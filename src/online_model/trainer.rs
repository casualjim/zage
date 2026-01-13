use std::collections::{HashMap, HashSet};

use libsql::Connection;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rkyv::{Archive, Deserialize, Serialize};

use crate::config::OnlineModelConfig;
use crate::hash_util::stable_hash;
use crate::repo::find_repo_root;
use crate::tokenize::{extract_command_parts, tokenize_index};
use crate::{Result, ZageError};

use super::replay::{ReplayConfig, sample_global_replay, sample_workspace_replay, store_replay};
use super::sampler::{GlobalCommandPool, NegativeSampler, SamplerPools};

const LR_EMBED: f32 = 0.05;
const LR_GROUP_SCALAR: f32 = 0.005;
const L2_GROUP_SCALAR: f32 = 0.001;
const ADAGRAD_EPS: f32 = 1e-6;
const GRAD_CLIP_NORM: f32 = 5.0;

const GROUP_WORKSPACE_ROOT: &str = "workspace_root";
const GROUP_CWD: &str = "cwd";
const GROUP_EXIT: &str = "exit";
const GROUP_HOST: &str = "host";
const GROUP_USER: &str = "user";
const GROUP_TIMEBUCKET: &str = "timebucket";
const GROUP_SESSION: &str = "session";
const GROUP_RECENT_HEADS: &str = "recent_heads";
const GROUP_RECENT_FLAGS: &str = "recent_flags";

const W_WORKSPACE_ROOT: f32 = 2.0;
const W_CWD: f32 = 1.5;
const W_EXIT: f32 = 1.0;
const W_HOST: f32 = 0.2;
const W_USER: f32 = 0.2;
const W_TIMEBUCKET: f32 = 0.1;
const W_SESSION: f32 = 0.1;
const W_RECENT_HEADS: f32 = 0.8;
const W_RECENT_FLAGS: f32 = 0.6;
const W_RECENT_ARGS: f32 = 0.4;

#[derive(Debug, Clone, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub(crate) struct OnlineExample {
  pub shellname: String,
  pub repo_root: String,
  pub cwd: Option<String>,
  pub workspace_key: Option<String>,
  pub positive_command_hash: u64,
  pub positive_head_hash: u64,
  pub now: i64,
  pub ctx_workspace: Vec<(u32, f32)>,
  pub ctx_cwd: Vec<(u32, f32)>,
  pub ctx_exit: Vec<(u32, f32)>,
  pub ctx_host: Vec<(u32, f32)>,
  pub ctx_user: Vec<(u32, f32)>,
  pub ctx_timebucket: Vec<(u32, f32)>,
  pub ctx_session: Vec<(u32, f32)>,
  pub recent_heads: Vec<(u32, f32)>,
  pub recent_flags: Vec<(u32, f32)>,
  pub recent_args: Vec<(u32, f32)>,
  pub cmd_buckets: Vec<(u32, f32)>,
}

pub(crate) async fn train_on_invocations(
  conn: &Connection,
  invocations: &[crate::core::Invocation],
) -> Result<()> {
  if invocations.is_empty() {
    return Ok(());
  }

  let config = OnlineModelConfig::load()?;
  conn.execute("BEGIN", ()).await?;
  let result: Result<()> = async {
    let now_fallback = unix_now();
    let seed = stable_hash("zage-online-model-v1");
    let mut cache = EmbeddingCache::new(seed, config.embedding_dim);
    let mut group_scalars = GroupScalarStore::default();

    let global_pool = GlobalCommandPool::load(conn).await?;
    let mut updates = 0u64;

    for inv in invocations {
      let Some(example_input) = build_example_input(
        conn,
        inv,
        now_fallback,
        config.window,
        config.bucket_count,
      )
      .await?
      else {
        continue;
      };
      updates += 1;

      // Deterministic per-event RNG derived from the event.
      let mut rng = StdRng::seed_from_u64(seed ^ example_input.example.positive_command_hash);

      // Sample replay before storing the new event.
      let global_replay = sample_global_replay(conn, &mut rng).await?;
      let workspace_replay = match example_input.example.workspace_key.as_deref() {
        Some(key) if !key.is_empty() => sample_workspace_replay(conn, key, &mut rng).await?,
        _ => None,
      };

      let mut sampler = NegativeSampler::new(&global_pool, config.bucket_count);
      train_one(
        conn,
        &mut cache,
        &mut group_scalars,
        &mut sampler,
        &example_input,
        config.negatives,
        &mut rng,
      )
      .await?;
      if let Some(replay) = global_replay {
        train_replay(
          conn,
          &mut cache,
          &mut group_scalars,
          &mut sampler,
          &replay,
          config.negatives,
          &mut rng,
        )
        .await?;
      }
      if let Some(replay) = workspace_replay {
        train_replay(
          conn,
          &mut cache,
          &mut group_scalars,
          &mut sampler,
          &replay,
          config.negatives,
          &mut rng,
        )
        .await?;
      }

      store_replay(
        conn,
        &example_input.example,
        &ReplayConfig {
          global_capacity: config.replay.global_capacity,
          workspace_capacity: config.replay.workspace_capacity,
          max_workspaces: config.replay.max_workspaces,
        },
        &mut rng,
      )
      .await?;
    }

    let write_now = unix_now();
    cache.flush(conn, write_now).await?;
    group_scalars.flush(conn, write_now).await?;
    if updates > 0 {
      crate::db::bump_online_model_update_count(conn, updates).await?;
    }
    Ok(())
  }
  .await;

  match result {
    Ok(()) => {
      conn.execute("COMMIT", ()).await?;
      Ok(())
    }
    Err(err) => {
      let _ = conn.execute("ROLLBACK", ()).await;
      Err(err)
    }
  }
}

struct ExampleInput {
  example: OnlineExample,
  recent_commands: Vec<String>,
}

async fn build_example_input(
  conn: &Connection,
  inv: &crate::core::Invocation,
  now_fallback: i64,
  window: usize,
  bucket_count: u32,
) -> Result<Option<ExampleInput>> {
  let stats_command = if inv.expanded_command.is_empty() {
    inv.command.as_str()
  } else {
    inv.expanded_command.as_str()
  };
  if stats_command.trim().is_empty() {
    return Ok(None);
  }

  let now = inv
    .end_unix_timestamp
    .or(inv.start_unix_timestamp)
    .unwrap_or(now_fallback);

  let cwd = inv.working_directory.clone();
  let repo_root = cwd
    .as_deref()
    .and_then(find_repo_root)
    .unwrap_or_default()
    .to_string();
  let workspace_key = if !repo_root.is_empty() {
    Some(repo_root.clone())
  } else {
    cwd.clone()
  };

  let head_hash = head_hash_for_command(&inv.shellname, stats_command).unwrap_or(0);
  let pos_hash = stable_hash(stats_command);

  let recent_commands = load_recent_session_commands(conn, inv.session_id, now, window).await?;

  let mut scratch_indices = Vec::new();
  let mut scratch_buckets = Vec::new();

  let mut ctx_workspace = HashMap::<u32, f32>::new();
  let mut ctx_cwd = HashMap::<u32, f32>::new();
  let mut ctx_exit = HashMap::<u32, f32>::new();
  let mut ctx_host = HashMap::<u32, f32>::new();
  let mut ctx_user = HashMap::<u32, f32>::new();
  let mut ctx_timebucket = HashMap::<u32, f32>::new();
  let mut ctx_session = HashMap::<u32, f32>::new();

  for tok in super::context_tokens_from_invocation(inv) {
    if let Some((group, _)) = tok.split_once(':') {
      let map = match group {
        "ctx" => {
          if tok.starts_with("ctx:workspace_root=") {
            &mut ctx_workspace
          } else if tok.starts_with("ctx:cwd=") {
            &mut ctx_cwd
          } else if tok.starts_with("ctx:exit=") {
            &mut ctx_exit
          } else if tok.starts_with("ctx:host=") {
            &mut ctx_host
          } else if tok.starts_with("ctx:user=") {
            &mut ctx_user
          } else if tok.starts_with("ctx:timebucket=") {
            &mut ctx_timebucket
          } else if tok.starts_with("ctx:session=") {
            &mut ctx_session
          } else {
            continue;
          }
        }
        _ => continue,
      };
      add_token_to_map(
        &tok,
        1.0,
        map,
        &mut scratch_indices,
        &mut scratch_buckets,
        bucket_count,
      );
    }
  }

  let mut recent_heads = HashMap::<u32, f32>::new();
  let mut recent_flags = HashMap::<u32, f32>::new();
  let mut recent_args = HashMap::<u32, f32>::new();
  for tok in super::window_tokens(&inv.shellname, &recent_commands, window) {
    let Some((_, rest)) = tok.split_once(':') else {
      continue;
    };
    let group_map = if rest.starts_with("head:") {
      &mut recent_heads
    } else if rest.starts_with("flag:") {
      &mut recent_flags
    } else if rest.starts_with("arg:") {
      &mut recent_args
    } else {
      continue;
    };
    add_token_to_map(
      &tok,
      1.0,
      group_map,
      &mut scratch_indices,
      &mut scratch_buckets,
      bucket_count,
    );
  }

  let mut cmd_map = HashMap::<u32, f32>::new();
  for tok in super::command_tokens(&inv.shellname, stats_command) {
    add_token_to_map(
      &tok,
      1.0,
      &mut cmd_map,
      &mut scratch_indices,
      &mut scratch_buckets,
      bucket_count,
    );
  }

  Ok(Some(ExampleInput {
    example: OnlineExample {
      shellname: inv.shellname.clone(),
      repo_root,
      cwd,
      workspace_key,
      positive_command_hash: pos_hash,
      positive_head_hash: head_hash,
      now,
      ctx_workspace: map_to_sorted_vec(ctx_workspace),
      ctx_cwd: map_to_sorted_vec(ctx_cwd),
      ctx_exit: map_to_sorted_vec(ctx_exit),
      ctx_host: map_to_sorted_vec(ctx_host),
      ctx_user: map_to_sorted_vec(ctx_user),
      ctx_timebucket: map_to_sorted_vec(ctx_timebucket),
      ctx_session: map_to_sorted_vec(ctx_session),
      recent_heads: map_to_sorted_vec(recent_heads),
      recent_flags: map_to_sorted_vec(recent_flags),
      recent_args: map_to_sorted_vec(recent_args),
      cmd_buckets: map_to_sorted_vec(cmd_map),
    },
    recent_commands,
  }))
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OnlineScoreContext<'a> {
  pub shellname: &'a str,
  pub repo_root: &'a str,
  pub cwd: Option<&'a str>,
  pub hostname: Option<&'a str>,
  pub username: Option<&'a str>,
  pub exit_status: Option<i64>,
  pub session_id: Option<i64>,
  pub unix_timestamp: i64,
  pub recent_commands: &'a [String],
  pub window: usize,
}

pub(crate) async fn score_commands(
  conn: &Connection,
  ctx: OnlineScoreContext<'_>,
  commands: &[String],
  config: &OnlineModelConfig,
) -> Result<Vec<f32>> {
  if commands.is_empty() {
    return Ok(Vec::new());
  }

  let bucket_count = config.bucket_count;
  let dim = config.embedding_dim;
  let workspace_root = if !ctx.repo_root.is_empty() {
    Some(ctx.repo_root)
  } else {
    ctx.cwd
  };

  let mut scratch_indices = Vec::new();
  let mut scratch_buckets = Vec::new();

  let mut ctx_workspace = HashMap::<u32, f32>::new();
  let mut ctx_cwd = HashMap::<u32, f32>::new();
  let mut ctx_exit = HashMap::<u32, f32>::new();
  let mut ctx_host = HashMap::<u32, f32>::new();
  let mut ctx_user = HashMap::<u32, f32>::new();
  let mut ctx_timebucket = HashMap::<u32, f32>::new();
  let mut ctx_session = HashMap::<u32, f32>::new();

  for tok in super::context_tokens(super::OnlineContextInput {
    workspace_root,
    cwd: ctx.cwd,
    hostname: ctx.hostname,
    username: ctx.username,
    exit_status: ctx.exit_status,
    session_id: ctx.session_id,
    unix_timestamp: Some(ctx.unix_timestamp),
  }) {
    if tok.starts_with("ctx:workspace_root=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_workspace,
        &mut scratch_indices,
        &mut scratch_buckets,
        bucket_count,
      );
    } else if tok.starts_with("ctx:cwd=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_cwd,
        &mut scratch_indices,
        &mut scratch_buckets,
        bucket_count,
      );
    } else if tok.starts_with("ctx:exit=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_exit,
        &mut scratch_indices,
        &mut scratch_buckets,
        bucket_count,
      );
    } else if tok.starts_with("ctx:host=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_host,
        &mut scratch_indices,
        &mut scratch_buckets,
        bucket_count,
      );
    } else if tok.starts_with("ctx:user=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_user,
        &mut scratch_indices,
        &mut scratch_buckets,
        bucket_count,
      );
    } else if tok.starts_with("ctx:timebucket=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_timebucket,
        &mut scratch_indices,
        &mut scratch_buckets,
        bucket_count,
      );
    } else if tok.starts_with("ctx:session=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_session,
        &mut scratch_indices,
        &mut scratch_buckets,
        bucket_count,
      );
    }
  }

  let mut recent_heads = HashMap::<u32, f32>::new();
  let mut recent_flags = HashMap::<u32, f32>::new();
  let mut recent_args = HashMap::<u32, f32>::new();
  let window = ctx.window.min(ctx.recent_commands.len());
  for tok in super::window_tokens(ctx.shellname, ctx.recent_commands, window) {
    let Some((_, rest)) = tok.split_once(':') else {
      continue;
    };
    let map = if rest.starts_with("head:") {
      &mut recent_heads
    } else if rest.starts_with("flag:") {
      &mut recent_flags
    } else if rest.starts_with("arg:") {
      &mut recent_args
    } else {
      continue;
    };
    add_token_to_map(
      &tok,
      1.0,
      map,
      &mut scratch_indices,
      &mut scratch_buckets,
      bucket_count,
    );
  }

  let ctx_workspace = map_to_sorted_vec(ctx_workspace);
  let ctx_cwd = map_to_sorted_vec(ctx_cwd);
  let ctx_exit = map_to_sorted_vec(ctx_exit);
  let ctx_host = map_to_sorted_vec(ctx_host);
  let ctx_user = map_to_sorted_vec(ctx_user);
  let ctx_timebucket = map_to_sorted_vec(ctx_timebucket);
  let ctx_session = map_to_sorted_vec(ctx_session);
  let recent_heads = map_to_sorted_vec(recent_heads);
  let recent_flags = map_to_sorted_vec(recent_flags);
  let recent_args = map_to_sorted_vec(recent_args);

  let mut buckets: HashSet<u32> = HashSet::new();
  for (b, _) in ctx_workspace
    .iter()
    .chain(ctx_cwd.iter())
    .chain(ctx_exit.iter())
    .chain(ctx_host.iter())
    .chain(ctx_user.iter())
    .chain(ctx_timebucket.iter())
    .chain(ctx_session.iter())
    .chain(recent_heads.iter())
    .chain(recent_flags.iter())
    .chain(recent_args.iter())
  {
    buckets.insert(*b);
  }

  let mut cmd_buckets: Vec<Vec<(u32, f32)>> = Vec::with_capacity(commands.len());
  for cmd in commands {
    let mut cmd_map = HashMap::<u32, f32>::new();
    for tok in super::command_tokens(ctx.shellname, cmd) {
      add_token_to_map(
        &tok,
        1.0,
        &mut cmd_map,
        &mut scratch_indices,
        &mut scratch_buckets,
        bucket_count,
      );
    }
    let vec = map_to_sorted_vec(cmd_map);
    for (b, _) in &vec {
      buckets.insert(*b);
    }
    cmd_buckets.push(vec);
  }

  let mut cache = EmbeddingCache::new(stable_hash("zage-online-model-v1"), dim);
  cache
    .preload(conn, buckets.into_iter(), ctx.unix_timestamp)
    .await?;

  let group_scalars = load_group_scalar_snapshot(conn).await?;

  let mut u_ctx = vec![0.0f32; dim];
  add_group_to_context_inference(
    &cache,
    &ctx_workspace,
    W_WORKSPACE_ROOT,
    group_scalars.workspace,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    &cache,
    &ctx_cwd,
    W_CWD,
    group_scalars.cwd,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    &cache,
    &ctx_exit,
    W_EXIT,
    group_scalars.exit,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    &cache,
    &ctx_host,
    W_HOST,
    group_scalars.host,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    &cache,
    &ctx_user,
    W_USER,
    group_scalars.user,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    &cache,
    &ctx_timebucket,
    W_TIMEBUCKET,
    group_scalars.timebucket,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    &cache,
    &ctx_session,
    W_SESSION,
    group_scalars.session,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    &cache,
    &recent_heads,
    W_RECENT_HEADS,
    group_scalars.recent_heads,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    &cache,
    &recent_flags,
    W_RECENT_FLAGS,
    group_scalars.recent_flags,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    &cache,
    &recent_args,
    W_RECENT_ARGS,
    1.0,
    u_ctx.as_mut_slice(),
  );

  let norm_ctx = l2_norm(&u_ctx).max(1e-8);
  let v_ctx = scale_vec(&u_ctx, 1.0 / norm_ctx);

  let mut out = Vec::with_capacity(commands.len());
  for buckets in cmd_buckets {
    let (u_cmd, _) = command_tower_vector(&cache, &buckets);
    let norm_cmd = l2_norm(&u_cmd).max(1e-8);
    let v_cmd = scale_vec(&u_cmd, 1.0 / norm_cmd);
    out.push(dot(&v_ctx, &v_cmd));
  }

  Ok(out)
}

#[derive(Debug, Clone, Copy)]
struct GroupScalarSnapshot {
  workspace: f32,
  cwd: f32,
  exit: f32,
  host: f32,
  user: f32,
  timebucket: f32,
  session: f32,
  recent_heads: f32,
  recent_flags: f32,
}

async fn load_group_scalar_snapshot(conn: &Connection) -> Result<GroupScalarSnapshot> {
  let mut snapshot = GroupScalarSnapshot {
    workspace: 1.0,
    cwd: 1.0,
    exit: 1.0,
    host: 1.0,
    user: 1.0,
    timebucket: 1.0,
    session: 1.0,
    recent_heads: 1.0,
    recent_flags: 1.0,
  };

  let mut rows = conn
    .query("SELECT group_name, value FROM online_group_scalar", ())
    .await?;
  while let Some(row) = rows.next().await? {
    let group: String = row.get(0)?;
    let value: f32 = row.get::<f64>(1)? as f32;
    match group.as_str() {
      GROUP_WORKSPACE_ROOT => snapshot.workspace = value,
      GROUP_CWD => snapshot.cwd = value,
      GROUP_EXIT => snapshot.exit = value,
      GROUP_HOST => snapshot.host = value,
      GROUP_USER => snapshot.user = value,
      GROUP_TIMEBUCKET => snapshot.timebucket = value,
      GROUP_SESSION => snapshot.session = value,
      GROUP_RECENT_HEADS => snapshot.recent_heads = value,
      GROUP_RECENT_FLAGS => snapshot.recent_flags = value,
      _ => {}
    }
  }
  Ok(snapshot)
}

fn add_group_to_context_inference(
  cache: &EmbeddingCache,
  buckets: &[(u32, f32)],
  base_weight: f32,
  scalar: f32,
  u_ctx: &mut [f32],
) {
  if buckets.is_empty() || base_weight == 0.0 || scalar == 0.0 {
    return;
  }

  for (bucket, w) in buckets {
    if let Some(e) = cache.get(*bucket) {
      for (idx, v) in e.iter().enumerate() {
        u_ctx[idx] += base_weight * scalar * (*w) * (*v);
      }
    }
  }
}

async fn train_one(
  conn: &Connection,
  cache: &mut EmbeddingCache,
  group_scalars: &mut GroupScalarStore,
  sampler: &mut NegativeSampler<'_>,
  input: &ExampleInput,
  negatives: usize,
  rng: &mut StdRng,
) -> Result<()> {
  let pools = sampler
    .build_pools(
      conn,
      &input.example.shellname,
      &input.example.repo_root,
      input.example.cwd.as_deref(),
      &input.recent_commands,
      input.example.positive_head_hash,
    )
    .await?;
  train_example_with_pools(
    conn,
    cache,
    group_scalars,
    sampler,
    &pools,
    &input.example,
    Some(&input.recent_commands),
    negatives,
    rng,
  )
  .await
}

async fn train_replay(
  conn: &Connection,
  cache: &mut EmbeddingCache,
  group_scalars: &mut GroupScalarStore,
  sampler: &mut NegativeSampler<'_>,
  replay: &OnlineExample,
  negatives: usize,
  rng: &mut StdRng,
) -> Result<()> {
  let pools = sampler
    .build_pools(
      conn,
      &replay.shellname,
      &replay.repo_root,
      replay.cwd.as_deref(),
      &[],
      replay.positive_head_hash,
    )
    .await?;
  train_example_with_pools(
    conn,
    cache,
    group_scalars,
    sampler,
    &pools,
    replay,
    None,
    negatives,
    rng,
  )
  .await
}

async fn train_example_with_pools(
  conn: &Connection,
  cache: &mut EmbeddingCache,
  group_scalars: &mut GroupScalarStore,
  sampler: &mut NegativeSampler<'_>,
  pools: &SamplerPools,
  example: &OnlineExample,
  recent_commands: Option<&[String]>,
  negatives: usize,
  rng: &mut StdRng,
) -> Result<()> {
  let (negatives, log_q_pos) = sampler.sample_with_logq(
    pools,
    &example.shellname,
    example.positive_command_hash,
    negatives,
    rng,
  )?;

  // Hash command vectors for negatives (we keep the positive vector in the example).
  let mut command_vecs: Vec<(Vec<(u32, f32)>, f32)> = Vec::with_capacity(1 + negatives.len());
  command_vecs.push((example.cmd_buckets.clone(), log_q_pos));
  for neg in negatives {
    command_vecs.push((neg.cmd_buckets, neg.log_q));
  }

  // Collect all bucket ids needed for this training step.
  let mut buckets: HashSet<u32> = HashSet::new();
  for (b, _) in example
    .ctx_workspace
    .iter()
    .chain(example.ctx_cwd.iter())
    .chain(example.ctx_exit.iter())
    .chain(example.ctx_host.iter())
    .chain(example.ctx_user.iter())
    .chain(example.ctx_timebucket.iter())
    .chain(example.ctx_session.iter())
    .chain(example.recent_heads.iter())
    .chain(example.recent_flags.iter())
    .chain(example.recent_args.iter())
  {
    buckets.insert(*b);
  }
  for (cmd_buckets, _) in &command_vecs {
    for (b, _) in cmd_buckets {
      buckets.insert(*b);
    }
  }

  cache
    .preload(conn, buckets.into_iter(), example.now)
    .await?;

  // Resolve group scalars (lazy init in DB).
  let s_workspace = group_scalars
    .get_or_init(conn, GROUP_WORKSPACE_ROOT)
    .await?;
  let s_cwd = group_scalars.get_or_init(conn, GROUP_CWD).await?;
  let s_exit = group_scalars.get_or_init(conn, GROUP_EXIT).await?;
  let s_host = group_scalars.get_or_init(conn, GROUP_HOST).await?;
  let s_user = group_scalars.get_or_init(conn, GROUP_USER).await?;
  let s_timebucket = group_scalars.get_or_init(conn, GROUP_TIMEBUCKET).await?;
  let s_session = group_scalars.get_or_init(conn, GROUP_SESSION).await?;
  let s_recent_heads = group_scalars.get_or_init(conn, GROUP_RECENT_HEADS).await?;
  let s_recent_flags = group_scalars.get_or_init(conn, GROUP_RECENT_FLAGS).await?;

  // Build context vector and bucket weights used in the context tower.
  let mut u_ctx = vec![0.0f32; cache.dim];
  let mut ctx_weights: HashMap<u32, f32> = HashMap::new();
  let mut group_u_no_scalar: HashMap<&'static str, Vec<f32>> = HashMap::new();

  add_group_to_context(
    cache,
    &example.ctx_workspace,
    W_WORKSPACE_ROOT,
    s_workspace,
    GROUP_WORKSPACE_ROOT,
    &mut u_ctx,
    &mut ctx_weights,
    &mut group_u_no_scalar,
  );
  add_group_to_context(
    cache,
    &example.ctx_cwd,
    W_CWD,
    s_cwd,
    GROUP_CWD,
    &mut u_ctx,
    &mut ctx_weights,
    &mut group_u_no_scalar,
  );
  add_group_to_context(
    cache,
    &example.ctx_exit,
    W_EXIT,
    s_exit,
    GROUP_EXIT,
    &mut u_ctx,
    &mut ctx_weights,
    &mut group_u_no_scalar,
  );
  add_group_to_context(
    cache,
    &example.ctx_host,
    W_HOST,
    s_host,
    GROUP_HOST,
    &mut u_ctx,
    &mut ctx_weights,
    &mut group_u_no_scalar,
  );
  add_group_to_context(
    cache,
    &example.ctx_user,
    W_USER,
    s_user,
    GROUP_USER,
    &mut u_ctx,
    &mut ctx_weights,
    &mut group_u_no_scalar,
  );
  add_group_to_context(
    cache,
    &example.ctx_timebucket,
    W_TIMEBUCKET,
    s_timebucket,
    GROUP_TIMEBUCKET,
    &mut u_ctx,
    &mut ctx_weights,
    &mut group_u_no_scalar,
  );
  add_group_to_context(
    cache,
    &example.ctx_session,
    W_SESSION,
    s_session,
    GROUP_SESSION,
    &mut u_ctx,
    &mut ctx_weights,
    &mut group_u_no_scalar,
  );

  add_group_to_context(
    cache,
    &example.recent_heads,
    W_RECENT_HEADS,
    s_recent_heads,
    GROUP_RECENT_HEADS,
    &mut u_ctx,
    &mut ctx_weights,
    &mut group_u_no_scalar,
  );
  add_group_to_context(
    cache,
    &example.recent_flags,
    W_RECENT_FLAGS,
    s_recent_flags,
    GROUP_RECENT_FLAGS,
    &mut u_ctx,
    &mut ctx_weights,
    &mut group_u_no_scalar,
  );

  // recent_args have a fixed base weight in v1.
  add_fixed_group_to_context(
    cache,
    &example.recent_args,
    W_RECENT_ARGS,
    &mut u_ctx,
    &mut ctx_weights,
  );

  let norm_ctx = l2_norm(&u_ctx);
  if norm_ctx <= 1e-8 {
    return Ok(());
  }
  let v_ctx = scale_vec(&u_ctx, 1.0 / norm_ctx);

  // Accumulate gradients (per bucket) for all samples in this event.
  let mut bucket_grads: HashMap<u32, Vec<f32>> = HashMap::new();
  let mut scalar_grads: HashMap<&'static str, f32> = HashMap::new();

  struct Candidate {
    label: f32,
    cmd_weights: HashMap<u32, f32>,
    v_cmd: Vec<f32>,
    norm_cmd: f32,
    s0: f32,
    logit: f32,
  }

  let mut candidates: Vec<Candidate> = Vec::with_capacity(command_vecs.len());
  for (idx, (cmd_buckets, log_q)) in command_vecs.iter().enumerate() {
    let label = if idx == 0 { 1.0 } else { 0.0 };
    let (u_cmd, cmd_weights) = command_tower_vector(cache, cmd_buckets);
    let norm_cmd = l2_norm(&u_cmd);
    if norm_cmd <= 1e-8 {
      continue;
    }
    let v_cmd = scale_vec(&u_cmd, 1.0 / norm_cmd);
    let s0 = dot(&v_ctx, &v_cmd);
    let logit = s0 - *log_q;
    candidates.push(Candidate {
      label,
      cmd_weights,
      v_cmd,
      norm_cmd,
      s0,
      logit,
    });
  }

  if candidates.is_empty() {
    return Ok(());
  }

  // Sampled-softmax (log-Q corrected): p = softmax(score - logQ).
  let max_logit = candidates
    .iter()
    .map(|c| c.logit)
    .fold(f32::NEG_INFINITY, f32::max);
  let mut sum = 0.0f32;
  let mut probs = Vec::with_capacity(candidates.len());
  for c in &candidates {
    let e = (c.logit - max_logit).exp();
    sum += e;
    probs.push(e);
  }
  if sum <= 1e-8 {
    return Ok(());
  }
  for p in probs.iter_mut() {
    *p /= sum;
  }

  for (candidate, p) in candidates.iter().zip(probs.iter().copied()) {
    let g = p - candidate.label;

    let mut grad_u_ctx = vec_sub(&candidate.v_cmd, &scale_vec(&v_ctx, candidate.s0));
    grad_u_ctx = scale_vec(&grad_u_ctx, g / norm_ctx);
    clip_in_place(&mut grad_u_ctx, GRAD_CLIP_NORM);

    let mut grad_u_cmd = vec_sub(&v_ctx, &scale_vec(&candidate.v_cmd, candidate.s0));
    grad_u_cmd = scale_vec(&grad_u_cmd, g / candidate.norm_cmd);
    clip_in_place(&mut grad_u_cmd, GRAD_CLIP_NORM);

    for (bucket, w) in &ctx_weights {
      add_scaled_bucket_grad(&mut bucket_grads, *bucket, &grad_u_ctx, *w);
    }
    for (bucket, w) in &candidate.cmd_weights {
      add_scaled_bucket_grad(&mut bucket_grads, *bucket, &grad_u_cmd, *w);
    }
    for (group, u_no_scalar) in &group_u_no_scalar {
      let grad = dot(&grad_u_ctx, u_no_scalar);
      *scalar_grads.entry(group).or_insert(0.0) += grad;
    }
  }

  // Apply embedding updates.
  for (bucket, grad) in bucket_grads {
    cache.apply_adagrad_update(bucket, &grad, LR_EMBED)?;
  }

  // Apply scalar updates with L2 reg toward 1.0.
  for (group, grad) in scalar_grads {
    let current = group_scalars.get_or_init(conn, group).await?;
    let reg = L2_GROUP_SCALAR * (current - 1.0);
    let updated = (current - LR_GROUP_SCALAR * (grad + reg)).clamp(0.05, 20.0);
    group_scalars.set(group, updated);
  }

  // Slight sanity: keep recent_commands used to build pools from being unused in the signature.
  let _ = recent_commands;

  Ok(())
}

fn add_group_to_context(
  cache: &EmbeddingCache,
  buckets: &[(u32, f32)],
  base_weight: f32,
  scalar: f32,
  group: &'static str,
  u_ctx: &mut Vec<f32>,
  ctx_weights: &mut HashMap<u32, f32>,
  group_u_no_scalar: &mut HashMap<&'static str, Vec<f32>>,
) {
  if buckets.is_empty() || base_weight == 0.0 {
    return;
  }

  let mut u = vec![0.0f32; cache.dim];
  for (bucket, w) in buckets {
    if let Some(e) = cache.get(*bucket) {
      for (idx, v) in e.iter().enumerate() {
        u[idx] += base_weight * (*w) * (*v);
      }
      *ctx_weights.entry(*bucket).or_insert(0.0) += base_weight * scalar * (*w);
    }
  }
  for idx in 0..u_ctx.len() {
    u_ctx[idx] += scalar * u[idx];
  }
  group_u_no_scalar.insert(group, u);
}

fn add_fixed_group_to_context(
  cache: &EmbeddingCache,
  buckets: &[(u32, f32)],
  base_weight: f32,
  u_ctx: &mut Vec<f32>,
  ctx_weights: &mut HashMap<u32, f32>,
) {
  if buckets.is_empty() || base_weight == 0.0 {
    return;
  }
  for (bucket, w) in buckets {
    if let Some(e) = cache.get(*bucket) {
      for (idx, v) in e.iter().enumerate() {
        u_ctx[idx] += base_weight * (*w) * (*v);
      }
      *ctx_weights.entry(*bucket).or_insert(0.0) += base_weight * (*w);
    }
  }
}

fn command_tower_vector(
  cache: &EmbeddingCache,
  buckets: &[(u32, f32)],
) -> (Vec<f32>, HashMap<u32, f32>) {
  let mut u = vec![0.0f32; cache.dim];
  let mut weights = HashMap::new();
  for (bucket, w) in buckets {
    if let Some(e) = cache.get(*bucket) {
      for (idx, v) in e.iter().enumerate() {
        u[idx] += (*w) * (*v);
      }
      *weights.entry(*bucket).or_insert(0.0) += *w;
    }
  }
  (u, weights)
}

fn add_scaled_bucket_grad(
  grads: &mut HashMap<u32, Vec<f32>>,
  bucket: u32,
  grad_u: &[f32],
  weight: f32,
) {
  if weight == 0.0 {
    return;
  }
  let entry = grads
    .entry(bucket)
    .or_insert_with(|| vec![0.0f32; grad_u.len()]);
  for (idx, g) in grad_u.iter().enumerate() {
    entry[idx] += weight * (*g);
  }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
  a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
  dot(v, v).sqrt()
}

fn scale_vec(v: &[f32], s: f32) -> Vec<f32> {
  v.iter().map(|x| x * s).collect()
}

fn vec_sub(a: &[f32], b: &[f32]) -> Vec<f32> {
  a.iter().zip(b).map(|(x, y)| x - y).collect()
}

fn clip_in_place(v: &mut [f32], max_norm: f32) {
  if max_norm <= 0.0 {
    return;
  }
  let n = l2_norm(v);
  if n <= max_norm || n <= 1e-8 {
    return;
  }
  let s = max_norm / n;
  for x in v.iter_mut() {
    *x *= s;
  }
}

fn add_token_to_map(
  token: &str,
  scale: f32,
  out: &mut HashMap<u32, f32>,
  scratch_indices: &mut Vec<usize>,
  scratch: &mut Vec<(u32, f32)>,
  bucket_count: u32,
) {
  if scale == 0.0 {
    return;
  }
  crate::hash_util::stable_char_ngrams_buckets(
    token,
    bucket_count,
    scratch_indices,
    scratch,
  );
  for (bucket, sign) in scratch.iter().copied() {
    *out.entry(bucket).or_insert(0.0) += scale * sign;
  }
}

fn map_to_sorted_vec(mut map: HashMap<u32, f32>) -> Vec<(u32, f32)> {
  map.retain(|_, v| v.abs() > f32::EPSILON);
  let mut out: Vec<(u32, f32)> = map.into_iter().collect();
  out.sort_by(|a, b| a.0.cmp(&b.0));
  out
}

fn head_hash_for_command(shellname: &str, command: &str) -> Option<u64> {
  let tokens = tokenize_index(shellname, command);
  let parts = extract_command_parts(command, &tokens)?;
  if parts.head.trim().is_empty() {
    return None;
  }
  Some(stable_hash(parts.head.trim()))
}

async fn load_recent_session_commands(
  conn: &Connection,
  session_id: i64,
  before_ts: i64,
  limit: usize,
) -> Result<Vec<String>> {
  if limit == 0 {
    return Ok(Vec::new());
  }
  let mut rows = conn
    .query(
      "SELECT command, expanded_command FROM shell_history
       WHERE session_id = ? AND COALESCE(start_unix_timestamp, 0) < ?
       ORDER BY COALESCE(start_unix_timestamp, 0) DESC, id DESC
       LIMIT ?",
      libsql::params![session_id, before_ts, limit as i64],
    )
    .await?;
  let mut out = Vec::new();
  while let Some(row) = rows.next().await? {
    let command: String = row.get(0)?;
    let expanded: String = row.get(1)?;
    if expanded.is_empty() {
      out.push(command);
    } else {
      out.push(expanded);
    }
  }
  out.reverse();
  Ok(out)
}

fn unix_now() -> i64 {
  std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64
}

struct EmbeddingCache {
  seed: u64,
  dim: usize,
  map: HashMap<u32, EmbeddingRow>,
}

struct EmbeddingRow {
  vec: Vec<f32>,
  acc: Vec<f32>,
  dirty: bool,
}

impl EmbeddingCache {
  fn new(seed: u64, dim: usize) -> Self {
    Self {
      seed,
      dim,
      map: HashMap::new(),
    }
  }

  async fn preload(
    &mut self,
    conn: &Connection,
    buckets: impl Iterator<Item = u32>,
    now: i64,
  ) -> Result<()> {
    for bucket in buckets {
      if self.map.contains_key(&bucket) {
        continue;
      }
      let mut rows = conn
        .query(
          "SELECT vec, opt_state FROM online_token_embedding WHERE bucket = ?",
          libsql::params![bucket as i64],
        )
        .await?;
      if let Some(row) = rows.next().await? {
        let vec_blob: Vec<u8> = row.get(0)?;
        let opt_blob: Option<Vec<u8>> = row.get(1)?;
        let vec = decode_f32_blob(&vec_blob, self.dim).ok_or_else(|| {
          ZageError::ConfigError(format!("invalid embedding blob for bucket {bucket}"))
        })?;
        let acc = match opt_blob {
          Some(blob) => {
            decode_f32_blob(&blob, self.dim).unwrap_or_else(|| vec![0.0; self.dim])
          }
          None => vec![0.0; self.dim],
        };
        self.map.insert(
          bucket,
          EmbeddingRow {
            vec,
            acc,
            dirty: false,
          },
        );
      } else {
        let vec = init_embedding(bucket, self.seed, self.dim);
        self.map.insert(
          bucket,
          EmbeddingRow {
            vec,
            acc: vec![0.0; self.dim],
            dirty: true,
          },
        );
        // We don't write immediately; flush() will persist.
        let _ = now;
      }
    }
    Ok(())
  }

  fn get(&self, bucket: u32) -> Option<&[f32]> {
    self.map.get(&bucket).map(|r| r.vec.as_slice())
  }

  fn apply_adagrad_update(&mut self, bucket: u32, grad: &[f32], lr: f32) -> Result<()> {
    let Some(row) = self.map.get_mut(&bucket) else {
      return Err(ZageError::ConfigError(format!(
        "missing embedding bucket {bucket} (not preloaded)"
      )));
    };
    if row.vec.len() != self.dim
      || row.acc.len() != self.dim
      || grad.len() != self.dim
    {
      return Err(ZageError::ConfigError(format!(
        "dimension mismatch for bucket {bucket}"
      )));
    }
    for idx in 0..self.dim {
      let g = grad[idx];
      row.acc[idx] += g * g;
      row.vec[idx] -= lr * g / (row.acc[idx] + ADAGRAD_EPS).sqrt();
    }
    row.dirty = true;
    Ok(())
  }

  async fn flush(&mut self, conn: &Connection, now: i64) -> Result<()> {
    for (bucket, row) in self.map.iter_mut() {
      if !row.dirty {
        continue;
      }
      let vec_blob = encode_f32_blob(&row.vec);
      let acc_blob = encode_f32_blob(&row.acc);
      conn
        .execute(
          "INSERT INTO online_token_embedding (bucket, vec, opt_state, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(bucket) DO UPDATE SET
             vec = excluded.vec,
             opt_state = excluded.opt_state,
             updated_at = excluded.updated_at",
          libsql::params![*bucket as i64, vec_blob, acc_blob, now],
        )
        .await?;
      row.dirty = false;
    }
    Ok(())
  }
}

fn init_embedding(bucket: u32, seed: u64, dim: usize) -> Vec<f32> {
  let mut rng = StdRng::seed_from_u64(seed ^ (bucket as u64));
  // Small symmetric init.
  (0..dim)
    .map(|_| rng.random_range(-0.01f32..0.01f32))
    .collect()
}

fn encode_f32_blob(values: &[f32]) -> Vec<u8> {
  let mut out = Vec::with_capacity(values.len() * 4);
  for v in values {
    out.extend_from_slice(&v.to_le_bytes());
  }
  out
}

fn decode_f32_blob(blob: &[u8], dim: usize) -> Option<Vec<f32>> {
  let expected = dim.checked_mul(4)?;
  if blob.len() != expected {
    return None;
  }
  let mut out = Vec::with_capacity(dim);
  for chunk in blob.chunks_exact(4) {
    out.push(f32::from_le_bytes(chunk.try_into().ok()?));
  }
  Some(out)
}

#[derive(Default)]
struct GroupScalarStore {
  values: HashMap<&'static str, f32>,
  dirty: HashSet<&'static str>,
}

impl GroupScalarStore {
  async fn get_or_init(&mut self, conn: &Connection, group: &'static str) -> Result<f32> {
    if let Some(v) = self.values.get(group) {
      return Ok(*v);
    }
    let mut rows = conn
      .query(
        "SELECT value FROM online_group_scalar WHERE group_name = ?",
        libsql::params![group.to_string()],
      )
      .await?;
    let value = if let Some(row) = rows.next().await? {
      row.get::<f64>(0)? as f32
    } else {
      conn
        .execute(
          "INSERT INTO online_group_scalar (group_name, value, updated_at) VALUES (?, ?, ?)",
          libsql::params![group.to_string(), 1.0f64, unix_now()],
        )
        .await?;
      1.0
    };
    self.values.insert(group, value);
    Ok(value)
  }

  fn set(&mut self, group: &'static str, value: f32) {
    self.values.insert(group, value);
    self.dirty.insert(group);
  }

  async fn flush(&mut self, conn: &Connection, now: i64) -> Result<()> {
    for group in self.dirty.drain() {
      let Some(value) = self.values.get(group).copied() else {
        continue;
      };
      conn
        .execute(
          "INSERT INTO online_group_scalar (group_name, value, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT(group_name) DO UPDATE SET
             value = excluded.value,
             updated_at = excluded.updated_at",
          libsql::params![group.to_string(), value as f64, now],
        )
        .await?;
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::core::Invocation;
  use crate::db::{init, insert_invocation, open_db, update_stats_for_invocation};
  use crate::predict::verifier::{TestConfig, suggest_for_test};
  use crate::core::RankingWeights;
  use std::fs;
  use std::path::Path;
  use std::sync::{Mutex, OnceLock};
  use tempfile::NamedTempFile;
  use tempfile::TempDir;

  fn command_buckets_for(shellname: &str, command: &str) -> Vec<(u32, f32)> {
    let bucket_count = OnlineModelConfig::default().bucket_count;
    let mut map = HashMap::<u32, f32>::new();
    let mut scratch_indices = Vec::new();
    let mut scratch = Vec::new();
    for tok in super::super::command_tokens(shellname, command) {
      crate::hash_util::stable_char_ngrams_buckets(
        &tok,
        bucket_count,
        &mut scratch_indices,
        &mut scratch,
      );
      for (bucket, sign) in scratch.iter().copied() {
        *map.entry(bucket).or_insert(0.0) += sign;
      }
    }
    let mut out: Vec<(u32, f32)> = map.into_iter().collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
  }

  async fn score_for_context_and_command(
    conn: &Connection,
    inv: &Invocation,
    command: &str,
  ) -> Result<f32> {
    let config = OnlineModelConfig::default();
    let input = build_example_input(
      conn,
      inv,
      unix_now(),
      config.window,
      config.bucket_count,
    )
    .await?;
    let Some(input) = input else {
      return Err(ZageError::ConfigError("missing example".to_string()));
    };

    let mut buckets = HashSet::new();
    for (b, _) in input
      .example
      .ctx_workspace
      .iter()
      .chain(input.example.ctx_cwd.iter())
      .chain(input.example.ctx_exit.iter())
      .chain(input.example.ctx_host.iter())
      .chain(input.example.ctx_user.iter())
      .chain(input.example.ctx_timebucket.iter())
      .chain(input.example.ctx_session.iter())
      .chain(input.example.recent_heads.iter())
      .chain(input.example.recent_flags.iter())
      .chain(input.example.recent_args.iter())
    {
      buckets.insert(*b);
    }

    let cmd_buckets = command_buckets_for(&input.example.shellname, command);
    for (b, _) in &cmd_buckets {
      buckets.insert(*b);
    }

    let mut cache = EmbeddingCache::new(stable_hash("zage-online-model-v1"), config.embedding_dim);
    cache
      .preload(conn, buckets.into_iter(), input.example.now)
      .await?;

    let mut u_ctx = vec![0.0f32; cache.dim];
    let mut ctx_weights = HashMap::new();
    add_fixed_group_to_context(
      &cache,
      &input.example.ctx_workspace,
      W_WORKSPACE_ROOT,
      &mut u_ctx,
      &mut ctx_weights,
    );
    add_fixed_group_to_context(
      &cache,
      &input.example.ctx_cwd,
      W_CWD,
      &mut u_ctx,
      &mut ctx_weights,
    );
    add_fixed_group_to_context(
      &cache,
      &input.example.ctx_exit,
      W_EXIT,
      &mut u_ctx,
      &mut ctx_weights,
    );
    add_fixed_group_to_context(
      &cache,
      &input.example.ctx_host,
      W_HOST,
      &mut u_ctx,
      &mut ctx_weights,
    );
    add_fixed_group_to_context(
      &cache,
      &input.example.ctx_user,
      W_USER,
      &mut u_ctx,
      &mut ctx_weights,
    );
    add_fixed_group_to_context(
      &cache,
      &input.example.ctx_timebucket,
      W_TIMEBUCKET,
      &mut u_ctx,
      &mut ctx_weights,
    );
    add_fixed_group_to_context(
      &cache,
      &input.example.ctx_session,
      W_SESSION,
      &mut u_ctx,
      &mut ctx_weights,
    );
    add_fixed_group_to_context(
      &cache,
      &input.example.recent_heads,
      W_RECENT_HEADS,
      &mut u_ctx,
      &mut ctx_weights,
    );
    add_fixed_group_to_context(
      &cache,
      &input.example.recent_flags,
      W_RECENT_FLAGS,
      &mut u_ctx,
      &mut ctx_weights,
    );
    add_fixed_group_to_context(
      &cache,
      &input.example.recent_args,
      W_RECENT_ARGS,
      &mut u_ctx,
      &mut ctx_weights,
    );

    let (u_cmd, _) = command_tower_vector(&cache, &cmd_buckets);
    let norm_ctx = l2_norm(&u_ctx).max(1e-8);
    let norm_cmd = l2_norm(&u_cmd).max(1e-8);
    let v_ctx = scale_vec(&u_ctx, 1.0 / norm_ctx);
    let v_cmd = scale_vec(&u_cmd, 1.0 / norm_cmd);
    Ok(dot(&v_ctx, &v_cmd))
  }

  #[derive(Debug, Clone)]
  struct EvalMetrics {
    mrr_at_k: f64,
    recall_at_k: f64,
    coverage_at_k: f64,
    leakage_rate: f64,
    total: usize,
    step_mrr: Vec<f64>,
  }

  struct EnvGuard {
    key: &'static str,
    previous: Option<String>,
  }

  impl Drop for EnvGuard {
    fn drop(&mut self) {
      if let Some(value) = self.previous.as_ref() {
        unsafe {
          std::env::set_var(self.key, value);
        }
      } else {
        unsafe {
          std::env::remove_var(self.key);
        }
      }
    }
  }

  fn env_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(())).lock().unwrap()
  }

  fn set_env_guard(key: &'static str, value: Option<String>) -> EnvGuard {
    let previous = std::env::var(key).ok();
    match value {
      Some(value) => unsafe {
        std::env::set_var(key, value);
      },
      None => unsafe {
        std::env::remove_var(key);
      },
    }
    EnvGuard { key, previous }
  }

  fn workspace_key(invocation: &Invocation) -> String {
    let Some(cwd) = invocation.working_directory.as_deref() else {
      return String::new();
    };
    find_repo_root(cwd).unwrap_or_else(|| cwd.to_string())
  }

  fn push_invocation(
    invocations: &mut Vec<Invocation>,
    command: &str,
    cwd: &Path,
    session_id: i64,
    ts: &mut i64,
  ) {
    invocations.push(Invocation {
      command: command.to_string(),
      expanded_command: command.to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some(cwd.to_string_lossy().to_string()),
      workspace: None,
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(*ts),
      end_unix_timestamp: Some(*ts + 1),
      session_id,
    });
    *ts += 2;
  }

  fn build_fixture(root: &TempDir) -> Result<Vec<Invocation>> {
    let repo_a = root.path().join("repo-a");
    let repo_b = root.path().join("repo-b");
    fs::create_dir_all(repo_a.join(".git"))?;
    fs::create_dir_all(repo_b.join(".git"))?;

    let repo_a_work = repo_a.join("src");
    let repo_b_work = repo_b.join("work");
    fs::create_dir_all(&repo_a_work)?;
    fs::create_dir_all(&repo_b_work)?;

    let mut invocations = Vec::new();
    let mut ts = 1_700_000_000i64;

    for _ in 0..4 {
      push_invocation(
        &mut invocations,
        "git status",
        &repo_b_work,
        2,
        &mut ts,
      );
      push_invocation(
        &mut invocations,
        "git log",
        &repo_b_work,
        2,
        &mut ts,
      );
      push_invocation(
        &mut invocations,
        "git status",
        &repo_b_work,
        2,
        &mut ts,
      );
      push_invocation(
        &mut invocations,
        "git show",
        &repo_b_work,
        2,
        &mut ts,
      );
    }

    for _ in 0..20 {
      push_invocation(
        &mut invocations,
        "git status",
        &repo_a_work,
        1,
        &mut ts,
      );
      push_invocation(
        &mut invocations,
        "git diff",
        &repo_a_work,
        1,
        &mut ts,
      );
    }

    Ok(invocations)
  }

  fn dominant_workspace(invocations: &[Invocation]) -> HashMap<String, String> {
    let mut counts: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for inv in invocations {
      let workspace = workspace_key(inv);
      let cmd = if inv.expanded_command.is_empty() {
        inv.command.clone()
      } else {
        inv.expanded_command.clone()
      };
      let entry = counts.entry(cmd).or_default();
      *entry.entry(workspace).or_insert(0) += 1;
    }

    let mut dominant = HashMap::new();
    for (command, workspaces) in counts {
      let mut best: Option<(String, usize)> = None;
      for (workspace, count) in workspaces {
        match &best {
          Some((_, best_count)) if *best_count >= count => {}
          _ => best = Some((workspace, count)),
        }
      }
      if let Some((workspace, _)) = best {
        dominant.insert(command, workspace);
      }
    }
    dominant
  }

  async fn run_prequential(
    invocations: &[Invocation],
    eval_start: usize,
    max_results: usize,
    recent_limit: usize,
    weights: RankingWeights,
  ) -> Result<EvalMetrics> {
    let tmp = NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    let dominant = dominant_workspace(invocations);
    let mut unique_predicted = HashSet::<String>::new();

    let mut mrr_sum = 0.0;
    let mut recall_hits = 0usize;
    let mut leakage_hits = 0usize;
    let mut total = 0usize;
    let mut step_mrr = Vec::new();

    for (idx, inv) in invocations.iter().enumerate() {
      if idx >= 1 && idx >= eval_start {
        let now = inv.start_unix_timestamp.unwrap_or(0);
        let config = crate::core::SuggestConfig {
          max_results,
          recent_limit,
          prefix: None,
          cwd: inv.working_directory.clone(),
          hostname: inv.hostname.clone(),
          username: inv.username.clone(),
          session_id: Some(inv.session_id),
          shellname: Some(inv.shellname.clone()),
          use_sequences: true,
          prefer_full_line: true,
        };
        let test_config = TestConfig {
          now: Some(now),
          weights: Some(weights.clone()),
          recency_half_life: Some(5.0),
          debug: false,
        };

        let suggestions = suggest_for_test(&db.conn, config, test_config).await?;
        let true_command = if inv.expanded_command.is_empty() {
          inv.command.clone()
        } else {
          inv.expanded_command.clone()
        };
        let mut rank: Option<usize> = None;
        for suggestion in &suggestions {
          if suggestion.command == true_command {
            rank = Some(suggestion.rank);
            break;
          }
        }

        let mut step_score = 0.0;
        if let Some(rank) = rank {
          if rank <= max_results {
            step_score = 1.0 / rank as f64;
            mrr_sum += step_score;
            recall_hits += 1;
          }
        }
        step_mrr.push(step_score);

        let mut leaked = false;
        let current_workspace = workspace_key(inv);
        for suggestion in suggestions.iter().take(max_results) {
          unique_predicted.insert(suggestion.command.clone());
          if let Some(expected_workspace) = dominant.get(&suggestion.command)
            && *expected_workspace != current_workspace
          {
            leaked = true;
          }
        }
        if leaked {
          leakage_hits += 1;
        }
        total += 1;
      }

      let _ = insert_invocation(&db.conn, inv).await?;
      update_stats_for_invocation(&db.conn, inv).await?;
      train_on_invocations(&db.conn, std::slice::from_ref(inv)).await?;
    }

    let denom = total.max(1) as f64;
    let coverage = if total == 0 || max_results == 0 {
      0.0
    } else {
      unique_predicted.len() as f64 / (total * max_results) as f64
    };
    Ok(EvalMetrics {
      mrr_at_k: mrr_sum / denom,
      recall_at_k: recall_hits as f64 / denom,
      coverage_at_k: coverage,
      leakage_rate: leakage_hits as f64 / denom,
      total,
      step_mrr,
    })
  }

  #[tokio::test]
  async fn online_training_increases_margin_for_seen_transition() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    let mut invocations = Vec::new();
    let mut ts = 10i64;
    // Strong signal: repeat (status -> diff) many times.
    for _ in 0..30 {
      invocations.push(Invocation {
        command: "git status".to_string(),
        expanded_command: "git status".to_string(),
        shellname: "zsh".to_string(),
        working_directory: Some("/tmp".to_string()),
        workspace: None,
        hostname: Some("host".to_string()),
        username: Some("user".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(ts),
        end_unix_timestamp: Some(ts + 1),
        session_id: 1,
      });
      ts += 2;
      invocations.push(Invocation {
        command: "git diff".to_string(),
        expanded_command: "git diff".to_string(),
        shellname: "zsh".to_string(),
        working_directory: Some("/tmp".to_string()),
        workspace: None,
        hostname: Some("host".to_string()),
        username: Some("user".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(ts),
        end_unix_timestamp: Some(ts + 1),
        session_id: 1,
      });
      ts += 2;
    }
    // Add a few negatives so global pool has plausible distractors.
    for _ in 0..5 {
      invocations.push(Invocation {
        command: "ls -la".to_string(),
        expanded_command: "ls -la".to_string(),
        shellname: "zsh".to_string(),
        working_directory: Some("/tmp".to_string()),
        workspace: None,
        hostname: Some("host".to_string()),
        username: Some("user".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(ts),
        end_unix_timestamp: Some(ts + 1),
        session_id: 1,
      });
      ts += 2;
    }

    for inv in &invocations {
      let _ = insert_invocation(&db.conn, inv).await?;
      update_stats_for_invocation(&db.conn, inv).await?;
    }

    // Pick a representative "git diff" invocation near the end.
    let target = invocations
      .iter()
      .rev()
      .find(|inv| inv.expanded_command == "git diff")
      .ok_or_else(|| ZageError::ConfigError("missing git diff".to_string()))?;

    let base_pos = score_for_context_and_command(&db.conn, target, "git diff").await?;
    let base_neg = score_for_context_and_command(&db.conn, target, "ls -la").await?;
    let base_margin = base_pos - base_neg;

    train_on_invocations(&db.conn, &invocations).await?;

    let after_pos = score_for_context_and_command(&db.conn, target, "git diff").await?;
    let after_neg = score_for_context_and_command(&db.conn, target, "ls -la").await?;
    let after_margin = after_pos - after_neg;

    assert!(
      after_margin > base_margin,
      "expected margin to improve after training"
    );
    Ok(())
  }

  #[tokio::test]
  async fn prequential_eval_tracks_online_mrr_and_leakage() -> Result<()> {
    let _env_lock = env_lock();
    let temp_dir = TempDir::new()?;
    let invocations = build_fixture(&temp_dir)?;

    let max_results = 5;
    let recent_limit = OnlineModelConfig::default().window;
    let first_a = invocations
      .iter()
      .position(|inv| inv.session_id == 1)
      .unwrap_or(0);
    let eval_start = first_a + recent_limit + 2;

    let model_dir = temp_dir.path().join("model");
    fs::create_dir_all(&model_dir)?;
    let _model_guard =
      set_env_guard("ZAGE_MODEL_PATH", Some(model_dir.to_string_lossy().to_string()));

    let weights = RankingWeights {
      recency: 0.10,
      frequency: 0.0,
      transition: 0.0,
      context: 0.0,
      sequence: 0.0,
      similarity: 0.0,
    };

    let online =
      run_prequential(&invocations, eval_start, max_results, recent_limit, weights.clone())
        .await?;

    let split = online.step_mrr.len() / 2;
    let (early_slice, late_slice) = if split > 0 {
      (&online.step_mrr[..split], &online.step_mrr[split..])
    } else {
      (&online.step_mrr[..], &online.step_mrr[..])
    };
    let early_mrr = if early_slice.is_empty() {
      0.0
    } else {
      early_slice.iter().sum::<f64>() / early_slice.len() as f64
    };
    let late_mrr = if late_slice.is_empty() {
      0.0
    } else {
      late_slice.iter().sum::<f64>() / late_slice.len() as f64
    };

    eprintln!(
      "prequential@{} online:   mrr={:.3} recall={:.3} coverage={:.3} leakage={:.3} (n={})",
      max_results,
      online.mrr_at_k,
      online.recall_at_k,
      online.coverage_at_k,
      online.leakage_rate,
      online.total
    );
    eprintln!(
      "prequential@{} online split: early_mrr={:.3} late_mrr={:.3}",
      max_results,
      early_mrr,
      late_mrr
    );

    assert!(
      late_mrr >= early_mrr,
      "expected online model MRR@{} to not regress over time",
      max_results
    );
    assert!(
      online.leakage_rate <= f64::EPSILON,
      "expected zero leakage on fixture"
    );
    Ok(())
  }
}
