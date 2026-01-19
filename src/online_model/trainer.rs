use std::collections::{HashMap, HashSet};

use libsql::{Connection, Value};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rkyv::{Archive, Deserialize, Serialize};

use crate::config::OnlineModelConfig;
use crate::hash_util::stable_hash;
use crate::tokenize::{extract_command_parts, normalize_command_whitespace, tokenize_index};
use crate::{Result, ZageError};

use super::replay::{ReplayConfig, sample_global_replay, sample_workspace_replay, store_replay};
use super::sampler::{GlobalCommandPool, NegativeSampler, SamplerPools};

const LR_EMBED: f32 = 0.05;
const LR_GROUP_SCALAR: f32 = 0.005;
const L2_GROUP_SCALAR: f32 = 0.001;
const LR_BIAS: f32 = 0.02;
const L2_BIAS: f32 = 0.001;
const BIAS_CLAMP: f32 = 5.0;
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
  pub workspace_root: String,
  pub cwd: Option<String>,
  pub workspace_key: Option<String>,
  pub positive_command: String,
  pub positive_head: Option<String>,
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
    let mut workspace_root_cache: HashMap<String, String> = HashMap::new();
    let mut group_scalars = GroupScalarStore::default();
    let mut bias_store = BiasStore::default();

    let global_pool = GlobalCommandPool::load(conn).await?;
    let mut sampler = NegativeSampler::new(&global_pool, config.bucket_count);
    let mut updates = 0u64;

    let use_in_memory_recent = config.window > 0 && {
      let mut last_by_session: HashMap<i64, i64> = HashMap::new();
      invocations.iter().all(|inv| {
        let now = inv
          .end_unix_timestamp
          .or(inv.start_unix_timestamp)
          .unwrap_or(now_fallback);
        match last_by_session.get(&inv.session_id) {
          Some(prev) if now < *prev => false,
          _ => {
            last_by_session.insert(inv.session_id, now);
            true
          }
        }
      })
    };

    #[derive(Debug, Default)]
    struct RecentSessionState {
      current_ts: Option<i64>,
      deferred: Vec<String>,
      window: Vec<String>,
    }

    fn push_window(window: &mut Vec<String>, command: String, max_len: usize) {
      window.push(command);
      if max_len == 0 {
        window.clear();
        return;
      }
      while window.len() > max_len {
        window.remove(0);
      }
    }

    let mut recent_by_session: HashMap<i64, RecentSessionState> = HashMap::new();
    if use_in_memory_recent && config.window > 0 {
      let mut min_ts_by_session: HashMap<i64, i64> = HashMap::new();
      for inv in invocations {
        let now = inv
          .end_unix_timestamp
          .or(inv.start_unix_timestamp)
          .unwrap_or(now_fallback);
        min_ts_by_session
          .entry(inv.session_id)
          .and_modify(|min| {
            if now < *min {
              *min = now;
            }
          })
          .or_insert(now);
      }

      for (session_id, min_ts) in min_ts_by_session {
        let seed_window =
          load_recent_session_commands(conn, session_id, min_ts, config.window).await?;
        recent_by_session.insert(
          session_id,
          RecentSessionState {
            current_ts: None,
            deferred: Vec::new(),
            window: seed_window,
          },
        );
      }
    }

    for inv in invocations {
      let now = inv
        .end_unix_timestamp
        .or(inv.start_unix_timestamp)
        .unwrap_or(now_fallback);

      let stats_command = if inv.expanded_command.is_empty() {
        inv.command.clone()
      } else {
        inv.expanded_command.clone()
      };
      let stats_command = normalize_command_whitespace(&stats_command);

      let recent_commands = if use_in_memory_recent {
        let state = recent_by_session.entry(inv.session_id).or_default();
        if state.current_ts != Some(now) {
          let deferred = std::mem::take(&mut state.deferred);
          for cmd in deferred {
            push_window(&mut state.window, cmd, config.window);
          }
          state.current_ts = Some(now);
        }
        state.window.clone()
      } else {
        load_recent_session_commands(conn, inv.session_id, now, config.window).await?
      };

      if use_in_memory_recent {
        let state = recent_by_session.entry(inv.session_id).or_default();
        state.deferred.push(stats_command.clone());
      }

      let Some(example_input) = build_example_input_from_recent(ExampleInputFromRecentArgs {
        inv,
        stats_command: stats_command.clone(),
        now,
        recent_commands,
        window: config.window,
        bucket_count: config.bucket_count,
        workspace_root_cache: &mut workspace_root_cache,
      })?
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

      {
        let mut ctx = TrainStepContext {
          conn,
          cache: &mut cache,
          group_scalars: &mut group_scalars,
          bias_store: &mut bias_store,
          sampler: &mut sampler,
          rng: &mut rng,
        };
        train_one(&mut ctx, &example_input, config.negatives).await?;
        if let Some(replay) = global_replay {
          train_replay(&mut ctx, &replay, config.negatives).await?;
        }
        if let Some(replay) = workspace_replay {
          train_replay(&mut ctx, &replay, config.negatives).await?;
        }
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
    bias_store.flush(conn, write_now).await?;
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

pub(crate) async fn train_on_invocations_bulk(
  conn: &Connection,
  invocations: &[crate::core::Invocation],
) -> Result<()> {
  if invocations.is_empty() {
    return Ok(());
  }

  let config = OnlineModelConfig::load()?;

  let now_fallback = unix_now();
  let seed = stable_hash("zage-online-model-v1");
  let mut cache = EmbeddingCache::new(seed, config.embedding_dim);
  let mut workspace_root_cache: HashMap<String, String> = HashMap::new();
  let mut group_scalars = GroupScalarStore::default();
  let mut bias_store = BiasStore::default();
  bias_store.preload_all(conn).await?;

  let global_pool = GlobalCommandPool::load(conn).await?;
  let mut sampler = NegativeSampler::new(&global_pool, config.bucket_count);
  let mut updates = 0u64;

  let replay_config = ReplayConfig {
    global_capacity: config.replay.global_capacity,
    workspace_capacity: config.replay.workspace_capacity,
    max_workspaces: config.replay.max_workspaces,
  };
  let mut replay_store = BulkReplayStore::load(conn, &replay_config).await?;

  let use_in_memory_recent = config.window > 0 && {
    let mut last_by_session: HashMap<i64, i64> = HashMap::new();
    invocations.iter().all(|inv| {
      let now = inv
        .end_unix_timestamp
        .or(inv.start_unix_timestamp)
        .unwrap_or(now_fallback);
      match last_by_session.get(&inv.session_id) {
        Some(prev) if now < *prev => false,
        _ => {
          last_by_session.insert(inv.session_id, now);
          true
        }
      }
    })
  };

  #[derive(Debug, Default)]
  struct RecentSessionState {
    current_ts: Option<i64>,
    deferred: Vec<String>,
    window: Vec<String>,
  }

  fn push_window(window: &mut Vec<String>, command: String, max_len: usize) {
    window.push(command);
    if max_len == 0 {
      window.clear();
      return;
    }
    while window.len() > max_len {
      window.remove(0);
    }
  }

  let mut recent_by_session: HashMap<i64, RecentSessionState> = HashMap::new();
  if use_in_memory_recent {
    let mut min_ts_by_session: HashMap<i64, i64> = HashMap::new();
    for inv in invocations {
      let now = inv
        .end_unix_timestamp
        .or(inv.start_unix_timestamp)
        .unwrap_or(now_fallback);
      min_ts_by_session
        .entry(inv.session_id)
        .and_modify(|min| {
          if now < *min {
            *min = now;
          }
        })
        .or_insert(now);
    }

    for (session_id, min_ts) in min_ts_by_session {
      let seed_window =
        load_recent_session_commands(conn, session_id, min_ts, config.window).await?;
      recent_by_session.insert(
        session_id,
        RecentSessionState {
          current_ts: None,
          deferred: Vec::new(),
          window: seed_window,
        },
      );
    }
  }

  for inv in invocations {
    let now = inv
      .end_unix_timestamp
      .or(inv.start_unix_timestamp)
      .unwrap_or(now_fallback);

    let stats_command = if inv.expanded_command.is_empty() {
      inv.command.clone()
    } else {
      inv.expanded_command.clone()
    };
    let stats_command = normalize_command_whitespace(&stats_command);

    let recent_commands = if use_in_memory_recent {
      let state = recent_by_session.entry(inv.session_id).or_default();
      if state.current_ts != Some(now) {
        let deferred = std::mem::take(&mut state.deferred);
        for cmd in deferred {
          push_window(&mut state.window, cmd, config.window);
        }
        state.current_ts = Some(now);
      }
      state.window.clone()
    } else {
      load_recent_session_commands(conn, inv.session_id, now, config.window).await?
    };

    if use_in_memory_recent {
      let state = recent_by_session.entry(inv.session_id).or_default();
      state.deferred.push(stats_command.clone());
    }

    let Some(example_input) = build_example_input_from_recent(ExampleInputFromRecentArgs {
      inv,
      stats_command: stats_command.clone(),
      now,
      recent_commands,
      window: config.window,
      bucket_count: config.bucket_count,
      workspace_root_cache: &mut workspace_root_cache,
    })?
    else {
      continue;
    };

    updates += 1;

    // Deterministic per-event RNG derived from the event.
    let mut rng = StdRng::seed_from_u64(seed ^ example_input.example.positive_command_hash);

    // Sample replay before storing the new event.
    let global_replay = replay_store.sample_global(&mut rng)?;
    let workspace_replay = match example_input.example.workspace_key.as_deref() {
      Some(key) if !key.is_empty() => replay_store.sample_workspace(key, &mut rng)?,
      _ => None,
    };

    {
      let mut ctx = TrainStepContext {
        conn,
        cache: &mut cache,
        group_scalars: &mut group_scalars,
        bias_store: &mut bias_store,
        sampler: &mut sampler,
        rng: &mut rng,
      };
      train_one(&mut ctx, &example_input, config.negatives).await?;
      if let Some(replay) = global_replay {
        train_replay(&mut ctx, &replay, config.negatives).await?;
      }
      if let Some(replay) = workspace_replay {
        train_replay(&mut ctx, &replay, config.negatives).await?;
      }
    }

    replay_store.store(&example_input.example, &replay_config, &mut rng)?;
  }

  let write_now = unix_now();
  conn.execute("BEGIN", ()).await?;
  let result: Result<()> = async {
    cache.flush(conn, write_now).await?;
    group_scalars.flush(conn, write_now).await?;
    bias_store.flush(conn, write_now).await?;
    replay_store.flush(conn, &replay_config).await?;

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

struct BulkReplayStore {
  global_seen: i64,
  global: Vec<(Vec<u8>, i64)>,
  workspaces: HashMap<String, BulkWorkspaceReplay>,
}

struct BulkWorkspaceReplay {
  seq: i64,
  last: i64,
  slots: Vec<Option<Vec<u8>>>,
  filled_slots: Vec<usize>,
}

impl BulkReplayStore {
  async fn load(conn: &Connection, config: &ReplayConfig) -> Result<Self> {
    let global_seen = bulk_get_meta_i64(conn, "online_replay_global_seen")
      .await?
      .unwrap_or(0);

    let mut global = Vec::new();
    if config.global_capacity > 0 {
      let mut rows = conn
        .query(
          "SELECT payload, sampled_at FROM online_replay_global ORDER BY event_id ASC",
          (),
        )
        .await?;
      while let Some(row) = rows.next().await? {
        let payload: Vec<u8> = row.get(0)?;
        let sampled_at: i64 = row.get(1)?;
        global.push((payload, sampled_at));
      }
    }

    let mut ws_seq: HashMap<String, i64> = HashMap::new();
    let mut ws_last: HashMap<String, i64> = HashMap::new();

    let mut last_rows = conn
      .query(
        "SELECT key, value FROM online_model_meta WHERE key LIKE ?",
        libsql::params!["online_replay_ws_last:%".to_string()],
      )
      .await?;
    while let Some(row) = last_rows.next().await? {
      let key: String = row.get(0)?;
      let value: String = row.get(1)?;
      let Some(workspace) = key.strip_prefix("online_replay_ws_last:") else {
        continue;
      };
      let ts = value.parse::<i64>().unwrap_or(0);
      ws_last.insert(workspace.to_string(), ts);
    }

    let mut seq_rows = conn
      .query(
        "SELECT key, value FROM online_model_meta WHERE key LIKE ?",
        libsql::params!["online_replay_ws_seq:%".to_string()],
      )
      .await?;
    while let Some(row) = seq_rows.next().await? {
      let key: String = row.get(0)?;
      let value: String = row.get(1)?;
      let Some(workspace) = key.strip_prefix("online_replay_ws_seq:") else {
        continue;
      };
      let seq = value.parse::<i64>().unwrap_or(0);
      ws_seq.insert(workspace.to_string(), seq);
    }

    let mut workspaces: HashMap<String, BulkWorkspaceReplay> = HashMap::new();
    if config.workspace_capacity > 0 && config.max_workspaces > 0 {
      let mut rows = conn
        .query(
          "SELECT workspace_root, seq, payload FROM online_replay_workspace",
          (),
        )
        .await?;
      while let Some(row) = rows.next().await? {
        let workspace_root: String = row.get(0)?;
        let slot: i64 = row.get(1)?;
        let payload: Vec<u8> = row.get(2)?;
        if workspace_root.trim().is_empty() {
          continue;
        }
        let slot_usize = usize::try_from(slot).unwrap_or(0);
        if slot_usize >= config.workspace_capacity {
          continue;
        }
        let entry = workspaces.entry(workspace_root.clone()).or_insert_with(|| {
          let seq = ws_seq.get(&workspace_root).copied().unwrap_or(0);
          let last = ws_last.get(&workspace_root).copied().unwrap_or(0);
          BulkWorkspaceReplay {
            seq,
            last,
            slots: vec![None; config.workspace_capacity],
            filled_slots: Vec::new(),
          }
        });
        if entry.slots[slot_usize].is_none() {
          entry.filled_slots.push(slot_usize);
        }
        entry.slots[slot_usize] = Some(payload);
      }

      // Ensure we don't exceed the configured max workspace count.
      if workspaces.len() > config.max_workspaces {
        let mut by_last = workspaces
          .iter()
          .map(|(k, v)| (k.clone(), v.last))
          .collect::<Vec<_>>();
        by_last.sort_by_key(|(_, ts)| *ts);
        let evict_count = by_last.len() - config.max_workspaces;
        for (workspace, _) in by_last.into_iter().take(evict_count) {
          workspaces.remove(&workspace);
        }
      }
    }

    Ok(Self {
      global_seen,
      global,
      workspaces,
    })
  }

  fn sample_global(&self, rng: &mut StdRng) -> Result<Option<OnlineExample>> {
    if self.global.is_empty() {
      return Ok(None);
    }
    let idx = rng.random_range(0..self.global.len());
    let (payload, _) = &self.global[idx];
    Ok(Some(bulk_decode_example(payload)?))
  }

  fn sample_workspace(
    &self,
    workspace_root: &str,
    rng: &mut StdRng,
  ) -> Result<Option<OnlineExample>> {
    let Some(ws) = self.workspaces.get(workspace_root) else {
      return Ok(None);
    };
    if ws.filled_slots.is_empty() {
      return Ok(None);
    }
    let pick = rng.random_range(0..ws.filled_slots.len());
    let slot = ws.filled_slots[pick];
    let Some(payload) = ws.slots.get(slot).and_then(|p| p.as_ref()) else {
      return Ok(None);
    };
    Ok(Some(bulk_decode_example(payload)?))
  }

  fn store(
    &mut self,
    example: &OnlineExample,
    config: &ReplayConfig,
    rng: &mut StdRng,
  ) -> Result<()> {
    let payload = bulk_encode_example(example)?;
    let now = example.now;

    self.store_global(&payload, now, config.global_capacity, rng)?;
    if let Some(workspace) = example.workspace_key.as_deref()
      && !workspace.trim().is_empty()
    {
      self.store_workspace(
        workspace,
        payload,
        now,
        config.workspace_capacity,
        config.max_workspaces,
      )?;
    }
    Ok(())
  }

  fn store_global(
    &mut self,
    payload: &[u8],
    now: i64,
    capacity: usize,
    rng: &mut StdRng,
  ) -> Result<()> {
    if capacity == 0 {
      return Ok(());
    }
    self.global_seen += 1;
    if self.global.len() < capacity {
      self.global.push((payload.to_vec(), now));
      return Ok(());
    }

    let keep_prob = (capacity as f64) / (self.global_seen as f64);
    if rng.random::<f64>() >= keep_prob {
      return Ok(());
    }
    let replace_offset = rng.random_range(0..capacity);
    if let Some(entry) = self.global.get_mut(replace_offset) {
      *entry = (payload.to_vec(), now);
    }
    Ok(())
  }

  fn store_workspace(
    &mut self,
    workspace_root: &str,
    payload: Vec<u8>,
    now: i64,
    capacity: usize,
    max_workspaces: usize,
  ) -> Result<()> {
    if capacity == 0 || max_workspaces == 0 {
      return Ok(());
    }

    if !self.workspaces.contains_key(workspace_root)
      && self.workspaces.len() >= max_workspaces
      && let Some((evict_key, _)) = self
        .workspaces
        .iter()
        .min_by_key(|(_, ws)| ws.last)
        .map(|(k, ws)| (k.clone(), ws.last))
    {
      self.workspaces.remove(&evict_key);
    }

    let ws = self
      .workspaces
      .entry(workspace_root.to_string())
      .or_insert_with(|| BulkWorkspaceReplay {
        seq: 0,
        last: 0,
        slots: vec![None; capacity],
        filled_slots: Vec::new(),
      });

    ws.seq += 1;
    ws.last = now;
    let slot = (ws.seq.rem_euclid(capacity as i64)) as usize;
    if ws.slots[slot].is_none() {
      ws.filled_slots.push(slot);
    }
    ws.slots[slot] = Some(payload);
    Ok(())
  }

  async fn flush(&self, conn: &Connection, config: &ReplayConfig) -> Result<()> {
    if config.global_capacity > 0 {
      conn.execute("DELETE FROM online_replay_global", ()).await?;
      let insert_global = conn
        .prepare("INSERT INTO online_replay_global (payload, sampled_at) VALUES (?, ?)")
        .await?;
      for (payload, sampled_at) in &self.global {
        insert_global
          .execute(libsql::params![payload.clone(), *sampled_at])
          .await?;
      }
    }

    if config.workspace_capacity > 0 && config.max_workspaces > 0 {
      conn
        .execute("DELETE FROM online_replay_workspace", ())
        .await?;
      conn
        .execute(
          "DELETE FROM online_model_meta WHERE key LIKE ?",
          libsql::params!["online_replay_ws_seq:%".to_string()],
        )
        .await?;
      conn
        .execute(
          "DELETE FROM online_model_meta WHERE key LIKE ?",
          libsql::params!["online_replay_ws_last:%".to_string()],
        )
        .await?;

      let insert_ws = conn
        .prepare(
          "INSERT OR REPLACE INTO online_replay_workspace (workspace_root, seq, payload)
           VALUES (?, ?, ?)",
        )
        .await?;
      let set_meta = conn
        .prepare(
          "INSERT INTO online_model_meta (key, value) VALUES (?, ?)
           ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        )
        .await?;

      for (workspace, ws) in &self.workspaces {
        set_meta
          .execute(libsql::params![
            format!("online_replay_ws_seq:{workspace}"),
            ws.seq.to_string()
          ])
          .await?;
        set_meta
          .execute(libsql::params![
            format!("online_replay_ws_last:{workspace}"),
            ws.last.to_string()
          ])
          .await?;

        for &slot in &ws.filled_slots {
          let Some(payload) = ws.slots.get(slot).and_then(|p| p.as_ref()) else {
            continue;
          };
          insert_ws
            .execute(libsql::params![
              workspace.clone(),
              slot as i64,
              payload.clone()
            ])
            .await?;
        }
      }
    }

    bulk_set_meta_i64(conn, "online_replay_global_seen", self.global_seen).await?;
    Ok(())
  }
}

async fn bulk_get_meta_i64(conn: &Connection, key: &str) -> Result<Option<i64>> {
  let mut rows = conn
    .query(
      "SELECT value FROM online_model_meta WHERE key = ?",
      libsql::params![key.to_string()],
    )
    .await?;
  let Some(row) = rows.next().await? else {
    return Ok(None);
  };
  let value: String = row.get(0)?;
  Ok(value.parse::<i64>().ok())
}

async fn bulk_set_meta_i64(conn: &Connection, key: &str, value: i64) -> Result<()> {
  conn
    .execute(
      "INSERT INTO online_model_meta (key, value) VALUES (?, ?)
       ON CONFLICT(key) DO UPDATE SET value = excluded.value",
      libsql::params![key.to_string(), value.to_string()],
    )
    .await?;
  Ok(())
}

fn bulk_encode_example(example: &OnlineExample) -> Result<Vec<u8>> {
  rkyv::to_bytes::<_, 4_096>(example)
    .map(|b| b.to_vec())
    .map_err(|err| ZageError::ConfigError(err.to_string()))
}

fn bulk_decode_example(payload: &[u8]) -> Result<OnlineExample> {
  rkyv::from_bytes::<OnlineExample>(payload).map_err(|err| ZageError::ConfigError(err.to_string()))
}

fn resolve_workspace_root(cwd: Option<&str>) -> String {
  let Some(cwd) = cwd else {
    return String::new();
  };
  crate::workspace::workspace_root_for_cwd(cwd).unwrap_or_default()
}

fn resolve_workspace_root_cached(cwd: Option<&str>, cache: &mut HashMap<String, String>) -> String {
  let Some(cwd) = cwd else {
    return String::new();
  };
  if let Some(root) = cache.get(cwd) {
    return root.clone();
  }
  let root = resolve_workspace_root(Some(cwd));
  cache.insert(cwd.to_string(), root.clone());
  root
}

#[cfg(test)]
async fn build_example_input(
  conn: &Connection,
  inv: &crate::core::Invocation,
  now_fallback: i64,
  window: usize,
  bucket_count: u32,
  workspace_root_cache: &mut HashMap<String, String>,
) -> Result<Option<ExampleInput>> {
  let now = inv
    .end_unix_timestamp
    .or(inv.start_unix_timestamp)
    .unwrap_or(now_fallback);

  let stats_command = if inv.expanded_command.is_empty() {
    inv.command.as_str()
  } else {
    inv.expanded_command.as_str()
  };
  let stats_command = normalize_command_whitespace(stats_command);
  let recent_commands = load_recent_session_commands(conn, inv.session_id, now, window).await?;
  build_example_input_from_recent(ExampleInputFromRecentArgs {
    inv,
    stats_command,
    now,
    recent_commands,
    window,
    bucket_count,
    workspace_root_cache,
  })
}

struct ExampleInputFromRecentArgs<'a> {
  inv: &'a crate::core::Invocation,
  stats_command: String,
  now: i64,
  recent_commands: Vec<String>,
  window: usize,
  bucket_count: u32,
  workspace_root_cache: &'a mut HashMap<String, String>,
}

fn build_example_input_from_recent(
  args: ExampleInputFromRecentArgs<'_>,
) -> Result<Option<ExampleInput>> {
  if args.stats_command.trim().is_empty() {
    return Ok(None);
  }

  let cwd = args.inv.working_directory.clone();
  let workspace_root = resolve_workspace_root_cached(cwd.as_deref(), args.workspace_root_cache);
  let workspace_key = if !workspace_root.is_empty() {
    Some(workspace_root.clone())
  } else {
    cwd.clone()
  };

  let positive_command = args.stats_command.clone();
  let positive_head = head_for_command(&args.inv.shellname, &args.stats_command);
  let head_hash = positive_head.as_deref().map(stable_hash).unwrap_or(0);
  let pos_hash = stable_hash(&args.stats_command);

  let mut scratch_indices = Vec::new();
  let mut scratch_buckets = Vec::new();

  let mut ctx_workspace = HashMap::<u32, f32>::new();
  let mut ctx_cwd = HashMap::<u32, f32>::new();
  let mut ctx_exit = HashMap::<u32, f32>::new();
  let mut ctx_host = HashMap::<u32, f32>::new();
  let mut ctx_user = HashMap::<u32, f32>::new();
  let mut ctx_timebucket = HashMap::<u32, f32>::new();
  let mut ctx_session = HashMap::<u32, f32>::new();

  for tok in super::context_tokens_from_invocation(args.inv) {
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
        args.bucket_count,
      );
    }
  }

  let mut recent_heads = HashMap::<u32, f32>::new();
  let mut recent_flags = HashMap::<u32, f32>::new();
  let mut recent_args = HashMap::<u32, f32>::new();
  for tok in super::window_tokens(&args.inv.shellname, &args.recent_commands, args.window) {
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
      args.bucket_count,
    );
  }

  let mut cmd_map = HashMap::<u32, f32>::new();
  for tok in super::command_tokens(&args.inv.shellname, &args.stats_command) {
    add_token_to_map(
      &tok,
      1.0,
      &mut cmd_map,
      &mut scratch_indices,
      &mut scratch_buckets,
      args.bucket_count,
    );
  }

  Ok(Some(ExampleInput {
    example: OnlineExample {
      shellname: args.inv.shellname.clone(),
      workspace_root,
      cwd,
      workspace_key,
      positive_command,
      positive_head,
      positive_command_hash: pos_hash,
      positive_head_hash: head_hash,
      now: args.now,
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
    recent_commands: args.recent_commands,
  }))
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OnlineScoreContext<'a> {
  pub shellname: &'a str,
  pub workspace_root: &'a str,
  pub cwd: Option<&'a str>,
  pub hostname: Option<&'a str>,
  pub username: Option<&'a str>,
  pub exit_status: Option<i64>,
  pub session_id: Option<i64>,
  pub unix_timestamp: i64,
  pub recent_commands: &'a [String],
  pub window: usize,
}

struct ContextMaps {
  ctx_workspace: Vec<(u32, f32)>,
  ctx_cwd: Vec<(u32, f32)>,
  ctx_exit: Vec<(u32, f32)>,
  ctx_host: Vec<(u32, f32)>,
  ctx_user: Vec<(u32, f32)>,
  ctx_timebucket: Vec<(u32, f32)>,
  ctx_session: Vec<(u32, f32)>,
  recent_heads: Vec<(u32, f32)>,
  recent_flags: Vec<(u32, f32)>,
  recent_args: Vec<(u32, f32)>,
}

fn context_bias_bucket_from_groups(groups: &[&[(u32, f32)]]) -> Option<(u32, f32)> {
  for group in groups {
    if let Some((bucket, sign)) = group.first().copied() {
      return Some((bucket, sign));
    }
  }
  None
}

fn context_bias_bucket_from_maps(context: &ContextMaps) -> Option<(u32, f32)> {
  context_bias_bucket_from_groups(&[
    &context.ctx_workspace,
    &context.ctx_cwd,
    &context.ctx_exit,
    &context.ctx_host,
    &context.ctx_user,
    &context.ctx_timebucket,
    &context.ctx_session,
  ])
}

fn context_bias_bucket_from_example(example: &OnlineExample) -> Option<(u32, f32)> {
  context_bias_bucket_from_groups(&[
    &example.ctx_workspace,
    &example.ctx_cwd,
    &example.ctx_exit,
    &example.ctx_host,
    &example.ctx_user,
    &example.ctx_timebucket,
    &example.ctx_session,
  ])
}

fn build_context_maps(
  ctx: OnlineScoreContext<'_>,
  bucket_count: u32,
  scratch_indices: &mut Vec<usize>,
  scratch_buckets: &mut Vec<(u32, f32)>,
) -> ContextMaps {
  let workspace_root = if !ctx.workspace_root.is_empty() {
    Some(ctx.workspace_root)
  } else {
    ctx.cwd
  };

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
    git_branch: None,
    exit_status: ctx.exit_status,
    session_id: ctx.session_id,
    unix_timestamp: Some(ctx.unix_timestamp),
  }) {
    if tok.starts_with("ctx:workspace_root=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_workspace,
        scratch_indices,
        scratch_buckets,
        bucket_count,
      );
    } else if tok.starts_with("ctx:cwd=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_cwd,
        scratch_indices,
        scratch_buckets,
        bucket_count,
      );
    } else if tok.starts_with("ctx:exit=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_exit,
        scratch_indices,
        scratch_buckets,
        bucket_count,
      );
    } else if tok.starts_with("ctx:host=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_host,
        scratch_indices,
        scratch_buckets,
        bucket_count,
      );
    } else if tok.starts_with("ctx:user=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_user,
        scratch_indices,
        scratch_buckets,
        bucket_count,
      );
    } else if tok.starts_with("ctx:timebucket=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_timebucket,
        scratch_indices,
        scratch_buckets,
        bucket_count,
      );
    } else if tok.starts_with("ctx:session=") {
      add_token_to_map(
        &tok,
        1.0,
        &mut ctx_session,
        scratch_indices,
        scratch_buckets,
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
      scratch_indices,
      scratch_buckets,
      bucket_count,
    );
  }

  ContextMaps {
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
  }
}

fn collect_context_buckets(context: &ContextMaps) -> HashSet<u32> {
  let mut buckets: HashSet<u32> = HashSet::new();
  for (b, _) in context
    .ctx_workspace
    .iter()
    .chain(context.ctx_cwd.iter())
    .chain(context.ctx_exit.iter())
    .chain(context.ctx_host.iter())
    .chain(context.ctx_user.iter())
    .chain(context.ctx_timebucket.iter())
    .chain(context.ctx_session.iter())
    .chain(context.recent_heads.iter())
    .chain(context.recent_flags.iter())
    .chain(context.recent_args.iter())
  {
    buckets.insert(*b);
  }
  buckets
}

fn context_embedding_from_maps(
  cache: &EmbeddingCache,
  group_scalars: &GroupScalarSnapshot,
  context: &ContextMaps,
) -> (Vec<f32>, f32) {
  let mut u_ctx = vec![0.0f32; cache.dim];
  add_group_to_context_inference(
    cache,
    &context.ctx_workspace,
    W_WORKSPACE_ROOT,
    group_scalars.workspace,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    cache,
    &context.ctx_cwd,
    W_CWD,
    group_scalars.cwd,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    cache,
    &context.ctx_exit,
    W_EXIT,
    group_scalars.exit,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    cache,
    &context.ctx_host,
    W_HOST,
    group_scalars.host,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    cache,
    &context.ctx_user,
    W_USER,
    group_scalars.user,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    cache,
    &context.ctx_timebucket,
    W_TIMEBUCKET,
    group_scalars.timebucket,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    cache,
    &context.ctx_session,
    W_SESSION,
    group_scalars.session,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    cache,
    &context.recent_heads,
    W_RECENT_HEADS,
    group_scalars.recent_heads,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    cache,
    &context.recent_flags,
    W_RECENT_FLAGS,
    group_scalars.recent_flags,
    u_ctx.as_mut_slice(),
  );
  add_group_to_context_inference(
    cache,
    &context.recent_args,
    W_RECENT_ARGS,
    1.0,
    u_ctx.as_mut_slice(),
  );

  let norm_ctx = l2_norm(&u_ctx);
  let denom = norm_ctx.max(1e-8);
  (scale_vec(&u_ctx, 1.0 / denom), norm_ctx)
}

pub(crate) async fn context_embedding_for_retrieval(
  conn: &Connection,
  ctx: OnlineScoreContext<'_>,
  config: &OnlineModelConfig,
) -> Result<Option<Vec<f32>>> {
  let bucket_count = config.bucket_count;
  let dim = config.embedding_dim;

  let mut scratch_indices = Vec::new();
  let mut scratch_buckets = Vec::new();
  let context_maps = build_context_maps(
    ctx,
    bucket_count,
    &mut scratch_indices,
    &mut scratch_buckets,
  );
  let buckets = collect_context_buckets(&context_maps);
  if buckets.is_empty() {
    return Ok(None);
  }

  let mut cache = EmbeddingCache::new(stable_hash("zage-online-model-v1"), dim);
  cache
    .preload(conn, buckets.into_iter(), ctx.unix_timestamp)
    .await?;
  let group_scalars = load_group_scalar_snapshot(conn).await?;
  let (v_ctx, norm_ctx) = context_embedding_from_maps(&cache, &group_scalars, &context_maps);
  if norm_ctx <= 1e-8 {
    return Ok(None);
  }
  Ok(Some(v_ctx))
}

pub(crate) async fn command_embeddings_for_commands(
  conn: &Connection,
  commands: &[(String, String)],
  config: &OnlineModelConfig,
) -> Result<Vec<Vec<f32>>> {
  if commands.is_empty() {
    return Ok(Vec::new());
  }

  let bucket_count = config.bucket_count;
  let dim = config.embedding_dim;

  let mut scratch_indices = Vec::new();
  let mut scratch_buckets = Vec::new();
  let mut buckets: HashSet<u32> = HashSet::new();
  let mut cmd_buckets: Vec<Vec<(u32, f32)>> = Vec::with_capacity(commands.len());
  for (command, shellname) in commands {
    let mut cmd_map = HashMap::<u32, f32>::new();
    for tok in super::command_tokens(shellname, command) {
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
  cache.preload(conn, buckets.into_iter(), unix_now()).await?;

  let mut out = Vec::with_capacity(commands.len());
  for buckets in cmd_buckets {
    let (u_cmd, _) = command_tower_vector(&cache, &buckets);
    let norm_cmd = l2_norm(&u_cmd);
    if norm_cmd <= 1e-8 {
      out.push(vec![0.0f32; dim]);
    } else {
      out.push(scale_vec(&u_cmd, 1.0 / norm_cmd));
    }
  }
  Ok(out)
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

  let mut scratch_indices = Vec::new();
  let mut scratch_buckets = Vec::new();
  let context_maps = build_context_maps(
    ctx,
    bucket_count,
    &mut scratch_indices,
    &mut scratch_buckets,
  );

  let mut buckets = collect_context_buckets(&context_maps);

  let mut cmd_buckets: Vec<Vec<(u32, f32)>> = Vec::with_capacity(commands.len());
  let mut command_heads: Vec<Option<String>> = Vec::with_capacity(commands.len());
  let mut unique_heads: HashSet<String> = HashSet::new();
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
    let head = head_for_command(ctx.shellname, cmd);
    if let Some(head) = head.as_ref() {
      unique_heads.insert(head.clone());
    }
    command_heads.push(head);
  }

  let mut cache = EmbeddingCache::new(stable_hash("zage-online-model-v1"), dim);
  cache
    .preload(conn, buckets.into_iter(), ctx.unix_timestamp)
    .await?;

  let group_scalars = load_group_scalar_snapshot(conn).await?;
  let (v_ctx, _) = context_embedding_from_maps(&cache, &group_scalars, &context_maps);

  let command_biases = load_command_biases(conn, commands).await?;
  let head_biases = load_head_biases(conn, &unique_heads).await?;
  let ctx_bias_value = if let Some((bucket, sign)) = context_bias_bucket_from_maps(&context_maps) {
    sign * load_context_bias(conn, bucket).await?
  } else {
    0.0
  };

  let mut out = Vec::with_capacity(commands.len());
  for (idx, buckets) in cmd_buckets.into_iter().enumerate() {
    let (u_cmd, _) = command_tower_vector(&cache, &buckets);
    let norm_cmd = l2_norm(&u_cmd).max(1e-8);
    let v_cmd = scale_vec(&u_cmd, 1.0 / norm_cmd);
    let mut score = dot(&v_ctx, &v_cmd) + ctx_bias_value;
    if let Some(bias) = command_biases.get(commands[idx].as_str()) {
      score += *bias;
    }
    if let Some(head) = command_heads.get(idx).and_then(|head| head.as_ref())
      && let Some(bias) = head_biases.get(head.as_str())
    {
      score += *bias;
    }
    out.push(score);
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

async fn load_context_bias(conn: &Connection, bucket: u32) -> Result<f32> {
  let mut rows = conn
    .query(
      "SELECT bias FROM online_context_bias WHERE bucket = ?",
      libsql::params![bucket as i64],
    )
    .await?;
  let value = if let Some(row) = rows.next().await? {
    row.get::<f64>(0)? as f32
  } else {
    0.0
  };
  Ok(value)
}

async fn load_command_biases(
  conn: &Connection,
  commands: &[String],
) -> Result<HashMap<String, f32>> {
  if commands.is_empty() {
    return Ok(HashMap::new());
  }

  let mut placeholders = String::new();
  for idx in 0..commands.len() {
    if idx > 0 {
      placeholders.push(',');
    }
    placeholders.push('?');
  }
  let mut params: Vec<Value> = Vec::with_capacity(commands.len());
  for cmd in commands {
    params.push(Value::from(cmd.clone()));
  }

  let sql = format!(
    "SELECT command, bias FROM online_command_bias WHERE command IN ({})",
    placeholders
  );
  let mut rows = conn.query(&sql, params).await?;
  let mut out = HashMap::new();
  while let Some(row) = rows.next().await? {
    let command: String = row.get(0)?;
    let bias: f32 = row.get::<f64>(1)? as f32;
    out.insert(command, bias);
  }
  Ok(out)
}

async fn load_head_biases(
  conn: &Connection,
  heads: &HashSet<String>,
) -> Result<HashMap<String, f32>> {
  if heads.is_empty() {
    return Ok(HashMap::new());
  }

  let mut placeholders = String::new();
  for idx in 0..heads.len() {
    if idx > 0 {
      placeholders.push(',');
    }
    placeholders.push('?');
  }
  let mut params: Vec<Value> = Vec::with_capacity(heads.len());
  for head in heads {
    params.push(Value::from(head.clone()));
  }

  let sql = format!(
    "SELECT head, bias FROM online_head_bias WHERE head IN ({})",
    placeholders
  );
  let mut rows = conn.query(&sql, params).await?;
  let mut out = HashMap::new();
  while let Some(row) = rows.next().await? {
    let head: String = row.get(0)?;
    let bias: f32 = row.get::<f64>(1)? as f32;
    out.insert(head, bias);
  }
  Ok(out)
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
  ctx: &mut TrainStepContext<'_, '_>,
  input: &ExampleInput,
  negatives: usize,
) -> Result<()> {
  let pools = ctx
    .sampler
    .build_pools(
      ctx.conn,
      &input.example.shellname,
      &input.example.workspace_root,
      input.example.cwd.as_deref(),
      &input.recent_commands,
      input.example.positive_head_hash,
    )
    .await?;
  train_example_with_pools(ctx, &pools, &input.example, negatives).await
}

async fn train_replay(
  ctx: &mut TrainStepContext<'_, '_>,
  replay: &OnlineExample,
  negatives: usize,
) -> Result<()> {
  let pools = ctx
    .sampler
    .build_pools(
      ctx.conn,
      &replay.shellname,
      &replay.workspace_root,
      replay.cwd.as_deref(),
      &[],
      replay.positive_head_hash,
    )
    .await?;
  train_example_with_pools(ctx, &pools, replay, negatives).await
}

struct TrainStepContext<'a, 'g> {
  conn: &'a Connection,
  cache: &'a mut EmbeddingCache,
  group_scalars: &'a mut GroupScalarStore,
  bias_store: &'a mut BiasStore,
  sampler: &'a mut NegativeSampler<'g>,
  rng: &'a mut StdRng,
}

struct CommandVec {
  command: String,
  buckets: Vec<(u32, f32)>,
  log_q: f32,
}

async fn train_example_with_pools(
  ctx: &mut TrainStepContext<'_, '_>,
  pools: &SamplerPools,
  example: &OnlineExample,
  negatives: usize,
) -> Result<()> {
  let (negatives, log_q_pos) = ctx.sampler.sample_with_logq(
    pools,
    &example.shellname,
    example.positive_command_hash,
    negatives,
    ctx.rng,
  )?;

  // Hash command vectors for negatives (we keep the positive vector in the example).
  let mut command_vecs: Vec<CommandVec> = Vec::with_capacity(1 + negatives.len());
  command_vecs.push(CommandVec {
    command: example.positive_command.clone(),
    buckets: example.cmd_buckets.clone(),
    log_q: log_q_pos,
  });
  for neg in negatives {
    command_vecs.push(CommandVec {
      command: neg.command,
      buckets: neg.cmd_buckets,
      log_q: neg.log_q,
    });
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
  for command_vec in &command_vecs {
    for (b, _) in &command_vec.buckets {
      buckets.insert(*b);
    }
  }

  ctx
    .cache
    .preload(ctx.conn, buckets.into_iter(), example.now)
    .await?;

  // Resolve group scalars (lazy init in DB).
  let s_workspace = ctx
    .group_scalars
    .get_or_init(ctx.conn, GROUP_WORKSPACE_ROOT)
    .await?;
  let s_cwd = ctx.group_scalars.get_or_init(ctx.conn, GROUP_CWD).await?;
  let s_exit = ctx.group_scalars.get_or_init(ctx.conn, GROUP_EXIT).await?;
  let s_host = ctx.group_scalars.get_or_init(ctx.conn, GROUP_HOST).await?;
  let s_user = ctx.group_scalars.get_or_init(ctx.conn, GROUP_USER).await?;
  let s_timebucket = ctx
    .group_scalars
    .get_or_init(ctx.conn, GROUP_TIMEBUCKET)
    .await?;
  let s_session = ctx
    .group_scalars
    .get_or_init(ctx.conn, GROUP_SESSION)
    .await?;
  let s_recent_heads = ctx
    .group_scalars
    .get_or_init(ctx.conn, GROUP_RECENT_HEADS)
    .await?;
  let s_recent_flags = ctx
    .group_scalars
    .get_or_init(ctx.conn, GROUP_RECENT_FLAGS)
    .await?;

  // Build context vector and bucket weights used in the context tower.
  let mut u_ctx = vec![0.0f32; ctx.cache.dim];
  let mut ctx_weights: HashMap<u32, f32> = HashMap::new();
  let mut group_u_no_scalar: HashMap<&'static str, Vec<f32>> = HashMap::new();

  {
    let mut acc = ContextAccumulator {
      u_ctx: &mut u_ctx,
      ctx_weights: &mut ctx_weights,
      group_u_no_scalar: &mut group_u_no_scalar,
    };
    add_group_to_context(
      ctx.cache,
      &example.ctx_workspace,
      W_WORKSPACE_ROOT,
      s_workspace,
      GROUP_WORKSPACE_ROOT,
      &mut acc,
    );
    add_group_to_context(
      ctx.cache,
      &example.ctx_cwd,
      W_CWD,
      s_cwd,
      GROUP_CWD,
      &mut acc,
    );
    add_group_to_context(
      ctx.cache,
      &example.ctx_exit,
      W_EXIT,
      s_exit,
      GROUP_EXIT,
      &mut acc,
    );
    add_group_to_context(
      ctx.cache,
      &example.ctx_host,
      W_HOST,
      s_host,
      GROUP_HOST,
      &mut acc,
    );
    add_group_to_context(
      ctx.cache,
      &example.ctx_user,
      W_USER,
      s_user,
      GROUP_USER,
      &mut acc,
    );
    add_group_to_context(
      ctx.cache,
      &example.ctx_timebucket,
      W_TIMEBUCKET,
      s_timebucket,
      GROUP_TIMEBUCKET,
      &mut acc,
    );
    add_group_to_context(
      ctx.cache,
      &example.ctx_session,
      W_SESSION,
      s_session,
      GROUP_SESSION,
      &mut acc,
    );
    add_group_to_context(
      ctx.cache,
      &example.recent_heads,
      W_RECENT_HEADS,
      s_recent_heads,
      GROUP_RECENT_HEADS,
      &mut acc,
    );
    add_group_to_context(
      ctx.cache,
      &example.recent_flags,
      W_RECENT_FLAGS,
      s_recent_flags,
      GROUP_RECENT_FLAGS,
      &mut acc,
    );
  }

  // recent_args have a fixed base weight in v1.
  {
    let mut acc = FixedContextAccumulator {
      u_ctx: &mut u_ctx,
      ctx_weights: &mut ctx_weights,
    };
    add_fixed_group_to_context(ctx.cache, &example.recent_args, W_RECENT_ARGS, &mut acc);
  }

  let norm_ctx = l2_norm(&u_ctx);
  if norm_ctx <= 1e-8 {
    return Ok(());
  }
  let v_ctx = scale_vec(&u_ctx, 1.0 / norm_ctx);
  let ctx_bias_bucket = context_bias_bucket_from_example(example);
  let ctx_bias_value = if let Some((bucket, sign)) = ctx_bias_bucket {
    sign * ctx.bias_store.get_context(ctx.conn, bucket).await?
  } else {
    0.0
  };

  let mut ctx_weight_entries: Vec<(u32, f32)> = ctx_weights
    .iter()
    .map(|(bucket, weight)| (*bucket, *weight))
    .collect();
  ctx_weight_entries.sort_by(|a, b| a.0.cmp(&b.0));

  // Accumulate gradients (per bucket) for all samples in this event.
  let mut bucket_grads: HashMap<u32, Vec<f32>> = HashMap::new();
  let mut scalar_grads: HashMap<&'static str, f32> = HashMap::new();
  let mut command_bias_grads: HashMap<String, f32> = HashMap::new();
  let mut head_bias_grads: HashMap<String, f32> = HashMap::new();
  let mut context_bias_grads: HashMap<u32, f32> = HashMap::new();

  struct Candidate {
    label: f32,
    command: String,
    head: Option<String>,
    cmd_weights: Vec<(u32, f32)>,
    v_cmd: Vec<f32>,
    norm_cmd: f32,
    s0: f32,
    logit: f32,
  }

  let mut candidates: Vec<Candidate> = Vec::with_capacity(command_vecs.len());
  for (idx, command_vec) in command_vecs.iter().enumerate() {
    let label = if idx == 0 { 1.0 } else { 0.0 };
    let (u_cmd, cmd_weights) = command_tower_vector(ctx.cache, &command_vec.buckets);
    let norm_cmd = l2_norm(&u_cmd);
    if norm_cmd <= 1e-8 {
      continue;
    }
    let v_cmd = scale_vec(&u_cmd, 1.0 / norm_cmd);
    let s0 = dot(&v_ctx, &v_cmd);
    let head = head_for_command(&example.shellname, &command_vec.command);
    let cmd_bias = ctx
      .bias_store
      .get_command(ctx.conn, &command_vec.command)
      .await?;
    let head_bias = match head.as_deref() {
      Some(head) => ctx.bias_store.get_head(ctx.conn, head).await?,
      None => 0.0,
    };
    let logit = s0 + cmd_bias + head_bias + ctx_bias_value - command_vec.log_q;
    candidates.push(Candidate {
      label,
      command: command_vec.command.clone(),
      head,
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

    for (bucket, w) in &ctx_weight_entries {
      add_scaled_bucket_grad(&mut bucket_grads, *bucket, &grad_u_ctx, *w);
    }
    for (bucket, w) in &candidate.cmd_weights {
      add_scaled_bucket_grad(&mut bucket_grads, *bucket, &grad_u_cmd, *w);
    }

    // Ensure stable iteration order for scalar gradients.
    for group in [
      GROUP_WORKSPACE_ROOT,
      GROUP_CWD,
      GROUP_EXIT,
      GROUP_HOST,
      GROUP_USER,
      GROUP_TIMEBUCKET,
      GROUP_SESSION,
      GROUP_RECENT_HEADS,
      GROUP_RECENT_FLAGS,
    ] {
      let Some(u_no_scalar) = group_u_no_scalar.get(group) else {
        continue;
      };
      let grad = dot(&grad_u_ctx, u_no_scalar);
      *scalar_grads.entry(group).or_insert(0.0) += grad;
    }
    *command_bias_grads
      .entry(candidate.command.clone())
      .or_insert(0.0) += g;
    if let Some(head) = candidate.head.as_deref() {
      *head_bias_grads.entry(head.to_string()).or_insert(0.0) += g;
    }
    if let Some((bucket, sign)) = ctx_bias_bucket {
      *context_bias_grads.entry(bucket).or_insert(0.0) += g * sign;
    }
  }

  // Apply embedding updates.
  for (bucket, grad) in bucket_grads {
    ctx.cache.apply_adagrad_update(bucket, &grad, LR_EMBED)?;
  }

  // Apply scalar updates with L2 reg toward 1.0.
  for (group, grad) in scalar_grads {
    let current = ctx.group_scalars.get_or_init(ctx.conn, group).await?;
    let reg = L2_GROUP_SCALAR * (current - 1.0);
    let updated = (current - LR_GROUP_SCALAR * (grad + reg)).clamp(0.05, 20.0);
    ctx.group_scalars.set(group, updated);
  }

  // Apply bias updates with L2 reg toward 0.
  for (command, grad) in command_bias_grads {
    let current = ctx.bias_store.get_command(ctx.conn, &command).await?;
    let reg = L2_BIAS * current;
    let updated = (current - LR_BIAS * (grad + reg)).clamp(-BIAS_CLAMP, BIAS_CLAMP);
    ctx.bias_store.set_command(command, updated);
  }
  for (head, grad) in head_bias_grads {
    let current = ctx.bias_store.get_head(ctx.conn, &head).await?;
    let reg = L2_BIAS * current;
    let updated = (current - LR_BIAS * (grad + reg)).clamp(-BIAS_CLAMP, BIAS_CLAMP);
    ctx.bias_store.set_head(head, updated);
  }
  for (bucket, grad) in context_bias_grads {
    let current = ctx.bias_store.get_context(ctx.conn, bucket).await?;
    let reg = L2_BIAS * current;
    let updated = (current - LR_BIAS * (grad + reg)).clamp(-BIAS_CLAMP, BIAS_CLAMP);
    ctx.bias_store.set_context(bucket, updated);
  }

  Ok(())
}

struct ContextAccumulator<'a> {
  u_ctx: &'a mut [f32],
  ctx_weights: &'a mut HashMap<u32, f32>,
  group_u_no_scalar: &'a mut HashMap<&'static str, Vec<f32>>,
}

struct FixedContextAccumulator<'a> {
  u_ctx: &'a mut [f32],
  ctx_weights: &'a mut HashMap<u32, f32>,
}

fn add_group_to_context(
  cache: &EmbeddingCache,
  buckets: &[(u32, f32)],
  base_weight: f32,
  scalar: f32,
  group: &'static str,
  acc: &mut ContextAccumulator<'_>,
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
      *acc.ctx_weights.entry(*bucket).or_insert(0.0) += base_weight * scalar * (*w);
    }
  }
  for (dst, value) in acc.u_ctx.iter_mut().zip(u.iter()) {
    *dst += scalar * value;
  }
  acc.group_u_no_scalar.insert(group, u);
}

fn add_fixed_group_to_context(
  cache: &EmbeddingCache,
  buckets: &[(u32, f32)],
  base_weight: f32,
  acc: &mut FixedContextAccumulator<'_>,
) {
  if buckets.is_empty() || base_weight == 0.0 {
    return;
  }
  for (bucket, w) in buckets {
    if let Some(e) = cache.get(*bucket) {
      for (idx, v) in e.iter().enumerate() {
        acc.u_ctx[idx] += base_weight * (*w) * (*v);
      }
      *acc.ctx_weights.entry(*bucket).or_insert(0.0) += base_weight * (*w);
    }
  }
}

fn command_tower_vector(
  cache: &EmbeddingCache,
  buckets: &[(u32, f32)],
) -> (Vec<f32>, Vec<(u32, f32)>) {
  let mut u = vec![0.0f32; cache.dim];
  let mut weights = HashMap::<u32, f32>::new();
  for (bucket, w) in buckets {
    if let Some(e) = cache.get(*bucket) {
      for (idx, v) in e.iter().enumerate() {
        u[idx] += (*w) * (*v);
      }
      *weights.entry(*bucket).or_insert(0.0) += *w;
    }
  }
  let mut out: Vec<(u32, f32)> = weights.into_iter().collect();
  out.sort_by(|a, b| a.0.cmp(&b.0));
  (u, out)
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
  crate::hash_util::stable_char_ngrams_buckets(token, bucket_count, scratch_indices, scratch);
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

fn head_for_command(shellname: &str, command: &str) -> Option<String> {
  let tokens = tokenize_index(shellname, command);
  let parts = extract_command_parts(command, &tokens)?;
  let head = parts.head.trim();
  if head.is_empty() {
    return None;
  }
  Some(head.to_string())
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
      out.push(normalize_command_whitespace(&command));
    } else {
      out.push(normalize_command_whitespace(&expanded));
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

  fn append_multi_row_placeholders(sql: &mut String, rows: usize) {
    for idx in 0..rows {
      if idx > 0 {
        sql.push(',');
      }
      sql.push('?');
    }
  }

  async fn preload(
    &mut self,
    conn: &Connection,
    buckets: impl Iterator<Item = u32>,
    now: i64,
  ) -> Result<()> {
    // Avoid per-bucket DB queries; load unknown buckets in IN(...) batches.
    // This is a major hotspot during bulk training/import.
    let mut missing: Vec<u32> = Vec::new();
    for bucket in buckets {
      if self.map.contains_key(&bucket) {
        continue;
      }
      missing.push(bucket);
    }
    if missing.is_empty() {
      return Ok(());
    }

    missing.sort_unstable();
    missing.dedup();

    // Keep well under SQLite's max variable limit (often 999).
    const MAX_BUCKET_PARAMS: usize = 500;
    let mut loaded: HashSet<u32> = HashSet::new();

    for chunk in missing.chunks(MAX_BUCKET_PARAMS) {
      let mut sql =
        String::from("SELECT bucket, vec, opt_state FROM online_token_embedding WHERE bucket IN (");
      Self::append_multi_row_placeholders(&mut sql, chunk.len());
      sql.push(')');

      let mut params: Vec<libsql::Value> = Vec::with_capacity(chunk.len());
      for bucket in chunk {
        params.push(libsql::Value::Integer(*bucket as i64));
      }

      let mut rows = conn.query(&sql, params).await?;
      while let Some(row) = rows.next().await? {
        let bucket: i64 = row.get(0)?;
        let Ok(bucket) = u32::try_from(bucket) else {
          continue;
        };
        let vec_blob: Vec<u8> = row.get(1)?;
        let opt_blob: Option<Vec<u8>> = row.get(2)?;
        let vec = decode_f32_blob(&vec_blob, self.dim).ok_or_else(|| {
          ZageError::ConfigError(format!("invalid embedding blob for bucket {bucket}"))
        })?;
        let acc = match opt_blob {
          Some(blob) => decode_f32_blob(&blob, self.dim).unwrap_or_else(|| vec![0.0; self.dim]),
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
        loaded.insert(bucket);
      }
    }

    for bucket in missing {
      if loaded.contains(&bucket) {
        continue;
      }
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
    if row.vec.len() != self.dim || row.acc.len() != self.dim || grad.len() != self.dim {
      return Err(ZageError::ConfigError(format!(
        "dimension mismatch for bucket {bucket}"
      )));
    }
    for (idx, g) in grad.iter().enumerate() {
      let g = *g;
      row.acc[idx] += g * g;
      row.vec[idx] -= lr * g / (row.acc[idx] + ADAGRAD_EPS).sqrt();
    }
    row.dirty = true;
    Ok(())
  }

  async fn flush(&mut self, conn: &Connection, now: i64) -> Result<()> {
    let stmt = conn
      .prepare(
        "INSERT INTO online_token_embedding (bucket, vec, opt_state, updated_at)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(bucket) DO UPDATE SET
           vec = excluded.vec,
           opt_state = excluded.opt_state,
           updated_at = excluded.updated_at",
      )
      .await?;
    for (bucket, row) in self.map.iter_mut() {
      if !row.dirty {
        continue;
      }
      let vec_blob = encode_f32_blob(&row.vec);
      let acc_blob = encode_f32_blob(&row.acc);
      stmt
        .execute(libsql::params![*bucket as i64, vec_blob, acc_blob, now])
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
      // Don't write during training; flush() will persist defaults in a single transaction.
      self.dirty.insert(group);
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

#[derive(Default)]
struct BiasStore {
  command: HashMap<String, f32>,
  head: HashMap<String, f32>,
  context: HashMap<u32, f32>,
  dirty_command: HashSet<String>,
  dirty_head: HashSet<String>,
  dirty_context: HashSet<u32>,
}

impl BiasStore {
  async fn preload_all(&mut self, conn: &Connection) -> Result<()> {
    self.command.clear();
    self.head.clear();
    self.context.clear();
    self.dirty_command.clear();
    self.dirty_head.clear();
    self.dirty_context.clear();

    let mut rows = conn
      .query("SELECT command, bias FROM online_command_bias", ())
      .await?;
    while let Some(row) = rows.next().await? {
      let command: String = row.get(0)?;
      let bias: f64 = row.get(1)?;
      self.command.insert(command, bias as f32);
    }

    let mut rows = conn
      .query("SELECT head, bias FROM online_head_bias", ())
      .await?;
    while let Some(row) = rows.next().await? {
      let head: String = row.get(0)?;
      let bias: f64 = row.get(1)?;
      self.head.insert(head, bias as f32);
    }

    let mut rows = conn
      .query("SELECT bucket, bias FROM online_context_bias", ())
      .await?;
    while let Some(row) = rows.next().await? {
      let bucket: i64 = row.get(0)?;
      let bias: f64 = row.get(1)?;
      if let Ok(bucket) = u32::try_from(bucket) {
        self.context.insert(bucket, bias as f32);
      }
    }

    Ok(())
  }

  async fn get_command(&mut self, conn: &Connection, command: &str) -> Result<f32> {
    if let Some(value) = self.command.get(command).copied() {
      return Ok(value);
    }
    let mut rows = conn
      .query(
        "SELECT bias FROM online_command_bias WHERE command = ?",
        libsql::params![command.to_string()],
      )
      .await?;
    let value = if let Some(row) = rows.next().await? {
      row.get::<f64>(0)? as f32
    } else {
      0.0
    };
    self.command.insert(command.to_string(), value);
    Ok(value)
  }

  async fn get_head(&mut self, conn: &Connection, head: &str) -> Result<f32> {
    if let Some(value) = self.head.get(head).copied() {
      return Ok(value);
    }
    let mut rows = conn
      .query(
        "SELECT bias FROM online_head_bias WHERE head = ?",
        libsql::params![head.to_string()],
      )
      .await?;
    let value = if let Some(row) = rows.next().await? {
      row.get::<f64>(0)? as f32
    } else {
      0.0
    };
    self.head.insert(head.to_string(), value);
    Ok(value)
  }

  async fn get_context(&mut self, conn: &Connection, bucket: u32) -> Result<f32> {
    if let Some(value) = self.context.get(&bucket).copied() {
      return Ok(value);
    }
    let mut rows = conn
      .query(
        "SELECT bias FROM online_context_bias WHERE bucket = ?",
        libsql::params![bucket as i64],
      )
      .await?;
    let value = if let Some(row) = rows.next().await? {
      row.get::<f64>(0)? as f32
    } else {
      0.0
    };
    self.context.insert(bucket, value);
    Ok(value)
  }

  fn set_command(&mut self, command: String, value: f32) {
    self.command.insert(command.clone(), value);
    self.dirty_command.insert(command);
  }

  fn set_head(&mut self, head: String, value: f32) {
    self.head.insert(head.clone(), value);
    self.dirty_head.insert(head);
  }

  fn set_context(&mut self, bucket: u32, value: f32) {
    self.context.insert(bucket, value);
    self.dirty_context.insert(bucket);
  }

  async fn flush(&mut self, conn: &Connection, now: i64) -> Result<()> {
    for command in self.dirty_command.drain() {
      let Some(value) = self.command.get(&command).copied() else {
        continue;
      };
      conn
        .execute(
          "INSERT INTO online_command_bias (command, bias, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT(command) DO UPDATE SET
             bias = excluded.bias,
             updated_at = excluded.updated_at",
          libsql::params![command, value as f64, now],
        )
        .await?;
    }
    for head in self.dirty_head.drain() {
      let Some(value) = self.head.get(&head).copied() else {
        continue;
      };
      conn
        .execute(
          "INSERT INTO online_head_bias (head, bias, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT(head) DO UPDATE SET
             bias = excluded.bias,
             updated_at = excluded.updated_at",
          libsql::params![head, value as f64, now],
        )
        .await?;
    }
    for bucket in self.dirty_context.drain() {
      let Some(value) = self.context.get(&bucket).copied() else {
        continue;
      };
      conn
        .execute(
          "INSERT INTO online_context_bias (bucket, bias, updated_at)
           VALUES (?, ?, ?)
           ON CONFLICT(bucket) DO UPDATE SET
             bias = excluded.bias,
             updated_at = excluded.updated_at",
          libsql::params![bucket as i64, value as f64, now],
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
  use crate::core::RankingWeights;
  use crate::db::{init, insert_invocation, open_db, update_stats_for_invocation};
  use crate::predict::verifier::{TestConfig, suggest_for_test};
  use std::fs;
  use std::path::Path;
  use tempfile::NamedTempFile;
  use tempfile::TempDir;

  async fn score_for_context_and_command(
    conn: &Connection,
    inv: &Invocation,
    command: &str,
  ) -> Result<f32> {
    let config = OnlineModelConfig::default();
    let mut workspace_root_cache: HashMap<String, String> = HashMap::new();
    let input = build_example_input(
      conn,
      inv,
      unix_now(),
      config.window,
      config.bucket_count,
      &mut workspace_root_cache,
    )
    .await?;
    let Some(input) = input else {
      return Err(ZageError::ConfigError("missing example".to_string()));
    };
    let workspace_root = input.example.workspace_root.clone();
    let commands = vec![command.to_string()];
    let scores = score_commands(
      conn,
      OnlineScoreContext {
        shellname: input.example.shellname.as_str(),
        workspace_root: workspace_root.as_str(),
        cwd: input.example.cwd.as_deref(),
        hostname: inv.hostname.as_deref(),
        username: inv.username.as_deref(),
        exit_status: inv.exit_status,
        session_id: Some(inv.session_id),
        unix_timestamp: input.example.now,
        recent_commands: &input.recent_commands,
        window: config.window,
      },
      &commands,
      &config,
    )
    .await?;
    Ok(scores.first().copied().unwrap_or(0.0))
  }

  #[tokio::test]
  async fn retrieval_embedding_matches_scoring_vector() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    let config = OnlineModelConfig::default();
    let recent_commands = vec!["git status".to_string()];
    let command = "git status".to_string();
    let make_ctx = || OnlineScoreContext {
      shellname: "zsh",
      workspace_root: "",
      cwd: Some("/tmp"),
      hostname: Some("host"),
      username: Some("user"),
      exit_status: Some(0),
      session_id: Some(1),
      unix_timestamp: 100,
      recent_commands: &recent_commands,
      window: config.window,
    };

    let ctx_embedding = context_embedding_for_retrieval(&db.conn, make_ctx(), &config).await?;
    let ctx_embedding = ctx_embedding.expect("context embedding");

    let command_pairs = vec![(command.clone(), "zsh".to_string())];
    let cmd_embeddings = command_embeddings_for_commands(&db.conn, &command_pairs, &config).await?;

    let commands = vec![command.clone()];
    let scores = score_commands(&db.conn, make_ctx(), &commands, &config).await?;
    let score = scores.first().copied().unwrap_or(0.0);
    let dot_score = dot(&ctx_embedding, &cmd_embeddings[0]);

    assert!((score - dot_score).abs() < 1e-5);
    Ok(())
  }

  #[derive(Debug, Clone)]
  struct EvalMetrics {
    mrr_at_k: f64,
    recall_at_k: f64,
    coverage_at_k: f64,
    leakage_rate: f64,
    total: usize,
  }

  fn workspace_key(invocation: &Invocation) -> String {
    let Some(cwd) = invocation.working_directory.as_deref() else {
      return String::new();
    };
    resolve_workspace_root(Some(cwd))
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
    let workspace_a = root.path().join("workspace-a");
    let workspace_b = root.path().join("workspace-b");
    fs::create_dir_all(&workspace_a)?;
    fs::create_dir_all(&workspace_b)?;
    std::fs::write(workspace_a.join("Cargo.toml"), "[workspace]\n")?;
    std::fs::write(workspace_b.join("Cargo.toml"), "[workspace]\n")?;

    let workspace_a_work = workspace_a.join("src");
    let workspace_b_work = workspace_b.join("work");
    fs::create_dir_all(&workspace_a_work)?;
    fs::create_dir_all(&workspace_b_work)?;

    let mut invocations = Vec::new();
    let mut ts = 1_700_000_000i64;

    for _ in 0..4 {
      push_invocation(
        &mut invocations,
        "git status",
        &workspace_b_work,
        2,
        &mut ts,
      );
      push_invocation(&mut invocations, "git log", &workspace_b_work, 2, &mut ts);
      push_invocation(
        &mut invocations,
        "git status",
        &workspace_b_work,
        2,
        &mut ts,
      );
      push_invocation(&mut invocations, "git show", &workspace_b_work, 2, &mut ts);
    }

    for _ in 0..20 {
      push_invocation(
        &mut invocations,
        "git status",
        &workspace_a_work,
        1,
        &mut ts,
      );
      push_invocation(&mut invocations, "git diff", &workspace_a_work, 1, &mut ts);
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
          include_debug: false,
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

        if let Some(rank) = rank
          && rank <= max_results
        {
          mrr_sum += 1.0 / rank as f64;
          recall_hits += 1;
        }

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
  async fn online_training_updates_command_and_head_biases() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    let mut invocations = Vec::new();
    let mut ts = 10i64;
    for _ in 0..40 {
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
    }
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

    train_on_invocations(&db.conn, &invocations).await?;

    let mut rows = db
      .conn
      .query(
        "SELECT bias FROM online_command_bias WHERE command = ?",
        libsql::params!["git status".to_string()],
      )
      .await?;
    let row = rows
      .next()
      .await?
      .ok_or_else(|| ZageError::ConfigError("missing command bias".to_string()))?;
    let bias: f64 = row.get(0)?;
    assert!(bias > 0.0);

    let mut rows = db
      .conn
      .query(
        "SELECT bias FROM online_head_bias WHERE head = ?",
        libsql::params!["git".to_string()],
      )
      .await?;
    let row = rows
      .next()
      .await?
      .ok_or_else(|| ZageError::ConfigError("missing head bias".to_string()))?;
    let bias: f64 = row.get(0)?;
    assert!(bias > 0.0);
    Ok(())
  }

  #[tokio::test]
  async fn command_bias_affects_scoring() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    db.conn
      .execute(
        "INSERT INTO online_command_bias (command, bias, updated_at) VALUES (?, ?, ?)",
        libsql::params!["alpha".to_string(), 5.0f64, 1i64],
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO online_command_bias (command, bias, updated_at) VALUES (?, ?, ?)",
        libsql::params!["beta".to_string(), -5.0f64, 1i64],
      )
      .await?;

    let config = OnlineModelConfig::default();
    let recent_commands: Vec<String> = Vec::new();
    let commands = vec!["alpha".to_string(), "beta".to_string()];
    let scores = score_commands(
      &db.conn,
      OnlineScoreContext {
        shellname: "zsh",
        workspace_root: "",
        cwd: None,
        hostname: None,
        username: None,
        exit_status: None,
        session_id: None,
        unix_timestamp: 0,
        recent_commands: &recent_commands,
        window: 0,
      },
      &commands,
      &config,
    )
    .await?;
    assert!(scores.len() == 2);
    assert!(scores[0] > scores[1]);
    Ok(())
  }

  #[tokio::test]
  async fn context_bias_affects_scoring() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    let config = OnlineModelConfig::default();
    let recent_commands: Vec<String> = Vec::new();
    let commands = vec!["alpha".to_string()];
    let make_ctx = || OnlineScoreContext {
      shellname: "zsh",
      workspace_root: "/workspace",
      cwd: Some("/workspace"),
      hostname: Some("host"),
      username: Some("user"),
      exit_status: Some(0),
      session_id: Some(1),
      unix_timestamp: 0,
      recent_commands: &recent_commands,
      window: 0,
    };

    let scores_no_bias = score_commands(&db.conn, make_ctx(), &commands, &config).await?;

    let mut scratch_indices = Vec::new();
    let mut scratch_buckets = Vec::new();
    let context_maps = build_context_maps(
      make_ctx(),
      config.bucket_count,
      &mut scratch_indices,
      &mut scratch_buckets,
    );
    let (bucket, sign) = context_bias_bucket_from_maps(&context_maps).expect("context bias bucket");

    let bias = 2.5f32;
    db.conn
      .execute(
        "INSERT INTO online_context_bias (bucket, bias, updated_at) VALUES (?, ?, ?)",
        libsql::params![bucket as i64, bias as f64, 1i64],
      )
      .await?;

    let scores_with_bias = score_commands(&db.conn, make_ctx(), &commands, &config).await?;
    let delta = scores_with_bias[0] - scores_no_bias[0];
    assert!((delta - sign * bias).abs() < 1e-4);
    Ok(())
  }

  #[tokio::test]
  async fn prequential_eval_tracks_online_mrr_and_leakage() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let invocations = build_fixture(&temp_dir)?;

    let max_results = 5;
    let recent_limit = OnlineModelConfig::default().window;
    let first_a = invocations
      .iter()
      .position(|inv| inv.session_id == 1)
      .unwrap_or(0);
    let eval_start = first_a + recent_limit + 2;

    let weights = RankingWeights {
      recency: 0.10,
      frequency: 0.0,
      transition: 0.0,
      context: 0.0,
      sequence: 0.0,
      similarity: 0.0,
    };

    let online = run_prequential(
      &invocations,
      eval_start,
      max_results,
      recent_limit,
      weights.clone(),
    )
    .await?;

    eprintln!(
      "prequential@{} online:   mrr={:.3} recall={:.3} coverage={:.3} leakage={:.3} (n={})",
      max_results,
      online.mrr_at_k,
      online.recall_at_k,
      online.coverage_at_k,
      online.leakage_rate,
      online.total
    );
    assert!(
      online.mrr_at_k >= 0.5,
      "expected online model MRR@{} to be >= 0.50 (got {:.3})",
      max_results,
      online.mrr_at_k
    );
    assert!(
      online.recall_at_k >= 0.9,
      "expected online model recall@{} to be >= 0.90 (got {:.3})",
      max_results,
      online.recall_at_k
    );
    assert!(
      online.leakage_rate <= f64::EPSILON,
      "expected zero leakage on fixture"
    );
    Ok(())
  }
}
