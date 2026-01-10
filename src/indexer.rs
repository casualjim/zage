use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use libsql::Connection;
use serde_json;
use tracing::info;

use crate::Result;
use crate::phase::{PhaseConfig, PhaseSample, features_from_tokens, train_phase_predictor};
use crate::predict::aliases::{expand_alias, load_aliases};
use crate::repo::find_repo_root;
use crate::tokenize::{extract_command_parts, normalize_token, tokenize_index};

#[derive(Debug, Default)]
pub struct IndexReport {
  pub commands: usize,
  pub transitions: usize,
  pub contexts: usize,
  pub token_cache: usize,
  pub phase_stats: usize,
}

#[derive(Debug, Default)]
struct Stat {
  freq: i64,
  last_seen: i64,
}

#[derive(Debug, Default)]
struct ArgStat {
  freq: i64,
  last_seen: i64,
  arg_norm: String,
}

#[derive(Debug, Default)]
struct PhaseStat {
  freq: i64,
  last_seen: i64,
  confidence_sum: f64,
}

type ContextKey = (String, Option<String>, Option<String>, Option<String>);

pub async fn rebuild_stats(conn: &Connection, max_commands: Option<usize>) -> Result<IndexReport> {
  let mut command_stats: HashMap<String, Stat> = HashMap::new();
  let mut transition_stats: HashMap<(String, Option<i64>, String), Stat> = HashMap::new();
  let mut repo_command_stats: HashMap<(String, String), Stat> = HashMap::new();
  let mut repo_transition_stats: HashMap<(String, String, Option<i64>, String), Stat> =
    HashMap::new();
  let mut context_stats: HashMap<ContextKey, Stat> = HashMap::new();
  let mut arg_stats: HashMap<(String, String, String, i64, String), ArgStat> = HashMap::new();
  let mut arg_stats_any: HashMap<(String, String, String, String), ArgStat> = HashMap::new();
  let mut flag_stats: HashMap<(String, String, String, String), Stat> = HashMap::new();
  let mut env_stats: HashMap<(String, String, String, String, String), Stat> = HashMap::new();
  let mut token_cache: HashMap<String, (Vec<String>, Vec<String>)> = HashMap::new();
  let mut phase_stats: HashMap<(String, String), PhaseStat> = HashMap::new();
  let mut command_shell: HashMap<String, String> = HashMap::new();
  let phase_config = PhaseConfig::load()?;
  let mut phase_samples: Vec<PhaseSample> = Vec::new();
  let mut phase_unlabeled: Vec<Vec<f64>> = Vec::new();

  let mut prev_command: Option<String> = None;
  let mut prev_exit_status: Option<i64> = None;
  let mut prev_repo_root: String = String::new();
  let mut processed: usize = 0;
  let progress_interval = 50_000usize;
  let max_unlabeled = 10_000usize;
  let aliases = load_aliases();

  let mut sql = String::from(
    "SELECT command, expanded_command, shellname, working_directory, hostname, username, exit_status, start_unix_timestamp
     FROM shell_history
     ORDER BY COALESCE(start_unix_timestamp, 0) ASC, id ASC",
  );
  if max_commands.is_some() {
    sql.push_str(" LIMIT ?");
  }

  let mut rows = if let Some(limit) = max_commands {
    conn.query(&sql, libsql::params![limit as i64]).await?
  } else {
    conn.query(&sql, ()).await?
  };

  while let Some(row) = rows.next().await? {
    let command = row.get::<String>(0)?;
    let expanded_command = row.get::<String>(1)?;
    let shellname = row.get::<String>(2)?;
    let working_directory = row.get::<Option<String>>(3)?;
    let hostname = row.get::<Option<String>>(4)?;
    let username = row.get::<Option<String>>(5)?;
    let exit_status = row.get::<Option<i64>>(6)?;
    let ts: Option<i64> = row.get(7)?;
    let ts = ts.unwrap_or(0);
    let repo_root = working_directory
      .as_deref()
      .and_then(find_repo_root)
      .unwrap_or_default();

    let stats_command = if !expanded_command.is_empty() {
      expanded_command
    } else {
      expand_alias(&command, &aliases).unwrap_or(command.clone())
    };

    update_stat(&mut command_stats, &stats_command, ts);
    update_stat_key(
      &mut repo_command_stats,
      (repo_root.clone(), stats_command.clone()),
      ts,
    );

    let ctx_key = (
      stats_command.clone(),
      working_directory.clone(),
      hostname.clone(),
      username.clone(),
    );
    update_stat_key(&mut context_stats, ctx_key, ts);

    if let Some(prev) = &prev_command {
      update_stat_key(
        &mut transition_stats,
        (prev.clone(), prev_exit_status, stats_command.clone()),
        ts,
      );
      update_stat_key(
        &mut transition_stats,
        (prev.clone(), None, stats_command.clone()),
        ts,
      );
      update_stat_key(
        &mut repo_transition_stats,
        (
          prev_repo_root.clone(),
          prev.clone(),
          prev_exit_status,
          stats_command.clone(),
        ),
        ts,
      );
      update_stat_key(
        &mut repo_transition_stats,
        (
          prev_repo_root.clone(),
          prev.clone(),
          None,
          stats_command.clone(),
        ),
        ts,
      );
    }

    command_shell.insert(stats_command.clone(), shellname.clone());

    let tokens = tokenize_index(&shellname, &stats_command);
    if !token_cache.contains_key(&stats_command) {
      let raw = tokens.iter().map(|t| t.raw.clone()).collect();
      let normalized = tokens.iter().map(|t| t.normalized.clone()).collect();
      token_cache.insert(stats_command.clone(), (raw, normalized));
    }

    if phase_config.labels().len() > 1 {
      let features = features_from_tokens(&tokens, phase_config.hash_size());
      if let Some(label) = phase_config.match_label(&stats_command) {
        phase_samples.push(PhaseSample { features, label });
      } else if phase_unlabeled.len() < max_unlabeled {
        phase_unlabeled.push(features);
      }
    }
    if let Some(parts) = extract_command_parts(&stats_command, &tokens) {
      let mut flags = parts.flags;
      flags.sort();
      let flags_json = serde_json::to_string(&flags)?;
      for flag in &flags {
        let flag_norm = normalize_token(flag);
        update_stat_key(
          &mut flag_stats,
          (
            repo_root.clone(),
            parts.head.clone(),
            flag.clone(),
            flag_norm,
          ),
          ts,
        );
      }
      for env in &parts.env {
        let env_key = env.raw.split('=').next().unwrap_or_default().to_string();
        update_stat_key(
          &mut env_stats,
          (
            repo_root.clone(),
            parts.head.clone(),
            env_key,
            env.raw.clone(),
            env.normalized.clone(),
          ),
          ts,
        );
      }
      for (idx, arg) in parts.args.iter().enumerate() {
        update_arg_stat(
          &mut arg_stats,
          (
            repo_root.clone(),
            parts.head.clone(),
            flags_json.clone(),
            idx as i64,
            arg.raw.clone(),
          ),
          &arg.normalized,
          ts,
        );
        update_arg_stat(
          &mut arg_stats_any,
          (
            repo_root.clone(),
            parts.head.clone(),
            flags_json.clone(),
            arg.raw.clone(),
          ),
          &arg.normalized,
          ts,
        );
      }
    }

    prev_command = Some(stats_command);
    prev_exit_status = exit_status;
    prev_repo_root = repo_root;
    processed += 1;
    if processed.is_multiple_of(progress_interval) {
      info!("Indexed {} commands so far", processed);
    }
  }

  if processed == 0 {
    return Ok(IndexReport::default());
  }

  let phase_predictor = train_phase_predictor(&phase_config, phase_samples, phase_unlabeled);
  let phase_labels: Vec<String> = phase_predictor
    .as_ref()
    .map(|predictor| predictor.labels().to_vec())
    .unwrap_or_else(|| phase_config.labels().to_vec());

  if phase_labels.len() > 1 {
    for (command, stat) in &command_stats {
      let shellname = command_shell
        .get(command)
        .map(|s| s.as_str())
        .unwrap_or("sh");
      let tokens = tokenize_index(shellname, command);
      let parts = extract_command_parts(command, &tokens);
      let Some(parts) = parts else {
        continue;
      };
      let features = features_from_tokens(&tokens, phase_config.hash_size());
      let probs = if let Some(predictor) = &phase_predictor {
        predictor.predict(&features)
      } else {
        phase_config.pattern_distribution(command)
      };
      if probs.len() != phase_labels.len() {
        continue;
      }
      for (idx, phase) in phase_labels.iter().enumerate() {
        let prob = probs[idx] as f64;
        if prob <= 0.0 {
          continue;
        }
        let entry = phase_stats
          .entry((parts.head.clone(), phase.clone()))
          .or_default();
        entry.freq += stat.freq;
        if stat.last_seen > entry.last_seen {
          entry.last_seen = stat.last_seen;
        }
        entry.confidence_sum += prob * stat.freq as f64;
      }
    }
  }

  let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64;

  conn.execute("BEGIN", ()).await?;
  let write_result: Result<()> = async {
    conn.execute("DELETE FROM command_stats", ()).await?;
    conn.execute("DELETE FROM transition_stats", ()).await?;
    conn.execute("DELETE FROM repo_command_stats", ()).await?;
    conn.execute("DELETE FROM repo_transition_stats", ()).await?;
    conn.execute("DELETE FROM context_stats", ()).await?;
    conn.execute("DELETE FROM arg_stats", ()).await?;
    conn.execute("DELETE FROM arg_stats_any", ()).await?;
    conn.execute("DELETE FROM flag_stats", ()).await?;
    conn.execute("DELETE FROM env_stats", ()).await?;
    conn.execute("DELETE FROM token_cache", ()).await?;
    conn.execute("DELETE FROM phase_stats", ()).await?;

    for (command, stat) in &command_stats {
      conn
        .execute(
          "INSERT INTO command_stats (command, freq, last_seen) VALUES (?, ?, ?)",
          (command.clone(), stat.freq, stat.last_seen),
        )
        .await?;
    }

    for ((repo_root, command), stat) in &repo_command_stats {
      conn
        .execute(
          "INSERT INTO repo_command_stats (repo_root, command, freq, last_seen)
           VALUES (?, ?, ?, ?)",
          (repo_root.clone(), command.clone(), stat.freq, stat.last_seen),
        )
        .await?;
    }

    for ((prev, status, next), stat) in &transition_stats {
      conn
        .execute(
          "INSERT INTO transition_stats (prev_command, prev_exit_status, next_command, freq, last_seen)
           VALUES (?, ?, ?, ?, ?)",
          (prev.clone(), *status, next.clone(), stat.freq, stat.last_seen),
        )
        .await?;
    }

    for ((repo_root, prev, status, next), stat) in &repo_transition_stats {
      conn
        .execute(
          "INSERT INTO repo_transition_stats (repo_root, prev_command, prev_exit_status, next_command, freq, last_seen)
           VALUES (?, ?, ?, ?, ?, ?)",
          (
            repo_root.clone(),
            prev.clone(),
            *status,
            next.clone(),
            stat.freq,
            stat.last_seen,
          ),
        )
        .await?;
    }

    for ((command, wd, host, user), stat) in &context_stats {
      conn
        .execute(
          "INSERT INTO context_stats (command, working_directory, hostname, username, freq, last_seen)
           VALUES (?, ?, ?, ?, ?, ?)",
          (
            command.clone(),
            wd.clone(),
            host.clone(),
            user.clone(),
            stat.freq,
            stat.last_seen,
          ),
        )
        .await?;
    }

    for ((repo_root, head, flags_json, arg_index, arg_raw), stat) in &arg_stats {
      conn
        .execute(
          "INSERT INTO arg_stats (repo_root, command_head, flags_json, arg_index, arg_raw, arg_norm, freq, last_seen)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
          (
            repo_root.clone(),
            head.clone(),
            flags_json.clone(),
            *arg_index,
            arg_raw.clone(),
            stat.arg_norm.clone(),
            stat.freq,
            stat.last_seen,
          ),
        )
        .await?;
    }

    for ((repo_root, head, flags_json, arg_raw), stat) in &arg_stats_any {
      conn
        .execute(
          "INSERT INTO arg_stats_any (repo_root, command_head, flags_json, arg_raw, arg_norm, freq, last_seen)
           VALUES (?, ?, ?, ?, ?, ?, ?)",
          (
            repo_root.clone(),
            head.clone(),
            flags_json.clone(),
            arg_raw.clone(),
            stat.arg_norm.clone(),
            stat.freq,
            stat.last_seen,
          ),
        )
        .await?;
    }

    for ((repo_root, head, flag_raw, flag_norm), stat) in &flag_stats {
      conn
        .execute(
          "INSERT INTO flag_stats (repo_root, command_head, flag_raw, flag_norm, freq, last_seen)
           VALUES (?, ?, ?, ?, ?, ?)",
          (
            repo_root.clone(),
            head.clone(),
            flag_raw.clone(),
            flag_norm.clone(),
            stat.freq,
            stat.last_seen,
          ),
        )
        .await?;
    }

    for ((repo_root, head, env_key, env_raw, env_norm), stat) in &env_stats {
      conn
        .execute(
          "INSERT INTO env_stats (repo_root, command_head, env_key, env_raw, env_norm, freq, last_seen)
           VALUES (?, ?, ?, ?, ?, ?, ?)",
          (
            repo_root.clone(),
            head.clone(),
            env_key.clone(),
            env_raw.clone(),
            env_norm.clone(),
            stat.freq,
            stat.last_seen,
          ),
        )
        .await?;
    }

    for (command, (raw, norm)) in &token_cache {
      let raw_json = serde_json::to_string(raw)?;
      let norm_json = serde_json::to_string(norm)?;
      conn
        .execute(
          "INSERT INTO token_cache (command, tokens_json, normalized_json, updated_at)
           VALUES (?, ?, ?, ?)",
          (command.clone(), raw_json, norm_json, now),
        )
        .await?;
    }

    for ((command_head, phase), stat) in &phase_stats {
      let confidence = if stat.freq > 0 {
        stat.confidence_sum / stat.freq as f64
      } else {
        0.0
      };
      conn
        .execute(
          "INSERT INTO phase_stats (command_head, phase, confidence, freq, last_seen)
           VALUES (?, ?, ?, ?, ?)",
          (
            command_head.clone(),
            phase.clone(),
            confidence,
            stat.freq,
            stat.last_seen,
          ),
        )
        .await?;
    }

    Ok(())
  }
  .await;

  if let Err(err) = write_result {
    let _ = conn.execute("ROLLBACK", ()).await;
    return Err(err);
  }
  conn.execute("COMMIT", ()).await?;

  Ok(IndexReport {
    commands: command_stats.len(),
    transitions: transition_stats.len(),
    contexts: context_stats.len(),
    token_cache: token_cache.len(),
    phase_stats: phase_stats.len(),
  })
}

fn update_stat(map: &mut HashMap<String, Stat>, key: &str, ts: i64) {
  let entry = map.entry(key.to_string()).or_default();
  entry.freq += 1;
  if ts > entry.last_seen {
    entry.last_seen = ts;
  }
}

fn update_stat_key<K: std::hash::Hash + Eq>(map: &mut HashMap<K, Stat>, key: K, ts: i64) {
  let entry = map.entry(key).or_default();
  entry.freq += 1;
  if ts > entry.last_seen {
    entry.last_seen = ts;
  }
}

fn update_arg_stat<K: std::hash::Hash + Eq>(
  map: &mut HashMap<K, ArgStat>,
  key: K,
  arg_norm: &str,
  ts: i64,
) {
  let entry = map.entry(key).or_default();
  entry.freq += 1;
  if ts > entry.last_seen {
    entry.last_seen = ts;
  }
  if entry.arg_norm != arg_norm {
    entry.arg_norm = arg_norm.to_string();
  }
}
