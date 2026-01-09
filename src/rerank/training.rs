use std::collections::{HashMap, VecDeque};
use std::fs;

use gbrt_rs::boosting::GBRTConfig;
use gbrt_rs::{Dataset, GBRTModel, ModelIO};
use libsql::Connection;
use rand::rng;
use rand::seq::SliceRandom;
use serde_json;

use crate::repo::{find_repo_root, read_git_branch};
use crate::shell_history::Invocation;
use crate::tokenize::normalized_tokens;
use crate::{Result, ZageError};

use super::calibration::{CalibrationParams, fit_platt, fit_stack, sigmoid};
use super::config::{TrainConfig, TrainReport};
use super::features::{
  ContextWindow, TrainingStats, build_feature_matrix, build_feature_vector, command_head,
  feature_names, normalize_sequence_command, tier1_score_from_stats,
};
use super::model::{ModelStatus, clear_model_cache, model_location, timestamp_bucket, unix_now};

fn invocation_command(invocation: &Invocation) -> &str {
  if invocation.expanded_command.is_empty() {
    invocation.command.as_str()
  } else {
    invocation.expanded_command.as_str()
  }
}

pub async fn train_model(conn: &Connection, config: TrainConfig) -> Result<TrainReport> {
  let invocations = load_invocations(conn, Some(config.max_samples)).await?;
  if invocations.len() < config.min_history {
    return Err(ZageError::ConfigError(format!(
      "need at least {} history entries to train (have {})",
      config.min_history,
      invocations.len()
    )));
  }

  let split_idx = ((invocations.len() as f64) * 0.9).round() as usize;
  let (train_set, val_set) = invocations.split_at(split_idx.max(1));

  let phase_config = crate::phase::PhaseConfig::load().ok();
  let mut stats = TrainingStats::default();
  let mut recent = VecDeque::new();

  let mut features: Vec<Vec<f64>> = Vec::new();
  let mut labels: Vec<f64> = Vec::new();
  let mut pairs = 0usize;

  let history_window = config.min_history.max(1);

  for invocation in train_set {
    if recent.len() < history_window {
      update_training_stats(&mut stats, invocation, None, None);
      recent.push_back(invocation.clone());
      continue;
    }

    let context = build_context(&recent, invocation, phase_config.as_ref());
    if let Some(pos) = build_feature_vector(invocation_command(invocation), &context, &stats) {
      features.push(pos.values);
      labels.push(1.0);

      let negatives = sample_negatives(
        &stats,
        &pos.head,
        invocation.session_id,
        config.negatives_per_pos,
        &mut rng(),
      );
      pairs += negatives.len();
      for neg_cmd in negatives {
        if let Some(neg) = build_feature_vector(&neg_cmd, &context, &stats) {
          features.push(neg.values);
          labels.push(0.0);
        }
      }
    }

    update_training_stats(
      &mut stats,
      invocation,
      context.prev_command.as_ref(),
      context.prev_exit_status,
    );
    recent.pop_front();
    recent.push_back(invocation.clone());
  }

  if features.is_empty() {
    return Err(ZageError::ConfigError(
      "no training samples available".to_string(),
    ));
  }

  let feature_matrix = build_feature_matrix(&features)?;
  let dataset = Dataset::new(feature_matrix.clone(), labels)
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;

  let mut gbrt_config = GBRTConfig::for_binary_classification();
  gbrt_config.n_estimators = config.epochs.max(10);
  let mut model =
    GBRTModel::with_config(gbrt_config).map_err(|err| ZageError::GenericError(Box::new(err)))?;
  model.set_feature_names(feature_names());
  model
    .fit(&dataset)
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;

  let location = model_location()?;
  let model_io = ModelIO::new().map_err(|err| ZageError::GenericError(Box::new(err)))?;
  let booster = model.booster();
  model_io
    .save_model(booster, &location.dir, &location.name)
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;

  let calibration = calibrate_model(&model, val_set, &mut stats, phase_config.as_ref())?;

  let status = ModelStatus {
    version: "gbrt-rs-0.2.0".to_string(),
    n_trees: booster.n_trees(),
    objective: "binary".to_string(),
    loss: format!("{:?}", booster.config().loss),
    created_at: unix_now().to_string(),
    model_path: location.model_path.clone(),
    calibration,
  };
  let payload = serde_json::to_vec_pretty(&status).map_err(ZageError::SerializationError)?;
  fs::write(&location.metadata_path, payload)?;
  clear_model_cache();

  let accuracy = evaluate_model(&model, val_set, phase_config.as_ref())?;
  let model_path = location.model_path;

  Ok(TrainReport {
    samples: features.len(),
    pairs,
    validation_accuracy: accuracy,
    model_path,
  })
}

fn build_context(
  recent: &VecDeque<Invocation>,
  current: &Invocation,
  phase_config: Option<&crate::phase::PhaseConfig>,
) -> ContextWindow {
  let recent_commands: Vec<String> = recent
    .iter()
    .map(|inv| invocation_command(inv).to_string())
    .collect();
  let recent_heads: Vec<String> = recent
    .iter()
    .map(|inv| command_head(invocation_command(inv)))
    .collect();
  let session_tokens = recent_commands
    .iter()
    .flat_map(|cmd| normalized_tokens(cmd))
    .collect::<Vec<_>>();

  let session_phase = phase_config.and_then(|cfg| {
    for cmd in recent_commands.iter().rev() {
      if let Some(label) = cfg.match_label(cmd) {
        return cfg.labels().get(label).cloned();
      }
    }
    None
  });

  let repo_root = current
    .working_directory
    .as_deref()
    .and_then(find_repo_root)
    .unwrap_or_default();
  let branch = read_git_branch(&repo_root).ok().flatten();
  let time_bucket = timestamp_bucket(current.start_unix_timestamp.unwrap_or(0));

  ContextWindow {
    recent_commands,
    recent_heads,
    session_tokens,
    session_phase,
    repo_root,
    branch,
    time_bucket,
    shellname: current.shellname.clone(),
    working_directory: current.working_directory.clone(),
    hostname: current.hostname.clone(),
    username: current.username.clone(),
    session_id: current.session_id,
    prev_command: recent.back().map(|inv| invocation_command(inv).to_string()),
    prev_exit_status: recent.back().and_then(|inv| inv.exit_status),
    now: current.start_unix_timestamp.unwrap_or(0),
  }
}

fn update_training_stats(
  stats: &mut TrainingStats,
  invocation: &Invocation,
  prev_command: Option<&String>,
  prev_exit_status: Option<i64>,
) {
  let ts = invocation.start_unix_timestamp.unwrap_or(0);
  let command = invocation_command(invocation);
  update_stat(&mut stats.command_stats, command, ts);

  if let Some(prev) = prev_command {
    let key = (prev.clone(), prev_exit_status, command.to_string());
    *stats.transition_stats.entry(key).or_insert(0) += 1;
  }

  let context_key = (
    command.to_string(),
    invocation.working_directory.clone(),
    invocation.hostname.clone(),
    invocation.username.clone(),
  );
  *stats.context_stats.entry(context_key).or_insert(0) += 1;

  if let Some(ref cwd) = invocation.working_directory
    && let Some(repo_root) = find_repo_root(cwd)
  {
    update_repo_stat(&mut stats.repo_stats, &repo_root, command, ts);
  }

  let session_key = (invocation.session_id, command.to_string());
  update_stat_pair(&mut stats.session_stats, session_key, ts);

  let head = command_head(command);
  if !head.is_empty() {
    stats
      .head_to_commands
      .entry(head)
      .or_default()
      .push(command.to_string());
  }

  stats
    .session_commands
    .entry(invocation.session_id)
    .or_default()
    .push(command.to_string());
  stats.commands_seen.push(command.to_string());

  let sequence_command = normalize_sequence_command(command, &invocation.shellname);
  *stats
    .sequence_unigram
    .entry(sequence_command.clone())
    .or_insert(0) += 1;
  stats.sequence_total += 1;
  if let Some(prev1) = stats.sequence_prev1.as_ref() {
    *stats
      .sequence_bigram
      .entry((prev1.clone(), sequence_command.clone()))
      .or_insert(0) += 1;
  }
  if let (Some(prev2), Some(prev1)) = (stats.sequence_prev2.as_ref(), stats.sequence_prev1.as_ref())
  {
    *stats
      .sequence_trigram
      .entry((prev2.clone(), prev1.clone(), sequence_command.clone()))
      .or_insert(0) += 1;
  }
  stats.sequence_prev2 = stats.sequence_prev1.take();
  stats.sequence_prev1 = Some(sequence_command);
}

fn update_stat(map: &mut HashMap<String, super::features::Stat>, key: &str, ts: i64) {
  let entry = map.entry(key.to_string()).or_insert(super::features::Stat {
    freq: 0,
    last_seen: 0,
  });
  entry.freq += 1;
  if ts > entry.last_seen {
    entry.last_seen = ts;
  }
}

fn update_stat_pair(
  map: &mut HashMap<(i64, String), super::features::Stat>,
  key: (i64, String),
  ts: i64,
) {
  let entry = map.entry(key).or_insert(super::features::Stat {
    freq: 0,
    last_seen: 0,
  });
  entry.freq += 1;
  if ts > entry.last_seen {
    entry.last_seen = ts;
  }
}

fn update_repo_stat(
  map: &mut HashMap<(String, String), super::features::Stat>,
  repo: &str,
  cmd: &str,
  ts: i64,
) {
  let key = (repo.to_string(), cmd.to_string());
  let entry = map.entry(key).or_insert(super::features::Stat {
    freq: 0,
    last_seen: 0,
  });
  entry.freq += 1;
  if ts > entry.last_seen {
    entry.last_seen = ts;
  }
}

fn sample_negatives(
  stats: &TrainingStats,
  head: &str,
  session_id: i64,
  limit: usize,
  rng: &mut impl rand::Rng,
) -> Vec<String> {
  let mut candidates = Vec::new();
  if let Some(list) = stats.head_to_commands.get(head) {
    candidates.extend(list.iter().cloned());
  }
  if let Some(session_list) = stats.session_commands.get(&session_id) {
    candidates.extend(session_list.iter().cloned());
  }
  if candidates.len() < limit {
    let mut global = stats.commands_seen.clone();
    global.shuffle(rng);
    candidates.extend(global.into_iter().take(limit));
  }
  candidates.shuffle(rng);
  candidates.truncate(limit);
  candidates
}

fn evaluate_model(
  model: &GBRTModel,
  val_set: &[Invocation],
  phase_config: Option<&crate::phase::PhaseConfig>,
) -> Result<f64> {
  let mut stats = TrainingStats::default();
  let mut recent = VecDeque::new();
  let mut correct = 0usize;
  let mut total = 0usize;

  for invocation in val_set {
    if recent.len() < 6 {
      update_training_stats(&mut stats, invocation, None, None);
      recent.push_back(invocation.clone());
      continue;
    }

    let context = build_context(&recent, invocation, phase_config);
    let Some(pos) = build_feature_vector(invocation_command(invocation), &context, &stats) else {
      update_training_stats(
        &mut stats,
        invocation,
        context.prev_command.as_ref(),
        context.prev_exit_status,
      );
      recent.pop_front();
      recent.push_back(invocation.clone());
      continue;
    };

    let pos_score = predict_single(model, &pos.values)?;
    let negatives = sample_negatives(&stats, &pos.head, invocation.session_id, 4, &mut rng());
    let mut best = pos_score;
    let mut best_is_pos = true;

    for neg_cmd in negatives {
      if let Some(neg) = build_feature_vector(&neg_cmd, &context, &stats) {
        let score = predict_single(model, &neg.values)?;
        if score > best {
          best = score;
          best_is_pos = false;
        }
      }
    }

    if best_is_pos {
      correct += 1;
    }
    total += 1;

    update_training_stats(
      &mut stats,
      invocation,
      context.prev_command.as_ref(),
      context.prev_exit_status,
    );
    recent.pop_front();
    recent.push_back(invocation.clone());
  }

  Ok(if total == 0 {
    0.0
  } else {
    (correct as f64) / (total as f64)
  })
}

fn calibrate_model(
  model: &GBRTModel,
  val_set: &[Invocation],
  stats: &mut TrainingStats,
  phase_config: Option<&crate::phase::PhaseConfig>,
) -> Result<Option<CalibrationParams>> {
  let mut recent = VecDeque::new();
  let mut feature_vectors: Vec<Vec<f64>> = Vec::new();
  let mut tier1_scores: Vec<f64> = Vec::new();
  let mut labels: Vec<f64> = Vec::new();

  for invocation in val_set {
    if recent.len() < 6 {
      update_training_stats(stats, invocation, None, None);
      recent.push_back(invocation.clone());
      continue;
    }

    let context = build_context(&recent, invocation, phase_config);
    let mut samples: Vec<(String, f64)> = Vec::new();
    samples.push((invocation_command(invocation).to_string(), 1.0));
    let negatives = sample_negatives(
      stats,
      &command_head(invocation_command(invocation)),
      invocation.session_id,
      4,
      &mut rng(),
    );
    for neg in negatives {
      samples.push((neg, 0.0));
    }

    for (cmd, label) in samples {
      if let Some(vector) = build_feature_vector(&cmd, &context, stats) {
        feature_vectors.push(vector.values);
        tier1_scores.push(tier1_score_from_stats(&cmd, &context, stats));
        labels.push(label);
      }
    }

    update_training_stats(
      stats,
      invocation,
      context.prev_command.as_ref(),
      context.prev_exit_status,
    );
    recent.pop_front();
    recent.push_back(invocation.clone());
  }

  if feature_vectors.len() < 50 {
    return Ok(None);
  }

  let feature_matrix = build_feature_matrix(&feature_vectors)?;
  let predictions = model
    .predict(&feature_matrix)
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;

  let (tier1_a, tier1_b) = fit_platt(&tier1_scores, &labels);
  let (model_a, model_b) = fit_platt(&predictions, &labels);

  let p_tier1: Vec<f64> = tier1_scores
    .iter()
    .map(|score| sigmoid(tier1_a * score + tier1_b))
    .collect();
  let p_model: Vec<f64> = predictions
    .iter()
    .map(|score| sigmoid(model_a * score + model_b))
    .collect();

  let (stack_w0, stack_w1, stack_w2) = fit_stack(&p_tier1, &p_model, &labels);

  Ok(Some(CalibrationParams {
    tier1_a,
    tier1_b,
    model_a,
    model_b,
    stack_w0,
    stack_w1,
    stack_w2,
  }))
}

fn predict_single(model: &GBRTModel, features: &[f64]) -> Result<f64> {
  model
    .predict_single(features)
    .map_err(|err| ZageError::GenericError(Box::new(err)))
}

async fn load_invocations(conn: &Connection, limit: Option<usize>) -> Result<Vec<Invocation>> {
  let mut sql = String::from(
    "SELECT command, expanded_command, shellname, working_directory, hostname, username, exit_status, start_unix_timestamp, end_unix_timestamp, session_id
     FROM shell_history
     ORDER BY COALESCE(start_unix_timestamp, 0) ASC, id ASC",
  );
  if limit.is_some() {
    sql.push_str(" LIMIT ?");
  }

  let mut rows = if let Some(limit) = limit {
    conn.query(&sql, libsql::params![limit as i64]).await?
  } else {
    conn.query(&sql, ()).await?
  };

  let mut invocations = Vec::new();
  while let Some(row) = rows.next().await? {
    invocations.push(Invocation {
      command: row.get(0)?,
      expanded_command: row.get(1)?,
      shellname: row.get(2)?,
      working_directory: row.get(3)?,
      hostname: row.get(4)?,
      username: row.get(5)?,
      exit_status: row.get(6)?,
      start_unix_timestamp: row.get(7)?,
      end_unix_timestamp: row.get(8)?,
      session_id: row.get::<Option<i64>>(9)?.unwrap_or(0),
    });
  }

  Ok(invocations)
}
