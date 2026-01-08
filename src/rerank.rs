use std::collections::{HashMap, HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

use gbrt_rs::boosting::GBRTConfig;
use gbrt_rs::{Dataset, FeatureMatrix, GBRTModel, GradientBooster, ModelIO};
use libsql::Connection;
use ndarray::Array2;
use rand::rng;
use rand::seq::SliceRandom;
use serde::Deserialize;
use serde_json;
use tracing::warn;

use crate::hash_util::stable_hash;
use crate::predict::ranking::{recency_score, token_similarity};
use crate::predict::{Candidate, Suggestion};
use crate::repo::{find_repo_root, read_git_branch};
use crate::rerank_config::RerankConfig;
use crate::shell_history::Invocation;
use crate::tokenize::{extract_command_parts, normalized_tokens, tokenize};
use crate::{Result, ZageError};

const MODEL_NAME: &str = "rerank";
const HASH_FEATURES: usize = 64;
const BASE_FEATURES: usize = 9;
const FEATURE_COUNT: usize = BASE_FEATURES + HASH_FEATURES;

const DEFAULT_EPOCHS: usize = 150;
const DEFAULT_NEGATIVES: usize = 6;
const DEFAULT_MIN_HISTORY: usize = 1000;
const DEFAULT_MAX_SAMPLES: usize = 25_000;

#[derive(Debug, Clone)]
pub struct TrainConfig {
  pub epochs: usize,
  pub negatives_per_pos: usize,
  pub min_history: usize,
  pub max_samples: usize,
}

impl Default for TrainConfig {
  fn default() -> Self {
    Self {
      epochs: DEFAULT_EPOCHS,
      negatives_per_pos: DEFAULT_NEGATIVES,
      min_history: DEFAULT_MIN_HISTORY,
      max_samples: DEFAULT_MAX_SAMPLES,
    }
  }
}

#[derive(Debug, Clone)]
pub struct TrainReport {
  pub samples: usize,
  pub pairs: usize,
  pub validation_accuracy: f64,
  pub model_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct RerankContext {
  pub repo_root: String,
  pub recent_heads: Vec<String>,
  pub session_tokens: Vec<String>,
  pub session_phase: Option<String>,
  pub branch: Option<String>,
  pub time_bucket: u8,
}

#[derive(Debug, Clone, Default, Deserialize, serde::Serialize)]
pub struct ModelStatus {
  #[serde(default)]
  pub version: String,
  #[serde(default)]
  pub n_trees: usize,
  #[serde(default)]
  pub objective: String,
  #[serde(default)]
  pub loss: String,
  #[serde(default)]
  pub created_at: String,
  #[serde(default)]
  pub model_path: PathBuf,
}

struct ModelLocation {
  dir: PathBuf,
  name: String,
  model_path: PathBuf,
  metadata_path: PathBuf,
}

#[derive(Debug, Clone)]
struct FeatureVector {
  values: Vec<f64>,
  head: String,
}

#[derive(Debug, Clone)]
struct Stat {
  freq: i64,
  last_seen: i64,
}

type ContextKey = (String, Option<String>, Option<String>, Option<String>);

#[derive(Default)]
struct TrainingStats {
  command_stats: HashMap<String, Stat>,
  transition_stats: HashMap<(String, Option<i64>, String), i64>,
  context_stats: HashMap<ContextKey, i64>,
  repo_stats: HashMap<(String, String), Stat>,
  session_stats: HashMap<(i64, String), Stat>,
  session_commands: HashMap<i64, Vec<String>>,
  head_to_commands: HashMap<String, Vec<String>>,
  commands_seen: Vec<String>,
}

#[derive(Debug, Clone)]
struct ContextWindow {
  recent_heads: Vec<String>,
  session_tokens: Vec<String>,
  session_phase: Option<String>,
  repo_root: String,
  branch: Option<String>,
  time_bucket: u8,
  working_directory: Option<String>,
  hostname: Option<String>,
  username: Option<String>,
  session_id: i64,
  prev_command: Option<String>,
  prev_exit_status: Option<i64>,
  now: i64,
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

  for invocation in train_set {
    if recent.len() < 6 {
      update_training_stats(&mut stats, invocation, None, None);
      recent.push_back(invocation.clone());
      continue;
    }

    let context = build_context(&recent, invocation, phase_config.as_ref());
    if let Some(pos) = build_feature_vector(invocation.command.as_str(), &context, &stats) {
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

  let status = ModelStatus {
    version: "gbrt-rs-0.2.0".to_string(),
    n_trees: booster.n_trees(),
    objective: "binary".to_string(),
    loss: format!("{:?}", booster.config().loss),
    created_at: unix_now().to_string(),
    model_path: location.model_path.clone(),
  };
  let payload = serde_json::to_vec_pretty(&status).map_err(ZageError::SerializationError)?;
  fs::write(&location.metadata_path, payload)?;

  let accuracy = evaluate_model(&model, val_set, phase_config.as_ref())?;
  let model_path = location.model_path;

  Ok(TrainReport {
    samples: features.len(),
    pairs,
    validation_accuracy: accuracy,
    model_path,
  })
}

pub fn model_status() -> Result<Option<ModelStatus>> {
  let location = model_location()?;
  if !location.model_path.exists() {
    return Ok(None);
  }

  let mut status = if location.metadata_path.exists() {
    let data = fs::read(&location.metadata_path)?;
    match serde_json::from_slice::<ModelStatus>(&data) {
      Ok(status) => status,
      Err(err) => {
        warn!(
          "Failed to read reranker metadata at {}: {}",
          location.metadata_path.display(),
          err
        );
        ModelStatus::default()
      }
    }
  } else {
    ModelStatus::default()
  };
  status.model_path = location.model_path;
  Ok(Some(status))
}

pub fn reset_model() -> Result<()> {
  let location = model_location()?;
  if location.model_path.exists() {
    fs::remove_file(location.model_path)?;
  }
  if location.metadata_path.exists() {
    fs::remove_file(location.metadata_path)?;
  }
  Ok(())
}

pub(crate) fn rerank_suggestions(
  suggestions: &mut Vec<Suggestion>,
  candidates: &HashMap<String, Candidate>,
  context: &RerankContext,
  config: &RerankConfig,
) -> Result<bool> {
  let Some(model) = load_model()? else {
    return Ok(false);
  };
  if suggestions.is_empty() {
    return Ok(false);
  }

  let recent_heads: HashSet<String> = context.recent_heads.iter().cloned().collect();
  let mut vectors: Vec<Vec<f64>> = Vec::new();
  let mut indices: Vec<usize> = Vec::new();

  for (idx, suggestion) in suggestions.iter().enumerate() {
    if let Some(candidate) = candidates.get(&suggestion.command)
      && let Some(vector) = features_from_suggestion(suggestion, candidate, context, &recent_heads)
    {
      vectors.push(vector.values);
      indices.push(idx);
    }
  }

  if vectors.is_empty() {
    return Ok(false);
  }

  let feature_matrix = build_feature_matrix(&vectors)?;
  let predictions = model
    .predict(&feature_matrix)
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;

  let mut updated = suggestions.clone();
  for (offset, idx) in indices.iter().enumerate() {
    if let Some(score) = predictions.get(offset) {
      updated[*idx].score = *score;
    }
  }

  updated.sort_by(|a, b| {
    b.score
      .partial_cmp(&a.score)
      .unwrap_or(std::cmp::Ordering::Equal)
  });

  let top = updated.first().map(|s| s.score).unwrap_or(0.0);
  let second = updated.get(1).map(|s| s.score).unwrap_or(0.0);
  if top < config.rerank_min_prob || (top - second) < config.rerank_min_margin {
    return Ok(false);
  }

  *suggestions = updated;
  Ok(true)
}

fn features_from_suggestion(
  suggestion: &Suggestion,
  candidate: &Candidate,
  context: &RerankContext,
  recent_heads: &HashSet<String>,
) -> Option<FeatureVector> {
  let head = command_head(&suggestion.command);
  if head.is_empty() {
    return None;
  }

  let tier1_score = suggestion.score;
  let recency = suggestion.breakdown.recency;
  let frequency = suggestion.breakdown.frequency;
  let transition = suggestion.breakdown.transition;
  let context_score = suggestion.breakdown.context;

  let candidate_tokens = normalized_tokens(&suggestion.command);
  let similarity = token_similarity(&context.session_tokens, &candidate_tokens);

  let repo_match = if candidate.repo_freq > 0 { 1.0 } else { 0.0 };
  let session_match = if candidate.session_freq > 0 { 1.0 } else { 0.0 };
  let head_recent = if recent_heads.contains(&head) {
    1.0
  } else {
    0.0
  };

  let mut values = vec![0.0; FEATURE_COUNT];
  values[0] = tier1_score;
  values[1] = recency;
  values[2] = frequency;
  values[3] = transition;
  values[4] = context_score;
  values[5] = similarity;
  values[6] = repo_match;
  values[7] = session_match;
  values[8] = head_recent;

  add_hash(&mut values, format!("head:{head}").as_str());
  if let Some(branch) = context.branch.as_ref() {
    add_hash(&mut values, format!("branch:{branch}:{head}").as_str());
  }
  add_hash(
    &mut values,
    format!("time:{}:{head}", context.time_bucket).as_str(),
  );
  if let Some(phase) = context.session_phase.as_ref() {
    add_hash(&mut values, format!("phase:{phase}:{head}").as_str());
  }

  Some(FeatureVector { values, head })
}

fn build_feature_vector(
  command: &str,
  context: &ContextWindow,
  stats: &TrainingStats,
) -> Option<FeatureVector> {
  let head = command_head(command);
  if head.is_empty() {
    return None;
  }

  let stat = stats.command_stats.get(command);
  let freq = stat.map(|s| s.freq).unwrap_or(0);
  let last_seen = stat.map(|s| s.last_seen).unwrap_or(0);

  let transition_freq = context
    .prev_command
    .as_ref()
    .and_then(|prev| {
      stats
        .transition_stats
        .get(&(prev.clone(), context.prev_exit_status, command.to_string()))
    })
    .copied()
    .unwrap_or(0);

  let context_freq = stats
    .context_stats
    .get(&(
      command.to_string(),
      context.working_directory.clone(),
      context.hostname.clone(),
      context.username.clone(),
    ))
    .copied()
    .unwrap_or(0);

  let repo_freq = if !context.repo_root.is_empty() {
    stats
      .repo_stats
      .get(&(context.repo_root.clone(), command.to_string()))
      .map(|s| s.freq)
      .unwrap_or(0)
  } else {
    0
  };

  let session_freq = stats
    .session_stats
    .get(&(context.session_id, command.to_string()))
    .map(|s| s.freq)
    .unwrap_or(0);

  let recency = recency_score(context.now, last_seen);
  let frequency = (freq as f64).ln_1p() + 0.5 * (repo_freq as f64).ln_1p();
  let transition = (transition_freq as f64).ln_1p();
  let context_score = (context_freq as f64).ln_1p() + 0.8 * (session_freq as f64).ln_1p();

  let candidate_tokens = normalized_tokens(command);
  let similarity = token_similarity(&context.session_tokens, &candidate_tokens);

  let tier1_score = recency + frequency + transition + context_score + similarity;
  let repo_match = if repo_freq > 0 { 1.0 } else { 0.0 };
  let session_match = if session_freq > 0 { 1.0 } else { 0.0 };
  let head_recent = if context.recent_heads.iter().any(|h| h == &head) {
    1.0
  } else {
    0.0
  };

  let mut values = vec![0.0; FEATURE_COUNT];
  values[0] = tier1_score;
  values[1] = recency;
  values[2] = frequency;
  values[3] = transition;
  values[4] = context_score;
  values[5] = similarity;
  values[6] = repo_match;
  values[7] = session_match;
  values[8] = head_recent;

  add_hash(&mut values, format!("head:{head}").as_str());
  if let Some(branch) = context.branch.as_ref() {
    add_hash(&mut values, format!("branch:{branch}:{head}").as_str());
  }
  add_hash(
    &mut values,
    format!("time:{}:{head}", context.time_bucket).as_str(),
  );
  if let Some(phase) = context.session_phase.as_ref() {
    add_hash(&mut values, format!("phase:{phase}:{head}").as_str());
  }

  Some(FeatureVector { values, head })
}

fn build_context(
  recent: &VecDeque<Invocation>,
  current: &Invocation,
  phase_config: Option<&crate::phase::PhaseConfig>,
) -> ContextWindow {
  let recent_commands: Vec<String> = recent.iter().map(|inv| inv.command.clone()).collect();
  let recent_heads: Vec<String> = recent
    .iter()
    .map(|inv| command_head(&inv.command))
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
    recent_heads,
    session_tokens,
    session_phase,
    repo_root,
    branch,
    time_bucket,
    working_directory: current.working_directory.clone(),
    hostname: current.hostname.clone(),
    username: current.username.clone(),
    session_id: current.session_id,
    prev_command: recent.back().map(|inv| inv.command.clone()),
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
  update_stat(&mut stats.command_stats, &invocation.command, ts);

  if let Some(prev) = prev_command {
    let key = (prev.clone(), prev_exit_status, invocation.command.clone());
    *stats.transition_stats.entry(key).or_insert(0) += 1;
  }

  let context_key = (
    invocation.command.clone(),
    invocation.working_directory.clone(),
    invocation.hostname.clone(),
    invocation.username.clone(),
  );
  *stats.context_stats.entry(context_key).or_insert(0) += 1;

  if let Some(ref cwd) = invocation.working_directory
    && let Some(repo_root) = find_repo_root(cwd)
  {
    update_repo_stat(&mut stats.repo_stats, &repo_root, &invocation.command, ts);
  }

  let session_key = (invocation.session_id, invocation.command.clone());
  update_stat_pair(&mut stats.session_stats, session_key, ts);

  let head = command_head(&invocation.command);
  if !head.is_empty() {
    stats
      .head_to_commands
      .entry(head)
      .or_default()
      .push(invocation.command.clone());
  }

  stats
    .session_commands
    .entry(invocation.session_id)
    .or_default()
    .push(invocation.command.clone());
  stats.commands_seen.push(invocation.command.clone());
}

fn update_stat(map: &mut HashMap<String, Stat>, key: &str, ts: i64) {
  let entry = map.entry(key.to_string()).or_insert(Stat {
    freq: 0,
    last_seen: 0,
  });
  entry.freq += 1;
  if ts > entry.last_seen {
    entry.last_seen = ts;
  }
}

fn update_stat_pair(map: &mut HashMap<(i64, String), Stat>, key: (i64, String), ts: i64) {
  let entry = map.entry(key).or_insert(Stat {
    freq: 0,
    last_seen: 0,
  });
  entry.freq += 1;
  if ts > entry.last_seen {
    entry.last_seen = ts;
  }
}

fn update_repo_stat(map: &mut HashMap<(String, String), Stat>, repo: &str, cmd: &str, ts: i64) {
  let key = (repo.to_string(), cmd.to_string());
  let entry = map.entry(key).or_insert(Stat {
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
    let Some(pos) = build_feature_vector(invocation.command.as_str(), &context, &stats) else {
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

fn predict_single(model: &GBRTModel, features: &[f64]) -> Result<f64> {
  model
    .predict_single(features)
    .map_err(|err| ZageError::GenericError(Box::new(err)))
}

fn build_feature_matrix(vectors: &[Vec<f64>]) -> Result<FeatureMatrix> {
  let rows = vectors.len();
  let cols = FEATURE_COUNT;
  let mut data = Array2::<f64>::zeros((rows, cols));
  for (row, vector) in vectors.iter().enumerate() {
    if vector.len() != cols {
      return Err(ZageError::ConfigError(format!(
        "feature length mismatch: expected {cols}, got {}",
        vector.len()
      )));
    }
    for (col, value) in vector.iter().enumerate() {
      data[(row, col)] = *value;
    }
  }
  FeatureMatrix::new(data).map_err(|err| ZageError::GenericError(Box::new(err)))
}

fn feature_names() -> Vec<String> {
  (0..FEATURE_COUNT).map(|i| format!("f{i}")).collect()
}

fn command_head(command: &str) -> String {
  let tokens = tokenize(command);
  if let Some(parts) = extract_command_parts(command, &tokens) {
    return parts.head;
  }
  tokens
    .first()
    .map(|token| token.raw.clone())
    .unwrap_or_default()
}

fn timestamp_bucket(ts: i64) -> u8 {
  if ts <= 0 {
    return 0;
  }
  let tz = jiff::tz::TimeZone::system();
  let hour = jiff::Timestamp::from_second(ts)
    .map(|t| t.to_zoned(tz))
    .map(|z| z.hour())
    .unwrap_or(0);
  (hour.clamp(0, 23) / 4) as u8
}

fn add_hash(values: &mut [f64], label: &str) {
  let hash = stable_hash(label);
  let idx = BASE_FEATURES + (hash as usize % HASH_FEATURES);
  values[idx] += 1.0;
}

fn default_model_dir() -> Result<PathBuf> {
  if let Ok(path) = std::env::var("ZAGE_MODEL_PATH") {
    return Ok(PathBuf::from(path));
  }
  let base =
    dirs::data_dir().ok_or_else(|| ZageError::ConfigError("missing data dir".to_string()))?;
  Ok(base.join("zage/model"))
}

fn model_location() -> Result<ModelLocation> {
  let path = default_model_dir()?;
  if path.extension().is_some() {
    let dir = path.parent().unwrap_or(Path::new(".")).to_path_buf();
    let stem = path
      .file_stem()
      .map(|s| s.to_string_lossy().into_owned())
      .unwrap_or_else(|| MODEL_NAME.to_string());
    let name = stem.to_string();
    let model_path = path.clone();
    let metadata_path = dir.join(format!("{name}_metadata.json"));
    return Ok(ModelLocation {
      dir,
      name,
      model_path,
      metadata_path,
    });
  }

  let model_path = path.join(format!("{MODEL_NAME}.json"));
  let metadata_path = path.join(format!("{MODEL_NAME}_metadata.json"));
  Ok(ModelLocation {
    dir: path,
    name: MODEL_NAME.to_string(),
    model_path,
    metadata_path,
  })
}

fn load_model() -> Result<Option<GradientBooster>> {
  let location = model_location()?;
  if !location.model_path.exists() {
    return Ok(None);
  }
  let model_io = ModelIO::new().map_err(|err| ZageError::GenericError(Box::new(err)))?;
  let booster = model_io
    .load_model(&location.model_path)
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;
  Ok(Some(booster))
}

pub fn runtime_context(
  repo_root: &str,
  recent_heads: &[String],
  session_tokens: Vec<String>,
  session_phase: Option<&str>,
) -> RerankContext {
  let branch = if repo_root.is_empty() {
    None
  } else {
    read_git_branch(repo_root).ok().flatten()
  };
  let time_bucket = timestamp_bucket(unix_now());
  RerankContext {
    repo_root: repo_root.to_string(),
    recent_heads: recent_heads.to_vec(),
    session_tokens,
    session_phase: session_phase.map(|s| s.to_string()),
    branch,
    time_bucket,
  }
}

async fn load_invocations(conn: &Connection, limit: Option<usize>) -> Result<Vec<Invocation>> {
  let mut sql = String::from(
    "SELECT command, shellname, working_directory, hostname, username, exit_status, start_unix_timestamp, end_unix_timestamp, session_id
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
      shellname: row.get(1)?,
      working_directory: row.get(2)?,
      hostname: row.get(3)?,
      username: row.get(4)?,
      exit_status: row.get(5)?,
      start_unix_timestamp: row.get(6)?,
      end_unix_timestamp: row.get(7)?,
      session_id: row.get::<Option<i64>>(8)?.unwrap_or(0),
    });
  }
  Ok(invocations)
}

fn unix_now() -> i64 {
  std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::predict::{ScoreBreakdown, candidate_for_test};
  use std::sync::Mutex;
  use tempfile::tempdir;

  static ENV_LOCK: Mutex<()> = Mutex::new(());

  #[test]
  fn feature_matrix_is_deterministic() {
    let a = vec![vec![1.0; FEATURE_COUNT], vec![0.5; FEATURE_COUNT]];
    let b = vec![vec![1.0; FEATURE_COUNT], vec![0.5; FEATURE_COUNT]];
    let left = build_feature_matrix(&a).unwrap();
    let right = build_feature_matrix(&b).unwrap();
    assert_eq!(left.data(), right.data());
  }

  #[test]
  fn hash_features_stable() {
    let mut values = vec![0.0; FEATURE_COUNT];
    add_hash(&mut values, "head:git");
    let mut values2 = vec![0.0; FEATURE_COUNT];
    add_hash(&mut values2, "head:git");
    assert_eq!(values, values2);
  }

  #[test]
  fn rerank_prefers_trained_positive() {
    let _guard = ENV_LOCK.lock().unwrap();
    let temp = tempdir().unwrap();
    let prev_env = std::env::var("ZAGE_MODEL_PATH").ok();
    unsafe {
      std::env::set_var("ZAGE_MODEL_PATH", temp.path());
    }

    let good_cmd = "git status";
    let bad_cmd = "cargo build";
    let suggestions = [
      Suggestion {
        command: bad_cmd.to_string(),
        score: 0.1,
        breakdown: ScoreBreakdown::default(),
      },
      Suggestion {
        command: good_cmd.to_string(),
        score: 2.0,
        breakdown: ScoreBreakdown::default(),
      },
    ];

    let mut candidates: HashMap<String, Candidate> = HashMap::new();
    candidates.insert(good_cmd.to_string(), candidate_for_test(good_cmd));
    candidates.insert(bad_cmd.to_string(), candidate_for_test(bad_cmd));

    let context = RerankContext {
      repo_root: String::new(),
      recent_heads: vec!["git".to_string()],
      session_tokens: Vec::new(),
      session_phase: None,
      branch: None,
      time_bucket: 0,
    };

    let recent_heads: HashSet<String> = context.recent_heads.iter().cloned().collect();
    let good_features = features_from_suggestion(
      &suggestions[1],
      candidates.get(good_cmd).unwrap(),
      &context,
      &recent_heads,
    )
    .unwrap();
    let bad_features = features_from_suggestion(
      &suggestions[0],
      candidates.get(bad_cmd).unwrap(),
      &context,
      &recent_heads,
    )
    .unwrap();

    let feature_matrix =
      build_feature_matrix(&[good_features.values.clone(), bad_features.values.clone()]).unwrap();
    let dataset = Dataset::new(feature_matrix, vec![1.0, 0.0]).unwrap();
    let mut gbrt_config = GBRTConfig::for_binary_classification();
    gbrt_config.n_estimators = 32;
    let mut model = GBRTModel::with_config(gbrt_config).unwrap();
    model.set_feature_names(feature_names());
    model.fit(&dataset).unwrap();

    let model_io = ModelIO::new().unwrap();
    model_io
      .save_model(model.booster(), temp.path(), MODEL_NAME)
      .unwrap();

    let loaded = load_model().unwrap().expect("model should load");
    let matrix =
      build_feature_matrix(&[good_features.values.clone(), bad_features.values.clone()]).unwrap();
    let scores = loaded.predict(&matrix).unwrap();
    assert!(scores.len() >= 2);
    assert!(scores[0] > scores[1]);

    unsafe {
      if let Some(prev) = prev_env {
        std::env::set_var("ZAGE_MODEL_PATH", prev);
      } else {
        std::env::remove_var("ZAGE_MODEL_PATH");
      }
    }
  }
}
