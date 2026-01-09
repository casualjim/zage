use std::collections::{HashMap, HashSet};

use crate::core::{Candidate, Suggestion};
use crate::hash_util::stable_hash;
use crate::predict::ranking::{DEFAULT_RECENCY_HALF_LIFE_SECONDS, recency_score, token_similarity};
use crate::tokenize::{extract_command_parts, normalized_tokens, tokenize, tokenize_index};

use super::config::RerankContext;
use super::{BASE_FEATURES, FEATURE_COUNT, HASH_FEATURES};

#[derive(Debug, Clone)]
pub(crate) struct FeatureVector {
  pub(crate) values: Vec<f64>,
  pub(crate) head: String,
}

#[derive(Debug, Clone)]
pub(crate) struct Stat {
  pub(crate) freq: i64,
  pub(crate) last_seen: i64,
}

type ContextKey = (String, Option<String>, Option<String>, Option<String>);

#[derive(Default)]
pub(crate) struct TrainingStats {
  pub(crate) command_stats: HashMap<String, Stat>,
  pub(crate) transition_stats: HashMap<(String, Option<i64>, String), i64>,
  pub(crate) context_stats: HashMap<ContextKey, i64>,
  pub(crate) repo_stats: HashMap<(String, String), Stat>,
  pub(crate) session_stats: HashMap<(i64, String), Stat>,
  pub(crate) session_commands: HashMap<i64, Vec<String>>,
  pub(crate) head_to_commands: HashMap<String, Vec<String>>,
  pub(crate) commands_seen: Vec<String>,
  pub(crate) sequence_unigram: HashMap<String, i64>,
  pub(crate) sequence_bigram: HashMap<(String, String), i64>,
  pub(crate) sequence_trigram: HashMap<(String, String, String), i64>,
  pub(crate) sequence_total: i64,
  pub(crate) sequence_prev1: Option<String>,
  pub(crate) sequence_prev2: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ContextWindow {
  pub(crate) recent_commands: Vec<String>,
  pub(crate) recent_heads: Vec<String>,
  pub(crate) session_tokens: Vec<String>,
  pub(crate) session_phase: Option<String>,
  pub(crate) repo_root: String,
  pub(crate) branch: Option<String>,
  pub(crate) time_bucket: u8,
  pub(crate) shellname: String,
  pub(crate) working_directory: Option<String>,
  pub(crate) hostname: Option<String>,
  pub(crate) username: Option<String>,
  pub(crate) session_id: i64,
  pub(crate) prev_command: Option<String>,
  pub(crate) prev_exit_status: Option<i64>,
  pub(crate) now: i64,
}

pub(crate) fn features_from_suggestion(
  suggestion: &Suggestion,
  candidate: &Candidate,
  context: &RerankContext,
  recent_heads: &HashSet<String>,
) -> Option<FeatureVector> {
  let head = command_head(&suggestion.command);
  if head.is_empty() {
    return None;
  }

  let time_bucket = context.time_bucket;
  let recency = suggestion.breakdown.recency;
  let frequency = suggestion.breakdown.frequency;
  let transition = suggestion.breakdown.transition;
  let context_score = suggestion.breakdown.context;
  let sequence_score = suggestion.breakdown.sequence;

  let candidate_tokens = normalized_tokens(&suggestion.command);
  let similarity = token_similarity(&context.session_tokens, &candidate_tokens);
  let tier1_score = recency + frequency + transition + context_score + sequence_score + similarity;

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
  values[9] = sequence_score;
  values[10] = candidate.sequence_confidence;
  values[11] = candidate.sequence_lift;
  values[12] = candidate.sequence_prefix_len as f64;

  add_hash(&mut values, format!("head:{head}").as_str());
  if let Some(branch) = context.branch.as_ref() {
    add_hash(&mut values, format!("branch:{branch}:{head}").as_str());
  }
  add_hash(&mut values, format!("time:{time_bucket}:{head}").as_str());
  if let Some(phase) = context.session_phase.as_ref() {
    add_hash(&mut values, format!("phase:{phase}:{head}").as_str());
  }

  Some(FeatureVector { values, head })
}

pub(crate) fn build_feature_vector(
  command: &str,
  context: &ContextWindow,
  stats: &TrainingStats,
) -> Option<FeatureVector> {
  let head = command_head(command);
  if head.is_empty() {
    return None;
  }

  let time_bucket = context.time_bucket;
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

  let recency = recency_score(context.now, last_seen, DEFAULT_RECENCY_HALF_LIFE_SECONDS);
  let frequency = (freq as f64).ln_1p() + 0.5 * (repo_freq as f64).ln_1p();
  let transition = (transition_freq as f64).ln_1p();
  let context_score = (context_freq as f64).ln_1p() + 0.8 * (session_freq as f64).ln_1p();

  let candidate_tokens = normalized_tokens(command);
  let similarity = token_similarity(&context.session_tokens, &candidate_tokens);

  let (sequence_score, sequence_confidence, sequence_lift, sequence_prefix_len) =
    sequence_features(command, context, stats);
  let tier1_score = recency + frequency + transition + context_score + sequence_score + similarity;
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
  values[9] = sequence_score;
  values[10] = sequence_confidence;
  values[11] = sequence_lift;
  values[12] = sequence_prefix_len;

  add_hash(&mut values, format!("head:{head}").as_str());
  if let Some(branch) = context.branch.as_ref() {
    add_hash(&mut values, format!("branch:{branch}:{head}").as_str());
  }
  add_hash(&mut values, format!("time:{time_bucket}:{head}").as_str());
  if let Some(phase) = context.session_phase.as_ref() {
    add_hash(&mut values, format!("phase:{phase}:{head}").as_str());
  }

  Some(FeatureVector { values, head })
}

pub(crate) fn sequence_features(
  command: &str,
  context: &ContextWindow,
  stats: &TrainingStats,
) -> (f64, f64, f64, f64) {
  if stats.sequence_total <= 0 || context.recent_commands.is_empty() {
    return (0.0, 0.0, 0.0, 0.0);
  }

  let candidate = normalize_sequence_command(command, &context.shellname);
  let total = stats.sequence_total as f64;
  let base = stats
    .sequence_unigram
    .get(&candidate)
    .map(|v| *v as f64 / total)
    .unwrap_or(0.0);
  if base <= 0.0 {
    return (0.0, 0.0, 0.0, 0.0);
  }

  let mut best_conf = 0.0;
  let mut best_lift = 0.0;
  let mut best_prefix = 0.0;

  if context.recent_commands.len() >= 2 {
    let prev2 = normalize_sequence_command(
      context
        .recent_commands
        .get(context.recent_commands.len() - 2)
        .map(|s| s.as_str())
        .unwrap_or_default(),
      &context.shellname,
    );
    let prev1 = normalize_sequence_command(
      context
        .recent_commands
        .last()
        .map(|s| s.as_str())
        .unwrap_or_default(),
      &context.shellname,
    );
    let trigram = stats
      .sequence_trigram
      .get(&(prev2.clone(), prev1.clone(), candidate.clone()))
      .copied()
      .unwrap_or(0);
    let prefix = stats
      .sequence_bigram
      .get(&(prev2, prev1))
      .copied()
      .unwrap_or(0);
    if trigram > 0 && prefix > 0 {
      let conf = (trigram as f64) / (prefix as f64);
      let lift = conf / base;
      best_conf = conf;
      best_lift = lift;
      best_prefix = 2.0;
    }
  }

  if best_prefix == 0.0 && !context.recent_commands.is_empty() {
    let prev1 = normalize_sequence_command(
      context
        .recent_commands
        .last()
        .map(|s| s.as_str())
        .unwrap_or_default(),
      &context.shellname,
    );
    let bigram = stats
      .sequence_bigram
      .get(&(prev1.clone(), candidate.clone()))
      .copied()
      .unwrap_or(0);
    let prefix = stats.sequence_unigram.get(&prev1).copied().unwrap_or(0);
    if bigram > 0 && prefix > 0 {
      let conf = (bigram as f64) / (prefix as f64);
      let lift = conf / base;
      best_conf = conf;
      best_lift = lift;
      best_prefix = 1.0;
    }
  }

  if best_prefix == 0.0 || best_conf <= 0.0 {
    return (0.0, 0.0, 0.0, 0.0);
  }

  let order_weight = if best_prefix >= 2.0 { 1.0 } else { 0.7 };
  let score = best_conf * best_lift.max(1.0) * order_weight;
  (score, best_conf, best_lift, best_prefix)
}

pub(crate) fn tier1_score_from_stats(
  command: &str,
  context: &ContextWindow,
  stats: &TrainingStats,
) -> f64 {
  let recency = stats
    .command_stats
    .get(command)
    .map(|stat| {
      recency_score(
        context.now,
        stat.last_seen,
        DEFAULT_RECENCY_HALF_LIFE_SECONDS,
      )
    })
    .unwrap_or(0.0);
  let frequency = stats
    .command_stats
    .get(command)
    .map(|stat| (stat.freq as f64).ln_1p())
    .unwrap_or(0.0);

  let transition = if let Some(prev) = context.prev_command.as_ref() {
    let key = (prev.clone(), context.prev_exit_status, command.to_string());
    stats
      .transition_stats
      .get(&key)
      .map(|val| (*val as f64).ln_1p())
      .unwrap_or(0.0)
  } else {
    0.0
  };

  let mut context_hits = 0.0;
  let context_key = (
    command.to_string(),
    context.working_directory.clone(),
    context.hostname.clone(),
    context.username.clone(),
  );
  if let Some(freq) = stats.context_stats.get(&context_key) {
    context_hits += (*freq as f64).ln_1p();
  }
  let session_key = (context.session_id, command.to_string());
  if let Some(stat) = stats.session_stats.get(&session_key) {
    context_hits += 0.8 * (stat.freq as f64).ln_1p();
  }

  0.25 * recency + 0.25 * frequency + 0.2 * transition + 0.15 * context_hits
}

pub(crate) fn build_feature_matrix(vectors: &[Vec<f64>]) -> crate::Result<gbrt_rs::FeatureMatrix> {
  use ndarray::Array2;

  let rows = vectors.len();
  let cols = FEATURE_COUNT;
  let mut data = Array2::<f64>::zeros((rows, cols));
  for (row, vector) in vectors.iter().enumerate() {
    if vector.len() != cols {
      return Err(crate::ZageError::ConfigError(format!(
        "feature length mismatch: expected {cols}, got {}",
        vector.len()
      )));
    }
    for (col, value) in vector.iter().enumerate() {
      data[(row, col)] = *value;
    }
  }
  gbrt_rs::FeatureMatrix::new(data).map_err(|err| crate::ZageError::GenericError(Box::new(err)))
}

pub(crate) fn feature_names() -> Vec<String> {
  (0..FEATURE_COUNT).map(|i| format!("f{i}")).collect()
}

pub(crate) fn command_head(command: &str) -> String {
  let tokens = tokenize(command);
  if let Some(parts) = extract_command_parts(command, &tokens) {
    return parts.head;
  }
  tokens
    .first()
    .map(|tok| tok.raw.clone())
    .unwrap_or_default()
}

pub(crate) fn command_signature(command: &str, shellname: &str) -> Option<String> {
  let tokens = tokenize_index(shellname, command);
  let parts = extract_command_parts(command, &tokens)?;
  let mut signature = parts.head;
  for arg in parts.args.iter() {
    if let Some(spec) = arg.normalized.strip_prefix("ARG=") {
      signature.push(':');
      signature.push_str(spec);
    } else if arg.normalized == "PATH" {
      signature.push_str(":PATH");
    } else if arg.normalized == "IP" {
      signature.push_str(":IP");
    } else if arg.normalized == "NUM" {
      signature.push_str(":NUM");
    }
  }
  Some(signature)
}

pub(crate) fn normalize_sequence_command(command: &str, shellname: &str) -> String {
  command_signature(command, shellname).unwrap_or_else(|| command.to_string())
}

pub(crate) fn add_hash(values: &mut [f64], label: &str) {
  let hash = stable_hash(label);
  let idx = BASE_FEATURES + (hash as usize % HASH_FEATURES);
  values[idx] += 1.0;
}
