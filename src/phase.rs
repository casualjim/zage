use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use gbrt_rs::boosting::GBRTConfig;
use gbrt_rs::{Dataset, FeatureMatrix, GBRTModel};
use ndarray::Array2;
use serde::Deserialize;

use crate::hash_util::stable_hash;
use crate::tokenize::Token;
use crate::{Result, ZageError};

const DEFAULT_PHASE: &str = "default";
const PHASE_FEATURES: usize = 512;
const PHASE_MIN_SAMPLES: usize = 100;
const PHASE_MIN_PER_CLASS: usize = 8;
const PHASE_DEFAULT_LIMIT: usize = 5_000;
const PHASE_ESTIMATORS: usize = 120;

#[derive(Debug, Clone, Deserialize)]
struct PhaseConfigFile {
  phases: HashMap<String, PhaseRule>,
}

#[derive(Debug, Clone, Deserialize)]
struct PhaseRule {
  patterns: Vec<String>,
}

#[derive(Debug, Clone)]
struct PhasePattern {
  head_glob: String,
  flag_globs: Vec<String>,
  arg_globs: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PhaseConfig {
  labels: Vec<String>,
  patterns: Vec<(usize, Vec<PhasePattern>)>,
  default_idx: usize,
}

#[derive(Debug, Clone)]
pub struct PhaseSample {
  pub features: Vec<f64>,
  pub label: usize,
}

pub struct PhasePredictor {
  labels: Vec<String>,
  hash_size: usize,
  default_idx: usize,
  models: Vec<Option<GBRTModel>>,
}

impl PhaseConfig {
  pub fn load() -> Result<Self> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = env::var("ZAGE_PHASES_CONFIG") {
      candidates.push(PathBuf::from(path));
    }
    candidates.push(PathBuf::from("config/phases.toml"));
    if let Some(config_dir) = dirs::config_dir() {
      candidates.push(config_dir.join("zage").join("phases.toml"));
    }

    for path in candidates {
      if path.exists() {
        let contents = fs::read_to_string(&path)?;
        return Self::from_str(&contents);
      }
    }

    Ok(Self::empty())
  }

  #[cfg(any(test, feature = "tier1-tests"))]
  pub(crate) fn load_from_path(path: &std::path::Path) -> Result<Self> {
    let contents = fs::read_to_string(path)?;
    Self::from_str(&contents)
  }

  fn from_str(contents: &str) -> Result<Self> {
    let parsed: PhaseConfigFile =
      toml::from_str(contents).map_err(|err| ZageError::ConfigError(err.to_string()))?;
    Ok(Self::from_map(parsed.phases))
  }

  fn from_map(map: HashMap<String, PhaseRule>) -> Self {
    let mut labels: Vec<String> = map.keys().cloned().collect();
    labels.sort();
    if !labels.iter().any(|label| label == DEFAULT_PHASE) {
      labels.push(DEFAULT_PHASE.to_string());
    }
    let default_idx = labels
      .iter()
      .position(|label| label == DEFAULT_PHASE)
      .unwrap_or(labels.len().saturating_sub(1));
    let mut patterns: Vec<(usize, Vec<PhasePattern>)> = Vec::new();
    for (label, rule) in map {
      let idx = labels.iter().position(|name| name == &label);
      if let Some(idx) = idx {
        let parsed = rule
          .patterns
          .into_iter()
          .filter_map(|pattern| parse_phase_pattern(&pattern))
          .collect::<Vec<_>>();
        patterns.push((idx, parsed));
      }
    }
    Self {
      labels,
      patterns,
      default_idx,
    }
  }

  fn empty() -> Self {
    Self {
      labels: vec![DEFAULT_PHASE.to_string()],
      patterns: Vec::new(),
      default_idx: 0,
    }
  }

  pub fn labels(&self) -> &[String] {
    &self.labels
  }

  pub fn default_idx(&self) -> usize {
    self.default_idx
  }

  pub fn hash_size(&self) -> usize {
    PHASE_FEATURES
  }

  pub fn match_label(&self, command: &str) -> Option<usize> {
    if self.patterns.is_empty() {
      return None;
    }
    let tokens = crate::tokenize::tokenize(command);
    let parts = crate::tokenize::extract_command_parts(command, &tokens)?;
    for (idx, patterns) in &self.patterns {
      if patterns.iter().any(|pattern| pattern.matches(&parts)) {
        return Some(*idx);
      }
    }
    None
  }

  pub fn pattern_distribution(&self, command: &str) -> Vec<f32> {
    let mut scores = vec![0.0; self.labels.len()];
    if let Some(label) = self.match_label(command) {
      scores[label] = 1.0;
      return scores;
    }
    if self.default_idx < scores.len() {
      scores[self.default_idx] = 1.0;
    }
    scores
  }
}

pub fn detect_phase_from_commands(
  recent_commands: &[String],
  phase_config: &PhaseConfig,
) -> Option<(String, f64)> {
  if phase_config.labels().len() <= 1 {
    return None;
  }
  let mut scores: HashMap<String, f64> = HashMap::new();
  let mut total = 0.0f64;
  for (idx, command) in recent_commands.iter().rev().take(6).enumerate() {
    let weight = 0.5_f64.powi(idx as i32);
    let label_idx = phase_config
      .match_label(command)
      .unwrap_or_else(|| phase_config.default_idx());
    let Some(phase) = phase_config.labels().get(label_idx).cloned() else {
      continue;
    };
    *scores.entry(phase).or_insert(0.0) += weight;
    total += weight;
  }
  let (phase, score) = scores
    .into_iter()
    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;
  let confidence = if total > 0.0 { score / total } else { 0.0 };
  Some((phase, confidence))
}

pub fn features_from_tokens(tokens: &[Token], hash_size: usize) -> Vec<f64> {
  let mut features = vec![0.0f64; hash_size];
  for token in tokens {
    let idx = (stable_hash(&token.normalized) as usize) % hash_size;
    features[idx] += 1.0;
  }
  features
}

pub fn train_phase_predictor(
  config: &PhaseConfig,
  mut samples: Vec<PhaseSample>,
  unlabeled: Vec<Vec<f64>>,
) -> Option<PhasePredictor> {
  if config.labels.len() <= 1 {
    return None;
  }

  let mut counts = vec![0usize; config.labels.len()];
  for sample in &samples {
    if let Some(count) = counts.get_mut(sample.label) {
      *count += 1;
    }
  }

  let total_labeled = samples.len();
  if total_labeled < PHASE_MIN_SAMPLES {
    return None;
  }
  for (idx, count) in counts.iter().enumerate() {
    if idx == config.default_idx {
      continue;
    }
    if *count < PHASE_MIN_PER_CLASS {
      return None;
    }
  }

  let default_take = total_labeled.min(unlabeled.len()).min(PHASE_DEFAULT_LIMIT);
  for features in unlabeled.into_iter().take(default_take) {
    samples.push(PhaseSample {
      features,
      label: config.default_idx,
    });
  }

  let feature_names = phase_feature_names();
  let feature_matrix = match build_feature_matrix(&samples, config.hash_size()) {
    Ok(matrix) => matrix,
    Err(_) => return None,
  };

  let mut models: Vec<Option<GBRTModel>> = Vec::with_capacity(config.labels.len());
  for label_idx in 0..config.labels.len() {
    let targets: Vec<f64> = samples
      .iter()
      .map(|sample| if sample.label == label_idx { 1.0 } else { 0.0 })
      .collect();
    let dataset = match Dataset::new(feature_matrix.clone(), targets) {
      Ok(dataset) => dataset,
      Err(_) => {
        models.push(None);
        continue;
      }
    };
    let mut gbrt_config = GBRTConfig::for_binary_classification();
    gbrt_config.n_estimators = PHASE_ESTIMATORS;
    let mut model = match GBRTModel::with_config(gbrt_config) {
      Ok(model) => model,
      Err(_) => {
        models.push(None);
        continue;
      }
    };
    model.set_feature_names(feature_names.clone());
    if model.fit(&dataset).is_err() {
      models.push(None);
      continue;
    }
    models.push(Some(model));
  }

  Some(PhasePredictor {
    labels: config.labels.clone(),
    hash_size: PHASE_FEATURES,
    default_idx: config.default_idx,
    models,
  })
}

impl PhasePredictor {
  pub fn labels(&self) -> &[String] {
    &self.labels
  }

  pub fn hash_size(&self) -> usize {
    self.hash_size
  }

  pub fn predict(&self, features: &[f64]) -> Vec<f32> {
    if features.len() != self.hash_size {
      return self.default_distribution();
    }
    let mut scores = vec![0.0f64; self.labels.len()];
    for (idx, model) in self.models.iter().enumerate() {
      if let Some(model) = model
        && let Ok(score) = model.predict_single(features)
      {
        scores[idx] = score;
      }
    }
    let sum: f64 = scores.iter().sum();
    if sum > 0.0 {
      scores.iter_mut().for_each(|s| *s /= sum);
      scores.into_iter().map(|s| s as f32).collect()
    } else {
      self.default_distribution()
    }
  }

  fn default_distribution(&self) -> Vec<f32> {
    let mut scores = vec![0.0; self.labels.len()];
    if self.default_idx < scores.len() {
      scores[self.default_idx] = 1.0;
    }
    scores
  }
}

fn build_feature_matrix(samples: &[PhaseSample], feature_len: usize) -> Result<FeatureMatrix> {
  let rows = samples.len();
  let cols = feature_len;
  let mut data = Array2::<f64>::zeros((rows, cols));
  for (row, sample) in samples.iter().enumerate() {
    if sample.features.len() != cols {
      return Err(ZageError::ConfigError(format!(
        "phase feature length mismatch: expected {cols}, got {}",
        sample.features.len()
      )));
    }
    for (col, value) in sample.features.iter().enumerate() {
      data[(row, col)] = *value;
    }
  }
  FeatureMatrix::new(data).map_err(|err| ZageError::GenericError(Box::new(err)))
}

fn phase_feature_names() -> Vec<String> {
  (0..PHASE_FEATURES).map(|i| format!("p{i}")).collect()
}

fn parse_phase_pattern(pattern: &str) -> Option<PhasePattern> {
  let tokens = crate::tokenize::tokenize(pattern);
  let parts = crate::tokenize::extract_command_parts(pattern, &tokens)?;
  let head_glob = parts.head.trim().to_string();
  if head_glob.is_empty() {
    return None;
  }
  let flag_globs = parts.flags;
  let arg_globs = parts
    .args
    .into_iter()
    .map(|arg| arg.raw)
    .collect::<Vec<_>>();
  Some(PhasePattern {
    head_glob,
    flag_globs,
    arg_globs,
  })
}

impl PhasePattern {
  fn matches(&self, parts: &crate::tokenize::CommandParts) -> bool {
    if !glob_match(&self.head_glob, parts.head.trim()) {
      return false;
    }

    if !self.flag_globs.is_empty() {
      let mut remaining: Vec<&str> = parts.flags.iter().map(|f| f.as_str()).collect();
      for flag_glob in &self.flag_globs {
        if let Some(pos) = remaining
          .iter()
          .position(|flag| glob_match(flag_glob, flag))
        {
          remaining.remove(pos);
        } else {
          return false;
        }
      }
    }

    if !self.arg_globs.is_empty() {
      if self.arg_globs.len() > parts.args.len() {
        return false;
      }
      for (idx, arg_glob) in self.arg_globs.iter().enumerate() {
        let arg = &parts.args[idx].raw;
        if !glob_match(arg_glob, arg) {
          return false;
        }
      }
    }

    true
  }
}

fn glob_match(pattern: &str, text: &str) -> bool {
  glob_match_bytes(pattern.as_bytes(), text.as_bytes())
}

fn glob_match_bytes(pattern: &[u8], text: &[u8]) -> bool {
  let mut pi = 0usize;
  let mut ti = 0usize;
  let mut star_idx: Option<usize> = None;
  let mut match_idx = 0usize;

  while ti < text.len() {
    if pi < pattern.len() && (pattern[pi] == b'?' || pattern[pi] == text[ti]) {
      pi += 1;
      ti += 1;
      continue;
    }
    if pi < pattern.len() && pattern[pi] == b'*' {
      star_idx = Some(pi);
      match_idx = ti;
      pi += 1;
      continue;
    }
    if let Some(star) = star_idx {
      pi = star + 1;
      match_idx += 1;
      ti = match_idx;
      continue;
    }
    return false;
  }

  while pi < pattern.len() && pattern[pi] == b'*' {
    pi += 1;
  }

  pi == pattern.len()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::tokenize::tokenize;

  #[test]
  fn matches_patterns_with_word_boundary() {
    assert!(glob_match("git push*", "git push origin"));
    assert!(glob_match("git push", "git push"));
    assert!(!glob_match("git push", "git pushy"));
  }

  #[test]
  fn features_hash_to_fixed_size() {
    let tokens = tokenize("cargo build");
    let features = features_from_tokens(&tokens, PHASE_FEATURES);
    assert_eq!(features.len(), PHASE_FEATURES);
    assert!(features.iter().any(|v| *v > 0.0));
  }

  #[test]
  fn glob_match_supports_wildcards() {
    assert!(glob_match("git *", "git push"));
    assert!(glob_match("kubectl ?pply", "kubectl apply"));
    assert!(!glob_match("git push", "git pull"));
  }

  #[test]
  fn phase_pattern_matches_flags_any_order() {
    let pattern = parse_phase_pattern("git commit -m -S").unwrap();
    let tokens = crate::tokenize::tokenize("git commit -S -m \"msg\"");
    let parts =
      crate::tokenize::extract_command_parts("git commit -S -m \"msg\"", &tokens).unwrap();
    assert!(pattern.matches(&parts));
  }
}
