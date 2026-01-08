use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::PathBuf;

use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::tensor::TensorData;
use burn::tensor::activation::softmax;
use rand::seq::SliceRandom;
use serde::Deserialize;

use crate::tokenize::Token;
use crate::{Result, ZageError};

const DEFAULT_PHASE: &str = "default";
const PHASE_FEATURES: usize = 512;
const PHASE_MIN_SAMPLES: usize = 100;
const PHASE_MIN_PER_CLASS: usize = 8;
const PHASE_DEFAULT_LIMIT: usize = 5_000;
const PHASE_EPOCHS: usize = 12;
const PHASE_BATCH_SIZE: usize = 64;
const PHASE_LR: f64 = 1e-2;

type PhaseBackend = NdArray<f32>;
type PhaseAutodiff = Autodiff<PhaseBackend>;

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
  pub features: Vec<f32>,
  pub label: usize,
}

#[derive(Debug, Clone)]
pub struct PhasePredictor {
  labels: Vec<String>,
  hash_size: usize,
  model: PhaseClassifier<PhaseBackend>,
}

#[derive(Module, Debug)]
struct PhaseClassifier<B: Backend> {
  linear: Linear<B>,
}

impl<B: Backend> PhaseClassifier<B> {
  fn new(features: usize, classes: usize, device: &B::Device) -> Self {
    Self {
      linear: LinearConfig::new(features, classes).init(device),
    }
  }

  fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
    self.linear.forward(input)
  }
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
    let tokens = crate::tokenize::tokenize(command);
    let Some(parts) = crate::tokenize::extract_command_parts(command, &tokens) else {
      return None;
    };
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

pub fn features_from_tokens(tokens: &[Token], hash_size: usize) -> Vec<f32> {
  let mut features = vec![0.0f32; hash_size];
  for token in tokens {
    let idx = (hash_token(&token.normalized) as usize) % hash_size;
    features[idx] += 1.0;
  }
  features
}

pub fn train_phase_predictor(
  config: &PhaseConfig,
  mut samples: Vec<PhaseSample>,
  unlabeled: Vec<Vec<f32>>,
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

  let device = Default::default();
  let mut model =
    PhaseClassifier::<PhaseAutodiff>::new(PHASE_FEATURES, config.labels.len(), &device);
  let mut optimizer = AdamConfig::new().init();

  let mut rng = rand::rng();
  for _epoch in 0..PHASE_EPOCHS {
    samples.shuffle(&mut rng);
    for batch in samples.chunks(PHASE_BATCH_SIZE) {
      if batch.is_empty() {
        continue;
      }
      let (x, y) = batch_tensors::<PhaseAutodiff>(batch, PHASE_FEATURES, &device);
      let logits = model.forward(x);
      let loss = CrossEntropyLossConfig::new()
        .init(&device)
        .forward(logits, y);
      let grads = loss.backward();
      let grads = GradientsParams::from_grads(grads, &model);
      model = optimizer.step(PHASE_LR, model, grads);
    }
  }

  let model = model.valid();
  Some(PhasePredictor {
    labels: config.labels.clone(),
    hash_size: PHASE_FEATURES,
    model,
  })
}

impl PhasePredictor {
  pub fn labels(&self) -> &[String] {
    &self.labels
  }

  pub fn hash_size(&self) -> usize {
    self.hash_size
  }

  pub fn predict(&self, features: &[f32]) -> Vec<f32> {
    let device = Default::default();
    let data = TensorData::new(features.to_vec(), [1, self.hash_size]);
    let input = Tensor::<PhaseBackend, 2>::from_data(data, &device);
    let logits = self.model.forward(input);
    let probs = softmax(logits, 1);
    probs
      .into_data()
      .into_vec::<f32>()
      .unwrap_or_else(|_| vec![0.0; self.labels.len()])
  }
}

fn batch_tensors<B: burn::tensor::backend::AutodiffBackend>(
  batch: &[PhaseSample],
  features: usize,
  device: &B::Device,
) -> (Tensor<B, 2>, Tensor<B, 1, Int>) {
  let mut flattened = Vec::with_capacity(batch.len() * features);
  let mut labels = Vec::with_capacity(batch.len());
  for sample in batch {
    flattened.extend_from_slice(&sample.features);
    labels.push(sample.label as i32);
  }
  let data = TensorData::new(flattened, [batch.len(), features]);
  let input = Tensor::<B, 2>::from_data(data, device);
  let label_data = TensorData::new(labels, [batch.len()]);
  let target = Tensor::<B, 1, Int>::from_data(label_data, device);
  (input, target)
}

fn hash_token(token: &str) -> u64 {
  const FNV_OFFSET: u64 = 0xcbf29ce484222325;
  const FNV_PRIME: u64 = 0x100000001b3;
  let mut hash = FNV_OFFSET;
  for byte in token.as_bytes() {
    hash ^= *byte as u64;
    hash = hash.wrapping_mul(FNV_PRIME);
  }
  hash
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
