use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Instant;

use gbrt_rs::{GradientBooster, ModelIO};
use serde::Deserialize;
use serde_json;
use tracing::warn;

use crate::core::{Candidate, Suggestion};
use crate::repo::read_git_branch;
use crate::rerank_config::RerankConfig;
use crate::{Result, ZageError};

use super::MODEL_NAME;
use super::calibration::{CalibrationParams, sigmoid};
use super::config::RerankContext;
use super::features::{build_feature_matrix, features_from_suggestion};

static MODEL_CACHE: OnceLock<RwLock<Option<Arc<GradientBooster>>>> = OnceLock::new();
static CALIBRATION_CACHE: OnceLock<RwLock<Option<CalibrationParams>>> = OnceLock::new();

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
  #[serde(default)]
  pub calibration: Option<CalibrationParams>,
}

pub(crate) struct ModelLocation {
  pub(crate) dir: PathBuf,
  pub(crate) name: String,
  pub(crate) model_path: PathBuf,
  pub(crate) metadata_path: PathBuf,
}

fn model_cache() -> &'static RwLock<Option<Arc<GradientBooster>>> {
  MODEL_CACHE.get_or_init(|| RwLock::new(None))
}

fn calibration_cache() -> &'static RwLock<Option<CalibrationParams>> {
  CALIBRATION_CACHE.get_or_init(|| RwLock::new(None))
}

fn load_model_cached() -> Result<Option<Arc<GradientBooster>>> {
  let cache = model_cache();
  let cached = match cache.read() {
    Ok(guard) => guard.clone(),
    Err(poisoned) => poisoned.into_inner().clone(),
  };
  if let Some(model) = cached {
    return Ok(Some(model));
  }
  let loaded = load_model()?.map(Arc::new);
  if let Some(ref model) = loaded
    && let Ok(mut guard) = cache.write()
  {
    *guard = Some(model.clone());
  }
  Ok(loaded)
}

fn load_calibration_cached() -> Result<Option<CalibrationParams>> {
  let cache = calibration_cache();
  let cached = match cache.read() {
    Ok(guard) => guard.clone(),
    Err(poisoned) => poisoned.into_inner().clone(),
  };
  if cached.is_some() {
    return Ok(cached);
  }
  let calibration = model_status()?.and_then(|status| status.calibration);
  if let Some(ref params) = calibration
    && let Ok(mut guard) = cache.write()
  {
    *guard = Some(params.clone());
  }
  Ok(calibration)
}

pub fn clear_model_cache() {
  if let Ok(mut guard) = model_cache().write() {
    *guard = None;
  }
  if let Ok(mut guard) = calibration_cache().write() {
    *guard = None;
  }
}

pub fn warm_model_cache() -> Result<()> {
  let _ = load_model_cached()?;
  let _ = load_calibration_cached()?;
  Ok(())
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
  clear_model_cache();
  Ok(())
}

pub(crate) fn rerank_suggestions(
  suggestions: &mut Vec<Suggestion>,
  candidates: &std::collections::HashMap<String, Candidate>,
  context: &RerankContext,
  config: &RerankConfig,
) -> Result<bool> {
  let Some(model) = load_model_cached()? else {
    return Ok(false);
  };
  if suggestions.is_empty() {
    return Ok(false);
  }

  let recent_heads: HashSet<String> = context.recent_heads.iter().cloned().collect();
  let mut vectors: Vec<Vec<f64>> = Vec::new();
  let mut indices: Vec<usize> = Vec::new();
  let mut tier1_scores: Vec<f64> = Vec::new();

  for (idx, suggestion) in suggestions.iter().enumerate() {
    if let Some(candidate) = candidates.get(&suggestion.command)
      && let Some(vector) = features_from_suggestion(suggestion, candidate, context, &recent_heads)
    {
      vectors.push(vector.values);
      indices.push(idx);
      tier1_scores.push(suggestion.score);
    }
  }

  if vectors.is_empty() {
    return Ok(false);
  }

  let start = Instant::now();
  let feature_matrix = build_feature_matrix(&vectors)?;
  let predictions = model
    .predict(&feature_matrix)
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;
  if config.rerank_timeout_ms > 0 && start.elapsed().as_millis() > config.rerank_timeout_ms as u128
  {
    warn!(
      "reranker inference exceeded {}ms (elapsed {}ms); skipping rerank",
      config.rerank_timeout_ms,
      start.elapsed().as_millis()
    );
    return Ok(false);
  }

  let calibration = load_calibration_cached()?;
  let mut updated = suggestions.clone();
  for (offset, idx) in indices.iter().enumerate() {
    if let Some(score) = predictions.get(offset) {
      let final_score = if let Some(ref params) = calibration {
        let p_tier1 = sigmoid(params.tier1_a * tier1_scores[offset] + params.tier1_b);
        let p_model = sigmoid(params.model_a * score + params.model_b);
        sigmoid(params.stack_w0 + params.stack_w1 * p_tier1 + params.stack_w2 * p_model)
      } else {
        *score
      };
      updated[*idx].score = final_score;
    }
  }

  updated.sort_by(|a, b| b.score.total_cmp(&a.score));

  let top = updated.first().map(|s| s.score).unwrap_or(0.0);
  let second = updated.get(1).map(|s| s.score).unwrap_or(0.0);
  if top < config.rerank_min_prob || (top - second) < config.rerank_min_margin {
    return Ok(false);
  }

  *suggestions = updated;
  Ok(true)
}

fn default_model_dir() -> Result<PathBuf> {
  if let Ok(path) = std::env::var("ZAGE_MODEL_PATH") {
    return Ok(PathBuf::from(path));
  }
  let base =
    dirs::data_dir().ok_or_else(|| ZageError::ConfigError("missing data dir".to_string()))?;
  Ok(base.join("zage/model"))
}

pub(crate) fn model_location() -> Result<ModelLocation> {
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

pub(crate) fn load_model() -> Result<Option<GradientBooster>> {
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

#[allow(clippy::too_many_arguments)]
pub fn runtime_context(
  repo_root: &str,
  recent_heads: &[String],
  session_tokens: Vec<String>,
  session_phase: Option<&str>,
  shellname: &str,
  working_directory: Option<&str>,
  hostname: Option<&str>,
  username: Option<&str>,
  session_id: Option<i64>,
  prev_exit_status: Option<i64>,
  now: i64,
) -> RerankContext {
  let branch = if repo_root.is_empty() {
    None
  } else {
    read_git_branch(repo_root).ok().flatten()
  };
  let time_bucket = timestamp_bucket(now);
  RerankContext {
    repo_root: repo_root.to_string(),
    recent_heads: recent_heads.to_vec(),
    session_tokens,
    session_phase: session_phase.map(|s| s.to_string()),
    shellname: shellname.to_string(),
    branch,
    time_bucket,
    working_directory: working_directory.map(|value| value.to_string()),
    hostname: hostname.map(|value| value.to_string()),
    username: username.map(|value| value.to_string()),
    session_id,
    prev_exit_status,
    now,
  }
}

pub(crate) fn unix_now() -> i64 {
  std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64
}

pub(crate) fn timestamp_bucket(ts: i64) -> u8 {
  if ts <= 0 {
    return 0;
  }
  let hour = ((ts / 3600) % 24) as u8;
  match hour {
    0..=5 => 1,
    6..=11 => 2,
    12..=17 => 3,
    _ => 4,
  }
}
