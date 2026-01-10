use std::path::PathBuf;

pub(crate) const DEFAULT_EPOCHS: usize = 150;
pub(crate) const DEFAULT_NEGATIVES: usize = 6;
pub(crate) const DEFAULT_MIN_HISTORY: usize = 0;
pub(crate) const DEFAULT_MAX_SAMPLES: usize = 0;

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
  pub validation_top1: f64,
  pub model_path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct RerankContext {
  pub repo_root: String,
  pub recent_heads: Vec<String>,
  pub session_tokens: Vec<String>,
  pub session_phase: Option<String>,
  pub shellname: String,
  pub branch: Option<String>,
  pub time_bucket: u8,
  pub working_directory: Option<String>,
  pub hostname: Option<String>,
  pub username: Option<String>,
  pub session_id: Option<i64>,
  pub prev_exit_status: Option<i64>,
  pub now: i64,
}
