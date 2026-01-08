use std::env;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

use crate::{Result, ZageError};

const DEFAULT_LOW_CONF_TOP: f64 = 0.15;
const DEFAULT_LOW_CONF_MARGIN: f64 = 0.02;
const DEFAULT_RERANK_MIN_PROB: f64 = 0.30;
const DEFAULT_RERANK_MIN_MARGIN: f64 = 0.02;

#[derive(Debug, Clone, Deserialize, Default)]
struct LowConfidenceConfig {
  top: Option<f64>,
  margin: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RerankThresholdConfig {
  min_prob: Option<f64>,
  min_margin: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Default)]
struct RerankConfigFile {
  low_confidence: Option<LowConfidenceConfig>,
  rerank: Option<RerankThresholdConfig>,
}

#[derive(Debug, Clone)]
pub struct RerankConfig {
  pub low_confidence_top: f64,
  pub low_confidence_margin: f64,
  pub rerank_min_prob: f64,
  pub rerank_min_margin: f64,
}

impl Default for RerankConfig {
  fn default() -> Self {
    Self {
      low_confidence_top: DEFAULT_LOW_CONF_TOP,
      low_confidence_margin: DEFAULT_LOW_CONF_MARGIN,
      rerank_min_prob: DEFAULT_RERANK_MIN_PROB,
      rerank_min_margin: DEFAULT_RERANK_MIN_MARGIN,
    }
  }
}

impl RerankConfig {
  pub fn load() -> Result<Self> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = env::var("ZAGE_RERANK_CONFIG") {
      candidates.push(PathBuf::from(path));
    }
    candidates.push(PathBuf::from("config/rerank.toml"));
    if let Some(config_dir) = dirs::config_dir() {
      candidates.push(config_dir.join("zage").join("rerank.toml"));
    }

    for path in candidates {
      if path.exists() {
        let contents = fs::read_to_string(&path)?;
        return Self::from_str(&contents);
      }
    }

    Ok(Self::default())
  }

  fn from_str(contents: &str) -> Result<Self> {
    let parsed: RerankConfigFile =
      toml::from_str(contents).map_err(|err| ZageError::ConfigError(err.to_string()))?;
    let mut config = Self::default();
    if let Some(low) = parsed.low_confidence {
      if let Some(top) = low.top {
        config.low_confidence_top = top;
      }
      if let Some(margin) = low.margin {
        config.low_confidence_margin = margin;
      }
    }
    if let Some(rerank) = parsed.rerank {
      if let Some(min_prob) = rerank.min_prob {
        config.rerank_min_prob = min_prob;
      }
      if let Some(min_margin) = rerank.min_margin {
        config.rerank_min_margin = min_margin;
      }
    }
    Ok(config)
  }
}
