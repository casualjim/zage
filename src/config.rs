use std::env;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;

use serde::Deserialize;

use crate::{Result, ZageError};

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BackendMode {
  #[default]
  Server,
  Embedded,
}

impl BackendMode {
  pub fn is_embedded(self) -> bool {
    matches!(self, Self::Embedded)
  }
}

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum DbKind {
  #[default]
  Local,
  Remote,
  RemoteReplica,
}

#[derive(Debug, Clone)]
pub struct DbConfig {
  pub kind: DbKind,
  pub path: PathBuf,
  pub url: Option<String>,
  pub auth_token: Option<String>,
  pub encryption_key: Option<String>,
  pub encryption_cipher: Option<String>,
  pub remote_encryption_key: Option<String>,
  pub sync_interval_ms: Option<u64>,
}

impl DbConfig {
  pub fn with_cli_path(&self, cli_path: Option<&PathBuf>) -> Self {
    let mut next = self.clone();
    if let Some(path) = cli_path {
      next.path = path.clone();
    }
    next
  }

  pub fn resolved_auth_token(&self) -> Option<String> {
    self
      .auth_token
      .clone()
      .or_else(|| env::var("ZAGE_DB_AUTH_TOKEN").ok())
  }

  pub fn resolved_encryption_key(&self) -> Option<String> {
    self
      .encryption_key
      .clone()
      .or_else(|| env::var("ZAGE_DB_ENCRYPTION_KEY").ok())
  }

  pub fn resolved_remote_encryption_key(&self) -> Option<String> {
    self
      .remote_encryption_key
      .clone()
      .or_else(|| env::var("ZAGE_DB_REMOTE_ENCRYPTION_KEY").ok())
  }

  pub fn resolved_cipher(&self) -> Result<Option<libsql::Cipher>> {
    let Some(cipher) = self.encryption_cipher.as_ref() else {
      return Ok(None);
    };
    let parsed = libsql::Cipher::from_str(cipher)
      .map_err(|_| ZageError::ConfigError(format!("Unknown cipher '{cipher}'")))?;
    Ok(Some(parsed))
  }

  pub fn resolved_sync_interval_ms(&self) -> Option<u64> {
    self.sync_interval_ms.or_else(|| {
      env::var("ZAGE_DB_SYNC_INTERVAL_MS")
        .ok()
        .and_then(|v| v.parse().ok())
    })
  }
}

#[derive(Debug, Clone)]
pub struct AppConfig {
  pub backend: BackendMode,
  pub db: DbConfig,
  pub online_model: OnlineModelConfig,
}

#[derive(Debug, Deserialize, Default)]
struct AppConfigFile {
  backend: Option<BackendMode>,
  db: Option<DbConfigFile>,
  online_model: Option<OnlineModelConfigFile>,
}

#[derive(Debug, Deserialize, Default)]
struct DbConfigFile {
  #[serde(rename = "type")]
  kind: Option<DbKind>,
  path: Option<String>,
  url: Option<String>,
  auth_token: Option<String>,
  encryption_key: Option<String>,
  encryption_cipher: Option<String>,
  remote_encryption_key: Option<String>,
  sync_interval_ms: Option<u64>,
}

impl AppConfig {
  pub fn load() -> Result<Self> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(path) = env::var("ZAGE_CONFIG") {
      candidates.push(PathBuf::from(path));
    }
    candidates.push(PathBuf::from("config/zage.toml"));
    if let Some(config_dir) = dirs::config_dir() {
      candidates.push(config_dir.join("zage").join("config.toml"));
    }

    for path in candidates {
      if path.exists() {
        let contents = fs::read_to_string(&path)?;
        return Self::from_str(&contents);
      }
    }

    Self::default_config()
  }

  fn from_str(contents: &str) -> Result<Self> {
    let parsed: AppConfigFile =
      toml::from_str(contents).map_err(|err| ZageError::ConfigError(err.to_string()))?;
    Self::from_file(parsed)
  }

  fn from_file(parsed: AppConfigFile) -> Result<Self> {
    let mut config = Self::default_config()?;
    if let Some(backend) = parsed.backend {
      config.backend = backend;
    }
    if let Some(db) = parsed.db {
      if let Some(kind) = db.kind {
        config.db.kind = kind;
      }
      if let Some(path) = db.path {
        config.db.path = PathBuf::from(path);
      }
      if let Some(url) = db.url {
        config.db.url = Some(url);
      }
      if let Some(token) = db.auth_token {
        config.db.auth_token = Some(token);
      }
      if let Some(key) = db.encryption_key {
        config.db.encryption_key = Some(key);
      }
      if let Some(cipher) = db.encryption_cipher {
        config.db.encryption_cipher = Some(cipher);
      }
      if let Some(key) = db.remote_encryption_key {
        config.db.remote_encryption_key = Some(key);
      }
      if let Some(interval) = db.sync_interval_ms {
        config.db.sync_interval_ms = Some(interval);
      }
    }
    if let Some(online) = parsed.online_model {
      config.online_model.apply_file(online);
    }
    Ok(config)
  }

  fn default_config() -> Result<Self> {
    let path = default_db_path()?;
    Ok(Self {
      backend: BackendMode::default(),
      db: DbConfig {
        kind: DbKind::Local,
        path,
        url: None,
        auth_token: None,
        encryption_key: None,
        encryption_cipher: None,
        remote_encryption_key: None,
        sync_interval_ms: None,
      },
      online_model: OnlineModelConfig::default(),
    })
  }
}

const DEFAULT_ONLINE_DIM: usize = 64;
const DEFAULT_ONLINE_NEGATIVES: usize = 16;
const DEFAULT_ONLINE_WINDOW: usize = 10;
const DEFAULT_ONLINE_BUCKETS: u32 = crate::hash_util::SUBWORD_BUCKETS;
const DEFAULT_REPLAY_GLOBAL: usize = 20_000;
const DEFAULT_REPLAY_WORKSPACE: usize = 5_000;
const DEFAULT_REPLAY_MAX_WORKSPACES: usize = 50;
const DEFAULT_BLEND_ALPHA: f64 = 0.25;
const DEFAULT_BLEND_MARGIN_GATE: f64 = 0.05;
const DEFAULT_BLEND_MIN_SCORE_GATE: f64 = 0.0;

#[derive(Debug, Clone)]
pub struct OnlineModelConfig {
  pub embedding_dim: usize,
  pub negatives: usize,
  pub window: usize,
  pub bucket_count: u32,
  pub replay: OnlineModelReplayConfig,
  pub blend: OnlineModelBlendConfig,
}

#[derive(Debug, Clone)]
pub struct OnlineModelReplayConfig {
  pub global_capacity: usize,
  pub workspace_capacity: usize,
  pub max_workspaces: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct OnlineModelBlendConfig {
  pub alpha: f64,
  pub margin_gate: f64,
  pub min_score_gate: f64,
}

#[derive(Debug, Deserialize, Default)]
struct OnlineModelConfigFile {
  embedding_dim: Option<usize>,
  negatives: Option<usize>,
  window: Option<usize>,
  bucket_count: Option<u32>,
  replay: Option<OnlineModelReplayConfigFile>,
  blend: Option<OnlineModelBlendConfigFile>,
}

#[derive(Debug, Deserialize, Default)]
struct OnlineModelReplayConfigFile {
  global_capacity: Option<usize>,
  workspace_capacity: Option<usize>,
  max_workspaces: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
struct OnlineModelBlendConfigFile {
  alpha: Option<f64>,
  margin_gate: Option<f64>,
  min_score_gate: Option<f64>,
}

impl Default for OnlineModelConfig {
  fn default() -> Self {
    Self {
      embedding_dim: DEFAULT_ONLINE_DIM,
      negatives: DEFAULT_ONLINE_NEGATIVES,
      window: DEFAULT_ONLINE_WINDOW,
      bucket_count: DEFAULT_ONLINE_BUCKETS,
      replay: OnlineModelReplayConfig {
        global_capacity: DEFAULT_REPLAY_GLOBAL,
        workspace_capacity: DEFAULT_REPLAY_WORKSPACE,
        max_workspaces: DEFAULT_REPLAY_MAX_WORKSPACES,
      },
      blend: OnlineModelBlendConfig {
        alpha: DEFAULT_BLEND_ALPHA,
        margin_gate: DEFAULT_BLEND_MARGIN_GATE,
        min_score_gate: DEFAULT_BLEND_MIN_SCORE_GATE,
      },
    }
  }
}

impl OnlineModelConfig {
  pub fn load() -> Result<Self> {
    let config = AppConfig::load()?.online_model;
    config.validate()?;
    Ok(config)
  }

  fn apply_file(&mut self, parsed: OnlineModelConfigFile) {
    if let Some(dim) = parsed.embedding_dim {
      self.embedding_dim = dim;
    }
    if let Some(negatives) = parsed.negatives {
      self.negatives = negatives;
    }
    if let Some(window) = parsed.window {
      self.window = window;
    }
    if let Some(bucket_count) = parsed.bucket_count {
      self.bucket_count = bucket_count;
    }
    if let Some(replay) = parsed.replay {
      if let Some(global_capacity) = replay.global_capacity {
        self.replay.global_capacity = global_capacity;
      }
      if let Some(workspace_capacity) = replay.workspace_capacity {
        self.replay.workspace_capacity = workspace_capacity;
      }
      if let Some(max_workspaces) = replay.max_workspaces {
        self.replay.max_workspaces = max_workspaces;
      }
    }
    if let Some(blend) = parsed.blend {
      if let Some(alpha) = blend.alpha {
        self.blend.alpha = alpha;
      }
      if let Some(margin_gate) = blend.margin_gate {
        self.blend.margin_gate = margin_gate;
      }
      if let Some(min_score_gate) = blend.min_score_gate {
        self.blend.min_score_gate = min_score_gate;
      }
    }
    self.blend.margin_gate = self.blend.margin_gate.max(0.0);
  }

  pub fn model_version(&self) -> String {
    format!("v1-d{}-b{}", self.embedding_dim, self.bucket_count)
  }

  fn validate(&self) -> Result<()> {
    if self.embedding_dim == 0 {
      return Err(ZageError::ConfigError(
        "online_model.embedding_dim must be > 0".to_string(),
      ));
    }
    if self.negatives == 0 {
      return Err(ZageError::ConfigError(
        "online_model.negatives must be > 0".to_string(),
      ));
    }
    if self.window == 0 {
      return Err(ZageError::ConfigError(
        "online_model.window must be > 0".to_string(),
      ));
    }
    if self.bucket_count == 0 || !self.bucket_count.is_power_of_two() {
      return Err(ZageError::ConfigError(
        "online_model.bucket_count must be a power of two".to_string(),
      ));
    }
    if self.replay.global_capacity == 0
      || self.replay.workspace_capacity == 0
      || self.replay.max_workspaces == 0
    {
      return Err(ZageError::ConfigError(
        "online_model.replay capacities must be > 0".to_string(),
      ));
    }
    Ok(())
  }
}

fn default_db_path() -> Result<PathBuf> {
  if let Ok(path) = env::var("ZAGE_DB_PATH") {
    return Ok(PathBuf::from(path));
  }
  dirs::data_dir()
    .map(|v| v.join("zage/zage.db"))
    .ok_or_else(|| ZageError::ConfigError("Could not determine data directory".to_string()))
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn online_model_defaults_match_docs() {
    let config = OnlineModelConfig::default();
    assert_eq!(config.embedding_dim, 64);
    assert_eq!(config.negatives, 16);
    assert_eq!(config.window, 10);
    assert_eq!(config.bucket_count, 1 << 17);
    assert_eq!(config.replay.global_capacity, 20_000);
    assert_eq!(config.replay.workspace_capacity, 5_000);
    assert_eq!(config.replay.max_workspaces, 50);
    assert_eq!(config.blend.alpha, 0.25);
    assert_eq!(config.blend.margin_gate, 0.05);
    assert_eq!(config.blend.min_score_gate, 0.0);
  }

  #[test]
  fn online_model_parses_from_app_config() -> Result<()> {
    let config = AppConfig::from_str(
      r#"
[online_model]
embedding_dim = 96
negatives = 24
window = 12
bucket_count = 131072

[online_model.replay]
global_capacity = 1000
workspace_capacity = 200
max_workspaces = 8

[online_model.blend]
alpha = 0.5
margin_gate = 0.1
min_score_gate = 0.2
"#,
    )?;
    let online = config.online_model;
    assert_eq!(online.embedding_dim, 96);
    assert_eq!(online.negatives, 24);
    assert_eq!(online.window, 12);
    assert_eq!(online.bucket_count, 131072);
    assert_eq!(online.replay.global_capacity, 1000);
    assert_eq!(online.replay.workspace_capacity, 200);
    assert_eq!(online.replay.max_workspaces, 8);
    assert_eq!(online.blend.alpha, 0.5);
    assert_eq!(online.blend.margin_gate, 0.1);
    assert_eq!(online.blend.min_score_gate, 0.2);
    Ok(())
  }
}
