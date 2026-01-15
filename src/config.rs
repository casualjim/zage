use std::env;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use clap::Args;
use confique::Config as _;
use confique::Layer as _;
use serde::Deserialize;

use crate::{Result, ZageError};

#[derive(Args, Debug, Clone, Default)]
pub struct ConfigArgs {
  /// Optional path to a config file to load in addition to the standard locations.
  #[arg(long = "config-file", global = true)]
  pub config_file: Option<PathBuf>,

  /// Path to the SQLite database file (overrides config/env).
  #[arg(long = "db-path", global = true)]
  pub db_path: Option<PathBuf>,
}

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

#[derive(confique::Config, Debug, Clone)]
pub struct AppConfig {
  #[config(default = "server")]
  pub backend: BackendMode,
  #[config(nested)]
  pub db: DbConfig,
  #[config(nested)]
  pub online_model: OnlineModelConfig,
}

#[derive(confique::Config, Debug, Clone)]
pub struct DbConfig {
  #[config(default = "local")]
  pub kind: DbKind,
  #[config(env = "ZAGE_DB_PATH")]
  pub path: PathBuf,
  pub url: Option<String>,
  #[config(env = "ZAGE_DB_AUTH_TOKEN")]
  pub auth_token: Option<String>,
  #[config(env = "ZAGE_DB_ENCRYPTION_KEY")]
  pub encryption_key: Option<String>,
  pub encryption_cipher: Option<String>,
  #[config(env = "ZAGE_DB_REMOTE_ENCRYPTION_KEY")]
  pub remote_encryption_key: Option<String>,
  #[config(env = "ZAGE_DB_SYNC_INTERVAL_MS")]
  pub sync_interval_ms: Option<u64>,
}

impl DbConfig {
  pub fn resolved_auth_token(&self) -> Option<String> {
    self.auth_token.clone()
  }

  pub fn resolved_encryption_key(&self) -> Option<String> {
    self.encryption_key.clone()
  }

  pub fn resolved_remote_encryption_key(&self) -> Option<String> {
    self.remote_encryption_key.clone()
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
    self.sync_interval_ms
  }
}

#[derive(confique::Config, Debug, Clone)]
#[config(validate = Self::validate)]
pub struct OnlineModelConfig {
  #[config(default = 64)]
  pub embedding_dim: usize,
  #[config(default = 16)]
  pub negatives: usize,
  #[config(default = 10)]
  pub window: usize,
  #[config(default = 131072)]
  pub bucket_count: u32,
  #[config(nested)]
  pub replay: OnlineModelReplayConfig,
  #[config(nested)]
  pub blend: OnlineModelBlendConfig,
}

#[derive(confique::Config, Debug, Clone)]
pub struct OnlineModelReplayConfig {
  #[config(default = 20000)]
  pub global_capacity: usize,
  #[config(default = 5000)]
  pub workspace_capacity: usize,
  #[config(default = 50)]
  pub max_workspaces: usize,
}

#[derive(confique::Config, Debug, Clone, Copy)]
pub struct OnlineModelBlendConfig {
  #[config(default = 0.25)]
  pub alpha: f64,
  #[config(default = 0.05)]
  pub margin_gate: f64,
  #[config(default = 0.0)]
  pub min_score_gate: f64,
}

impl Default for OnlineModelConfig {
  fn default() -> Self {
    Self {
      embedding_dim: 64,
      negatives: 16,
      window: 10,
      bucket_count: crate::hash_util::SUBWORD_BUCKETS,
      replay: OnlineModelReplayConfig::default(),
      blend: OnlineModelBlendConfig::default(),
    }
  }
}

impl Default for OnlineModelReplayConfig {
  fn default() -> Self {
    Self {
      global_capacity: 20_000,
      workspace_capacity: 5_000,
      max_workspaces: 50,
    }
  }
}

impl Default for OnlineModelBlendConfig {
  fn default() -> Self {
    Self {
      alpha: 0.25,
      margin_gate: 0.05,
      min_score_gate: 0.0,
    }
  }
}

impl AppConfig {
  pub fn load() -> Result<Self> {
    Self::load_with_args(None)
  }

  pub fn load_with_args(args: Option<&ConfigArgs>) -> Result<Self> {
    let cli_layer = cli_layer(args);
    let mut builder = AppConfig::builder().preloaded(cli_layer).env();

    if let Some(path) = explicit_config_path(args) {
      builder = builder.file(path);
    }

    if let Some(dir) = dirs::config_dir() {
      builder = add_existing_files(builder, xdg_config_candidates(&dir));
    }

    builder = builder.preloaded(dynamic_defaults()?);

    builder
      .load()
      .map_err(|err| ZageError::ConfigError(err.to_string()))
  }

  #[cfg(test)]
  fn from_str(contents: &str) -> Result<Self> {
    let layer: <AppConfig as confique::Config>::Layer =
      toml::from_str(contents).map_err(|err| ZageError::ConfigError(err.to_string()))?;
    AppConfig::builder()
      .preloaded(layer)
      .preloaded(dynamic_defaults()?)
      .load()
      .map_err(|err| ZageError::ConfigError(err.to_string()))
  }
}

impl OnlineModelConfig {
  pub fn load() -> Result<Self> {
    Ok(AppConfig::load()?.online_model)
  }

  pub fn model_version(&self) -> String {
    format!("v1-d{}-b{}", self.embedding_dim, self.bucket_count)
  }

  fn validate(&self) -> std::result::Result<(), String> {
    if self.embedding_dim == 0 {
      return Err("online_model.embedding_dim must be > 0".to_string());
    }
    if self.negatives == 0 {
      return Err("online_model.negatives must be > 0".to_string());
    }
    if self.window == 0 {
      return Err("online_model.window must be > 0".to_string());
    }
    if self.bucket_count == 0 || !self.bucket_count.is_power_of_two() {
      return Err("online_model.bucket_count must be a power of two".to_string());
    }
    if self.replay.global_capacity == 0
      || self.replay.workspace_capacity == 0
      || self.replay.max_workspaces == 0
    {
      return Err("online_model.replay capacities must be > 0".to_string());
    }
    Ok(())
  }
}

fn cli_layer(args: Option<&ConfigArgs>) -> <AppConfig as confique::Config>::Layer {
  let mut layer = <AppConfig as confique::Config>::Layer::empty();
  if let Some(args) = args
    && let Some(path) = args.db_path.clone()
  {
    layer.db.path = Some(path);
  }
  layer
}

fn explicit_config_path(args: Option<&ConfigArgs>) -> Option<PathBuf> {
  if let Some(args) = args
    && let Some(path) = args.config_file.clone()
  {
    return Some(path);
  }
  env::var("ZAGE_CONFIG").ok().map(PathBuf::from)
}

fn add_existing_files(
  mut builder: confique::Builder<AppConfig>,
  paths: Vec<PathBuf>,
) -> confique::Builder<AppConfig> {
  for path in paths {
    if path.exists() {
      builder = builder.file(path);
    }
  }
  builder
}

fn xdg_config_candidates(dir: &Path) -> Vec<PathBuf> {
  vec![dir.join("zage").join("config.toml")]
}

fn dynamic_defaults() -> Result<<AppConfig as confique::Config>::Layer> {
  let mut layer = <AppConfig as confique::Config>::Layer::empty();
  layer.db.path = Some(default_db_path()?);
  Ok(layer)
}

fn default_db_path() -> Result<PathBuf> {
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
