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
}

#[derive(Debug, Deserialize, Default)]
struct AppConfigFile {
  backend: Option<BackendMode>,
  db: Option<DbConfigFile>,
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
    })
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
