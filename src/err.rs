use miette::Diagnostic;
use std::error::Error as StdError;
use thiserror::Error;

pub type Result<T, E = ZageError> = std::result::Result<T, E>;

#[derive(Debug, Error, Diagnostic)]
pub enum ZageError {
  #[error("Invalid configuration: {0}")]
  #[diagnostic(code(zage::config_error))]
  ConfigError(String),

  #[error("Server request failed: {0}")]
  #[diagnostic(code(zage::server_request_error))]
  ServerRequestError(String),

  #[error("Server request timed out after {timeout_ms}ms: {context}")]
  #[diagnostic(code(zage::server_request_timeout))]
  ServerRequestTimeout { timeout_ms: u64, context: String },

  #[error("I/O error: {0}")]
  #[diagnostic(code(zage::io_error))]
  IoError(#[from] std::io::Error),

  #[error("Parse error: {0}")]
  #[diagnostic(code(zage::parse_error))]
  ParseError(#[from] std::num::ParseIntError),

  #[error("Database error: {0}")]
  #[diagnostic(code(zage::db_error))]
  DbError(#[from] libsql::Error),

  #[error("Invalid utf-8 bytes: {0}")]
  #[diagnostic(code(zage::invalid_utf8))]
  InvalidUtf8(#[from] std::str::Utf8Error),

  #[error("Serialization error: {0}")]
  #[diagnostic(code(zage::serialization_error))]
  SerializationError(#[from] serde_json::Error),

  #[error("YAML error: {0}")]
  #[diagnostic(code(zage::yaml_error))]
  YamlError(#[from] serde_yaml::Error),

  #[error("TOML edit error: {0}")]
  #[diagnostic(code(zage::toml_edit_error))]
  TomlEditError(#[from] toml_edit::TomlError),

  #[error("Generic error: {0}")]
  #[diagnostic(code(zage::generic_error))]
  GenericError(Box<dyn StdError + Send + Sync>),
}
