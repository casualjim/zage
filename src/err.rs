use miette::Diagnostic;
use std::error::Error as StdError;
use thiserror::Error;

pub type Result<T, E = ZageError> = std::result::Result<T, E>;

#[derive(Debug, Error, Diagnostic)]
pub enum ZageError {
  #[error("Invalid configuration: {0}")]
  #[diagnostic(code(zage::config_error))]
  ConfigError(String),

  #[error("I/O error")]
  #[diagnostic(code(zage::io_error))]
  IoError(#[from] std::io::Error),

  #[error("Parse error")]
  #[diagnostic(code(zage::parse_error))]
  ParseError(#[from] std::num::ParseIntError),

  #[error("Database error")]
  #[diagnostic(code(zage::db_error))]
  DbError(#[from] libsql::Error),

  #[error("Invalid utf-8 bytes")]
  #[diagnostic(code(zage::invalid_utf8))]
  InvalidUtf8(#[from] std::str::Utf8Error),

  #[error("Serialization error")]
  #[diagnostic(code(zage::serialization_error))]
  SerializationError(#[from] serde_json::Error),

  #[error("Generic error")]
  #[diagnostic(code(zage::generic_error))]
  GenericError(Box<dyn StdError + Send + Sync>),
}
