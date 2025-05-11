use crate::{Result, ZageError};

/// Protocol message types
#[derive(Debug, Clone, PartialEq)]
pub enum MessageType {
  /// Request to embed text
  EmbedRequest,
  /// Response with embedding vector
  EmbedResponse,
  /// Error response
  ErrorResponse,
}

impl MessageType {
  /// Convert message type to byte
  pub fn to_byte(&self) -> u8 {
    match self {
      MessageType::EmbedRequest => 0x01,
      MessageType::EmbedResponse => 0x02,
      MessageType::ErrorResponse => 0xFF,
    }
  }

  /// Convert byte to message type
  pub fn from_byte(byte: u8) -> Result<Self> {
    match byte {
      0x01 => Ok(MessageType::EmbedRequest),
      0x02 => Ok(MessageType::EmbedResponse),
      0xFF => Ok(MessageType::ErrorResponse),
      _ => Err(ZageError::ConfigError(format!(
        "Invalid message type: {}",
        byte
      ))),
    }
  }
}
