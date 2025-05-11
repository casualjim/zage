//! Length-delimited protocol implementation for efficient message encoding/decoding.
//!
//! This module provides a simple protocol for encoding and decoding messages
//! sent over a local socket. Each message is prefixed with its length (u32 LE),
//! followed by the raw bytes. This is efficient for both text and binary data.

use crate::Result;
use crate::ZageError;
use std::str;

/// Encoder for length-delimited messages
pub struct LengthDelimitedEncoder {
  buffer: Vec<u8>,
}

impl LengthDelimitedEncoder {
  /// Create a new encoder
  pub fn new() -> Self {
    Self { buffer: Vec::new() }
  }

  /// Encode a string (as UTF-8 bytes)
  pub fn encode_string(&mut self, s: &str) {
    self.encode_bytes(s.as_bytes());
  }

  /// Encode a vector of f32 values
  pub fn encode_f32_vec(&mut self, vec: &[f32]) {
    let mut bytes = Vec::with_capacity(vec.len() * 4);
    for &val in vec {
      bytes.extend_from_slice(&val.to_le_bytes());
    }
    self.encode_bytes(&bytes);
  }

  /// Encode a byte slice with length prefix
  pub fn encode_bytes(&mut self, bytes: &[u8]) {
    let len = bytes.len() as u32;
    self.buffer.extend_from_slice(&len.to_le_bytes());
    self.buffer.extend_from_slice(bytes);
  }

  /// Finish encoding and return the encoded data
  pub fn finish(self) -> Vec<u8> {
    self.buffer
  }
}

/// Decoder for length-delimited messages
pub struct LengthDelimitedDecoder<'a> {
  data: &'a [u8],
  position: usize,
}

impl<'a> LengthDelimitedDecoder<'a> {
  /// Create a new decoder
  pub fn new(data: &'a [u8]) -> Self {
    Self { data, position: 0 }
  }

  /// Decode the next message as a string
  pub fn decode_string(&mut self) -> Result<String> {
    let bytes = self.decode_bytes()?;
    let s = str::from_utf8(&bytes)
      .map_err(|e| ZageError::InvalidUtf8(e))?
      .to_string();
    Ok(s)
  }

  /// Decode the next message as a vector of f32 values
  pub fn decode_f32_vec(&mut self) -> Result<Vec<f32>> {
    let bytes = self.decode_bytes()?;
    if bytes.len() % 4 != 0 {
      return Err(ZageError::ConfigError(
        "Invalid f32 vector data: length not a multiple of 4".to_string(),
      ));
    }
    let mut result = Vec::with_capacity(bytes.len() / 4);
    let mut i = 0;
    while i < bytes.len() {
      let mut float_bytes = [0u8; 4];
      float_bytes.copy_from_slice(&bytes[i..i + 4]);
      result.push(f32::from_le_bytes(float_bytes));
      i += 4;
    }
    Ok(result)
  }

  /// Decode the next message as a byte vector
  pub fn decode_bytes(&mut self) -> Result<Vec<u8>> {
    if self.position + 4 > self.data.len() {
      return Err(ZageError::ConfigError(
        "Incomplete data: missing length prefix".to_string(),
      ));
    }
    let len_bytes = &self.data[self.position..self.position + 4];
    let len = u32::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
    self.position += 4;
    if self.position + len > self.data.len() {
      return Err(ZageError::ConfigError(
        "Incomplete data: not enough bytes for message".to_string(),
      ));
    }
    let bytes = self.data[self.position..self.position + len].to_vec();
    self.position += len;
    Ok(bytes)
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_length_delimited_string_roundtrip() {
    let test_str = "Hello, world! This is a test string with some repetition...";
    let mut encoder = LengthDelimitedEncoder::new();
    encoder.encode_string(test_str);
    let encoded = encoder.finish();
    let mut decoder = LengthDelimitedDecoder::new(&encoded);
    let decoded = decoder.decode_string().unwrap();
    assert_eq!(test_str, decoded);
  }

  #[test]
  fn test_length_delimited_f32_vec_roundtrip() {
    let test_vec = vec![1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0, 6.0];
    let mut encoder = LengthDelimitedEncoder::new();
    encoder.encode_f32_vec(&test_vec);
    let encoded = encoder.finish();
    let mut decoder = LengthDelimitedDecoder::new(&encoded);
    let decoded = decoder.decode_f32_vec().unwrap();
    assert_eq!(test_vec, decoded);
  }

  #[test]
  fn test_length_delimited_bytes_roundtrip() {
    let test_vec = vec![42u8; 300];
    let mut encoder = LengthDelimitedEncoder::new();
    encoder.encode_bytes(&test_vec);
    let encoded = encoder.finish();
    let mut decoder = LengthDelimitedDecoder::new(&encoded);
    let decoded = decoder.decode_bytes().unwrap();
    assert_eq!(test_vec, decoded);
  }
}
