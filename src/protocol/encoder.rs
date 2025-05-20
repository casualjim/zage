//! Length-delimited protocol implementation for efficient message encoding/decoding.
//!
//! This module provides a simple protocol for encoding and decoding messages
//! sent over a local socket. Each message is prefixed with its length (u32 LE),
//! followed by the raw bytes. This is efficient for both text and binary data.
//!
//! # Protocol Format
//!
//! Each message has the following binary format:
//! ```text
//! +-------------------+--------------------+
//! | Length (4 bytes)  | Payload (N bytes)  |
//! +-------------------+--------------------+
//! ```
//!
//! - Length: 32-bit unsigned integer in little-endian format, representing the length of the payload in bytes
//! - Payload: The actual message content (can be text, binary data, or serialized structures)

use crate::Result;
use crate::ZageError;
use std::str;

/// Encoder for length-delimited messages
///
/// This encoder handles the serialization of messages according to the length-delimited protocol.
/// It maintains an internal buffer that accumulates all encoded messages until `finish()` is called.
pub struct LengthDelimitedEncoder {
    buffer: Vec<u8>,
}

impl LengthDelimitedEncoder {
    /// Create a new encoder with an empty buffer
    ///
    /// # Examples
    ///
    /// ```
    /// use zage::protocol::encoder::LengthDelimitedEncoder;
    ///
    /// let encoder = LengthDelimitedEncoder::new();
    /// ```
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }
    
    /// Create a new encoder with a pre-allocated buffer capacity
    ///
    /// This can improve performance when encoding many messages or large payloads.
    ///
    /// # Examples
    ///
    /// ```
    /// use zage::protocol::encoder::LengthDelimitedEncoder;
    ///
    /// let encoder = LengthDelimitedEncoder::with_capacity(1024);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self { buffer: Vec::with_capacity(capacity) }
    }

    /// Encode a string as UTF-8 bytes with length prefix
    ///
    /// # Examples
    ///
    /// ```
    /// use zage::protocol::encoder::LengthDelimitedEncoder;
    ///
    /// let mut encoder = LengthDelimitedEncoder::new();
    /// encoder.encode_string("Hello, world!");
    /// ```
    pub fn encode_string(&mut self, s: &str) {
        self.encode_bytes(s.as_bytes());
    }

    /// Encode a vector of f32 values with length prefix
    ///
    /// Each f32 value is encoded in little-endian byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// use zage::protocol::encoder::LengthDelimitedEncoder;
    ///
    /// let mut encoder = LengthDelimitedEncoder::new();
    /// encoder.encode_f32_vec(&[1.0, 2.0, 3.0]);
    /// ```
    pub fn encode_f32_vec(&mut self, vec: &[f32]) {
        let mut bytes = Vec::with_capacity(vec.len() * 4);
        for &val in vec {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        self.encode_bytes(&bytes);
    }

    /// Encode a byte slice with length prefix
    ///
    /// This is the core encoding method that other encoding methods build upon.
    /// It prefixes the byte slice with its length as a 32-bit little-endian integer.
    ///
    /// # Examples
    ///
    /// ```
    /// use zage::protocol::encoder::LengthDelimitedEncoder;
    ///
    /// let mut encoder = LengthDelimitedEncoder::new();
    /// encoder.encode_bytes(&[1, 2, 3, 4]);
    /// ```
    ///
    /// # Panics
    ///
    /// This method will panic if the byte slice is larger than 4 GiB (2^32 - 1 bytes),
    /// as the length is encoded as a 32-bit integer.
    pub fn encode_bytes(&mut self, bytes: &[u8]) {
        let len = bytes.len();
        if len > u32::MAX as usize {
            panic!("Byte slice too large to encode: {} bytes exceeds maximum of {} bytes", len, u32::MAX);
        }
        
        let len = len as u32;
        self.buffer.extend_from_slice(&len.to_le_bytes());
        self.buffer.extend_from_slice(bytes);
    }

    /// Finish encoding and return the encoded data
    ///
    /// This consumes the encoder and returns the internal buffer containing all encoded messages.
    ///
    /// # Examples
    ///
    /// ```
    /// use zage::protocol::encoder::LengthDelimitedEncoder;
    ///
    /// let mut encoder = LengthDelimitedEncoder::new();
    /// encoder.encode_string("Hello");
    /// encoder.encode_string("World");
    /// let encoded_data = encoder.finish();
    /// ```
    pub fn finish(self) -> Vec<u8> {
        self.buffer
    }
    
    /// Returns the current size of the internal buffer in bytes
    ///
    /// This can be useful for monitoring the size of the encoded data before finishing.
    pub fn buffer_size(&self) -> usize {
        self.buffer.len()
    }
}

/// Decoder for length-delimited messages
///
/// This decoder handles the deserialization of messages according to the length-delimited protocol.
/// It maintains a position cursor that advances through the data buffer as messages are decoded.
pub struct LengthDelimitedDecoder<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> LengthDelimitedDecoder<'a> {
    /// Create a new decoder for the given data buffer
    ///
    /// # Examples
    ///
    /// ```
    /// use zage::protocol::encoder::LengthDelimitedDecoder;
    ///
    /// let data = vec![4, 0, 0, 0, b'T', b'e', b's', b't'];
    /// let decoder = LengthDelimitedDecoder::new(&data);
    /// ```
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, position: 0 }
    }
    
    /// Returns true if there is more data to decode
    ///
    /// # Examples
    ///
    /// ```
    /// use zage::protocol::encoder::LengthDelimitedDecoder;
    ///
    /// let data = vec![4, 0, 0, 0, b'T', b'e', b's', b't'];
    /// let mut decoder = LengthDelimitedDecoder::new(&data);
    /// assert!(decoder.has_more());
    /// ```
    pub fn has_more(&self) -> bool {
        // Need at least 4 bytes for the length prefix
        self.position + 4 <= self.data.len()
    }
    
    /// Returns the current position in the data buffer
    pub fn position(&self) -> usize {
        self.position
    }

    /// Decode the next message as a UTF-8 string
    ///
    /// # Examples
    ///
    /// ```
    /// use zage::protocol::encoder::{LengthDelimitedEncoder, LengthDelimitedDecoder};
    ///
    /// let mut encoder = LengthDelimitedEncoder::new();
    /// encoder.encode_string("Hello, world!");
    /// let data = encoder.finish();
    ///
    /// let mut decoder = LengthDelimitedDecoder::new(&data);
    /// let decoded = decoder.decode_string().unwrap();
    /// assert_eq!(decoded, "Hello, world!");
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - There is not enough data in the buffer
    /// - The length prefix is invalid
    /// - The data is not valid UTF-8
    pub fn decode_string(&mut self) -> Result<String> {
        let bytes = self.decode_bytes()?;
        let s = str::from_utf8(&bytes)
            .map_err(|e| ZageError::InvalidUtf8(e))?
            .to_string();
        Ok(s)
    }

    /// Decode the next message as a vector of f32 values
    ///
    /// Each f32 value is decoded from little-endian byte order.
    ///
    /// # Examples
    ///
    /// ```
    /// use zage::protocol::encoder::{LengthDelimitedEncoder, LengthDelimitedDecoder};
    ///
    /// let mut encoder = LengthDelimitedEncoder::new();
    /// encoder.encode_f32_vec(&[1.0, 2.0, 3.0]);
    /// let data = encoder.finish();
    ///
    /// let mut decoder = LengthDelimitedDecoder::new(&data);
    /// let decoded = decoder.decode_f32_vec().unwrap();
    /// assert_eq!(decoded, vec![1.0, 2.0, 3.0]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - There is not enough data in the buffer
    /// - The length prefix is invalid
    /// - The data length is not a multiple of 4 bytes (required for f32 values)
    pub fn decode_f32_vec(&mut self) -> Result<Vec<f32>> {
        let bytes = self.decode_bytes()?;
        if bytes.len() % 4 != 0 {
            return Err(ZageError::ConfigError(
                "Invalid f32 vector data: length not a multiple of 4 bytes".to_string(),
            ));
        }
        
        let mut result = Vec::with_capacity(bytes.len() / 4);
        for chunk in bytes.chunks_exact(4) {
            let mut float_bytes = [0u8; 4];
            float_bytes.copy_from_slice(chunk);
            result.push(f32::from_le_bytes(float_bytes));
        }
        
        Ok(result)
    }

    /// Decode the next message as a byte vector
    ///
    /// This is the core decoding method that other decoding methods build upon.
    /// It reads a 32-bit little-endian length prefix followed by that many bytes of data.
    ///
    /// # Examples
    ///
    /// ```
    /// use zage::protocol::encoder::{LengthDelimitedEncoder, LengthDelimitedDecoder};
    ///
    /// let mut encoder = LengthDelimitedEncoder::new();
    /// encoder.encode_bytes(&[1, 2, 3, 4]);
    /// let data = encoder.finish();
    ///
    /// let mut decoder = LengthDelimitedDecoder::new(&data);
    /// let decoded = decoder.decode_bytes().unwrap();
    /// assert_eq!(decoded, vec![1, 2, 3, 4]);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - There is not enough data in the buffer for the length prefix (4 bytes)
    /// - There is not enough data in the buffer for the message content
    pub fn decode_bytes(&mut self) -> Result<Vec<u8>> {
        // Ensure we have at least 4 bytes for the length
        if self.position + 4 > self.data.len() {
            return Err(ZageError::ConfigError(
                "Incomplete message: not enough data for length prefix".to_string(),
            ));
        }

        // Read the length prefix
        let mut len_bytes = [0u8; 4];
        len_bytes.copy_from_slice(&self.data[self.position..self.position + 4]);
        let len = u32::from_le_bytes(len_bytes) as usize;
        self.position += 4;

        // Ensure we have enough data for the message
        if self.position + len > self.data.len() {
            return Err(ZageError::ConfigError(
                format!(
                    "Incomplete message: expected {} bytes, but only {} available",
                    len,
                    self.data.len() - self.position
                )
            ));
        }

        // Extract the message bytes
        let result = if len == 0 {
            Vec::new()
        } else {
            self.data[self.position..self.position + len].to_vec()
        };
        self.position += len;

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_length_delimited_string_roundtrip() {
    let original = "hello world";
    let mut encoder = LengthDelimitedEncoder::new();
    encoder.encode_string(original);
    let encoded = encoder.finish();

    let mut decoder = LengthDelimitedDecoder::new(&encoded);
    let decoded = decoder.decode_string().unwrap();
    assert_eq!(original, decoded);
  }

  #[test]
  fn test_length_delimited_f32_vec_roundtrip() {
    let original = vec![1.0, 2.5, -3.0];
    let mut encoder = LengthDelimitedEncoder::new();
    encoder.encode_f32_vec(&original);
    let encoded = encoder.finish();

    let mut decoder = LengthDelimitedDecoder::new(&encoded);
    let decoded = decoder.decode_f32_vec().unwrap();
    assert_eq!(original, decoded);
  }

  #[test]
  fn test_length_delimited_bytes_roundtrip() {
    let original = vec![0x01, 0x02, 0x03, 0x04];
    let mut encoder = LengthDelimitedEncoder::new();
    encoder.encode_bytes(&original);
    let encoded = encoder.finish();

    let mut decoder = LengthDelimitedDecoder::new(&encoded);
    let decoded = decoder.decode_bytes().unwrap();
    assert_eq!(original, decoded);
  }
}
