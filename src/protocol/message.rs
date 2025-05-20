use std::io::{Read, Write};

use crate::protocol::encoder::{LengthDelimitedDecoder, LengthDelimitedEncoder};
use crate::{Result, ZageError};

/// Protocol message enum for embedding requests and responses
#[derive(Debug, Clone)]
pub enum ProtocolMessage {
  /// Request to embed text
  EmbedRequest(String),

  /// Response with embedding vector
  EmbedResponse(Vec<f32>),

  /// Error response
  ErrorResponse(String),
}

impl ProtocolMessage {
  /// Write the message to a stream
  pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<()> {
    match self {
      Self::EmbedRequest(text) => {
        // Write message type (1 byte)
        writer.write_all(&[0x01])?;

        // Encode the text
        let mut encoder = LengthDelimitedEncoder::new();
        encoder.encode_string(text);
        let encoded = encoder.finish();

        // Write payload length (4 bytes, little endian)
        let len_bytes = (encoded.len() as u32).to_le_bytes();
        writer.write_all(&len_bytes)?;

        // Write payload
        writer.write_all(&encoded)?;
      }
      Self::EmbedResponse(embedding) => {
        // Write message type (1 byte)
        writer.write_all(&[0x02])?;

        // Encode the embedding vector
        let mut encoder = LengthDelimitedEncoder::new();
        encoder.encode_f32_vec(embedding);
        let encoded = encoder.finish();

        // Write payload length (4 bytes, little endian)
        let len_bytes = (encoded.len() as u32).to_le_bytes();
        writer.write_all(&len_bytes)?;

        // Write payload
        writer.write_all(&encoded)?;
      }
      Self::ErrorResponse(error) => {
        // Write message type (1 byte)
        writer.write_all(&[0x03])?;

        // Encode the error message
        let mut encoder = LengthDelimitedEncoder::new();
        encoder.encode_string(error);
        let encoded = encoder.finish();

        // Write payload length (4 bytes, little endian)
        let len_bytes = (encoded.len() as u32).to_le_bytes();
        writer.write_all(&len_bytes)?;

        // Write payload
        writer.write_all(&encoded)?;
      }
    }

    writer.flush()?;
    Ok(())
  }

  /// Read a message from a stream
  pub fn read_from<R: Read>(reader: &mut R) -> Result<Self> {
    // Read message type (1 byte)
    let mut type_buf = [0u8; 1];
    reader.read_exact(&mut type_buf)?;

    // Read payload length (4 bytes, little endian)
    let mut len_buf = [0u8; 4];
    reader.read_exact(&mut len_buf)?;
    let payload_len = u32::from_le_bytes(len_buf) as usize;

    // Read payload
    let mut payload = vec![0u8; payload_len];
    reader.read_exact(&mut payload)?;

    // Decode based on message type
    match type_buf[0] {
      0x01 => {
        // Embed request
        let mut decoder = LengthDelimitedDecoder::new(&payload);
        let text = decoder.decode_string()?;
        Ok(Self::EmbedRequest(text))
      }
      0x02 => {
        // Embed response
        let mut decoder = LengthDelimitedDecoder::new(&payload);
        let embedding = decoder.decode_f32_vec()?;
        Ok(Self::EmbedResponse(embedding))
      }
      0x03 => {
        // Error response
        let mut decoder = LengthDelimitedDecoder::new(&payload);
        let error = decoder.decode_string()?;
        Ok(Self::ErrorResponse(error))
      }
      _ => Err(ZageError::ConfigError(format!(
        "Invalid message type: {}",
        type_buf[0]
      ))),
    }
  }
}
