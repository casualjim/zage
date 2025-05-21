use std::io::{Read, Write};

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
                let encoded = encode_string(text);
                
                // Write payload length (4 bytes, little endian)
                let len_bytes = (encoded.len() as u32).to_le_bytes();
                writer.write_all(&len_bytes)?;
                
                // Write payload
                writer.write_all(&encoded)?;
            },
            Self::EmbedResponse(embedding) => {
                // Write message type (1 byte)
                writer.write_all(&[0x02])?;
                
                // Encode the embedding vector
                let encoded = encode_f32_vec(embedding);
                
                // Write payload length (4 bytes, little endian)
                let len_bytes = (encoded.len() as u32).to_le_bytes();
                writer.write_all(&len_bytes)?;
                
                // Write payload
                writer.write_all(&encoded)?;
            },
            Self::ErrorResponse(error) => {
                // Write message type (1 byte)
                writer.write_all(&[0x03])?;
                
                // Encode the error message
                let encoded = encode_string(error);
                
                // Write payload length (4 bytes, little endian)
                let len_bytes = (encoded.len() as u32).to_le_bytes();
                writer.write_all(&len_bytes)?;
                
                // Write payload
                writer.write_all(&encoded)?;
            },
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
                let text = decode_string(&payload)?;
                Ok(Self::EmbedRequest(text))
            },
            0x02 => {
                // Embed response
                let embedding = decode_f32_vec(&payload)?;
                Ok(Self::EmbedResponse(embedding))
            },
            0x03 => {
                // Error response
                let error = decode_string(&payload)?;
                Ok(Self::ErrorResponse(error))
            },
            _ => Err(ZageError::ConfigError(format!(
                "Invalid message type: {}", type_buf[0]
            ))),
        }
    }
}

// Simple encoding functions to replace the LengthDelimitedEncoder/Decoder

fn encode_string(s: &str) -> Vec<u8> {
    let bytes = s.as_bytes();
    let mut result = Vec::with_capacity(4 + bytes.len());
    
    // Add length prefix (4 bytes, little endian)
    result.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    
    // Add string bytes
    result.extend_from_slice(bytes);
    
    result
}

fn decode_string(data: &[u8]) -> Result<String> {
    if data.len() < 4 {
        return Err(ZageError::ConfigError("Invalid string encoding: too short".to_string()));
    }
    
    // Read length prefix
    let mut len_bytes = [0u8; 4];
    len_bytes.copy_from_slice(&data[0..4]);
    let str_len = u32::from_le_bytes(len_bytes) as usize;
    
    if data.len() < 4 + str_len {
        return Err(ZageError::ConfigError("Invalid string encoding: truncated data".to_string()));
    }
    
    // Convert bytes to string
    let string_bytes = &data[4..4+str_len];
    String::from_utf8(string_bytes.to_vec())
        .map_err(|e| ZageError::ConfigError(format!("Invalid UTF-8 string: {}", e)))
}

fn encode_f32_vec(vec: &[f32]) -> Vec<u8> {
    let mut result = Vec::with_capacity(4 + vec.len() * 4);
    
    // Add length prefix (4 bytes, little endian)
    result.extend_from_slice(&(vec.len() as u32).to_le_bytes());
    
    // Add f32 values
    for &val in vec {
        result.extend_from_slice(&val.to_le_bytes());
    }
    
    result
}

fn decode_f32_vec(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() < 4 {
        return Err(ZageError::ConfigError("Invalid f32 vector encoding: too short".to_string()));
    }
    
    // Read length prefix
    let mut len_bytes = [0u8; 4];
    len_bytes.copy_from_slice(&data[0..4]);
    let vec_len = u32::from_le_bytes(len_bytes) as usize;
    
    if data.len() < 4 + vec_len * 4 {
        return Err(ZageError::ConfigError("Invalid f32 vector encoding: truncated data".to_string()));
    }
    
    // Read f32 values
    let mut result = Vec::with_capacity(vec_len);
    for i in 0..vec_len {
        let start = 4 + i * 4;
        let mut val_bytes = [0u8; 4];
        val_bytes.copy_from_slice(&data[start..start+4]);
        result.push(f32::from_le_bytes(val_bytes));
    }
    
    Ok(result)
}
