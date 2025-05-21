use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

use crate::Result;
use crate::ZageError;
use crate::embedding::{Embedder, protocol::ProtocolMessage};

/// Client for the embedding socket server
pub struct RemoteEmbedder {
    socket_path: PathBuf,
    timeout_secs: u64,
}

impl Default for RemoteEmbedder {
    fn default() -> Self {
        Self {
            socket_path: "/tmp/zage_embedder.sock".into(),
            timeout_secs: 30,
        }
    }
}

impl RemoteEmbedder {
    /// Create a new client with custom settings
    pub fn new<P: Into<PathBuf>>(socket_path: P, timeout_secs: u64) -> Self {
        Self {
            socket_path: socket_path.into(),
            timeout_secs,
        }
    }

    /// Embed a text string
    pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
        // Connect to the socket
        let mut stream = UnixStream::connect(&self.socket_path)
            .map_err(|e| ZageError::ConfigError(format!("Failed to connect to socket: {}", e)))?;

        // Set timeouts
        stream.set_read_timeout(Some(Duration::from_secs(self.timeout_secs)))?;
        stream.set_write_timeout(Some(Duration::from_secs(self.timeout_secs)))?;

        // Create and send the embedding request message
        let request = ProtocolMessage::EmbedRequest(text.to_string());
        request.write_to(&mut stream)?;

        // Read the response message
        let response = ProtocolMessage::read_from(&mut stream)?;

        // Process the response based on its type
        match response {
            ProtocolMessage::EmbedResponse(embedding) => Ok(embedding),
            ProtocolMessage::ErrorResponse(error_msg) => Err(ZageError::ConfigError(format!(
                "Server error: {}",
                error_msg
            ))),
            _ => Err(ZageError::ConfigError(format!(
                "Unexpected response type: {:?}",
                response
            ))),
        }
    }
}

impl Embedder for RemoteEmbedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed(text)
    }
}
