mod protocol;
mod in_process;
mod remote;

use std::env;
use std::sync::Arc;
use candle_core::Device;

use crate::Result;

/// Trait defining the embedding interface
pub trait Embedder: Send + Sync {
    /// Embed a single text into a vector
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Environment variable to control socket path for remote embedder
pub const EMBEDDER_SOCKET_PATH_ENV: &str = "ZAGE_EMBEDDER_SOCKET_PATH";

/// Create an appropriate embedder based on environment configuration
///
/// If ZAGE_EMBEDDER_SOCKET_PATH is set, creates a remote embedder client
/// that connects to the specified socket path.
/// Otherwise, creates an in-process embedder that loads the model locally.
pub fn create_embedder(device: Device) -> Result<Arc<dyn Embedder>> {
    match env::var(EMBEDDER_SOCKET_PATH_ENV) {
        Ok(socket_path) => {
            // Create client with the specified socket path
            let client = remote::RemoteEmbedder::new(&socket_path, 30); // 30 second timeout
            Ok(Arc::new(client))
        }
        Err(_) => {
            // Create in-process embedder
            let embedder = in_process::InProcessEmbedder::new(device)?;
            Ok(Arc::new(embedder))
        }
    }
}

// Re-export the protocol message for the socket server
pub use protocol::ProtocolMessage;
