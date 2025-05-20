mod context;
pub mod contextual_features;
mod embedder;
pub mod feature_integration;
pub mod markov;
pub mod neural;
pub mod ngram;
pub mod sequence;
pub mod sequence_context;
pub mod training_dataset;

use candle_core::Device;
use std::env;
use std::sync::Arc;

use self::embedder::{InProcessEmbedder, RemoteEmbedder};
use crate::Result;
use crate::shell_history::Invocation;

pub use embedder::Embedder;

/// Environment variable to control socket path for remote embedder
pub const EMBEDDER_SOCKET_PATH_ENV: &str = "ZAGE_EMBEDDER_SOCKET_PATH";

/// Create a new embedder instance wrapped in an Arc
///
/// This function creates an embedder based on the `ZAGE_EMBEDDER_SOCKET_PATH` environment variable:
///
/// - If the environment variable is set, it creates an EmbedderClient that connects to the
///   socket server at the specified path
/// - If the environment variable is not set, it creates a PretrainedEmbedder that runs in-process
///
/// This allows for flexible testing and deployment configurations without changing code.
pub fn create_embedder(device: Device) -> Result<Arc<dyn Embedder>> {
  // Check if socket path environment variable is set
  match env::var(EMBEDDER_SOCKET_PATH_ENV) {
    Ok(socket_path) => {
      // Create client with the specified socket path
      let client = RemoteEmbedder::new(&socket_path, 30); // 30 second timeout
      Ok(Arc::new(client))
    }
    Err(_) => {
      // Environment variable not set, use in-process embedder
      let embedder = InProcessEmbedder::new(device)?;
      Ok(Arc::new(embedder))
    }
  }
}

/// Trait for command prediction models
pub trait PredictionModel {
  /// Train the model on a set of invocations
  fn train<I>(&mut self, invocations: I) -> Result<()>
  where
    I: IntoIterator<Item = Invocation>;

  /// Predict the next command based on recent history
  fn predict(&self, recent_history: &[Invocation], max_predictions: usize) -> Result<Vec<String>>;

  /// Update the model with new invocations
  fn update<I>(&mut self, invocations: I) -> Result<()>
  where
    I: IntoIterator<Item = Invocation>;
}
