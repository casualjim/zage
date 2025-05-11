mod context;
pub mod contextual_features;
pub mod feature_integration;
pub mod markov;
pub mod neural;
pub mod ngram;
pub mod pretrained_embedder;
pub mod sequence;
mod sequence_context;
mod tokenizer;
pub mod training_dataset;

use crate::Result;
use crate::shell_history::Invocation;

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
