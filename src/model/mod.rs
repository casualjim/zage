// Module for prediction models
mod context;
pub mod lstm;
pub mod markov;
pub mod ngram;
pub mod sequence;
mod sequence_context;

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
