//! # Training Dataset Generation for Neural Model
//!
//! This module handles the preparation of training data for the neural model,
//! including sequence windowing, embedding extraction, and batching.
//!
//! The primary goal is to convert raw shell command history into structured
//! tensor data that can be efficiently processed by the Candle-based neural network.

use candle_core::{Device, Result as CandleResult, Tensor};
use rand::seq::SliceRandom;
use std::sync::Arc;

use crate::Result;
use crate::model::contextual_features::ContextualFeatures;
use crate::model::pretrained_embedder::PretrainedEmbedder;
use crate::shell_history::Invocation;

/// Represents a single training example for the neural model
pub struct TrainingExample {
  /// Input sequence of command embeddings (sequence of previous commands)
  pub input_embeddings: Vec<Vec<f32>>,
  /// Target command embedding (command to predict)
  pub target_embedding: Vec<f32>,
  /// Contextual features for the example
  pub context_features: Vec<f32>,
  /// Target command as string (for evaluation)
  pub target_command: String,
}

/// A dataset for training neural models
pub struct TrainingDataset {
  /// The sequence of training examples
  examples: Vec<TrainingExample>,
  /// Command embedder for converting text to vectors
  embedder: Arc<PretrainedEmbedder>,
  /// Contextual features extractor
  context_features: Arc<ContextualFeatures>,
  /// Maximum sequence length to consider
  sequence_length: usize,
  /// Device to use for tensor operations
  device: Device,
}

impl TrainingDataset {
  /// Create a new training dataset
  pub fn new(
    embedder: Arc<PretrainedEmbedder>,
    context_features: Arc<ContextualFeatures>,
    sequence_length: usize,
    device: Device,
  ) -> Self {
    Self {
      examples: Vec::new(),
      embedder,
      context_features,
      sequence_length,
      device,
    }
  }

  /// Generate training examples from invocation history
  pub fn generate_from_history(&mut self, invocations: &[Invocation]) -> Result<()> {
    if invocations.len() < self.sequence_length + 1 {
      // Not enough data for a single example
      return Ok(());
    }

    // Extract sliding windows of commands to create examples
    for window_start in 0..=(invocations.len() - self.sequence_length - 1) {
      let window_end = window_start + self.sequence_length;
      let input_sequence = &invocations[window_start..window_end];
      let target = &invocations[window_end];

      // Process this window into a training example
      let example = self.process_sequence(input_sequence, target)?;
      self.examples.push(example);
    }

    Ok(())
  }

  /// Process a sequence of invocations into a training example
  fn process_sequence(
    &self,
    input_sequence: &[Invocation],
    target: &Invocation,
  ) -> Result<TrainingExample> {
    // Embed each command in the input sequence
    let mut input_embeddings = Vec::with_capacity(input_sequence.len());
    for inv in input_sequence {
      let embedding = self.embedder.embed(&inv.command)?;
      input_embeddings.push(embedding);
    }

    // Embed the target command
    let target_embedding = self.embedder.embed(&target.command)?;

    // Extract contextual features from the target (the context where we make prediction)
    let context_features = self.context_features.encode_all_features(target)?;

    Ok(TrainingExample {
      input_embeddings,
      target_embedding,
      context_features,
      target_command: target.command.clone(),
    })
  }

  /// Split the dataset into training and validation sets
  pub fn split_train_val(
    &self,
    validation_ratio: f64,
  ) -> (Vec<&TrainingExample>, Vec<&TrainingExample>) {
    let validation_size = (self.examples.len() as f64 * validation_ratio) as usize;
    let training_size = self.examples.len() - validation_size;

    let train = self.examples.iter().take(training_size).collect();
    let val = self.examples.iter().skip(training_size).collect();

    (train, val)
  }

  /// Shuffle the training examples
  pub fn shuffle(&mut self) {
    let mut rng = rand::thread_rng();
    self.examples.shuffle(&mut rng);
  }

  /// Create batches for training
  pub fn create_batches(&self, batch_size: usize) -> Vec<TrainingBatch> {
    let mut batches = Vec::new();
    for chunk in self.examples.chunks(batch_size) {
      if let Ok(batch) = self.create_batch(chunk) {
        batches.push(batch);
      }
    }
    batches
  }

  /// Create a single batch from examples
  fn create_batch(&self, examples: &[TrainingExample]) -> Result<TrainingBatch> {
    if examples.is_empty() {
      return Err(crate::ZageError::ConfigError(
        "Cannot create batch from empty examples".to_string(),
      ));
    }

    // Get dimensions
    let batch_size = examples.len();
    let seq_len = self.sequence_length;
    let embedding_dim = examples[0].input_embeddings[0].len();
    let context_dim = examples[0].context_features.len();

    // Prepare data arrays
    let mut input_data = Vec::with_capacity(batch_size * seq_len * embedding_dim);
    let mut target_data = Vec::with_capacity(batch_size * embedding_dim);
    let mut context_data = Vec::with_capacity(batch_size * context_dim);
    let mut target_commands = Vec::with_capacity(batch_size);

    // Fill data arrays
    for example in examples {
      // Flatten input embeddings
      for embedding in &example.input_embeddings {
        input_data.extend_from_slice(embedding);
      }

      // Add target embedding
      target_data.extend_from_slice(&example.target_embedding);

      // Add context features
      context_data.extend_from_slice(&example.context_features);

      // Add target command string
      target_commands.push(example.target_command.clone());
    }

    // Create tensors
    let input_tensor = Tensor::from_vec(
      input_data,
      (batch_size, seq_len, embedding_dim),
      &self.device,
    )
    .map_err(|e| crate::ZageError::CandleError(e))?;

    let target_tensor = Tensor::from_vec(target_data, (batch_size, embedding_dim), &self.device)
      .map_err(|e| crate::ZageError::CandleError(e))?;

    let context_tensor = Tensor::from_vec(context_data, (batch_size, context_dim), &self.device)
      .map_err(|e| crate::ZageError::CandleError(e))?;

    Ok(TrainingBatch {
      input: input_tensor,
      target: target_tensor,
      context: context_tensor,
      target_commands,
    })
  }

  /// Get the number of examples in the dataset
  pub fn len(&self) -> usize {
    self.examples.len()
  }

  /// Check if the dataset is empty
  pub fn is_empty(&self) -> bool {
    self.examples.is_empty()
  }
}

/// A batch of training data as tensors
pub struct TrainingBatch {
  /// Input tensor of shape [batch_size, sequence_length, embedding_dim]
  pub input: Tensor,
  /// Target tensor of shape [batch_size, embedding_dim]
  pub target: Tensor,
  /// Context tensor of shape [batch_size, context_dim]
  pub context: Tensor,
  /// Target commands as strings (for evaluation)
  pub target_commands: Vec<String>,
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::shell_history::Invocation;
  use candle_core::Device;

  // Helper function to create test invocations
  fn create_test_invocations() -> Vec<Invocation> {
    vec![
      Invocation {
        command: "git status".to_string(),
        shellname: "bash".to_string(),
        working_directory: Some("/project".to_string()),
        hostname: Some("localhost".to_string()),
        username: Some("user".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(1000),
        end_unix_timestamp: Some(1005),
        session_id: 1,
      },
      Invocation {
        command: "git add .".to_string(),
        shellname: "bash".to_string(),
        working_directory: Some("/project".to_string()),
        hostname: Some("localhost".to_string()),
        username: Some("user".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(1010),
        end_unix_timestamp: Some(1015),
        session_id: 1,
      },
      Invocation {
        command: "git commit -m 'update'".to_string(),
        shellname: "bash".to_string(),
        working_directory: Some("/project".to_string()),
        hostname: Some("localhost".to_string()),
        username: Some("user".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(1020),
        end_unix_timestamp: Some(1025),
        session_id: 1,
      },
      Invocation {
        command: "git push".to_string(),
        shellname: "bash".to_string(),
        working_directory: Some("/project".to_string()),
        hostname: Some("localhost".to_string()),
        username: Some("user".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(1030),
        end_unix_timestamp: Some(1035),
        session_id: 1,
      },
    ]
  }

  #[test]
  fn test_dataset_generation() -> Result<()> {
    let device = Device::Cpu;

    // Initialize the embedder and context features
    let embedder = Arc::new(PretrainedEmbedder::new(device.clone())?);
    let context_features = Arc::new(ContextualFeatures::new(device.clone())?);

    // Create a dataset with sequence length 2
    let mut dataset = TrainingDataset::new(embedder.clone(), context_features.clone(), 2, device);

    // Generate examples from test invocations
    let invocations = create_test_invocations();
    dataset.generate_from_history(&invocations)?;

    // We should have 2 examples:
    // 1. [git status, git add .] → git commit -m 'update'
    // 2. [git add ., git commit -m 'update'] → git push
    assert_eq!(dataset.len(), 2, "Expected 2 training examples");

    // Test shuffling (just make sure it doesn't crash)
    dataset.shuffle();

    // Test train/val split
    let (train, val) = dataset.split_train_val(0.5);
    assert_eq!(train.len(), 1, "Expected 1 training example");
    assert_eq!(val.len(), 1, "Expected 1 validation example");

    // Test batch creation
    let batches = dataset.create_batches(1);
    assert_eq!(batches.len(), 2, "Expected 2 batches of size 1");

    Ok(())
  }
}
