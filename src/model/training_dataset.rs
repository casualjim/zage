//! # Training Dataset Generation for Neural Model
//!
//! This module handles the preparation of training data for the neural model,
//! including sequence windowing, embedding extraction, and batching.
//!
//! The primary goal is to convert raw shell command history into structured
//! tensor data that can be efficiently processed by the Candle-based neural network.

use candle_core::{Device, Tensor};
use rand::seq::SliceRandom;
use std::sync::Arc;

use crate::Result;
use crate::model::contextual_features::ContextualFeatures;
use crate::embedding::Embedder;
use crate::shell_history::Invocation;

/// Represents a single training example for the neural model
pub struct TrainingExample {
  /// Input sequence of command embeddings (sequence of previous commands)
  pub input_embeddings: Vec<Vec<f32>>,
  /// Input sequence of context features (for each command)
  pub input_context_features: Vec<Vec<f32>>,
  /// Target command embedding
  pub target_embedding: Vec<f32>,
  /// Target context features
  pub target_context_features: Vec<f32>,
  /// Target command as string (for evaluation)
  pub target_command: String,
}

/// A dataset for training neural models
pub struct TrainingDataset {
  /// Pretrained embedder for command embeddings
  embedder: Arc<dyn Embedder>,
  /// Contextual feature extractor
  context_features: Arc<ContextualFeatures>,
  /// Length of input sequences
  sequence_length: usize,
  /// Training examples
  examples: Vec<TrainingExample>,
  /// Device to use for tensors
  device: Device,
}

impl TrainingDataset {
  /// Create a new training dataset
  pub fn new(
    embedder: Arc<dyn Embedder>,
    context_features: Arc<ContextualFeatures>,
    sequence_length: usize,
    device: Device,
  ) -> Self {
    Self {
      embedder,
      context_features,
      sequence_length,
      examples: Vec::new(),
      device,
    }
  }

  /// Generate training examples from invocation history
  pub fn generate_from_history(&mut self, invocations: &[Invocation]) -> Result<()> {
    // Need at least sequence_length + 1 invocations to create an example
    if invocations.len() <= self.sequence_length {
      return Ok(());
    }

    // Create sliding windows over the invocations
    for i in 0..invocations.len() - self.sequence_length {
      let input_sequence = &invocations[i..i + self.sequence_length];
      let target = &invocations[i + self.sequence_length];

      let example = self.process_sequence(input_sequence, target)?;
      self.examples.push(example);
    }

    Ok(())
  }

  /// Process a sequence of invocations into a training example
  pub fn process_sequence(
    &self,
    input_sequence: &[Invocation],
    target: &Invocation,
  ) -> Result<TrainingExample> {
    // Process input sequence
    let mut input_embeddings = Vec::with_capacity(input_sequence.len());
    let mut input_context_features = Vec::with_capacity(input_sequence.len());

    for inv in input_sequence {
      // Get command embedding
      let embedding = self.embedder.embed(&inv.command)?;
      input_embeddings.push(embedding);

      // Get context features
      let features = self.context_features.encode_all_features(inv)?;
      input_context_features.push(features);
    }

    // Process target
    let target_embedding = self.embedder.embed(&target.command)?;
    let target_context_features = self.context_features.encode_all_features(target)?;

    Ok(TrainingExample {
      input_embeddings,
      input_context_features,
      target_embedding,
      target_context_features,
      target_command: target.command.clone(),
    })
  }

  /// Split the dataset into training and validation sets
  pub fn split_train_val(
    &self,
    validation_ratio: f64,
  ) -> (Vec<&TrainingExample>, Vec<&TrainingExample>) {
    let val_size = (self.examples.len() as f64 * validation_ratio).round() as usize;
    let train_size = self.examples.len() - val_size;

    let train: Vec<&TrainingExample> = self.examples[..train_size].iter().collect();
    let val: Vec<&TrainingExample> = self.examples[train_size..].iter().collect();

    (train, val)
  }

  /// Shuffle the training examples
  pub fn shuffle(&mut self) {
    let mut rng = rand::rng();
    self.examples.shuffle(&mut rng);
  }

  /// Create batches for training
  pub fn create_batches(&self, batch_size: usize) -> Vec<TrainingBatch> {
    let num_batches = (self.examples.len() + batch_size - 1) / batch_size;
    let mut batches = Vec::with_capacity(num_batches);

    for i in 0..num_batches {
      let start = i * batch_size;
      let end = (start + batch_size).min(self.examples.len());
      let examples = &self.examples[start..end];

      if let Ok(batch) = self.create_batch(examples) {
        batches.push(batch);
      }
    }

    batches
  }

  /// Create a single batch from examples
  pub fn create_batch(&self, examples: &[TrainingExample]) -> Result<TrainingBatch> {
    if examples.is_empty() {
      return Err(crate::ZageError::ConfigError(
        "Cannot create batch from empty examples".to_string(),
      ));
    }

    // Dimensions
    let batch_size = examples.len();
    let seq_len = self.sequence_length;
    let emb_dim = examples[0].input_embeddings[0].len();
    let ctx_dim = examples[0].input_context_features[0].len();

    // Flatten input embeddings
    let mut flat_input_emb = Vec::with_capacity(batch_size * seq_len * emb_dim);
    for example in examples {
      for emb in &example.input_embeddings {
        flat_input_emb.extend_from_slice(emb);
      }
    }

    // Flatten input context features
    let mut flat_input_ctx = Vec::with_capacity(batch_size * seq_len * ctx_dim);
    for example in examples {
      for ctx in &example.input_context_features {
        flat_input_ctx.extend_from_slice(ctx);
      }
    }

    // Flatten target embeddings
    let mut flat_target_emb = Vec::with_capacity(batch_size * emb_dim);
    for example in examples {
      flat_target_emb.extend_from_slice(&example.target_embedding);
    }

    // Flatten target context features
    let mut flat_target_ctx = Vec::with_capacity(batch_size * ctx_dim);
    for example in examples {
      flat_target_ctx.extend_from_slice(&example.target_context_features);
    }

    // Create tensors
    let input_emb_tensor = Tensor::from_vec(
      flat_input_emb,
      &[batch_size, seq_len, emb_dim],
      &self.device,
    )?;

    let input_ctx_tensor = Tensor::from_vec(
      flat_input_ctx,
      &[batch_size, seq_len, ctx_dim],
      &self.device,
    )?;

    let target_emb_tensor =
      Tensor::from_vec(flat_target_emb, &[batch_size, emb_dim], &self.device)?;

    let target_ctx_tensor =
      Tensor::from_vec(flat_target_ctx, &[batch_size, ctx_dim], &self.device)?;

    // Collect target commands for evaluation
    let target_commands: Vec<String> = examples.iter().map(|e| e.target_command.clone()).collect();

    Ok(TrainingBatch {
      input_embeddings: input_emb_tensor,
      input_context_features: input_ctx_tensor,
      target_embeddings: target_emb_tensor,
      target_context_features: target_ctx_tensor,
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
  /// Input sequence embeddings [batch_size, seq_len, emb_dim]
  pub input_embeddings: Tensor,
  /// Input sequence context features [batch_size, seq_len, ctx_dim]
  pub input_context_features: Tensor,
  /// Target embeddings [batch_size, emb_dim]
  pub target_embeddings: Tensor,
  /// Target context features [batch_size, ctx_dim]
  pub target_context_features: Tensor,
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
        command: "cd /home/user/projects".to_string(),
        working_directory: Some("/home/user".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(1620000000),
        shellname: "bash".to_string(),
        session_id: 1,
        hostname: None,
        username: None,
        end_unix_timestamp: None,
      },
      Invocation {
        command: "git status".to_string(),
        working_directory: Some("/home/user/projects".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(1620000100),
        shellname: "bash".to_string(),
        session_id: 1,
        hostname: None,
        username: None,
        end_unix_timestamp: None,
      },
      Invocation {
        command: "git add .".to_string(),
        working_directory: Some("/home/user/projects".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(1620000200),
        shellname: "bash".to_string(),
        session_id: 1,
        hostname: None,
        username: None,
        end_unix_timestamp: None,
      },
      Invocation {
        command: "git commit -m 'update'".to_string(),
        working_directory: Some("/home/user/projects".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(1620000300),
        shellname: "bash".to_string(),
        session_id: 1,
        hostname: None,
        username: None,
        end_unix_timestamp: None,
      },
      Invocation {
        command: "git push".to_string(),
        working_directory: Some("/home/user/projects".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(1620000400),
        shellname: "bash".to_string(),
        session_id: 1,
        hostname: None,
        username: None,
        end_unix_timestamp: None,
      },
    ]
  }

  #[test]
  fn test_dataset_generation() -> Result<()> {
    let device = Device::Cpu;

    // Initialize the embedder and context features
    let embedder = crate::embedding::create_embedder(device.clone())?;
    let context_features = Arc::new(crate::model::contextual_features::ContextualFeatures::new(
      device.clone(),
      embedder.clone(),
    )?);

    // Create a dataset with sequence length 2
    let mut dataset = TrainingDataset::new(embedder.clone(), context_features.clone(), 2, device);

    // Generate examples from test invocations
    let invocations = create_test_invocations();
    dataset.generate_from_history(&invocations)?;

    // Check that we have the expected number of examples
    // With 5 invocations and sequence length 2, we should have 3 examples
    assert_eq!(
      dataset.len(),
      3,
      "Should have 3 examples with sequence length 2"
    );

    // Create batches
    let batches = dataset.create_batches(2);
    assert_eq!(batches.len(), 2, "Should have 2 batches with batch size 2");

    // Check first batch dimensions
    let batch = &batches[0];
    assert_eq!(
      batch.input_embeddings.shape().dims(),
      &[2, 2, batch.input_embeddings.shape().dims()[2]],
      "Input embeddings should have shape [batch_size, seq_len, emb_dim]"
    );

    assert_eq!(
      batch.target_commands.len(),
      2,
      "Should have 2 target commands in the batch"
    );

    Ok(())
  }
}
