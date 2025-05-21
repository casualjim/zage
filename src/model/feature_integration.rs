//! # Feature Integration for Neural Models
//!
//! This module handles the integration of command embeddings and contextual features
//! into a unified representation for neural network input.
//!
//! The main functions are:
//! 1. Concatenate command embeddings with contextual features
//! 2. Optionally normalize the combined features
//! 3. Format data into tensors suitable for the neural model

use candle_core::{Device, Tensor};
use std::sync::Arc;

use crate::Result;
use crate::embedding::Embedder;
use crate::model::contextual_features::ContextualFeatures;
use crate::shell_history::Invocation;

/// Configuration for feature integration
#[derive(Debug, Clone, Copy)]
pub struct FeatureIntegrationConfig {
  /// Whether to normalize features to unit length
  pub normalize_features: bool,
}

impl Default for FeatureIntegrationConfig {
  fn default() -> Self {
    Self {
      normalize_features: true,
    }
  }
}

/// Handles the integration of command embeddings and contextual features
pub struct FeatureIntegrator {
  /// Command embedder
  embedder: Arc<dyn Embedder>,
  /// Context feature extractor
  context_features: Arc<ContextualFeatures>,
  /// Configuration for feature integration
  config: FeatureIntegrationConfig,
  /// Device to use for tensor operations
  device: Device,
}

impl FeatureIntegrator {
  /// Create a new feature integrator with default configuration
  pub fn new(
    embedder: Arc<dyn Embedder>,
    context_features: Arc<ContextualFeatures>,
    device: Device,
  ) -> Result<Self> {
    Self::with_config(
      embedder,
      context_features,
      FeatureIntegrationConfig::default(),
      device,
    )
  }

  /// Create a new feature integrator with custom configuration
  pub fn with_config(
    embedder: Arc<dyn Embedder>,
    context_features: Arc<ContextualFeatures>,
    config: FeatureIntegrationConfig,
    device: Device,
  ) -> Result<Self> {
    Ok(Self {
      embedder,
      context_features,
      config,
      device,
    })
  }

  /// Integrate command embedding and context features for a single invocation
  /// using simple concatenation
  pub fn integrate_features(&self, invocation: &Invocation) -> Result<Vec<f32>> {
    // Get command embedding
    let cmd_embedding = self.embedder.embed(&invocation.command)?;

    // Get contextual features
    let ctx_features = self.context_features.encode_all_features(invocation)?;

    // Concatenate features
    let mut combined = Vec::with_capacity(cmd_embedding.len() + ctx_features.len());
    combined.extend_from_slice(&cmd_embedding);
    combined.extend_from_slice(&ctx_features);

    // Normalize if configured
    if self.config.normalize_features {
      Ok(Self::normalize_vector(&combined))
    } else {
      Ok(combined)
    }
  }

  /// Create integrated feature tensor for a sequence of invocations
  pub fn create_integrated_tensor(&self, invocations: &[Invocation]) -> Result<Tensor> {
    if invocations.is_empty() {
      // Handle empty input with a zero tensor of appropriate shape
      // This is a fallback that should rarely be needed
      let feature_dim = match invocations.first() {
        Some(inv) => self.get_feature_dimension(inv)?,
        None => {
          // If we have no example, create a dummy invocation to determine dimensions
          let dummy = Invocation::default();
          self.get_feature_dimension(&dummy)?
        }
      };
      return Ok(Tensor::zeros(
        &[0, feature_dim],
        candle_core::DType::F32,
        &self.device,
      )?);
    }

    // Process each invocation to get integrated features
    let mut all_features = Vec::with_capacity(invocations.len());
    for inv in invocations {
      let features = self.integrate_features(inv)?;
      all_features.push(features);
    }

    // Convert to tensor
    let feature_dim = all_features[0].len();
    let mut flat_features = Vec::with_capacity(invocations.len() * feature_dim);
    for features in &all_features {
      flat_features.extend_from_slice(features);
    }

    // Create tensor with shape [sequence_length, feature_dim]
    let tensor = Tensor::from_vec(
      flat_features,
      &[invocations.len(), feature_dim],
      &self.device,
    )?;

    Ok(tensor)
  }

  /// Normalize a vector to unit length (L2 normalization)
  pub fn normalize_vector(vector: &[f32]) -> Vec<f32> {
    let norm_squared: f32 = vector.iter().map(|&x| x * x).sum();

    // Avoid division by zero
    if norm_squared < 1e-10 {
      return vector.to_vec();
    }

    let norm = norm_squared.sqrt();
    vector.iter().map(|&x| x / norm).collect()
  }

  /// Legacy method for backward compatibility
  pub fn normalize_features(v: &[f32]) -> Vec<f32> {
    Self::normalize_vector(v)
  }

  /// Get the dimension of the integrated features
  pub fn get_feature_dimension(&self, invocation: &Invocation) -> Result<usize> {
    let cmd_dim = self.embedder.embed(&invocation.command)?.len();
    let ctx_dim = self.context_features.encode_all_features(invocation)?.len();
    Ok(cmd_dim + ctx_dim)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::embedding::create_embedder;
  use crate::shell_history::Invocation;
  use candle_core::Device;

  // Helper function to create a test invocation
  fn create_test_invocation() -> Invocation {
    Invocation {
      command: "git commit -m 'test commit'".to_string(),
      working_directory: Some("/home/user/projects/test".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(1620000000),
      hostname: Some("testhost".to_string()),
      username: Some("testuser".to_string()),
      shellname: "bash".to_string(),
      session_id: 1,
      end_unix_timestamp: None,
    }
  }

  // Helper function to create multiple test invocations
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
    ]
  }

  #[test]
  fn test_feature_integration_basic() -> Result<()> {
    let device = Device::Cpu;

    // Initialize embedder and context features
    let embedder = crate::embedding::create_embedder(device.clone())?;
    let context_features = Arc::new(crate::model::contextual_features::ContextualFeatures::new(
      device.clone(),
      embedder.clone(),
    )?);

    // Create integrator with default config
    let integrator =
      FeatureIntegrator::new(embedder.clone(), context_features.clone(), device.clone())?;

    // Get a sample invocation
    let invocation = create_test_invocation();

    // Get integrated features
    let features = integrator.integrate_features(&invocation)?;

    // Verify we got some features
    assert!(
      !features.is_empty(),
      "Integrated features should not be empty"
    );

    // Get the individual components to compare
    let cmd_emb = embedder.embed(&invocation.command)?;
    let ctx_feat = context_features.encode_all_features(&invocation)?;

    // Check dimensions (should be sum when concatenated)
    assert_eq!(
      features.len(),
      cmd_emb.len() + ctx_feat.len(),
      "Concatenated features should have combined length"
    );

    Ok(())
  }

  #[test]
  fn test_feature_integration_with_normalization() -> Result<()> {
    let device = Device::Cpu;
    let embedder = crate::embedding::create_embedder(device.clone())?;
    let context_features = Arc::new(crate::model::contextual_features::ContextualFeatures::new(
      device.clone(),
      embedder.clone(),
    )?);

    // Create integrator with normalization enabled
    let config_norm = FeatureIntegrationConfig {
      normalize_features: true,
      ..Default::default()
    };
    let integrator_norm = FeatureIntegrator::with_config(
      embedder.clone(),
      context_features.clone(),
      config_norm,
      device.clone(),
    )?;

    // Create integrator with normalization disabled
    let config_no_norm = FeatureIntegrationConfig {
      normalize_features: false,
      ..Default::default()
    };
    let integrator_no_norm = FeatureIntegrator::with_config(
      embedder.clone(),
      context_features.clone(),
      config_no_norm,
      device.clone(),
    )?;

    // Get a sample invocation
    let invocation = create_test_invocation();

    // Get features with and without normalization
    let features_norm = integrator_norm.integrate_features(&invocation)?;
    let features_no_norm = integrator_no_norm.integrate_features(&invocation)?;

    // Verify we got some features in both cases
    assert!(
      !features_norm.is_empty(),
      "Normalized features should not be empty"
    );
    assert!(
      !features_no_norm.is_empty(),
      "Non-normalized features should not be empty"
    );

    // Verify dimensions are the same
    assert_eq!(
      features_norm.len(),
      features_no_norm.len(),
      "Feature dimensions should match regardless of normalization"
    );

    // Check that normalized features have unit length (approximately)
    let norm_squared: f32 = features_norm.iter().map(|&x| x * x).sum();
    assert!(
      (norm_squared - 1.0).abs() < 1e-5,
      "Normalized features should have unit length"
    );

    // Check that non-normalized features don't have unit length
    // (unless by coincidence, which is extremely unlikely)
    let non_norm_squared: f32 = features_no_norm.iter().map(|&x| x * x).sum();
    // Check that non-normalized features don't have unit length
    // (unless by coincidence, which is extremely unlikely)
    assert!(
      (non_norm_squared - 1.0).abs() > 1e-5 || non_norm_squared < 1e-10,
      "Non-normalized features should not have unit length (unless they're all zeros)"
    );

    Ok(())
  }

  #[test]
  fn test_tensor_creation() -> Result<()> {
    let device = Device::Cpu;

    // Initialize embedder and context features
    let embedder = crate::embedding::create_embedder(device.clone())?;
    let context_features = Arc::new(crate::model::contextual_features::ContextualFeatures::new(
      device.clone(),
      embedder.clone(),
    )?);

    // Create integrator
    let integrator =
      FeatureIntegrator::new(embedder.clone(), context_features.clone(), device.clone())?;

    // Get test invocations
    let invocations = create_test_invocations();
    let num_invocations = invocations.len();

    // Create tensor
    let tensor = integrator.create_integrated_tensor(&invocations)?;

    // Check tensor shape
    let shape = tensor.shape();
    assert_eq!(shape.dims().len(), 2, "Tensor should be 2D");

    // Convert dimensions for comparison
    let dim0 = shape.dims()[0] as usize;
    let dim1 = shape.dims()[1] as usize;

    assert_eq!(
      dim0, num_invocations,
      "First dimension should match number of invocations"
    );

    // Check feature dimension
    let feature_dim = integrator.get_feature_dimension(&invocations[0])?;
    assert_eq!(
      dim1, feature_dim,
      "Second dimension should match feature dimension"
    );

    Ok(())
  }

  #[test]
  fn test_empty_invocations() -> Result<()> {
    let device = Device::Cpu;

    // Initialize embedder and context features
    let embedder = crate::embedding::create_embedder(device.clone())?;
    let context_features = Arc::new(crate::model::contextual_features::ContextualFeatures::new(
      device.clone(),
      embedder.clone(),
    )?);

    // Create integrator
    let integrator =
      FeatureIntegrator::new(embedder.clone(), context_features.clone(), device.clone())?;

    // Create tensor from empty invocations
    let tensor = integrator.create_integrated_tensor(&[])?;

    // Check tensor shape
    let shape = tensor.shape();
    assert_eq!(shape.dims().len(), 2, "Tensor should be 2D");
    assert_eq!(
      shape.dims()[0],
      0,
      "First dimension should be 0 for empty input"
    );

    Ok(())
  }

  #[test]
  fn test_normalize_vector() {
    // Test with a simple vector
    let v = vec![3.0, 4.0];
    let normalized = FeatureIntegrator::normalize_vector(&v);

    // Should have length 1.0 (3-4-5 triangle)
    let norm: f32 = normalized.iter().map(|x| x * x).sum();
    assert!(
      (norm - 1.0).abs() < 1e-6,
      "Normalized vector should have unit length"
    );

    // Components should be 3/5 and 4/5
    assert!(
      (normalized[0] - 0.6).abs() < 1e-6,
      "First component should be 3/5"
    );
    assert!(
      (normalized[1] - 0.8).abs() < 1e-6,
      "Second component should be 4/5"
    );

    // Test with zero vector
    let zero_vec = vec![0.0, 0.0, 0.0];
    let normalized_zero = FeatureIntegrator::normalize_vector(&zero_vec);

    // Should return the original vector
    assert_eq!(
      normalized_zero, zero_vec,
      "Zero vector should remain unchanged"
    );
  }

  #[test]
  fn test_consistency_across_invocations() -> Result<()> {
    let device = Device::Cpu;

    // Initialize embedder and context features
    let embedder = create_embedder(device.clone())?;
    let context_features = Arc::new(ContextualFeatures::new(device.clone(), embedder.clone())?);

    // Create integrator
    let integrator =
      FeatureIntegrator::new(embedder.clone(), context_features.clone(), device.clone())?;

    // Get test invocations
    let invocations = create_test_invocations();

    // Process each invocation individually
    let mut individual_features = Vec::new();
    for inv in &invocations {
      individual_features.push(integrator.integrate_features(inv)?);
    }

    // Process as a batch
    let tensor = integrator.create_integrated_tensor(&invocations)?;

    // Extract features from tensor for comparison
    let tensor_vec = tensor.to_vec2::<f32>()?;

    // Compare dimensions
    assert_eq!(
      tensor_vec.len(),
      individual_features.len(),
      "Batch and individual processing should yield same number of examples"
    );

    // Compare each feature vector
    for (i, features) in individual_features.iter().enumerate() {
      assert_eq!(
        tensor_vec[i].len(),
        features.len(),
        "Feature dimensions should match"
      );

      // Compare values (allowing for small floating point differences)
      for (j, val) in features.iter().enumerate() {
        assert!(
          (tensor_vec[i][j] - val).abs() < 1e-5,
          "Feature values should match between batch and individual processing"
        );
      }
    }

    Ok(())
  }
}
