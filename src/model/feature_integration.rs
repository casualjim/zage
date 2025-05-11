//! # Feature Integration for Neural Models
//!
//! This module handles the integration of command embeddings and contextual features
//! into a unified representation for neural network input. 
//!
//! The main functions are:
//! 1. Concatenate command embeddings with contextual features
//! 2. Optionally normalize the combined features
//! 3. Format data into tensors suitable for the neural model

use candle_core::{Device, Result as CandleResult, Tensor};
use std::sync::Arc;

use crate::Result;
use crate::model::contextual_features::ContextualFeatures;
use crate::model::pretrained_embedder::PretrainedEmbedder;
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
    embedder: Arc<PretrainedEmbedder>,
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
        embedder: Arc<PretrainedEmbedder>,
        context_features: Arc<ContextualFeatures>,
        device: Device,
    ) -> Result<Self> {
        Self::with_config(embedder, context_features, FeatureIntegrationConfig::default(), device)
    }

    /// Create a new feature integrator with custom configuration
    pub fn with_config(
        embedder: Arc<PretrainedEmbedder>,
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

        // Concatenate the features
        let mut combined = Vec::with_capacity(cmd_embedding.len() + ctx_features.len());
        combined.extend_from_slice(&cmd_embedding);
        combined.extend_from_slice(&ctx_features);
        
        if self.config.normalize_features {
            Ok(Self::normalize_vector(&combined))
        } else {
            Ok(combined)
        }
    }

    /// Create integrated feature tensor for a sequence of invocations
    pub fn create_integrated_tensor(&self, invocations: &[Invocation]) -> Result<Tensor> {
        if invocations.is_empty() {
            return Err(crate::ZageError::ConfigError(
                "Cannot create tensor from empty invocations".to_string()
            ));
        }

        // Get integrated features for each invocation
        let mut features_list = Vec::with_capacity(invocations.len());
        for inv in invocations {
            let features = self.integrate_features(inv)?;
            features_list.push(features);
        }

        // Determine feature dimension from the first integrated feature
        let feature_dim = features_list[0].len();

        // Create a vector to hold all flattened features
        let mut all_features = Vec::with_capacity(invocations.len() * feature_dim);
        for features in &features_list {
            all_features.extend_from_slice(features);
        }

        // Create tensor with shape [batch_size, feature_dim]
        let tensor = Tensor::from_vec(
            all_features,
            (invocations.len(), feature_dim),
            &self.device,
        ).map_err(|e| crate::ZageError::CandleError(e))?;

        Ok(tensor)
    }

    /// Normalize a vector to unit length (L2 norm)
    fn normalize_vector(v: &[f32]) -> Vec<f32> {
        let mut norm = 0.0;
        for &x in v {
            norm += x * x;
        }
        norm = norm.sqrt();

        if norm == 0.0 {
            return v.to_vec();
        }

        let mut normalized = Vec::with_capacity(v.len());
        for &x in v {
            normalized.push(x / norm);
        }
        normalized
    }
    
    /// Get the dimension of the integrated features
    pub fn get_feature_dimension(&self, invocation: &Invocation) -> Result<usize> {
        let features = self.integrate_features(invocation)?;
        Ok(features.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // Helper function to create a test invocation
    fn create_test_invocation() -> Invocation {
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
        }
    }
    
    // Helper function to create multiple test invocations
    fn create_test_invocations() -> Vec<Invocation> {
        vec![
            create_test_invocation(),
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
        ]
    }

    #[test]
    fn test_feature_integration_basic() -> Result<()> {
        let device = Device::Cpu;
        
        // Initialize embedder and context features
        let embedder = Arc::new(PretrainedEmbedder::new(device.clone())?);
        let context_features = Arc::new(ContextualFeatures::new(device.clone())?);
        
        // Create integrator with default config
        let integrator = FeatureIntegrator::new(
            embedder.clone(),
            context_features.clone(),
            device.clone(),
        )?;
        
        // Get a sample invocation
        let invocation = create_test_invocation();
        
        // Get integrated features
        let features = integrator.integrate_features(&invocation)?;
        
        // Verify we got some features
        assert!(!features.is_empty(), "Integrated features should not be empty");
        
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
        
        // Initialize embedder and context features
        let embedder = Arc::new(PretrainedEmbedder::new(device.clone())?);
        let context_features = Arc::new(ContextualFeatures::new(device.clone())?);
        
        // Create integrator with normalization enabled (default)
        let integrator_with_norm = FeatureIntegrator::new(
            embedder.clone(),
            context_features.clone(),
            device.clone(),
        )?;
        
        // Create integrator with normalization disabled
        let config_no_norm = FeatureIntegrationConfig {
            normalize_features: false,
        };
        let integrator_no_norm = FeatureIntegrator::with_config(
            embedder,
            context_features,
            config_no_norm,
            device,
        )?;
        
        // Get a sample invocation
        let invocation = create_test_invocation();
        
        // Get features with and without normalization
        let features_norm = integrator_with_norm.integrate_features(&invocation)?;
        let features_no_norm = integrator_no_norm.integrate_features(&invocation)?;
        
        // Both should have same dimension
        assert_eq!(
            features_norm.len(),
            features_no_norm.len(),
            "Features should have same dimension regardless of normalization"
        );
        
        // Normalized features should have unit length (approximately)
        let mut norm_sum = 0.0;
        for &x in &features_norm {
            norm_sum += x * x;
        }
        let length = norm_sum.sqrt();
        assert!(
            (length - 1.0).abs() < 1e-5,
            "Normalized features should have unit length"
        );
        
        // Non-normalized features should have different length
        let mut non_norm_sum = 0.0;
        for &x in &features_no_norm {
            non_norm_sum += x * x;
        }
        let non_norm_length = non_norm_sum.sqrt();
        
        // They should be different unless the original vector happened to have unit length
        if non_norm_length != 0.0 {  // Avoid comparing if zero vector
            assert!(
                (non_norm_length - 1.0).abs() > 1e-5 || non_norm_length == 0.0,
                "Non-normalized features should not have unit length unless original did"
            );
        }
        
        Ok(())
    }
    
    #[test]
    fn test_tensor_creation() -> Result<()> {
        let device = Device::Cpu;
        
        // Initialize embedder, context features, and integrator
        let embedder = Arc::new(PretrainedEmbedder::new(device.clone())?);
        let context_features = Arc::new(ContextualFeatures::new(device.clone())?);
        let integrator = FeatureIntegrator::new(
            embedder,
            context_features,
            device,
        )?;
        
        // Get test invocations
        let invocations = create_test_invocations();
        
        // Create integrated tensor
        let tensor = integrator.create_integrated_tensor(&invocations)?;
        
        // Get feature dimension for a single invocation
        let feature_dim = integrator.get_feature_dimension(&invocations[0])?;
        
        // Verify tensor dimensions
        assert_eq!(
            tensor.dims(),
            &[invocations.len(), feature_dim],
            "Tensor should have shape [batch_size, feature_dim]"
        );
        
        Ok(())
    }
    
    #[test]
    fn test_empty_invocations() -> Result<()> {
        let device = Device::Cpu;
        
        // Initialize embedder, context features, and integrator
        let embedder = Arc::new(PretrainedEmbedder::new(device.clone())?);
        let context_features = Arc::new(ContextualFeatures::new(device.clone())?);
        let integrator = FeatureIntegrator::new(
            embedder,
            context_features,
            device,
        )?;
        
        // Try to create tensor from empty invocations
        let result = integrator.create_integrated_tensor(&[]);
        
        // Should return an error
        assert!(result.is_err(), "Should error on empty invocations");
        
        Ok(())
    }

    #[test]
    fn test_normalize_vector() {
        // Test with a simple vector
        let v = vec![3.0, 4.0];
        let normalized = FeatureIntegrator::normalize_vector(&v);
        
        // Length should be 5, so normalized vector should be [3/5, 4/5]
        let expected = vec![0.6, 0.8];
        
        assert_eq!(normalized.len(), expected.len());
        for i in 0..normalized.len() {
            assert!((normalized[i] - expected[i]).abs() < 1e-6);
        }
        
        // Test with zero vector
        let zero_v = vec![0.0, 0.0];
        let normalized_zero = FeatureIntegrator::normalize_vector(&zero_v);
        
        // Should return the original vector
        assert_eq!(normalized_zero, zero_v);
    }
    
    #[test]
    fn test_consistency_across_invocations() -> Result<()> {
        let device = Device::Cpu;
        
        // Initialize embedder, context features, and integrator
        let embedder = Arc::new(PretrainedEmbedder::new(device.clone())?);
        let context_features = Arc::new(ContextualFeatures::new(device.clone())?);
        let integrator = FeatureIntegrator::new(
            embedder,
            context_features,
            device,
        )?;
        
        // Get test invocations
        let invocations = create_test_invocations();
        
        // Get feature dimension for each invocation
        let dims: Vec<usize> = invocations
            .iter()
            .map(|inv| integrator.get_feature_dimension(inv))
            .collect::<Result<Vec<usize>>>()?;
        
        // All dimensions should be the same
        for i in 1..dims.len() {
            assert_eq!(
                dims[0], dims[i],
                "Feature dimensions should be consistent across invocations"
            );
        }
        
        Ok(())
    }
}