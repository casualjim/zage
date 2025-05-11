use crate::Result;
use crate::model::context::Context;
use crate::model::pretrained_embedder::PretrainedEmbedder;
use crate::shell_history::Invocation;
use candle_core::{Device, Tensor};
use std::f32::consts::PI;
use std::time::{SystemTime, UNIX_EPOCH};

/// ContextualFeatures encodes various context information into tensors
/// for use in the neural prediction model.
pub struct ContextualFeatures {
  embedder: PretrainedEmbedder,
  device: Device,
}

impl ContextualFeatures {
  /// Create a new ContextualFeatures processor with the given embedder
  pub fn new(device: Device) -> Result<Self> {
    let embedder = PretrainedEmbedder::new(device.clone())?;
    Ok(Self { embedder, device })
  }

  /// Extract and encode working directory into embeddings
  pub fn encode_working_directory(&self, cwd: &str) -> Result<Vec<f32>> {
    // Path components are meaningful, so encode the full path
    self.embedder.embed(cwd)
  }

  /// Encode exit status as a binary feature (0 for success, 1 for failure)
  pub fn encode_exit_status(&self, exit_status: Option<i64>) -> Result<Vec<f32>> {
    match exit_status {
      Some(0) => Ok(vec![1.0, 0.0]), // Success: [1, 0]
      Some(_) => Ok(vec![0.0, 1.0]), // Failure: [0, 1]
      None => Ok(vec![0.5, 0.5]),    // Unknown: [0.5, 0.5]
    }
  }

  /// Encode time features using cyclic encoding (sine/cosine)
  pub fn encode_time_features(&self, timestamp: Option<i64>) -> Result<Vec<f32>> {
    let now = match timestamp {
      Some(ts) => ts,
      None => SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64,
    };

    // Convert timestamp to various time units
    let secs_in_day = 24 * 60 * 60;
    // let secs_in_week = 7 * secs_in_day;

    // Extract time components
    let second_of_day = (now % secs_in_day) as f32;
    let day_of_week = ((now / secs_in_day) % 7) as f32;

    // Normalize to [0, 2π]
    let hour_angle = 2.0 * PI * second_of_day / (secs_in_day as f32);
    let day_angle = 2.0 * PI * day_of_week / 7.0;

    // Encode using sine and cosine for cyclical representation
    let hour_sin = hour_angle.sin();
    let hour_cos = hour_angle.cos();
    let day_sin = day_angle.sin();
    let day_cos = day_angle.cos();

    Ok(vec![hour_sin, hour_cos, day_sin, day_cos])
  }

  /// Encode host and user information
  pub fn encode_host_user(
    &self,
    hostname: Option<&str>,
    username: Option<&str>,
  ) -> Result<Vec<f32>> {
    // Combine hostname and username for embedding
    let host_user = match (hostname, username) {
      (Some(host), Some(user)) => format!("{} {}", host, user),
      (Some(host), None) => host.to_string(),
      (None, Some(user)) => user.to_string(),
      (None, None) => "unknown".to_string(),
    };

    self.embedder.embed(&host_user)
  }

  /// Encode remote/local session info (determines if within SSH)
  pub fn encode_remote_session(&self) -> Result<Vec<f32>> {
    // Check if SSH_CONNECTION or SSH_CLIENT env vars are set
    let is_remote = std::env::var("SSH_CONNECTION").is_ok() || std::env::var("SSH_CLIENT").is_ok();

    if is_remote {
      Ok(vec![0.0, 1.0]) // Remote: [0, 1]
    } else {
      Ok(vec![1.0, 0.0]) // Local: [1, 0]
    }
  }

  /// Combine all contextual features for an invocation
  pub fn encode_all_features(&self, invocation: &Invocation) -> Result<Vec<f32>> {
    // Extract context from invocation
    let context = Context::from_invocation(invocation);

    // Encode each feature type
    let cwd_features = self.encode_working_directory(&context.cwd)?;
    let exit_features = self.encode_exit_status(context.exit_status)?;
    let time_features = self.encode_time_features(invocation.end_unix_timestamp)?;
    let host_user_features =
      self.encode_host_user(context.hostname.as_deref(), context.username.as_deref())?;
    let remote_features = self.encode_remote_session()?;

    // Combine all features into a single vector
    let mut combined_features = Vec::new();
    combined_features.extend_from_slice(&cwd_features);
    combined_features.extend_from_slice(&exit_features);
    combined_features.extend_from_slice(&time_features);
    combined_features.extend_from_slice(&host_user_features);
    combined_features.extend_from_slice(&remote_features);

    Ok(combined_features)
  }

  /// Convert combined features to tensor
  pub fn features_to_tensor(&self, features: &[f32]) -> Result<Tensor> {
    Tensor::new(features, &self.device).map_err(|e| e.into())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use candle_core::Device;
  use std::time::{SystemTime, UNIX_EPOCH};

  #[test]
  fn test_encode_working_directory() -> Result<()> {
    let features = ContextualFeatures::new(Device::Cpu)?;

    // Test various paths
    let home_emb = features.encode_working_directory("/home/user")?;
    let root_emb = features.encode_working_directory("/")?;

    // Embeddings should have consistent length and be non-empty
    assert!(!home_emb.is_empty());
    assert!(!root_emb.is_empty());
    assert_eq!(home_emb.len(), root_emb.len());

    // Different paths should produce different embeddings
    assert_ne!(home_emb, root_emb);

    Ok(())
  }

  #[test]
  fn test_encode_exit_status() -> Result<()> {
    let features = ContextualFeatures::new(Device::Cpu)?;

    // Test success (0), failure (non-zero), and None
    let success = features.encode_exit_status(Some(0))?;
    let failure = features.encode_exit_status(Some(1))?;
    let unknown = features.encode_exit_status(None)?;

    // All should be 2-dimensional
    assert_eq!(success.len(), 2);
    assert_eq!(failure.len(), 2);
    assert_eq!(unknown.len(), 2);

    // Success should be [1.0, 0.0]
    assert_eq!(success, vec![1.0, 0.0]);

    // Failure should be [0.0, 1.0]
    assert_eq!(failure, vec![0.0, 1.0]);

    // Unknown should be [0.5, 0.5]
    assert_eq!(unknown, vec![0.5, 0.5]);

    Ok(())
  }

  #[test]
  fn test_encode_time_features() -> Result<()> {
    let features = ContextualFeatures::new(Device::Cpu)?;

    // Test with specific timestamp (midnight on Monday)
    let monday_midnight = 1620604800; // 2021-05-10 00:00:00 UTC (a Monday)
    let time_feats = features.encode_time_features(Some(monday_midnight))?;

    // Should have 4 features: hour_sin, hour_cos, day_sin, day_cos
    assert_eq!(time_feats.len(), 4);

    // At midnight, hour_sin should be 0.0 and hour_cos should be 1.0 (or very close)
    assert!(time_feats[0].abs() < 1e-6);
    assert!((time_feats[1] - 1.0).abs() < 1e-6);

    // Test current time (for coverage)
    let now_feats = features.encode_time_features(None)?;
    assert_eq!(now_feats.len(), 4);

    // All values should be between -1.0 and 1.0
    for &val in &now_feats {
      assert!(val >= -1.0 && val <= 1.0);
    }

    Ok(())
  }

  #[test]
  fn test_encode_host_user() -> Result<()> {
    let features = ContextualFeatures::new(Device::Cpu)?;

    // Test with host and user
    let both = features.encode_host_user(Some("localhost"), Some("testuser"))?;

    // Test with only host
    let host_only = features.encode_host_user(Some("localhost"), None)?;

    // Test with only user
    let user_only = features.encode_host_user(None, Some("testuser"))?;

    // Test with neither
    let neither = features.encode_host_user(None, None)?;

    // All should have same dimension and be non-empty
    assert!(!both.is_empty());
    assert!(!host_only.is_empty());
    assert!(!user_only.is_empty());
    assert!(!neither.is_empty());

    // Different inputs should produce different embeddings
    assert_ne!(both, host_only);
    assert_ne!(both, user_only);
    assert_ne!(host_only, user_only);

    Ok(())
  }

  #[test]
  fn test_encode_all_features() -> Result<()> {
    let features = ContextualFeatures::new(Device::Cpu)?;

    // Create a test invocation
    let now = SystemTime::now();
    let now_unix = now.duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;

    let invocation = Invocation {
      command: "ls -l".to_string(),
      shellname: "bash".to_string(),
      working_directory: Some("/home/user".to_string()),
      hostname: Some("testhost".to_string()),
      username: Some("testuser".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(now_unix - 5),
      end_unix_timestamp: Some(now_unix),
      session_id: 1,
    };

    // Get all features
    let all_features = features.encode_all_features(&invocation)?;

    // Should have substantial number of features
    assert!(all_features.len() > 10);

    // Convert to tensor
    let tensor = features.features_to_tensor(&all_features)?;

    // Tensor should have correct shape (1D with same length as features)
    assert_eq!(tensor.dims(), &[all_features.len()]);

    Ok(())
  }
}
