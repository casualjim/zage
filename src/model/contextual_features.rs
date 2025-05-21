use crate::Result;
// Using dependency injection for embedder
use crate::embedding::Embedder;
use crate::shell_history::Invocation;
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Tracks document frequency statistics for path components
/// to identify common components that can be replaced with placeholders
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathComponentStats {
  /// Total number of paths processed
  total_paths: usize,

  /// Document frequency for each path component
  /// Maps component -> number of paths containing this component
  component_df: HashMap<String, usize>,

  /// Threshold for high document frequency
  /// Components with DF above this threshold are considered common
  /// and can be replaced with placeholders
  df_threshold: f32,
}

impl PathComponentStats {
  /// Get the total number of paths processed
  pub fn total_paths(&self) -> usize {
    self.total_paths
  }

  /// Set the total number of paths
  pub fn set_total_paths(&mut self, total: usize) {
    self.total_paths = total;
  }

  /// Get a reference to the component frequency map
  pub fn component_frequencies(&self) -> &HashMap<String, usize> {
    &self.component_df
  }

  /// Set the frequency for a specific component
  pub fn set_component_frequency(&mut self, component: &str, frequency: usize) {
    self.component_df.insert(component.to_string(), frequency);
  }

  /// Create a new PathComponentStats with default threshold
  pub fn new() -> Self {
    Self {
      total_paths: 0,
      component_df: HashMap::new(),
      df_threshold: 0.5, // Default: components in >50% of paths are considered common
    }
  }

  /// Create a new PathComponentStats with custom threshold
  pub fn with_threshold(df_threshold: f32) -> Self {
    Self {
      total_paths: 0,
      component_df: HashMap::new(),
      df_threshold,
    }
  }

  /// Update statistics with a new path
  pub fn update(&mut self, path: &str) {
    // Increment total paths count
    self.total_paths += 1;

    // Extract unique components from the path
    let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

    // Update document frequency for each unique component
    let unique_components: std::collections::HashSet<&str> = components.iter().copied().collect();
    for component in unique_components {
      *self.component_df.entry(component.to_string()).or_insert(0) += 1;
    }
  }

  /// Check if a component has high document frequency
  pub fn is_high_df(&self, component: &str) -> bool {
    if self.total_paths == 0 {
      return false;
    }

    if let Some(count) = self.component_df.get(component) {
      let df = *count as f32 / self.total_paths as f32;
      df >= self.df_threshold
    } else {
      false
    }
  }

  /// Get the document frequency of a component
  pub fn get_df(&self, component: &str) -> f32 {
    if self.total_paths == 0 {
      return 0.0;
    }

    if let Some(count) = self.component_df.get(component) {
      *count as f32 / self.total_paths as f32
    } else {
      0.0
    }
  }

  /// Get all components with document frequency above the threshold
  pub fn get_high_df_components(&self) -> Vec<(String, f32)> {
    if self.total_paths == 0 {
      return Vec::new();
    }

    self
      .component_df
      .iter()
      .filter_map(|(component, count)| {
        let df = *count as f32 / self.total_paths as f32;
        if df >= self.df_threshold {
          Some((component.clone(), df))
        } else {
          None
        }
      })
      .collect()
  }
}

/// ContextualFeatures encodes various context information into tensors
/// for use in the neural prediction model.
pub struct ContextualFeatures {
  embedder: Arc<dyn Embedder>,
  device: Device,
  path_stats: PathComponentStats,
}

impl ContextualFeatures {
  /// Create a new ContextualFeatures processor with default path stats and provided embedder
  pub fn new(device: Device, embedder: Arc<dyn Embedder>) -> Result<Self> {
    Self::with_path_stats_and_embedder(device, PathComponentStats::new(), embedder)
  }

  /// Create a new ContextualFeatures processor with custom path stats and provided embedder
  pub fn with_path_stats_and_embedder(
    device: Device,
    path_stats: PathComponentStats,
    embedder: Arc<dyn Embedder>,
  ) -> Result<Self> {
    Ok(Self {
      embedder,
      device,
      path_stats,
    })
  }

  /// Update path statistics with a new path
  pub fn update_path_stats(&mut self, path: &str) {
    self.path_stats.update(path);
  }

  /// Maximum length for each path component before truncation
  const MAX_COMPONENT_LENGTH: usize = 25;

  /// Maximum number of path components to keep (from the end)
  const MAX_COMPONENTS: usize = 10;

  /// Number of root components to preserve for deep paths
  const PRESERVE_ROOT_COMPONENTS: usize = 2;

  /// Number of end components to preserve for deep paths
  const PRESERVE_END_COMPONENTS: usize = 3;

  /// Placeholder for omitted path segments
  const PATH_OMISSION_PLACEHOLDER: &'static str = "....";

  /// Common project indicators that should be preserved in paths
  const PROJECT_INDICATORS: [&'static str; 7] = [
    "github.com",
    "gitlab.com",
    "bitbucket.org",
    "src",
    "source",
    "projects",
    "workspace",
  ];

  /// Extract and encode working directory into embeddings
  pub fn encode_working_directory(&self, cwd: &str) -> Result<Vec<f32>> {
    // Normalize the path before embedding
    let normalized_path = self.normalize_path(cwd);
    self.embedder.embed(&normalized_path)
  }

  /// Update path statistics with a batch of working directories
  /// This is used during history import to build document frequency statistics
  pub fn update_path_stats_batch(&mut self, paths: &[&str]) {
    for path in paths {
      self.path_stats.update(path);
    }
  }

  /// Get the document frequency of a path component
  pub fn get_component_df(&self, component: &str) -> f32 {
    self.path_stats.get_df(component)
  }

  /// Check if a component has high document frequency
  pub fn is_high_df_component(&self, component: &str) -> bool {
    self.path_stats.is_high_df(component)
  }

  /// Normalize a path for embedding by handling long paths and components
  /// Uses document frequency statistics to identify common components that can be replaced
  fn normalize_path(&self, path: &str) -> String {
    // Split path into components
    let components: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

    // If path is short enough, use it directly
    if components.len() <= Self::MAX_COMPONENTS
      && components
        .iter()
        .all(|c| c.len() <= Self::MAX_COMPONENT_LENGTH)
    {
      return path.to_string();
    }

    // For deep paths, we want to preserve both the root and the most recent components
    let mut normalized_components = Vec::new();

    if components.len() > Self::MAX_COMPONENTS {
      // Try to find a project root indicator
      let root_idx = self.find_project_root(&components);

      if let Some(idx) = root_idx {
        // We found a project indicator, preserve up to that point
        let preserve_until = idx.min(Self::PRESERVE_ROOT_COMPONENTS);
        normalized_components.extend(
          components[..=preserve_until]
            .iter()
            .map(|&c| self.truncate_component(c)),
        );

        // Process middle components using DF statistics
        if preserve_until + 1 < components.len() - Self::PRESERVE_END_COMPONENTS {
          // Check if there are high-DF components in the middle section
          let mut added_placeholder = false;
          for i in (preserve_until + 1)..(components.len() - Self::PRESERVE_END_COMPONENTS) {
            let component = components[i];

            // If it's a low-DF component (more distinctive), keep it
            // Otherwise, replace with placeholder
            if !self.is_high_df_component(component) {
              normalized_components.push(self.truncate_component(component));
            } else if !added_placeholder {
              // Only add the placeholder once for consecutive high-DF components
              normalized_components.push(Self::PATH_OMISSION_PLACEHOLDER.to_string());
              added_placeholder = true;
            }
          }
        }
      } else {
        // No project indicator found, use DF statistics for initial components
        let middle_start = Self::PRESERVE_ROOT_COMPONENTS.min(components.len());
        let middle_end = components
          .len()
          .saturating_sub(Self::PRESERVE_END_COMPONENTS);

        // Always preserve the root components
        normalized_components.extend(
          components[..middle_start]
            .iter()
            .map(|&c| self.truncate_component(c)),
        );

        // Use DF statistics for middle components
        if middle_start < middle_end {
          let mut added_placeholder = false;
          for i in middle_start..middle_end {
            let component = components[i];

            // If it's a low-DF component (more distinctive), keep it
            // Otherwise, replace with placeholder
            if !self.is_high_df_component(component) {
              normalized_components.push(self.truncate_component(component));
            } else if !added_placeholder {
              // Only add the placeholder once for consecutive high-DF components
              normalized_components.push(Self::PATH_OMISSION_PLACEHOLDER.to_string());
              added_placeholder = true;
            }
          }
        }
      }

      // Always preserve the end components
      let end_start = components
        .len()
        .saturating_sub(Self::PRESERVE_END_COMPONENTS);
      normalized_components.extend(
        components[end_start..]
          .iter()
          .map(|&c| self.truncate_component(c)),
      );
    } else {
      // Path is not too deep, but may have long components
      normalized_components = components
        .iter()
        .map(|&component| self.truncate_component(component))
        .collect();
    }

    // Reconstruct the path
    if path.starts_with('/') {
      format!("/{}", normalized_components.join("/"))
    } else {
      normalized_components.join("/")
    }
  }

  /// Truncate a path component while preserving semantic meaning
  fn truncate_component(&self, component: &str) -> String {
    if component.len() <= Self::MAX_COMPONENT_LENGTH {
      return component.to_string();
    }

    // For long components, keep the start and end parts
    let half_len = Self::MAX_COMPONENT_LENGTH / 2;
    let start = &component[..half_len.saturating_sub(1)];
    let end = &component[component.len().saturating_sub(half_len)..];

    format!("{}..{}", start, end)
  }

  /// Find the index of a project indicator in path components
  fn find_project_root(&self, components: &[&str]) -> Option<usize> {
    for (i, component) in components.iter().enumerate() {
      if Self::PROJECT_INDICATORS.contains(component) && i + 1 < components.len() {
        // Found a project indicator, return the index of the component after it
        return Some(i + 1);
      }
    }
    None
  }

  /// Alternative approach: hash long path components
  pub fn hash_path_component(&self, component: &str) -> String {
    if component.len() <= Self::MAX_COMPONENT_LENGTH {
      return component.to_string();
    }

    // Use a simple hash function for demonstration
    // In production, consider using a more robust hash
    let hash_value = component.bytes().fold(0u64, |acc, byte| {
      acc.wrapping_mul(31).wrapping_add(byte as u64)
    });

    // Format as a short hex string and combine with prefix
    let prefix = &component[..4.min(component.len())];
    format!("{}_{:x}", prefix, hash_value % 10000)
  }

  /// Encode exit status as a binary feature (0 for success, 1 for failure)
  pub fn encode_exit_status(&self, exit_status: Option<i64>) -> Result<Vec<f32>> {
    match exit_status {
      Some(0) => Ok(vec![1.0, 0.0]), // Success: [1, 0]
      Some(_) => Ok(vec![0.0, 1.0]), // Failure: [0, 1]
      None => Ok(vec![0.5, 0.5]),    // Unknown: [0.5, 0.5]
    }
  }

  /// Encode host and user information
  pub fn encode_host_user(
    &self,
    hostname: Option<&str>,
    username: Option<&str>,
  ) -> Result<Vec<f32>> {
    let mut features = Vec::new();

    // Encode hostname if available
    if let Some(host) = hostname {
      let host_embedding = self.embedder.embed(host)?;
      features.extend_from_slice(&host_embedding);
    } else {
      // Use a zero vector as placeholder
      features.push(0.0);
    }

    // Encode username if available
    if let Some(user) = username {
      let user_embedding = self.embedder.embed(user)?;
      features.extend_from_slice(&user_embedding);
    } else {
      // Use a zero vector as placeholder
      features.push(0.0);
    }

    Ok(features)
  }

  /// Encode remote/local session info (determines if within SSH)
  pub fn encode_remote_session(&self) -> Result<Vec<f32>> {
    // Check if we're in an SSH session
    let is_ssh = std::env::var("SSH_CLIENT").is_ok() || std::env::var("SSH_TTY").is_ok();

    if is_ssh {
      Ok(vec![1.0]) // Remote session
    } else {
      Ok(vec![0.0]) // Local session
    }
  }

  /// Combine all contextual features for an invocation
  pub fn encode_all_features(&self, invocation: &Invocation) -> Result<Vec<f32>> {
    let mut features = Vec::new();

    // Working directory features
    if let Some(cwd) = &invocation.working_directory {
      let cwd_features = self.encode_working_directory(cwd)?;
      features.extend_from_slice(&cwd_features);
    } else {
      // Use a zero vector as placeholder
      features.push(0.0);
    }

    // Exit status features
    let exit_features = self.encode_exit_status(invocation.exit_status)?;
    features.extend_from_slice(&exit_features);

    // Host and user features
    let host_user_features = self.encode_host_user(
      invocation.hostname.as_deref(),
      invocation.username.as_deref(),
    )?;
    features.extend_from_slice(&host_user_features);

    // Remote session features
    let remote_features = self.encode_remote_session()?;
    features.extend_from_slice(&remote_features);

    Ok(features)
  }

  /// Convert combined features to tensor
  pub fn features_to_tensor(&self, features: &[f32]) -> Result<Tensor> {
    Ok(Tensor::from_vec(
      features.to_vec(),
      &[features.len()],
      &self.device,
    )?)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::embedding::create_embedder;
  use crate::shell_history::Invocation;

  #[test]
  fn test_encode_working_directory() -> Result<()> {
    let device = Device::Cpu;
    let embedder = create_embedder(device.clone())?;
    let features = ContextualFeatures::new(device.clone(), embedder)?;

    // Test with various paths
    let paths = vec![
      "/home/user",
      "/home/user/projects",
      "/home/user/projects/rust-project",
      "/very/long/path/with/many/components/that/should/be/truncated",
    ];

    for path in paths {
      let embedding = features.encode_working_directory(path)?;
      assert!(
        !embedding.is_empty(),
        "Should produce a non-empty embedding for path: {}",
        path
      );
      assert_eq!(
        embedding.len(),
        768,
        "Embedding dimension should be 768 for path: {}",
        path
      );
    }
    Ok(())
  }

  #[test]
  fn test_path_normalization() -> Result<()> {
    let device = Device::Cpu;
    let embedder = create_embedder(device.clone())?;
    let mut features = ContextualFeatures::new(device.clone(), embedder)?;

    // Build up some DF statistics first
    features.update_path_stats(
      "/home/user/very/deep/nested/directory/structure/with/many/components/file1.txt",
    );
    features.update_path_stats(
      "/home/user/very/deep/nested/directory/structure/with/many/components/file2.txt",
    );
    features.update_path_stats(
      "/home/user/very/deep/nested/directory/structure/with/other/components/file3.txt",
    );

    // Test with short path (should remain unchanged if no high-DF components affect it)
    let short_path = "/home/user/projects";
    let normalized_short = features.normalize_path(short_path);
    assert_eq!(
      normalized_short, short_path,
      "Short path normalization failed. Expected: {}, Got: {}",
      short_path, normalized_short
    );

    // Test with very long path
    let long_path = "/home/user/very/deep/nested/directory/structure/with/many/components/file.txt";
    let normalized_long = features.normalize_path(long_path);
    assert_ne!(
      normalized_long, long_path,
      "Long path should be normalized. Original: {}, Normalized: {}",
      long_path, normalized_long
    );
    assert!(
      normalized_long.contains(ContextualFeatures::PATH_OMISSION_PLACEHOLDER),
      "Normalized long path should contain placeholder. Actual: {}",
      normalized_long
    );
    assert!(
      normalized_long.starts_with('/'),
      "Normalized path should preserve leading slash. Actual: {}",
      normalized_long
    );

    // Test with project indicator
    features.update_path_stats("/data/github.com/another/repo/file.txt"); // ensure 'github.com' isn't high-DF from previous
    let project_path = "/home/user/github.com/username/repository/src/main/rust/file.rs";
    let normalized_project = features.normalize_path(project_path);
    assert!(
      normalized_project.contains("github.com"),
      "Project indicator 'github.com' should be preserved. Actual: {}",
      normalized_project
    );

    // Test with long component names
    let long_component_path = "/home/user/thisisareallyverylongcomponentnamethatshouldbeshortened/anotherlongcomponent/file.txt";
    let normalized_component_path = features.normalize_path(long_component_path);
    assert!(
      !normalized_component_path
        .contains("thisisareallyverylongcomponentnamethatshouldbeshortened"),
      "Long components should be shortened. Actual: {}",
      normalized_component_path
    );

    Ok(())
  }

  #[test]
  fn test_truncate_component() -> Result<()> {
    let device = Device::Cpu;
    let embedder = create_embedder(device.clone())?;
    let features = ContextualFeatures::new(device.clone(), embedder)?;

    // Test case 1: Component shorter than MAX_COMPONENT_LENGTH
    let short_component = "short";
    let truncated_short = features.truncate_component(short_component);
    assert_eq!(
      truncated_short, short_component,
      "Short component should remain unchanged"
    );

    // Test case 2: Component exactly at MAX_COMPONENT_LENGTH
    let at_limit_component = "a".repeat(ContextualFeatures::MAX_COMPONENT_LENGTH);
    let truncated_limit = features.truncate_component(&at_limit_component);
    assert_eq!(
      truncated_limit, at_limit_component,
      "Component at limit should remain unchanged"
    );

    // Test case 3: Component longer than MAX_COMPONENT_LENGTH
    let long_component = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let truncated_long = features.truncate_component(long_component);

    // Verify the truncated component is shorter
    assert!(
      truncated_long.len() <= ContextualFeatures::MAX_COMPONENT_LENGTH,
      "Truncated component should be within length limit"
    );

    // Verify it contains ".." to indicate truncation
    assert!(
      truncated_long.contains(".."),
      "Truncated component should contain '..' to indicate truncation"
    );

    // Verify it preserves the start and end
    assert!(
      truncated_long.starts_with('a'),
      "Truncated component should preserve the start"
    );
    assert!(
      truncated_long.ends_with('Z'),
      "Truncated component should preserve the end"
    );

    Ok(())
  }

  #[test]
  fn test_hash_path_component() -> Result<()> {
    let device = Device::Cpu;
    let embedder = create_embedder(device.clone())?;
    let features = ContextualFeatures::new(device.clone(), embedder)?;

    // Test case 1: Component shorter than MAX_COMPONENT_LENGTH
    let short_component = "short_component_name";
    let hashed_short = features.hash_path_component(short_component);
    assert_eq!(
      hashed_short, short_component,
      "Short component should remain unchanged"
    );

    // Test case 2: Component longer than MAX_COMPONENT_LENGTH
    let long_component = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    let hashed_long = features.hash_path_component(long_component);

    // Verify the hashed component is shorter
    assert!(
      hashed_long.len() <= ContextualFeatures::MAX_COMPONENT_LENGTH,
      "Hashed component should be within length limit"
    );

    // Verify it preserves the prefix
    assert!(
      hashed_long.starts_with("abcd"),
      "Hashed component should preserve the prefix"
    );

    // Verify it contains a hash part
    assert!(
      hashed_long.contains('_'),
      "Hashed component should contain a separator"
    );

    // Test consistency
    let hashed_again = features.hash_path_component(long_component);
    assert_eq!(hashed_long, hashed_again, "Hashing should be deterministic");

    Ok(())
  }

  #[test]
  fn test_embedding_consistency_with_normalization() -> Result<()> {
    let device = Device::Cpu;
    let embedder = create_embedder(device.clone())?;
    let mut features = ContextualFeatures::new(device.clone(), embedder)?;

    features.update_path_stats("/home/user/common_project/src/main.rs");
    features.update_path_stats("/home/user/common_project/src/lib.rs");

    // Test that similar paths have similar embeddings
    let path1 = "/home/user/common_project/src/main.rs";
    let path2 = "/home/user/common_project/src/lib.rs";

    let embedding1 = features.encode_working_directory(path1)?;
    let embedding2 = features.encode_working_directory(path2)?;

    // Calculate cosine similarity
    let dot_product: f32 = embedding1
      .iter()
      .zip(embedding2.iter())
      .map(|(a, b)| a * b)
      .sum();

    let norm1: f32 = embedding1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = embedding2.iter().map(|x| x * x).sum::<f32>().sqrt();

    let similarity = dot_product / (norm1 * norm2);

    // Similar paths should have high similarity
    assert!(
      similarity > 0.7,
      "Similar paths should have similar embeddings"
    );

    Ok(())
  }

  #[test]
  fn test_encode_exit_status() -> Result<()> {
    let device = Device::Cpu;
    let embedder = create_embedder(device.clone())?;
    let features = ContextualFeatures::new(device.clone(), embedder)?;

    // Test success status
    let success_status = Some(0);
    let success = features.encode_exit_status(success_status)?;
    assert_eq!(
      success,
      vec![1.0, 0.0],
      "Success should be encoded as [1.0, 0.0]"
    );

    // Test failure status
    let failure_status = Some(1);
    let failure = features.encode_exit_status(failure_status)?;
    assert_eq!(
      failure,
      vec![0.0, 1.0],
      "Failure should be encoded as [0.0, 1.0]"
    );

    // Test unknown status
    let unknown_status = None;
    let unknown = features.encode_exit_status(unknown_status)?;
    assert_eq!(
      unknown,
      vec![0.5, 0.5],
      "Unknown should be encoded as [0.5, 0.5]"
    );

    // Test other non-zero status
    let other_failure_status = Some(127);
    let other_failure = features.encode_exit_status(other_failure_status)?;
    assert_eq!(
      other_failure,
      vec![0.0, 1.0],
      "Any non-zero exit status should be encoded as failure"
    );

    Ok(())
  }

  #[test]
  fn test_encode_host_user() -> Result<()> {
    let device = Device::Cpu;
    let embedder = crate::embedding::create_embedder(device.clone())?;
    let features = ContextualFeatures::new(device.clone(), embedder)?;

    // Test with hostname and username
    let hostname = Some("localhost");
    let username = Some("testuser");
    let both = features.encode_host_user(hostname, username)?;
    assert!(!both.is_empty(), "Should produce non-empty features");

    // Test with only hostname
    let host_only = features.encode_host_user(Some("localhost"), None)?;
    assert!(
      !host_only.is_empty(),
      "Should produce features with hostname only"
    );

    // Test with only username
    let user_only = features.encode_host_user(None, Some("testuser"))?;
    assert!(
      !user_only.is_empty(),
      "Should produce features with username only"
    );

    // Test with neither
    let neither = features.encode_host_user(None, None)?;
    assert!(
      !neither.is_empty(),
      "Should produce features even with no input"
    );

    // Verify consistency - use the same input for both calls
    let both_again = features.encode_host_user(hostname, username)?;
    assert_eq!(both, both_again, "Encoding should be deterministic");

    Ok(())
  }

  #[test]
  fn test_encode_all_features() -> Result<()> {
    let device = Device::Cpu;
    let embedder = crate::embedding::create_embedder(device.clone())?;
    let features = ContextualFeatures::new(device.clone(), embedder)?;

    let invocation = Invocation {
      start_unix_timestamp: Some(1678886400),
      command: "ls -la".to_string(),
      working_directory: Some("/home/user/projects".to_string()),
      exit_status: Some(0),
      hostname: Some("localhost".to_string()),
      username: Some("testuser".to_string()),
      shellname: "bash".to_string(),
      session_id: 1,
      end_unix_timestamp: None,
    };

    // Encode all features
    let all_features = features.encode_all_features(&invocation)?;

    // Verify we get a non-empty result
    assert!(
      !all_features.is_empty(),
      "Should produce non-empty features"
    );

    // Verify we can convert to tensor
    let tensor = features.features_to_tensor(&all_features)?;
    assert_eq!(
      tensor.dims()[0] as usize,
      all_features.len(),
      "Tensor dimension should match feature count"
    );

    // Test with minimal invocation
    let minimal = Invocation {
      start_unix_timestamp: Some(1678886400),
      command: "pwd".to_string(),
      shellname: "bash".to_string(),
      session_id: 1,
      ..Default::default()
    };

    let minimal_features = features.encode_all_features(&minimal)?;
    assert!(
      !minimal_features.is_empty(),
      "Should handle minimal invocation"
    );

    Ok(())
  }

  #[test]
  fn test_path_stats_and_normalization() -> Result<()> {
    let device = Device::Cpu;
    let embedder = create_embedder(device.clone())?;
    let mut features = ContextualFeatures::new(device.clone(), embedder)?;

    // Add some paths to build statistics
    features.update_path_stats("/home/user/projects/rust-project/src/main.rs");
    features.update_path_stats("/home/user/projects/rust-project/src/lib.rs");
    features.update_path_stats("/home/user/projects/python-project/src/main.py");
    features.update_path_stats("/home/user/projects/go-project/src/main.go");
    features.update_path_stats("/home/user/documents/notes.txt");

    // Test DF values
    assert!(
      features.get_component_df("src") > 0.5,
      "src should have high DF"
    );
    assert!(
      features.get_component_df("projects") > 0.5,
      "projects should have high DF"
    );
    assert!(
      features.get_component_df("rust-project") < 0.5,
      "rust-project should have low DF"
    );

    // Test normalization with DF stats
    let long_path =
      "/home/user/projects/rust-project/src/module/submodule/deep/nested/very/long/path/file.rs";
    let normalized = features.normalize_path(long_path);

    // Print paths for debugging
    println!("Original path: {}", long_path);
    println!("Normalized path: {}", normalized);
    println!("Original length: {}", long_path.len());
    println!("Normalized length: {}", normalized.len());

    // Print DF stats for components
    println!("DF for 'src': {}", features.get_component_df("src"));
    println!("DF for 'module': {}", features.get_component_df("module"));
    println!(
      "DF for 'submodule': {}",
      features.get_component_df("submodule")
    );
    println!(
      "DF for 'rust-project': {}",
      features.get_component_df("rust-project")
    );

    // Verify that high-DF components like "src" might be replaced with placeholders
    // while distinctive components like "rust-project" are preserved
    assert!(
      normalized.contains("rust-project"),
      "Distinctive components should be preserved"
    );

    // Verify that high-DF components are replaced with placeholders
    assert!(
      normalized.contains(ContextualFeatures::PATH_OMISSION_PLACEHOLDER),
      "High-DF components should be replaced with placeholders"
    );

    // Verify the normalized path is different from the original
    assert!(
      normalized != long_path,
      "Normalized path should be different from the original"
    );

    Ok(())
  }
}
