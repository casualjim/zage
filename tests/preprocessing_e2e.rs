//! End-to-end tests for the neural network preprocessing pipeline
//!
//! This file contains comprehensive tests for the complete preprocessing
//! pipeline, validating that all components work correctly together from
//! raw shell commands to tensor data ready for neural network training.

use std::sync::Arc;
use std::time::Instant;

use candle_core::Device;
use zage::Result;
use zage::model::contextual_features::ContextualFeatures;
use zage::model::create_embedder;
use zage::model::feature_integration::{FeatureIntegrationConfig, FeatureIntegrator};
use zage::model::training_dataset::TrainingDataset;
use zage::shell_history::Invocation;

// Helper function to create diverse test invocations
fn create_diverse_test_invocations() -> Vec<Invocation> {
  vec![
    // Git workflow in project directory
    Invocation {
      command: "git status".to_string(),
      shellname: "bash".to_string(),
      working_directory: Some("/home/user/project".to_string()),
      hostname: Some("laptop".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(1620000000),
      end_unix_timestamp: Some(1620000005),
      session_id: 1,
    },
    Invocation {
      command: "git add .".to_string(),
      shellname: "bash".to_string(),
      working_directory: Some("/home/user/project".to_string()),
      hostname: Some("laptop".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(1620000010),
      end_unix_timestamp: Some(1620000015),
      session_id: 1,
    },
    Invocation {
      command: "git commit -m 'update readme'".to_string(),
      shellname: "bash".to_string(),
      working_directory: Some("/home/user/project".to_string()),
      hostname: Some("laptop".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(1620000020),
      end_unix_timestamp: Some(1620000025),
      session_id: 1,
    },
    Invocation {
      command: "git push origin main".to_string(),
      shellname: "bash".to_string(),
      working_directory: Some("/home/user/project".to_string()),
      hostname: Some("laptop".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(1620000030),
      end_unix_timestamp: Some(1620000040),
      session_id: 1,
    },
    // Docker commands in another directory
    Invocation {
      command: "docker ps".to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/home/user/docker-project".to_string()),
      hostname: Some("laptop".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(1620001000),
      end_unix_timestamp: Some(1620001002),
      session_id: 1,
    },
    Invocation {
      command: "docker build -t myapp:latest .".to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/home/user/docker-project".to_string()),
      hostname: Some("laptop".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(1620001010),
      end_unix_timestamp: Some(1620001060),
      session_id: 1,
    },
    Invocation {
      command: "docker run -p 8080:80 myapp:latest".to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/home/user/docker-project".to_string()),
      hostname: Some("laptop".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(1), // Failed with error
      start_unix_timestamp: Some(1620001070),
      end_unix_timestamp: Some(1620001075),
      session_id: 1,
    },
    // File operations in home directory
    Invocation {
      command: "ls -la".to_string(),
      shellname: "bash".to_string(),
      working_directory: Some("/home/user".to_string()),
      hostname: Some("laptop".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(1620002000),
      end_unix_timestamp: Some(1620002001),
      session_id: 1,
    },
    Invocation {
      command: "mkdir -p test/subdir".to_string(),
      shellname: "bash".to_string(),
      working_directory: Some("/home/user".to_string()),
      hostname: Some("laptop".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(1620002010),
      end_unix_timestamp: Some(1620002011),
      session_id: 1,
    },
    Invocation {
      command: "touch test/subdir/file.txt".to_string(),
      shellname: "bash".to_string(),
      working_directory: Some("/home/user".to_string()),
      hostname: Some("laptop".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(1620002020),
      end_unix_timestamp: Some(1620002021),
      session_id: 1,
    },
  ]
}

// Helper function to create a complex command with special characters
fn create_complex_command() -> Invocation {
  Invocation {
    command: "find . -type f -name \"*.rs\" | xargs grep -l \"fn main\" | sort -r | head -n 5"
      .to_string(),
    shellname: "bash".to_string(),
    working_directory: Some("/home/user/project".to_string()),
    hostname: Some("laptop".to_string()),
    username: Some("user".to_string()),
    exit_status: Some(0),
    start_unix_timestamp: Some(1620003000),
    end_unix_timestamp: Some(1620003010),
    session_id: 1,
  }
}

#[test]
fn test_embedding_various_commands() -> Result<()> {
  let device = Device::Cpu;
  let embedder = zage::model::create_embedder(device.clone())?;

  // Test different types of commands
  let basic_cmd = embedder.embed("ls -la")?;
  let git_cmd = embedder.embed("git commit -m 'update code'")?;
  let complex_cmd =
    embedder.embed("find . -type f -name \"*.rs\" | xargs grep \"fn main\" | sort | head -n 5")?;

  // Verify embeddings have consistent dimensions
  assert_eq!(
    basic_cmd.len(),
    git_cmd.len(),
    "Embeddings should have consistent dimensions"
  );
  assert_eq!(
    basic_cmd.len(),
    complex_cmd.len(),
    "Embeddings should have consistent dimensions"
  );

  // Verify similar commands have more similar embeddings than dissimilar ones
  let git_status = embedder.embed("git status")?;
  let git_add = embedder.embed("git add .")?;
  let ls_cmd = embedder.embed("ls")?;

  let git_distance = cosine_distance(&git_status, &git_add);
  let unrelated_distance = cosine_distance(&git_status, &ls_cmd);

  assert!(
    git_distance < unrelated_distance,
    "Similar commands should have more similar embeddings than dissimilar ones"
  );

  Ok(())
}

#[test]
fn test_context_feature_extraction() -> Result<()> {
  let device = Device::Cpu;
  let context_features = ContextualFeatures::new(device.clone(), create_embedder(device.clone())?)?;

  // Extract features from invocations with different contexts
  let invocations = create_diverse_test_invocations();

  // Test working directory context
  let project_features = context_features
    .encode_working_directory(&invocations[0].working_directory.clone().unwrap())?;
  let docker_features = context_features
    .encode_working_directory(&invocations[5].working_directory.clone().unwrap())?;

  // Different directories should produce different embeddings
  assert_ne!(
    project_features, docker_features,
    "Different directories should produce different embeddings"
  );

  // Test exit status encoding
  let success_features = context_features.encode_exit_status(Some(0))?;
  let error_features = context_features.encode_exit_status(Some(1))?;

  assert_ne!(
    success_features, error_features,
    "Different exit statuses should produce different features"
  );
  assert_eq!(
    success_features.len(),
    2,
    "Exit status features should have 2 dimensions"
  );

  // Test all features together
  let all_features = context_features.encode_all_features(&invocations[0])?;

  // Should have substantial number of features
  assert!(
    all_features.len() > 10,
    "Combined features should have substantial dimensions"
  );

  // Convert to tensor and verify shape
  let tensor = context_features.features_to_tensor(&all_features)?;
  assert_eq!(
    tensor.dims(),
    &[all_features.len()],
    "Tensor should have correct dimensions"
  );

  Ok(())
}

#[test]
fn test_feature_integration() -> Result<()> {
  let device = Device::Cpu;

  // Initialize components
  let embedder = create_embedder(device.clone())?;
  let context_features = Arc::new(ContextualFeatures::new(device.clone(), embedder.clone())?);

  // Create integrator with normalization enabled
  let config = FeatureIntegrationConfig {
    normalize_features: true,
  };
  let integrator = FeatureIntegrator::with_config(
    embedder.clone(),
    context_features.clone(),
    config,
    device.clone(),
  )?;

  let invocations = create_diverse_test_invocations();

  // Get integrated features for different invocations
  let git_features = integrator.integrate_features(&invocations[0])?;
  let docker_features = integrator.integrate_features(&invocations[5])?;
  let file_features = integrator.integrate_features(&invocations[8])?;

  // All features should have same dimensions
  assert_eq!(
    git_features.len(),
    docker_features.len(),
    "Integrated features should have consistent dimensions"
  );
  assert_eq!(
    git_features.len(),
    file_features.len(),
    "Integrated features should have consistent dimensions"
  );

  // Normalized features should have unit length
  assert_almost_eq(
    vector_norm(&git_features),
    1.0,
    1e-5,
    "Normalized features should have unit length",
  );

  // Test tensor creation for batch
  let tensor = integrator.create_integrated_tensor(&invocations[0..3])?;

  // Tensor should have correct dimensions
  assert_eq!(
    tensor.dims(),
    &[3, git_features.len()],
    "Tensor should have shape [batch_size, feature_dim]"
  );

  Ok(())
}

#[test]
fn test_training_dataset_creation() -> Result<()> {
  let device = Device::Cpu;

  // Initialize components
  let embedder = create_embedder(device.clone())?;
  let context_features = Arc::new(ContextualFeatures::new(device.clone(), embedder.clone())?);

  // Create dataset with sequence length 2
  let mut dataset = TrainingDataset::new(
    embedder.clone(),
    context_features.clone(),
    2,
    device.clone(),
  );

  // Generate examples from diverse invocations
  let invocations = create_diverse_test_invocations();
  dataset.generate_from_history(&invocations)?;

  // Should have expected number of examples
  let expected_examples = invocations.len() - 2;
  assert_eq!(
    dataset.len(),
    expected_examples,
    "Should have expected number of training examples"
  );

  // Test shuffling
  dataset.shuffle();
  assert_eq!(
    dataset.len(),
    expected_examples,
    "Shuffling should preserve dataset size"
  );

  // Test train/val split
  let validation_ratio = 0.2;
  let (train, val) = dataset.split_train_val(validation_ratio);

  let expected_val_size = (expected_examples as f64 * validation_ratio).round() as usize;
  let expected_train_size = expected_examples - expected_val_size;

  assert_eq!(
    train.len(),
    expected_train_size,
    "Training set should have expected size"
  );
  assert_eq!(
    val.len(),
    expected_val_size,
    "Validation set should have expected size"
  );

  // Test batch creation with different batch sizes
  let batch_size = 2;
  let batches = dataset.create_batches(batch_size);

  let expected_batches = (expected_examples + batch_size - 1) / batch_size;
  assert_eq!(
    batches.len(),
    expected_batches,
    "Should create expected number of batches"
  );

  // Verify first batch dimensions
  if !batches.is_empty() {
    let first_batch = &batches[0];
    assert_eq!(
      first_batch.input_embeddings.dims()[0],
      batch_size.min(expected_examples),
      "Batch should have correct batch size dimension"
    );

    // Verify the target tensor has the expected shape
    assert_eq!(
      first_batch.target_embeddings.dims().len(),
      2,
      "Target tensor should be 2-dimensional"
    );
  }

  Ok(())
}

#[test]
fn test_end_to_end_preprocessing() -> Result<()> {
  let device = Device::Cpu;

  // Initialize all components
  let embedder = create_embedder(device.clone())?;
  let context_features = Arc::new(ContextualFeatures::new(device.clone(), embedder.clone())?);
  let _integrator =
    FeatureIntegrator::new(embedder.clone(), context_features.clone(), device.clone())?;

  // Get diverse test data
  let invocations = create_diverse_test_invocations();

  // Process a sequence with complex command
  let mut test_sequence = invocations.clone();
  test_sequence.push(create_complex_command());

  // Create dataset
  let mut dataset = TrainingDataset::new(
    embedder.clone(),
    context_features.clone(),
    3, // Sequence length of 3
    device.clone(),
  );

  // Generate examples
  dataset.generate_from_history(&test_sequence)?;

  // Create batches
  let batches = dataset.create_batches(4);

  // Verify batches were created successfully
  assert!(!batches.is_empty(), "Should create at least one batch");

  // Check that all tensors have valid dimensions and values
  for batch in &batches {
    // Input should be 3D: [batch_size, seq_len, embedding_dim]
    assert_eq!(
      batch.input_embeddings.dims().len(),
      3,
      "Input tensor should be 3-dimensional"
    );

    // Target should be 2D: [batch_size, embedding_dim]
    assert_eq!(
      batch.target_embeddings.dims().len(),
      2,
      "Target tensor should be 2-dimensional"
    );

    // Context should be 2D: [batch_size, context_dim]
    assert_eq!(
      batch.target_context_features.dims().len(),
      2,
      "Context tensor should be 2-dimensional"
    );

    // All batches should have the same number of examples
    assert_eq!(
      batch.target_commands.len(),
      batch.input_embeddings.dims()[0],
      "Number of target commands should match batch dimension"
    );
  }

  Ok(())
}

#[test]
fn benchmark_preprocessing_pipeline() -> Result<()> {
  let device = Device::Cpu;

  // Initialize components
  let embedder = create_embedder(device.clone())?;
  let context_features = Arc::new(ContextualFeatures::new(device.clone(), embedder.clone())?);
  let integrator =
    FeatureIntegrator::new(embedder.clone(), context_features.clone(), device.clone())?;

  let invocations = create_diverse_test_invocations();

  // Benchmark embedding
  let embedding_start = Instant::now();
  for inv in &invocations {
    let _ = embedder.embed(&inv.command)?;
  }
  let embedding_duration = embedding_start.elapsed();
  let embedding_avg = embedding_duration / invocations.len() as u32;

  // Benchmark context feature extraction
  let context_start = Instant::now();
  for inv in &invocations {
    let _ = context_features.encode_all_features(inv)?;
  }
  let context_duration = context_start.elapsed();
  let context_avg = context_duration / invocations.len() as u32;

  // Benchmark feature integration
  let integration_start = Instant::now();
  for inv in &invocations {
    let _ = integrator.integrate_features(inv)?;
  }
  let integration_duration = integration_start.elapsed();
  let integration_avg = integration_duration / invocations.len() as u32;

  // Benchmark dataset creation
  let dataset_start = Instant::now();
  let mut dataset = TrainingDataset::new(
    embedder.clone(),
    context_features.clone(),
    2,
    device.clone(),
  );
  dataset.generate_from_history(&invocations)?;
  let dataset_duration = dataset_start.elapsed();

  // Benchmark batch creation
  let batch_start = Instant::now();
  let _ = dataset.create_batches(4);
  let batch_duration = batch_start.elapsed();

  // Display benchmark results
  println!("Preprocessing Pipeline Benchmarks:");
  println!(
    "  Command Embedding:        {:?} avg/command",
    embedding_avg
  );
  println!("  Context Feature Extract:  {:?} avg/command", context_avg);
  println!(
    "  Feature Integration:      {:?} avg/command",
    integration_avg
  );
  println!("  Dataset Creation:         {:?} total", dataset_duration);
  println!("  Batch Creation:           {:?} total", batch_duration);

  // This test does not assert anything - it just provides benchmark information
  Ok(())
}

// Helper function to calculate cosine distance between vectors
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
  if a.len() != b.len() || a.is_empty() {
    return 1.0; // Maximum distance for incomparable vectors
  }

  let mut dot_product = 0.0;
  let mut norm_a = 0.0;
  let mut norm_b = 0.0;

  for i in 0..a.len() {
    dot_product += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }

  norm_a = norm_a.sqrt();
  norm_b = norm_b.sqrt();

  if norm_a == 0.0 || norm_b == 0.0 {
    return 1.0; // Maximum distance for zero vectors
  }

  1.0 - (dot_product / (norm_a * norm_b))
}

// Helper function to calculate the L2 norm of a vector
fn vector_norm(v: &[f32]) -> f32 {
  let mut sum_squared = 0.0;
  for val in v {
    sum_squared += val * val;
  }
  sum_squared.sqrt()
}

// Helper function for approximate float comparison
fn assert_almost_eq(a: f32, b: f32, epsilon: f32, message: &str) {
  assert!((a - b).abs() < epsilon, "{}: {} vs {}", message, a, b);
}
