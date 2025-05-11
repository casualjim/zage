# Neural Network Implementation Plan for Zage

This plan describes the steps to replace all existing models with a Candle-based neural network for predicting the next shell command, ensuring comprehensive testing at every stage. Use the checkboxes to track progress for each work item.

## 1. Project Setup & Dependencies

- [x] Update `Cargo.toml` with Candle dependencies (`candle-core`, `candle-nn`, etc.)
- [ ] Add testing utilities for model validation
- [x] Create foundation for neural network module (`src/model/neural.rs`)
- [x] Define trait implementations compatible with existing model interface
- [ ] Define data structures for training & inference
- [x] Working tests for module imports and trait implementations

## 2. Data Preprocessing

### 2.1 Command Embeddings

- [x] Add HuggingFace dependency `hf-hub` to Cargo.toml
- [x] Add tokenizers dependency to Cargo.toml
- [x] Implement `PretrainedEmbedder` using a HuggingFace model
- [x] Use CodeRankEmbed model for embedding shell commands
- [x] Add functions to embed text into fixed-size vectors
- [x] Test embedding output shapes and values
- [x] Measure embedding latency for performance benchmarking

### 2.2 Contextual Features

- [x] Design and implement context extraction:
  - [x] Working directory embedding:
    - [x] Use pretrained embedder to encode path components
  - [x] Command success/failure representation:
    - [x] Encode exit status as binary or categorical feature
  - [x] Time features:
    - [x] Extract hour of day, day of week, etc.
    - [x] Encode as cyclic features using sine/cosine transformations
  - [x] User/host representation if applicable
  - [x] Remote/local session
  - [x] Test context feature extraction with diverse inputs

### 2.3 Training Data Generation

- [x] Create comprehensive dataset from shell history:
  - [x] Implement sequence windowing to create training examples
  - [x] For each example, extract:
    - [x] Input sequence (previous commands as embeddings)
    - [x] Target sequence (command to predict)
    - [x] Associated contextual features
  - [x] Split into training/validation sets
  - [x] Implement batching and shuffling mechanisms
  - [x] Create data loaders compatible with Candle
  - [x] Test dataset creation and iteration

### 2.4 Feature Integration

- [x] Create combined input representation:
  - [x] Concatenate or otherwise merge command embeddings with contextual features
  - [x] Implement feature normalization if needed
  - [x] Create input tensor formatting for model
  - [x] Test integrated feature representation

### 2.5 Testing and Validation

- [x] Implement comprehensive tests for preprocessing pipeline:
  - [x] Tests for embedding functionality with various shell commands
  - [x] Tests for context feature extraction with diverse inputs
  - [x] Tests for dataset creation and batching
  - [x] End-to-end tests for complete preprocessing pipeline
  - [x] Benchmark preprocessing pipeline performance

## 3. Model Architecture

### 3.1 Architecture Selection & Evaluation

- [ ] Evaluate LSTM-based architecture:
  - [ ] Define number of layers, hidden size, dropout, embedding size
  - [ ] Prototype using Candle's RNN modules
  - [ ] Benchmark on sample dataset (accuracy, latency)
- [ ] Evaluate Transformer-based architecture:
  - [ ] Define number of layers, attention heads, model dimension, positional encoding
  - [ ] Prototype using Candle's transformer modules
  - [ ] Benchmark on sample dataset (accuracy, latency)
- [ ] Compare architectures and select best-performing model

### 3.2 Model Definition

- [ ] Define `ModelConfig` struct with hyperparameters
- [ ] Implement model struct and constructor
- [ ] Implement forward pass:
  - [ ] Sequence encoding pipeline (e.g., RNN or self-attention)
  - [ ] Integrate contextual feature vectors
  - [ ] Prediction head outputting command probabilities

### 3.3 Inference Optimization

- [ ] Add top-k and top-p sampling methods
- [ ] Implement caching for recurrent states or attention key/value tensors
- [ ] Optimize tensor operations for low-latency inference

### 3.4 Model Persistence

- [ ] Implement `save` method to serialize model parameters to disk (e.g., `.ckpt` files)
- [ ] Implement `load` method to restore model from disk
- [ ] Test round-trip serialization integrity

### 3.5 Testing & Validation

- [ ] Unit tests for model components:
  - [ ] Test shapes and types of forward pass outputs
  - [ ] Test behavior with dummy inputs
- [ ] Integration tests:
  - [ ] Load sample model, run inference, validate output distribution
  - [ ] Ensure reproducibility across runs
- [ ] Benchmark model runtime and memory usage on sample inputs

## 4. Training Pipeline

- [ ] Implement loss function for command prediction
- [ ] Create training loop with optimization:
  - [ ] Set up optimizer with learning rate schedule
  - [ ] Implement gradient accumulation if needed
  - [ ] Add regularization techniques
  - [ ] Implement early stopping and checkpointing
  - [ ] Create validation loop
- [ ] Implement evaluation metrics:
  - [ ] Accuracy, perplexity, etc.
  - [ ] Command suggestion quality metrics
- [ ] Add logging and visualization
- [ ] Working tests for training components

## 5. Inference & Integration

- [ ] Create efficient inference pipeline:
  - [ ] Implement top-k and top-p sampling for suggestions
  - [ ] Add cache mechanism for faster repeated predictions
  - [ ] Optimize for low-latency prediction
- [ ] Integrate with shell:
  - [ ] Update shell integration scripts to use neural model
  - [ ] Implement fallback to simpler models when appropriate
  - [ ] Add confidence-based model selection
- [ ] Comprehensive test suite:
  - [ ] Unit tests for inference components
  - [ ] Integration tests with shell environment
  - [ ] Performance benchmarks

## 6. Advanced Features

- [ ] Add advanced context awareness:
  - [ ] Incorporate command output as context
  - [ ] Implement time-based patterns (time of day, day of week)
  - [ ] Add user activity patterns
- [ ] Improve multi-terminal support:
  - [ ] Create shared model state across terminal instances
  - [ ] Implement synchronization mechanism
- [ ] Optimize performance:
  - [ ] Profile and optimize for latency
  - [ ] Implement quantization for reduced memory footprint
  - [ ] Add model compression techniques if needed
- [ ] Working tests for advanced features

## 7. Testing & Documentation

- [ ] Comprehensive test coverage:
  - [ ] Unit tests for all components
  - [ ] Integration tests for end-to-end functionality
  - [ ] Property-based tests for edge cases
  - [ ] Cross-shell environment tests
- [ ] Documentation:
  - [ ] Update development docs with neural model architecture
  - [ ] Add examples for custom model training
  - [ ] Create user guide for model configuration
- [ ] Benchmarking:
  - [ ] Prediction accuracy benchmarks
  - [ ] Latency and resource usage benchmarks

## Implementation Guidelines

- Write tests first for each component (TDD approach)
- Maintain or improve test coverage with all code changes
- Follow Rust idioms and naming conventions
- Use expressive variable names and comprehensive error handling
- Balance accuracy with prediction latency and resource constraints
- Set up CI pipeline for automated validation (tests, benchmarks, docs)
- Require all tests to pass before merging
