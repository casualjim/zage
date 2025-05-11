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

- [ ] Create comprehensive dataset from shell history:
  - [ ] Implement sequence windowing to create training examples
  - [ ] For each example, extract:
    - [ ] Input sequence (previous commands as embeddings)
    - [ ] Target sequence (command to predict)
    - [ ] Associated contextual features
  - [ ] Split into training/validation sets
  - [ ] Implement batching and shuffling mechanisms
  - [ ] Create data loaders compatible with Candle
  - [ ] Test dataset creation and iteration

### 2.4 Feature Integration

- [ ] Create combined input representation:
  - [ ] Concatenate or otherwise merge command embeddings with contextual features
  - [ ] Implement feature normalization if needed
  - [ ] Create input tensor formatting for model
  - [ ] Test integrated feature representation

### 2.5 Testing and Validation

- [ ] Implement comprehensive tests for preprocessing pipeline:
  - [ ] Tests for embedding functionality with various shell commands
  - [ ] Tests for context feature extraction with diverse inputs
  - [ ] Tests for dataset creation and batching
  - [ ] End-to-end tests for complete preprocessing pipeline
  - [ ] Benchmark preprocessing pipeline performance

## 3. Model Architecture

- [ ] Design and implement neural network architecture:
  - [ ] Choose between LSTM or Transformer-based architecture
  - [ ] Define model parameters (layers, dimensions, etc.)
  - [ ] Implement forward pass for sequence prediction
  - [ ] Integrate context vectors into model
  - [ ] Create prediction head for command probabilities
  - [ ] Add optional caching for efficient inference
  - [ ] Implement model saving/loading to/from disk
  - [ ] Working tests for model components and forward pass

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
