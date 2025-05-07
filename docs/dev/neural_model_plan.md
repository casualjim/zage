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

### 2.1 Shell Command Tokenization (Simplified)

- [x] Add `shell-words` dependency to `Cargo.toml` for POSIX-compliant splitting
- [x] Remove `yash-syntax` dependency; no AST-based parsing
- [ ] Ensure `ShellTokenizer` uses `shell-words` with whitespace fallback
- [ ] Make `ShellTokenizer` stateful, tracking token counts
- [ ] Add token frequency counting for vocabulary building
- [ ] Test tokenization across diverse shell command patterns

### 2.2 Vocabulary Building

- [x] Enhance `VocabularyBuilder` to process command history:
  - [x] Store tokenized corpus for embedding generation
  - [x] Implement token frequency-based vocabulary selection
  - [x] Add special tokens (UNK, PAD, BOS, EOS)
  - [x] Implement configurable parameters:
    - [x] Minimum token frequency
    - [x] Maximum vocabulary size
    - [x] Special token handling
  - [x] Create token-to-ID and ID-to-token mappings
  - [x] Test with real command history samples

### 2.3 Command Encoding

- [x] Implement multiple encoding strategies inspired by SLP:
  - [x] TF-IDF encoding:
    - [x] Calculate document frequency for each token
    - [x] Calculate term frequency for tokens in each command
    - [x] Combine into TF-IDF representation
  - [x] One-hot encoding:
    - [x] Create sparse vectors of vocabulary size
    - [x] Set corresponding values to 1 for present tokens
  - [x] Label encoding:
    - [x] Convert token sequences to ID sequences
    - [x] Add padding to fixed length
  - [x] Test each encoding strategy with varied command inputs

### 2.4 Embedding Layer

- [ ] Implement embedding layer using Candle:
  - [ ] Create embedding table with dimensions (vocab_size, embedding_dim)
  - [ ] Initialize with uniform or normal distribution
  - [ ] Implement lookup operation for token IDs
  - [ ] Add positional encoding for sequence position information
  - [ ] Implement optional embedding normalization
  - [ ] Test embedding output shapes and values

### 2.5 Contextual Features

- [ ] Design and implement context extraction:
  - [ ] Working directory embedding:
    - [ ] Tokenize path components
    - [ ] Encode using vocabulary or separate path encoder
  - [ ] Command success/failure representation:
    - [ ] Encode exit status as binary or categorical feature
  - [ ] Time features:
    - [ ] Extract hour of day, day of week, etc.
    - [ ] Encode as cyclic features using sine/cosine transformations
  - [ ] User/host representation if applicable
  - [ ] Test context feature extraction with diverse inputs

### 2.6 Syntax-Aware Embeddings

- [ ] Leverage AST structure for improved embeddings:
  - [ ] Develop position-aware token embeddings based on AST depth
  - [ ] Create node type embeddings (command, argument, operator, etc.)
  - [ ] Implement parent-child relationship encoding
  - [ ] Create command-argument relationship encoding
  - [ ] Test syntax-aware embeddings against baseline

### 2.7 Training Data Generation

- [ ] Create comprehensive dataset from shell history:
  - [ ] Implement sequence windowing to create training examples
  - [ ] For each example, extract:
    - [ ] Input sequence (previous commands)
    - [ ] Target sequence (command to predict)
    - [ ] Associated contextual features
    - [ ] Syntax information from AST
  - [ ] Split into training/validation sets
  - [ ] Implement batching and shuffling mechanisms
  - [ ] Add data augmentation techniques if applicable
  - [ ] Create data loaders compatible with Candle
  - [ ] Test dataset creation and iteration

### 2.8 Feature Integration

- [ ] Create combined input representation:
  - [ ] Concatenate or otherwise merge command embeddings with contextual features
  - [ ] Implement feature normalization if needed
  - [ ] Create input tensor formatting for model
  - [ ] Test integrated feature representation

### 2.9 Testing and Validation

- [ ] Implement comprehensive tests for preprocessing pipeline:
  - [ ] Unit tests for tokenizer with edge cases
  - [ ] Tests for AST parsing with various shell constructs
  - [ ] Tests for vocabulary building and selection
  - [ ] Tests for each encoding strategy
  - [ ] Tests for embedding layer functionality
  - [ ] End-to-end tests for complete preprocessing pipeline
  - [ ] Benchmark preprocessing pipeline performance

## 3. Model Architecture

- [ ] Design LSTM/Transformer architecture in Candle
- [ ] Implement forward pass for sequence prediction
- [ ] Integrate context vectors into model
- [ ] Implement attention mechanism for context awareness
- [ ] Create prediction layer for command probabilities
- [ ] Implement beam search for command generation
- [ ] Implement model saving/loading to/from disk
- [ ] Model versioning for compatibility
- [ ] Working tests for model forward pass, context injection, prediction head, and serialization

## 4. Training Pipeline

- [ ] Implement loss function for command prediction
- [ ] Add regularization techniques
- [ ] Create training loop with learning rate scheduling
- [ ] Implement early stopping and checkpointing
- [ ] Implement accuracy, perplexity, and other metrics
- [ ] Logging system for training progress
- [ ] Hyperparameter tuning system (simple grid search)
- [ ] Working tests for loss, training loop, metrics, and parameter tuning

## 5. Inference & Integration

- [ ] Create efficient inference pipeline for prediction
- [ ] Implement top-k and top-p sampling for suggestions
- [ ] Model evaluation and benchmarking suite
- [ ] Update shell integration scripts to use neural model
- [ ] Optimize for low-latency prediction
- [ ] Implement fallback to simpler models when appropriate
- [ ] Confidence-based model selection
- [ ] Working tests for inference, integration, and fallback

## 6. Advanced Features

- [ ] Incorporate command output as context for predictions
- [ ] Implement filtering based on output patterns
- [ ] Add time-of-day and day-of-week features
- [ ] Implement seasonal pattern detection
- [ ] Create shared model state across terminal instances
- [ ] Implement synchronization for multi-terminal prediction
- [ ] Profile and optimize model for latency reduction
- [ ] Implement quantization for reduced memory footprint
- [ ] Working tests for advanced features and optimizations

## 7. Testing & Documentation (Continuous)

- [ ] Ensure comprehensive unit test coverage for all components
- [ ] Implement property-based testing for edge cases
- [ ] Create end-to-end integration tests for full prediction pipeline
- [ ] Test across different shell environments
- [ ] Update `development.md` and create model architecture documentation
- [ ] Add examples for custom model training
- [ ] Create benchmark suite for prediction accuracy and latency
- [ ] Working tests for documentation and benchmarks

## Implementation Guidelines

- Write tests first for each component (TDD)
- Maintain or improve test coverage with all code changes
- Follow Rust idioms and naming conventions
- Use expressive variable names and comprehensive error handling
- Balance accuracy with prediction latency and resource constraints
- Set up CI pipeline for automated validation (tests, benchmarks, docs)
- Require all tests to pass before merging
