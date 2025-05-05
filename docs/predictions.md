# Zage Neural Network Prediction System

This document provides a comprehensive explanation of how Zage's LSTM neural network prediction system works, from high-level concepts to implementation details.

## Table of Contents

1. [Introduction](#introduction)
2. [How Neural Networks Work](#how-neural-networks-work)
3. [LSTM Networks Explained](#lstm-networks-explained)
4. [Zage Prediction System Architecture](#zage-prediction-system-architecture)
5. [Data Preparation](#data-preparation)
6. [Training Process](#training-process)
7. [Inference Process](#inference-process)
8. [Implementation Guide](#implementation-guide)
9. [Evaluation and Metrics](#evaluation-and-metrics)
10. [Troubleshooting](#troubleshooting)

## Introduction

Zage's prediction system uses a Long Short-Term Memory (LSTM) neural network to predict the next shell command a user is likely to execute based on their command history and contextual information. This document explains how this system works in detail, without requiring prior machine learning expertise.

## How Neural Networks Work

### Basic Concepts

Neural networks are computational systems inspired by the human brain. They consist of:

1. **Neurons (Nodes)**: Basic computational units that process input data
2. **Connections**: Links between neurons with associated weights
3. **Layers**: Groups of neurons, typically organized as:
   - Input layer: Receives initial data
   - Hidden layer(s): Performs intermediate computations
   - Output layer: Produces the final result

### Simple Example

Imagine predicting whether a user will run `git commit` after running `git add`. A simple neural network might:

1. Take inputs like:
   - Previous command was `git add`
   - Current directory is a git repository
   - Time of day is evening

2. Process these inputs through hidden layers that learn patterns

3. Output a probability (e.g., 85%) that the next command will be `git commit`

### How Learning Happens

Neural networks learn through a process called **backpropagation**:

1. **Forward pass**: The network makes a prediction
2. **Loss calculation**: The prediction is compared to the actual outcome
3. **Backward pass**: The network adjusts its weights to reduce errors
4. **Iteration**: This process repeats with many examples until the network becomes accurate

## LSTM Networks Explained

### The Problem with Simple Networks

Standard neural networks struggle with sequence data (like command history) because they don't maintain "memory" of past inputs. For example, the commands `cd project`, `git status`, and `git add .` form a logical sequence that should inform the prediction of `git commit -m "..."`.

### LSTM Solution

LSTM (Long Short-Term Memory) networks solve this problem by incorporating memory cells that can remember information over long sequences.

#### Key Components

An LSTM cell contains:

1. **Cell State**: Long-term memory that runs through the entire sequence
2. **Hidden State**: Short-term memory updated at each step
3. **Gates**: Control mechanisms that regulate information flow:
   - **Forget Gate**: Decides what to discard from cell state
   - **Input Gate**: Decides what new information to store
   - **Output Gate**: Decides what to output based on cell state

#### Visual Explanation

```text
Input: command sequence ["cd project", "git status", "git add ."]
                  ↓
┌─────────────────────────────────────┐
│                LSTM                 │
│  ┌─────┐     ┌─────┐     ┌─────┐   │
│  │Cell │     │Cell │     │Cell │   │
│  │  1  │ → → │  2  │ → → │  3  │   │
│  └─────┘     └─────┘     └─────┘   │
└─────────────────────────────────────┘
                  ↓
Output: prediction ["git commit -m "..."]
```

Each LSTM cell processes one command, maintains memory of important patterns, and passes this information to the next cell.

## Zage Prediction System Architecture

Zage's prediction system consists of several components:

### 1. Command Embedding

Raw shell commands need to be converted into numerical form for neural network processing:

```text
"git commit -m 'update readme'" → [0.2, -0.5, 0.8, ..., 0.1]
```

This is done through:

- Tokenization (splitting commands into meaningful parts)
- Embedding (converting tokens to numerical vectors)

### 2. Context Enrichment

Beyond just the command text, Zage incorporates:

- Current working directory
- Previous command exit status
- Time of day/week patterns
- Session information

Each of these is encoded and added to the network input.

### 3. LSTM Model

The core prediction engine consists of:

- Input layer matching the embedding dimensions
- LSTM layers (typically 2-3) for sequence processing
- Dense output layer that produces command predictions

### 4. Candidate Selection

The raw network output is processed to:

- Filter invalid or unsafe commands
- Apply relevance scoring
- Sort and rank predictions

## Data Preparation

### Collection Process

Zage collects training data from:

1. User's existing shell history files
2. Ongoing command execution (with user permission)
3. Directory context and other metadata

### Preprocessing Steps

1. **Cleaning**: Remove sensitive information, standardize formatting
2. **Tokenization**: Break commands into tokens

   ```text
   "git commit -m 'update'" → ["git", "commit", "-m", "update"]
   ```

3. **Embedding**: Convert tokens to numerical vectors
4. **Sequencing**: Group commands into sequences with context
5. **Augmentation**: Generate variations of commands for robustness

### Feature Engineering

To help the model recognize patterns, we extract features like:

- Command categories (git, file operations, network, etc.)
- Argument patterns
- Time and session patterns
- Directory structure information

## Training Process

### Initial Training

1. **Data Split**: Dividing data into training (80%) and validation (20%) sets
2. **Batch Processing**: Processing sequences in small batches (typically 32-64)
3. **Epoch Iteration**: Repeating training over the entire dataset multiple times
4. **Learning Rate Scheduling**: Adjusting how quickly the model adapts
5. **Regularization**: Preventing overfitting to training data

### Continuous Learning

Zage's model isn't static—it continues to learn from user behavior:

1. Collect new command sequences
2. Periodically retrain the model with both historical and new data
3. Update model parameters while preserving existing knowledge

### Implementation with Candle

[Candle](https://github.com/huggingface/candle) is a Rust-native machine learning framework that powers Zage's neural network. Key advantages include:

- Pure Rust implementation (no Python dependencies)
- Memory efficiency and speed
- First-class support for different hardware acceleration options

## Inference Process

When predicting the next command, Zage:

1. **Context Gathering**: Collects current working directory, recent commands, etc.
2. **Preprocessing**: Transforms context into the format expected by the model
3. **Forward Pass**: Runs the preprocessed data through the trained LSTM
4. **Post-processing**: Transforms raw output into human-readable command suggestions
5. **Presentation**: Shows top predictions to the user through shell integration

## Implementation Guide

This section provides concrete guidance for implementing the prediction system.

### Command Embedding Implementation

```rust
use candle_core::{Tensor, Device};
use candle_nn::{Embedding, VarBuilder};

struct CommandEncoder {
    vocabulary: HashMap<String, usize>,
    embedding: Embedding,
    max_tokens: usize,
}

impl CommandEncoder {
    pub fn new(vocabulary_size: usize, embedding_dim: usize, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(vocabulary_size, embedding_dim, vb.pp("embedding"))?;
        
        Ok(Self {
            vocabulary: HashMap::new(),
            embedding,
            max_tokens: 50,
        })
    }
    
    pub fn encode(&self, command: &str, device: &Device) -> Result<Tensor> {
        // Tokenize the command
        let tokens = self.tokenize(command);
        
        // Convert tokens to indices
        let indices: Vec<usize> = tokens.iter()
            .map(|token| self.vocabulary.get(token).copied().unwrap_or(0))
            .collect();
            
        // Create tensor from indices
        let indices_tensor = Tensor::new(&indices[..], device)?;
        
        // Embed the indices
        self.embedding.forward(&indices_tensor)
    }
    
    fn tokenize(&self, command: &str) -> Vec<String> {
        // Simple whitespace tokenization for example purposes
        // A real implementation should handle quotes, escapes, etc.
        command.split_whitespace()
            .map(|s| s.to_lowercase())
            .take(self.max_tokens)
            .map(String::from)
            .collect()
    }
}
```

### LSTM Model Implementation

```rust
use candle_core::{Tensor, Device, Result};
use candle_nn::{LSTM, Linear, Module, VarBuilder};

struct CommandPredictor {
    lstm: LSTM,
    output: Linear,
    hidden_size: usize,
    vocabulary_size: usize,
}

impl CommandPredictor {
    pub fn new(
        embedding_dim: usize,
        hidden_size: usize,
        vocabulary_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Initialize LSTM layer
        let lstm = candle_nn::lstm(embedding_dim, hidden_size, vb.pp("lstm"))?;
        
        // Initialize output projection
        let output = candle_nn::linear(
            hidden_size, 
            vocabulary_size, 
            vb.pp("output")
        )?;
        
        Ok(Self {
            lstm,
            output,
            hidden_size,
            vocabulary_size,
        })
    }
    
    pub fn forward(&self, embeddings: &Tensor) -> Result<Tensor> {
        // Process the sequence through LSTM
        let (outputs, _) = self.lstm.seq_init(embeddings)?;
        
        // Take the final output and project to vocabulary size
        let prediction = self.output.forward(&outputs)?;
        
        // Apply softmax to get probabilities
        candle_nn::ops::softmax(&prediction, candle_core::D::Minus1)
    }
    
    pub fn predict_next_command(
        &self,
        command_history: &[String],
        encoder: &CommandEncoder,
        device: &Device,
    ) -> Result<Vec<(String, f32)>> {
        // Encode command history
        let encoded_commands = command_history.iter()
            .map(|cmd| encoder.encode(cmd, device))
            .collect::<Result<Vec<_>>>()?;
            
        // Stack tensors into a batch
        let batch = Tensor::stack(&encoded_commands, 0)?;
        
        // Forward pass through the model
        let predictions = self.forward(&batch)?;
        
        // Get top-k predictions
        let (values, indices) = candle_nn::ops::topk(&predictions, 5, candle_core::D::Minus1, true)?;
        
        // Convert predictions to commands
        let values_vec = values.to_vec1::<f32>()?;
        let indices_vec = indices.to_vec1::<usize>()?;
        
        // Map indices back to commands (in a real implementation)
        // This is simplified here
        let predictions = indices_vec.iter()
            .zip(values_vec.iter())
            .map(|(idx, score)| (format!("predicted_command_{}", idx), *score))
            .collect();
            
        Ok(predictions)
    }
}
```

### Training Loop Implementation

```rust
fn train_model(
    model: &mut CommandPredictor,
    encoder: &CommandEncoder,
    dataset: &CommandDataset,
    learning_rate: f64,
    epochs: usize,
    device: &Device,
) -> Result<()> {
    // Initialize optimizer
    let mut opt = candle_nn::AdamW::new(
        model.parameters(),
        candle_nn::ParamsAdamW {
            lr: learning_rate,
            ..Default::default()
        }
    )?;
    
    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut samples = 0;
        
        // Iterate through batches
        for (inputs, targets) in dataset.batches(32)? {
            // Forward pass
            let predictions = model.forward(&inputs)?;
            
            // Calculate loss
            let loss = candle_nn::loss::cross_entropy(&predictions, &targets)?;
            
            // Backward pass
            opt.backward_step(&loss)?;
            
            total_loss += loss.to_scalar::<f32>()?;
            samples += inputs.dim(0)?;
        }
        
        println!("Epoch {}: Avg loss {}", epoch, total_loss / samples as f32);
    }
    
    Ok(())
}
```

## Evaluation and Metrics

To measure prediction system performance, we track:

### Accuracy Metrics

1. **Top-k Accuracy**: Percentage of times the correct command is in top k predictions
2. **MRR (Mean Reciprocal Rank)**: Average of 1/position for the correct command
3. **Precision@k**: Proportion of relevant commands in top k predictions

### User Experience Metrics

1. **Acceptance Rate**: How often users select a suggested command
2. **Time Saved**: Reduction in time/keystrokes when using predictions
3. **Learning Curve**: How prediction accuracy improves over time

## Troubleshooting

### Common Issues

1. **Low Prediction Accuracy**
   - Possible causes: Insufficient training data, over-regularization
   - Solution: Collect more diverse command history, adjust model hyperparameters

2. **Slow Inference Time**
   - Possible causes: Model too large, inefficient embedding
   - Solution: Model quantization, caching frequent predictions, optimizing inference path

3. **Irrelevant Predictions**
   - Possible causes: Context not properly incorporated, overfit to common commands
   - Solution: Improve context weighting, adjust candidate scoring algorithm

### Debugging Techniques

1. **Logging Prediction Scores**: Record confidence scores for each prediction
2. **Command Sequence Visualization**: Plot command sequences and predictions
3. **Embedding Analysis**: Visualize command embeddings to see clustering

## Conclusion

Zage's LSTM-based prediction system transforms shell command usage by learning from patterns in user behavior. By combining neural network techniques with contextual awareness, it can significantly improve productivity through accurate command suggestions.

Building this system requires careful attention to data preparation, model architecture, and user experience—but the result is a powerful tool that gets more useful over time as it adapts to individual usage patterns.
