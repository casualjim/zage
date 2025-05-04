# Zage Development Plan

This document outlines the development plan and architecture for Zage, an intelligent shell command prediction plugin built in Rust.

## Project Overview

Zage aims to predict the next shell command a user is likely to run based on:
- Command history
- Current working directory
- Recent command sequences
- Exit status of previous commands
- Time and contextual patterns

Unlike simple history search tools, Zage uses LSTM neural networks to identify and learn complex command sequences within different contexts, working as a seamless plugin for Zsh (and potentially Bash).

## Current Project Structure

```
zage/
├── src/
│   ├── config.rs          # Configuration management
│   ├── db.rs              # Database connection and operations
│   ├── err.rs             # Error types and handling
│   ├── lib.rs             # Library entry point
│   ├── main.rs            # CLI entry point
│   └── shell_history/     # Shell history parsing
│       ├── mod.rs
│       ├── bash.rs        # Bash history parser
│       └── zsh.rs         # Zsh history parser
└── docs/
    └── dev/
        └── development.md # This file
```

## Target Architecture

```
zage/
├── src/
│   ├── cli.rs             # Command-line interface handling
│   ├── config.rs          # Configuration management
│   ├── db.rs              # Database connection and operations
│   ├── err.rs             # Error types and handling
│   ├── lib.rs             # Library entry point
│   ├── main.rs            # CLI entry point
│   ├── model/             # ML model implementation
│   │   ├── mod.rs
│   │   ├── features.rs    # Feature extraction
│   │   ├── lstm.rs        # LSTM implementation
│   │   ├── markov.rs      # Simpler Markov model (initial implementation)
│   │   └── training.rs    # Model training logic
│   ├── shell_history/     # Shell history parsing
│   │   ├── mod.rs
│   │   ├── bash.rs        # Bash history parser
│   │   └── zsh.rs         # Zsh history parser
│   └── shell_integration/ # Shell integration scripts
│       ├── mod.rs
│       ├── bash.rs        # Bash integration (future)
│       └── zsh.rs         # Zsh integration
└── docs/
    └── dev/
        └── development.md # This file
```

### Data Flow

1. **Data Collection**:
   - Hook into shell to capture commands as they're executed
   - Store commands with metadata (directory, timestamp, exit status)
   - Parse existing history files for initial training data

2. **Feature Extraction**:
   - Extract contextual features from commands
   - Generate sequence embeddings for LSTM input

3. **Model Training**:
   - Initial simple model: N-gram or Markov chain
   - Advanced model: LSTM neural network
   - Periodic retraining as new commands are collected

4. **Prediction**:
   - Generate command predictions based on current context
   - Rank predictions by confidence score
   - Automatically suggest the next command

## Implementation Plan

The development will proceed in phases, each building on the previous:

### Phase 1: Foundation (Current)

- [x] Project setup with CLI framework
- [x] Basic error handling
- [x] Configuration system
- [x] Shell history parsing (Bash, Zsh)
- [x] SQLite database schema and operations  # Completed: includes schema init, insert_invocation, and tests
- [ ] Command collection system

### Phase 2: Simple Prediction

- [ ] Implement N-gram model for baseline predictions
- [ ] Add Markov chain model with context awareness
- [ ] Zsh plugin integration
- [ ] Initial sequence detection algorithm

### Phase 3: LSTM Implementation

- [ ] Command embedding generation
- [ ] Feature extraction from commands and context
- [ ] LSTM model implementation using tch-rs
- [ ] Training pipeline
- [ ] Prediction pipeline

### Phase 4: Advanced Features

- [ ] Context enhancement with directory, exit status
- [ ] Time-based patterns detection
- [ ] Multi-terminal awareness
- [ ] Performance optimizations
- [ ] Bash plugin integration (if feasible)

## Database Schema

```sql
CREATE TABLE commands (
    id BLOB PRIMARY KEY,
    command TEXT NOT NULL,
    command_template TEXT,
    working_directory TEXT,
    exit_status INTEGER,
    start_timestamp INTEGER NOT NULL,
    end_timestamp INTEGER,
    terminal_id TEXT,
    session_id TEXT
);

CREATE TABLE sequences (
    id BLOB PRIMARY KEY,
    name TEXT,
    context TEXT,
    detected_count INTEGER DEFAULT 1
);

CREATE TABLE sequence_commands (
    sequence_id BLOB,
    command_id BLOB,
    position INTEGER,
    PRIMARY KEY (sequence_id, position),
    FOREIGN KEY (sequence_id) REFERENCES sequences(id),
    FOREIGN KEY (command_id) REFERENCES commands(id)
);

CREATE TABLE predictions (
    id BLOB PRIMARY KEY,
    context TEXT,
    predicted_command_id BLOB,
    actual_command_id BLOB,
    confidence REAL,
    was_used BOOLEAN,
    timestamp INTEGER,
    FOREIGN KEY (predicted_command_id) REFERENCES commands(id),
    FOREIGN KEY (actual_command_id) REFERENCES commands(id)
);
```

## Model Details

### LSTM Architecture

```
Input Layer (Features) → Embedding Layer → LSTM Layer(s) → Dense Layer → Output (Command Prediction)
```

- **Input Features**:
  - Command embeddings (tokenized commands)
  - Directory embeddings
  - Time features (hour of day, day of week)
  - Exit status encoding
  - Session/terminal ID

- **LSTM Configuration**:
  - Hidden size: 128
  - Number of layers: 2
  - Dropout: 0.2 (for regularization)

- **Training Parameters**:
  - Loss: Cross-entropy
  - Optimizer: Adam
  - Learning rate: 0.001
  - Batch size: 64
  - Epochs: Dynamic based on validation performance

### Feature Engineering

Commands will be processed as follows:

1. **Command Tokenization**:
   - Split into command, subcommand, flags, arguments
   - Handle special characters and quotes

2. **Directory Processing**:
   - Full path
   - Parent directory
   - Project root detection (based on .git, etc.)

3. **Temporal Features**:
   - Time of day (hour)
   - Day of week
   - Working hours vs. non-working hours

4. **Contextual Features**:
   - Exit status of previous command (success/failure)
   - Command type (git, docker, etc.)

## Shell Integration

Shell integration will be implemented as plugins:

### Zsh Integration (Primary)

```zsh
# Initialization (to be added to .zshrc)
eval "$(zage init zsh)"

# Key components:
# 1. precmd and preexec hooks for capturing commands
# 2. Automatic prediction display
# 3. Integration with Zsh line editor
```

### Bash Integration (Potential Future Support)

```bash
# Initialization (to be added to .bashrc)
eval "$(zage init bash)"

# Key components:
# 1. PROMPT_COMMAND to record executed commands
# 2. Hooks for capturing exit status
# 3. Prediction integration
```

## Development Guidelines

1. **Code Structure**:
   - Keep modules small and focused
   - Use Rust's type system to enforce correctness
   - Document public APIs

2. **Error Handling**:
   - Use thiserror/anyhow for error types
   - Graceful degradation on failures

3. **Testing**:
   - Unit tests for all components
   - Integration tests for end-to-end flows
   - Test with real-world shell history data

4. **Performance**:
   - Profile database operations
   - Optimize model inference for fast predictions
   - Ensure predictions are generated quickly enough for real-time use

## Current Status and Next Steps

The project currently has shell history parsing implemented. The next steps are:

1. Complete the command collection system
2. Create a simple N-gram prediction model
3. Implement Zsh plugin integration

## Contributing

When contributing to Zage:

1. Ensure code follows Rust style guidelines
2. Add tests for new functionality
3. Update this development document with design decisions
4. Keep the README.md updated with user-facing changes

## Future Directions

After the core functionality is complete, possible extensions include:

- Support for additional shells (Fish, etc.)
- Team sharing of command sequences
- Integration with other development tools
- More advanced sequence prediction models