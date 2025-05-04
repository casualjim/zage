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

```rust
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

```rust
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

## CLI Usage

Zage provides a command-line interface with the following subcommands:

```bash
# Import shell history
zage import [OPTIONS]

Options:
  --file <FILE>       Path to history file (defaults to $HISTFILE env var)
  --hostname <NAME>   Override hostname for import
  --username <NAME>   Override username for import
  --shell <SHELL>     Shell type (bash or zsh); defaults to $SHELL env var
```

## Implementation Plan

The development will proceed in phases, each building on the previous:

### Phase 1: Foundation (Current)

- [x] Project setup with CLI framework
- [x] Basic error handling
- [x] Configuration system
- [x] Shell history parsing (Bash, Zsh)
- [x] SQLite database schema and operations  # Completed: includes schema init, insert_invocation, and tests
- [x] Command collection system with CLI import command

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
CREATE TABLE shell_history (
    id TEXT PRIMARY KEY,
    command TEXT NOT NULL,
    shellname TEXT NOT NULL,
    working_directory TEXT,
    hostname TEXT,
    username TEXT,
    exit_status INTEGER,
    start_unix_timestamp INTEGER,
    end_unix_timestamp INTEGER,
    session_id INTEGER
);

CREATE UNIQUE INDEX idx_shell_history_unique ON shell_history (
    command,
    shellname,
    working_directory,
    hostname,
    username,
    exit_status,
    start_unix_timestamp,
    end_unix_timestamp,
    session_id
);
```

Future tables for sequence detection and model training:

```sql
CREATE TABLE sequences (
    id TEXT PRIMARY KEY,
    name TEXT,
    context TEXT,
    detected_count INTEGER DEFAULT 1
);

CREATE TABLE sequence_commands (
    sequence_id TEXT,
    command_id TEXT,
    position INTEGER,
    PRIMARY KEY (sequence_id, position),
    FOREIGN KEY (sequence_id) REFERENCES sequences(id),
    FOREIGN KEY (command_id) REFERENCES shell_history(id)
);
```

## Development Notes

### Shell History Format

#### Zsh History Format

The Zsh history format consists of lines with the following structure:

```text
: <timestamp>:<elapsed seconds>;<command>
```

For example:

```text
: 1610000000:0;echo hello
```

#### Bash History Format

The Bash history format can have two forms:

1. Simple commands:

```text
echo hello
```

1. With timestamps (when HISTTIMEFORMAT is set):

```text
#1610000000
echo hello
```

### Async Implementation

This project uses Tokio for asynchronous operations:

- Database operations are performed asynchronously
- Shell integration uses async channels for communication
- Model training and prediction run in background tasks

## Testing Strategy

- Unit tests for individual components
- Integration tests for end-to-end functionality
- Fuzzing tests for history parsing to handle edge cases
- Performance benchmarks for critical paths

## Future Considerations

- Containerized deployment for CI/CD
- Cross-platform support (Windows, macOS, Linux)
- Plugin system for custom prediction strategies
- Privacy controls and sensitive command filtering
- Remote synchronization between machines