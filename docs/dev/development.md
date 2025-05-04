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
  --hostname <n>      Override hostname for import
  --username <n>      Override username for import
  --shell <SHELL>     Shell type (bash or zsh); defaults to $SHELL env var

# Train prediction model
zage train [OPTIONS]

Options:
  --model <MODEL>     Model type to train (ngram, default: ngram)
  --n <N>             N value for N-gram model (default: 3)
  --limit <LIMIT>     Limit number of history entries to use (default: all)

# Predict next command
zage predict [OPTIONS]

Options:
  --model <MODEL>     Model to use for prediction (ngram, default: ngram)
  --n <N>             Number of predictions to return (default: 5)
  --context <DIR>     Use directory context for prediction (default: true)

# Show model statistics
zage stats [OPTIONS]

Options:
  --model <MODEL>     Model to show statistics for (ngram, default: ngram)
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

- [x] Implement N-gram model for baseline predictions
  - [x] Core N-gram implementation with frequency tracking
  - [x] Directory-aware context for improved predictions
  - [x] Serialization and database persistence
  - [x] Comprehensive test suite for model validation
- [x] Add Markov chain model with context awareness
- [x] Zsh plugin integration  # Shell hook script with debug logging and silent background recording
- [ ] Initial sequence detection algorithm

### Phase 3: LSTM Implementation

- [ ] Command embedding generation
- [ ] Feature extraction from commands and context
- [ ] LSTM model implementation using tch-rs
- [ ] Training pipeline
- [ ] Prediction pipeline

### Phase 4: Advanced Features

- [x] Context enhancement with directory, exit status
  - [x] Directory-aware predictions implemented in N-gram model
  - [ ] Exit status awareness for improved context
- [ ] Time-based patterns detection
- [ ] Multi-terminal awareness
- [ ] Performance optimizations
- [x] Bash plugin integration (if feasible)  # Added `bash.sh` using bash-preexec hooks

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

CREATE TABLE models (
    model_type TEXT PRIMARY KEY,
    model_data BLOB NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
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
echo "Example command"
```

When timestamps are enabled (`HISTTIMEFORMAT`), the format becomes:

```text
#1430458594
echo "Example command"
```

### Async Implementation

This project uses Tokio for asynchronous operations:

- Database operations are performed asynchronously
- Shell integration uses async channels for communication
- Model training and prediction run in background tasks

## N-gram Model Implementation

The N-gram model is the first prediction model implemented in Zage. It provides a solid baseline for command prediction based on command history and working directory context.

### Model Overview

The N-gram model works by:

1. **Analyzing command sequences**: It tracks sequences of N commands and calculates the frequency of each command following a specific context of N-1 commands.

2. **Directory-aware predictions**: The model maintains separate frequency tables for different working directories, allowing it to make context-specific predictions.

3. **Efficient storage**: The model uses BTreeMap for storing frequency data, which provides a good balance between performance and memory usage for potentially large datasets.

### Serialization

The model is serialized to JSON and stored in the SQLite database, allowing it to persist between sessions. The serialization process handles complex data structures like nested maps with non-string keys by using a custom serialization approach.

### Prediction Process

When predicting the next command, the model:

1. Takes the most recent N-1 commands as context
2. Looks up this context in both the global frequency table and the directory-specific table
3. Combines and ranks predictions based on frequency, giving higher weight to directory-specific matches
4. Returns the top K predictions sorted by probability

### Future Enhancements

While the current N-gram model provides good baseline predictions, future enhancements will include:

- Incorporating command exit status for better context awareness
- Time-based weighting to prioritize more recent patterns
- Integration with more sophisticated models like Markov chains and LSTMs

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
