# Zage Development Plan

This document outlines the development plan and architecture for Zage, an intelligent shell command prediction plugin built in Rust.

## Project Overview

Zage aims to predict the next shell command a user is likely to run based on:

- Command history
- Current working directory
- Recent command sequences
- Exit status of previous commands
- Time and contextual patterns

Unlike simple history search tools, Zage uses advanced models (starting with N-gram, with Markov and LSTM planned) to identify and learn complex command sequences within different contexts, working as a seamless plugin for Zsh and Bash.

## Current Project Structure

```text
zage/
├── src/
│   ├── config.rs           # Configuration management
│   ├── db.rs               # Database connection and operations
│   ├── err.rs              # Error types and handling
│   ├── lib.rs              # Library entry point
│   ├── main.rs             # CLI entry point and logic
│   ├── model/              # ML model implementations
│   │   ├── context.rs      # Context struct for predictions
│   │   ├── markov.rs       # Markov model implementation
│   │   ├── mod.rs
│   │   ├── ngram.rs        # N-gram model (default)
│   │   ├── sequence.rs     # Sequence detection
│   │   └── sequence_context.rs # Sequence context logic
│   ├── shell_history/      # Shell history parsing
│   │   ├── bash.rs         # Bash history parser
│   │   ├── mod.rs
│   │   └── zsh.rs          # Zsh history parser
│   └── shell_integration/  # Shell integration scripts
│       ├── bash.sh         # Bash integration script
│       └── zsh.zsh         # Zsh integration script
└── docs/
    └── dev/
        └── development.md  # This file
```

## Dependencies

Zage uses the following main dependencies:

- `clap` for CLI parsing
- `color-eyre`, `thiserror`, `miette` for error handling
- `tracing` for logging
- `serde`, `serde_json`, `bstr` for serialization
- `rusqlite` for SQLite database
- `uuid`, `dirs`, `rand`, `itertools`, `jiff`, `sqlite-vss` for various utilities

See `Cargo.toml` for full list and versions. Rust 2024 edition is required.

## CLI Usage

Zage provides a command-line interface with the following subcommands (see `zage --help` for all options):

```bash
# Import shell history
zage import [--file <FILE>] [--hostname <HOST>] [--username <USER>] [--shell <SHELL>]

# Train prediction model
zage train [--model_type <MODEL>] [--n <N>] [--max_commands <LIMIT>] [--use_context <BOOL>]

# Predict next command
zage predict [--count <N>] [--model_type <MODEL>] [--n <N>] [--show_probability]

# Show model statistics
zage stats [--model_type <MODEL>] [--n <N>]

# Analyze and store frequent command sequences
zage analyze-sequences [--min_support <N>] [--min_length <N>]
```

- `model_type` can be `ngram` (default) or `markov` (experimental).
- `n` defaults to 2 for N-gram.
- `use_context` enables context-aware predictions (directory, hostname, username, exit status).

## Development Phases

### Phase 1: Foundation (Complete)

- Project setup, CLI, error handling, config, shell history parsing, SQLite schema, command collection.

### Phase 2: Simple Prediction (Mostly Complete)

- [x] N-gram model with context (directory, host, user, exit status)
- [x] Markov model (experimental)
- [x] Zsh and Bash integration scripts
- [x] Sequence detection (complete)

### Phase 3: LSTM Implementation (Planned)

- [ ] Command embedding generation
- [ ] Feature extraction (including command output)
- [ ] LSTM model (using `tch-rs`)
- [ ] Training/prediction pipeline

### Phase 4: Advanced Features (Planned)

- [ ] Output awareness, time-based patterns, multi-terminal, performance optimizations
- [ ] BitNet model (experimental)

## Database Schema

Schema is up-to-date and matches the codebase. See `src/db.rs` for implementation.

## Testing

- Unit tests and integration tests are provided for all core models and features.
- Run tests with:

```bash
cargo test
```

## Shell Integration

- Zsh and Bash integration scripts are in `src/shell_integration/` as `zsh.zsh` and `bash.sh`.
- These scripts hook into the shell to record command execution and output.

## Coding Style & Conventions

- Idiomatic Rust (2024 edition)
- Error handling with `Result`, `color-eyre`, `thiserror`
- Modular code organization
- Use of environment variables for config
- Comprehensive Rustdoc and inline comments

## Contributing

- Ensure you have the correct Rust version (`rustup update stable`)
- Run tests before submitting PRs
- Follow the module structure and error handling conventions
- Document new features and update this file as needed

## Data Flow

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

### Phase 1: Foundation

- [x] Project setup with CLI framework
- [x] Basic error handling
- [x] Configuration system
- [x] Shell history parsing (Bash, Zsh)
- [x] SQLite database schema and operations  # Completed: includes schema init, insert_invocation, and tests
- [x] Command collection system with CLI import command

### Phase 2: Simple Prediction (Current)

- [x] Implement N-gram model for baseline predictions
  - [x] Core N-gram implementation with frequency tracking
  - [x] Context-aware predictions (directory, hostname, username, exit status)
  - [x] Serialization and database persistence
  - [x] Refactored JSON serialization format to entry lists to avoid map-key issues
  - [x] Comprehensive test suite for model validation
- [x] Add Markov chain model with context awareness
- [x] Zsh plugin integration  # Shell hook script with debug logging and silent background recording
- [x] Initial sequence detection algorithm
  - [x] Define sequence detection objectives and requirements
  - [x] Design multi-dimensional context representation for sequences
  - [x] Implement sequence scoring and threshold criteria
  - [x] Integrate detection algorithm with existing models and contexts
  - [x] Develop unit tests and benchmarking suite for sequence detection
  - [x] Validate detection performance on real shell history samples

### Phase 3: LSTM Implementation

- [ ] Command embedding generation
- [ ] Feature extraction from commands and context
  - [ ] Command output (stdout/stderr) feature extraction
- [ ] LSTM model implementation using tch-rs
- [ ] Training pipeline
- [ ] Prediction pipeline

### Phase 4: Advanced Features

- [x] Context enhancement with directory, exit status
  - [x] Directory-aware predictions implemented in N-gram model
  - [x] Exit status awareness for improved context
- [ ] Command output awareness (stdout/stderr)
- [ ] Time-based patterns detection
- [ ] Multi-terminal awareness
- [ ] Performance optimizations
- [x] Bash plugin integration (if feasible)  # Added `bash.sh` using bash-preexec hooks
- [ ] BitNet model exploration (alternative to LSTM)

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
    session_id INTEGER,
    stdout_summary TEXT,
    stderr_summary TEXT,
    has_output BOOLEAN
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

CREATE TABLE command_outputs (
    history_id TEXT PRIMARY KEY,
    stdout BLOB,
    stderr BLOB,
    FOREIGN KEY(history_id) REFERENCES shell_history(id)
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

## N-gram Model Implementation

The N-gram model is the first prediction model implemented in Zage. It provides a solid baseline for command prediction based on command history and working directory context.

### Model Overview

The N-gram model works by:

1. **Analyzing command sequences**: It tracks sequences of N commands and calculates the frequency of each command following a specific context of N-1 commands.

2. **Context-aware predictions**: The model maintains separate frequency tables for different working directories, hostnames, usernames, and exit statuses, allowing it to make context-specific predictions.

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

## Command Output Capture Implementation

Incorporating stdout and stderr from commands can significantly enhance prediction accuracy, as users often run the same commands when they see specific outputs (error messages, successful results, etc.).

### Capture Approach

1. **Shell Integration**:

   ```bash
   # Example of command output capture in shell hooks
   command_exec_hook() {
     local cmd="$1"
     local tmp_stdout=$(mktemp)
     local tmp_stderr=$(mktemp)
     
     # Capture stdout and stderr while still displaying them
     eval "$cmd" > >(tee "$tmp_stdout") 2> >(tee "$tmp_stderr" >&2)
     local exit_status=$?
     
     # Process the captured output
     zage_record_output "$cmd" "$tmp_stdout" "$tmp_stderr" "$exit_status"
     
     # Cleanup
     rm "$tmp_stdout" "$tmp_stderr"
   }
   ```

2. **Rust Implementation**:

   ```rust
   // Function to process and store command outputs
   pub async fn store_command_output(
       db: &DB,
       history_id: &str,
       stdout_path: &Path,
       stderr_path: &Path
   ) -> Result<()> {
       // Read files
       let stdout = tokio::fs::read(stdout_path).await?;
       let stderr = tokio::fs::read(stderr_path).await?;
       
       // Generate summaries
       let stdout_summary = summarize_output(&stdout)?;
       let stderr_summary = summarize_output(&stderr)?;
       
       // Update history with summaries
       let has_output = !stdout.is_empty() || !stderr.is_empty();
       db.execute(
           "UPDATE shell_history SET stdout_summary = ?, stderr_summary = ?, has_output = ? WHERE id = ?",
           params![stdout_summary, stderr_summary, has_output, history_id],
       ).await?;
       
       // Store full output if significant
       if has_output {
           db.execute(
               "INSERT INTO command_outputs (history_id, stdout, stderr) VALUES (?, ?, ?)",
               params![history_id, stdout, stderr],
           ).await?;
       }
       
       Ok(())
   }
   
   // Function to summarize output (limit length, extract key patterns)
   fn summarize_output(output: &[u8]) -> Result<String> {
       // Implementation depends on your specific needs
       // Could use regex patterns to extract errors, important results, etc.
       // Or just truncate to a reasonable length
       let output_str = String::from_utf8_lossy(output);
       Ok(output_str.chars().take(200).collect())
   }
   ```

### Feature Extraction

1. **Basic Text Features**:
   - Error message presence/absence
   - Output length
   - Common patterns (file lists, error codes, etc.)

2. **Advanced Features**:
   - Text embeddings of output using a small language model
   - Classification of output types (error, normal output, empty, etc.)
   - Pattern matching for specific command types

## BitNet Model Exploration

BitNet with ternary weights (-1, 0, +1) offers an efficient alternative to LSTMs for command prediction, potentially providing similar accuracy with lower resource usage.

### Implementation Considerations

1. **Model Architecture**:
   - Custom BitLinear layers replacing standard linear layers
   - Ternary weight quantization during forward pass
   - Efficient matrix operations that avoid multiplication

2. **Integration Strategy**:

   ```rust
   pub struct BitNetModel {
       // Model architecture fields
       weights: Vec<TernaryWeights>,
       // Other model parameters
   }
   
   impl BitNetModel {
       pub fn new(config: ModelConfig) -> Self {
           // Initialize model with ternary weights
       }
       
       pub fn predict(&self, features: &[Feature]) -> Vec<Prediction> {
           // Implement efficient prediction using ternary weights
       }
       
       pub fn train(&mut self, training_data: &[TrainingExample]) -> Result<()> {
           // Train model with ternary weight constraints
       }
   }
   ```

3. **Comparison Benchmarks**:
   - Memory usage: BitNet vs. LSTM
   - Prediction speed: BitNet vs. LSTM
   - Prediction accuracy: BitNet vs. LSTM
   - Training time: BitNet vs. LSTM

### Phased Approach

1. Implement and test LSTM model with stdout/stderr features first
2. Develop BitNet model in parallel as an experimental feature
3. Benchmark both approaches and determine the best fit for Zage

## Future Considerations

- Containerized deployment for CI/CD
- Cross-platform support (Windows, macOS, Linux)
- Plugin system for custom prediction strategies
- Privacy controls and sensitive command filtering
- Remote synchronization between machines
- Pre-trained models for common command patterns

### Testing and Benchmarking

#### Unit Tests

Run the unit tests for sequence detection:

```bash
cargo test -- --nocapture
```

#### Benchmarks

Run the benchmarks using Criterion:

```bash
cargo bench
```
