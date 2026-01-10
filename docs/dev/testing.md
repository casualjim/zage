# Testing and Simulation Infrastructure for Zage

This document outlines the testing and simulation infrastructure needed for the Zage project, which aims to recreate Warp terminal's intelligent command prediction functionality as a ZSH plugin.

## Project Overview

Zage is a command prediction system with:

1. **Multiple prediction models**:
   - Markov Chain model for command transition prediction
   - N-gram model for sequence-based predictions
   - Context-aware predictions considering working directory, hostname, and command exit status

2. **Persistence and learning**:
   - SQLite database for model storage
   - History import from existing shell files
   - Continuous learning from new command executions

## Implementation Approach

We're using a straightforward approach for implementing Warp-like command prediction by:

1. **Leveraging zsh-autosuggestions plugin** as the UI/UX foundation
   - It already handles displaying suggestions inline
   - Manages accepting suggestions with keyboard shortcuts
   - Handles styling of suggestions

2. **Creating a custom strategy for zsh-autosuggestions** that calls the Rust binary

```zsh
_zsh_autosuggest_strategy_zage() {
    local suggestion=$(zage predict --current-line "$1" --pwd "$PWD" 2>/dev/null)
    [[ -n "$suggestion" ]] && echo "$suggestion"
}
```

3. **Configuring zsh-autosuggestions** to use this strategy

```zsh
ZSH_AUTOSUGGEST_STRATEGY=(zage)
```

## Testing Infrastructure Requirements

### 1. Command History Datasets

To effectively test the prediction models, we need diverse and representative command history datasets:

- **Real-world shell history files**:
  - ZSH history files from various users and use cases
  - Bash history files to test compatibility
  - History files with different formats and encodings

- **Synthetic datasets**:
  - Generated datasets with specific patterns to test corner cases
  - Datasets with controlled frequency distributions
  - Datasets with command sequences of different complexity

- **Project-specific datasets**:
  - Command sequences typical in different development environments (Node.js, Rust, Python, etc.)
  - DevOps/cloud management command patterns
  - System administration command sequences

### 2. Context Simulation

Since context is crucial for accurate predictions, we need to simulate different working environments:

- **Directory Structure Mock-ups**:
  - Create mock project directories that mimic real-world scenarios
  - Include various project types (web, data science, DevOps, etc.)
  - Simulate navigational patterns between directories

- **Environment Variables**:
  - Test with different sets of environment variables
  - Simulate different shell configurations

- **Git Repository Context**:
  - Mock repos with different branch structures
  - Simulate common git workflow patterns

### 3. Performance Benchmarks

For measuring and optimizing performance:

- **Response Time Measurements**:
  - Time from input to suggestion display
  - Track p50, p90, p99 latencies
  - Measure cold start vs. warm suggestion times

- **Prediction Quality Metrics**:
  - Precision/recall for suggestions
  - Acceptance rate (how often suggestions are used)
  - Context relevance scores

- **Resource Usage**:
  - CPU and memory utilization
  - Database access patterns and efficiency
  - Startup and initialization overhead

### 4. User Experience Simulation

To validate the user interface aspects:

- **Interactive Typing Simulation**:
  - Script-based typing with realistic timing
  - Different typing speeds and patterns
  - Interruptions and corrections

- **Command Acceptance Patterns**:
  - Different acceptance behaviors (tab, right arrow, etc.)
  - Partial suggestion acceptance
  - Suggestion rejection patterns

### 5. Integration Testing

To ensure smooth operation with ZSH and other components:

- **Shell Plugin Compatibility**:
  - Test with different versions of zsh-autosuggestions
  - Compatibility with other common ZSH plugins
  - Cross-shell behavior (ZSH vs. Bash)

- **Terminal Emulator Compatibility**:
  - Testing across different terminal emulators (iTerm2, Alacritty, Kitty, etc.)
  - Rendering and display compatibility

## Test Automation

### Automated Test Suites

1. **Unit Tests**:
   - Prediction algorithm correctness
   - Context parsing and handling
   - Database operations

2. **Functional Tests**:
   - End-to-end prediction flows
   - History import/export
   - Configuration management

3. **Simulation-based Tests**:
   - Automated typing and suggestion scenarios
   - Context switching simulations

### CI/CD Integration

- GitHub Actions workflow for automated testing
- Performance regression detection
- Compatibility matrix testing

## Evaluation Metrics

Key metrics to track for assessing Zage's effectiveness:

1. **Prediction Accuracy**:
   - Percentage of accepted suggestions
   - Edit distance between predictions and actual commands
   - Ranking quality (how often the best suggestion comes first)

2. **Performance Metrics**:
   - Average suggestion generation time
   - Memory usage over time
   - Database size growth rate

3. **User Experience Metrics**:
   - Time saved by using suggestions
   - Keystrokes saved
   - Learning curve (improvement in suggestion relevance over time)

## Testing Tools

Potential tools and frameworks to leverage:

- **Rust Testing**:
  - Criterion for benchmarking
  - Proptest for property-based testing
  - Mockall for mocking dependencies

- **Shell Testing**:
  - Bats (Bash Automated Testing System)
  - ShellCheck for script quality

- **User Experience Testing**:
  - xdotool for simulating keyboard input
  - ttyrec/asciinema for recording terminal sessions
  - Custom typing simulator

## Next Steps

1. Implement a basic test harness that can replay shell history files through the prediction engine
2. Create a small set of representative test datasets
3. Set up CI pipeline with basic unit and integration tests
4. Develop performance benchmarking suite
5. Create a simulation environment for UX testing
