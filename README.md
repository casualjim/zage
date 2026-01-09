# Zage - The Intelligent Shell Sage

Zage (derived from "Z Shell" and "Sage") is a Rust CLI that predicts the next command you're likely to run based on your shell history, working directory, and command context.

## Features

- 🔮 **Command Prediction**: Predicts the most likely next command based on your history
- 🧠 **Context Awareness**: Uses working directory, hostname, username, and session signals
- 📊 **Sequence Learning**: Mines frequent bigrams/trigrams from your history
- 🤖 **ML-Powered Reranking**: Optional GBRT model learns from your behavior patterns
- 🔎 **Token Similarity**: Tree-sitter-based parsing for accurate command analysis
- 🪄 **Sensible Defaults**: Works out of the box, but fully configurable

## Installation

Build from source:

```bash
git clone https://github.com/casualjim/zage.git
cd zage
cargo build --release
```

For development, use the pinned toolchain and tasks:

```bash
mise install
mise build:debug
```

## Quick Start

1. Import shell history:
   ```bash
   zage import --shell zsh
   ```
2. Build stats (and optional sequences):
   ```bash
   zage index --with-sequences
   ```
3. Ask for suggestions:
   ```bash
   zage suggest --current-line "git " --count 5
   ```
4. (Optional) Train the ML model after accumulating 1000+ commands:
   ```bash
   zage train
   ```

## Usage

Import your shell history:

```bash
# zsh (defaults to ~/.zsh_history if no file passed)
zage import --shell zsh

# bash (defaults to ~/.bash_history if no file passed)
zage import --shell bash
```

Build the stats tables:

```bash
zage index --with-sequences
```

Ask for suggestions:

```bash
zage suggest --current-line "git " --count 5
```

### Training the ML Model (Optional)

Once you have 1000+ commands in your history, train the reranking model:

```bash
# Train with default settings (150 epochs, 6 negatives per positive)
zage train

# Or customize training parameters
zage train --epochs 200 --negatives 8 --min-history 1000 --max-samples 30000
```

Check model status:

```bash
zage model-status
```

Reset (delete) the model:

```bash
zage model-reset
```

The model will automatically be used for reranking when available. It learns from your actual command execution patterns to improve suggestion quality.

Zsh autosuggestions (ghost text):

```bash
# 1) Install zsh-autosuggestions with your plugin manager
# 2) Source Zage's zsh integration after your plugins
source /path/to/zage/src/shell_integration/zsh.zsh

# Optional: disable zage's autosuggest backend
# export ZAGE_AUTOSUGGEST_DISABLE=1
```

Antidote users:

```bash
# Add Zage to your bundle list; Antidote will source zage.plugin.zsh
casualjim/zage
```

Completion tuning (optional):

```bash
# Provide session id for session-aware completions
export ZAGE_SESSION_ID=$$

# Aliases are auto-captured when you load the shell integration.
# You can still override manually if needed:
export ZAGE_ALIASES="gst=git status;gco=git checkout"
# or put alias lines in a file:
export ZAGE_ALIAS_FILE="$HOME/.config/zage/aliases"
```

Phase configuration (optional):

Zage learns workflow phases (build/test/deploy/edit/etc.) from your history. You can seed those
phases with a TOML file to make the learning fast and predictable.

Default config path:

- `config/phases.toml` (repo-local)
- or `~/.config/zage/phases.toml`

Override with:

- `ZAGE_PHASES_CONFIG=/path/to/phases.toml`

Pattern rules:

- Patterns are tokenized by the existing shell parser (term-limited).
- Terms can include glob wildcards: `*` and `?`.
- Multiple terms are allowed; quoted terms stay a single unit.
- Flags are **order-independent** (`git commit -m -S` matches `git commit -S -m "msg"`).
- Args are **order-dependent** (matched as a prefix of the argument list).

Example:

```toml
[phases.build]
patterns = [
  "cargo build",
  "make",
  "cmake --build *"
]

[phases.deploy]
patterns = [
  "git push",
  "kubectl apply *"
]
```

Optional: record a single invocation (intended for future shell hooks):

```bash
zage record \
  --command "ls -la" \
  --working-directory "$PWD" \
  --exit-status 0 \
  --start-timestamp 1710000000 \
  --end-timestamp 1710000001 \
  --session-id 12345
```

## Database

Default local path:

- `~/.local/share/zage/zage.db` (Linux)

Override with:

- `ZAGE_DB_PATH=/path/to/zage.db`

## How It Works

Zage uses a two-phase approach combining statistical ranking with machine learning:

### Phase 1: Statistical Candidate Generation
1. **History Collection**: Stores command history with rich contextual metadata (working directory, session, git repo, etc.)
2. **Tree-Sitter Parsing**: Uses tree-sitter-bash and tree-sitter-zsh for accurate command tokenization
3. **Stats Tables**: Maintains command, transition, and context frequencies with recency decay
4. **Sequence Mining**: Computes bigram/trigram support, confidence, and lift from command patterns
5. **Tier-1 Ranking**: Combines recency, frequency, transitions, context, sequence lift, and token similarity

### Phase 2: ML-Powered Reranking (Optional)
6. **Feature Extraction**: Extracts 77 features from candidates (13 base + 64 hash features)
7. **GBRT Model**: Gradient Boosted Regression Trees trained on your actual command choices
8. **Calibration**: Uses Platt scaling and stacking to produce well-calibrated probabilities
9. **Adaptive Learning**: The model learns what you actually run, not just statistical patterns

### Training the Model
After you have 1000+ commands in your history, train the model:
```bash
zage train
```

The model uses pairwise learning: for each command you ran, it samples negative examples (commands you *didn't* run) and learns to distinguish them. It considers:
- Your recent command sequence and workflow patterns
- Session context and repository location
- Time of day and git branch
- Phase detection (build/test/deploy/etc.)

The reranker automatically applies when available, with confidence thresholds and timeouts to ensure fast suggestions.

## Contributing

Contributions are welcome! Feel free to submit pull requests or open issues.

## License

MIT

## Acknowledgments

- Inspired by projects like:
  - [McFly](https://github.com/cantino/mcfly)
  - [Warp](https://github.com/warp-rs/warp)
  - [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)
  - [pxhist](https://github.com/chipturner/pxhist)
- Built with Rust 🦀
