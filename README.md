# Zage - The Intelligent Shell Sage

Zage (derived from "Z Shell" and "Sage") is a Rust CLI that predicts the next command you're likely to run based on your shell history, working directory, and command context.

## Features

- 🔮 **Command Prediction**: Predicts the most likely next command based on your history
- 🧠 **Context Awareness**: Uses working directory, hostname, and username signals
- 📊 **Sequence Learning**: Mines frequent bigrams/trigrams from your history
- 🔎 **Token Similarity**: Uses SLP-style tokenization to score partial input
- 🪄 **Sensible Defaults**: Works out of the box, but fully configurable

## Installation

Build from source:

```bash
git clone https://github.com/casualjim/zage.git
cd zage
cargo build --release
```

## Quick Start

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

Turso/libSQL (sqld) support:

- `TURSO_DATABASE_URL` (or `LIBSQL_URL`)
- `TURSO_AUTH_TOKEN` (or `LIBSQL_AUTH_TOKEN`)
- `TURSO_LOCAL_REPLICA_PATH` (optional, for local replica sync)

## How It Works

Zage uses statistical ranking and sequence mining to predict commands:

1. **History Collection**: Stores command history with contextual metadata
2. **SLP Tokenization**: Splits commands into normalized tokens for prefix matching
3. **Stats Tables**: Maintains command, transition, and context frequencies + recency
4. **Sequence Mining**: Computes bigram/trigram support, confidence, and lift
5. **Ranking**: Combines recency, frequency, transitions, context, sequence lift, and token similarity

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
