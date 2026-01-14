# Zage - The Intelligent Shell Sage

Zage ("Z Shell" + "Sage") is a Rust CLI that predicts the next command you're likely to run based
on your shell history, working directory, and command context. It can run fully embedded or as a
background server and integrates with zsh-autosuggestions.

## Features

- 🔮 **Command prediction** from your history
- 🧠 **Context-aware**: working directory, hostname, username, session
- 📊 **Sequence learning**: frequent bigrams/trigrams and token sequences
- 🔎 **Accurate parsing** via tree-sitter (bash + zsh)
- 🖥️ **Shell integration**: zsh autosuggest backend + hooks
- 🧰 **Flexible storage**: local, remote (Turso/libsql), or remote replica
- 🔐 **Encryption support** for private history data

## Installation

Build from source:

```bash
git clone https://github.com/casualjim/zage.git
cd zage
cargo build --release
```

Development setup:

```bash
mise install
mise build:debug
```

## Quick Start

1) Import history (automatically trains the online model):

```bash
zage import --shell zsh
```

For a clean model bootstrap, add `--reset-model`:

```bash
zage import --shell zsh --reset-model
```

2) Ask for suggestions:

```bash
zage suggest --current-line "git " --count 5
```

## Shell Integration

### Zsh (autosuggestions + recording)

```bash
# source after zsh-autosuggestions
source /path/to/zage/src/shell_integration/zsh.zsh
```

Antidote users:

```bash
# Add Zage to your bundle list; Antidote will source zage.plugin.zsh
casualjim/zage
```

Zsh options:

- `ZAGE_AUTOSUGGEST_DISABLE=1` disables zage autosuggestions
- `ZAGE_AUTOSUGGEST_ONLY=1` makes zage the only autosuggest strategy
- `ZAGE_ZSH_DEBUG=/path/to/log` writes debug logs

### Bash (requires bash-preexec)

```bash
# load bash-preexec first
source /path/to/bash-preexec.sh
source /path/to/zage/src/shell_integration/bash.sh
```

Bash options:

- `ZAGE_BASH_DEBUG=/path/to/log` writes debug logs

## Command Reference

- `zage import --shell {zsh|bash} [FILE] [--no-index] [--reset-model] [--embedded-db]`
  - Imports shell history and trains the online model.
  - Defaults to `$HISTFILE` when set, otherwise `~/.zsh_history` / `~/.bash_history`.
  - `--reset-model`: Start with a clean model (discards previous training).
  - `--no-index`: Skip rebuilding stats after import.
- `zage index [--with-sequences] [--max-commands N] [--embedded-db]`
  - Rebuilds statistics and optionally sequence patterns.
- `zage sequences analyze [--min-support N] [--min-confidence F] [--min-lift F] [--max-len N]`
  - Analyzes command sequence patterns.
- `zage yank "command" [--match-expanded] [--no-sequences] [--embedded-db]`
  - Removes matching history entries, then rebuilds stats.
- `zage suggest [--current-line "prefix"] [--count N] [--recent-limit N]
  [--no-sequences] [--autosuggest] [--completion-format plain|zsh] [--show-scores] [--timeout 2s]`
  - With `--current-line`, returns token completions for the active token.
  - Without it, returns full command suggestions ranked by the online model.
  - `--autosuggest`: Forces full-line output for autosuggest backends.
  - `--timeout`: Accepts human durations like `500ms`, `2s`, `1m` (server mode only).
- `zage model status`
  - Shows online model statistics (embedding count, training examples, etc.).
- `zage model reset`
  - Resets the online model (clears all learned embeddings).
- `zage server`
  - Runs the suggestion server in foreground mode.
- `zage service install|uninstall`
  - Installs/uninstalls the background service (systemd/launchd).
- `zage record ...`
  - Internal command used by shell hooks to record command execution.
  - Requires `--shell`, `--command`, `--working-directory`, `--exit-status`, `--start-timestamp`, `--end-timestamp`.
  - Updates the online model with each recorded command.

## Embedded vs Server Mode

Zage can run embedded (default) or via a background server.

- Embedded: direct DB access; easiest to start.
- Server: uses a Unix socket; best for low-latency suggestions and shared state.

Config or force it per command:

- Config: `backend = "embedded"` or `backend = "server"`
- Override: `--embedded-db` on commands

Server details:

- Socket path:
  - `ZAGE_SOCKET_PATH=/custom/zage.sock`
  - Defaults: `/tmp/zage.sock` (macOS), `$XDG_RUNTIME_DIR/zage.sock` or `/tmp/zage.sock` (Linux)
- Pool size: `ZAGE_DB_POOL_SIZE=30` (default 30)
- Logs: `ZAGE_LOG=info|debug|trace`

## Configuration

Zage can read a config file to set the default backend and database connection.

Load order:

- `ZAGE_CONFIG=/path/to/zage.toml`
- `config/zage.toml` (repo-local)
- `~/.config/zage/config.toml`

Example:

```toml
backend = "embedded" # or "server"

[db]
type = "local"       # or "remote" or "remote_replica"
path = "/path/to/zage.db" # local/replica only

# Remote libsql/sqld connection (optional for local)
# url = "libsql://your-host"
# auth_token = "your-token" # or set ZAGE_DB_AUTH_TOKEN

# Local at-rest encryption (requires libsql encryption feature)
# encryption_key = "super-secret"
# encryption_cipher = "aes256cbc"

# Remote encryption context (base64 key sent to server)
# remote_encryption_key = "base64-encoded-key" # or set ZAGE_DB_REMOTE_ENCRYPTION_KEY

# Remote replica sync tuning (remote_replica only)
# sync_interval_ms = 1000
```

Environment overrides:

- `ZAGE_DB_PATH` (local DB path)
- `ZAGE_DB_AUTH_TOKEN`
- `ZAGE_DB_ENCRYPTION_KEY`
- `ZAGE_DB_REMOTE_ENCRYPTION_KEY`
- `ZAGE_DB_SYNC_INTERVAL_MS`
- `ZAGE_SESSION_ID`
- `ZAGE_COMPLETION_FORMAT=plain|zsh`
- `ZAGE_ALIASES` or `ZAGE_ALIAS_FILE`
- `ZAGE_HOSTNAME`
- `ZAGE_SUGGEST_TIMEOUT` (human duration, server mode only)

## Database and Encryption

Default local path (Linux): `~/.local/share/zage/zage.db`

- Local at-rest encryption: set `encryption_key` (and optionally `encryption_cipher`).
- Remote encryption: set `remote_encryption_key` (base64 key).

## Multi-machine sync with Turso (recommended)

If you use multiple machines, point Zage at a Turso (libsql) database. Your shell history is
private by nature, so **enable encryption** whenever you use a remote database.

Pick a mode:

- `remote`: connect to Turso directly (no local replica).
- `remote_replica`: keep a local replica and sync changes to the remote.
  This keeps reads fast and still shares history across machines.

Example: remote (no local replica)

```toml
backend = "embedded"

[db]
type = "remote"
url = "libsql://your-host"
auth_token = "your-token"

# Required for remote encryption
remote_encryption_key = "base64-encoded-key"
```

Example: remote replica (recommended for laptops/desktops)

```toml
backend = "embedded"

[db]
type = "remote_replica"
path = "/path/to/zage.db"
url = "libsql://your-host"
auth_token = "your-token"
sync_interval_ms = 1000

# Encrypt the local replica at rest
encryption_key = "local-secret"
encryption_cipher = "aes256cbc"

# Encrypt remote traffic and server-side storage
remote_encryption_key = "base64-encoded-key"
```

Key generation tip (base64, 32 bytes):

```bash
openssl rand -base64 32
```

## How It Works

Zage uses an **online two-tower embedding model** that learns continuously from your shell history:

1) **Candidate generation**
   - Parses commands with tree‑sitter (bash/zsh) into structured tokens.
   - Generates candidates from transitions, context stats, and sequence patterns.
   - Applies hard constraints (prefix matching, syntax validity, deduplication).

2) **Online model ranking**
   - Learns a **context embedding** from workspace, directory, exit status, and recent commands.
   - Learns a **command embedding** from normalized command structure (head, flags, args).
   - Scores candidates by dot product plus calibrated priors (frecency, sequences).
   - Updates embeddings **online** after every command execution using negative sampling.
   - Uses replay buffers and confidence gates to prevent catastrophic forgetting.

The model trains incrementally as you work, adapting to your workflows without offline batch training. See [`docs/online_next_command_prediction.md`](docs/online_next_command_prediction.md) for the full design.

## Advanced Features

### Phase Configuration

Zage learns workflow phases (build/test/deploy/edit/etc.) from your history. You can seed those
phases with a TOML file to make the learning fast and predictable.

Default config path:

- `config/phases.toml` (repo-local)
- `~/.config/zage/phases.toml`

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

### Aliases

Zage can learn from aliases and their expansions:

- `ZAGE_ALIASES="gst=git status;gco=git checkout"`
- `ZAGE_ALIAS_FILE="$HOME/.config/zage/aliases"`

### Manual recording

Shell hooks already call this, but you can record a single invocation manually:

```bash
zage record \
  --shell "zsh" \
  --command "ls -la" \
  --working-directory "$PWD" \
  --exit-status 0 \
  --start-timestamp 1710000000 \
  --end-timestamp 1710000001 \
  --session-id 12345
```

## Troubleshooting

- **"Suggest server unavailable"**: Start the server (`zage server`) or install the service
  (`zage service install`), or switch to embedded mode (`backend = "embedded"` or `--embedded-db`).
- **No suggestions**: Run `zage import --shell zsh` to import your history and train the model.
- **Poor suggestions**: The online model needs training data. Run `zage model status` to check
  training progress. Consider `zage import --reset-model` for a clean start.
- **Autosuggest not working**: Ensure zsh-autosuggestions is loaded before the Zage script, and
  `ZAGE_AUTOSUGGEST_DISABLE` is not set to `1`.

## License

MIT

## Acknowledgments

- Inspired by projects like:
  - [McFly](https://github.com/cantino/mcfly)
  - [Warp](https://github.com/warp-rs/warp)
  - [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)
  - [pxhist](https://github.com/chipturner/pxhist)
- Built with Rust 🦀
