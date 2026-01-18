# Zage 🧙 — AI-powered shell suggestions

Zage is a small Rust CLI that **predicts what you’re about to type next** in your shell.
It learns from your history and context (repo, folder, exit status, recent commands) using a lightweight **online embedding model** stored in your local database.

- ✅ Works offline and stays local by default (no cloud required)
- ⚡ Learns your workflows per-repo ("in this project I usually run X after Y")
- 🧠 Uses **AI** to generalize beyond exact string matches (flags reorder, similar commands, similar args)

## Why you’ll want this

- 🔮 **Fewer keystrokes**: suggestions match what you *actually* do in *this* directory/project.
- 🧩 **Subtle improvements**: not just “history search” — it learns patterns and generalizes.
- 🏠 **Privacy-friendly**: everything can be local (or you can opt into remote sync + encryption).

## Quick demo (no shell setup yet)

```bash
# 1) Import history and train the model (pick your shell)
zage import --shell zsh

# 2) Ask for suggestions
zage suggest --count 5

# 3) Ask for token completions for the *current word*
zage suggest --current-line "git " --count 5
```

## Setup cheat-sheet (for beginners)

1) Figure out your shell:

```bash
echo "$SHELL"
```

2) Edit the matching file:

- **bash** → `~/.bashrc`
- **zsh** → `~/.zshrc`

Then follow the **Bash** or **Zsh** setup section below and restart your shell (or open a new terminal tab).

## Installation

Build from source:

```bash
git clone https://github.com/casualjim/zage.git
cd zage
cargo build --release

# optional: install `zage` into ~/.cargo/bin
cargo install --path .
```

Make sure `~/.cargo/bin` is on your PATH (restart your terminal after installing).

Dev setup:

```bash
mise install
mise build:debug
```

## Setup (pick one): simplest vs best experience

Zage can run in two modes:

- **Embedded (simplest)**: no daemon; commands talk directly to the DB.
- **Server/service (best experience)**: a background daemon handles requests and updates the online model continuously.

### Option A — Simplest setup (embedded; no daemon)

Create a config file:

```bash
mkdir -p ~/.config/zage
cat > ~/.config/zage/config.toml <<'TOML'
backend = "embedded"
TOML
```

Now `zage suggest` and `zage import` work without installing a service.

> Note: in embedded mode, the online model is trained during `zage import` (and other explicit training steps), not continuously after every command.

### Option B — Best experience (install background service)

Install the user service (systemd on Linux, launchd on macOS):

```bash
zage service install
```

If you prefer foreground (for debugging):

```bash
zage server
```

> In server/service mode, Zage can update the online model continuously as your shell records commands.

## Shell integration (recording + autosuggestions)

Once Zage is installed, add a small snippet to your **shell config** so Zage can:

- record commands as you run them (this is how it learns)
- show suggestions (zsh only, via zsh-autosuggestions)

### Zsh (recommended: zsh-autosuggestions + Zage)

Zsh is where Zage shines the most, because it can plug into **zsh-autosuggestions**.

#### Why this is better than zsh-autosuggestions alone

zsh-autosuggestions by itself mostly suggests from:

- your history (often global)
- simple matching

Zage adds a smarter backend that uses **AI-style embeddings + context**, so suggestions are more likely to match what you do *in this repo/dir*, and it can generalize a bit (flags reordered, similar commands).

#### Setup

1) Ensure you have zsh-autosuggestions installed (pick any method you like):

- https://github.com/zsh-users/zsh-autosuggestions

2) In `~/.zshrc`, load zsh-autosuggestions first, then load Zage:

```bash
# 1) zsh-autosuggestions
# Install it first: https://github.com/zsh-users/zsh-autosuggestions
# Then source it here (example path; yours may differ):
# source /path/to/zsh-autosuggestions/zsh-autosuggestions.zsh

# 2) Zage (adjust to where you cloned it)
ZAGE_DIR="$HOME/src/zage"
source "$ZAGE_DIR/src/shell_integration/zsh.zsh"
```

3) Restart your terminal (or `exec zsh`).

Zsh options:

- `ZAGE_AUTOSUGGEST_DISABLE=1` disables zage autosuggestions
- `ZAGE_AUTOSUGGEST_ONLY=1` makes zage the only autosuggest strategy
- `ZAGE_ZSH_DEBUG=/path/to/log` writes debug logs

Antidote users:

```bash
# Add Zage to your bundle list; Antidote will source zage.plugin.zsh
casualjim/zage
```

### Bash (beginner-friendly setup)

Many Linux distros default to **bash**, so here’s the shortest reliable setup.

#### 1) Install bash-preexec

Zage’s bash integration relies on **bash-preexec** to get `preexec`/`precmd` hooks (bash doesn’t provide those by default).

- https://github.com/rcaloras/bash-preexec

After installing it, you should have a `bash-preexec.sh` you can `source`.

Example (git clone):

```bash
mkdir -p ~/.config
git clone https://github.com/rcaloras/bash-preexec ~/.config/bash-preexec

# then in ~/.bashrc:
# source "$HOME/.config/bash-preexec/bash-preexec.sh"
```

#### 2) Add this to `~/.bashrc`

```bash
# 1) bash-preexec (adjust path)
source /path/to/bash-preexec.sh

# 2) Zage (adjust to where you cloned it)
ZAGE_DIR="$HOME/src/zage"
source "$ZAGE_DIR/src/shell_integration/bash.sh"
```

#### 3) Restart your terminal (or `exec bash`)

Bash options:

- `ZAGE_BASH_DEBUG=/path/to/log` writes debug logs

## Common commands

```bash
# Import your history (first-time setup / re-train)
zage import --shell zsh

# Ask for suggestions
zage suggest --count 5

# Inspect training progress
zage model status

# Run the always-on daemon (optional)
zage service install   # or: zage server
```

## Make it feel “AI-powered” (what’s actually happening)

Zage isn’t calling a hosted LLM.
Instead it maintains a small **online embedding model** (think: recommender system) that learns from:

- your recent commands (short context window)
- your current repo/folder
- exit status (failed commands change what you do next)
- and tokenized structure (command head, flags, normalized args)

This is why suggestions can improve in subtle ways over time: it’s learning patterns, not just replaying strings.

## Configuration

Zage can read a config file to set the default backend and database connection.

Load order (highest priority first):

- CLI flags
- Environment variables
- `--config-file /path/to/zage.toml`
- `ZAGE_CONFIG=/path/to/zage.toml`
- `~/.config/zage/config.toml`

Example:

```toml
backend = "embedded" # or "server"

[db]
kind = "local"       # or "remote" or "remote_replica"
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
   - Updates embeddings **online** in server/service mode (and during `zage import`) using negative sampling.
   - Uses replay buffers and confidence gates to prevent catastrophic forgetting.

The model trains incrementally as you work, adapting to your workflows without offline batch training. See [`docs/online_next_command_prediction.md`](docs/online_next_command_prediction.md) for the full design.

## Advanced Features

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

### I don’t know if I’m using zsh or bash

Run:

```bash
echo "$SHELL"
```

Then follow the matching section above (Zsh or Bash).

### I’m not seeing any suggestions / `zage suggest` is empty

1) Import history first:

```bash
zage import --shell zsh
# or
zage import --shell bash
```

2) Check model status:

```bash
zage model status
```

### I see “Suggest server unavailable”

That means you’re in **server mode** but the daemon isn’t running.
Pick one:

- Install the service: `zage service install`
- Run in foreground: `zage server`
- Or switch to embedded mode by setting `backend = "embedded"` in `~/.config/zage/config.toml`

### Zsh autosuggestions aren’t showing up

- Ensure zsh-autosuggestions is loaded **before** `zsh.zsh`.
- Ensure `ZAGE_AUTOSUGGEST_DISABLE` is not set to `1`.
- For debugging: set `ZAGE_ZSH_DEBUG=/tmp/zage-zsh.log` and restart your shell.

## License

MIT

## Acknowledgments

- Inspired by projects like:
  - [McFly](https://github.com/cantino/mcfly)
  - [Warp](https://github.com/warp-rs/warp)
  - [zsh-autosuggestions](https://github.com/zsh-users/zsh-autosuggestions)
  - [pxhist](https://github.com/chipturner/pxhist)
- Built with Rust 🦀
