# Tier-1 Test Specification: Deterministic Verification Framework

## 1. Overview

### 1.1 Purpose

This specification defines a comprehensive, deterministic verification framework for the Tier-1 statistical retrieval engine. The framework ensures that every ranking decision is predictable, testable, and explainable.

### 1.2 Design Principles

| Principle | Description |
|-----------|-------------|
| **Glass-Box Testing** | Inspect internal scores, not just final rankings |
| **Time Determinism** | Freeze `now` to make recency scores reproducible |
| **Weight Isolation** | Override ranking weights to test each force independently |
| **Real Filesystems** | Create ephemeral directories with `.git/` markers for repo detection |
| **Anti-Hallucination** | Verify the engine never suggests commands not grounded in history |
| **Structural Semantics** | Test flag commutativity, argument position strictness, and tokenization |

### 1.3 Scoring Formula Reference

From `src/predict.rs`, the Tier-1 ranking score is:

```
score = w_recency    × recency
      + w_frequency  × frequency
      + w_transition × transition
      + w_context    × context
      + w_sequence   × sequence
      + w_similarity × similarity
      + 0.1          × session_recency
```

Default weights (from `RankingWeights::default()`):
- `w_recency = 0.25`
- `w_frequency = 0.25`
- `w_transition = 0.20`
- `w_context = 0.15`
- `w_sequence = 0.10`
- `w_similarity = 0.05`

Additionally, `session_recency` has a hardcoded weight of `0.1` (not configurable via `RankingWeights`).

Each sub-score is computed as:
- **recency**: `exp(-age / half_life)` where `half_life = 604800 seconds (7 days)`
  - Returns `0.0` if `last_seen <= 0` or `now <= last_seen`
- **frequency**: `ln(freq + 1) + 0.5 × ln(repo_freq + 1)`
- **transition**: `ln(transition_freq + 1) + 0.7 × ln(repo_transition_freq + 1)`
- **context**: `ln(context_freq + 1) + 0.8 × ln(session_freq + 1) + phase_boost`
- **sequence**: `confidence × max(lift, 1.0) × order_weight`
- **similarity**: Sørensen–Dice coefficient on normalized tokens

---

## 2. Test File Schema

Test files use TOML format and are located in `src/testdata/tier1/`.

### 2.1 Complete Schema

```toml
#===============================================================================
# METADATA
#===============================================================================
[meta]
description = "Human-readable description of what this test verifies"
tags = ["recency", "transition", "anti-hallucination"]  # Optional categorization

#===============================================================================
# PHYSICS OVERRIDES
#===============================================================================
[physics]
# Freeze the current time (ISO 8601 format)
now = "2024-06-15T12:00:00Z"

# Override ranking weights (omit to use defaults from RankingWeights::default())
# Defaults: recency=0.25, frequency=0.25, transition=0.20, context=0.15, sequence=0.10, similarity=0.05
w_recency = 1.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

# Note: session_recency weight (0.1) is hardcoded and cannot be overridden

# Override recency half-life (default: 604800 seconds = 7 days)
recency_half_life_seconds = 604800

#===============================================================================
# EPHEMERAL FILESYSTEM
#===============================================================================
[fs]
# Keys are paths relative to temp root; values are "dir" or "file"
"project_a/.git/" = "dir"
"project_a/src/main.rs" = "file"
"project_a/Cargo.toml" = "file"
"project_b/.git/" = "dir"
"project_b/package.json" = "file"
"notes/todo.txt" = "file"

#===============================================================================
# ALIASES
#===============================================================================
[aliases]
gst = "git status"
gco = "git checkout"
gd = "git diff"

#===============================================================================
# HISTORY ENTRIES
#===============================================================================
[[history]]
cmd = "git status"
expanded = "git status"        # Optional: if different from cmd (alias expansion)
shell = "zsh"                  # Optional: default "zsh"
at = "-10m"                    # Relative to [physics].now: -10m, -1h, -2d, -1w, -1y
cwd = "project_a"              # Relative to temp root (resolves to /tmp/.../project_a)
hostname = "workstation"       # Optional: default "testhost"
username = "developer"         # Optional: default "testuser"
exit = 0                       # Exit status
session = "term_1"             # Session identifier (string, converted to stable i64)
count = 1                      # Optional: bulk insert count (default 1)

[[history]]
cmd = "git add ."
at = "-9m"
cwd = "project_a"
exit = 0
session = "term_1"

[[history]]
cmd = "git commit -m 'fix'"
at = "-8m"
cwd = "project_a"
exit = 0
session = "term_1"

#===============================================================================
# SCENARIOS
#===============================================================================
[[scenario]]
name = "unique_scenario_name"
mode = "next_command"          # "next_command" | "completion"

# For completion mode
input = "git c"                # The prefix typed by user
cursor = 5                     # Optional: cursor position (default: end of input)

# For next_command mode (context about previous command)
prev_command = "git add ."     # Optional: previous command for transition scoring
prev_exit = 0                  # Optional: exit status of previous command

[scenario.context]
cwd = "project_a/src"          # Relative to temp root
hostname = "workstation"       # Optional
username = "developer"         # Optional
session = "term_1"             # Optional

# Simple ranking assertions
[scenario.expect]
top = ["git commit -m 'fix'"]              # Exact match for top N results
contains = ["git status", "git diff"]       # Must appear somewhere in results
absent = ["npm install", "cargo build"]     # Must NOT appear in results
empty = false                               # true = expect no results (anti-hallucination)
min_results = 1                             # Minimum result count
max_results = 10                            # Maximum result count

# Deep inspection of a specific candidate
[[scenario.expect.candidate]]
cmd = "git commit -m 'fix'"

# Score bounds (all optional, 0.0 - 1.0 for normalized, or raw for composite)
min_score = 0.5
max_score = 1.0
min_recency = 0.8
max_recency = 1.0
min_frequency = 0.0
max_frequency = 0.5
min_transition = 0.3
max_transition = 1.0
min_context = 0.0
max_context = 1.0
min_sequence = 0.0
max_sequence = 0.0
min_similarity = 0.0
max_similarity = 1.0

# Rank bounds
min_rank = 1                   # Must be at least this rank (1 = first)
max_rank = 3                   # Must be at most this rank

# Database state assertions
[[scenario.expect.db]]
description = "Verify command_stats was indexed correctly"
sql = "SELECT freq FROM command_stats WHERE command = 'git status'"
operator = "eq"                # "eq" | "gt" | "gte" | "lt" | "lte" | "ne"
value = 1

[[scenario.expect.db]]
description = "Verify transition was recorded"
sql = "SELECT freq FROM transition_stats WHERE prev_command = 'git add .' AND next_command = 'git commit -m ''fix'''"
operator = "gte"
value = 1
```

### 2.2 Timestamp Format

The `at` field supports relative timestamps from `[physics].now`:

| Format | Meaning |
|--------|---------|
| `-5s` | 5 seconds ago |
| `-10m` | 10 minutes ago |
| `-2h` | 2 hours ago |
| `-3d` | 3 days ago |
| `-1w` | 1 week ago |
| `-2M` | 2 months ago (30 days each) |
| `-1y` | 1 year ago |
| `1710000000` | Absolute Unix timestamp |
| `2024-06-15T11:50:00Z` | Absolute ISO 8601 |

### 2.3 Session ID Resolution

Session identifiers are strings in TOML but resolve to `i64` in the database:

```
session_id = stable_hash(session_string) as i64
```

This allows readable test files while maintaining compatibility with the schema.

---

## 3. Verification Matrix

### 3.1 Recency Mechanics

#### 3.1.1 Decay Curve Verification

**Objective**: Verify `recency_score(now, last_seen) = exp(-age / half_life)`

```toml
[meta]
description = "Verify recency decay follows exponential curve"
tags = ["recency", "physics"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate recency by zeroing other weights (defaults: 0.25, 0.25, 0.20, 0.15, 0.10, 0.05)
w_recency = 1.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "recent_cmd"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "day_old_cmd"
at = "-1d"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "week_old_cmd"
at = "-1w"
cwd = "/tmp"
exit = 0
session = "s1"

[[scenario]]
name = "recency_ordering"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
top = ["recent_cmd", "day_old_cmd", "week_old_cmd"]

[[scenario.expect.candidate]]
cmd = "recent_cmd"
min_recency = 0.99    # 1 hour ago ≈ exp(-3600/604800) ≈ 0.9940

[[scenario.expect.candidate]]
cmd = "day_old_cmd"
min_recency = 0.86    # 1 day ago ≈ exp(-86400/604800) ≈ 0.8669
max_recency = 0.88

[[scenario.expect.candidate]]
cmd = "week_old_cmd"
min_recency = 0.36    # 1 week ago = half_life ≈ exp(-1) ≈ 0.3679
max_recency = 0.38
```

#### 3.1.2 Half-Life Verification

**Objective**: Verify that at exactly half-life, recency score ≈ 0.368 (1/e)

```toml
[meta]
description = "Verify half-life produces expected decay"
tags = ["recency", "physics"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 1.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0
recency_half_life_seconds = 86400  # Override default 604800 (7 days) to 1 day for easier testing

[[history]]
cmd = "at_half_life"
at = "-1d"
cwd = "/tmp"
exit = 0
session = "s1"

[[scenario]]
name = "half_life_check"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[[scenario.expect.candidate]]
cmd = "at_half_life"
min_recency = 0.36    # exp(-86400/86400) = exp(-1) ≈ 0.3679
max_recency = 0.38
```

### 3.2 Frequency Mechanics

#### 3.2.1 Frequency Ranking

**Objective**: Higher frequency commands rank higher (with recency disabled)

```toml
[meta]
description = "Verify frequency affects ranking"
tags = ["frequency"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate frequency scoring
w_recency = 0.0
w_frequency = 1.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "rare_cmd"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 1

[[history]]
cmd = "common_cmd"
at = "-2h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "very_common_cmd"
at = "-3h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 50

[[scenario]]
name = "frequency_ordering"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
top = ["very_common_cmd", "common_cmd", "rare_cmd"]

[[scenario.expect.candidate]]
cmd = "very_common_cmd"
min_frequency = 3.9  # ln(50+1) ≈ 3.93 (global freq only, no repo boost here)

[[scenario.expect.candidate]]
cmd = "common_cmd"
min_frequency = 2.3  # ln(10+1) ≈ 2.40
max_frequency = 2.5

[[scenario.expect.candidate]]
cmd = "rare_cmd"
min_frequency = 0.6  # ln(1+1) ≈ 0.69
max_frequency = 0.8

# Note: frequency score = ln(freq+1) + 0.5 * ln(repo_freq+1)
# In this test, repo_freq = 0 because no repo context, so it's just ln(freq+1)
```

#### 3.2.2 Repo Frequency Boost

**Objective**: Commands frequently used in current repo get boosted

```toml
[meta]
description = "Verify repo-specific frequency boost"
tags = ["frequency", "repo"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate frequency to test repo boost component
# frequency = ln(freq+1) + 0.5 * ln(repo_freq+1)
w_recency = 0.0
w_frequency = 1.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[fs]
"repo_a/.git/" = "dir"
"repo_b/.git/" = "dir"

[[history]]
cmd = "make build"
at = "-1h"
cwd = "repo_a"
exit = 0
session = "s1"
count = 20

[[history]]
cmd = "mvn build"
at = "-1h"
cwd = "repo_b"
exit = 0
session = "s1"
count = 20

[[scenario]]
name = "repo_frequency_isolation"
mode = "next_command"

[scenario.context]
cwd = "repo_a"
session = "s1"

[scenario.expect]
top = ["make build"]
absent = ["mvn build"]
```

### 3.3 Transition Mechanics

#### 3.3.1 Basic Transition Chain

**Objective**: Previous command influences next command ranking

```toml
[meta]
description = "Verify transition scoring from previous command"
tags = ["transition", "markov"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate transition scoring
# transition = ln(transition_freq+1) + 0.7 * ln(repo_transition_freq+1)
w_recency = 0.0
w_frequency = 0.0
w_transition = 1.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "git add ."
at = "-10m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git commit -m 'msg'"
at = "-9m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git add ."
at = "-8m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git commit -m 'fix'"
at = "-7m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git add ."
at = "-6m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git status"
at = "-5m"
cwd = "/tmp"
exit = 0
session = "s1"

[[scenario]]
name = "transition_after_git_add"
mode = "next_command"
prev_command = "git add ."
prev_exit = 0

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# git commit appears 2x after git add, git status appears 1x
top = ["git commit -m 'msg'", "git commit -m 'fix'"]
contains = ["git status"]

[[scenario.expect.candidate]]
cmd = "git commit -m 'msg'"
min_transition = 0.5
```

#### 3.3.2 Exit Status Aware Transitions

**Objective**: Failed commands influence different follow-up suggestions

```toml
[meta]
description = "Verify exit status affects transition scoring"
tags = ["transition", "exit_status"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate transition for exit-status testing
w_recency = 0.0
w_frequency = 0.0
w_transition = 1.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "cargo test"
at = "-10m"
cwd = "/tmp"
exit = 1  # Failed

[[history]]
cmd = "cargo test"  # Retry
at = "-9m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "cargo test"
at = "-8m"
cwd = "/tmp"
exit = 1  # Failed again

[[history]]
cmd = "git diff"  # Check what changed
at = "-7m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "cargo test"
at = "-6m"
cwd = "/tmp"
exit = 0  # Success

[[history]]
cmd = "git commit -m 'fix'"  # After success
at = "-5m"
cwd = "/tmp"
exit = 0
session = "s1"

[[scenario]]
name = "after_failed_test"
mode = "next_command"
prev_command = "cargo test"
prev_exit = 1

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# After failed test, suggest retry or investigation
contains = ["cargo test", "git diff"]
absent = ["git commit -m 'fix'"]  # Don't suggest commit after failure

[[scenario]]
name = "after_successful_test"
mode = "next_command"
prev_command = "cargo test"
prev_exit = 0

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["git commit -m 'fix'"]
```

#### 3.3.3 Repo-Specific Transitions

**Objective**: Different repos have different transition patterns

```toml
[meta]
description = "Verify repo-specific transition chains"
tags = ["transition", "repo"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate transition for repo-specific chain testing
# repo_transition_freq contributes 0.7 * ln(freq+1) to transition score
w_recency = 0.0
w_frequency = 0.0
w_transition = 1.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[fs]
"rust_project/.git/" = "dir"
"node_project/.git/" = "dir"

[[history]]
cmd = "git checkout"
at = "-10m"
cwd = "rust_project"
exit = 0
session = "s1"

[[history]]
cmd = "main"
at = "-9m"
cwd = "rust_project"
exit = 0
session = "s1"

[[history]]
cmd = "git checkout"
at = "-8m"
cwd = "node_project"
exit = 0
session = "s1"

[[history]]
cmd = "master"
at = "-7m"
cwd = "node_project"
exit = 0
session = "s1"

[[scenario]]
name = "rust_repo_branch"
mode = "next_command"
prev_command = "git checkout"
prev_exit = 0

[scenario.context]
cwd = "rust_project"
session = "s1"

[scenario.expect]
top = ["main"]
absent = ["master"]

[[scenario]]
name = "node_repo_branch"
mode = "next_command"
prev_command = "git checkout"
prev_exit = 0

[scenario.context]
cwd = "node_project"
session = "s1"

[scenario.expect]
top = ["master"]
absent = ["main"]
```

### 3.4 Context Mechanics

#### 3.4.1 Working Directory Context

**Objective**: Commands are boosted when CWD matches history

```toml
[meta]
description = "Verify CWD context affects ranking"
tags = ["context", "cwd"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate context scoring
# context = ln(context_freq+1) + 0.8 * ln(session_freq+1) + phase_match_boost
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 1.0
w_sequence = 0.0
w_similarity = 0.0

[fs]
"project_a/.git/" = "dir"
"project_b/.git/" = "dir"

[[history]]
cmd = "make test"
at = "-1h"
cwd = "project_a"
exit = 0
session = "s1"
count = 5

[[history]]
cmd = "npm test"
at = "-1h"
cwd = "project_b"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "cwd_context_project_a"
mode = "next_command"

[scenario.context]
cwd = "project_a"
session = "s1"

[scenario.expect]
top = ["make test"]

[[scenario]]
name = "cwd_context_project_b"
mode = "next_command"

[scenario.context]
cwd = "project_b"
session = "s1"

[scenario.expect]
top = ["npm test"]
```

#### 3.4.2 Subdirectory Fallback

**Objective**: Commands from repo root match when queried from subdirectory

```toml
[meta]
description = "Verify subdirectory inherits repo context"
tags = ["context", "cwd", "repo"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate context for subdirectory fallback testing
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 1.0
w_sequence = 0.0
w_similarity = 0.0

[fs]
"myproject/.git/" = "dir"
"myproject/src/" = "dir"
"myproject/src/lib/" = "dir"

[[history]]
cmd = "cargo build"
at = "-1h"
cwd = "myproject"
exit = 0
session = "s1"
count = 10

[[scenario]]
name = "subdir_inherits_repo_context"
mode = "next_command"

[scenario.context]
cwd = "myproject/src/lib"  # Deep subdirectory
session = "s1"

[scenario.expect]
contains = ["cargo build"]

[[scenario.expect.candidate]]
cmd = "cargo build"
min_context = 0.5  # Should still get context score from repo match
```

#### 3.4.3 Session Context Boost

**Objective**: Commands from current session are boosted

```toml
[meta]
description = "Verify session-specific context boost"
tags = ["context", "session"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate context for session boost testing
# session_freq contributes 0.8 * ln(freq+1) to context score
# Additionally, session_recency (hardcoded 0.1 weight) adds session-specific recency
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 1.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "current_session_cmd"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "current_term"
count = 2

[[history]]
cmd = "other_session_cmd"
at = "-30m"  # More recent, but different session
cwd = "/tmp"
exit = 0
session = "other_term"
count = 5

[[scenario]]
name = "session_boost"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "current_term"

[scenario.expect]
top = ["current_session_cmd"]

[[scenario.expect.candidate]]
cmd = "current_session_cmd"
min_context = 0.5  # Session boost should apply
```

#### 3.4.4 Hostname and Username Context

**Objective**: Commands are boosted when hostname/username match

```toml
[meta]
description = "Verify hostname and username context"
tags = ["context", "hostname", "username"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate context for hostname/username testing
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 1.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "deploy_prod"
at = "-1h"
cwd = "/tmp"
hostname = "prod-server"
username = "deploy"
exit = 0
session = "s1"

[[history]]
cmd = "local_dev"
at = "-1h"
cwd = "/tmp"
hostname = "workstation"
username = "developer"
exit = 0
session = "s1"

[[scenario]]
name = "hostname_context"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
hostname = "prod-server"
username = "deploy"
session = "s1"

[scenario.expect]
top = ["deploy_prod"]
absent = ["local_dev"]
```

### 3.5 Sequence Mechanics (N-gram Mining)

#### 3.5.1 Bigram Sequences

**Objective**: Two-command sequences are learned and suggested

```toml
[meta]
description = "Verify bigram sequence mining"
tags = ["sequence", "bigram"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate sequence scoring
# sequence = confidence * max(lift, 1.0) * order_weight
# order_weight = 1.0 if prefix_len >= 2 (trigram), else 0.7 (bigram)
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 1.0
w_similarity = 0.0

# Enable sequence mining for this test
[options]
use_sequences = true
run_sequence_analysis = true
min_sequence_support = 2
min_sequence_confidence = 0.5
min_sequence_lift = 1.0

[[history]]
cmd = "git pull"
at = "-20m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "cargo build"
at = "-19m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git pull"
at = "-18m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "cargo build"
at = "-17m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git pull"
at = "-16m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "cargo build"
at = "-15m"
cwd = "/tmp"
exit = 0
session = "s1"

[[scenario]]
name = "bigram_git_pull_cargo_build"
mode = "next_command"
prev_command = "git pull"
prev_exit = 0

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
top = ["cargo build"]

[[scenario.expect.candidate]]
cmd = "cargo build"
min_sequence = 0.5  # High sequence score due to consistent pattern
```

#### 3.5.2 Trigram Sequences

**Objective**: Three-command sequences are learned

```toml
[meta]
description = "Verify trigram sequence mining"
tags = ["sequence", "trigram"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate sequence for trigram testing
# Trigrams have order_weight = 1.0 (vs 0.7 for bigrams)
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 1.0
w_similarity = 0.0

[options]
use_sequences = true
run_sequence_analysis = true
min_sequence_support = 2

# Pattern: git add -> git commit -> git push
[[history]]
cmd = "git add ."
at = "-30m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git commit -m 'a'"
at = "-29m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git push"
at = "-28m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git add ."
at = "-20m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git commit -m 'b'"
at = "-19m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git push"
at = "-18m"
cwd = "/tmp"
exit = 0
session = "s1"

# Current context: just ran git add, then git commit
[[scenario]]
name = "trigram_predicts_push"
mode = "next_command"
prev_command = "git commit -m 'c'"
prev_exit = 0

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
top = ["git push"]
```

### 3.6 Similarity Mechanics

#### 3.6.1 Token Similarity Scoring

**Objective**: Partial token matches boost results

```toml
[meta]
description = "Verify token similarity scoring"
tags = ["similarity", "tokens"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate similarity scoring
# similarity = Sørensen–Dice coefficient = 2*|intersection| / (|A| + |B|)
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 1.0

[[history]]
cmd = "cargo build --release"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "cargo test --release"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "npm run build"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"

[[scenario]]
name = "similarity_cargo_release"
mode = "completion"
input = "cargo --release"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# Both cargo commands share tokens with input
contains = ["cargo build --release", "cargo test --release"]
absent = ["npm run build"]  # No token overlap with "cargo --release"
```

### 3.7 Structure Semantics

#### 3.7.1 Flag Commutativity (Position Independence)

**Objective**: Flags can appear in any order

```toml
[meta]
description = "Verify flags are order-independent"
tags = ["structure", "flags"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "ls -a -l -h"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "flag_order_la"
mode = "completion"
input = "ls -l -a"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["ls -a -l -h"]  # Should match despite different flag order

[[scenario]]
name = "flag_order_al"
mode = "completion"
input = "ls -a -l"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["ls -a -l -h"]
```

#### 3.7.2 Argument Position Strictness

**Objective**: Arguments are position-sensitive

```toml
[meta]
description = "Verify argument positions are strict"
tags = ["structure", "args"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "cp source.txt dest.txt"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "correct_arg_position"
mode = "completion"
input = "cp source.txt "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["cp source.txt dest.txt"]

[[scenario]]
name = "wrong_arg_position"
mode = "completion"
input = "cp dest.txt "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# Should NOT suggest "source.txt" because it's in wrong position
absent = ["cp dest.txt source.txt"]
```

#### 3.7.3 Flag-Argument Binding

**Objective**: Flags that take arguments are handled correctly

```toml
[meta]
description = "Verify flag-argument binding"
tags = ["structure", "flags", "args"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "gcc -o myprogram main.c"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "flag_arg_binding"
mode = "completion"
input = "gcc -o "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["gcc -o myprogram main.c"]

[[scenario]]
name = "positional_after_flag_arg"
mode = "completion"
input = "gcc -o myprogram "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["gcc -o myprogram main.c"]
```

#### 3.7.4 Quoted Argument Handling

**Objective**: Quoted strings are treated as single tokens

```toml
[meta]
description = "Verify quoted arguments are handled correctly"
tags = ["structure", "quotes", "tokens"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "git commit -m \"fix: resolve bug\""
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[history]]
cmd = "git commit -m 'add feature'"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 3

[[scenario]]
name = "quoted_double"
mode = "completion"
input = "git commit -m \""

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["git commit -m \"fix: resolve bug\""]

[[scenario]]
name = "quoted_single"
mode = "completion"
input = "git commit -m '"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["git commit -m 'add feature'"]
```

#### 3.7.5 Environment Variable Prefix

**Objective**: Commands with env prefix are handled correctly

```toml
[meta]
description = "Verify environment variable prefix handling"
tags = ["structure", "env"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "RUST_LOG=debug cargo test"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[history]]
cmd = "cargo test"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[scenario]]
name = "env_prefix_completion"
mode = "completion"
input = "RUST_LOG=debug "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
top = ["RUST_LOG=debug cargo test"]
# Should NOT suggest bare "cargo test" when env prefix is present
absent = ["cargo test"]

[[scenario]]
name = "no_env_prefix"
mode = "completion"
input = "cargo "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["cargo test"]
# Should NOT inject env prefix the user didn't type
absent = ["RUST_LOG="]
```

#### 3.7.6 Pipe and Multi-Command Handling

**Objective**: Piped commands are tokenized correctly

```toml
[meta]
description = "Verify pipe and multi-command handling"
tags = ["structure", "pipes"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "ls -la | grep foo"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[history]]
cmd = "cat file.txt | wc -l"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 3

[[scenario]]
name = "pipe_completion"
mode = "completion"
input = "ls -la | "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["ls -la | grep foo"]

[[scenario]]
name = "pipe_grep"
mode = "completion"
input = "ls -la | grep "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["ls -la | grep foo"]
```

### 3.8 Anti-Hallucination

#### 3.8.1 Empty History Rule

**Objective**: No suggestions when history is empty

```toml
[meta]
description = "Verify empty history produces no suggestions"
tags = ["anti-hallucination", "empty"]

[physics]
now = "2024-06-15T12:00:00Z"

# NO history entries

[[scenario]]
name = "empty_history_next_command"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
empty = true

[[scenario]]
name = "empty_history_completion"
mode = "completion"
input = "git c"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
empty = true
```

#### 3.8.2 Strict Prefix Adherence

**Objective**: Suggestions must match the typed prefix

```toml
[meta]
description = "Verify strict prefix matching"
tags = ["anti-hallucination", "prefix"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "cargo build"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "cargo test"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[scenario]]
name = "prefix_match"
mode = "completion"
input = "cargo b"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["cargo build"]
absent = ["cargo test"]  # "cargo t" doesn't match "cargo b"

[[scenario]]
name = "prefix_no_match"
mode = "completion"
input = "cargo x"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
empty = true  # Nothing in history starts with "cargo x"
```

#### 3.8.3 No Ghost Candidates

**Objective**: Don't suggest completions that contradict user input

```toml
[meta]
description = "Verify no ghost candidates are suggested"
tags = ["anti-hallucination", "ghost"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "git commit -m 'fix'"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[scenario]]
name = "contradictory_flag"
mode = "completion"
input = "git commit -a"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# User typed "-a" but history has "-m", don't force "-m"
absent = ["git commit -m"]

[[scenario]]
name = "partial_contradictory"
mode = "completion"
input = "git commit --am"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# "--am" could be start of "--amend", but we don't have "--amend" in history
empty = true
```

#### 3.8.4 No Cross-Contamination

**Objective**: Commands from unrelated contexts don't appear

```toml
[meta]
description = "Verify context isolation prevents cross-contamination"
tags = ["anti-hallucination", "context"]

[physics]
now = "2024-06-15T12:00:00Z"
# Isolate context for cross-contamination testing
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 1.0
w_sequence = 0.0
w_similarity = 0.0

[fs]
"work/.git/" = "dir"
"personal/.git/" = "dir"

[[history]]
cmd = "kubectl apply -f deployment.yaml"
at = "-1h"
cwd = "work"
exit = 0
session = "work_session"
count = 20

[[history]]
cmd = "brew install neovim"
at = "-1h"
cwd = "personal"
exit = 0
session = "personal_session"
count = 20

[[scenario]]
name = "work_context_no_personal"
mode = "next_command"

[scenario.context]
cwd = "work"
session = "work_session"

[scenario.expect]
contains = ["kubectl apply -f deployment.yaml"]
absent = ["brew install neovim"]

[[scenario]]
name = "personal_context_no_work"
mode = "next_command"

[scenario.context]
cwd = "personal"
session = "personal_session"

[scenario.expect]
contains = ["brew install neovim"]
absent = ["kubectl apply -f deployment.yaml"]
```

### 3.9 Alias Mechanics

#### 3.9.1 Alias Expansion Matching

**Objective**: Aliases match their expanded forms

```toml
[meta]
description = "Verify alias expansion and matching"
tags = ["aliases"]

[physics]
now = "2024-06-15T12:00:00Z"

[aliases]
gst = "git status"
gd = "git diff"
gco = "git checkout"

[[history]]
cmd = "git status"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "git diff"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "alias_completion"
mode = "completion"
input = "gst"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# Should suggest the alias form
contains = ["gst"]

[[scenario]]
name = "alias_prefix"
mode = "completion"
input = "gs"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["gst"]
```

#### 3.9.2 Alias Suffix Completion

**Objective**: Aliases work with additional arguments

```toml
[meta]
description = "Verify alias with suffix arguments"
tags = ["aliases"]

[physics]
now = "2024-06-15T12:00:00Z"

[aliases]
gco = "git checkout"

[[history]]
cmd = "git checkout main"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "git checkout feature-branch"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "alias_with_arg"
mode = "completion"
input = "gco "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["gco main", "gco feature-branch"]
```

### 3.10 Phase Mechanics

#### 3.10.1 Phase Detection

**Objective**: Workflow phase is detected from recent commands

```toml
[meta]
description = "Verify phase detection from command history"
tags = ["phase"]

[physics]
now = "2024-06-15T12:00:00Z"
# Phase boost is added to context score via phase_match_boost()
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 1.0
w_sequence = 0.0
w_similarity = 0.0

[options]
run_phase_indexing = true

[phases]
build = ["cargo build", "make", "npm run build"]
test = ["cargo test", "pytest", "npm test"]
deploy = ["kubectl apply", "docker push"]

[[history]]
cmd = "cargo build"
at = "-10m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "cargo build --release"
at = "-9m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "cargo test"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 20

[[scenario]]
name = "build_phase_boost"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# Recent commands are build-phase, so build commands should be boosted
top = ["cargo build", "cargo build --release"]
```

### 3.11 Template Mechanics

#### 3.11.1 Argument Templates

**Objective**: Common argument patterns are suggested

```toml
[meta]
description = "Verify argument template suggestions"
tags = ["templates", "args"]

[physics]
now = "2024-06-15T12:00:00Z"

[fs]
"myrepo/.git/" = "dir"

[[history]]
cmd = "git checkout main"
at = "-1h"
cwd = "myrepo"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "git checkout develop"
at = "-1h"
cwd = "myrepo"
exit = 0
session = "s1"
count = 5

[[history]]
cmd = "git checkout feature-x"
at = "-30m"
cwd = "myrepo"
exit = 0
session = "s1"
count = 3

[[scenario]]
name = "branch_template"
mode = "completion"
input = "git checkout "

[scenario.context]
cwd = "myrepo"
session = "s1"

[scenario.expect]
# Should suggest known branches from this repo
contains = ["git checkout main", "git checkout develop", "git checkout feature-x"]
```

#### 3.11.2 Flag Templates

**Objective**: Common flag combinations are suggested

```toml
[meta]
description = "Verify flag template suggestions"
tags = ["templates", "flags"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "docker run -it --rm ubuntu"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "docker run -d --name mycontainer nginx"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "docker_flag_template"
mode = "completion"
input = "docker run -"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["docker run -it", "docker run -d"]
```

### 3.12 Integration Scenarios

#### 3.12.1 Competing Signals

**Objective**: Verify score interactions when multiple signals conflict

```toml
[meta]
description = "Verify recency can beat moderate frequency under default weights"
tags = ["integration", "tradeoff"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "frequent_old_cmd"
at = "-7d"
cwd = "/tmp/other"
exit = 0
session = "s2"
count = 5

[[history]]
cmd = "recent_cmd"
at = "-1h"
cwd = "/tmp/project"
exit = 0
session = "s1"
count = 1

[[scenario]]
name = "competing_signals_recency_wins"
mode = "next_command"

[scenario.context]
cwd = "/tmp/project"
session = "s1"

[scenario.expect]
top = ["recent_cmd"]
contains = ["frequent_old_cmd"]
```

#### 3.12.2 Default Weights Verification

**Objective**: Verify the default weight balance produces expected behavior

```toml
[meta]
description = "Verify default weights produce balanced ranking"
tags = ["integration", "weights"]

[physics]
now = "2024-06-15T12:00:00Z"
# Using default weights - no overrides

[fs]
"project/.git/" = "dir"

[[history]]
cmd = "balanced_winner"
at = "-30m"
cwd = "project"
exit = 0
session = "s1"
count = 5

[[history]]
cmd = "recency_only"
at = "-5m"
cwd = "/other"
exit = 0
session = "s2"
count = 1

[[scenario]]
name = "default_weights_balance"
mode = "next_command"

[scenario.context]
cwd = "project"
session = "s1"

[scenario.expect]
# Balanced winner has better context + frequency, should beat pure recency
top = ["balanced_winner"]
contains = ["recency_only"]
```

#### 3.12.3 Full Stack Integration

**Objective**: Test all components working together

```toml
[meta]
description = "Verify full stack integration with all features"
tags = ["integration", "full"]

[physics]
now = "2024-06-15T12:00:00Z"

[fs]
"myproject/.git/" = "dir"
"myproject/src/" = "dir"

[aliases]
gst = "git status"

[options]
use_sequences = true
run_sequence_analysis = true
min_sequence_support = 2

[[history]]
cmd = "git status"
at = "-10m"
cwd = "myproject"
exit = 0
session = "s1"

[[history]]
cmd = "git add ."
at = "-9m"
cwd = "myproject"
exit = 0
session = "s1"

[[history]]
cmd = "git commit -m 'fix'"
at = "-8m"
cwd = "myproject"
exit = 0
session = "s1"

[[history]]
cmd = "git status"
at = "-7m"
cwd = "myproject"
exit = 0
session = "s1"

[[history]]
cmd = "git add ."
at = "-6m"
cwd = "myproject"
exit = 0
session = "s1"

[[history]]
cmd = "git commit -m 'update'"
at = "-5m"
cwd = "myproject"
exit = 0
session = "s1"

[[scenario]]
name = "full_stack_after_add"
mode = "next_command"
prev_command = "git add ."
prev_exit = 0

[scenario.context]
cwd = "myproject/src"
session = "s1"

[scenario.expect]
# git commit should be top due to transition + sequence + context
contains = ["git commit -m 'fix'", "git commit -m 'update'"]
```

---

## 4. Database State Assertions

### 4.1 Command Stats Verification

```toml
[[scenario.expect.db]]
description = "Verify command frequency was indexed"
sql = "SELECT freq FROM command_stats WHERE command = ?"
params = ["git status"]
operator = "eq"
value = 5
```

### 4.2 Transition Stats Verification

```toml
[[scenario.expect.db]]
description = "Verify transition was recorded"
sql = "SELECT freq FROM transition_stats WHERE prev_command = ? AND next_command = ?"
params = ["git add .", "git commit -m 'fix'"]
operator = "gte"
value = 1
```

### 4.3 Repo Stats Verification

```toml
[[scenario.expect.db]]
description = "Verify repo-specific stats"
sql = "SELECT freq FROM repo_command_stats WHERE repo_root LIKE ? AND command = ?"
params = ["%/myrepo", "cargo build"]
operator = "eq"
value = 10
```

### 4.4 Sequence Stats Verification

```toml
[[scenario.expect.db]]
description = "Verify sequence was mined"
sql = "SELECT lift FROM sequence_stats WHERE sequence_json LIKE ?"
params = ["%git add%git commit%"]
operator = "gte"
value = 1.5
```

---

## 5. Implementation Requirements

### 5.1 Test Harness API

```rust
/// Ranking weights - mirrors src/predict.rs RankingWeights
/// Default values: recency=0.25, frequency=0.25, transition=0.20, 
///                 context=0.15, sequence=0.10, similarity=0.05
/// Note: session_recency has hardcoded weight of 0.1 in score_candidates()
pub struct RankingWeights {
    pub recency: f64,
    pub frequency: f64,
    pub transition: f64,
    pub context: f64,
    pub sequence: f64,
    pub similarity: f64,
}

/// Configuration for deterministic testing
pub struct TestConfig {
    /// Frozen timestamp for recency calculations (Unix seconds)
    /// If None, uses SystemTime::now()
    pub now: Option<i64>,
    
    /// Override ranking weights
    /// If None, uses RankingWeights::default()
    pub weights: Option<RankingWeights>,
    
    /// Override recency half-life in seconds
    /// Default: 604800 (7 days = 60*60*24*7)
    pub recency_half_life: Option<f64>,
    
    /// Enable debug mode for score breakdown
    pub debug: bool,
}

/// Score breakdown - mirrors src/predict.rs ScoreBreakdown
/// These are the raw sub-scores BEFORE applying weights
pub struct ScoreBreakdown {
    pub recency: f64,      // exp(-age / half_life), 0.0 if invalid
    pub frequency: f64,    // ln(freq+1) + 0.5 * ln(repo_freq+1)
    pub transition: f64,   // ln(trans_freq+1) + 0.7 * ln(repo_trans_freq+1)
    pub context: f64,      // ln(ctx_freq+1) + 0.8 * ln(session_freq+1) + phase_boost
    pub sequence: f64,     // confidence * max(lift, 1.0) * order_weight
    pub similarity: f64,   // Sørensen–Dice coefficient on tokens
}

/// Extended suggestion with full breakdown for testing
pub struct TestSuggestion {
    pub command: String,
    pub score: f64,
    pub breakdown: ScoreBreakdown,
    pub rank: usize,
}

/// Test harness entry point
pub async fn suggest_for_test(
    conn: &Connection,
    config: SuggestConfig,
    test_config: TestConfig,
) -> Result<Vec<TestSuggestion>>;
```

### 5.2 Required Refactoring

1. **TimeProvider Trait**
   - Extract `SystemTime::now()` calls in `suggest()` and `score_candidates()` into injectable `TimeProvider`
   - Default implementation uses wall clock
   - Test implementation uses frozen time from `TestConfig.now`

2. **Configurable Weights**
   - Make `RankingWeights` a parameter to `score_candidates()` or part of `SuggestConfig`
   - Current hardcoded `RankingWeights::default()` (0.25, 0.25, 0.20, 0.15, 0.10, 0.05) becomes the default
   - Note: `session_recency` weight (0.1) is hardcoded in `score_candidates()` line 375

3. **Configurable Half-Life**
   - Extract `half_life = 60.0 * 60.0 * 24.0 * 7.0` in `ranking.rs:recency_score()` to be injectable
   - Default remains 604800 seconds (7 days)
   - Allow override via `TestConfig.recency_half_life`

4. **Score Breakdown Exposure**
   - `ScoreBreakdown` is already populated in `score_candidates()` lines 380-387
   - Add `rank` field to test output (1-indexed position after sorting)

5. **Filesystem Fixture**
   - Helper to materialize `[fs]` block into `tempfile::TempDir`
   - Path resolution: relative paths → absolute paths in temp dir
   - Ensure `.git/` directories are created so `find_repo_root()` works

6. **Session ID Resolution**
   - Stable hash function for session string → i64
   - Same string always produces same ID

### 5.3 File Organization

```
src/
├── testdata/
│   └── tier1/
│       ├── recency/
│       │   ├── decay_curve.toml
│       │   └── half_life.toml
│       ├── frequency/
│       │   ├── basic_ranking.toml
│       │   └── repo_boost.toml
│       ├── transition/
│       │   ├── basic_chain.toml
│       │   ├── exit_status.toml
│       │   └── repo_specific.toml
│       ├── context/
│       │   ├── cwd.toml
│       │   ├── subdirectory.toml
│       │   ├── session.toml
│       │   └── hostname.toml
│       ├── sequence/
│       │   ├── bigram.toml
│       │   └── trigram.toml
│       ├── similarity/
│       │   └── token_matching.toml
│       ├── structure/
│       │   ├── flag_commutativity.toml
│       │   ├── arg_position.toml
│       │   ├── flag_arg_binding.toml
│       │   ├── quotes.toml
│       │   ├── env_prefix.toml
│       │   └── pipes.toml
│       ├── anti_hallucination/
│       │   ├── empty_history.toml
│       │   ├── strict_prefix.toml
│       │   ├── ghost_candidates.toml
│       │   └── cross_contamination.toml
│       ├── aliases/
│       │   ├── expansion.toml
│       │   └── suffix.toml
│       ├── phase/
│       │   └── detection.toml
│       └── templates/
│           ├── args.toml
│           └── flags.toml
├── predict/
│   └── verifier.rs          # Test harness implementation
└── predict.rs               # Modified with TimeProvider, configurable weights
```

### 5.4 Test Runner

```rust
#[tokio::test]
async fn tier1_verification_suite() {
    let test_files = glob("src/testdata/tier1/**/*.toml").unwrap();
    
    for path in test_files {
        let content = fs::read_to_string(&path).unwrap();
        let spec: TestSpec = toml::from_str(&content).unwrap();
        
        // Create temp directory with filesystem structure
        let temp_dir = materialize_filesystem(&spec.fs);
        
        // Initialize database
        let db = open_db(temp_dir.path().join("test.db")).await.unwrap();
        init(&db.conn).await.unwrap();
        
        // Seed history with path resolution
        seed_history(&db.conn, &spec.history, &temp_dir).await;
        
        // Rebuild stats (and optionally sequences)
        rebuild_stats(&db.conn, None).await.unwrap();
        if spec.options.run_sequence_analysis {
            analyze_sequences(&db.conn, spec.sequence_config()).await.unwrap();
        }
        
        // Run each scenario
        for scenario in &spec.scenarios {
            let config = build_suggest_config(&scenario, &temp_dir);
            let test_config = build_test_config(&spec.physics);
            
            let results = suggest_for_test(&db.conn, config, test_config).await.unwrap();
            
            // Assert expectations
            assert_scenario_expectations(&scenario.expect, &results, &scenario.name);
        }
    }
}
```

---

## 6. Coverage Checklist

### 6.1 Scoring Components

| Component | Test File | Status |
|-----------|-----------|--------|
| Recency decay | `recency/decay_curve.toml` | ✓ |
| Recency half-life | `recency/half_life.toml` | ✓ |
| Recency future timestamps | `recency/future_timestamps.toml` | ✓ |
| Recency invalid timestamps | `recency/invalid_timestamps.toml` | ✓ |
| Frequency basic | `frequency/basic_ranking.toml` | ✓ |
| Frequency repo boost | `frequency/repo_boost.toml` | ✓ |
| Frequency zero | `frequency/zero_frequency.toml` | ✓ |
| Transition basic | `transition/basic_chain.toml` | ✓ |
| Transition exit status | `transition/exit_status.toml` | ✓ |
| Transition no previous | `transition/no_previous.toml` | ✓ |
| Transition repo-specific | `transition/repo_specific.toml` | ✓ |
| Context CWD | `context/cwd.toml` | ✓ |
| Context subdirectory | `context/subdirectory.toml` | ✓ |
| Context parent inheritance | `context/parent_inheritance.toml` | ✓ |
| Context session | `context/session.toml` | ✓ |
| Context session recency | `context/session_recency_weight.toml` | ✓ |
| Context hostname/user | `context/hostname.toml` | ✓ |
| Sequence bigram | `sequence/bigram.toml` | ✓ |
| Sequence trigram | `sequence/trigram.toml` | ✓ |
| Sequence disabled | `sequence/disabled.toml` | ✓ |
| Sequence low confidence | `sequence/low_confidence.toml` | ✓ |
| Similarity tokens | `similarity/token_matching.toml` | ✓ |
| Similarity no overlap | `similarity/no_overlap.toml` | ✓ |

### 6.2 Structural Semantics

| Component | Test File | Status |
|-----------|-----------|--------|
| Flag commutativity | `structure/flag_commutativity.toml` | ✓ |
| Arg position strictness | `structure/arg_position.toml` | ✓ |
| Flag-arg binding | `structure/flag_arg_binding.toml` | ✓ |
| Quoted arguments | `structure/quotes.toml` | ✓ |
| Environment prefix | `structure/env_prefix.toml` | ✓ |
| Pipe handling | `structure/pipes.toml` | ✓ |
| Cursor position | `structure/cursor.toml` | ✓ |

### 6.3 Anti-Hallucination

| Component | Test File | Status |
|-----------|-----------|--------|
| Empty history | `anti_hallucination/empty_history.toml` | ✓ |
| Strict prefix | `anti_hallucination/strict_prefix.toml` | ✓ |
| Ghost candidates | `anti_hallucination/ghost_candidates.toml` | ✓ |
| Cross-contamination | `anti_hallucination/cross_contamination.toml` | ✓ |

### 6.4 Features

| Component | Test File | Status |
|-----------|-----------|--------|
| Alias expansion | `aliases/expansion.toml` | ✓ |
| Alias suffix | `aliases/suffix.toml` | ✓ |
| Phase detection | `phase/detection.toml` | ✓ |
| Arg templates | `templates/args.toml` | ✓ |
| Flag templates | `templates/flags.toml` | ✓ |

### 6.5 Integration Tests

| Component | Test File | Status |
|-----------|-----------|--------|
| Competing signals | `integration/competing_signals.toml` | ✓ |
| Default weights | `integration/default_weights.toml` | ✓ |
| Full stack | `integration/full_stack.toml` | ✓ |
| DB assertions | `integration/db_assertions.toml` | ✓ |

---

## 7. Definition of Done

- [ ] Test harness supports `[physics]` weight overrides
- [ ] Test harness supports frozen timestamps via `now`
- [ ] Test harness supports configurable `recency_half_life`
- [ ] Test harness supports real ephemeral filesystems from `[fs]`
- [ ] Test harness supports `[aliases]` injection
- [ ] Test harness can inspect `ScoreBreakdown` with `[[scenario.expect.candidate]]`
- [ ] Test harness can run SQL assertions with `[[scenario.expect.db]]`
- [ ] All 20+ test files in coverage checklist pass
- [ ] Anti-hallucination suite passes (empty history, strict prefix, no ghosts)
- [ ] CI runs full tier1 verification suite on every PR
- [ ] Test execution time < 30 seconds for full suite

---

## 8. Migration Path

### 8.1 Existing `completion_cases.toml`

The 5 existing cases will be migrated to the new format:

| Old Case | New Location | Enhancements |
|----------|--------------|--------------|
| `head_completion_long_history` | `structure/env_prefix.toml` | Add timestamps, explicit weights |
| `first_arg_position` | `templates/args.toml` | Add repo context |
| `second_arg_position` | `templates/args.toml` | Add position strictness checks |
| `flag_position` | `templates/flags.toml` | Add flag commutativity checks |
| `head_after_env_prefix` | `structure/env_prefix.toml` | Add anti-hallucination checks |

### 8.2 Deprecation

Once the new test suite is complete and passing:
1. Remove `src/testdata/completion_cases.toml`
2. Remove the old test runner in `predict.rs` tests
3. Update CI to run only the new verification suite

---

## 9. Gap Analysis and Additional Edge Cases

This section documents edge cases and additional test scenarios that should be added to achieve comprehensive coverage.

### 9.1 Recency Edge Cases

#### 9.1.1 Future Timestamps

**Objective**: Commands with future timestamps should return recency score of 0.0

```toml
[meta]
description = "Verify future timestamps are handled gracefully"
tags = ["recency", "edge", "future"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 1.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "future_cmd"
at = "+1h"  # 1 hour in the future
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "past_cmd"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"

[[scenario]]
name = "future_timestamp_ignored"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
top = ["past_cmd"]

[[scenario.expect.candidate]]
cmd = "future_cmd"
min_recency = 0.0
max_recency = 0.0
```

#### 9.1.2 Invalid/Zero Timestamps

**Objective**: Commands with zero or negative timestamps return recency 0.0

```toml
[meta]
description = "Verify invalid timestamps are handled"
tags = ["recency", "edge", "invalid"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 1.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "zero_ts_cmd"
at = 0
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "valid_cmd"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"

[[scenario]]
name = "zero_timestamp_handled"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
top = ["valid_cmd"]
```

#### 9.1.3 Multiple Half-Lives

**Objective**: Very old commands decay exponentially across multiple half-lives

```toml
[meta]
description = "Verify decay over multiple half-lives"
tags = ["recency", "physics", "edge"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 1.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "two_half_lives"
at = "-14d"  # 2 weeks = 2 half-lives
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "three_half_lives"
at = "-21d"  # 3 weeks = 3 half-lives
cwd = "/tmp"
exit = 0
session = "s1"

[[scenario]]
name = "multi_half_life_decay"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[[scenario.expect.candidate]]
cmd = "two_half_lives"
min_recency = 0.13  # exp(-2) ≈ 0.135
max_recency = 0.14

[[scenario.expect.candidate]]
cmd = "three_half_lives"
min_recency = 0.04  # exp(-3) ≈ 0.050
max_recency = 0.06
```

### 9.2 Frequency Edge Cases

#### 9.2.1 Zero Frequency

**Objective**: Commands with count=0 produce frequency score of 0.0

```toml
[meta]
description = "Verify zero frequency handling"
tags = ["frequency", "edge"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 0.0
w_frequency = 1.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "single_cmd"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 1

[[scenario]]
name = "minimal_frequency"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[[scenario.expect.candidate]]
cmd = "single_cmd"
min_frequency = 0.69  # ln(1+1) = ln(2) ≈ 0.693
max_frequency = 0.70
```

#### 9.2.2 High Frequency Logarithmic Behavior

**Objective**: Verify logarithmic scaling at high frequencies

```toml
[meta]
description = "Verify logarithmic scaling for high frequency commands"
tags = ["frequency", "edge", "scaling"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 0.0
w_frequency = 1.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "freq_100"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 100

[[history]]
cmd = "freq_1000"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 1000

[[scenario]]
name = "high_frequency_scaling"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[[scenario.expect.candidate]]
cmd = "freq_100"
min_frequency = 4.6  # ln(101) ≈ 4.615
max_frequency = 4.7

[[scenario.expect.candidate]]
cmd = "freq_1000"
min_frequency = 6.9  # ln(1001) ≈ 6.909
max_frequency = 7.0
```

#### 9.2.3 Equal Frequency Tie-Breaking

**Objective**: Commands with equal frequency are tie-broken by recency or alphabetically

```toml
[meta]
description = "Verify tie-breaking for equal frequency"
tags = ["frequency", "edge", "tiebreak"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 0.0
w_frequency = 1.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "cmd_a"
at = "-2h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "cmd_b"
at = "-1h"  # More recent
cwd = "/tmp"
exit = 0
session = "s1"
count = 10  # Same frequency

[[scenario]]
name = "equal_frequency_tiebreak"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# Should be ordered by recency as tiebreaker
top = ["cmd_b", "cmd_a"]
```

### 9.3 Transition Edge Cases

#### 9.3.1 No Previous Command

**Objective**: Transition scoring when there's no previous command

```toml
[meta]
description = "Verify behavior when no previous command exists"
tags = ["transition", "edge"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 0.0
w_frequency = 0.0
w_transition = 1.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "any_cmd"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "no_previous_command"
mode = "next_command"
# No prev_command specified

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# With only transition weight and no prev_command, should have no results
# or all results have 0 transition score
empty = false  # May still return results from other sources
```

#### 9.3.2 Long Transition Chains

**Objective**: Multi-hop transitions (A→B→C→D) patterns

```toml
[meta]
description = "Verify long transition chains"
tags = ["transition", "edge", "chain"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 0.0
w_frequency = 0.0
w_transition = 1.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

# Establish a chain: make → test → deploy → notify
[[history]]
cmd = "make"
at = "-10m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "test"
at = "-9m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "deploy"
at = "-8m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "notify"
at = "-7m"
cwd = "/tmp"
exit = 0
session = "s1"

# Repeat the chain
[[history]]
cmd = "make"
at = "-6m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "test"
at = "-5m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "deploy"
at = "-4m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "notify"
at = "-3m"
cwd = "/tmp"
exit = 0
session = "s1"

[[scenario]]
name = "chain_step_1"
mode = "next_command"
prev_command = "make"
prev_exit = 0

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
top = ["test"]

[[scenario]]
name = "chain_step_2"
mode = "next_command"
prev_command = "test"
prev_exit = 0

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
top = ["deploy"]

[[scenario]]
name = "chain_step_3"
mode = "next_command"
prev_command = "deploy"
prev_exit = 0

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
top = ["notify"]
```

### 9.4 Context Edge Cases

#### 9.4.1 Deep Subdirectory Nesting

**Objective**: Context inheritance works at 4+ levels deep

```toml
[meta]
description = "Verify context inheritance at deep nesting levels"
tags = ["context", "cwd", "edge", "deep"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 1.0
w_sequence = 0.0
w_similarity = 0.0

[fs]
"project/.git/" = "dir"
"project/src/lib/util/helpers/" = "dir"

[[history]]
cmd = "root_cmd"
at = "-1h"
cwd = "project"
exit = 0
session = "s1"
count = 10

[[scenario]]
name = "deep_nesting_context"
mode = "next_command"

[scenario.context]
cwd = "project/src/lib/util/helpers"
session = "s1"

[scenario.expect]
contains = ["root_cmd"]

[[scenario.expect.candidate]]
cmd = "root_cmd"
min_context = 0.1  # Should still get context from repo root
```

#### 9.4.2 Non-Git Repository (No .git marker)

**Objective**: Directories without .git don't get repo-specific context

```toml
[meta]
description = "Verify non-git directories don't get repo context"
tags = ["context", "repo", "edge"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 0.0
w_frequency = 1.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[fs]
"project_with_git/.git/" = "dir"
"project_no_git/" = "dir"  # No .git

[[history]]
cmd = "git_project_cmd"
at = "-1h"
cwd = "project_with_git"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "no_git_cmd"
at = "-1h"
cwd = "project_no_git"
exit = 0
session = "s1"
count = 10

[[scenario]]
name = "no_git_marker"
mode = "next_command"

[scenario.context]
cwd = "project_no_git"
session = "s1"

[scenario.expect]
# Both should appear since frequency is isolated
# but git_project_cmd shouldn't get repo_freq boost
contains = ["no_git_cmd", "git_project_cmd"]
```

#### 9.4.3 Host Switching

**Objective**: Same history, different host context

```toml
[meta]
description = "Verify host context isolation"
tags = ["context", "hostname", "edge"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 1.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "production_cmd"
at = "-1h"
cwd = "/app"
hostname = "prod-server-1"
exit = 0
session = "s1"
count = 20

[[history]]
cmd = "staging_cmd"
at = "-1h"
cwd = "/app"
hostname = "staging-server"
exit = 0
session = "s1"
count = 20

[[history]]
cmd = "local_cmd"
at = "-1h"
cwd = "/app"
hostname = "dev-laptop"
exit = 0
session = "s1"
count = 20

[[scenario]]
name = "host_prod"
mode = "next_command"

[scenario.context]
cwd = "/app"
hostname = "prod-server-1"
session = "s1"

[scenario.expect]
top = ["production_cmd"]

[[scenario]]
name = "host_staging"
mode = "next_command"

[scenario.context]
cwd = "/app"
hostname = "staging-server"
session = "s1"

[scenario.expect]
top = ["staging_cmd"]
```

### 9.5 Sequence Edge Cases

#### 9.5.1 Conflicting Sequences

**Objective**: Multiple valid sequences compete

```toml
[meta]
description = "Verify conflicting sequence resolution"
tags = ["sequence", "edge", "conflict"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 1.0
w_similarity = 0.0

[options]
use_sequences = true
run_sequence_analysis = true
min_sequence_support = 2

# Pattern 1: git add → git commit (3 times)
[[history]]
cmd = "git add"
at = "-30m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git commit"
at = "-29m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git add"
at = "-28m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git commit"
at = "-27m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git add"
at = "-26m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git commit"
at = "-25m"
cwd = "/tmp"
exit = 0
session = "s1"

# Pattern 2: git add → git status (2 times)
[[history]]
cmd = "git add"
at = "-20m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git status"
at = "-19m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git add"
at = "-18m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "git status"
at = "-17m"
cwd = "/tmp"
exit = 0
session = "s1"

[[scenario]]
name = "conflicting_sequences"
mode = "next_command"
prev_command = "git add"
prev_exit = 0

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# git commit should win with higher support
top = ["git commit"]
contains = ["git status"]
```

#### 9.5.2 Sequences Disabled

**Objective**: Verify sequences don't affect results when disabled

```toml
[meta]
description = "Verify sequences can be disabled"
tags = ["sequence", "edge", "disabled"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 1.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0  # Sequence weight is zero
w_similarity = 0.0

[options]
use_sequences = false

[[history]]
cmd = "sequence_a"
at = "-10m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "sequence_b"
at = "-9m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "sequence_a"
at = "-8m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "sequence_b"
at = "-7m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "recent_only"
at = "-1m"
cwd = "/tmp"
exit = 0
session = "s1"

[[scenario]]
name = "sequences_disabled"
mode = "next_command"
prev_command = "sequence_a"
prev_exit = 0

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# Most recent command should win since only recency is enabled
top = ["recent_only"]
```

### 9.6 Structure Edge Cases

#### 9.6.1 Cursor Position Mid-Word

**Objective**: Completion with cursor not at end of input

```toml
[meta]
description = "Verify completion with cursor mid-word"
tags = ["structure", "cursor", "edge"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "git status"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "git stash"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "cursor_mid_word"
mode = "completion"
input = "git staXXX"
cursor = 6  # Cursor at position 6, after "git sta"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["git status", "git stash"]
```

#### 9.6.2 Very Long Commands

**Objective**: Handle commands exceeding typical length

```toml
[meta]
description = "Verify handling of very long commands"
tags = ["structure", "edge", "long"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "kubectl exec -it pod-name-very-long-identifier-12345 --namespace production-environment -- /bin/bash -c 'cat /var/log/application.log | grep ERROR | tail -n 100'"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "long_command_completion"
mode = "completion"
input = "kubectl exec -it pod"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
min_results = 1
```

#### 9.6.3 Command Substitution

**Objective**: Handle $() and backticks correctly

```toml
[meta]
description = "Verify command substitution tokenization"
tags = ["structure", "edge", "substitution"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "echo $(date +%Y-%m-%d)"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[history]]
cmd = "echo `hostname`"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 3

[[scenario]]
name = "dollar_paren_substitution"
mode = "completion"
input = "echo $("

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["echo $(date +%Y-%m-%d)"]

[[scenario]]
name = "backtick_substitution"
mode = "completion"
input = "echo `"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["echo `hostname`"]
```

#### 9.6.4 Redirection Operators

**Objective**: Handle >, >>, <, 2>&1 correctly

```toml
[meta]
description = "Verify redirection operator handling"
tags = ["structure", "edge", "redirection"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "make build > build.log 2>&1"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[history]]
cmd = "cat input.txt < data.csv"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 3

[[history]]
cmd = "echo 'log' >> app.log"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 4

[[scenario]]
name = "redirect_stdout"
mode = "completion"
input = "make build >"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["make build > build.log 2>&1"]

[[scenario]]
name = "redirect_append"
mode = "completion"
input = "echo 'log' >>"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["echo 'log' >> app.log"]
```

#### 9.6.5 Background and Control Operators

**Objective**: Handle &, &&, ||, ; correctly

```toml
[meta]
description = "Verify control operator handling"
tags = ["structure", "edge", "operators"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "sleep 100 &"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 3

[[history]]
cmd = "make build && make test"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "make build || echo 'build failed'"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[history]]
cmd = "cd /tmp; ls -la"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 4

[[scenario]]
name = "and_chain"
mode = "completion"
input = "make build &&"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["make build && make test"]

[[scenario]]
name = "or_chain"
mode = "completion"
input = "make build ||"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["make build || echo 'build failed'"]

[[scenario]]
name = "semicolon_chain"
mode = "completion"
input = "cd /tmp;"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["cd /tmp; ls -la"]
```

#### 9.6.6 Glob Patterns

**Objective**: Handle * and ? in commands

```toml
[meta]
description = "Verify glob pattern handling"
tags = ["structure", "edge", "glob"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "ls *.txt"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "rm -f temp?.log"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[history]]
cmd = "find . -name '*.rs'"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 8

[[scenario]]
name = "star_glob"
mode = "completion"
input = "ls *"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["ls *.txt"]

[[scenario]]
name = "question_glob"
mode = "completion"
input = "rm -f temp?"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["rm -f temp?.log"]
```

#### 9.6.7 Unicode in Commands

**Objective**: Handle non-ASCII characters correctly

```toml
[meta]
description = "Verify Unicode character handling"
tags = ["structure", "edge", "unicode"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "echo 'Héllo Wörld'"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[history]]
cmd = "cat 日本語ファイル.txt"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 3

[[history]]
cmd = "echo '🚀 Deployment started'"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 4

[[scenario]]
name = "unicode_accents"
mode = "completion"
input = "echo 'Héllo"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["echo 'Héllo Wörld'"]

[[scenario]]
name = "unicode_emoji"
mode = "completion"
input = "echo '🚀"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["echo '🚀 Deployment started'"]
```

### 9.7 Anti-Hallucination Edge Cases

#### 9.7.1 Case Sensitivity

**Objective**: Verify case-sensitive prefix matching

```toml
[meta]
description = "Verify case sensitivity in prefix matching"
tags = ["anti-hallucination", "edge", "case"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "Git status"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[history]]
cmd = "git status"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "GIT status"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 3

[[scenario]]
name = "lowercase_prefix"
mode = "completion"
input = "git"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["git status"]
absent = ["Git status", "GIT status"]

[[scenario]]
name = "uppercase_prefix"
mode = "completion"
input = "Git"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["Git status"]
absent = ["git status", "GIT status"]
```

#### 9.7.2 Whitespace Handling

**Objective**: Leading/trailing whitespace in prefix

```toml
[meta]
description = "Verify whitespace handling in prefix"
tags = ["anti-hallucination", "edge", "whitespace"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "git status"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[scenario]]
name = "leading_space"
mode = "completion"
input = " git"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# Leading space means no command starts with " git"
empty = true

[[scenario]]
name = "trailing_spaces"
mode = "completion"
input = "git   "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["git status"]
```

#### 9.7.3 Partial vs Prefix Match

**Objective**: Only prefix matches, not substring matches

```toml
[meta]
description = "Verify only prefix matching, not substring"
tags = ["anti-hallucination", "edge", "substring"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "git status"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "fugitive git"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "prefix_not_substring"
mode = "completion"
input = "git"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["git status"]
absent = ["fugitive git"]  # "git" is substring, not prefix
```

### 9.8 Alias Edge Cases

#### 9.8.1 Alias with Arguments

**Objective**: Aliases that include their own arguments

```toml
[meta]
description = "Verify alias with embedded arguments"
tags = ["aliases", "edge"]

[physics]
now = "2024-06-15T12:00:00Z"

[aliases]
ll = "ls -la"
glog = "git log --oneline --graph"

[[history]]
cmd = "ls -la /tmp"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "git log --oneline --graph --all"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "alias_with_extra_args"
mode = "completion"
input = "ll "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["ll /tmp"]

[[scenario]]
name = "alias_extends_history"
mode = "completion"
input = "glog "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
contains = ["glog --all"]
```

#### 9.8.2 Alias Shadowing

**Objective**: Alias shadows a real command name

```toml
[meta]
description = "Verify alias shadowing behavior"
tags = ["aliases", "edge", "shadow"]

[physics]
now = "2024-06-15T12:00:00Z"

[aliases]
ls = "exa --icons"  # Shadows the real 'ls' command

[[history]]
cmd = "exa --icons -la"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[history]]
cmd = "ls -la"
at = "-2h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "alias_shadows_command"
mode = "completion"
input = "ls "

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# Both alias expansion and direct command should work
min_results = 1
```

### 9.9 Phase Edge Cases

#### 9.9.1 Phase Transitions

**Objective**: Rapid phase changes in session

```toml
[meta]
description = "Verify phase detection with rapid transitions"
tags = ["phase", "edge", "transitions"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 1.0
w_sequence = 0.0
w_similarity = 0.0

[options]
run_phase_indexing = true

[phases]
build = ["cargo build", "make"]
test = ["cargo test", "pytest"]
deploy = ["kubectl apply", "docker push"]

# Quick phase transition: build → test → deploy
[[history]]
cmd = "cargo build"
at = "-5m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "cargo test"
at = "-4m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "kubectl apply -f deployment.yaml"
at = "-3m"
cwd = "/tmp"
exit = 0
session = "s1"

[[history]]
cmd = "docker push myimage"
at = "-2m"
cwd = "/tmp"
exit = 0
session = "s1"
count = 5

[[scenario]]
name = "recent_phase_wins"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# Most recent phase is deploy
top = ["docker push myimage", "kubectl apply -f deployment.yaml"]
```

### 9.10 Integration Edge Cases

#### 9.10.1 All Weights Zero

**Objective**: Edge case where all weights are zero

```toml
[meta]
description = "Verify behavior when all weights are zero"
tags = ["integration", "edge", "zero"]

[physics]
now = "2024-06-15T12:00:00Z"
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

[[history]]
cmd = "any_command"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[scenario]]
name = "all_zero_weights"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# With all weights zero, only session_recency (hardcoded 0.1) applies
# or the system should gracefully handle this edge case
min_results = 0  # May return nothing or may rely on session_recency
```

#### 9.10.2 Large History Volume

**Objective**: Performance with large history

```toml
[meta]
description = "Verify performance with large history"
tags = ["integration", "edge", "performance"]

[physics]
now = "2024-06-15T12:00:00Z"

# Simulate large history with count
[[history]]
cmd = "common_cmd"
at = "-1d"
cwd = "/tmp"
exit = 0
session = "s1"
count = 1000

[[history]]
cmd = "rare_cmd"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 1

[[scenario]]
name = "large_history_query"
mode = "next_command"

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# Should still return results in reasonable time
min_results = 1
max_results = 10
```

#### 9.10.3 Empty Prefix in Completion Mode

**Objective**: Completion mode with empty prefix

```toml
[meta]
description = "Verify completion with empty prefix"
tags = ["integration", "edge", "empty"]

[physics]
now = "2024-06-15T12:00:00Z"

[[history]]
cmd = "git status"
at = "-1h"
cwd = "/tmp"
exit = 0
session = "s1"
count = 10

[[scenario]]
name = "empty_prefix_completion"
mode = "completion"
input = ""

[scenario.context]
cwd = "/tmp"
session = "s1"

[scenario.expect]
# Empty prefix should behave like next_command mode
min_results = 1
```

---

## 10. Updated Coverage Checklist

### 10.1 Scoring Components (Extended)

| Component | Test File | Status |
|-----------|-----------|--------|
| Recency decay | `recency/decay_curve.toml` | ☐ |
| Recency half-life | `recency/half_life.toml` | ☐ |
| Recency future timestamps | `recency/future_timestamps.toml` | ☐ |
| Recency invalid timestamps | `recency/invalid_timestamps.toml` | ☐ |
| Recency multi-half-life | `recency/multi_half_life.toml` | ☐ |
| Frequency basic | `frequency/basic_ranking.toml` | ☐ |
| Frequency repo boost | `frequency/repo_boost.toml` | ☐ |
| Frequency zero | `frequency/zero_frequency.toml` | ☐ |
| Frequency high scaling | `frequency/high_scaling.toml` | ☐ |
| Frequency tiebreak | `frequency/tiebreak.toml` | ☐ |
| Transition basic | `transition/basic_chain.toml` | ☐ |
| Transition exit status | `transition/exit_status.toml` | ☐ |
| Transition no previous | `transition/no_previous.toml` | ☐ |
| Transition repo-specific | `transition/repo_specific.toml` | ☐ |
| Transition long chains | `transition/long_chains.toml` | ☐ |
| Context CWD | `context/cwd.toml` | ☐ |
| Context subdirectory | `context/subdirectory.toml` | ☐ |
| Context parent inheritance | `context/parent_inheritance.toml` | ☐ |
| Context deep nesting | `context/deep_nesting.toml` | ☐ |
| Context session | `context/session.toml` | ☐ |
| Context session recency | `context/session_recency_weight.toml` | ☐ |
| Context hostname/user | `context/hostname.toml` | ☐ |
| Context host switching | `context/host_switching.toml` | ☐ |
| Context non-git repo | `context/non_git_repo.toml` | ☐ |
| Sequence bigram | `sequence/bigram.toml` | ☐ |
| Sequence trigram | `sequence/trigram.toml` | ☐ |
| Sequence disabled | `sequence/disabled.toml` | ☐ |
| Sequence low confidence | `sequence/low_confidence.toml` | ☐ |
| Sequence conflicts | `sequence/conflicts.toml` | ☐ |
| Similarity tokens | `similarity/token_matching.toml` | ☐ |
| Similarity no overlap | `similarity/no_overlap.toml` | ☐ |

### 10.2 Structural Semantics (Extended)

| Component | Test File | Status |
|-----------|-----------|--------|
| Flag commutativity | `structure/flag_commutativity.toml` | ☐ |
| Arg position strictness | `structure/arg_position.toml` | ☐ |
| Flag-arg binding | `structure/flag_arg_binding.toml` | ☐ |
| Quoted arguments | `structure/quotes.toml` | ☐ |
| Environment prefix | `structure/env_prefix.toml` | ☐ |
| Pipe handling | `structure/pipes.toml` | ☐ |
| Cursor position | `structure/cursor.toml` | ☐ |
| Long commands | `structure/long_commands.toml` | ☐ |
| Command substitution | `structure/command_substitution.toml` | ☐ |
| Redirection | `structure/redirection.toml` | ☐ |
| Control operators | `structure/control_operators.toml` | ☐ |
| Glob patterns | `structure/glob_patterns.toml` | ☐ |
| Unicode | `structure/unicode.toml` | ☐ |

### 10.3 Anti-Hallucination (Extended)

| Component | Test File | Status |
|-----------|-----------|--------|
| Empty history | `anti_hallucination/empty_history.toml` | ☐ |
| Strict prefix | `anti_hallucination/strict_prefix.toml` | ☐ |
| Ghost candidates | `anti_hallucination/ghost_candidates.toml` | ☐ |
| Cross-contamination | `anti_hallucination/cross_contamination.toml` | ☐ |
| Case sensitivity | `anti_hallucination/case_sensitivity.toml` | ☐ |
| Whitespace handling | `anti_hallucination/whitespace.toml` | ☐ |
| Substring vs prefix | `anti_hallucination/substring_prefix.toml` | ☐ |

### 10.4 Features (Extended)

| Component | Test File | Status |
|-----------|-----------|--------|
| Alias expansion | `aliases/expansion.toml` | ☐ |
| Alias suffix | `aliases/suffix.toml` | ☐ |
| Alias with args | `aliases/with_args.toml` | ☐ |
| Alias shadowing | `aliases/shadowing.toml` | ☐ |
| Phase detection | `phase/detection.toml` | ☐ |
| Phase transitions | `phase/transitions.toml` | ☐ |
| Arg templates | `templates/args.toml` | ☐ |
| Flag templates | `templates/flags.toml` | ☐ |

### 10.5 Integration Tests

| Component | Test File | Status |
|-----------|-----------|--------|
| Competing signals | `integration/competing_signals.toml` | ☐ |
| Default weights | `integration/default_weights.toml` | ☐ |
| Full stack | `integration/full_stack.toml` | ☐ |
| DB assertions | `integration/db_assertions.toml` | ☐ |
| All zero weights | `integration/zero_weights.toml` | ☐ |
| Large history | `integration/large_history.toml` | ☐ |
| Empty prefix | `integration/empty_prefix.toml` | ☐ |

---

## 11. Implementation Notes

### 11.1 Files Already Implemented

The following test files already exist in `src/testdata/tier1/` and should be verified against this specification:

- `recency/decay_curve.toml`, `half_life.toml`, `future_timestamps.toml`, `invalid_timestamps.toml`
- `frequency/basic_ranking.toml`, `repo_boost.toml`, `zero_frequency.toml`
- `transition/basic_chain.toml`, `exit_status.toml`, `no_previous.toml`, `repo_specific.toml`
- `context/cwd.toml`, `hostname.toml`, `parent_inheritance.toml`, `session_recency_weight.toml`, `session.toml`, `subdirectory.toml`
- `sequence/bigram.toml`, `disabled.toml`, `low_confidence.toml`, `trigram.toml`
- `similarity/no_overlap.toml`, `token_matching.toml`
- `structure/arg_position.toml`, `cursor.toml`, `env_prefix.toml`, `flag_arg_binding.toml`, `flag_commutativity.toml`, `pipes.toml`, `quotes.toml`
- `anti_hallucination/cross_contamination.toml`, `empty_history.toml`, `ghost_candidates.toml`, `strict_prefix.toml`
- `aliases/expansion.toml`, `suffix.toml`
- `phase/detection.toml`
- `templates/args.toml`, `flags.toml`
- `integration/competing_signals.toml`, `db_assertions.toml`, `default_weights.toml`, `full_stack.toml`

### 11.2 Files Needing Addition

Based on the gap analysis, the following new test files should be created:

1. `recency/multi_half_life.toml` - Verify decay across multiple half-lives
2. `frequency/high_scaling.toml` - Verify logarithmic behavior at high frequencies
3. `frequency/tiebreak.toml` - Verify tie-breaking behavior
4. `transition/long_chains.toml` - Verify multi-hop transition chains
5. `context/deep_nesting.toml` - Verify 4+ level subdirectory inheritance
6. `context/non_git_repo.toml` - Verify behavior without .git marker
7. `context/host_switching.toml` - Verify host context isolation
8. `sequence/conflicts.toml` - Verify conflicting sequence resolution
9. `structure/long_commands.toml` - Verify handling of very long commands
10. `structure/command_substitution.toml` - Verify $() and backtick handling
11. `structure/redirection.toml` - Verify >, >>, < handling
12. `structure/control_operators.toml` - Verify &&, ||, ;, & handling
13. `structure/glob_patterns.toml` - Verify * and ? handling
14. `structure/unicode.toml` - Verify non-ASCII character handling
15. `anti_hallucination/case_sensitivity.toml` - Verify case-sensitive matching
16. `anti_hallucination/whitespace.toml` - Verify whitespace handling
17. `anti_hallucination/substring_prefix.toml` - Verify only prefix matches
18. `aliases/with_args.toml` - Verify aliases with embedded arguments
19. `aliases/shadowing.toml` - Verify alias shadowing behavior
20. `phase/transitions.toml` - Verify rapid phase transitions
21. `integration/zero_weights.toml` - Verify all-zero weights edge case
22. `integration/large_history.toml` - Verify performance with large history
23. `integration/empty_prefix.toml` - Verify empty prefix behavior

### 11.3 Priority Order for Implementation

**P0 - Critical (must have for correctness):**
1. Anti-hallucination edge cases (case sensitivity, whitespace, substring)
2. Structure edge cases (redirection, control operators, command substitution)
3. Context edge cases (non-git repo, deep nesting)

**P1 - High (important for robustness):**
4. Frequency edge cases (tiebreak, high scaling)
5. Transition edge cases (long chains)
6. Sequence edge cases (conflicts)

**P2 - Medium (good to have):**
7. Unicode handling
8. Alias edge cases
9. Phase transitions

**P3 - Low (nice to have):**
10. Integration edge cases (zero weights, large history)
11. Glob patterns
12. Multi-half-life recency