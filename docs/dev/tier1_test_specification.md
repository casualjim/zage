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

Default weights:
- `w_recency = 0.25`
- `w_frequency = 0.20`
- `w_transition = 0.15`
- `w_context = 0.20`
- `w_sequence = 0.10`
- `w_similarity = 0.10`

Each sub-score is computed as:
- **recency**: `exp(-age / half_life)` where `half_life = 7 days`
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

# Override ranking weights (omit to use defaults)
w_recency = 1.0
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

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
min_recency = 0.99    # 1 hour ago ≈ exp(-3600/604800) ≈ 0.994

[[scenario.expect.candidate]]
cmd = "day_old_cmd"
min_recency = 0.86    # 1 day ago ≈ exp(-86400/604800) ≈ 0.867
max_recency = 0.88

[[scenario.expect.candidate]]
cmd = "week_old_cmd"
min_recency = 0.36    # 1 week ago ≈ exp(-604800/604800) ≈ 0.368
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
recency_half_life_seconds = 86400  # 1 day for easier testing

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
min_recency = 0.36
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
min_frequency = 3.9  # ln(50+1) ≈ 3.93

[[scenario.expect.candidate]]
cmd = "common_cmd"
min_frequency = 2.3  # ln(10+1) ≈ 2.40
max_frequency = 2.5

[[scenario.expect.candidate]]
cmd = "rare_cmd"
min_frequency = 0.6  # ln(1+1) ≈ 0.69
max_frequency = 0.8
```

#### 3.2.2 Repo Frequency Boost

**Objective**: Commands frequently used in current repo get boosted

```toml
[meta]
description = "Verify repo-specific frequency boost"
tags = ["frequency", "repo"]

[physics]
now = "2024-06-15T12:00:00Z"
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
w_recency = 0.0
w_frequency = 0.0
w_transition = 0.0
w_context = 1.0  # Phase boost is part of context
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
/// Configuration for deterministic testing
pub struct TestConfig {
    /// Frozen timestamp for recency calculations
    pub now: Option<i64>,
    
    /// Override ranking weights
    pub weights: Option<RankingWeights>,
    
    /// Override recency half-life
    pub recency_half_life: Option<f64>,
    
    /// Enable debug mode for score breakdown
    pub debug: bool,
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
   - Extract `SystemTime::now()` calls into injectable `TimeProvider`
   - Default implementation uses wall clock
   - Test implementation uses frozen time

2. **Configurable Weights**
   - Make `RankingWeights` a parameter to `suggest()` or part of `SuggestConfig`
   - Current hardcoded `RankingWeights::default()` becomes the default

3. **Score Breakdown Exposure**
   - Ensure `ScoreBreakdown` is always populated (currently is)
   - Add `rank` field to test output

4. **Filesystem Fixture**
   - Helper to materialize `[fs]` block into `tempfile::TempDir`
   - Path resolution: relative paths → absolute paths in temp dir

5. **Session ID Resolution**
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
| Recency decay | `recency/decay_curve.toml` | ☐ |
| Recency half-life | `recency/half_life.toml` | ☐ |
| Frequency basic | `frequency/basic_ranking.toml` | ☐ |
| Frequency repo boost | `frequency/repo_boost.toml` | ☐ |
| Transition basic | `transition/basic_chain.toml` | ☐ |
| Transition exit status | `transition/exit_status.toml` | ☐ |
| Transition repo-specific | `transition/repo_specific.toml` | ☐ |
| Context CWD | `context/cwd.toml` | ☐ |
| Context subdirectory | `context/subdirectory.toml` | ☐ |
| Context session | `context/session.toml` | ☐ |
| Context hostname/user | `context/hostname.toml` | ☐ |
| Sequence bigram | `sequence/bigram.toml` | ☐ |
| Sequence trigram | `sequence/trigram.toml` | ☐ |
| Similarity tokens | `similarity/token_matching.toml` | ☐ |

### 6.2 Structural Semantics

| Component | Test File | Status |
|-----------|-----------|--------|
| Flag commutativity | `structure/flag_commutativity.toml` | ☐ |
| Arg position strictness | `structure/arg_position.toml` | ☐ |
| Flag-arg binding | `structure/flag_arg_binding.toml` | ☐ |
| Quoted arguments | `structure/quotes.toml` | ☐ |
| Environment prefix | `structure/env_prefix.toml` | ☐ |
| Pipe handling | `structure/pipes.toml` | ☐ |

### 6.3 Anti-Hallucination

| Component | Test File | Status |
|-----------|-----------|--------|
| Empty history | `anti_hallucination/empty_history.toml` | ☐ |
| Strict prefix | `anti_hallucination/strict_prefix.toml` | ☐ |
| Ghost candidates | `anti_hallucination/ghost_candidates.toml` | ☐ |
| Cross-contamination | `anti_hallucination/cross_contamination.toml` | ☐ |

### 6.4 Features

| Component | Test File | Status |
|-----------|-----------|--------|
| Alias expansion | `aliases/expansion.toml` | ☐ |
| Alias suffix | `aliases/suffix.toml` | ☐ |
| Phase detection | `phase/detection.toml` | ☐ |
| Arg templates | `templates/args.toml` | ☐ |
| Flag templates | `templates/flags.toml` | ☐ |

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