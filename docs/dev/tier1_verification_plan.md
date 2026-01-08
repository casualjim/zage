# Tier-1 Verification Plan: "Deep State"

## Objective

Establish a high-fidelity, deterministic verification framework for the Tier-1 statistical retrieval engine. This framework must go beyond simple input/output pairs to verify the *internal physics* of the ranking model: probability distributions, decay functions, context affinity, and structure semantics.

## The Core Problem

The current testing approach treats the engine as a black box. We need a "glass box" approach where we can:
1.  **Freeze Time**: precise control over `now` to test recency decay curves.
2.  **Mock Physics**: override gravitational constants (weights) to test forces in isolation.
3.  **Real Context**: create actual temporary directory structures to verify repo detection integration.
4.  **Inspect Internals**: assert on intermediate scores (e.g., "Recency Score must be > 0.8") not just final rank.

## Schema Specification

The test runner will ingest a TOML file defining a "World" and a set of "Scenarios".

### 1. Global Configuration & Physics

Allows overriding hardcoded constants in `src/predict.rs` to verify specific mechanics.

```toml
[meta]
description = "Verifies recency decay half-life behavior"

# Override default ranking weights to isolate components
[physics]
w_recency = 1.0    # Test pure recency
w_frequency = 0.0
w_transition = 0.0
w_context = 0.0
w_sequence = 0.0
w_similarity = 0.0

# Mock the "Current Time" for the test execution
now = "2024-06-15T12:00:00Z"
```

### 2. Ephemeral Filesystem

Instead of mocking filesystem calls, we will create a real, isolated directory structure using `tempfile`. The test runner will resolve relative paths in the TOML to the absolute path of the temp directory. This ensures `find_repo_root` works exactly as it does in production.

```toml
[fs]
# Define the structure relative to the test's temp root
"project_a/.git/" = "dir"      # Creates a .git directory (repo root marker)
"project_a/src/" = "dir"
"notes/todo.txt" = "file"
```

When the test runs:
1. A temp dir is created (e.g., `/tmp/zage_test_123`).
2. The structure is built inside it.
3. If a test specifies `cwd = "project_a/src"`, the engine receives `/tmp/zage_test_123/project_a/src`.

### 3. History Injection (The Corpus)

We define history events with relative timestamps and rich context. Note that `cwd` here can refer to the paths defined in `[fs]`.

```toml
[[history]]
cmd = "cargo test"
at = "-10m"            # 10 minutes before 'now'
cwd = "project_a"      # Resolves to /tmp/.../project_a
exit = 0
session = "term_1"

[[history]]
cmd = "npm test"
at = "-1y"             # 1 year ago
cwd = "frontend_repo"  # Directory might not even exist in FS for this test, which is valid state
count = 50             # Bulk insert simulation
```

### 4. Scenarios (Queries & Assertions)

A single test file can contain multiple query scenarios against the same seeded history.

#### Scenario A: Simple Ranking

```toml
[[scenario]]
name = "Repo Affinity Test"
input = "test"
cursor = 4

[scenario.context]
cwd = "project_a/src" # Inside the real git repo created in [fs]

[scenario.expect]
# Simple ranking assertion
top = ["cargo test"]
absent = ["npm test"] 
```

#### Scenario B: Deep Inspection

This is the critical upgrade. We verify *why* a result was chosen.

```toml
[[scenario]]
name = "Decay Curve Verification"
input = "cargo"

[scenario.expect.candidate]
cmd = "cargo test"

# Assert on specific sub-scores (0.0 - 1.0)
# Derived from `ScoreBreakdown` struct in predict.rs
min_recency_score = 0.95 
max_frequency_score = 0.1

[scenario.expect.db]
# verify the indexer did its job correctly
# checks raw SQL counts in `command_stats` table
sql = "SELECT freq FROM command_stats WHERE command = 'cargo test'"
value = 1
```

## Verification Matrix

The plan requires implementing tests for these specific mechanical interactions.

### 1. The Time Machine (Recency)
*   **Decay Function**: With `w_recency=1.0`, verify that `score(t-1h) > score(t-1d) > score(t-1w)`.
*   **Half-Life Check**: Verify that `score(t - half_life) ≈ 0.5`.

### 2. Context Gravity
*   **Repo Isolation**: 
    *   Setup: `make build` in Repo A, `mvn build` in Repo B.
    *   Query: "build" while CWD is inside Repo A.
    *   Expect: `make build` >> `mvn build`.
*   **Directory Fallback**:
    *   Query in a subdir (`/repo/src/lib`) should still match history from root (`/repo`).

### 3. Session Short-Term Memory
*   **The "Up Arrow" Simulator**:
    *   Inject `cmd_A` with `session_id=current`.
    *   Inject `cmd_A` with `session_id=other` (older).
    *   Verify `current` session boost logic (often handled via `Phase` or distinct session features).

### 4. Transition & Sequence (Markov)
*   **Chain Reaction**:
    *   History: `git add .` -> `git commit`.
    *   Query Context: Prev command = `git add .`.
    *   Expect: `git commit` has `transition_score > 0`.
*   **Repo-Specific Chains**:
    *   Repo A: `git checkout` -> `main`.
    *   Repo B: `git checkout` -> `master`.
    *   Verify correct branch suggested based on CWD.

### 5. Structure Semantics & Syntax
*   **Flag Commutativity (Position Independence)**:
    *   History: `ls -a -l`.
    *   Query: `ls -l` -> Expect `ls -l -a` (or `ls -a -l`).
    *   Query: `ls -a` -> Expect `ls -a -l`.
    *   The retrieval engine must recognize that flags can be reordered (bag-of-flags model).
*   **Argument Position Strictness**:
    *   History: `cp source dest`.
    *   Query: `cp dest` -> Expect **NO** suggestion (or very low score). `dest` is arg position 2, but query puts it in position 1.
*   **Flag-Argument Binding**:
    *   History: `gcc -o main`.
    *   Query: `gcc -o` -> Expect `main`.
    *   Query: `gcc main` -> Expect **NO** suggestion of `-o`. The `-o` flag *requires* an argument, and `main` is that argument.

### 6. Anti-Hallucination (Recall Precision)
*   **The "Clean Slate" Rule**:
    *   Setup: Empty history.
    *   Query: `git c` -> Expect **Empty**. Tier 1 should not hallucinate commands not in history (unless explicit templates/aliases are loaded).
*   **No Ghost Candidates**:
    *   History: `git commit -m "fix"`.
    *   Query: `git commit -a` -> Expect **Empty**. We should not suggest completion if the user has already typed something (`-a`) that contradicts the history (`-m`), even if the prefix matches `git commit`.
*   **Strict Prefix Adherence**:
    *   History: `cargo build`.
    *   Query: `cargo b` -> Match.
    *   Query: `cargo x` -> **No Match**. Fuzzy matching should be strictly controlled or disabled for Tier 1 to prevent noise.

## Implementation Roadmap

### Phase 1: Test Harness (`src/predict/verifier.rs`)
1.  **Refactor `SuggestConfig`**: Ensure it can accept a `Box<dyn TimeProvider>` for deterministic time.
2.  **Refactor `RankingWeights`**: Make it configurable at runtime via `SuggestConfig` rather than strictly compile-time default.
3.  **Instrumentation**: Add a `debug: bool` flag to `suggest()` that returns the full `ScoreBreakdown` for every candidate, not just the final score.

### Phase 2: Test Infrastructure
1.  **Filesystem Fixture**: A helper that accepts the `[fs]` TOML block and materializes it to a `tempfile::TempDir`. It should auto-clean on drop.
2.  **SQL Loader**: A helper to translate the TOML `[[history]]` block into standard SQLite `INSERT` statements for `shell_history`, then triggering the `indexer::rebuild_stats`.

### Phase 3: Case Migration
1.  Convert high-value existing cases.
2.  Write the "Physics Verification" suite (isolating each weight).
3.  Write the "Structure & Semantics" suite (flags vs args).

## Definition of Done
- [ ] Test harness supports `[physics]` weight overrides.
- [ ] Test harness supports virtual timestamps (`at = "-5m"`).
- [ ] Test harness supports real ephemeral filesystems defined in TOML.
- [ ] Test harness can inspect `ScoreBreakdown`.
- [ ] "Tier-1 Mechanics" test suite passes in CI with anti-hallucination checks.