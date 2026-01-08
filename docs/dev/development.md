# Zage Development Plan (Research-Aligned)

This plan replaces the previous model/embedding roadmap. The new approach follows research-validated techniques for shell command prediction:

- **SHREC-style ranking**: candidate generation + context-aware ranking (recency, frequency, scope) with string similarity.
- **SLP-style tokenization**: shell-aware parsing and normalization to preserve operators and structure.
- **Sequence mining**: mine frequent command sequences and use support/confidence/lift in ranking.

The goal is a fast, on-device predictor that is accurate for personal histories without heavyweight embeddings.

---

## 1. Goals & Non-Goals

### Goals
- Predict next command from local shell history with low latency.
- Use **lightweight statistical signals** (recency, frequency, context) + **sequence mining**.
- Preserve shell syntax with SLP-style tokenization for better similarity and mining.
- Keep the system fully local, fast, and deterministic.

### Non-Goals
- No embedding-first retrieval or vector search.
- No external inference services.
- No requirement for GPU (optional only).

---

## 2. Architecture Overview

### Pipeline (High-Level)
1. **Ingest history** (existing parsers) into SQLite.
2. **Index** history into stats tables:
   - command frequency, recency
   - transition frequency (prev -> next)
   - context stats (cwd/host/user)
   - mined sequences (support/confidence/lift)
3. **Suggest**:
   - Tokenize current context and recent history
   - Generate candidates from multiple sources
   - Rank candidates with weighted scoring

### Key Principles
- **Candidate generation is broad, ranking is precise.**
- **Normalization and tokenization reduce noise.**
- **Context is first-class** (cwd, host, user, session).

---

## 3. Data Model & Storage

### Existing Tables (keep)
- `shell_history`

### New Tables (add)
- `command_stats`:
  - `command TEXT PRIMARY KEY`
  - `freq INTEGER NOT NULL`
  - `last_seen INTEGER NOT NULL`

- `transition_stats`:
  - `prev_command TEXT`
  - `next_command TEXT`
  - `freq INTEGER NOT NULL`
  - `last_seen INTEGER NOT NULL`
  - PRIMARY KEY (`prev_command`, `next_command`)

- `context_stats`:
  - `command TEXT`
  - `working_directory TEXT`
  - `hostname TEXT`
  - `username TEXT`
  - `freq INTEGER NOT NULL`
  - `last_seen INTEGER NOT NULL`
  - PRIMARY KEY (`command`, `working_directory`, `hostname`, `username`)

- `sequence_stats`:
  - `sequence_json TEXT PRIMARY KEY`  -- JSON array of commands
  - `support INTEGER NOT NULL`
  - `confidence REAL NOT NULL`
  - `lift REAL NOT NULL`
  - `context_json TEXT`               -- optional JSON (cwd/host/user/session)

- `token_cache`:
  - `command TEXT PRIMARY KEY`
  - `tokens_json TEXT NOT NULL`        -- raw tokens
  - `normalized_json TEXT NOT NULL`    -- normalized tokens

- `phase_stats`:
  - `command_head TEXT`
  - `phase TEXT`
  - `confidence REAL NOT NULL`
  - `freq INTEGER NOT NULL`
  - `last_seen INTEGER NOT NULL`
  - PRIMARY KEY (`command_head`, `phase`)

---

## 4. Core Modules (New / Updated)

### New Modules
- `src/tokenize/mod.rs`
  - Shell-aware tokenizer (SLP-style)
  - Preserves `|`, `&&`, `||`, `>`, `<`, `2>`, `;`, `()`
  - Recognizes quoted strings and env vars
  - Normalization rules: PATH, NUM, IP, HASH, USER, HOST

- `src/phase.rs`
  - Phase config loader (TOML)
  - Term-limited glob pattern matching (tokenized)
  - Local lightweight classifier to generalize phase labels
  - Writes `phase_stats` during indexing

- `src/indexer/mod.rs`
  - Builds stats tables from `shell_history`
  - Incremental updates based on `last_seen`

- `src/candidates/mod.rs`
  - Candidate sources:
    1) exact-prefix history matches
    2) transition stats (prev -> next)
    3) sequence stats (top support/confidence/lift)
    4) context stats (cwd/host/user)

- `src/features/mod.rs`
  - Feature extraction for ranking:
    - recency score
    - frequency score
    - context match score
    - token overlap score (Dice/Jaccard)
    - sequence strength score

- `src/ranking/mod.rs`
  - Weighted scoring and final top-K selection
  - Optional Tier-2 reranking (GBDT/linear) once candidates are generated

- `src/predict/mod.rs`
  - Orchestrates tokenize -> candidate -> rank

### Updated Modules
- `src/db.rs`:
  - Add indexing, stats queries, and sequence mining helpers.
- `src/db/schema-v0.sql`:
  - Add new stats tables and token cache.
- `src/main.rs`:
  - Replace old model CLI with `index` and `suggest`.

---

## 5. Tokenization & Normalization (SLP-Style)

### Token Categories
- words/identifiers
- flags (`-a`, `--long`)
- operators (`|`, `&&`, `||`, `;`, `>`)
- redirections (`>`, `>>`, `<`, `2>`, `2>&1`)
- substitutions (`$(...)`, `$(command)`)

### Normalization Rules
- File paths -> `PATH`
- IP addresses -> `IP`
- Numbers -> `NUM`
- Hash-like strings -> `HASH`
- Users/hosts -> `USER`, `HOST`

This normalization improves sequence mining and string similarity metrics.

---

## 6. Candidate Generation (SHREC-Style)

For a given recent history window, build candidate set by:

1. **Transition candidates**
   - From `transition_stats` with last command as key.

2. **Sequence candidates**
   - From `sequence_stats` where prefix matches recent history.

3. **Context candidates**
   - From `context_stats` matching cwd/host/user.

4. **History prefix candidates**
   - Commands starting with the current input prefix.

5. **Phase-aware candidates**
   - Boost candidates whose command heads match the current session phase.
   - Phase is learned from user history and seeded by config patterns.

Candidates are deduped and tagged with their source.

---

## 7. Ranking Model (Lightweight)

Score each candidate as a weighted sum:

```
score =
  w_recency * recency_score +
  w_freq    * frequency_score +
  w_ctx     * context_match_score +
  w_sim     * token_similarity_score +
  w_seq     * sequence_strength_score
```

### Feature Definitions
- **Recency**: exponential decay of `last_seen` time.
- **Frequency**: log-scaled global freq, optionally per-context.
- **Context match**: exact match on cwd/host/user or partial match.
- **Token similarity**: Dice/Jaccard on normalized tokens.
- **Sequence strength**: combination of confidence + lift.
- **Phase boost**: categorical match for session phase derived from `phase_stats`.

Weights will be configurable via config file and tuned with offline replay.

### Phase Configuration (TOML)
Phase patterns are tokenized (term-limited) and support glob terms (`*`, `?`).
Flags are matched order-independently; args are matched as a prefix in order.
Patterns live in `config/phases.toml` or `~/.config/zage/phases.toml`.

---

## 8. Sequence Mining

- Mine bigrams/trigrams (extend to N if cheap).
- Compute:
  - **support**: count of sequence
  - **confidence**: P(next | prefix)
  - **lift**: confidence / P(next)
- Store in `sequence_stats` and re-rank suggestions.

---

## 9. CLI Changes

Replace previous model commands with:

- `zage index`:
  - Build/refresh stats tables and token cache.
- `zage suggest`:
  - Generate predictions for current context.
- `zage analyze-sequences` (optional manual refresh)

Import/record remain unchanged.

---

## 10. Testing & Evaluation

### Offline evaluation
- Replay history and compute:
  - top-1 / top-5 accuracy
  - keystroke savings
  - mean reciprocal rank

### Unit tests
- tokenizer (edge cases)
- candidate generation
- ranking scoring
- sequence mining correctness

---

## 11. Implementation Phases

### Phase 1: Core tokenization + indexing
- Implement tokenizer + normalization
- Add stats tables + indexer
- Build token cache

### Phase 2: Candidate generation + ranking
- Implement candidate sources
- Implement ranking features + scoring

### Phase 3: Sequence mining + integration
- Compute sequence stats
- Integrate sequence features into ranking

### Phase 4: Evaluation + tuning
- History replay harness
- Tune weights and thresholds

---

## 12. Risks & Mitigations

- **Tokenization errors**: add fuzz tests for shell syntax.
- **Performance regressions**: keep indices small, prefer precomputed stats.
- **Overfitting to history**: balance recency with global frequency.

---

## 13. Deliverables Checklist

- [ ] New schema + indexer
- [ ] Tokenizer + normalization
- [ ] Candidate generation
- [ ] Ranking model
- [ ] Sequence mining
- [ ] CLI updates (`index`, `suggest`)
- [ ] Offline evaluation harness
- [ ] Updated tests
