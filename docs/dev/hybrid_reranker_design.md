# Hybrid Reranker Design (Draft)

## Status
Draft for review and iteration.

## Summary
Keep the system **simple and local**: strengthen Tier 1 candidate recall (templates, session awareness, alias expansion, repo priors) and add an **optional
lightweight reranker** (GBDT/linear). No embedding-first retrieval. The statistical pipeline stays the backbone; ML only reorders candidates.

## Goals
- Maximize **context relevance** (repo/cwd/branch/session-aware).
- Improve **recall** via better candidate generation (templates + session awareness).
- Improve ranking quality without heavy ML infrastructure.
- Use local data only; no external services.

## Non-goals
- Replacing the SQLite statistical pipeline.
- Training on shared or public user history by default.
- Large embedding models or vector search infrastructure.

## Architecture Overview
```
shell_history -> indexer.rs -> SQLite stats (Tier 1 retrieval)
                                   |
                                   v
                         candidate set (~50–200)
                                   |
                                   v
                     lightweight reranker (Tier 2)
                                   |
                                   v
                          final ordered suggestions
```

## Tier 1: Retrieval (Existing)
- Uses frequency/recency/context/sequence stats in SQLite.
- Produces top N candidates (default N=50–200).
- Guarantee: always returns recently seen commands.

## Tier 1+: Recall Upgrades (New)
Improve recall before ML reranking:
- **Template-based expansion** (command + flags + argument slots).
- **Session-aware candidate generation** (workflow phase / recent task cluster).
- **Alias expansion** (from dotfiles and runtime aliases).
- **Repo priors** (per-repo top commands and sequences).
- **Fallback retrieval** when confidence low (recency + frequency + session templates).

## Tier 2: Optional Reranking (Lightweight)
- Takes Tier 1 candidates + context features.
- Outputs a score per candidate.
- Uses **GBDT or linear model** (fast, debuggable).
- Blends with Tier 1 score using **stacked calibration**, not a fixed alpha.

### Candidate Features
Base features (cheap, always available):
- Tier 1 score (normalized).
- Recency (log time since last seen).
- Frequency (log count).
- Last exit status (from last command).
- Session id match.
- Repo root match / cwd match.

Contextual features (v1):
- Last K command tokens (token IDs); default K=12 tokens.
- Last K commands (command-level tokens); default K=6 commands.
- Recent failure rate (last N commands).
- Session phase tag (build/test/deploy/edit).
- Git branch token (cached).
- Time-of-day bucket (lower weight).

Similarity features:
- Token overlap or normalized token similarity between **session window** and candidate.

## Session Phase Detection
Detect workflow phase from recent command patterns (last 3–5 commands):
- **build**: `cargo build`, `make`, `npm run build`, `go build`
- **test**: `cargo test`, `pytest`, `npm test`, `go test`
- **deploy**: `git push`, `kubectl apply`, `docker push`
- **edit**: editor commands like `vim`, `code`, `nano`
- **default**: no clear pattern\n
Phase is inferred from simple keyword rules and used as a categorical feature.

## Model Options
### Option A: GBDT / Linear Reranker (preferred)
- Fast to ship, easy to debug.
- Handles mixed numeric/categorical features well.

### Option B: Tiny Transformer (future)
- Only if GBDT plateaus and recall is high.

### GBDT Implementation
- Prefer **lightgbm-rs** (LightGBM bindings) for speed/quality.
- Fallback: **linfa** (pure Rust) if avoiding C dependencies is critical.
- Model size target: <1MB for fast loading.

## Training Data
Source: local `shell_history` table (user’s own data) plus optional public corpora for pretraining.
- Build sequences of last N commands.
- Training task: **pairwise ranking** (context, next command) vs negatives.
- Only local data by default; optional public corpora are opt-in.

### Negative Sampling (Pairwise Ranking)
- **Easy negatives**: random commands from history.
- **Hard negatives**: commands from the same session but not the next step.
- **Hardest negatives**: commands with similar prefixes or same head (e.g., `git status` vs `git diff`).

### Data Leakage & Evaluation Split
- Use a **time-based split** for local evaluation (e.g., train on the oldest 80–90%, validate on the newest 10–20%).
- For replay evaluation, predict each command using only history up to *t-1* (no peeking at future commands).
- Keep a small “recent holdout” window (e.g., last 1–2 days or last 1k commands) to detect overfitting to recency.

### Recommended Corpora (from recent research)
- **Verified NL2Bash / NL2Bash‑EABench (2025)** for supervised NL→Bash pairs (clean, execution‑validated).
- **Magnum bash_gen** to cover rare flags and syntactic combinations.
- **Masaryk Hands‑on Cybersecurity Training dataset** for session modeling with rich context (cwd, timestamps).
- Avoid raw NL2Bash without filtering (known error rate).

### Sequence Construction
- Tokenize commands (reuse SLP tokenization).
- Build windows of size N (e.g., 10 commands).
- Optionally include exit status, time bucket, repo root id as extra features.

### Training Triggers & Minimum Data
- Minimum local history before training: **~1k–2k commands**.
- Retrain cadence: when new history exceeds a threshold (e.g., +2k commands) or on a weekly schedule.
- Training should be **background/idle** and cancellable; never block suggestions.

### Incremental Updates
- GBDT supports warm-start from previous model when available.
- Add new training examples incrementally without full retrain.
- Full retrain only on major history changes (e.g., new repo focus or >10k new commands).

## Inference
1. Build a context window (K=6 commands, 12 tokens).
2. Generate candidates from Tier 1 + Tier 1+ expansions.
3. Apply hard constraints (prefix, slots) for completion mode.
4. Score candidates with reranker.
5. Blend scores using **stacked calibration**:
   - Calibrate Tier 1 and model scores to probabilities (isotonic regression or Platt scaling).
   - Combine via learned logistic stacker or weighted geometric mean.
   - Fall back to Tier 1 only when model confidence is below threshold (e.g., low calibrated top‑1 probability or small top‑1 vs top‑2 margin).
6. Output ranked suggestions (one for autosuggest, top k for list).

## Graceful Degradation
- If model unavailable: use Tier 1 scores only (silent fallback).
- If reranker exceeds 50ms: short-circuit to Tier 1.
- If history < 1k commands: skip training, rely on Tier 1.
- Log degradation events for debugging (optional, off by default).

## Storage
- Model file: `~/.local/share/zage/model/model.bin` (global)
- Vocab file: `~/.local/share/zage/model/vocab.json` (global)
- Training metadata: `~/.local/share/zage/model/metadata.json` (global)

## CLI & Config
Add new commands:
- `zage train --backend wgpu --epochs N`
- `zage model status`
- `zage model reset`

Config/env:
- Reranking is **always on** once available.
- `ZAGE_MODEL_PATH` to override default model path.
- `ZAGE_RERANK_ALPHA` to override blending for debugging (optional).

## Performance Targets
- No strict latency targets in v1; prioritize relevance.
- Track latency to avoid regressions.

### Latency & Tokenization Budget
- Avoid heavy tokenization in the hot path where possible.
- Reuse **token_cache** for history/candidates; only tokenize current line + last N commands at inference.
- For candidates, prefer **precomputed token features** from the indexer.

### Vocabulary Strategy
- Keep a **fixed vocab size** (e.g., 5k) for optional bag‑of‑tokens features.
- Use **normalization tokens** (PATH/NUM/HASH/IP) to reduce the long tail.
- Use **hash buckets** for OOV tokens to capture user-specific paths without dynamic vocab growth.

## Evaluation & Validation
- Prefer **execution-based validation** over string match (functional equivalence in sandbox).
- Track offline metrics: top‑k accuracy, MRR, and failure‑rate on verified sets.
- Safety filter for destructive commands when running any automated evaluation.

### Execution-Based Validation Feasibility
- **Never** run user‑generated commands for validation.
- Restrict execution-based checks to vetted public datasets inside Docker/Podman sandboxes.

### Context Feature Latency
- Git branch/name should be **cached** (e.g., collected in shell hooks or repo watcher), not computed synchronously.
- Feature collection must not add latency to prompt rendering.

## Privacy
- Training uses only local shell history.
- No telemetry or uploads.
- Model artifacts stored locally.

## Rollout Plan (Proposed)
1. Add design doc + scaffolding (no model).
2. Add Tier 1+ recall upgrades (templates, session awareness, repo priors).
3. Add GBDT/linear reranker (optional).
4. Add tiny transformer only if needed.

## Future Context Features (non‑blocking)
- Error‑aware suggestions using “Fix‑it” style datasets (e.g., CLAI).
- Terminal output conditioning (asciinema‑style session data).
- Dotfiles/alias corpora for personalized expansion.

## Candidate Recall Limits
- Tier 2 cannot recover a suggestion not present in Tier 1.
- Ensure Tier 1 maintains **high recall** (larger candidate pool, multiple sources, fallback to recency).
- Expand candidate pool dynamically when confidence is low (e.g., low top‑1 margin).

## Open Questions
- Is K=6 commands / 12 tokens optimal, or should it be tunable?

## Next Steps
- Review with team.
- Decide on opt-in behavior and feature set.
- Implement scaffolding and config plumbing.
