# Hybrid Reranker Design (Draft)

## Status
Draft for review and iteration.

## Summary
Strengthen Tier 1 candidate recall first (templates, session awareness, alias expansion, repo priors). Add an **optional lightweight reranker** (GBDT/linear)
only after recall is high. The ML layer is strictly additive and never replaces the statistical pipeline.

## Goals
- Preserve current speed and reliability of suggestions.
- Improve **recall** (candidate generation) before improving ranking.
- Improve ranking quality (esp. next-command suggestions).
- Keep inference fast on CPU (<20ms; target <5ms at tiny sizes).
- Use local data only; no external services.

## Non-goals
- Replacing the SQLite retrieval layer.
- Training on any shared or public user history by default.
- Building a large embedding model.

## Architecture Overview
```
shell_history -> indexer.rs -> SQLite stats (Tier 1 retrieval)
                                   |
                                   v
                           candidate set (~50)
                                   |
                                   v
                     optional reranker (Tier 2)
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
- Blends with Tier 1 score using **learned calibration**, not a fixed alpha.

### Candidate Features
Base features (cheap, always available):
- Tier 1 score (normalized).
- Recency (log time since last seen).
- Frequency (log count).
- Last exit status (from last command).
- Session id match.
- Repo root match / cwd match.
- Time-of-day bucket (optional).

Contextual features (v1):
- Last K command tokens (token IDs); default K=12 tokens.
- Last K commands (command-level tokens); default K=6 commands.
- Recent failure rate (last N commands).
- Session phase tag (build/test/deploy/edit).
- Git branch token (cached).
- Time-of-day bucket (lower weight).

Similarity features:
- Token overlap or normalized token similarity between **session window** and candidate.
- Optional lightweight embedding similarity (deferred).

## Model Options
### Option A: GBDT / Linear Reranker (preferred)
- Fast to ship, easy to debug.
- Handles mixed numeric/categorical features well.

### Option B: Tiny Transformer (future)
- Only if GBDT plateaus and recall is high.

## Training Data
Source: local `shell_history` table (user’s own data) plus optional public corpora for pretraining.
- Build sequences of last N commands.
- Training task: predict next token or classify candidate set.
- Only local data by default; optional public corpora are opt-in.

### Data Leakage & Evaluation Split
- Use a **time-based split** for local evaluation (e.g., train on the oldest 80–90%, validate on the newest 10–20%).
- For replay evaluation, predict each command using only history up to *t‑1* (no peeking at future commands).
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
- Minimum local history before training: **~1k–2k commands** for linear/GBDT; **5k–10k** for tiny transformer.
- Retrain cadence: when new history exceeds a threshold (e.g., +2k commands) or on a weekly schedule.
- Training should be **background/idle** and cancellable; never block suggestions.

## Inference
1. Get candidate set from Tier 1.
2. Encode last N commands (K=6 commands, 12 tokens).
3. Score candidates with model.
4. Blend scores using **stacked calibration**:
   - Calibrate Tier 1 and model scores to probabilities (isotonic regression or Platt scaling).
   - Combine via learned logistic stacker or weighted geometric mean.
   - Fall back to Tier 1 only when model confidence is below threshold (e.g., low calibrated top‑1 probability or small top‑1 vs top‑2 margin).
5. Output ranked suggestions (one for autosuggest, top k for list).

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
- Retrieval: <5ms.
- Rerank: <20ms (goal <5ms).
- Total suggest path: <30ms.

### Latency & Tokenization Budget
- Avoid heavy tokenization in the hot path.
- Reuse **token_cache** for history/candidates; only tokenize current line + last N commands at inference.
- For candidates, prefer **precomputed token features** from the indexer.

### Vocabulary Strategy
- Keep a **fixed vocab size** (e.g., 5k) to allow incremental updates without resizing embeddings.
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

### Non-goal Alignment
- This reranker remains optional and should not replace the Tier 1 statistical pipeline.

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
