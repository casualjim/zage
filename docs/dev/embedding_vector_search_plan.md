# Embedding + Vector Search Plan (Canonical)

This document is the **current plan** for the embedding/vector-search track in `zage`.
It is written to avoid losing context across long threads.

## Primary Goals

- Strongly improve **context-bound relevance** (workspace/cwd) without endlessly tuning heuristic filters.
- Make vector similarity a **first-class retrieval/ranking signal** backed by libsql + sqlite-vec.
- Keep everything **local-first and privacy-preserving**, while supporting **remote libsql/Turso** for multi-machine use.

## Definitions

- **Invocation**: one recorded command execution (`shell_history` row).
- **Command**: the canonical string we suggest (usually `expanded_command`).
- **Context window**: the last `N` commands (already exposed via `recent_limit`).
- **Scope**:
  - **workspace_root**: the root of a “workspace” (may be a git repo, monorepo root, or other workspace root).
  - **cwd**: the working directory of the current invocation/suggest request.
  - Note: earlier code used `repo_root`; treat this conceptually as **workspace_root** going forward.

## What We Are Building

### 1) Storage: embeddings in the DB (sqlite-vec)

- Store vectors in the database so we can do KNN queries via sqlite-vec.
- Store embeddings at the **unique command** level (not per history row) to avoid duplicating data.

Concrete choices:

- Table: `command_stats`
  - Add: `embedding F32_BLOB(dim)` (sqlite-vec)
  - Add: `embedding_updated_at INTEGER`
- Index: `command_stats_embedding_idx` on `command_stats(libsql_vector_idx(embedding))`
- Meta: `meta(key,value)` stores `command_embedding_dim`

### 2) Indexing: produce embeddings for commands

- `zage index --with-embeddings`:
  - Finds commands missing embeddings (`WHERE embedding IS NULL`).
  - Calls an embedder (configured via `[embedding]`) to compute vectors.
  - Writes vectors to `command_stats.embedding`.

Notes:
- The current implementation uses the `seasoning` embedder client as the embedding provider.
- This is **infrastructure** and can be swapped for a local model later.

### 3) Retrieval: vector search is scope-first

Vector similarity must be **context-bound**:

- Prefer retrieving within `workspace_root` (when available).
- Otherwise fall back to `cwd`.
- Only then fall back to global history.

This is the main mechanism for preventing “out-of-workspace” suggestions.

### 4) Ranking: embeddings become a core signal (not “extra candidates” forever)

Two-phase approach:

1. **Immediate**: vector search contributes candidates (recall) and is integrated into the existing pipeline.
2. **Next**: vector similarity becomes a primary scoring feature in ranking (and/or the ranker), reducing the need for layered heuristic filtering.

## Neural Net Plan (Sequence Model)

We previously agreed the “real model” goal is:

- **Primary:** full-line next-command prediction (sequence model).
- **Secondary:** edit-time next token/arg/flag prediction.

### Tokenization / normalization requirements

- Parse commands into `head`, `flags`, and `args` (shell-aware).
- **Flags must be position-independent**: represent flag *presence* as a set/multiset feature so
  `cmd --foo -b x` and `cmd -b x --foo` are treated equivalently for flag features.
- Args may remain order-dependent (many tools treat positional args as semantic).

The intended neural model shape:

- Sequence-contrastive **bi-encoder**:
  - Input A: context window + context fields
  - Input B: candidate command
  - Output: embeddings; score via dot-product/cosine
  - Loss: InfoNCE (in-batch negatives + hard negatives)

Key alignment decisions:

- Session affinity is a **soft feature**, not a gate.
- Context fields should reuse **existing tier-1 context dimensions** (don’t invent new ones).
- Avoid inventing an LRU cache for embeddings: the DB already supports similarity queries.

Integration points:

- Store learned embeddings in DB (sqlite-vec).
- Use sqlite-vec KNN queries for retrieval and/or ranking.
- Keep GBRT as a fallback path during A/B, but don’t let it become the primary driver long-term if neural works.

## Operational Rules

- **One way to do things**: server and embedded should behave the same for relevant flags and behavior.
- **No backwards compatibility**: schema/model changes are “break and rebuild”, not multiple paths.
- Prefer correctness and relevance over micro-optimizing away GPU usage.

## Current State (as of this document)

- Embedding vectors are stored on `command_stats.embedding` and queried via sqlite-vec.
- `zage index --with-embeddings` populates missing embeddings.
- Mean context embeddings are computed from the last `N` commands.
- There are known behavior/perf issues tracked by failing tests (in-code), and the plan is to make those tests pass and then iterate on scope-first vector retrieval.
