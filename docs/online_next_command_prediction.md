# Online Next-Command Prediction (Prior Art → Concrete Design)

This doc summarizes relevant prior art on “next shell/CLI command prediction” and distills it into a concrete, lightweight **online learning** design aligned with zage’s goals:

- **Sequence effects** (e.g., after `git add` suggest `git commit`) *but conditioned on workspace/cwd/exit*.
- **Soft semantic similarity / generalization** (flag reorder, different filenames, similar subcommands).
- **Higher-order context interactions** (workspace_root + cwd + exit_status + recent commands; host/user/time/session as low-weight signals).

The intent is to replace “heavy offline training” with an **always-on model** that learns immediately from `record` events and stays strongly context-bound.

---

## Terminology

- **Invocation**: a row in `shell_history` (a single executed command).
- **Command (unique)**: a distinct command line (represented in zage by `command_stats.command`, derived from `shell_history.expanded_command`).
- **Workspace root**: zage’s notion of a project/workspace root (often a git repo root, but not always).
- **Context**: structured fields + recent history window. Priority (high → low): `workspace_root > cwd > exit_status > host > user > time-of-day > session`.

---

## Prior Art (What We Should Steal)

### Korvemaker & Greiner (AAAI 2000): adaptive UNIX command prediction is hard

Paper: “Predicting UNIX Command Lines: Adjusting to User Patterns” (AAAI 2000). PDF mirror:

`https://papersdb.cs.ualberta.ca/~papersdb/uploaded_files/712/paper_korvemaker00predicting.pdf`

Key takeaways:

- A relatively simple adaptive model can predict next commands surprisingly often, but **it is difficult to squeeze out large additional gains** in an offline setting.
- The target distribution is **non-stationary** (goals shift; workflows start/stop).
- They explore “mixture of experts” ideas: different “experts” specialize on different slices/signals.

Distilled implication for zage:

- Prefer **online adaptation** and **context partitioning/backoff** over a single global model.
- Don’t treat “session boundaries” as hard gates; treat them as *features* and *backoff hints*.
- Favor “cheap signals that track non-stationarity” (online weights, decays, per-context statistics).

### Korvemaker (MSc thesis 2001): more depth on Unix command line prediction

Thesis landing page (includes DOI):

`https://era.library.ualberta.ca/items/25101b5c-a867-401c-adbf-21078e1c6f9a`

Key takeaways:

- Reinforces the core difficulty: offline “big model” improvements tend to be marginal because user behavior shifts.

Distilled implication for zage:

- Make “learning” incremental and cheap so we can update continuously and avoid long retrains.

### “Generalized command-lines” (1995): normalize syntax to generalize + correct errors

Paper: “Command-line prediction and error correction using generalized command-line” (1995, Elsevier/ScienceDirect):

`https://www.sciencedirect.com/science/article/pii/S0921264706801930`

Key takeaways:

- Predicting on raw command strings is brittle.
- A *generalized* command line (command name + argument symbols/templates) helps both prediction and error correction.

Distilled implication for zage:

- Model commands primarily by a **stable structural signature**:
  - `head:*`
  - **position-independent** `flag:*` set (sorted/dedup)
  - a bounded set of **normalized** `arg:*` (paths/ids/branches normalized)
  - optional “template”/slot tokens where appropriate

### Seq2Seq Unix command prediction (2020): deep models exist, but are heavy

Paper: “Seq2Seq and Joint Learning Based Unix Command Line Prediction System” (arXiv:2006.11558):

`https://arxiv.org/abs/2006.11558`

Key takeaways:

- Neural seq2seq can be trained for this problem, but training + inference costs are non-trivial.

Distilled implication for zage:

- We do **not** want a heavyweight encoder trained offline if we want “instant learning”.
- However, this supports the idea that “learned representations” can help beyond string matching.

### SimCURL (2022): command sequences are rich signal; session dropout is a useful trick

Paper: “SimCURL: Simple Contrastive User Representation Learning from Command Sequences” (arXiv:2207.14760):

`https://arxiv.org/abs/2207.14760`

Key takeaways:

- Even noisy “command sequences” encode intent and can support downstream tasks.
- They introduce **session dropout** as augmentation (drop some steps to avoid overfitting to exact sequences).

Distilled implication for zage:

- Use **windows** and **backoff** rather than rigid dependence on exact last-N.
- Consider “drop some recent commands” as a regularizer for online learning.

### SHREC (2024): workflows can be modeled as graphs (useful for SRE-like operations)

Paper: “SHREC: a SRE Behaviour Knowledge Graph Model for Shell Command Recommendations” (arXiv:2408.05592):

`https://arxiv.org/abs/2408.05592`

Key takeaways:

- Command histories can be structured into a graph of actions/entities/workflows, and the graph can drive real-time recommendations.

Distilled implication for zage:

- Keep the **sequence/transition layer** (bigrams/trigrams/backoff) as a strong candidate source.
- Don’t rely on graphs alone to solve semantic generalization (flags/args/etc.).

### CmdCaliper (EMNLP 2024): semantic command-line embeddings are a real thing

Paper: “CmdCaliper: A Semantic-Aware Command-Line Embedding Model and Dataset for Security Research” (arXiv:2411.01176; EMNLP 2024):

`https://arxiv.org/abs/2411.01176`

ACL anthology:

`https://aclanthology.org/2024.emnlp-main.1126/`

Key takeaways:

- Domain-specific command-line embeddings can outperform generic sentence embeddings on “similar command” tasks.
- These models are still fairly large and not ideal for “always-on online learning”.

Distilled implication for zage:

- We want CmdCaliper-like invariances but in a **tiny**, privacy-preserving, online-updatable model.

### ShIOEnv (2025): exit status / IO matter; constrain invalid commands (grammar masking)

Paper: “ShIOEnv: A CLI Behavior-Capturing Environment …” (arXiv:2505.18374):

`https://arxiv.org/abs/2505.18374`

Key takeaways:

- Datasets that omit exit status/output miss important behavior signals.
- Grammar constraints help prevent invalid command constructions.

Distilled implication for zage:

- Exit status should be an explicit feature/training signal.
- Keep **hard constraints** (prefix match, syntax validity, dedupe) outside the model.

---

## Additional Prior Art (New)

These are more implementation-adjacent / “what engineers actually ship” references that match our constraints (local, fast, explainable, privacy-sensitive):

- **NeuroShell / Intelligent Unix Shell (2023–2024, open source)**: hybrid approaches combining n-grams/Markov + similarity scoring for suggestions/typo correction, emphasizing local execution and explainable scoring.
  - `https://github.com/Anshika-111105/Intelligent-Unix-Shell`
- **ShellSuggest (2023–2024, open source)**: next-word prediction for Linux terminal commands (LM framing for shell completion).
  - `https://thedotproduct.github.io/shell-suggest-lang-model/`
- **First Shell Command Prediction Model (Jesse Claven)**: practical blog write-up of a transformer/time-series approach for shell command prediction.
  - `https://www.j-e-s-s-e.com/blog/first-shell-command-prediction-model`
- **LOTLDetector (2025, security detection)**: multi-view command-line embeddings (Token2Vec/Rule2Vec/Similarity2Vec) fused with CNN; evidence that multi-view representations work.
  - `https://link.springer.com/article/10.1186/s42400-025-00531-w`
- **Netflix Generative Recommenders (2024–2025, recsys practice)**: sampled-softmax efficiency, continuous training, calibration/metrics thinking (MRR).
  - `https://netflixtechblog.medium.com/towards-generalizable-and-efficient-large-scale-generative-recommenders-a7db648aa257`
- **DenseRec (2025, recsys)**: dual pathway embeddings (ID-based + content-based) with probabilistic mixing for cold-start.
  - `https://www.linkedin.com/posts/singhsidhukuldeep_solving-the-cold-start-problem-in-sequential-activity-7407955404971859968-FNZ6`
- **Two-tower retrieval overview**: good background on two-tower training objectives and ANN retrieval tradeoffs.
  - `https://www.emergentmind.com/topics/two-tower-retrieval`
- **PSAG (online continual learning)**: projection-based stabilization for online continual learning streams.
  - `https://openreview.net/forum?id=NvXpSvMrXS`
- **Continual learning overview (practical)**: stability/plasticity, replay sizing, embedded constraints.
  - `https://runtimerec.com/the-enduring-challenge-of-continual-learning-preventing-catastrophic-forgetting-in-long-lived-ai-systems/`
- **Lifelong deep learning survey (overview)**: additional background and pointers across the CL landscape.
  - `https://www.emergentmind.com/topics/lifelong-deep-learning-ldl`
- **Cold-start overview (auxiliary features / settings)**: cold-start settings and how side-info helps.
  - `https://www.sciencedirect.com/science/article/abs/pii/S0950705126000249`

Even when these aren’t “next shell command prediction” papers, the techniques directly map to our goals (next-item prediction + context + cold-start + online updates).

---

## Distilled Design: Lightweight Online Two-Tower Embedder + Context Backoff

### High-level

We want a model that:

1) Learns a **context vector** from (workspace_root, cwd, exit_status, optional host/user/time/session) + last N commands.
2) Learns a **command vector** from a normalized, generalized representation of the command line.
3) Scores candidates by **dot product** (or cosine) plus small bias terms.
4) Updates parameters **online** on every `record` (or micro-batches) using negative sampling.

This is intentionally “word2vec/item2vec-like” rather than a Transformer:

- Training per update is ~O((1+K)·D) where K is number of negatives and D is embedding dim.
- No expensive batch×batch logits matrix.
- No GPU required.

---

## Concrete v1 Decisions (So This Is Implementable)

This section pins down defaults so we don’t get stuck in “design feels plausible” territory.

### Targets

- **Inference latency (p99)**: < 15ms end-to-end for `suggest` over UDS on a typical dev machine, including DB I/O.
- **Update latency**: < 2ms average per `record` (micro-batch updates allowed).

### Dimensions / windows

- Embedding dim **D**: 64 (increase to 96/128 only if needed).
- Negative samples **K**: 16 (increase to 32 only if needed).
- Context window **N**: 10 commands (tunable).

### OOV handling (commit now)

We will implement **FastText-style subword hashing** in v1:

- Character n-grams: 3–6
- Bucket count: **131,072** (2^17), with sign hashing
- Hash function: stable 64-bit hash with good avalanche (reuse zage’s stable hash utility)

Rationale: shell token space is huge (paths/branches/tooling). We must not require “rebuild vocab”.

### Replay buffers (forgetting mitigation)

We will do replay in v1. Defaults:

- **Global replay**: reservoir sample of 20,000 events.
- **Per-workspace replay**: ring buffer of 5,000 events for up to 32 most-recently-active workspaces (LRU evict workspaces).
- Each update trains on: 1 new event + 2 replay events (1 global, 1 same-workspace if present).

Store events compactly (no raw strings):

- token IDs / hashed bucket IDs
- lightweight sampling keys (workspace hash, head hash, time bucket)

### Learned context importance (group scalars)

We will learn a small set of **group scalars** online (instead of hard-coding all context weights):

- Groups: `workspace_root`, `cwd`, `exit`, `host`, `user`, `timebucket`, `session`, `phase/workflow`, `recent_heads`, `recent_flags`.
- Init: 1.0; LR: 10× smaller than embeddings; L2 reg toward 1.0.

### Confidence gating (when to not suggest)

We will suppress embedding-driven suggestions when uncertain:

- Margin gate: if `top1 - top2 < 0.05`, down-weight learned score and rely more on tier-1/frecency.
- Absolute gate: if `top1 < 0.0` (post-calibration), don’t inject embedding-based candidates.

This ensures “model on” never performs worse than “model off”.

### Calibration / blending with priors

We will blend with priors using a log-linear model:

`final = α * model_score + β * log(frecency_prior) + γ * sequence_score + δ * tier1_score`

v1 calibration:

- Learn `α/β/γ/δ` online using **accept-only feedback** (exact suggestion accepted).
- No implicit negatives or time windows in v1; non-accepts are ignored (no-op).
- Use simple gradient descent on logistic loss; no complex bandits for v1.

### Privacy note (v1)

Embeddings capture associations, not raw text. They may still leak sensitive patterns statistically, but they should not be directly invertible back into exact command lines without additional side information.

### Representation (generalized command line)

For an invocation `(shellname, expanded_command)`:

- Tokenize using zage’s existing shell-aware tokenizer.
- Extract structured parts:
  - `head:<cmd>`
  - `flag:<flag>` (collect all flags, sort+dedup to make them **position-independent**)
  - `arg:<normalized>` (cap to M args; normalize paths/ids/branches, etc.)
- Optional structural tokens (if useful, but not required for v1):
  - `shape:pipe`, `shape:redir`, `shape:assign`, `shape:subshell`, …

These tokens are used to compute a command embedding.

### Context representation

Context tokens are:

- `ctx:workspace_root=<root>` (highest priority)
- `ctx:cwd=<cwd>`
- `ctx:exit=<exit>` (explicitly include failures)
- `ctx:host=<host>` (low weight)
- `ctx:user=<user>` (low weight)
- `ctx:timebucket=<bucket>` (very low weight)
- `ctx:session=<id>` (very low weight; *soft affinity*)
- Recent commands (last N invocations), expressed using the same generalized tokens but prefixed, e.g.:
  - `prev:head:*`, `prev:flag:*`, `prev:arg:*`

We can explicitly implement **backoff**:

1) If `workspace_root` exists: model uses it.
2) Else if `cwd` exists: use cwd.
3) Else: global.

This backoff is not a hard gate; it just controls which statistics/features dominate.

### Model

Two towers that share the same token vocabulary:

- `E[token] -> R^D` (embedding lookup)
- Context vector:
  - `v_ctx = normalize(sum(E[t] * w_ctx(t)))`
- Command vector:
  - `v_cmd = normalize(sum(E[t] * w_cmd(t)))`
- Score:
  - `score(ctx, cmd) = v_ctx · v_cmd + b_cmd + b_head + b_ctx_bucket`

Where per-token weights are **initial priors**, and group scalars are learned online (see “Concrete v1 Decisions”):

- `ctx:workspace_root` weight > `ctx:cwd` > `ctx:exit` > `ctx:host` > `ctx:user` > `ctx:timebucket` > `ctx:session`.
- `flag:*` tokens weight is non-zero and position-independent (flags must matter).

### Online training objective (negative sampling)

On each new `record` event we get an example:

- Input: context (last N commands + context fields)
- Positive: the command that was actually executed next

We update embeddings with a binary logistic objective using K negatives:

- Sample negatives from a mixed pool that intentionally includes “hard negatives”:
  - same `head:*`
  - same workspace/cwd (if available)
  - globally frequent commands
  - recent commands

Loss (conceptual):

- For positive command `c+` and negatives `c-`:
  - maximize `σ(score(ctx, c+))`
  - minimize `σ(score(ctx, c-))`

This is the classic “skip-gram with negative sampling” pattern, applied to next-command prediction.

### Where frecency fits

Frecency is still useful as:

- a **baseline prior** (what you do often/recently within the same workspace/cwd)
- a **cold-start** mechanism before the online model has enough signal

But frecency should be **context-conditioned** and should back off:

`workspace_root frecency -> cwd frecency -> global frecency`

The online model score should be the primary rank signal once warmed up.

---

## Critical Failure Modes (and Concrete Mitigations)

This section addresses the major risks called out in review so we don’t ship a design that “sounds plausible” but fails in practice.

### 1) Catastrophic forgetting / non-stationarity

Problem:

- Pure online SGD will overfit to the most recent workflow and forget older associations when the user switches projects.

Mitigations (pick at least one; ideally two):

- **Replay buffer**: keep a reservoir-sampled buffer of past `(context, next_command)` events and train each update on:
  - 1 new event + R replay events (e.g., R=1–8).
- **Per-context replay**: maintain separate buffers per `workspace_root` (and optionally per `cwd`), so switching projects immediately replays that project’s past.
- **Decayed learning rate / Adagrad**: reduces step sizes for frequently-updated parameters; helps stabilize common tokens/commands.
- **Weight averaging**: maintain EMA (“shadow weights”) of embeddings and use EMA for inference to damp oscillation.

What we are explicitly *not* doing (v1):

- Heavy continual-learning methods (EWC, LwF) unless replay + Adagrad still isn’t stable enough.

### 2) Cold-start and OOV tokens/commands

Problem:

- New tools/flags/args appear continuously. A naive vocabulary table yields OOV issues and brittle behavior.

Mitigations (commit to one):

- **FastText-style subword hashing**:
  - Represent each token using character n-grams (e.g., 3–6) hashed into fixed buckets.
  - Command embedding becomes the sum of its token n-gram embeddings.
  - This makes new commands partially “understandable” immediately.
- **Dual pathway (DenseRec-style)**:
  - Maintain both:
    - ID-based embedding for known `command_stats.command` (fast, high quality when seen)
    - content-based embedding from tokens/subwords (works for unseen commands)
  - During training/inference, mix them (learned or heuristic) based on “how much history exists for this command”.

We should not require a “rebuild vocab” step.

### 3) Negative sampling strategy (popularity bias + bias correction)

Problem:

- If we sample “globally frequent commands” as negatives too aggressively, we suppress common-but-correct suggestions.
- If we don’t correct for the sampling distribution, scores become miscalibrated.

Mitigations:

- Use a mixed negative distribution `Q` with explicit weights, e.g.:
  - 40% from same `workspace_root` (or cwd backoff)
  - 30% from same `head:*` (hard negatives)
  - 20% from recent window (recency hard negatives)
  - 10% global, frequency-weighted with exponent 0.75 (word2vec-style)
- Use a **sampled-softmax / log-Q correction**:
  - Train on `logits = score(ctx, cmd) - log(Q(cmd))` so training isn’t biased toward the sampler.

Related reading:

- “Correct and Weight” (implicit feedback; negative sampling bias): `https://arxiv.org/abs/2601.04291`
- “Correcting the LogQ Correction” (notes subtle bias in standard log-Q correction): `https://arxiv.org/abs/2507.09331`

### 4) Multi-step workflows and interleaving

Problem:

- Real workflows are not pure bigrams and are often interleaved.
- “Context → next command” can still learn these, but only if the context representation preserves enough structure.

Mitigations:

- Include last-N commands in context, but avoid collapsing everything into a single bag:
  - Add lightweight structure tokens: `prev0:head:*`, `prev1:head:*`, … for heads (or for a small subset like heads + top flags).
  - Keep flags position-independent *within a command*, not across the entire window.
- Add “phase/workflow” as explicit context tokens (derived from zage phases/sequences), not as a separate heuristic scoring system.

If this still isn’t sufficient, the next step is a tiny sequence encoder (e.g., GRU over the last N command embeddings), still trained online with negative sampling.

### 5) Fixed context weights vs learned importance

Problem:

- Hard-coded `w_ctx` is brittle; different users and repos need different context importance.

Mitigations:

- Learn a small set of **group scalars** online:
  - one scalar per context group (`workspace_root`, `cwd`, `exit`, `host`, …)
  - optionally per “phase/workflow” token group
- Or learn a small “gating” function (tiny linear layer) that weights groups based on the current context.

The key is: keep it tiny enough to update online and store in DB.

### 6) Exit status: feature, not label

Exit status should be treated as context, not “this command was wrong”.

- `exit!=0` often means “retry the same command with flags” or “run the fix step”.
- So `ctx:exit=1` should allow the model to learn patterns like:
  - `cargo test` (exit 101) → `cargo test -- --nocapture`
  - `git push` (exit) → `git pull --rebase`

### 7) Dimensionality, collisions, and optimizer stability

Recommendations:

- Start with D=64–128 for online.
- Use Adagrad or RMSProp (per-parameter adaption) to avoid constant retuning.
- If using hashing buckets:
  - collisions are expected; mitigate by:
    - larger bucket count
    - sign hashing
    - keeping “ID-based” embeddings for frequent commands (dual pathway)
- Use gradient clipping to avoid exploding updates on rare events.
- If we use Adagrad, consider periodic accumulator rescaling or switch to RMSProp to avoid “learning rate goes to ~0 forever” for very frequent features.

### 8) Ranking and calibration (combining with frecency)

Dot products are not probabilities; we need a stable, interpretable blend with priors.

Recommended:

- Use a log-linear blend:
  - `final = α * model_score + β * log(frecency_prior) + γ * sequence_score + …`
- Learn `α/β/γ` (or at least `α`) online using acceptance feedback, or set them conservatively and expose them as config.

### 9) Evaluation: non-stationary + ranking metrics

Replace “single chronological split” with:

- rolling-window evaluation (prequential): train on past, evaluate on next chunk, slide forward.
- ranking metrics:
  - **MRR@K**
  - Recall@K
  - Coverage@K (diversity / not always predicting `ls`)
  - Context leakage rate (define as: top-K contains commands whose dominant workspace/cwd != current)

### 10) Implicit negative feedback (“undo”, Ctrl-C, rejects)

We do **not** use implicit negatives in v1:

- Only **accepted suggestions** are logged and used for training.
- No “rejected/ignored” outcomes, and no time-window matching.
- Ctrl-C and other aborts are ignored (no-op).

If we want negatives later, it must be explicit and UX-supported; we will not infer it from timing.

### Multi-host sync conflict behavior

If multiple hosts update the model against the same remote DB:

- Avoid last-write-wins on a single embedding row.
- Prefer one of:
  - append-only updates + periodic compaction
  - per-host shards (host-scoped embedding deltas) + read-time merge (or scheduled merge)

---

## Integration Into zage (Concrete)

### Inputs we already have

From `shell_history` we already capture:

- `expanded_command`
- `shellname`
- `working_directory`
- `hostname`, `username`
- `exit_status`
- timestamps
- `session_id`

And zage already computes/derives:

- `workspace_root` / workspace info (stored as JSON in `shell_history.workspace_json` today)
- sequence statistics (bigrams/trigrams)
- context statistics (tier-1)

### Candidate generation stays

Keep a staged pipeline approach:

1) Collect candidates from:
   - transitions (sequence stats)
   - context stats (workspace/cwd/session)
   - phase/templates
   - optionally vector similarity (sqlite-vec) if we keep it as a candidate source
2) Apply hard constraints:
   - strict prefix / cursor constraints
   - syntax validity
   - dedupe
3) Rank with the online model score (plus small priors/backoff).

### One embedding space (avoid conflicts)

If we keep sqlite-vec similarity search, we should avoid “two embedding spaces” that disagree.

Rule:

- The embeddings indexed in sqlite-vec should be **the same embeddings** used by the online model for scoring (or a strict subset of the same representation).
- Candidate source identity (transition/context/phase/vector) can still be included as a feature for blending/calibration.

### Storage (what should live in DB)

Minimal online model state:

- Token embedding table (or fixed hashed buckets):
  - `token_id -> vec[D]`
  - optimizer accumulators if using Adagrad/RMSProp
- Optional bias tables:
  - command bias
  - head bias

If we want multi-host sharing via libsql/Turso, storing this model state in the DB makes it syncable.
If we want maximum write performance, buffer updates and flush periodically.

### Update frequency (online)

Update the model on:

- every `record` (single-step update), or
- micro-batches (flush every N records), or
- a background worker in the daemon

This should be fast enough on CPU if D is small (e.g., 64–128) and K is modest (8–32).

---

## Evaluation Plan (What “good” looks like)

### Offline (repeatable)

Given a fixed history, prefer rolling-window evaluation (prequential), because user behavior is non-stationary:

- Evaluate on successive windows (train on [0..t], evaluate on (t..t+Δ], slide forward).
- Report ranking metrics:
  - top-1 / top-5 next-command accuracy
  - **MRR@K**
  - Recall@K
  - Coverage@K
  - Context leakage rate
  - the same metrics *within workspace_root*, to ensure context-bound behavior is real

### Online (what matters)

Track:

- acceptance rate (did the user execute a suggested command soon after?)
- time-to-next-command reduction (optional)
- “context leakage” events (suggestions from unrelated workspace/cwd)

---

## Why this satisfies the goals

- **Sequence effects**: context vector includes last-N; negative sampling can include “same head” to make sequences discriminative.
- **Soft semantic similarity**: generalized tokens + learned token embeddings make reorder/different filenames map close.
- **Higher-order interactions**: context tokens (workspace_root/cwd/exit) are first-class; the model can learn their interactions because they participate in the same embedding space.

The key property is that the model updates immediately from new data and continuously adapts to your workflows.
