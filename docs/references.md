# References

## Public datasets

- **NL2SH-ALFA** (MIT license) — natural-language to shell command pairs (bash).
  - Source: https://huggingface.co/datasets/westenfelder/NL2SH-ALFA
  - Local copy: tests/corpus/nl2sh-alfa (train.csv.xz, test.csv, README.md)
  - Intended use here: harden shell parsing/tokenization against real-world command constructs.
  - Non-goals: sequence modeling or personalized ranking (dataset has no temporal/session order).

## Pretrain workflow

- `mise pretrain` scans `data/pretrain/derived` for `.bash_history`, `.zsh_history`, or `.history` files.
  Convert the datasets below into one of those formats to include them in global pretraining.
  The built-in fetch pipeline writes to `data/pretrain/derived` and uses:
  - Masaryk Hands-on Cybersecurity Training (Zenodo data.zip)
  (History datasets with session/timestamp context only.)

## Shell history datasets (from "Public Shell Command History Datasets for Predictive Terminal Training")

- **Masaryk University Hands-on Cybersecurity Training** — sanitized, structured JSON with rich context
  (timestamps, cwd, host/session id, full commands). Good for contextual prediction.
  - UCI dataset entry: https://archive.ics.uci.edu/dataset/869/shell+commands+used+by+participants+of+hands-on+cybersecurity-training
  - Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8479389/
  - DOI (Zenodo): 10.5281/zenodo.8136017

- **GitHub-scraped shell histories (Aldo Cortesi, 2013)** — raw `.bash_history`/`.zsh_history` from public repos.
  Unsanitized; good for real-world distribution but requires careful filtering.
  - Summary: https://corte.si/posts/hacks/github-shhistory/

- **Greenberg Unix Command Traces (168 users, 1988)** — classic multi-user dataset with session structure.
  Real commands with arguments; no per-command cwd/exit.
  - Info: http://saul.cpsc.ucalgary.ca/pmwiki.php/HCIResources/HCIWWWUnixDataSets

- **SEA Masquerade Dataset (Schonlau et al., 2001)** — 50 users, 15k commands each, command names only
  (truncated). Useful for sequence-only modeling or anomaly detection. Not used for pretrain.
  - Report: https://www.niss.org/sites/default/files/technicalreports/tr95.pdf

- **Purdue Unix Command Dataset (Lane & Brodley, late 1990s)** — small user set, sanitized streams.
  Referenced in masquerade detection literature (may require contacting authors).

- **RACOON synthetic command generator (ACSAC 2019)** — synthetic, template-based logs for augmentation.

- **Asciinema public session recordings** — real command sequences with timing metadata; requires scraping/curation.
  - API / site: https://asciinema.org/

- **Atuin / Hishtory-style logs (potential data dumps)** — rich contextual logging (cwd, exit code, duration).
  Not a public dataset today, but a format target for opt-in aggregation.

## NL2SH datasets and benchmarks (from "Validated Datasets and Data Corpora for Context-Aware Shell Command Suggestion Systems")

- **NL2Bash (original, 2018)** — large NL→Bash one-liners corpus from QA/tutorial sources; broad utility coverage but
  known to contain errors and outdated syntax in the raw version.

- **Verified NL2Bash / NL2Bash‑EABench (2025)** — cleaned + expanded dataset with execution-based validation
  (Docker functional equivalence); current gold standard for supervised NL→Bash training.

- **InterCode / execution-based evaluation** — emphasizes functional correctness rather than string match; useful
  benchmark framing for “does the command work?” validation.

- **NLC2CMD (NeurIPS 2020)** — competition datasets derived from NL2Bash + Tellina query logs; designed to test
  generalization to unseen utilities and noisy user queries.

- **Magnum NLC2CMD “bash_gen”** — synthetically generated NL→Bash pairs using man‑page grammars and LLM
  back-translation; improves coverage of rare flags and complex combinations.

- **nl2bash‑custom** — aggregated NL→Bash corpus (community‑sourced; large volume, noisy).

- **CLI Commands Explained (commandlinefu)** — community-voted commands; votes can act as a “quality/idiomatic”
  signal for ranking.

## Context corpora and system signals (from the same report)

- **UCI UNIX User Data** — anonymized multi‑user command streams; useful for user‑specific transition modeling.
  Command‑only; not used for pretrain.

- **Asciinema session recordings** — real command sequences with timing + output; enables state-aware suggestion
  when paired with terminal output parsing.

- **Dotfiles corpora** — alias/function definitions for personalization (e.g., `g=git`); useful for alias expansion.

- **Loghub** — large system log corpora; can align command execution with system state changes for richer context.

- **IBM CLAI (Command Line AI)** — reference architecture + “Fix‑it” datasets mapping errors to corrective commands;
  highlights value of error-aware suggestions.
