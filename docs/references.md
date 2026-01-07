# References

## Public datasets

- **NL2SH-ALFA** (MIT license) — natural-language to shell command pairs (bash).
  - Source: https://huggingface.co/datasets/westenfelder/NL2SH-ALFA
  - Local copy: tests/corpus/nl2sh-alfa (train.csv.xz, test.csv, README.md)
  - Intended use here: harden shell parsing/tokenization against real-world command constructs.
  - Non-goals: sequence modeling or personalized ranking (dataset has no temporal/session order).
