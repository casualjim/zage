# Agent Coding Guidelines

> **Important:** Prefer the `mise` tasks for installs, builds, tests, and formatting. Only use raw toolchain commands when no `mise` wrapper exists, and call that out explicitly.
>
> **CRITICAL: DO NOT ASK KNOWABLE QUESTIONS** Do not ask the user for information that you can look up.


## Build, Test, and Development Commands
Always default to the `mise` tasks below; only run direct toolchain commands if no `mise` wrapper exists and note the deviation.

- `mise install`: Install pinned Rust, Bun, Wrangler, etc.
- `mise build:debug`: Build Rust
- `mise test`: All tests (Rust nextest + Workers via bun test).

## Code Style & Formatting

- Refactor, don't keep adding to the technical debt
- Clean up what is no longer needed
- KISS: keep it stupid simple
- YAGNI: you aren't going to need it, only implement what was asked do not make up requirements
- Backwards compatibility is not a goal, modify and break the existing code
- do not create a new file for every new task, work with what already exist.
- do not create limits or fallbacks for specialized libraries.
- Create the simplest possible thing that could possibly work
- Use Uuid::now_v7 not v4

- Rust:
  - Use `eyre::Result` for error handling, `thiserror` for domain errors
  - No `unwrap()` or `expect()` in public APIs
  - Async streaming first - avoid `collect()` patterns
  - Prefer streaming API's over batching API's. So return streams not vecs
  - Imports: Group std/core, external crates, and internal modules separately
  - Avoid allocations when you can, proactive cloning is not a good look
  - Formatting: run `mise format`; never invoke `cargo fmt` directly
  - Strict error handling - fail spectacularly, don't swallow errors
- TypeScript:
  - Strict mode with no `any` or `unknown`
  - Bun package manager
  - Double quotes for strings
- General:
  - 2-space indentation (except Python which uses 4)
  - LF line endings with final newline
  - Trim trailing whitespace
  - UTF-8 encoding

## Naming Conventions
- Rust: snake_case for variables/functions, PascalCase for types
- TypeScript: camelCase for variables/functions, PascalCase for types
- Files: snake_case for Rust, camelCase for TypeScript

## Error Handling
- Rust: Use `eyre::Result` for function returns, `thiserror` for domain-specific errors
- TypeScript: Proper error catching and handling without swallowing
- Never ignore errors - propagate or handle explicitly

## Commit Messages
- **MUST use conventional commits** - automated releases depend on them
- Format: `<type>: <description>` where type is one of: feat, fix, refactor, chore, docs, test, perf, style
- Include `bump:major`, `bump:minor`, or `bump:patch` in PR title/body for version bumps
- First line should be a clear summary (50-72 chars recommended)
- Use the body for detailed explanation if needed
- Reference issue IDs when relevant (e.g., "Closes: #123")

Good examples:
- "feat: Add reranking provider for DeepInfra Qwen3-Reranker"
- "fix: Stabilize online prediction scoring and tests"
- "refactor: Split search into dedicated Searcher service"

Bad examples:
- "Split search into dedicated Searcher service" (missing type)
- "Add reranking provider" (missing type prefix)
- "flaky test fix" (missing type prefix)

## SUPER IMPORTANT
- Do NOT run git commands that can result in loss of work unilaterally. ALWAYS get approval from the user for git checkout/reset/restore/revert/...
