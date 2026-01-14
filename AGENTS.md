# Agent Coding Guidelines

> **Important:** Prefer the `mise` tasks for installs, builds, tests, and formatting. Only use raw toolchain commands when no `mise` wrapper exists, and call that out explicitly.
>
> **CRITICAL: Prefer the Rust LSP for Rust code navigation.** The Rust LSP is the primary tool for Rust files because it is accurate, fast, and type-aware. That said, other tools (rg/find/manual browsing) are still acceptable when they are faster for the task, the LSP is unavailable, or you're working outside Rust code. See the "Code Navigation (Use Rust LSP!)" section for detailed commands.
>
> **CRITICAL: Do NOT run git mutations without explicit approval from the user** Do NOT ever run git checkout/revert/restore/reset without EXPLICIT APPROVAL from the USER
>
> **CRITICAL: DO NOT ASK KNOWABLE QUESTIONS** Do not ask the user for information that you can look up. 


## Build, Test, and Development Commands
Always default to the `mise` tasks below; only run direct toolchain commands if no `mise` wrapper exists and note the deviation.

**For Rust code navigation and understanding, use the Rust LSP first.** For non-Rust code or quick searches, it is fine to use rg/find/manual browsing when it is more appropriate.

- `mise install`: Install pinned Rust, Bun, Wrangler, etc.
- `mise build:debug`: Build Rust
- `mise test`: All tests (Rust nextest + Workers via bun test).

## Code Navigation (Use Rust LSP!)

**IMPORTANT: Prefer the Rust LSP for Rust code navigation.** The Rust LSP should be your primary tool for:
- Finding symbols and definitions
- Navigating to references
- Getting function signatures and documentation
- Understanding code structure
- Finding implementations and usages

**Prefer Rust LSP over:** grep/find/rg, manual file browsing, or any other navigation method **for Rust files**. Use other tools when they are faster for the task, the LSP is unavailable, or the code is not Rust.

### Rust LSP Commands Available

Use these `mcp__rust-lsp__*` tools for navigation:

```bash
# Get file structure and symbols
mcp__rust-lsp__outline <file_path>

# Search for symbols across the codebase
mcp__rust-lsp__search <query>

# Find all references to a symbol
mcp__rust-lsp__references <file_path> <line> <character>

# Get detailed info about a symbol at cursor position
mcp__rust-lsp__inspect <file_path> <line> <character>

# Get code completions at a position
mcp__rust-lsp__completion <file_path> <line> <character>

# Rename a symbol across the codebase
mcp__rust-lsp__rename <file_path> <line> <character> <new_name>

# Get diagnostics (errors/warnings) for a file
mcp__rust-lsp__diagnostics <file_path>
```

### Navigation Examples

```bash
# Find all search-related services
mcp__rust-lsp__search "SearchService"

# Explore the main application structure
mcp__rust-lsp__outline "crates/slipstreamd/src/lib.rs"

# Find all references to AppState
mcp__rust-lsp__references "crates/slipstreamd/src/app.rs" 16 1

# Inspect a function to get its documentation
mcp__rust-lsp__inspect "crates/embedding/src/lib.rs" 127 1

# Get completions for method calls
mcp__rust-lsp__completion "crates/slipstreamd/src/routes.rs" 42 20
```

### Why Use Rust LSP?

- **Accurate**: Understands Rust's type system and module resolution
- **Fast**: Instant navigation without scanning files
- **Context-aware**: Knows about imports, traits, generics
- **Complete**: Shows parameters, return types, documentation
- **IDE-quality**: Same experience as modern IDEs

**Remember: For Rust code, reach for the Rust LSP first; for everything else, use the best tool for the job.**


> REMINDER:
> ALWAYS get approval from the user for git checkout/reset/restore/revert/...
> NEVER run destructive git commands without explicit approval


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
- Write clear, descriptive commit messages in plain English
- Do NOT use conventional commits, semantic commits, or any commit prefixes (no "feat:", "fix:", "refactor:", etc.)
- Focus on WHAT changed and WHY, not the type of change
- First line should be a clear summary (50-72 chars recommended)
- Use the body for detailed explanation if needed
- Reference issue IDs when relevant (e.g., "Closes: slipstream-24")

Good examples:
- "Split search into dedicated Searcher service"
- "Add reranking provider for DeepInfra Qwen3-Reranker"
- "Fix flaky test by increasing tolerance for timing variance"

Bad examples:
- "refactor(embedding): Split search into dedicated Searcher service"
- "feat: add reranking provider"
- "fix: flaky test"

## SUPER IMPORTANT
- Do NOT run git commands that can result in loss of work unilaterally. ALWAYS get approval from the user for git checkout/reset/restore/revert/...
