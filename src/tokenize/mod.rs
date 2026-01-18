use serde::{Deserialize, Serialize};

mod command_parts;
mod lexer;
mod normalize;
mod tree_sitter;

pub use command_parts::{CommandParts, extract_command_parts};
pub use command_parts::{CommandStatsParts, extract_command_stats_parts};
pub use lexer::{token_strings, token_strings_index, tokenize, tokenize_index};
pub use normalize::{normalize_command_whitespace, normalize_token, normalized_tokens};

pub fn generalized_command_tokens(shellname: &str, command: &str, max_args: usize) -> Vec<String> {
  let tokens = tokenize_index(shellname, command);
  let Some(parts) = extract_command_parts(command, &tokens) else {
    return Vec::new();
  };

  let mut out = Vec::new();
  if !parts.head.is_empty() {
    out.push(format!("head:{}", parts.head));
  }

  // position-independent flags
  let mut flags = parts.flags;
  flags.sort();
  flags.dedup();
  for flag in flags {
    out.push(format!("flag:{flag}"));
  }

  // bounded normalized args
  for arg in parts.args.into_iter().take(max_args) {
    if !arg.normalized.is_empty() {
      out.push(format!("arg:{}", arg.normalized));
    }
  }

  out
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenKind {
  Word,
  Operator,
  Redirect,
  Quoted,
  Assignment,
  Variable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token {
  pub raw: String,
  pub kind: TokenKind,
  pub normalized: String,
}
