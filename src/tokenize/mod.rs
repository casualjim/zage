use serde::{Deserialize, Serialize};

mod command_parts;
mod lexer;
mod normalize;
mod tree_sitter;

pub use command_parts::{CommandParts, extract_command_parts};
pub use lexer::{token_strings, token_strings_index, tokenize, tokenize_index};
pub use normalize::{normalize_token, normalized_tokens};

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
