use std::cell::RefCell;

use tracing::warn;
use tree_sitter::{Node, Parser};

use super::Token;
use super::TokenKind;
use super::command_parts::merge_special_tokens;
use super::normalize::{classify_word, normalize};

thread_local! {
  static ZSH_PARSER: RefCell<Option<Parser>> = {
    let mut parser = Parser::new();
    match parser.set_language(&tree_sitter_zsh::LANGUAGE.into()) {
      Ok(()) => RefCell::new(Some(parser)),
      Err(err) => {
        warn!("failed to load zsh grammar: {}", err);
        RefCell::new(None)
      }
    }
  };
  static BASH_PARSER: RefCell<Option<Parser>> = {
    let mut parser = Parser::new();
    match parser.set_language(&tree_sitter_bash::LANGUAGE.into()) {
      Ok(()) => RefCell::new(Some(parser)),
      Err(err) => {
        warn!("failed to load bash grammar: {}", err);
        RefCell::new(None)
      }
    }
  };
}

pub(crate) fn tokenize_tree_zsh(input: &str) -> Option<Vec<Token>> {
  ZSH_PARSER.with(|parser| {
    let mut parser = parser.borrow_mut();
    let parser = parser.as_mut()?;
    let tree = parser.parse(input, None)?;
    if tree.root_node().has_error() {
      return None;
    }
    let tokens = tokens_from_tree(input, &tree);
    Some(merge_special_tokens(input, tokens))
  })
}

pub(crate) fn tokenize_tree_bash(input: &str) -> Option<Vec<Token>> {
  BASH_PARSER.with(|parser| {
    let mut parser = parser.borrow_mut();
    let parser = parser.as_mut()?;
    let tree = parser.parse(input, None)?;
    if tree.root_node().has_error() {
      return None;
    }
    let tokens = tokens_from_tree(input, &tree);
    Some(merge_special_tokens(input, tokens))
  })
}

fn tokens_from_tree(input: &str, tree: &tree_sitter::Tree) -> Vec<Token> {
  let mut tokens = Vec::new();
  let source = input.as_bytes();
  collect_tokens(tree.root_node(), source, &mut tokens);
  tokens
}

fn collect_tokens(node: Node<'_>, source: &[u8], tokens: &mut Vec<Token>) {
  let kind = node.kind();
  if is_heredoc_body_kind(kind) {
    return;
  }
  if is_atomic_expansion(kind) {
    if let Ok(text) = node.utf8_text(source) {
      let raw = strip_quote_chars(text.trim());
      if !raw.is_empty() {
        let kind = TokenKind::Word;
        let normalized = normalize(&raw, &kind);
        tokens.push(Token {
          raw,
          kind,
          normalized,
        });
      }
    }
    return;
  }
  if is_string_kind(kind) {
    if let Ok(text) = node.utf8_text(source) {
      let raw = strip_quotes(text);
      let kind = TokenKind::Quoted;
      let normalized = normalize(raw, &kind);
      tokens.push(Token {
        raw: raw.to_string(),
        kind,
        normalized,
      });
    }
    return;
  }

  if is_word_node(kind) {
    if let Ok(text) = node.utf8_text(source) {
      let raw = text.trim();
      if !raw.is_empty() {
        let kind = classify_word(raw);
        let normalized = normalize(raw, &kind);
        tokens.push(Token {
          raw: raw.to_string(),
          kind,
          normalized,
        });
      }
    }
    return;
  }

  if node.child_count() == 0 {
    if let Ok(text) = node.utf8_text(source) {
      let raw = text.trim();
      if raw.is_empty() {
        return;
      }
      if is_operator_text(raw) {
        tokens.push(Token {
          raw: raw.to_string(),
          kind: TokenKind::Operator,
          normalized: raw.to_string(),
        });
        return;
      }
      if is_redirect_text(raw) {
        tokens.push(Token {
          raw: raw.to_string(),
          kind: TokenKind::Redirect,
          normalized: raw.to_string(),
        });
        return;
      }
      if raw.starts_with('$') {
        let kind = TokenKind::Variable;
        tokens.push(Token {
          raw: raw.to_string(),
          kind: kind.clone(),
          normalized: normalize(raw, &kind),
        });
        return;
      }
      if is_wordish(raw) {
        let kind = classify_word(raw);
        let normalized = normalize(raw, &kind);
        tokens.push(Token {
          raw: raw.to_string(),
          kind,
          normalized,
        });
      }
    }
    return;
  }

  let mut cursor = node.walk();
  for child in node.children(&mut cursor) {
    collect_tokens(child, source, tokens);
  }
}

fn is_string_kind(kind: &str) -> bool {
  matches!(kind, "string" | "raw_string" | "regex")
}

fn is_heredoc_body_kind(kind: &str) -> bool {
  kind.contains("heredoc_body")
    || kind.contains("heredoc_content")
    || kind.contains("heredoc_end")
    || kind.contains("heredoc_delimiter")
}

fn is_word_node(kind: &str) -> bool {
  kind == "word" || kind.ends_with("_word")
}

fn strip_quotes(input: &str) -> &str {
  let bytes = input.as_bytes();
  if bytes.len() >= 2 {
    if bytes.len() >= 3 && bytes[0] as char == '$' {
      let second = bytes[1] as char;
      let last = bytes[bytes.len() - 1] as char;
      if (second == '\'' && last == '\'') || (second == '"' && last == '"') {
        return &input[2..bytes.len() - 1];
      }
    }
    let first = bytes[0] as char;
    let last = bytes[bytes.len() - 1] as char;
    if (first == '\'' && last == '\'') || (first == '"' && last == '"') {
      return &input[1..bytes.len() - 1];
    }
  }
  input
}

fn strip_quote_chars(input: &str) -> String {
  input.chars().filter(|c| !matches!(c, '"' | '\'')).collect()
}

fn is_operator_text(raw: &str) -> bool {
  matches!(raw, "|" | "||" | "&" | "&&" | ";" | "(" | ")")
}

fn is_redirect_text(raw: &str) -> bool {
  if !raw.contains('<') && !raw.contains('>') {
    return false;
  }
  raw
    .chars()
    .all(|c| c.is_ascii_digit() || c == '<' || c == '>' || c == '&')
}

fn is_atomic_expansion(kind: &str) -> bool {
  if kind.contains("expansion") || kind.contains("substitution") || kind.contains("variable_ref") {
    return true;
  }
  matches!(kind, "raw_string")
}

fn is_wordish(raw: &str) -> bool {
  raw.chars().any(|c| {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':' | '$' | '=' | '{' | '}')
  })
}
