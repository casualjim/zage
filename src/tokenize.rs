use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use tree_sitter::{Node, Parser};

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

pub fn tokenize(input: &str) -> Vec<Token> {
  let mut tokens = Vec::new();
  let mut chars = input.chars().peekable();

  while let Some(ch) = chars.peek().copied() {
    if ch.is_whitespace() {
      chars.next();
      continue;
    }

    if let Some(tok) = parse_redirect(&mut chars) {
      tokens.push(tok);
      continue;
    }

    if let Some(tok) = parse_operator(&mut chars) {
      tokens.push(tok);
      continue;
    }

    if ch == '\'' || ch == '"' {
      let tok = parse_quoted(&mut chars);
      tokens.push(tok);
      continue;
    }

    let tok = parse_word(&mut chars);
    tokens.push(tok);
  }

  tokens
}

pub fn token_strings(input: &str) -> (Vec<String>, Vec<String>) {
  let tokens = tokenize(input);
  let raw = tokens.iter().map(|t| t.raw.clone()).collect();
  let normalized = tokens.iter().map(|t| t.normalized.clone()).collect();
  (raw, normalized)
}

pub fn token_strings_index(shellname: &str, input: &str) -> (Vec<String>, Vec<String>) {
  let tokens = tokenize_index(shellname, input);
  let raw = tokens.iter().map(|t| t.raw.clone()).collect();
  let normalized = tokens.iter().map(|t| t.normalized.clone()).collect();
  (raw, normalized)
}

pub fn tokenize_index(shellname: &str, input: &str) -> Vec<Token> {
  match shellname {
    "zsh" => tokenize_tree_zsh(input).unwrap_or_else(|| tokenize(input)),
    "bash" | "sh" => tokenize_tree_bash(input).unwrap_or_else(|| tokenize(input)),
    _ => tokenize(input),
  }
}

#[derive(Debug, Clone)]
pub struct CommandParts {
  pub head: String,
  pub env: Vec<Token>,
  pub flags: Vec<String>,
  pub args: Vec<Token>,
}

pub fn extract_command_parts(input: &str, tokens: &[Token]) -> Option<CommandParts> {
  let spans = token_spans(input, tokens);
  let mut env: Vec<Token> = Vec::new();
  let mut idx = 0usize;
  let mut skip_next = false;
  while idx < tokens.len() {
    let token = &tokens[idx];
    if skip_next {
      skip_next = false;
      idx += 1;
      continue;
    }
    if matches!(token.kind, TokenKind::Redirect) {
      if redirect_needs_target(&token.raw) {
        skip_next = true;
      }
      idx += 1;
      continue;
    }
    if is_number(&token.raw) {
      if let Some(next) = tokens.get(idx + 1) {
        if matches!(next.kind, TokenKind::Redirect) {
          if redirect_needs_target(&next.raw) {
            skip_next = true;
          }
          idx += 2;
          continue;
        }
      }
    }
    if matches!(token.kind, TokenKind::Assignment) {
      if token.raw.ends_with('=') {
        if let Some(val) = tokens.get(idx + 1) {
          if matches!(val.kind, TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment) {
            let cur_span = spans.get(idx);
            let val_span = spans.get(idx + 1);
            if let (Some(cur), Some(val_span)) = (cur_span, val_span) {
              if is_adjacent_or_quoted(input, cur.end, val_span.start) {
                let raw = format!("{}{}", token.raw, val.raw);
                env.push(make_assignment_token(raw));
                idx += 2;
                continue;
              }
            }
          }
        }
      }
      env.push(token.clone());
      idx += 1;
      continue;
    }
    if looks_like_assignment_lhs(token.raw.as_str()) {
      if let Some(next) = tokens.get(idx + 1) {
        if next.raw == "=" {
          let lhs_span = spans.get(idx);
          let eq_span = spans.get(idx + 1);
          let adjacent = matches!((lhs_span, eq_span), (Some(lhs), Some(eq)) if lhs.end == eq.start);
          if adjacent {
            let mut raw = token.raw.clone();
            raw.push('=');
            if let Some(val) = tokens.get(idx + 2) {
              if matches!(val.kind, TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment) {
                let val_span = spans.get(idx + 2);
                let adjacent_val = match (eq_span, val_span) {
                  (Some(eq), Some(val)) => is_adjacent_or_quoted(input, eq.end, val.start),
                  _ => false,
                };
                if adjacent_val {
                  raw.push_str(&val.raw);
                  env.push(make_assignment_token(raw));
                  idx += 3;
                  continue;
                }
              }
            }
            env.push(make_assignment_token(raw));
            idx += 2;
            continue;
          }
        }
        if next.raw.starts_with('=') {
          let lhs_span = spans.get(idx);
          let eq_span = spans.get(idx + 1);
          let adjacent = matches!(
            (lhs_span, eq_span),
            (Some(lhs), Some(eq)) if is_adjacent_or_quoted(input, lhs.end, eq.start)
          );
          if adjacent {
            let raw = format!("{}{}", token.raw, next.raw);
            env.push(make_assignment_token(raw));
            idx += 2;
            continue;
          }
        }
      }
    }
    break;
  }

  if idx >= tokens.len() {
    return None;
  }
  let head_idx = idx;

  let head_raw = tokens[head_idx].raw.trim();
  if head_raw.is_empty() {
    return None;
  }

  let mut head = head_raw.to_string();
  let mut start_idx = head_idx + 1;
  if is_subcommand_cli(head_raw) {
    if let Some(next) = tokens.get(start_idx) {
      if matches!(next.kind, TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment)
        && !is_flag_token(&next.raw)
      {
        head = format!("{} {}", head_raw, next.raw.trim());
        start_idx += 1;
      }
    }
  }

  let mut flags = Vec::new();
  let mut args = Vec::new();
  let mut end_of_options = false;
  let mut skip_next = false;
  for idx in start_idx..tokens.len() {
    let token = &tokens[idx];
    if matches!(token.kind, TokenKind::Operator) {
      if is_command_separator(&token.raw) {
        break;
      }
      continue;
    }
    if matches!(token.kind, TokenKind::Redirect) {
      skip_next = true;
      continue;
    }
    if is_number(&token.raw) {
      if let Some(next) = tokens.get(idx + 1) {
        if matches!(next.kind, TokenKind::Redirect) {
          skip_next = true;
          continue;
        }
      }
    }
    if skip_next {
      if matches!(token.kind, TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment) {
        skip_next = false;
      }
      continue;
    }
    if token.raw == "--" {
      end_of_options = true;
      continue;
    }
    if !end_of_options && is_flag_token(&token.raw) {
      flags.push(token.raw.clone());
      continue;
    }
    if matches!(token.kind, TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment) {
      args.push(token.clone());
    }
  }

  let args = merge_url_tokens(args);

  Some(CommandParts {
    head,
    env,
    flags,
    args,
  })
}

pub fn normalized_tokens(input: &str) -> Vec<String> {
  tokenize(input)
    .into_iter()
    .map(|t| t.normalized)
    .collect()
}

pub fn normalize_token(raw: &str) -> String {
  tokenize(raw)
    .into_iter()
    .next()
    .map(|t| t.normalized)
    .unwrap_or_else(|| raw.to_string())
}

fn parse_operator(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Option<Token> {
  let ch = chars.peek().copied()?;
  let op = match ch {
    '&' => {
      chars.next();
      if matches!(chars.peek(), Some('&')) {
        chars.next();
        "&&".to_string()
      } else {
        "&".to_string()
      }
    }
    '|' => {
      chars.next();
      if matches!(chars.peek(), Some('|')) {
        chars.next();
        "||".to_string()
      } else {
        "|".to_string()
      }
    }
    ';' | '(' | ')' => {
      chars.next();
      ch.to_string()
    }
    _ => return None,
  };

  Some(Token {
    raw: op.clone(),
    kind: TokenKind::Operator,
    normalized: op,
  })
}

fn parse_redirect(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Option<Token> {
  let mut buf = String::new();
  let mut iter = chars.clone();
  let ch = iter.peek().copied()?;

  if ch.is_ascii_digit() {
    while let Some(c) = iter.peek().copied() {
      if c.is_ascii_digit() {
        buf.push(c);
        iter.next();
      } else {
        break;
      }
    }
  } else if ch == '&' {
    buf.push(ch);
    iter.next();
  }

  let op = match iter.peek().copied() {
    Some('>') | Some('<') => iter.next().unwrap(),
    _ => return None,
  };
  buf.push(op);

  if let Some(next) = iter.peek().copied() {
    if next == op {
      buf.push(next);
      iter.next();
    }
  }

  if let Some('&') = iter.peek().copied() {
    buf.push('&');
    iter.next();
    while let Some(c) = iter.peek().copied() {
      if c.is_ascii_digit() {
        buf.push(c);
        iter.next();
      } else {
        break;
      }
    }
  }

  for _ in 0..buf.chars().count() {
    chars.next();
  }

  Some(Token {
    raw: buf.clone(),
    kind: TokenKind::Redirect,
    normalized: buf,
  })
}

fn parse_quoted(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Token {
  let quote = chars.next().unwrap();
  let mut buf = String::new();
  while let Some(ch) = chars.next() {
    if ch == quote {
      break;
    }
    if quote == '"' && ch == '\\' {
      if let Some(escaped) = chars.next() {
        buf.push(escaped);
        continue;
      }
    }
    buf.push(ch);
  }

  let kind = classify_word(&buf);
  let normalized = normalize(&buf, &kind);
  Token {
    raw: buf,
    kind: TokenKind::Quoted,
    normalized,
  }
}

fn parse_word(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> Token {
  let mut buf = String::new();
  while let Some(ch) = chars.peek().copied() {
    if ch.is_whitespace() || is_operator_start(ch) || is_redirect_start(chars) {
      if (ch == '>' || ch == '<') && buf.ends_with('=') {
        chars.next();
        buf.push(ch);
        continue;
      }
      break;
    }
    chars.next();
    if ch == '\\' {
      if let Some(escaped) = chars.next() {
        buf.push(escaped);
      }
      continue;
    }
    buf.push(ch);
  }

  let kind = classify_word(&buf);
  let normalized = normalize(&buf, &kind);
  Token {
    raw: buf,
    kind,
    normalized,
  }
}

fn is_operator_start(ch: char) -> bool {
  matches!(ch, '&' | '|' | ';' | '(' | ')')
}

fn is_redirect_start(chars: &std::iter::Peekable<std::str::Chars<'_>>) -> bool {
  let mut iter = chars.clone();
  let ch = match iter.peek().copied() {
    Some(c) => c,
    None => return false,
  };
  if ch.is_ascii_digit() {
    while let Some(c) = iter.peek().copied() {
      if c.is_ascii_digit() {
        iter.next();
      } else {
        break;
      }
    }
    return matches!(iter.peek().copied(), Some('>') | Some('<'));
  }
  if ch == '>' || ch == '<' {
    return true;
  }
  if ch == '&' {
    iter.next();
    return matches!(iter.peek().copied(), Some('>') | Some('<'));
  }
  false
}

fn classify_word(raw: &str) -> TokenKind {
  if raw.starts_with('$') {
    return TokenKind::Variable;
  }
  if is_assignment(raw) {
    return TokenKind::Assignment;
  }
  TokenKind::Word
}

fn normalize(raw: &str, kind: &TokenKind) -> String {
  match kind {
    TokenKind::Operator | TokenKind::Redirect => raw.to_string(),
    TokenKind::Variable => "VAR".to_string(),
    TokenKind::Assignment => "ASSIGN".to_string(),
    _ => {
      if is_path(raw) {
        return "PATH".to_string();
      }
      if is_ip(raw) {
        return "IP".to_string();
      }
      if is_number(raw) {
        return "NUM".to_string();
      }
      if is_hash(raw) {
        return "HASH".to_string();
      }
      raw.to_ascii_lowercase()
    }
  }
}

fn is_assignment(raw: &str) -> bool {
  let mut parts = raw.splitn(2, '=');
  let lhs = match parts.next() {
    Some(p) => p,
    None => return false,
  };
  let _rhs = match parts.next() {
    Some(p) => p,
    None => return false,
  };
  if lhs.is_empty() {
    return false;
  }
  if lhs.starts_with('-') {
    return false;
  }
  looks_like_assignment_lhs(lhs)
}

fn looks_like_assignment_lhs(raw: &str) -> bool {
  if raw.is_empty() || raw.starts_with('-') {
    return false;
  }
  let mut chars = raw.chars();
  let first = match chars.next() {
    Some(c) => c,
    None => return false,
  };
  if !(first.is_ascii_alphabetic() || first == '_') {
    return false;
  }
  chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn is_path(raw: &str) -> bool {
  raw.starts_with('/')
    || raw.starts_with("./")
    || raw.starts_with("../")
    || raw.starts_with('~')
    || raw.contains('/')
}

fn is_number(raw: &str) -> bool {
  !raw.is_empty() && raw.chars().all(|c| c.is_ascii_digit())
}

fn is_hash(raw: &str) -> bool {
  if raw.len() < 8 {
    return false;
  }
  raw.chars()
    .all(|c| c.is_ascii_hexdigit())
}

fn is_ip(raw: &str) -> bool {
  let parts: Vec<&str> = raw.split('.').collect();
  if parts.len() != 4 {
    return false;
  }
  for part in parts {
    if part.is_empty() || part.len() > 3 {
      return false;
    }
    if let Ok(val) = part.parse::<u8>() {
      if val.to_string() != part && part.starts_with('0') && part != "0" {
        return false;
      }
    } else {
      return false;
    }
  }
  true
}

thread_local! {
  static ZSH_PARSER: RefCell<Parser> = {
    let mut parser = Parser::new();
    parser
      .set_language(&tree_sitter_zsh::LANGUAGE.into())
      .expect("zsh grammar");
    RefCell::new(parser)
  };
  static BASH_PARSER: RefCell<Parser> = {
    let mut parser = Parser::new();
    parser
      .set_language(&tree_sitter_bash::LANGUAGE.into())
      .expect("bash grammar");
    RefCell::new(parser)
  };
}

fn tokenize_tree_zsh(input: &str) -> Option<Vec<Token>> {
  ZSH_PARSER.with(|parser| {
    let mut parser = parser.borrow_mut();
    let tree = parser.parse(input, None)?;
    if tree.root_node().has_error() {
      return None;
    }
    let tokens = tokens_from_tree(input, &tree);
    Some(merge_special_tokens(input, tokens))
  })
}

fn tokenize_tree_bash(input: &str) -> Option<Vec<Token>> {
  BASH_PARSER.with(|parser| {
    let mut parser = parser.borrow_mut();
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
        tokens.push(Token { raw, kind, normalized });
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
  input
    .chars()
    .filter(|c| !matches!(c, '"' | '\''))
    .collect()
}

fn is_operator_text(raw: &str) -> bool {
  matches!(raw, "|" | "||" | "&" | "&&" | ";" | "(" | ")")
}

fn is_redirect_text(raw: &str) -> bool {
  if !raw.contains('<') && !raw.contains('>') {
    return false;
  }
  raw.chars()
    .all(|c| c.is_ascii_digit() || c == '<' || c == '>' || c == '&')
}

fn redirect_needs_target(raw: &str) -> bool {
  raw.ends_with('<') || raw.ends_with('>') || raw.ends_with('&')
}

fn is_adjacent_or_quoted(input: &str, left_end: usize, right_start: usize) -> bool {
  if left_end == right_start {
    return true;
  }
  if left_end > right_start || right_start > input.len() {
    return false;
  }
  let between = &input[left_end..right_start];
  !between.chars().any(|c| !matches!(c, '"' | '\''))
}

fn is_atomic_expansion(kind: &str) -> bool {
  if kind.contains("expansion")
    || kind.contains("substitution")
    || kind.contains("variable_ref")
  {
    return true;
  }
  matches!(
    kind,
    "raw_string"
  )
}

fn is_wordish(raw: &str) -> bool {
  raw.chars()
    .any(|c| {
      c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':' | '$' | '=' | '{' | '}')
    })
}

fn is_flag_token(raw: &str) -> bool {
  raw.starts_with('-') && raw.len() > 1
}

fn is_command_separator(raw: &str) -> bool {
  matches!(raw, "|" | "||" | "&&" | ";")
}

fn make_assignment_token(raw: String) -> Token {
  let kind = TokenKind::Assignment;
  let normalized = normalize(&raw, &kind);
  Token { raw, kind, normalized }
}

fn merge_url_tokens(args: Vec<Token>) -> Vec<Token> {
  let mut merged = Vec::new();
  let mut i = 0usize;
  while i < args.len() {
    if i + 2 < args.len()
      && (args[i].raw == "http" || args[i].raw == "https")
      && args[i + 1].raw == ":"
      && args[i + 2].raw.starts_with("//")
    {
      let raw = format!("{}{}{}", args[i].raw, args[i + 1].raw, args[i + 2].raw);
      let kind = TokenKind::Word;
      let normalized = normalize(&raw, &kind);
      merged.push(Token { raw, kind, normalized });
      i += 3;
      continue;
    }
    if i + 1 < args.len() && args[i].raw == "$" {
      let raw = format!("${}", args[i + 1].raw);
      let kind = TokenKind::Variable;
      let normalized = normalize(&raw, &kind);
      merged.push(Token { raw, kind, normalized });
      i += 2;
      continue;
    }
    merged.push(args[i].clone());
    i += 1;
  }
  merged
}

fn merge_special_tokens(input: &str, tokens: Vec<Token>) -> Vec<Token> {
  let spans = token_spans(input, &tokens);
  if spans.is_empty() {
    return tokens;
  }

  let mut merged = Vec::new();
  let mut i = 0usize;
  while i < spans.len() {
    let mut current = spans[i].token.clone();
    let mut j = i;
    while j + 1 < spans.len()
      && spans[j].found
      && spans[j + 1].found
      && is_adjacent_or_quoted(input, spans[j].end, spans[j + 1].start)
    {
      let next = &spans[j + 1].token;
      if should_merge_variable_modifier(&current.raw, &next.raw)
        || should_merge_colon_chain(&current.raw, &next.raw)
        || should_merge_wordish(&current, next)
      {
        let raw = format!("{}{}", current.raw, next.raw);
        let kind = classify_word(&raw);
        let normalized = normalize(&raw, &kind);
        current = Token { raw, kind, normalized };
        j += 1;
        continue;
      }
      break;
    }
    merged.push(current);
    i = j + 1;
  }

  merged
}

fn should_merge_variable_modifier(current: &str, next: &str) -> bool {
  current.starts_with('$')
    && current.ends_with(':')
    && next
      .chars()
      .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == ':' || c == '-')
}

fn should_merge_colon_chain(current: &str, next: &str) -> bool {
  let current_colon = current == ":" || current.contains(':');
  if !current_colon {
    return false;
  }
  let next_trimmed = next.strip_prefix(':').unwrap_or(next);
  next_trimmed
    .chars()
    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.')
}

fn should_merge_wordish(current: &Token, next: &Token) -> bool {
  is_wordish_kind(&current.kind) && is_wordish_kind(&next.kind)
}

fn is_wordish_kind(kind: &TokenKind) -> bool {
  matches!(
    kind,
    TokenKind::Word | TokenKind::Variable | TokenKind::Quoted | TokenKind::Assignment
  )
}

struct TokenSpan {
  token: Token,
  start: usize,
  end: usize,
  found: bool,
}

fn token_spans(input: &str, tokens: &[Token]) -> Vec<TokenSpan> {
  let mut spans = Vec::new();
  let mut search_start = 0usize;
  for token in tokens {
    if token.raw.is_empty() {
      continue;
    }
    if let Some(found) = input[search_start..].find(&token.raw) {
      let start = search_start + found;
      let end = start + token.raw.len();
      spans.push(TokenSpan {
        token: token.clone(),
        start,
        end,
        found: true,
      });
      search_start = end;
    } else {
      spans.push(TokenSpan {
        token: token.clone(),
        start: search_start,
        end: search_start,
        found: false,
      });
    }
  }
  spans
}

fn is_subcommand_cli(cmd: &str) -> bool {
  matches!(
    cmd,
    "git"
      | "cargo"
      | "kubectl"
      | "docker"
      | "docker-compose"
      | "npm"
      | "pnpm"
      | "yarn"
      | "gh"
      | "gcloud"
      | "aws"
      | "terraform"
      | "poetry"
      | "pip"
      | "pipx"
      | "uv"
  )
}
