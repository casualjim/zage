use super::Token;
use super::TokenKind;
use super::normalize::{classify_word, is_number, looks_like_assignment_lhs, normalize};

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
    if is_number(&token.raw)
      && let Some(next) = tokens.get(idx + 1)
      && matches!(next.kind, TokenKind::Redirect)
    {
      if redirect_needs_target(&next.raw) {
        skip_next = true;
      }
      idx += 2;
      continue;
    }
    if matches!(token.kind, TokenKind::Assignment) {
      if token.raw.ends_with('=')
        && let Some(val) = tokens.get(idx + 1)
        && matches!(
          val.kind,
          TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment
        )
      {
        let cur_span = spans.get(idx);
        let val_span = spans.get(idx + 1);
        if let (Some(cur), Some(val_span)) = (cur_span, val_span)
          && is_adjacent_or_quoted(input, cur.end, val_span.start)
        {
          let raw = format!("{}{}", token.raw, val.raw);
          env.push(make_assignment_token(raw));
          idx += 2;
          continue;
        }
      }
      env.push(token.clone());
      idx += 1;
      continue;
    }
    if looks_like_assignment_lhs(token.raw.as_str())
      && let Some(next) = tokens.get(idx + 1)
    {
      if next.raw == "=" {
        let lhs_span = spans.get(idx);
        let eq_span = spans.get(idx + 1);
        let adjacent = matches!((lhs_span, eq_span), (Some(lhs), Some(eq)) if lhs.end == eq.start);
        if adjacent {
          let mut raw = token.raw.clone();
          raw.push('=');
          if let Some(val) = tokens.get(idx + 2)
            && matches!(
              val.kind,
              TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment
            )
          {
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

  let head = head_raw.to_string();
  let start_idx = head_idx + 1;

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
    if is_number(&token.raw)
      && let Some(next) = tokens.get(idx + 1)
      && matches!(next.kind, TokenKind::Redirect)
    {
      skip_next = true;
      continue;
    }
    if skip_next {
      if matches!(
        token.kind,
        TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment
      ) {
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
    if matches!(
      token.kind,
      TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment
    ) {
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

pub(crate) fn merge_special_tokens(input: &str, tokens: Vec<Token>) -> Vec<Token> {
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
        current = Token {
          raw,
          kind,
          normalized,
        };
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

fn is_flag_token(raw: &str) -> bool {
  raw.starts_with('-') && raw.len() > 1
}

fn is_command_separator(raw: &str) -> bool {
  matches!(raw, "|" | "||" | "&&" | ";")
}

fn make_assignment_token(raw: String) -> Token {
  let kind = TokenKind::Assignment;
  let normalized = normalize(&raw, &kind);
  Token {
    raw,
    kind,
    normalized,
  }
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
      merged.push(Token {
        raw,
        kind,
        normalized,
      });
      i += 3;
      continue;
    }
    if i + 1 < args.len() && args[i].raw == "$" {
      let raw = format!("${}", args[i + 1].raw);
      let kind = TokenKind::Variable;
      let normalized = normalize(&raw, &kind);
      merged.push(Token {
        raw,
        kind,
        normalized,
      });
      i += 2;
      continue;
    }
    merged.push(args[i].clone());
    i += 1;
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
