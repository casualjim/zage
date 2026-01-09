use super::Token;
use super::TokenKind;
use super::normalize::{classify_word, normalize};
use super::tree_sitter::{tokenize_tree_bash, tokenize_tree_zsh};

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

  if let Some(next) = iter.peek().copied()
    && next == op
  {
    buf.push(next);
    iter.next();
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
    if quote == '"'
      && ch == '\\'
      && let Some(escaped) = chars.next()
    {
      buf.push(escaped);
      continue;
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
