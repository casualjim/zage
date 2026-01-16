use super::TokenKind;
use super::lexer::tokenize;

pub fn normalized_tokens(input: &str) -> Vec<String> {
  tokenize(input).into_iter().map(|t| t.normalized).collect()
}

pub fn normalize_token(raw: &str) -> String {
  tokenize(raw)
    .into_iter()
    .next()
    .map(|t| t.normalized)
    .unwrap_or_else(|| raw.to_string())
}

pub fn normalize_command_whitespace(input: &str) -> String {
  let mut out = String::with_capacity(input.len());
  let mut chars = input.chars().peekable();
  let mut in_single = false;
  let mut in_double = false;
  let mut last_was_space = false;

  while let Some(ch) = chars.next() {
    if ch == '\\' && !in_single {
      out.push(ch);
      if let Some(next) = chars.next() {
        out.push(next);
      }
      last_was_space = false;
      continue;
    }

    if ch == '\'' && !in_double {
      in_single = !in_single;
      out.push(ch);
      last_was_space = false;
      continue;
    }
    if ch == '"' && !in_single {
      in_double = !in_double;
      out.push(ch);
      last_was_space = false;
      continue;
    }

    if ch.is_whitespace() && !in_single && !in_double {
      if !out.is_empty() && !last_was_space {
        out.push(' ');
        last_was_space = true;
      }
      continue;
    }

    out.push(ch);
    last_was_space = false;
  }

  if last_was_space {
    out.pop();
  }

  out
}

pub(crate) fn classify_word(raw: &str) -> TokenKind {
  if raw.starts_with('$') {
    return TokenKind::Variable;
  }
  if is_assignment(raw) {
    return TokenKind::Assignment;
  }
  TokenKind::Word
}

pub(crate) fn normalize(raw: &str, kind: &TokenKind) -> String {
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

pub(crate) fn is_number(raw: &str) -> bool {
  !raw.is_empty() && raw.chars().all(|c| c.is_ascii_digit())
}

pub(crate) fn is_assignment(raw: &str) -> bool {
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

pub(crate) fn looks_like_assignment_lhs(raw: &str) -> bool {
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

fn is_hash(raw: &str) -> bool {
  if raw.len() < 8 {
    return false;
  }
  raw.chars().all(|c| c.is_ascii_hexdigit())
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

#[cfg(test)]
mod tests {
  use super::normalize_command_whitespace;

  #[test]
  fn normalize_command_whitespace_collapses_unquoted_spaces() {
    let input = "  cargo  install --path .  --force --locked ";
    assert_eq!(
      normalize_command_whitespace(input),
      "cargo install --path . --force --locked"
    );
  }

  #[test]
  fn normalize_command_whitespace_preserves_quoted_and_escaped_spaces() {
    let input = "echo  'a  b'  \"c  d\"  foo\\ bar";
    assert_eq!(
      normalize_command_whitespace(input),
      "echo 'a  b' \"c  d\" foo\\ bar"
    );
  }
}
