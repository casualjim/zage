use zage::tokenize::{TokenKind, tokenize, tokenize_index, normalized_tokens};

#[test]
fn test_basic_tokens() {
  let tokens = tokenize("git status");
  assert_eq!(tokens.len(), 2);
  assert_eq!(tokens[0].raw, "git");
  assert_eq!(tokens[1].raw, "status");
}

#[test]
fn test_quoted_tokens() {
  let tokens = tokenize("echo \"hello world\"");
  assert_eq!(tokens.len(), 2);
  assert_eq!(tokens[0].raw, "echo");
  assert_eq!(tokens[1].raw, "hello world");
  assert_eq!(tokens[1].kind, TokenKind::Quoted);
}

#[test]
fn test_redirection_tokens() {
  let tokens = tokenize("cat file.txt 2>err.log");
  let raws: Vec<String> = tokens.iter().map(|t| t.raw.clone()).collect();
  assert_eq!(raws, vec!["cat", "file.txt", "2>", "err.log"]);
  assert_eq!(tokens[2].kind, TokenKind::Redirect);
}

#[test]
fn test_operator_tokens() {
  let tokens = tokenize("cat a | grep b && echo ok");
  let raws: Vec<String> = tokens.iter().map(|t| t.raw.clone()).collect();
  assert_eq!(raws, vec!["cat", "a", "|", "grep", "b", "&&", "echo", "ok"]);
  assert_eq!(tokens[2].kind, TokenKind::Operator);
  assert_eq!(tokens[5].kind, TokenKind::Operator);
}

#[test]
fn test_normalization() {
  let tokens = normalized_tokens("curl http://127.0.0.1:8000/file.txt");
  assert!(tokens.contains(&"path".to_string()) || tokens.contains(&"PATH".to_string()));
}

#[test]
fn test_tokenize_index_bind_key_sequence() {
  let input = "bind -x '\"key_sequence\":command'";
  let tokens = tokenize_index("bash", input);
  let raws: Vec<String> = tokens.iter().map(|t| t.raw.clone()).collect();
  assert_eq!(raws, vec!["bind", "-x", "key_sequence:command"]);
}

#[test]
fn test_tokenize_index_printf_brace_expansion() {
  let input = "printf '=%.0s' {1..99}";
  let tokens = tokenize_index("bash", input);
  let raws: Vec<String> = tokens.iter().map(|t| t.raw.clone()).collect();
  assert_eq!(raws, vec!["printf", "=%.0s", "{1..99}"]);
}
