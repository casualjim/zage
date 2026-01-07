use std::fs;
use std::path::{Path, PathBuf};

use tree_sitter::{Node, Parser};

use csv::ReaderBuilder;
use xz2::read::XzDecoder;

use zage::tokenize::{extract_command_parts, tokenize_index, Token, TokenKind};

#[test]
fn test_bash_corpus_parsing() {
  let corpus_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("corpus")
    .join("bash");
  run_corpus(&corpus_dir, "bash", tree_sitter_bash::LANGUAGE.into());
}

#[test]
fn test_zsh_corpus_parsing() {
  let corpus_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("corpus")
    .join("zsh");
  run_corpus(&corpus_dir, "zsh", tree_sitter_zsh::LANGUAGE.into());
}

#[test]
fn test_nl2sh_alfa_parsing() {
  let corpus_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("corpus")
    .join("nl2sh-alfa");
  run_nl2sh_alfa(&corpus_dir);
}

fn run_corpus(corpus_dir: &Path, shell: &str, language: tree_sitter::Language) {
  let corpus_files = list_corpus_files(corpus_dir);
  assert!(
    !corpus_files.is_empty(),
    "no corpus files found in {}",
    corpus_dir.display()
  );

  let mut parser = Parser::new();
  parser.set_language(&language).expect("set language");

  for path in corpus_files {
    let content = fs::read_to_string(&path).expect("read corpus file");
    let cases = parse_corpus_cases(&content);
    for (idx, case) in cases.iter().enumerate() {
      if case.input.trim().is_empty() {
        continue;
      }
      let tree = parser.parse(&case.input, None).expect("parse corpus input");
      let source = case.input.as_bytes();
      let command_nodes = collect_command_nodes(tree.root_node());
      if command_nodes.is_empty() {
        continue;
      }
      for command in command_nodes {
        let text = command.utf8_text(source).unwrap_or_default();
        if text.trim().is_empty() {
          continue;
        }
        let ast_parts = extract_ast_parts(command, source);
        if ast_parts.head.is_empty() {
          continue;
        }
        let tokens = tokenize_index(shell, text);
        let parts = extract_command_parts(text, &tokens).unwrap_or_else(|| {
          panic!(
            "failed to extract command parts for {} case {}:{}\ninput: {}\ncommand: {}",
            shell,
            path.display(),
            idx,
            case.input,
            text
          )
        });

        let head_expected = normalize_for_compare(&ast_parts.head);
        let head_observed = normalize_for_compare(&parts.head);
        if is_simple_token(&head_expected) {
          assert!(
            head_observed.starts_with(&head_expected),
            "head mismatch for {} case {}:{}\ninput: {}\ncommand: {}\nexpected head: {}\nobserved head: {}",
            shell,
            path.display(),
            idx,
            case.input,
            text,
            head_expected,
            head_observed
          );
        }
        let env_raw: Vec<String> = parts.env.iter().map(|t| t.raw.clone()).collect();
        assert_eq!(
          env_raw, ast_parts.env,
          "env mismatch for {} case {}:{}\ninput: {}\ncommand: {}",
          shell,
          path.display(),
          idx,
          case.input,
          text
        );

        if ast_parts.args.iter().all(|arg| is_simple_token(arg)) && is_simple_command_tokens(&tokens) {
          let expected_joined = normalize_join(&ast_parts.args);
          let observed_args = collect_arg_sequence(text, &tokens);
          let observed_joined = normalize_join(&observed_args);
          let expected_norm = normalize_for_compare(&expected_joined);
          let observed_norm = normalize_for_compare(&observed_joined);
          assert_eq!(
            observed_norm,
            expected_norm,
            "arg sequence mismatch for {} case {}:{}\ninput: {}\ncommand: {}\nexpected args: {:?}\nobserved flags: {:?}\nobserved args: {:?}",
            shell,
            path.display(),
            idx,
            case.input,
            text,
            ast_parts.args,
            parts.flags,
            parts.args
          );
        }
      }
    }
  }
}

fn run_nl2sh_alfa(corpus_dir: &Path) {
  let train = choose_nl2sh_path(corpus_dir, "train");
  let test = corpus_dir.join("test.csv");
  assert!(train.exists(), "missing NL2SH-ALFA train.csv at {}", train.display());
  assert!(test.exists(), "missing NL2SH-ALFA test.csv at {}", test.display());

  let mut parser = Parser::new();
  parser
    .set_language(&tree_sitter_bash::LANGUAGE.into())
    .expect("set language");

  let mut total = 0usize;
  total += run_nl2sh_csv(&train, &mut parser, &["bash"]);
  total += run_nl2sh_csv(&test, &mut parser, &["bash", "bash2"]);

  assert!(total > 0, "no NL2SH-ALFA commands parsed");
}

fn run_nl2sh_csv(path: &Path, parser: &mut Parser, columns: &[&str]) -> usize {
  let mut reader = open_nl2sh_reader(path);

  let headers = reader
    .headers()
    .unwrap_or_else(|err| panic!("failed to read headers for {}: {}", path.display(), err))
    .clone();

  let mut col_indices = Vec::new();
  for col in columns {
    if let Some(idx) = headers.iter().position(|name| name == *col) {
      col_indices.push((col.to_string(), idx));
    }
  }
  assert!(
    !col_indices.is_empty(),
    "no requested columns {:?} in {}",
    columns,
    path.display()
  );

  let mut count = 0usize;
  for record in reader.records() {
    let record = record.unwrap_or_else(|err| {
      panic!("failed to parse CSV record in {}: {}", path.display(), err)
    });
    for (name, idx) in &col_indices {
      let value = record.get(*idx).unwrap_or("").trim();
      if value.is_empty() {
        continue;
      }
      run_command_case(
        parser,
        "bash",
        value,
        format!("{}:{}", path.display(), name),
      );
      count += 1;
    }
  }
  count
}

fn choose_nl2sh_path(dir: &Path, stem: &str) -> PathBuf {
  let xz_path = dir.join(format!("{}.csv.xz", stem));
  if xz_path.exists() {
    return xz_path;
  }
  dir.join(format!("{}.csv", stem))
}

fn open_nl2sh_reader(path: &Path) -> csv::Reader<Box<dyn std::io::Read>> {
  let file = fs::File::open(path)
    .unwrap_or_else(|err| panic!("failed to open {}: {}", path.display(), err));
  let reader: Box<dyn std::io::Read> = match path.extension().and_then(|ext| ext.to_str()) {
    Some("xz") => Box::new(XzDecoder::new(file)),
    _ => Box::new(file),
  };
  ReaderBuilder::new()
    .has_headers(true)
    .from_reader(reader)
}

fn run_command_case(parser: &mut Parser, shell: &str, input: &str, label: String) {
  let tree = parser.parse(input, None).expect("parse corpus input");
  let source = input.as_bytes();
  let command_nodes = collect_command_nodes(tree.root_node());
  if command_nodes.is_empty() {
    return;
  }
  for command in command_nodes {
    let text = command.utf8_text(source).unwrap_or_default();
    if text.trim().is_empty() {
      continue;
    }
    let ast_parts = extract_ast_parts(command, source);
    if ast_parts.head.is_empty() {
      continue;
    }
    let tokens = tokenize_index(shell, text);
    let parts = extract_command_parts(text, &tokens).unwrap_or_else(|| {
      panic!(
        "failed to extract command parts for {}\ninput: {}\ncommand: {}",
        label, input, text
      )
    });
    assert!(
      !parts.head.trim().is_empty(),
      "empty head for {}\ninput: {}\ncommand: {}",
      label,
      input,
      text
    );

    let head_expected = normalize_for_compare(&ast_parts.head);
    let head_observed = normalize_for_compare(&parts.head);
    let template_like = is_template_like(input);
    let tree_has_error = tree.root_node().has_error();
    if !template_like && !tree_has_error && is_simple_token(&head_expected) {
      assert!(
        head_observed.starts_with(&head_expected),
        "head mismatch for {}\ninput: {}\ncommand: {}\nexpected head: {}\nobserved head: {}",
        label,
        input,
        text,
        head_expected,
        head_observed
      );
    }
    let env_raw: Vec<String> = parts.env.iter().map(|t| t.raw.clone()).collect();
    if !template_like && !tree_has_error {
      assert_eq!(
        env_raw, ast_parts.env,
        "env mismatch for {}\ninput: {}\ncommand: {}",
        label, input, text
      );
    }

    if !template_like
      && !tree_has_error
      && ast_parts.args.iter().all(|arg| is_simple_token(arg))
      && is_simple_command_tokens(&tokens)
    {
      let expected_joined = normalize_join(&ast_parts.args);
      let observed_args = collect_arg_sequence(text, &tokens);
      let observed_joined = normalize_join(&observed_args);
      let expected_norm = normalize_for_compare(&expected_joined);
      let observed_norm = normalize_for_compare(&observed_joined);
      assert_eq!(
        observed_norm,
        expected_norm,
        "arg sequence mismatch for {}\ninput: {}\ncommand: {}\nexpected args: {:?}\nobserved flags: {:?}\nobserved args: {:?}",
        label,
        input,
        text,
        ast_parts.args,
        parts.flags,
        parts.args
      );
    }
  }
}

fn is_template_like(input: &str) -> bool {
  if input.contains("...") {
    return true;
  }
  for token in input.split_whitespace() {
    if token.contains('|') {
      return true;
    }
  }
  false
}

fn list_corpus_files(dir: &Path) -> Vec<PathBuf> {
  let mut files = Vec::new();
  let Ok(entries) = fs::read_dir(dir) else {
    return files;
  };
  for entry in entries.flatten() {
    let path = entry.path();
    if !path.is_file() {
      continue;
    }
    if path.extension().and_then(|s| s.to_str()) != Some("txt") {
      continue;
    }
    files.push(path);
  }
  files.sort();
  files
}

#[derive(Debug)]
struct CorpusCase {
  input: String,
}

fn parse_corpus_cases(content: &str) -> Vec<CorpusCase> {
  let mut cases = Vec::new();
  let mut lines = content.lines();
  let mut state = CorpusState::SeekHeader;
  let mut input_lines: Vec<String> = Vec::new();

  while let Some(line) = lines.next() {
    match state {
      CorpusState::SeekHeader => {
        if is_delim(line, '=') {
          state = CorpusState::ReadName;
        }
      }
      CorpusState::ReadName => {
        state = CorpusState::SeekInput;
      }
      CorpusState::SeekInput => {
        if is_delim(line, '=') {
          input_lines.clear();
          state = CorpusState::ReadInput;
        }
      }
      CorpusState::ReadInput => {
        if is_delim(line, '-') {
          cases.push(CorpusCase {
            input: input_lines.join("\n"),
          });
          state = CorpusState::SkipExpected;
        } else {
          input_lines.push(line.to_string());
        }
      }
      CorpusState::SkipExpected => {
        if is_delim(line, '=') {
          state = CorpusState::ReadName;
        }
      }
    }
  }

  cases
}

#[derive(Clone, Copy)]
enum CorpusState {
  SeekHeader,
  ReadName,
  SeekInput,
  ReadInput,
  SkipExpected,
}

fn is_delim(line: &str, ch: char) -> bool {
  let trimmed = line.trim();
  if trimmed.len() < 3 {
    return false;
  }
  trimmed.chars().all(|c| c == ch)
}

#[derive(Debug)]
struct AstParts {
  head: String,
  env: Vec<String>,
  args: Vec<String>,
}

struct ArgSpan {
  raw: String,
  start: usize,
  end: usize,
}

fn extract_ast_parts(command: Node<'_>, source: &[u8]) -> AstParts {
  let head = command
    .child_by_field_name("name")
    .and_then(|node| node.utf8_text(source).ok())
    .map(normalize_ast_text)
    .unwrap_or_default();

  let mut env = Vec::new();
  let mut args = Vec::new();
  let mut seen_head = false;
  for i in 0..command.child_count() {
    let i = i as u32;
    let child = command.child(i).unwrap();
    let field = command.field_name_for_child(i);
    if field == Some("name") {
      seen_head = true;
      continue;
    }
    if !seen_head {
      if is_variable_assignment_node(&child) {
        env.extend(extract_assignment_texts(child, source));
      }
      continue;
    }
    if field == Some("argument") {
      if let Ok(text) = child.utf8_text(source) {
        let raw = text.trim().to_string();
        if !raw.is_empty() {
          args.push(ArgSpan {
            raw,
            start: child.start_byte(),
            end: child.end_byte(),
          });
        }
      }
    }
  }

  let merged_args = merge_adjacent_args(args, source);
  let normalized_args = merged_args
    .into_iter()
    .flat_map(|arg| normalize_arg_text(&arg))
    .collect();
  let mut parts = fold_subcommand_head(AstParts {
    head,
    env,
    args: normalized_args,
  });
  parts = split_head_flags(parts);
  parts.args.retain(|arg| arg != "--");
  parts
}

fn is_variable_assignment_node(node: &Node<'_>) -> bool {
  matches!(
    node.kind(),
    "variable_assignment" | "variable_assignments"
  )
}

fn fold_subcommand_head(mut parts: AstParts) -> AstParts {
  if parts.args.is_empty() {
    return parts;
  }
  if is_subcommand_cli(&parts.head) {
    let sub = parts.args.remove(0);
    parts.head = format!("{} {}", parts.head, sub);
  }
  parts
}

fn split_head_flags(mut parts: AstParts) -> AstParts {
  let mut tokens = parts.head.split_whitespace().collect::<Vec<_>>();
  if tokens.len() <= 1 {
    return parts;
  }
  let head = tokens.remove(0);
  if !is_subcommand_cli(head) {
    return parts;
  }
  if !tokens.iter().all(|tok| tok.starts_with('-')) {
    return parts;
  }
  let mut new_args: Vec<String> = tokens.into_iter().map(|s| s.to_string()).collect();
  new_args.extend(parts.args);
  parts.head = head.to_string();
  parts.args = new_args;
  parts
}

fn extract_assignment_texts(node: Node<'_>, source: &[u8]) -> Vec<String> {
  let mut assignments = Vec::new();
  if node.kind() == "variable_assignment" {
    if let Ok(text) = node.utf8_text(source) {
      if text.contains('"') || text.contains('\'') || text.contains("$(") || text.contains('`') {
        if looks_like_assignment_text(text) {
          let raw = normalize_assignment_text(text);
          if !raw.is_empty() {
            assignments.push(raw);
          }
        }
      } else {
        for part in text.split_whitespace() {
          if looks_like_assignment_text(part) {
            let raw = normalize_assignment_text(part);
            if !raw.is_empty() {
              assignments.push(raw);
            }
          }
        }
      }
    }
    return assignments;
  }

  for i in 0..node.child_count() {
    let i = i as u32;
    let child = node.child(i).unwrap();
    if child.kind() == "variable_assignment" {
      if let Ok(text) = child.utf8_text(source) {
        if text.contains('"') || text.contains('\'') || text.contains("$(") || text.contains('`') {
          if looks_like_assignment_text(text) {
            let raw = normalize_assignment_text(text);
            if !raw.is_empty() {
              assignments.push(raw);
            }
          }
        } else {
          for part in text.split_whitespace() {
            if looks_like_assignment_text(part) {
              let raw = normalize_assignment_text(part);
              if !raw.is_empty() {
                assignments.push(raw);
              }
            }
          }
        }
      }
    }
  }
  assignments
}

fn merge_adjacent_args(args: Vec<ArgSpan>, source: &[u8]) -> Vec<String> {
  if args.is_empty() {
    return Vec::new();
  }
  let mut merged = Vec::new();
  let mut current = args[0].raw.clone();
  let mut current_end = args[0].end;

  for span in args.into_iter().skip(1) {
    let between = &source[current_end..span.start];
    let has_whitespace = between.iter().any(|b| b.is_ascii_whitespace());
    if !has_whitespace {
      let between_str = std::str::from_utf8(between).unwrap_or_default();
      let non_quote = between_str
        .chars()
        .filter(|c| !matches!(c, '"' | '\''))
        .collect::<String>();
      if !non_quote.is_empty() {
        current.push_str(between_str);
      }
      current.push_str(&span.raw);
      current_end = span.end;
    } else {
      merged.push(current);
      current = span.raw;
      current_end = span.end;
    }
  }
  merged.push(current);
  merged
}

fn collect_command_nodes(root: Node<'_>) -> Vec<Node<'_>> {
  let mut out = Vec::new();
  let mut stack = vec![root];
  while let Some(node) = stack.pop() {
    if node.kind() == "command" {
      out.push(node);
      continue;
    }
    for i in (0..node.child_count()).rev() {
      let i = i as u32;
      if let Some(child) = node.child(i) {
        stack.push(child);
      }
    }
  }
  out.sort_by_key(|node| node.start_byte());
  out
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

fn normalize_ast_text(raw: &str) -> String {
  let trimmed = raw.trim();
  if trimmed.len() >= 2 {
    let bytes = trimmed.as_bytes();
    let first = bytes[0] as char;
    let last = bytes[bytes.len() - 1] as char;
    if (first == '\'' && last == '\'') || (first == '"' && last == '"') {
      return trimmed[1..bytes.len() - 1].to_string();
    }
  }
  trimmed
    .chars()
    .filter(|c| !matches!(c, '"' | '\''))
    .collect()
}

fn normalize_assignment_text(raw: &str) -> String {
  let trimmed = raw.trim();
  if let Some((lhs, rhs)) = trimmed.split_once('=') {
    let rhs_norm = normalize_ast_text(rhs);
    return format!("{}={}", lhs.trim(), rhs_norm);
  }
  normalize_ast_text(trimmed)
}

fn normalize_arg_text(raw: &str) -> Vec<String> {
  let trimmed = raw.trim();
  if trimmed.is_empty() {
    return Vec::new();
  }
  if let Some(unquoted) = strip_outer_quotes(trimmed) {
    return vec![unquoted.to_string()];
  }
  let cleaned: String = trimmed
    .chars()
    .filter(|c| !matches!(c, '"' | '\''))
    .collect();
  cleaned
    .split_whitespace()
    .filter(|s| !s.is_empty())
    .map(|s| s.to_string())
    .collect()
}

fn strip_outer_quotes(raw: &str) -> Option<&str> {
  if raw.len() < 2 {
    return None;
  }
  let bytes = raw.as_bytes();
  let first = bytes[0] as char;
  let last = bytes[bytes.len() - 1] as char;
  if (first == '\'' && last == '\'') || (first == '"' && last == '"') {
    return Some(&raw[1..bytes.len() - 1]);
  }
  if raw.starts_with("$'") && last == '\'' {
    return Some(&raw[2..bytes.len() - 1]);
  }
  if raw.starts_with("$\"") && last == '"' {
    return Some(&raw[2..bytes.len() - 1]);
  }
  None
}

fn normalize_join(parts: &[String]) -> String {
  parts
    .join(" ")
    .split_whitespace()
    .collect::<Vec<_>>()
    .join(" ")
}

fn normalize_for_compare(input: &str) -> String {
  let cleaned: String = input
    .chars()
    .filter(|c| !matches!(c, '"' | '\'' | '\\'))
    .collect();
  cleaned
    .split_whitespace()
    .collect::<Vec<_>>()
    .join(" ")
}

fn is_simple_token(raw: &str) -> bool {
  let trimmed = raw.trim();
  if trimmed.is_empty() {
    return false;
  }
  if trimmed.chars().any(|c| c.is_whitespace()) {
    return false;
  }
  if trimmed.chars().any(|c| matches!(c, '$' | '`' | '\\' | '(' | ')' | '{' | '}' | '[' | ']' | '<' | '>' | '|' | '&' | ';')) {
    return false;
  }
  if trimmed.contains('*') || trimmed.contains('?') || trimmed.contains('!') {
    return false;
  }
  true
}

fn is_simple_command_tokens(tokens: &[Token]) -> bool {
  tokens.iter().all(|token| {
    match token.kind {
      TokenKind::Operator => {
        !matches!(
          token.raw.as_str(),
          "|" | "||" | "&&" | ";" | "&" | "(" | ")" | "{" | "}"
        )
      }
      _ => !matches!(token.raw.as_str(), "(" | ")" | "{" | "}"),
    }
  })
}

fn collect_arg_sequence(input: &str, tokens: &[Token]) -> Vec<String> {
  let spans = token_spans(input, tokens);
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
          if is_wordish_token(val) {
            let cur_span = spans.get(idx);
            let val_span = spans.get(idx + 1);
            if let (Some(cur), Some(val_span)) = (cur_span, val_span) {
              if is_adjacent_or_quoted_span(input, cur.end, val_span.start) {
                idx += 2;
                continue;
              }
            }
          }
        }
      }
      idx += 1;
      continue;
    }

    if looks_like_assignment_lhs(&token.raw) {
      if let Some(next) = tokens.get(idx + 1) {
        if next.raw == "=" || next.raw.starts_with('=') {
          idx += if next.raw == "=" { 2 } else { 2 };
          continue;
        }
      }
    }

    if is_wordish_token(token) {
      break;
    }
    idx += 1;
  }

  if idx >= tokens.len() {
    return Vec::new();
  }

  let mut head = tokens[idx].raw.clone();
  let mut start_idx = idx + 1;
  if is_subcommand_cli(&head) {
    if let Some(next) = tokens.get(start_idx) {
      if is_wordish_token(next) && !is_flag_like(&next.raw) {
        head = format!("{} {}", head, next.raw);
        start_idx += 1;
      }
    }
  }
  let _ = head;

  let mut args = Vec::new();
  let mut end_of_options = false;
  let mut skip_next = false;
  for token in tokens.iter().skip(start_idx) {
    if matches!(token.kind, TokenKind::Operator) {
      if is_separator(&token.raw) {
        break;
      }
      continue;
    }
    if matches!(token.kind, TokenKind::Redirect) {
      if redirect_needs_target(&token.raw) {
        skip_next = true;
      }
      continue;
    }
    if skip_next {
      if is_wordish_token(token) {
        skip_next = false;
      }
      continue;
    }
    if token.raw == "--" {
      end_of_options = true;
      continue;
    }
    if !end_of_options && is_flag_like(&token.raw) {
      args.push(token.raw.clone());
      continue;
    }
    if is_wordish_token(token) {
      args.push(token.raw.clone());
    }
  }

  args
}

fn is_wordish_token(token: &Token) -> bool {
  matches!(
    token.kind,
    TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment
  )
}

fn is_flag_like(raw: &str) -> bool {
  raw.starts_with('-') && raw.len() > 1
}

fn looks_like_assignment_lhs(raw: &str) -> bool {
  let trimmed = raw.trim();
  if trimmed.is_empty() || trimmed.starts_with('-') {
    return false;
  }
  let mut chars = trimmed.chars();
  let first = match chars.next() {
    Some(c) => c,
    None => return false,
  };
  if !(first.is_ascii_alphabetic() || first == '_') {
    return false;
  }
  chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn redirect_needs_target(raw: &str) -> bool {
  raw.ends_with('<') || raw.ends_with('>') || raw.ends_with('&')
}

fn is_separator(raw: &str) -> bool {
  matches!(raw, "|" | "||" | "&&" | ";")
}

fn is_number(raw: &str) -> bool {
  !raw.is_empty() && raw.chars().all(|c| c.is_ascii_digit())
}

fn is_adjacent_or_quoted_span(input: &str, left_end: usize, right_start: usize) -> bool {
  if left_end == right_start {
    return true;
  }
  if left_end > right_start || right_start > input.len() {
    return false;
  }
  let between = &input[left_end..right_start];
  !between.chars().any(|c| !matches!(c, '"' | '\''))
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
        start,
        end,
      });
      search_start = end;
    } else {
      spans.push(TokenSpan {
        start: search_start,
        end: search_start,
      });
    }
  }
  spans
}

struct TokenSpan {
  start: usize,
  end: usize,
}

fn looks_like_assignment_text(raw: &str) -> bool {
  let trimmed = raw.trim();
  let Some((lhs, _)) = trimmed.split_once('=') else {
    return false;
  };
  if lhs.is_empty() {
    return false;
  }
  let mut chars = lhs.chars();
  let first = match chars.next() {
    Some(c) => c,
    None => return false,
  };
  if !(first.is_ascii_alphabetic() || first == '_') {
    return false;
  }
  chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}
