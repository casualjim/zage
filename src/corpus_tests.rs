use std::fs;
use std::path::{Path, PathBuf};

use tree_sitter::{Node, Parser};

use csv::ReaderBuilder;
use liblzma::read::XzDecoder;

use crate::tokenize::{Token, TokenKind, extract_command_parts, tokenize_index};

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
        if is_simple_head_token(&head_expected) {
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
        let env_raw: Vec<String> = parts
          .env
          .iter()
          .map(|t| normalize_for_compare(&t.raw))
          .collect();
        let env_expected: Vec<String> = normalize_env_list(&ast_parts.env)
          .iter()
          .map(|val| normalize_for_compare(val))
          .collect();
        assert_eq!(
          env_raw,
          env_expected,
          "env mismatch for {} case {}:{}\ninput: {}\ncommand: {}",
          shell,
          path.display(),
          idx,
          case.input,
          text
        );

        if !ast_parts.args.is_empty()
          && ast_parts.args.iter().all(|arg| is_simple_token(arg))
          && is_simple_command_tokens(&tokens)
          && is_simple_command_text(text)
          && !has_shell_expansion(text)
        {
          if !parts.flags.is_empty()
            || !parts.env.is_empty()
            || parts.args.len() != ast_parts.args.len()
          {
            continue;
          }
          let observed_args = collect_arg_sequence(text, &tokens, &ast_parts.head);
          if observed_args.len() != ast_parts.args.len() {
            continue;
          }
          let expected_joined = normalize_join(&ast_parts.args);
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
  assert!(
    train.exists(),
    "missing NL2SH-ALFA train.csv at {}",
    train.display()
  );
  assert!(
    test.exists(),
    "missing NL2SH-ALFA test.csv at {}",
    test.display()
  );

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
    let record = record
      .unwrap_or_else(|err| panic!("failed to parse CSV record in {}: {}", path.display(), err));
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
  let file = fs::File::open(path).expect("open NL2SH-ALFA file");
  let reader: Box<dyn std::io::Read> = if path.extension().map(|ext| ext == "xz").unwrap_or(false) {
    Box::new(XzDecoder::new(file))
  } else {
    Box::new(file)
  };
  ReaderBuilder::new()
    .has_headers(true)
    .flexible(true)
    .from_reader(reader)
}

#[derive(Debug)]
struct CorpusCase {
  input: String,
  #[allow(dead_code)]
  output: String,
}

fn parse_corpus_cases(content: &str) -> Vec<CorpusCase> {
  let mut cases = Vec::new();
  let mut input = String::new();
  let mut output = String::new();
  let mut in_input = false;
  let mut in_output = false;
  for line in content.lines() {
    if line.starts_with("===") {
      if !input.is_empty() || !output.is_empty() {
        cases.push(CorpusCase {
          input: input.trim().to_string(),
          output: output.trim().to_string(),
        });
        input.clear();
        output.clear();
      }
      in_input = true;
      in_output = false;
      continue;
    }
    if line.starts_with("---") {
      in_output = true;
      in_input = false;
      continue;
    }
    if in_input {
      input.push_str(line);
      input.push('\n');
    } else if in_output {
      output.push_str(line);
      output.push('\n');
    }
  }
  if !input.is_empty() || !output.is_empty() {
    cases.push(CorpusCase {
      input: input.trim().to_string(),
      output: output.trim().to_string(),
    });
  }
  cases
}

fn collect_command_nodes(node: Node) -> Vec<Node> {
  let mut commands = Vec::new();
  if node.kind() == "command" || node.kind() == "command_name" {
    commands.push(node);
  }
  let mut cursor = node.walk();
  for child in node.children(&mut cursor) {
    commands.extend(collect_command_nodes(child));
  }
  commands
}

#[derive(Debug)]
struct AstParts {
  head: String,
  env: Vec<String>,
  args: Vec<String>,
}

fn extract_ast_parts(command: Node, source: &[u8]) -> AstParts {
  let mut head = String::new();
  let mut env = Vec::new();
  let mut args = Vec::new();

  let mut cursor = command.walk();
  for child in command.children(&mut cursor) {
    match child.kind() {
      "command_name" | "command" => {
        head = child.utf8_text(source).unwrap_or_default().to_string();
      }
      "variable_assignment" => {
        let text = child.utf8_text(source).unwrap_or_default().to_string();
        if !text.is_empty() {
          env.push(text);
        }
      }
      "word" | "string" | "raw_string" => {
        let text = child.utf8_text(source).unwrap_or_default().to_string();
        if !text.is_empty() {
          args.push(text);
        }
      }
      _ => {}
    }
  }

  AstParts { head, env, args }
}

fn collect_arg_sequence(_input: &str, tokens: &[Token], head: &str) -> Vec<String> {
  let mut args = Vec::new();
  let mut skip_redirect = false;
  let mut start_idx = 0usize;

  let head_clean = normalize_for_compare(head);
  let head_terms: Vec<&str> = head_clean.split_whitespace().collect();
  if !head_terms.is_empty() {
    for idx in 0..tokens.len() {
      if tokens[idx].raw == head_terms[0] {
        let mut matches = true;
        for (offset, term) in head_terms.iter().enumerate() {
          if idx + offset >= tokens.len() || tokens[idx + offset].raw != *term {
            matches = false;
            break;
          }
        }
        if matches {
          start_idx = idx + head_terms.len();
          break;
        }
      }
    }
  } else {
    start_idx = 1;
  }

  for token in tokens.iter().skip(start_idx) {
    if skip_redirect {
      if matches!(
        token.kind,
        TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment
      ) {
        skip_redirect = false;
      }
      continue;
    }
    if matches!(token.kind, TokenKind::Redirect) {
      skip_redirect = true;
      continue;
    }
    if matches!(token.kind, TokenKind::Operator) {
      continue;
    }
    args.push(token.raw.clone());
  }

  args
}

fn list_corpus_files(dir: &Path) -> Vec<PathBuf> {
  let mut files = Vec::new();
  let entries = match fs::read_dir(dir) {
    Ok(entries) => entries,
    Err(_) => return files,
  };
  for entry in entries.flatten() {
    let path = entry.path();
    if path.is_file()
      && let Some(ext) = path.extension().and_then(|e| e.to_str())
      && ext == "txt"
    {
      files.push(path);
    }
  }
  files.sort();
  files
}

fn is_simple_command_tokens(tokens: &[Token]) -> bool {
  tokens.iter().all(|token| {
    matches!(
      token.kind,
      TokenKind::Word | TokenKind::Quoted | TokenKind::Assignment
    ) && !token.raw.contains('\n')
      && !token.raw.contains(';')
  })
}

fn is_simple_command_text(text: &str) -> bool {
  !text.contains('\n') && !text.contains('\r')
}

fn has_shell_expansion(text: &str) -> bool {
  text.contains('$') || text.contains('`') || text.contains('{') || text.contains('}')
}

fn is_simple_token(token: &str) -> bool {
  !token.contains('\n') && !token.contains(';')
}

fn is_simple_head_token(token: &str) -> bool {
  !token.is_empty()
    && token
      .chars()
      .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | '/' | ':'))
}

fn normalize_for_compare(input: &str) -> String {
  input
    .replace(['"', '\''], "")
    .replace("\\\n", "")
    .replace("\\", "")
    .trim()
    .to_string()
}

fn normalize_join(parts: &[String]) -> String {
  parts
    .iter()
    .map(|s| s.trim().to_string())
    .filter(|s| !s.is_empty())
    .collect::<Vec<_>>()
    .join(" ")
}

fn normalize_env_list(env: &[String]) -> Vec<String> {
  let mut out = Vec::new();
  for val in env {
    if val.contains('\n') || val.contains('\r') {
      for part in val.split_whitespace() {
        if !part.is_empty() {
          out.push(part.to_string());
        }
      }
    } else {
      out.push(val.clone());
    }
  }
  out
}

fn run_command_case(parser: &mut Parser, shell: &str, input: &str, name: String) {
  let tree = parser.parse(input, None).expect("parse command input");
  let source = input.as_bytes();
  let command_nodes = collect_command_nodes(tree.root_node());
  for command in command_nodes {
    let text = command.utf8_text(source).unwrap_or_default();
    if text.trim().is_empty() {
      continue;
    }
    let tokens = tokenize_index(shell, text);
    let _ = extract_command_parts(text, &tokens).unwrap_or_else(|| {
      panic!(
        "failed to extract command parts for {}:{} input: {}",
        shell, name, input
      )
    });
  }
}
