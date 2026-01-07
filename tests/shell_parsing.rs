use std::path::Path;

use zage::shell_history;
use zage::tokenize::{extract_command_parts, tokenize_index, Token, TokenKind};

struct SegmentCase {
  head: &'static str,
  env: &'static [&'static str],
  flags: &'static [&'static str],
  args: &'static [&'static str],
}

struct ParseCase {
  shell: &'static str,
  input: &'static str,
  segments: &'static [SegmentCase],
}

fn assert_case(case: &ParseCase) {
  let segments = split_segments_with_text(case.shell, case.input);
  assert_eq!(
    segments.len(),
    case.segments.len(),
    "segment count mismatch for {}",
    case.input
  );

  for (idx, segment) in segments.iter().enumerate() {
    let expected = &case.segments[idx];
    let tokens = tokenize_index(case.shell, segment);
    let parts = extract_command_parts(segment, &tokens).unwrap_or_else(|| {
      panic!(
        "no command parts for shell={} input={} segment={}",
        case.shell, case.input, idx
      )
    });

    assert_eq!(
      parts.head, expected.head,
      "head mismatch for {} segment {}",
      case.input, idx
    );

    let env_raw: Vec<String> = parts.env.iter().map(|t| t.raw.clone()).collect();
    let expected_env: Vec<String> = expected.env.iter().map(|e| (*e).to_string()).collect();
    assert_eq!(
      env_raw, expected_env,
      "env mismatch for {} segment {}",
      case.input, idx
    );

    let mut flags = parts.flags.clone();
    flags.sort();
    let mut expected_flags: Vec<String> = expected.flags.iter().map(|f| (*f).to_string()).collect();
    expected_flags.sort();
    assert_eq!(
      flags, expected_flags,
      "flags mismatch for {} segment {}",
      case.input, idx
    );

    let args_raw: Vec<String> = parts.args.iter().map(|t| t.raw.clone()).collect();
    let expected_args: Vec<String> = expected.args.iter().map(|a| (*a).to_string()).collect();
    assert_eq!(
      args_raw, expected_args,
      "args mismatch for {} segment {}",
      case.input, idx
    );
  }
}

#[test]
fn test_shell_parsing_cases() {
  let cases = [
    ParseCase {
      shell: "zsh",
      input: "git status",
      segments: &[SegmentCase {
        head: "git status",
        env: &[],
        flags: &[],
        args: &[],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "git status",
      segments: &[SegmentCase {
        head: "git status",
        env: &[],
        flags: &[],
        args: &[],
      }],
    },
    ParseCase {
      shell: "zsh",
      input: "sops -d -i --age AGE .env.json",
      segments: &[SegmentCase {
        head: "sops",
        env: &[],
        flags: &["-d", "-i", "--age"],
        args: &["AGE", ".env.json"],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "sops -d -i --age AGE .env.json",
      segments: &[SegmentCase {
        head: "sops",
        env: &[],
        flags: &["-d", "-i", "--age"],
        args: &["AGE", ".env.json"],
      }],
    },
    ParseCase {
      shell: "zsh",
      input: "curl -sS https://example.com/file.txt",
      segments: &[SegmentCase {
        head: "curl",
        env: &[],
        flags: &["-sS"],
        args: &["https://example.com/file.txt"],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "grep -E \"foo|bar\" file.txt",
      segments: &[SegmentCase {
        head: "grep",
        env: &[],
        flags: &["-E"],
        args: &["foo|bar", "file.txt"],
      }],
    },
    ParseCase {
      shell: "zsh",
      input: "FOO=1 BAR=2 git status",
      segments: &[SegmentCase {
        head: "git status",
        env: &["FOO=1", "BAR=2"],
        flags: &[],
        args: &[],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "cmd -- -x -y",
      segments: &[SegmentCase {
        head: "cmd",
        env: &[],
        flags: &[],
        args: &["-x", "-y"],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "cmd -x -- -y",
      segments: &[SegmentCase {
        head: "cmd",
        env: &[],
        flags: &["-x"],
        args: &["-y"],
      }],
    },
    ParseCase {
      shell: "zsh",
      input: "cat file.txt 2>err.log",
      segments: &[SegmentCase {
        head: "cat",
        env: &[],
        flags: &[],
        args: &["file.txt"],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "cmd <in >out 2>err",
      segments: &[SegmentCase {
        head: "cmd",
        env: &[],
        flags: &[],
        args: &[],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "cmd arg1 2>err arg2",
      segments: &[SegmentCase {
        head: "cmd",
        env: &[],
        flags: &[],
        args: &["arg1", "arg2"],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "rg foo file | sort -u",
      segments: &[
        SegmentCase {
          head: "rg",
          env: &[],
          flags: &[],
          args: &["foo", "file"],
        },
        SegmentCase {
          head: "sort",
          env: &[],
          flags: &["-u"],
          args: &[],
        },
      ],
    },
    ParseCase {
      shell: "zsh",
      input: "a b | c d | e f",
      segments: &[
        SegmentCase {
          head: "a",
          env: &[],
          flags: &[],
          args: &["b"],
        },
        SegmentCase {
          head: "c",
          env: &[],
          flags: &[],
          args: &["d"],
        },
        SegmentCase {
          head: "e",
          env: &[],
          flags: &[],
          args: &["f"],
        },
      ],
    },
    ParseCase {
      shell: "zsh",
      input: "echo \"a | b\" | wc -l",
      segments: &[
        SegmentCase {
          head: "echo",
          env: &[],
          flags: &[],
          args: &["a | b"],
        },
        SegmentCase {
          head: "wc",
          env: &[],
          flags: &["-l"],
          args: &[],
        },
      ],
    },
    ParseCase {
      shell: "bash",
      input: "cmd -ab --foo=bar baz",
      segments: &[SegmentCase {
        head: "cmd",
        env: &[],
        flags: &["-ab", "--foo=bar"],
        args: &["baz"],
      }],
    },
    ParseCase {
      shell: "zsh",
      input: "cmd - bar",
      segments: &[SegmentCase {
        head: "cmd",
        env: &[],
        flags: &[],
        args: &["-", "bar"],
      }],
    },
    ParseCase {
      shell: "zsh",
      input: "printf '%s\\n' a b",
      segments: &[SegmentCase {
        head: "printf",
        env: &[],
        flags: &[],
        args: &["%s\\n", "a", "b"],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "cmd \"a b\" 'c d'",
      segments: &[SegmentCase {
        head: "cmd",
        env: &[],
        flags: &[],
        args: &["a b", "c d"],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "cmd $VAR",
      segments: &[SegmentCase {
        head: "cmd",
        env: &[],
        flags: &[],
        args: &["$VAR"],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "VAR=1 cmd VAR2=2",
      segments: &[SegmentCase {
        head: "cmd",
        env: &["VAR=1"],
        flags: &[],
        args: &["VAR2=2"],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "env VAR=1 cmd",
      segments: &[SegmentCase {
        head: "env",
        env: &[],
        flags: &[],
        args: &["VAR=1", "cmd"],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "gcloud compute instances list",
      segments: &[SegmentCase {
        head: "gcloud compute",
        env: &[],
        flags: &[],
        args: &["instances", "list"],
      }],
    },
    ParseCase {
      shell: "bash",
      input: "VAR=1 cmd arg && other",
      segments: &[
        SegmentCase {
          head: "cmd",
          env: &["VAR=1"],
          flags: &[],
          args: &["arg"],
        },
        SegmentCase {
          head: "other",
          env: &[],
          flags: &[],
          args: &[],
        },
      ],
    },
    ParseCase {
      shell: "bash",
      input: "cat <<EOF\nline one\nline two\nEOF",
      segments: &[SegmentCase {
        head: "cat",
        env: &[],
        flags: &[],
        args: &[],
      }],
    },
  ];

  for case in &cases {
    assert_case(case);
  }
}

#[test]
fn test_shell_parsing_history_smoke() {
  let zsh_history_path = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("data")
    .join("zsh.history");
  let zsh_invocations = shell_history::parse_zsh_history(&zsh_history_path, None, None)
    .expect("parse zsh history");
  for invocation in zsh_invocations {
    let tokens = tokenize_index("zsh", &invocation.command);
    assert!(
      !tokens.is_empty(),
      "no tokens for zsh command: {}",
      invocation.command
    );
  }

  let bash_history_path = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("data")
    .join("bash.history");
  let bash_invocations = shell_history::parse_bash_history(&bash_history_path, None, None)
    .expect("parse bash history");
  for invocation in bash_invocations {
    let tokens = tokenize_index("bash", &invocation.command);
    assert!(
      !tokens.is_empty(),
      "no tokens for bash command: {}",
      invocation.command
    );
  }
}

fn is_separator(raw: &str) -> bool {
  matches!(raw, "|" | "||" | "&&" | ";")
}

fn split_segments_with_text(shell: &str, input: &str) -> Vec<String> {
  let tokens = tokenize_index(shell, input);
  assert!(
    !tokens.is_empty() || input.trim().is_empty(),
    "no tokens for shell={} input={}",
    shell,
    input
  );

  let spans = token_spans(input, &tokens);
  if spans.is_empty() {
    return Vec::new();
  }

  let mut segments = Vec::new();
  let mut start = 0usize;
  for span in spans {
    if matches!(span.token.kind, TokenKind::Operator) && is_separator(&span.token.raw) {
      if start <= span.start {
        let slice = input[start..span.start].trim();
        if !slice.is_empty() {
          segments.push(slice.to_string());
        }
      }
      start = span.end;
    }
  }
  if start < input.len() {
    let slice = input[start..].trim();
    if !slice.is_empty() {
      segments.push(slice.to_string());
    }
  }

  segments
}

struct TokenSpan {
  token: Token,
  start: usize,
  end: usize,
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
      });
      search_start = end;
    } else {
      spans.push(TokenSpan {
        token: token.clone(),
        start: search_start,
        end: search_start,
      });
    }
  }
  spans
}
