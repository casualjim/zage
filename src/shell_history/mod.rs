use std::os::unix::fs::MetadataExt;
use std::path::Path;

mod bash;
mod zsh;

pub use crate::core::Invocation;
pub use bash::parse_history_file as parse_bash_history;
pub use zsh::parse_history_file as parse_zsh_history;

/// Which shell history format to import
#[derive(Clone, Copy, Debug)]
pub enum Shell {
  Bash,
  Zsh,
}

impl std::str::FromStr for Shell {
  type Err = String;
  fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
    let name = s.rsplit('/').next().unwrap_or(s);
    match name {
      "bash" => Ok(Shell::Bash),
      "zsh" => Ok(Shell::Zsh),
      other => Err(format!("Unknown shell: {}", other)),
    }
  }
}

// Try to generate a "stable" session id based on the file imported.
// If that fails, just create a random one.
fn generate_import_session_id(histfile: &Path) -> i64 {
  if let Ok(st) = std::fs::metadata(histfile) {
    ((st.ino() << 16) | st.dev()) as i64
  } else {
    (rand::random::<u64>() >> 1) as i64
  }
}

pub fn get_hostname() -> String {
  std::env::var("ZAGE_HOSTNAME").unwrap_or_else(|_| {
    hostname::get()
      .unwrap_or_default()
      .into_string()
      .unwrap_or_default()
  })
}

pub fn detect_shellname() -> String {
  let Some(shell) = std::env::var("SHELL").ok().and_then(|value| {
    Path::new(&value)
      .file_name()
      .map(|name| name.to_string_lossy().to_string())
  }) else {
    return "sh".to_string();
  };
  let normalized = shell.to_lowercase();
  match normalized.as_str() {
    "zsh" | "bash" | "sh" | "fish" | "nushell" | "nu" => normalized,
    _ => shell,
  }
}

fn dedup_invocations(invocations: Vec<Invocation>) -> Vec<Invocation> {
  let mut it = invocations.into_iter();
  let Some(first) = it.next() else {
    return vec![];
  };
  let mut ret = vec![first];
  for elem in it {
    if !elem.sameish(ret.last().unwrap()) {
      ret.push(elem);
    }
  }
  ret
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::fs::File;
  use std::io::Write;
  use tempfile::tempdir;

  #[test]
  fn test_bash_history_timestamps() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("bash.history");
    let mut file = File::create(&file_path).unwrap();
    let history = b"#1620000000\necho hello\n#1620000005\nls -la\n#0\ninvalid\n";
    file.write_all(history).unwrap();

    let invocations = parse_bash_history(&file_path, None, None).unwrap();
    assert_eq!(invocations.len(), 3);
    assert_eq!(invocations[0].command, "echo hello");
    assert_eq!(invocations[0].start_unix_timestamp, Some(1620000000));
    assert_eq!(invocations[1].command, "ls -la");
    assert_eq!(invocations[1].start_unix_timestamp, Some(1620000005));
    assert_eq!(invocations[2].command, "invalid");
    assert_eq!(invocations[2].start_unix_timestamp, Some(1620000005));
  }

  #[test]
  fn test_zsh_history_timestamps() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("zsh.history");
    let mut file = File::create(&file_path).unwrap();
    let history = b": 1620000000:2;echo hello\n: 1620000005:3;ls -la\n: 0:1;invalid\n";
    file.write_all(history).unwrap();

    let invocations = parse_zsh_history(&file_path, None, None).unwrap();
    assert_eq!(invocations.len(), 3);
    assert_eq!(invocations[0].command, "echo hello");
    assert_eq!(invocations[0].start_unix_timestamp, Some(1620000000));
    assert_eq!(invocations[0].end_unix_timestamp, Some(1620000002));
    assert_eq!(invocations[1].command, "ls -la");
    assert_eq!(invocations[1].start_unix_timestamp, Some(1620000005));
    assert_eq!(invocations[1].end_unix_timestamp, Some(1620000008));
    assert_eq!(invocations[2].command, "invalid");
    assert_eq!(invocations[2].start_unix_timestamp, Some(0));
    assert_eq!(invocations[2].end_unix_timestamp, Some(1));
  }

  #[test]
  fn test_bash_deduplication() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("bash.history");
    let mut file = File::create(&file_path).unwrap();
    let history = b"#1\necho foo\n#1\necho foo\n#2\necho foo\n#2\necho foo\nls\nls\n";
    file.write_all(history).unwrap();
    let invocations = parse_bash_history(&file_path, None, None).unwrap();
    assert_eq!(
      invocations
        .iter()
        .filter(|i| i.command == "echo foo")
        .count(),
      2
    );
    assert_eq!(invocations.iter().filter(|i| i.command == "ls").count(), 1);
  }

  #[test]
  fn test_zsh_deduplication() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("zsh.history");
    let mut file = File::create(&file_path).unwrap();
    let history = b": 100:1;foo\n: 100:1;foo\n: 101:1;foo\n: 101:1;foo\n: 102:1;bar\n: 102:1;bar\n";
    file.write_all(history).unwrap();
    let invocations = parse_zsh_history(&file_path, None, None).unwrap();
    assert_eq!(invocations.iter().filter(|i| i.command == "foo").count(), 2);
    assert_eq!(invocations.iter().filter(|i| i.command == "bar").count(), 1);
  }

  #[test]
  fn test_bash_edge_cases() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("bash.history");
    let mut file = File::create(&file_path).unwrap();
    let history = b"\n   \n#notanumber\n#0\n\0\n#123\n\n";
    file.write_all(history).unwrap();
    let invocations = parse_bash_history(&file_path, None, None).unwrap();
    assert!(invocations.iter().any(|i| i.command == "\u{0}"));
  }

  #[test]
  fn test_zsh_edge_cases() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("zsh.history");
    let mut file = File::create(&file_path).unwrap();
    let history =
      b"\n: notanumber:1;foo\n: 100:bad;bar\n: 100:1;foo\n: 100:1;foo\n: 101:1;baz\n;missingfields\n";
    file.write_all(history).unwrap();
    let invocations = parse_zsh_history(&file_path, None, None);
    assert!(invocations.is_ok());
    let invocations = invocations.unwrap();
    assert_eq!(invocations.iter().filter(|i| i.command == "foo").count(), 1);
    assert_eq!(invocations.iter().filter(|i| i.command == "bar").count(), 0);
    assert_eq!(invocations.iter().filter(|i| i.command == "baz").count(), 1);
  }

  #[test]
  fn test_bash_fuzz() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("bash.history");
    let mut file = File::create(&file_path).unwrap();
    let fuzz = (0..=255u8).collect::<Vec<_>>();
    file.write_all(&fuzz).unwrap();
    let _ = parse_bash_history(&file_path, None, None);
  }

  #[test]
  fn test_zsh_fuzz() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("zsh.history");
    let mut file = File::create(&file_path).unwrap();
    let fuzz = (0..=255u8).collect::<Vec<_>>();
    file.write_all(&fuzz).unwrap();
    let _ = parse_zsh_history(&file_path, None, None);
  }
}
