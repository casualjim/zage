use super::{Invocation, dedup_invocations, generate_import_session_id, get_hostname};
use crate::Result;
use itertools::Itertools;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Parse zsh history file into a vector of Invocations
/// Handles potential binary/non-UTF8 content by reading as bytes
pub fn parse_history_file(
  path: &Path,
  hostname: Option<String>,
  username: Option<String>,
) -> Result<Vec<Invocation>> {
  let session_id = generate_import_session_id(path);
  let file = File::open(path)?;
  let reader = BufReader::new(file);
  let mut invocations = Vec::new();

  // Track current directory, starting with the CWD where import is run
  let mut current_dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));

  let username_s = username.unwrap_or_else(|| {
    uzers::get_current_username()
      .map(|v| v.to_string_lossy().into_owned())
      .unwrap_or_else(|| "unknown".to_string())
  });
  let hostname_s = hostname.unwrap_or_else(get_hostname);

  for line_result in reader.split(b'\n') {
    let mut line_bytes = match line_result {
      Ok(bytes) => bytes,
      Err(_) => continue,
    };
    if line_bytes.ends_with(&[b'\r']) {
      line_bytes.pop();
    }
    if line_bytes.is_empty() {
      continue;
    }

    let Some((fields, command)) = line_bytes.splitn(2, |&ch| ch == b';').collect_tuple() else {
      continue;
    };
    let Some((_skip, start_time, duration_seconds)) =
      fields.splitn(3, |&ch| ch == b':').collect_tuple()
    else {
      continue;
    };

    let command_s = String::from_utf8_lossy(command).into_owned();

    // Update current directory based on 'cd' commands
    if command_s.starts_with("cd ") {
      let target = command_s[3..].trim();
      if !target.is_empty() {
        let target_path = PathBuf::from(target);
        if target_path.is_absolute() {
          current_dir = target_path;
        } else {
          current_dir = current_dir.join(target_path);
          // Basic normalization (doesn't handle symlinks etc.)
          current_dir = current_dir
            .components()
            .fold(PathBuf::new(), |mut acc, comp| {
              match comp {
                std::path::Component::RootDir => acc.push("/"),
                std::path::Component::CurDir => {}
                std::path::Component::ParentDir => {
                  acc.pop();
                }
                std::path::Component::Normal(name) => acc.push(name),
                std::path::Component::Prefix(_) => { /* ignore windows prefixes */ }
              }
              acc
            });
        }
      }
    }

    let start_unix_timestamp_value = match std::str::from_utf8(start_time) {
      Ok(s) => match s.trim().parse::<i64>() {
        Ok(val) => val,
        Err(_) => continue,
      },
      Err(_) => continue,
    };
    let duration = match std::str::from_utf8(duration_seconds) {
      Ok(s) => match s.trim().parse::<i64>() {
        Ok(val) => val,
        Err(_) => continue,
      },
      Err(_) => continue,
    };
    let start_unix_timestamp = Some(start_unix_timestamp_value);
    let end_unix_timestamp = Some(start_unix_timestamp_value + duration);

    let invocation = Invocation {
      command: command_s.clone(),
      shellname: "zsh".to_string(),
      hostname: Some(hostname_s.clone()),
      username: Some(username_s.clone()),
      start_unix_timestamp,
      end_unix_timestamp,
      session_id,
      // Set the working directory based on tracked state
      working_directory: Some(current_dir.to_string_lossy().into_owned()),
      exit_status: None, // exit_status is not available in zsh history
    };

    invocations.push(invocation);
  }

  Ok(dedup_invocations(invocations))
}
