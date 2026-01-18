use super::{Invocation, dedup_invocations, generate_import_session_id};
use crate::Result;
use itertools::Itertools;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

fn normalize_path(mut path: PathBuf) -> Option<PathBuf> {
  path = path.components().fold(PathBuf::new(), |mut acc, comp| {
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

  if path.as_os_str().is_empty() {
    None
  } else {
    Some(path)
  }
}

fn cd_target(command: &str) -> Option<&str> {
  let trimmed = command.trim();
  let mut parts = trimmed.split_whitespace();
  let head = parts.next()?;
  if head != "cd" {
    return None;
  }

  match parts.next() {
    Some("-") => None,
    Some(target) => Some(target),
    None => Some("~"),
  }
}

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

  // Track current directory only when the history explicitly reveals it (via `cd` commands).
  // Importing history from another machine should not "inherit" the importer's cwd.
  let mut current_dir: Option<PathBuf> = None;

  // For imports we should not "invent" identity info from the importer machine.
  // If the caller wants hostname/username, they can pass them explicitly.
  let username_s = username;
  let hostname_s = hostname;

  for line_result in reader.split(b'\n') {
    let mut line_bytes = match line_result {
      Ok(bytes) => bytes,
      Err(_) => continue,
    };
    if line_bytes.ends_with(b"\r") {
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
    if let Some(target) = cd_target(&command_s) {
      let target = target.trim();
      if !target.is_empty() {
        let target_path = PathBuf::from(target);
        if target_path.is_absolute() || target.starts_with('~') {
          current_dir = normalize_path(target_path);
        } else if let Some(base) = current_dir.as_ref() {
          let next = base.join(target_path);
          current_dir = normalize_path(next);
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
      expanded_command: String::new(),
      command: command_s.clone(),
      shellname: "zsh".to_string(),
      hostname: hostname_s.clone(),
      username: username_s.clone(),
      workspace: None,
      start_unix_timestamp,
      end_unix_timestamp,
      session_id,
      // Set the working directory based on tracked state
      working_directory: current_dir
        .as_ref()
        .map(|d| d.to_string_lossy().into_owned()),
      exit_status: None, // exit_status is not available in zsh history
    };

    invocations.push(invocation);
  }

  Ok(dedup_invocations(invocations))
}
