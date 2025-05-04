use super::{Invocation, dedup_invocations, generate_import_session_id, get_hostname};
use crate::Result;
use bstr::BString;
use itertools::Itertools;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Parse zsh history file into a vector of Invocations
/// Handles potential binary/non-UTF8 content by reading as bytes
pub fn parse_history_file(
  path: &Path,
  hostname: Option<BString>,
  username: Option<BString>,
) -> Result<Vec<Invocation>> {
  let session_id = generate_import_session_id(path);
  let file = File::open(path)?;
  let reader = BufReader::new(file);
  let mut invocations = Vec::new();

  let username = username
    .or_else(|| {
      users::get_current_username()
        .as_ref()
        .map(|v| BString::from(v.as_encoded_bytes()))
    })
    .unwrap_or_else(|| BString::from("unknown"));
  let hostname = hostname.unwrap_or_else(get_hostname);

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

    let command = BString::from(command);
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
      command,
      shellname: "zsh".to_string(),
      hostname: Some(hostname.clone()),
      username: Some(username.clone()),
      start_unix_timestamp,
      end_unix_timestamp,
      session_id,
      ..Default::default()
    };

    invocations.push(invocation);
  }

  Ok(dedup_invocations(invocations))
}
