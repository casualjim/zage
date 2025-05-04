use super::{Invocation, dedup_invocations, generate_import_session_id, get_hostname};
use crate::Result;
use bstr::BString;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use users;

/// Parse bash history file into a vector of Invocations
/// Handles potential binary/non-UTF8 content by reading line by line
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

  let mut last_ts = None;

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

    if line_bytes[0] == b'#' {
      if let Ok(ts) = str::parse::<i64>(std::str::from_utf8(&line_bytes[1..]).unwrap_or("0")) {
        if ts > 0 {
          last_ts = Some(ts);
        }
        continue;
      }
    }
    let command = BString::from(line_bytes);
    let invocation = Invocation {
      command,
      shellname: "bash".to_string(),
      hostname: Some(hostname.clone()),
      username: Some(username.clone()),
      start_unix_timestamp: last_ts,
      session_id,
      ..Default::default()
    };
    invocations.push(invocation);
  }
  Ok(dedup_invocations(invocations))
}
