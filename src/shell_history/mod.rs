use bstr::BString;
use serde::{Deserialize, Serialize};
use std::env;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

mod bash;
mod zsh;

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

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Invocation {
  pub command: BString,
  pub shellname: String,
  pub working_directory: Option<BString>,
  pub hostname: Option<BString>,
  pub username: Option<BString>,
  pub exit_status: Option<i64>,
  pub start_unix_timestamp: Option<i64>,
  pub end_unix_timestamp: Option<i64>,
  pub session_id: i64,
}

impl Invocation {
  fn sameish(&self, other: &Self) -> bool {
    self.command == other.command && self.start_unix_timestamp == other.start_unix_timestamp
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

pub fn get_hostname() -> BString {
  env::var_os("ZAGE_HOSTNAME")
    .unwrap_or_else(|| hostname::get().unwrap_or_default())
    .as_bytes()
    .into()
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
