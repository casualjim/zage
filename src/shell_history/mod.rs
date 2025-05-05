use serde::{Deserialize, Serialize};
use std::os::unix::fs::MetadataExt;
use std::path::Path;

mod bash;
mod zsh;

pub use bash::parse_history_file as parse_bash_history;
pub use zsh::parse_history_file as parse_zsh_history;

use crate::db::DBInvocation;

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
  pub command: String,
  pub shellname: String,
  pub working_directory: Option<String>,
  pub hostname: Option<String>,
  pub username: Option<String>,
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

impl From<DBInvocation> for Invocation {
  fn from(db_inv: DBInvocation) -> Self {
    Self {
      command: db_inv.command,
      shellname: db_inv.shellname,
      working_directory: db_inv.working_directory,
      hostname: db_inv.hostname,
      username: db_inv.username,
      exit_status: db_inv.exit_status,
      start_unix_timestamp: db_inv.start_unix_timestamp,
      end_unix_timestamp: db_inv.end_unix_timestamp,
      session_id: db_inv.session_id,
    }
  }
}

impl From<Invocation> for DBInvocation {
  fn from(inv: Invocation) -> Self {
    Self {
      id: uuid::Uuid::now_v7().to_string(),
      command: inv.command,
      shellname: inv.shellname,
      working_directory: inv.working_directory,
      hostname: inv.hostname,
      username: inv.username,
      exit_status: inv.exit_status,
      start_unix_timestamp: inv.start_unix_timestamp,
      end_unix_timestamp: inv.end_unix_timestamp,
      session_id: inv.session_id,
    }
  }
}

impl From<&Invocation> for DBInvocation {
  fn from(inv: &Invocation) -> Self {
    Self {
      id: uuid::Uuid::now_v7().to_string(),
      command: inv.command.clone(),
      shellname: inv.shellname.clone(),
      working_directory: inv.working_directory.clone(),
      hostname: inv.hostname.clone(),
      username: inv.username.clone(),
      exit_status: inv.exit_status,
      start_unix_timestamp: inv.start_unix_timestamp,
      end_unix_timestamp: inv.end_unix_timestamp,
      session_id: inv.session_id,
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
