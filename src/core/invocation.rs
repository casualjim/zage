use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct Invocation {
  pub command: String,
  pub expanded_command: String,
  pub shellname: String,
  pub working_directory: Option<String>,
  pub workspace: Option<crate::workspace::WorkspaceInfo>,
  pub hostname: Option<String>,
  pub username: Option<String>,
  pub exit_status: Option<i64>,
  pub start_unix_timestamp: Option<i64>,
  pub end_unix_timestamp: Option<i64>,
  pub session_id: i64,
}

impl Invocation {
  pub(crate) fn sameish(&self, other: &Self) -> bool {
    self.command == other.command && self.start_unix_timestamp == other.start_unix_timestamp
  }
}
