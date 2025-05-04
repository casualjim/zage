use serde::{Deserialize, Serialize};
use crate::shell_history::Invocation;

/// Context for command prediction, combining directory, host, user, and exit status
#[derive(Serialize, Deserialize, Hash, Eq, PartialEq, Clone, Debug, PartialOrd, Ord)]
pub struct Context {
    /// Current working directory
    pub cwd: String,
    /// Hostname where command ran
    pub hostname: Option<String>,
    /// Username who ran the command
    pub username: Option<String>,
    /// Exit status of the command
    pub exit_status: Option<i64>,
}

impl Context {
    /// Construct a Context from an Invocation
    pub fn from_invocation(inv: &Invocation) -> Self {
        Context {
            cwd: inv.working_directory
                .as_ref()
                .map(|b| String::from_utf8_lossy(b).to_string())
                .unwrap_or_else(|| String::from("")),
            hostname: inv.hostname
                .as_ref()
                .map(|b| String::from_utf8_lossy(b).to_string()),
            username: inv.username
                .as_ref()
                .map(|b| String::from_utf8_lossy(b).to_string()),
            exit_status: inv.exit_status,
        }
    }
}
