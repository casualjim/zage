use serde::{Deserialize, Serialize};
use crate::shell_history::Invocation;
use bstr::ByteSlice;

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
                .map(|b| b.to_str_lossy().to_string())
                .unwrap_or_else(|| String::from("")),
            hostname: inv.hostname
                .as_ref()
                .map(|b| b.to_str_lossy().to_string()),
            username: inv.username
                .as_ref()
                .map(|b| b.to_str_lossy().to_string()),
            exit_status: inv.exit_status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shell_history::Invocation;
    use bstr::BString;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[test]
    fn test_context_from_invocation() {
        let now = SystemTime::now();
        let now_unix = now.duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        let start_unix = (now - Duration::from_secs(10)).duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;

        // Test case 1: Standard invocation
        let inv1 = Invocation {
            command: BString::from("ls -l"),
            shellname: "bash".to_string(),
            exit_status: Some(0),
            start_unix_timestamp: Some(start_unix),
            end_unix_timestamp: Some(now_unix),
            working_directory: Some(BString::from("/home/user")),
            hostname: Some(BString::from("myhost")),
            username: Some(BString::from("user1")),
            session_id: 0,
        };
        let ctx1 = Context::from_invocation(&inv1);
        assert_eq!(ctx1.cwd, "/home/user");
        assert_eq!(ctx1.hostname, Some("myhost".to_string()));
        assert_eq!(ctx1.username, Some("user1".to_string()));
        assert_eq!(ctx1.exit_status, Some(0));

        // Test case 2: Invocation with missing optional fields
        let inv2 = Invocation {
            command: BString::from("cd"),
            shellname: "zsh".to_string(),
            exit_status: None,
            start_unix_timestamp: Some(now_unix),
            end_unix_timestamp: None,
            working_directory: Some(BString::from("/tmp")),
            hostname: None,
            username: None,
            session_id: 1,
        };
        let ctx2 = Context::from_invocation(&inv2);
        assert_eq!(ctx2.cwd, "/tmp");
        assert_eq!(ctx2.hostname, None);
        assert_eq!(ctx2.username, None);
        assert_eq!(ctx2.exit_status, None);

        // Test case 3: Invocation with empty working directory
        let inv3 = Invocation {
            command: BString::from("pwd"),
            shellname: "bash".to_string(),
            exit_status: Some(0),
            start_unix_timestamp: Some(now_unix),
            end_unix_timestamp: None,
            working_directory: None, // Represents empty or unknown CWD
            hostname: Some(BString::from("anotherhost")),
            username: Some(BString::from("user2")),
            session_id: 2,
        };
        let ctx3 = Context::from_invocation(&inv3);
        assert_eq!(ctx3.cwd, ""); // Expect empty string if working_directory is None
        assert_eq!(ctx3.hostname, Some("anotherhost".to_string()));
        assert_eq!(ctx3.username, Some("user2".to_string()));
        assert_eq!(ctx3.exit_status, Some(0));

        // Test case 4: Invocation with non-UTF8 bytes (should be lossily converted)
        let non_utf8_dir = BString::from(vec![0x2f, 0x68, 0x6f, 0x6d, 0x65, 0x2f, 0xf0, 0x9f, 0x91]); // Invalid sequence
        let non_utf8_host = BString::from(vec![0x6d, 0x79, 0x80, 0x68, 0x6f, 0x73, 0x74]); // Invalid byte 0x80
        let non_utf8_user = BString::from(vec![0x75, 0x73, 0xff, 0x65, 0x72]); // Invalid byte 0xff

        let inv4 = Invocation {
            command: BString::from("echo hello"),
            shellname: "bash".to_string(),
            exit_status: Some(0),
            start_unix_timestamp: Some(now_unix),
            end_unix_timestamp: None,
            working_directory: Some(non_utf8_dir),
            hostname: Some(non_utf8_host),
            username: Some(non_utf8_user),
            session_id: 3,
        };
        let ctx4 = Context::from_invocation(&inv4);
        // Expect replacement characters (U+FFFD) where conversion failed
        assert!(ctx4.cwd.contains('\u{fffd}'));
        assert!(ctx4.hostname.unwrap().contains('\u{fffd}'));
        assert!(ctx4.username.unwrap().contains('\u{fffd}'));
        assert_eq!(ctx4.exit_status, Some(0));
    }
}
