use bstr::BString;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;
use zage::shell_history;

#[test]
fn test_bash_history_timestamps() {
  // Create a temp directory and file
  let dir = tempdir().unwrap();
  let file_path = dir.path().join("bash.history");
  let mut file = File::create(&file_path).unwrap();

  // Write a bash history file with timestamps
  let history = b"#1620000000\necho hello\n#1620000005\nls -la\n#0\ninvalid\n";
  file.write_all(history).unwrap();

  // Parse the file
  let invocations = shell_history::parse_bash_history(&file_path, None, None).unwrap();

  // We expect 3 commands: echo hello, ls -la, invalid
  assert_eq!(invocations.len(), 3);
  assert_eq!(invocations[0].command, BString::from("echo hello"));
  assert_eq!(invocations[0].start_unix_timestamp, Some(1620000000));
  assert_eq!(invocations[1].command, BString::from("ls -la"));
  assert_eq!(invocations[1].start_unix_timestamp, Some(1620000005));
  assert_eq!(invocations[2].command, BString::from("invalid"));
  // The last timestamp is 0, which should not be set (should remain as previous or None)
  assert_eq!(invocations[2].start_unix_timestamp, Some(1620000005));
}
