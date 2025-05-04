use std::fs::File;
use std::io::Write;
use tempfile::tempdir;
use zage::shell_history;
use bstr::BString;

#[test]
fn test_zsh_history_timestamps() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("zsh.history");
    let mut file = File::create(&file_path).unwrap();

    // Zsh extended history format: ": <start>:<duration>;<command>"
    let history = b": 1620000000:5;echo hello\n: 1620000005:2;ls -la\n: 1620000007:1;invalid\ninvalid_line\n";
    file.write_all(history).unwrap();

    let invocations = shell_history::parse_zsh_history(&file_path, None, None).unwrap();

    // We expect 3 valid commands (invalid_line is skipped)
    assert_eq!(invocations.len(), 3);
    assert_eq!(invocations[0].command, BString::from("echo hello"));
    assert_eq!(invocations[0].start_unix_timestamp, Some(1620000000));
    assert_eq!(invocations[0].end_unix_timestamp, Some(1620000005));
    assert_eq!(invocations[1].command, BString::from("ls -la"));
    assert_eq!(invocations[1].start_unix_timestamp, Some(1620000005));
    assert_eq!(invocations[1].end_unix_timestamp, Some(1620000007));
    assert_eq!(invocations[2].command, BString::from("invalid"));
    assert_eq!(invocations[2].start_unix_timestamp, Some(1620000007));
    assert_eq!(invocations[2].end_unix_timestamp, Some(1620000008));
}
