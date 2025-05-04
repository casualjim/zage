use bstr::BString;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;
use zage::shell_history;

#[test]
fn test_bash_deduplication() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("bash.history");
    let mut file = File::create(&file_path).unwrap();
    // Duplicate and near-duplicate lines
    let history = b"#1\necho foo\n#1\necho foo\n#2\necho foo\n#2\necho foo\nls\nls\n";
    file.write_all(history).unwrap();
    let invocations = shell_history::parse_bash_history(&file_path, None, None).unwrap();
    // Should deduplicate by command and timestamp
    assert_eq!(
        invocations
            .iter()
            .filter(|i| i.command == BString::from("echo foo"))
            .count(),
        2
    );
    assert_eq!(
        invocations
            .iter()
            .filter(|i| i.command == BString::from("ls"))
            .count(),
        1
    );
}

#[test]
fn test_zsh_deduplication() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("zsh.history");
    let mut file = File::create(&file_path).unwrap();
    let history = b": 100:1;foo\n: 100:1;foo\n: 101:1;foo\n: 101:1;foo\n: 102:1;bar\n: 102:1;bar\n";
    file.write_all(history).unwrap();
    let invocations = shell_history::parse_zsh_history(&file_path, None, None).unwrap();
    assert_eq!(
        invocations
            .iter()
            .filter(|i| i.command == BString::from("foo"))
            .count(),
        2
    );
    assert_eq!(
        invocations
            .iter()
            .filter(|i| i.command == BString::from("bar"))
            .count(),
        1
    );
}

#[test]
fn test_bash_edge_cases() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("bash.history");
    let mut file = File::create(&file_path).unwrap();
    let history = b"\n   \n#notanumber\n#0\n\0\n#123\n\n";
    file.write_all(history).unwrap();
    let invocations = shell_history::parse_bash_history(&file_path, None, None).unwrap();
    // Should skip empty, whitespace, and invalid timestamp lines, but keep null byte line
    assert!(
        invocations
            .iter()
            .any(|i| i.command == BString::from(vec![0]))
    );
}

#[test]
fn test_zsh_edge_cases() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("zsh.history");
    let mut file = File::create(&file_path).unwrap();
    // Malformed and partial lines
    let history = b"\n: notanumber:1;foo\n: 100:bad;bar\n: 100:1;foo\n: 100:1;foo\n: 101:1;baz\n;missingfields\n";
    file.write_all(history).unwrap();
    let invocations = shell_history::parse_zsh_history(&file_path, None, None);
    // Should not panic, should parse valid lines, skip malformed
    assert!(invocations.is_ok());
    let invocations = invocations.unwrap();
    assert_eq!(
        invocations
            .iter()
            .filter(|i| i.command == BString::from("foo"))
            .count(),
        1
    );
    assert_eq!(
        invocations
            .iter()
            .filter(|i| i.command == BString::from("bar"))
            .count(),
        0
    );
    assert_eq!(
        invocations
            .iter()
            .filter(|i| i.command == BString::from("baz"))
            .count(),
        1
    );
}

#[test]
fn test_bash_fuzz() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("bash.history");
    let mut file = File::create(&file_path).unwrap();
    // Random bytes
    let fuzz = (0..=255u8).collect::<Vec<_>>();
    file.write_all(&fuzz).unwrap();
    let _ = shell_history::parse_bash_history(&file_path, None, None);
    // Should not panic
}

#[test]
fn test_zsh_fuzz() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("zsh.history");
    let mut file = File::create(&file_path).unwrap();
    let fuzz = (0..=255u8).collect::<Vec<_>>();
    file.write_all(&fuzz).unwrap();
    let _ = shell_history::parse_zsh_history(&file_path, None, None);
    // Should not panic
}
