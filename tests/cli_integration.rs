//! Integration tests for the Zage CLI binary.
//!
//! These tests run the built CLI as a subprocess and verify end-to-end behavior for
//! history import and command recording without touching the user's environment.

use assert_cmd::prelude::*;
use color_eyre::Result;
use predicates::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use tempfile::TempDir;
use zage::db::{init, open_db};

/// Helper to write a minimal shell history file (zsh format)
fn write_zsh_history(path: &Path) {
  let contents = ": 1610000000:0;echo foo\n: 1610000001:0;cd project\n: 1610000002:0;ls\n: 1610000003:0;cargo build\n: 1610000004:0;ls\n";
  fs::write(path, contents).unwrap();
}

fn setup_test_environment() -> Result<(TempDir, PathBuf, PathBuf)> {
  let temp_dir = TempDir::new().unwrap();
  let db_dir = temp_dir.path().join("zage_data");
  fs::create_dir_all(&db_dir).unwrap();
  let db_path = db_dir.join("zage.db");
  let hist_file = temp_dir.path().join(".zsh_history");
  write_zsh_history(&hist_file);

  Ok((temp_dir, db_path, hist_file))
}

#[test]
fn test_import() -> Result<()> {
  let temp_dir = TempDir::new().unwrap();
  let db_dir = temp_dir.path().join("zage_data");
  fs::create_dir_all(&db_dir).unwrap();
  let db_file = db_dir.join("zage.db");
  let hist_file = temp_dir.path().join(".zsh_history");
  let socket_path = temp_dir.path().join("zage.sock");
  write_zsh_history(&hist_file);

  // Import history
  let mut import_cmd = Command::new(assert_cmd::cargo::cargo_bin!("zage"));
  import_cmd
    .env("RUST_LOG", "info")
    .env("ZAGE_SOCKET_PATH", &socket_path)
    .arg("--db-path")
    .arg(db_file.to_str().unwrap())
    .arg("import")
    .arg("--embedded-db")
    .arg("--shell")
    .arg("zsh")
    .arg(hist_file.to_str().unwrap());
  import_cmd
    .assert()
    .success()
    .stderr(predicate::str::contains("Imported history"));

  Ok(())
}

#[test]
fn test_yank_command() -> Result<()> {
  let (temp_dir, db_path, hist_file) = setup_test_environment()?;
  let socket_path = temp_dir.path().join("zage.sock");

  let mut import_cmd = Command::new(assert_cmd::cargo::cargo_bin!("zage"));
  import_cmd
    .env("RUST_LOG", "info")
    .env("ZAGE_SOCKET_PATH", &socket_path)
    .arg("--db-path")
    .arg(&db_path)
    .arg("import")
    .arg("--embedded-db")
    .arg("--no-index")
    .arg("--shell")
    .arg("zsh")
    .arg(hist_file.to_str().unwrap());
  import_cmd
    .assert()
    .success()
    .stderr(predicate::str::contains("Imported history"));

  let mut yank_cmd = Command::new(assert_cmd::cargo::cargo_bin!("zage"));
  yank_cmd
    .env("RUST_LOG", "info")
    .env("ZAGE_SOCKET_PATH", &socket_path)
    .arg("--db-path")
    .arg(&db_path)
    .arg("yank")
    .arg("--embedded-db")
    .arg("--no-sequences")
    .arg("ls");
  yank_cmd
    .assert()
    .success()
    .stderr(predicate::str::contains("Removed"));

  let rt = tokio::runtime::Runtime::new()?;
  let remaining = rt.block_on(async {
    let db = open_db(&db_path).await?;
    init(&db.conn).await?;
    let mut rows = db
      .conn
      .query(
        "SELECT COUNT(*) FROM shell_history WHERE command = ?",
        libsql::params!["ls".to_string()],
      )
      .await?;
    let row = rows.next().await?.expect("expected row");
    Ok::<_, zage::ZageError>(row.get::<i64>(0)?)
  })?;

  assert_eq!(remaining, 0);
  Ok(())
}

#[test]
fn test_record_command() -> Result<()> {
  let (temp_dir, db_path, _) = setup_test_environment()?;
  let socket_path = temp_dir.path().join("zage.sock");
  let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("zage"));

  let cmd_str = "echo 'hello zage'";
  let wd = "/tmp/zage_test_dir";
  let exit_status = 0;
  let start_ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64 - 1;
  let end_ts = start_ts + 1;
  let session_id = 12345;

  // Run the record command
  let output = cmd
    .env("RUST_LOG", "info")
    .env("ZAGE_SOCKET_PATH", &socket_path)
    .arg("--db-path")
    .arg(&db_path)
    .arg("record")
    .arg("--embedded-db")
    .arg("--command")
    .arg(cmd_str)
    .arg("--working-directory")
    .arg(wd)
    .arg("--exit-status")
    .arg(exit_status.to_string())
    .arg("--start-timestamp")
    .arg(start_ts.to_string())
    .arg("--end-timestamp")
    .arg(end_ts.to_string())
    .arg("--session-id")
    .arg(session_id.to_string())
    .output()?;

  assert!(output.status.success(), "zage record failed: {:?}", output);
  // Optional: check stderr/stdout if needed

  // Verify the record in the database
  let rt = tokio::runtime::Runtime::new()?;
  let invocation = rt.block_on(async {
    let db = open_db(&db_path).await?;
    init(&db.conn).await?;
    let mut rows = db
      .conn
      .query(
        "SELECT command, working_directory, exit_status, start_unix_timestamp, end_unix_timestamp, session_id FROM shell_history WHERE command = ?",
        libsql::params![cmd_str.to_string()],
      )
      .await?;
    let row = rows.next().await?.expect("expected row");
    let record = (
      row.get::<String>(0)?,
      row.get::<String>(1)?,
      row.get::<i64>(2)?,
      row.get::<i64>(3)?,
      row.get::<i64>(4)?,
      row.get::<i64>(5)?,
    );
    Ok::<_, zage::ZageError>(record)
  })?;

  assert_eq!(invocation.0, cmd_str);
  assert_eq!(invocation.1, wd);
  assert_eq!(invocation.2, exit_status);
  assert_eq!(invocation.3, start_ts);
  assert_eq!(invocation.4, end_ts);
  assert_eq!(invocation.5, session_id);

  Ok(())
}
