//! Integration tests for the Zage CLI binary.
//!
//! These tests run the built CLI as a subprocess and verify end-to-end behavior, including database and model interactions.
//! They use temporary directories and files to avoid polluting the user's environment.

use assert_cmd::prelude::*;
use color_eyre::Result;
use predicates::prelude::*;
use rusqlite::Connection;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};
use tempfile::TempDir;

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
  write_zsh_history(&hist_file);

  // Import history
  let mut import_cmd = Command::cargo_bin("zage").unwrap();
  import_cmd
    .env("RUST_LOG", "info")
    .arg("--db-path")
    .arg(db_file.to_str().unwrap())
    .arg("import")
    .arg("--shell")
    .arg("zsh")
    .arg(hist_file.to_str().unwrap());
  import_cmd
    .assert()
    .success()
    .stdout(predicate::str::contains("Imported history"));

  Ok(())
}

#[test]
fn test_import_and_predict() -> Result<()> {
  let temp_dir = TempDir::new().unwrap();
  let db_dir = temp_dir.path().join("zage_data");
  fs::create_dir_all(&db_dir).unwrap();
  let db_file = db_dir.join("zage.db");
  let hist_file = temp_dir.path().join(".zsh_history");
  write_zsh_history(&hist_file);

  // Import history
  let mut import_cmd = Command::cargo_bin("zage").unwrap();
  import_cmd
    .env("RUST_LOG", "info")
    .arg("--db-path")
    .arg(db_file.to_str().unwrap())
    .arg("import")
    .arg("--shell")
    .arg("zsh")
    .arg(hist_file.to_str().unwrap());
  import_cmd
    .assert()
    .success()
    .stdout(predicate::str::contains("Imported history"));

  // Train model using the imported history (n=2 default)
  let mut train_cmd = Command::cargo_bin("zage").unwrap();
  train_cmd
    .env("RUST_LOG", "info")
    .arg("--db-path")
    .arg(db_file.to_str().unwrap())
    .arg("train")
    .arg("--model-type")
    .arg("ngram");
  train_cmd
    .assert()
    .success()
    .stdout(predicate::str::contains("Model trained successfully"));

  // Predict
  let mut predict_cmd = Command::cargo_bin("zage").unwrap();
  predict_cmd
    .env("RUST_LOG", "info")
    .arg("--db-path")
    .arg(db_file.to_str().unwrap())
    .arg("predict")
    .arg("--model-type")
    .arg("ngram");
  predict_cmd
    .assert()
    .success()
    .stdout(predicate::str::contains("Predicted commands"))
    .stdout(predicate::str::contains("cargo build")); // Check for specific prediction

  Ok(())
}

#[test]
fn test_predict_without_import() -> Result<()> {
  let temp_dir = TempDir::new().unwrap();
  let db_dir = temp_dir.path().join("zage_data");
  fs::create_dir_all(&db_dir).unwrap();
  let db_file = db_dir.join("zage.db");

  let mut predict_cmd = Command::cargo_bin("zage").unwrap();
  predict_cmd
    .env("RUST_LOG", "info")
    .arg("--db-path")
    .arg(db_file.to_str().unwrap())
    .arg("predict");
  predict_cmd
    .env("RUST_LOG", "info")
    .assert()
    .failure()
    .stdout(predicate::str::contains("No command history found"));

  Ok(())
}

#[test]
fn test_record_command() -> Result<()> {
  let (_temp_dir, db_path, _) = setup_test_environment()?;
  let mut cmd = Command::cargo_bin("zage").unwrap();

  let cmd_str = "echo 'hello zage'";
  let wd = "/tmp/zage_test_dir";
  let exit_status = 0;
  let start_ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs() as i64 - 1;
  let end_ts = start_ts + 1;
  let session_id = 12345;

  // Run the record command
  let output = cmd
    .env("RUST_LOG", "info")
    .arg("--db-path")
    .arg(&db_path)
    .arg("record")
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
  let conn = Connection::open(&db_path)?;
  let mut stmt = conn.prepare(
    "SELECT command, working_directory, exit_status, start_unix_timestamp, end_unix_timestamp, session_id FROM shell_history WHERE command = ?1",
  )?;
  let invocation = stmt.query_row([cmd_str], |row| {
    Ok((
      row.get::<_, String>(0)?,
      row.get::<_, String>(1)?,
      row.get::<_, i64>(2)?,
      row.get::<_, i64>(3)?,
      row.get::<_, i64>(4)?,
      row.get::<_, i64>(5)?,
    ))
  })?;

  assert_eq!(invocation.0, cmd_str);
  assert_eq!(invocation.1, wd);
  assert_eq!(invocation.2, exit_status);
  assert_eq!(invocation.3, start_ts);
  assert_eq!(invocation.4, end_ts);
  assert_eq!(invocation.5, session_id);

  Ok(())
}
