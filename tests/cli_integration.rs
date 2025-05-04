//! Integration tests for the Zage CLI binary.
//!
//! These tests run the built CLI as a subprocess and verify end-to-end behavior, including database and model interactions.
//! They use temporary directories and files to avoid polluting the user's environment.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

/// Helper to write a minimal shell history file (zsh format)
fn write_zsh_history(path: &Path) {
  let contents = ": 1610000000:0;echo foo\n: 1610000001:0;cd project\n: 1610000002:0;ls\n: 1610000003:0;cargo build\n: 1610000004:0;ls\n";
  fs::write(path, contents).unwrap();
}

#[test]
fn test_import() {
  let temp_dir = TempDir::new().unwrap();
  let db_dir = temp_dir.path().join("zage_data");
  fs::create_dir_all(&db_dir).unwrap();
  let db_file = db_dir.join("zage.db");
  let hist_file = temp_dir.path().join(".zsh_history");
  write_zsh_history(&hist_file);

  // Import history
  let mut import_cmd = Command::cargo_bin("zage").unwrap();
  import_cmd
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
}

#[test]
fn test_import_and_predict() {
  let temp_dir = TempDir::new().unwrap();
  let db_dir = temp_dir.path().join("zage_data");
  fs::create_dir_all(&db_dir).unwrap();
  let db_file = db_dir.join("zage.db");
  let hist_file = temp_dir.path().join(".zsh_history");
  write_zsh_history(&hist_file);

  // Import history
  let mut import_cmd = Command::cargo_bin("zage").unwrap();
  import_cmd
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
    .arg("--db-path")
    .arg(db_file.to_str().unwrap())
    .arg("train");
  train_cmd
    .assert()
    .success()
    .stdout(predicate::str::contains("Model trained successfully"));

  // Predict
  let mut predict_cmd = Command::cargo_bin("zage").unwrap();
  predict_cmd
    .arg("--db-path")
    .arg(db_file.to_str().unwrap())
    .arg("predict");
  predict_cmd
    .assert()
    .success()
    .stdout(predicate::str::contains("Predicted commands"))
    .stdout(predicate::str::contains("cargo build")); // Check for specific prediction
}

#[test]
fn test_predict_without_import() {
  let temp_dir = TempDir::new().unwrap();
  let db_dir = temp_dir.path().join("zage_data");
  fs::create_dir_all(&db_dir).unwrap();
  let db_file = db_dir.join("zage.db");

  let mut predict_cmd = Command::cargo_bin("zage").unwrap();
  predict_cmd
    .arg("--db-path")
    .arg(db_file.to_str().unwrap())
    .arg("predict");
  predict_cmd
    .assert()
    .failure()
    .stdout(predicate::str::contains("No command history found"));
}
