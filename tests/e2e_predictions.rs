use assert_cmd::Command;
use predicates::str::contains;
use tempfile::TempDir;

#[test]
fn test_e2e_ngram_prediction() -> Result<(), Box<dyn std::error::Error>> {
    // Setup temporary DB
    let tmp = TempDir::new()?;
    let db = tmp.path().join("zage.db");
    let db_str = db.to_str().unwrap();

    // Record a simple sequence: foo -> bar -> baz -> bar
    for (i, cmd_str) in ["foo", "bar", "baz", "bar"].iter().enumerate() {
        Command::cargo_bin("zage")?
            .args(&[
                "--db-path",
                db_str,
                "record",
                "--command",
                cmd_str,
                "--working-directory",
                "/tmp",
                "--exit-status",
                "0",
                "--start-timestamp",
                &(1 + i as i64).to_string(),
                "--end-timestamp",
                &(2 + i as i64).to_string(),
                "--session-id",
                "1",
            ])
            .assert()
            .success();
    }

    // Train NGram model
    Command::cargo_bin("zage")?
        .args(&["--db-path", db_str, "train", "--model-type", "ngram", "--n", "2"])
        .assert()
        .success();

    // Predict next command
    Command::cargo_bin("zage")?
        .args(&["--db-path", db_str, "predict", "--model-type", "ngram", "--count", "1"])
        .assert()
        .success()
        .stdout(contains("1. baz"));

    Ok(())
}

#[test]
fn test_e2e_markov_prediction() -> Result<(), Box<dyn std::error::Error>> {
    // Setup temporary DB
    let tmp = TempDir::new()?;
    let db = tmp.path().join("zage.db");
    let db_str = db.to_str().unwrap();

    // Record a simple sequence: foo -> bar -> baz
    for (i, cmd_str) in ["foo", "bar", "baz"].iter().enumerate() {
        Command::cargo_bin("zage")?
            .args(&[
                "--db-path",
                db_str,
                "record",
                "--command",
                cmd_str,
                "--working-directory",
                "/tmp",
                "--exit-status",
                "0",
                "--start-timestamp",
                &(1 + i as i64).to_string(),
                "--end-timestamp",
                &(2 + i as i64).to_string(),
                "--session-id",
                "1",
            ])
            .assert()
            .success();
    }

    // Train Markov model
    Command::cargo_bin("zage")?
        .args(&["--db-path", db_str, "train", "--model-type", "markov"])
        .assert()
        .success();

    // Predict next command
    Command::cargo_bin("zage")?
        .args(&["--db-path", db_str, "predict", "--model-type", "markov", "--count", "1"])
        .assert()
        .success()
        .stdout(contains("1. baz"));

    Ok(())
}
