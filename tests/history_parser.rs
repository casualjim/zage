use std::path::Path;
use tempfile::tempdir;
use zage::{AppConfig, Result, db, shell_history};

#[test]
fn test_history_import() -> Result<()> {
    // Create a temporary directory for the test database
    let temp_dir = tempdir()?;
    let db_path = temp_dir.path().join("test.db");
    let db_path_str = db_path.to_str().unwrap();

    // Initialize the database
    let config = AppConfig {
        db_path: db_path_str.to_string(),
    };
    db::init(&config.db_path)?;

    // Import bash history
    let bash_history_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("bash.history");

    let bash_invocations = shell_history::parse_bash_history(&bash_history_path, None, None)?;

    // Import zsh history
    let zsh_history_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("zsh.history");

    let zsh_invocations = shell_history::parse_zsh_history(&zsh_history_path, None, None)?;

    // Insert all invocations into database
    let mut conn = db::connect(&config.db_path)?;
    let mut tx = conn.transaction()?;

    for invocation in bash_invocations {
        db::insert_invocation(&mut tx, &invocation)?;
    }

    for invocation in zsh_invocations {
        db::insert_invocation(&mut tx, &invocation)?;
    }

    tx.commit()?;

    // Verify data was inserted correctly
    let count = count_history_entries(&conn)?;
    assert!(count > 0, "No history entries were imported");

    Ok(())
}

/// Count the number of entries in the shell_history table
fn count_history_entries(conn: &rusqlite::Connection) -> Result<usize> {
    let count: usize =
        conn.query_row("SELECT COUNT(*) FROM shell_history", [], |row| row.get(0))?;
    Ok(count)
}
