use rusqlite::params;
use std::io::Write;
use tempfile::NamedTempFile;
use zage::Result;
use zage::db::{connect, import_history};
use zage::shell_history::parse_zsh_history;

#[test]
fn test_import_history_basic() -> Result<()> {
  // Setup in-memory DB and create table
  let mut conn = connect(":memory:")?;
  conn.execute(
    "CREATE TABLE IF NOT EXISTS shell_history (
            id TEXT PRIMARY KEY,
            command TEXT NOT NULL,
            shellname TEXT NOT NULL,
            working_directory TEXT,
            hostname TEXT,
            username TEXT,
            exit_status INTEGER,
            start_unix_timestamp INTEGER,
            end_unix_timestamp INTEGER,
            session_id INTEGER
        )",
    params![],
  )?;

  // Write sample zsh history
  let mut tmp = NamedTempFile::new()?;
  let content = ":1610000000:2;echo hello
:1610000002:3;ls -la
:1610000005:1;echo hello
";
  tmp.write_all(content.as_bytes())?;
  tmp.flush()?;

  // Import history
  let invocations = parse_zsh_history(tmp.path(), None, None)?;
  import_history(&mut conn, invocations)?;

  // Expect three invocations (no dedup of same command at different times)
  let count: i64 = conn.query_row("SELECT COUNT(*) FROM shell_history", params![], |row| {
    row.get(0)
  })?;
  assert_eq!(count, 3);
  Ok(())
}
