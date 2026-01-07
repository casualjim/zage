use std::path::Path;
use tempfile::tempdir;
use zage::{Result, db, shell_history};

#[tokio::test]
async fn test_history_import() -> Result<()> {
  let temp_dir = tempdir()?;
  let db_path = temp_dir.path().join("test.db");

  let db = db::open_db(&db_path).await?;
  db::init(&db.conn).await?;

  let bash_history_path = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("data")
    .join("bash.history");
  let bash_invocations = shell_history::parse_bash_history(&bash_history_path, None, None)?;

  let zsh_history_path = Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("tests")
    .join("data")
    .join("zsh.history");
  let zsh_invocations = shell_history::parse_zsh_history(&zsh_history_path, None, None)?;

  for invocation in bash_invocations {
    let inserted = db::insert_invocation(&db.conn, &invocation).await?;
    assert!(inserted);
  }

  for invocation in zsh_invocations {
    let inserted = db::insert_invocation(&db.conn, &invocation).await?;
    assert!(inserted);
  }

  let count = count_history_entries(&db.conn).await?;
  assert!(count > 0, "No history entries were imported");

  Ok(())
}

async fn count_history_entries(conn: &libsql::Connection) -> Result<usize> {
  let mut rows = conn.query("SELECT COUNT(*) FROM shell_history", ()).await?;
  let row = rows.next().await?.expect("expected row");
  let count: i64 = row.get(0)?;
  Ok(count as usize)
}
