use std::io::Write;
use tempfile::NamedTempFile;
use zage::Result;
use zage::db::{open_db, import_history, init};
use zage::shell_history::parse_zsh_history;

#[tokio::test]
async fn test_import_history_basic() -> Result<()> {
  let tmp_db = NamedTempFile::new()?;
  let db = open_db(tmp_db.path()).await?;
  init(&db.conn).await?;

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
  import_history(&db.conn, invocations).await?;

  // Expect three invocations (no dedup of same command at different times)
  let mut rows = db
    .conn
    .query("SELECT COUNT(*) FROM shell_history", ())
    .await?;
  let row = rows.next().await?.expect("expected row");
  let count: i64 = row.get(0)?;
  assert_eq!(count, 3);
  Ok(())
}
