use std::{path::Path, time::Duration};

use crate::{Result, shell_history::Invocation};
use rusqlite::{Connection, Transaction, params};

pub fn connect<P: AsRef<Path>>(db_path: P) -> Result<Connection> {
  let conn = Connection::open(db_path)?;
  conn.busy_timeout(Duration::from_millis(500))?;
  conn.pragma_update(None, "journal_mode", "WAL")?;
  conn.pragma_update(None, "temp_store", "MEMORY")?;
  conn.pragma_update(None, "cache_size", "16777216")?;
  Ok(conn)
}

pub fn insert_invocation(tx: &mut Transaction, invocation: &Invocation) -> Result<()> {
  // Convert BString to &[u8] for SQLite parameters
  let working_directory = invocation
    .working_directory
    .as_ref()
    .map(|wd| wd.as_slice())
    .map(String::from_utf8_lossy);
  let hostname = invocation
    .hostname
    .as_ref()
    .map(|hn| hn.as_slice())
    .map(String::from_utf8_lossy);
  let username = invocation
    .username
    .as_ref()
    .map(|un| un.as_slice())
    .map(String::from_utf8_lossy);

  tx.execute(
    "INSERT INTO shell_history (
            id,
            command,
            shellname,
            working_directory,
            hostname,
            username,
            exit_status,
            start_unix_timestamp,
            end_unix_timestamp,
            session_id
        ) VALUES (
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?
        )",
    params![
      uuid::Uuid::now_v7().to_string(),
      String::from_utf8_lossy(invocation.command.as_slice()),
      String::from_utf8_lossy(invocation.shellname.as_bytes()),
      working_directory.unwrap_or_default(),
      hostname.unwrap_or_default(),
      username.unwrap_or_default(),
      &invocation.exit_status.unwrap_or(0),
      &invocation.start_unix_timestamp.unwrap_or(0),
      &invocation.end_unix_timestamp.unwrap_or(0),
      &invocation.session_id.to_string(),
    ],
  )?;
  Ok(())
}

pub fn init<P: AsRef<Path>>(db_path: P) -> Result<()> {
  let mut conn = connect(db_path)?;
  let mut tx = conn.transaction()?;
  init_table(&mut tx)?;
  tx.commit()?;
  Ok(())
}

/// Imports shell history from a sequence of invocations.
pub fn import_history<I>(conn: &mut Connection, invocations: I) -> Result<()>
where
  I: IntoIterator<Item = Invocation>,
{
  let mut tx = conn.transaction()?;
  for invocation in invocations {
    insert_invocation(&mut tx, &invocation)?;
  }
  tx.commit()?;
  Ok(())
}

fn init_table(tx: &mut Transaction) -> Result<()> {
  tx.execute(
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
    [],
  )?;
  tx.execute(
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_shell_history_unique ON shell_history (
            command,
            shellname,
            working_directory,
            hostname,
            username,
            exit_status,
            start_unix_timestamp,
            end_unix_timestamp,
            session_id
        )",
    [],
  )?;
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use bstr::BString;

  #[test]
  fn test_init_table() -> Result<()> {
    let mut conn = connect(":memory:")?;
    let mut tx = conn.transaction()?;
    init_table(&mut tx)?;
    tx.commit()?;

    let exists: i64 = conn.query_row(
      "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='shell_history'",
      [],
      |row| row.get(0),
    )?;
    assert_eq!(exists, 1);
    Ok(())
  }

  #[test]
  fn test_insert_invocation() -> Result<()> {
    let mut conn = connect(":memory:")?;
    let mut tx = conn.transaction()?;
    init_table(&mut tx)?;
    tx.commit()?;

    let invocation = Invocation {
      command: BString::from("ls -la"),
      shellname: "zsh".to_string(),
      working_directory: Some(BString::from("/tmp")),
      hostname: Some(BString::from("host")),
      username: Some(BString::from("user")),
      exit_status: Some(0),
      start_unix_timestamp: Some(123),
      end_unix_timestamp: Some(124),
      session_id: 42,
    };

    let mut tx = conn.transaction()?;
    insert_invocation(&mut tx, &invocation)?;
    tx.commit()?;

    let count: i64 = conn.query_row("SELECT COUNT(*) FROM shell_history", [], |row| row.get(0))?;
    assert_eq!(count, 1);
    Ok(())
  }
}
