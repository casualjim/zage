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
  tx.execute(
    "CREATE TABLE IF NOT EXISTS models (
            model_type TEXT PRIMARY KEY,
            model_data BLOB,
            created_at INTEGER,
            updated_at INTEGER
        )",
    [],
  )?;
  Ok(())
}

/// Get recent command invocations from the database
pub fn get_recent_invocations(conn: &mut Connection, limit: usize) -> Result<Vec<Invocation>> {
  let mut stmt = conn.prepare(
    "SELECT
            command,
            shellname,
            working_directory,
            hostname,
            username,
            exit_status,
            start_unix_timestamp,
            end_unix_timestamp,
            session_id
        FROM shell_history
        ORDER BY start_unix_timestamp DESC
        LIMIT ?",
  )?;

  let invocations = stmt
    .query_map([limit as i64], |row| {
      // Get raw bytes from the database and convert to appropriate types
      let command_bytes: Vec<u8> = row.get(0)?;
      let shellname_str: String = row.get(1)?;
      let working_dir: Option<Vec<u8>> = row.get(2)?;
      let hostname: Option<Vec<u8>> = row.get(3)?;
      let username: Option<Vec<u8>> = row.get(4)?;

      Ok(Invocation {
        command: bstr::BString::from(command_bytes),
        shellname: shellname_str,
        working_directory: working_dir.map(bstr::BString::from),
        hostname: hostname.map(bstr::BString::from),
        username: username.map(bstr::BString::from),
        exit_status: row.get(5)?,
        start_unix_timestamp: row.get(6)?,
        end_unix_timestamp: row.get(7)?,
        session_id: row.get(8)?,
      })
    })?
    .collect::<std::result::Result<Vec<_>, _>>()?;

  // Reverse the order to get chronological order for training
  let mut chronological = invocations;
  chronological.reverse();

  Ok(chronological)
}

/// Save a model to the database
pub fn save_model(conn: &mut Connection, model_type: &str, model_data: &[u8]) -> Result<()> {
  // Create the models table if it doesn't exist

  let now = std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap()
    .as_secs() as i64;

  // Check if the model already exists
  let mut stmt = conn.prepare("SELECT 1 FROM models WHERE model_type = ?")?;
  let exists = stmt.exists([model_type])?;

  if exists {
    // Update existing model
    conn.execute(
      "UPDATE models SET model_data = ?, updated_at = ? WHERE model_type = ?",
      params![model_data, now, model_type],
    )?;
  } else {
    // Insert new model
    conn.execute(
      "INSERT INTO models (model_type, model_data, created_at, updated_at) VALUES (?, ?, ?, ?)",
      params![model_type, model_data, now, now],
    )?;
  }

  Ok(())
}

/// Load a model from the database
pub fn load_model(conn: &mut Connection, model_type: &str) -> Result<Option<Vec<u8>>> {
  // Try to load the model
  let mut stmt = conn.prepare("SELECT model_data FROM models WHERE model_type = ?")?;
  let model_data = match stmt.query_row([model_type], |row| row.get::<_, Vec<u8>>(0)) {
    Ok(data) => Some(data),
    Err(rusqlite::Error::QueryReturnedNoRows) => None,
    Err(e) => return Err(e.into()),
  };

  Ok(model_data)
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
