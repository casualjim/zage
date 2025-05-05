use std::{path::Path, time::Duration};

use crate::{Result, shell_history::Invocation};
use exemplar::Model;
use rusqlite::{Connection, Transaction, params};

#[derive(Debug, Model)]
#[table("shell_history")]
pub struct DBInvocation {
  #[column("id")]
  pub id: String,
  pub command: String,
  pub shellname: String,
  #[column("working_directory")]
  pub working_directory: Option<String>,
  #[column("hostname")]
  pub hostname: Option<String>,
  #[column("username")]
  pub username: Option<String>,
  #[column("exit_status")]
  pub exit_status: Option<i64>,
  #[column("start_unix_timestamp")]
  pub start_unix_timestamp: Option<i64>,
  #[column("end_unix_timestamp")]
  pub end_unix_timestamp: Option<i64>,
  #[column("session_id")]
  pub session_id: i64,
}

pub fn connect<P: AsRef<Path>>(db_path: P) -> Result<Connection> {
  let conn = Connection::open(db_path)?;
  conn.busy_timeout(Duration::from_millis(500))?;
  conn.pragma_update(None, "journal_mode", "WAL")?;
  conn.pragma_update(None, "temp_store", "MEMORY")?;
  conn.pragma_update(None, "cache_size", "16777216")?;
  Ok(conn)
}

pub fn insert_invocation(tx: &mut Transaction, invocation: &Invocation) -> Result<()> {
  let db_inv: DBInvocation = invocation.into();
  db_inv.insert(tx)?;
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

pub fn init_table(tx: &mut Transaction) -> Result<()> {
  tx.execute_batch(&include_str!("db/schema-v0.sql"))?;

  Ok(())
}

/// Get recent command invocations from the database
pub fn get_recent_invocations(conn: &mut Connection, limit: usize) -> Result<Vec<Invocation>> {
  let mut stmt = conn.prepare(include_str!("db/get-recent-invocations.sql"))?;
  let db_invs = stmt
    .query_and_then(
      rusqlite::named_params! {":limit": limit as i64},
      DBInvocation::from_row,
    )?
    .collect::<rusqlite::Result<Vec<DBInvocation>>>()?;

  let mut invs = db_invs
    .into_iter()
    .map(|db_inv| Invocation {
      command: db_inv.command,
      shellname: db_inv.shellname,
      working_directory: db_inv.working_directory,
      hostname: db_inv.hostname,
      username: db_inv.username,
      exit_status: db_inv.exit_status,
      start_unix_timestamp: db_inv.start_unix_timestamp,
      end_unix_timestamp: db_inv.end_unix_timestamp,
      session_id: db_inv.session_id,
    })
    .collect::<Vec<_>>();
  invs.reverse();
  Ok(invs)
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

/// Clears all entries from sequence_scores table.
pub fn clear_sequence_scores_table(tx: &mut Transaction) -> Result<()> {
  tx.execute("DELETE FROM sequence_scores", [])?;
  Ok(())
}

/// Analyzes bigram and trigram sequences and stores scores in sequence_scores.
pub fn analyze_sequences(
  tx: &mut Transaction,
  min_support: usize,
  min_confidence: f64,
  min_lift: f64,
) -> Result<()> {
  // Bigram analysis
  let sql_bigram = include_str!("db/analyze-bigram.sql");

  tx.execute(
    sql_bigram,
    rusqlite::named_params! {
      ":min_support": min_support,
      ":min_confidence": min_confidence,
      ":min_lift": min_lift
    },
  )?;

  // Trigram analysis
  let sql_trigram = include_str!("db/analyze-trigram.sql");

  tx.execute(
    sql_trigram,
    rusqlite::named_params! {
      ":min_support": min_support,
      ":min_confidence": min_confidence,
      ":min_lift": min_lift
    },
  )?;

  Ok(())
}

/// Represents a raw sequence score from the database
#[derive(Debug, Clone, Model)]
#[table("sequence_scores")]
pub struct RawSequenceScore {
  #[column("sequence")]
  pub sequence_json: String,
  pub support: usize,
  pub confidence: f64,
  pub lift: f64,
  #[column("context")]
  pub context_json: Option<String>,
}

/// Retrieves top scored sequences by lift as a vector.
pub fn get_sequence_scores(conn: &mut Connection, limit: usize) -> Result<Vec<RawSequenceScore>> {
  let mut stmt = conn.prepare(
        "SELECT sequence, support, confidence, lift, context FROM sequence_scores ORDER BY lift DESC LIMIT :limit",
    )?;

  let seqs = stmt
    .query_map(
      rusqlite::named_params! {":limit": limit as i64},
      RawSequenceScore::from_row,
    )?
    .collect::<rusqlite::Result<Vec<_>>>()?;
  Ok(seqs)
}

#[cfg(test)]
mod tests {
  use super::*;

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
      command: "ls -la".to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/tmp".to_string()),
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
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
