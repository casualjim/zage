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
  let working_directory = invocation.working_directory.as_deref().unwrap_or_default();
  let hostname = invocation.hostname.as_deref().unwrap_or_default();
  let username = invocation.username.as_deref().unwrap_or_default();

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
      &invocation.command,
      &invocation.shellname,
      working_directory,
      hostname,
      username,
      &invocation.exit_status.unwrap_or(0),
      &invocation.start_unix_timestamp.unwrap_or(0),
      &invocation.end_unix_timestamp.unwrap_or(0),
      &invocation.session_id,
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

pub fn init_table(tx: &mut Transaction) -> Result<()> {
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
  tx.execute(
    "CREATE TABLE IF NOT EXISTS sequence_scores (
       sequence   TEXT PRIMARY KEY,
       support    INTEGER NOT NULL,
       confidence REAL    NOT NULL,
       lift       REAL    NOT NULL,
       context    TEXT    NULL
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
      // Use get for TEXT columns as String, then convert to BString
      let command_str: String = row.get(0)?;
      let shellname_str: String = row.get(1)?;
      let working_dir: Option<String> = row.get(2)?;
      let hostname: Option<String> = row.get(3)?;
      let username: Option<String> = row.get(4)?;

      Ok(Invocation {
        command: command_str,
        shellname: shellname_str,
        working_directory: working_dir,
        hostname,
        username,
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
  let sql_bigram = format!(
    r#"
  WITH bigrams AS (
    SELECT LAG(command) OVER (ORDER BY rowid) AS c1,
           command AS c2
    FROM shell_history
  ), counts AS (
    SELECT c1, c2, COUNT(*) AS support
    FROM bigrams
    WHERE c1 IS NOT NULL
    GROUP BY c1, c2
  ), prefix AS (
    SELECT c1, COUNT(*) AS sp
    FROM bigrams
    WHERE c1 IS NOT NULL
    GROUP BY c1
  ), suffix AS (
    SELECT c2, COUNT(*) AS ss
    FROM bigrams
    GROUP BY c2
  ), total AS (
    SELECT COUNT(*) AS tot FROM shell_history
  )
  INSERT OR REPLACE INTO sequence_scores(sequence, support, confidence, lift, context)
  SELECT
    json_array(counts.c1, counts.c2),
    counts.support,
    counts.support * 1.0 / prefix.sp,
    (counts.support * 1.0 / prefix.sp) / (suffix.ss * 1.0 / total.tot),
    json_object(
      'working_directory', MAX(h1.working_directory),
      'hostname', MAX(h1.hostname),
      'username', MAX(h1.username),
      'exit_status', json_array(MAX(h1.exit_status), MAX(h2.exit_status)),
      'session_id', MAX(h1.session_id),
      'time_info', json_object(
        'start_time', MIN(h1.start_unix_timestamp),
        'end_time', MAX(h2.end_unix_timestamp)
      )
    )
  FROM counts
  JOIN prefix ON counts.c1 = prefix.c1
  JOIN suffix ON counts.c2 = suffix.c2
  CROSS JOIN total
  JOIN shell_history h1 ON counts.c1 = h1.command
  JOIN shell_history h2 ON counts.c2 = h2.command
  WHERE
    counts.support    >= {min_support}
    AND (counts.support * 1.0 / prefix.sp)             >= {min_confidence}
    AND ((counts.support * 1.0 / prefix.sp) / (suffix.ss * 1.0 / total.tot)) >= {min_lift}
  GROUP BY counts.c1, counts.c2;
  "#,
    min_support = min_support,
    min_confidence = min_confidence,
    min_lift = min_lift
  );
  tx.execute_batch(&sql_bigram)?;

  // Trigram analysis
  let sql_trigram = format!(
    r#"
  WITH trigrams AS (
    SELECT LAG(command,2) OVER (ORDER BY rowid) AS c1,
           LAG(command,1) OVER (ORDER BY rowid) AS c2,
           command AS c3
    FROM shell_history
  ), counts AS (
    SELECT c1, c2, c3, COUNT(*) AS support
    FROM trigrams
    WHERE c1 IS NOT NULL
    GROUP BY c1, c2, c3
  ), prefix AS (
    SELECT c1, c2, COUNT(*) AS sp
    FROM trigrams
    WHERE c1 IS NOT NULL
    GROUP BY c1, c2
  ), suffix AS (
    SELECT c3, COUNT(*) AS ss
    FROM trigrams
    GROUP BY c3
  ), total AS (
    SELECT COUNT(*) AS tot FROM shell_history
  )
  INSERT OR REPLACE INTO sequence_scores(sequence, support, confidence, lift, context)
  SELECT
    json_array(counts.c1, counts.c2, counts.c3),
    counts.support,
    counts.support * 1.0 / prefix.sp,
    (counts.support * 1.0 / prefix.sp) / (suffix.ss * 1.0 / total.tot),
    json_object(
      'working_directory', MAX(h1.working_directory),
      'hostname', MAX(h1.hostname),
      'username', MAX(h1.username),
      'exit_status', json_array(MAX(h1.exit_status), MAX(h2.exit_status), MAX(h3.exit_status)),
      'session_id', MAX(h1.session_id),
      'time_info', json_object(
        'start_time', MIN(h1.start_unix_timestamp),
        'end_time', MAX(h3.end_unix_timestamp)
      )
    )
  FROM counts
  JOIN prefix ON counts.c1 = prefix.c1 AND counts.c2 = prefix.c2
  JOIN suffix ON counts.c3 = suffix.c3
  CROSS JOIN total
  JOIN shell_history h1 ON counts.c1 = h1.command
  JOIN shell_history h2 ON counts.c2 = h2.command
  JOIN shell_history h3 ON counts.c3 = h3.command
  WHERE
    counts.support    >= {min_support}
    AND (counts.support * 1.0 / prefix.sp)             >= {min_confidence}
    AND ((counts.support * 1.0 / prefix.sp) / (suffix.ss * 1.0 / total.tot)) >= {min_lift}
  GROUP BY counts.c1, counts.c2, counts.c3;
  "#,
    min_support = min_support,
    min_confidence = min_confidence,
    min_lift = min_lift
  );
  tx.execute_batch(&sql_trigram)?;
  Ok(())
}

/// Represents a raw sequence score from the database
#[derive(Debug, Clone)]
pub struct RawSequenceScore {
  pub sequence_json: String,
  pub support: usize,
  pub confidence: f64,
  pub lift: f64,
  pub context_json: Option<String>,
}

/// Retrieves top scored sequences by lift.
pub fn get_sequence_scores(conn: &mut Connection, limit: usize) -> Result<Vec<RawSequenceScore>> {
  let mut stmt = conn.prepare(
    "SELECT sequence, support, confidence, lift, context
     FROM sequence_scores
     ORDER BY lift DESC
     LIMIT ?",
  )?;
  let rows = stmt.query_map([limit as i64], |row| {
    let sequence_json: String = row.get(0)?;

    Ok(RawSequenceScore {
      sequence_json,
      support: row.get(1)?,
      confidence: row.get(2)?,
      lift: row.get(3)?,
      context_json: row.get(4)?,
    })
  })?;
  let mut results = Vec::new();
  for res in rows {
    results.push(res?);
  }
  Ok(results)
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
