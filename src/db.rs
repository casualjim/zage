use libsql::{
  Builder, Cipher, Connection, Database, EncryptionConfig, EncryptionContext, EncryptionKey,
};
use std::collections::{BTreeSet, HashSet};
use std::fs;
use std::path::Path;

use crate::config::{DbConfig, DbKind};
use crate::sequence::{SequenceConfig, normalize_sequence_command};
use crate::tokenize::{
  extract_command_parts, extract_command_stats_parts, normalize_command_whitespace,
  normalize_token, tokenize_index,
};
use crate::{Result, shell_history::Invocation};
use serde_json;
use tracing::info;

pub struct Db {
  pub db: Database,
  pub conn: Connection,
}

async fn apply_pragma(conn: &Connection, sql: &str) -> Result<()> {
  let mut rows = conn.query(sql, ()).await?;
  let _ = rows.next().await?;
  Ok(())
}

fn db_busy_timeout_ms() -> u64 {
  std::env::var("ZAGE_DB_BUSY_TIMEOUT_MS")
    .ok()
    .and_then(|val| val.parse::<u64>().ok())
    .unwrap_or(300_000)
}

async fn configure_local_connection(conn: &Connection) -> Result<()> {
  // WAL gives much better read/write concurrency and tends to improve throughput on bulk imports.
  // Ignore the returned row; we only care that it succeeds.
  apply_pragma(conn, "PRAGMA journal_mode=WAL").await?;
  // With WAL, NORMAL is the usual durability/perf tradeoff and is far faster than FULL for
  // bulk imports and index rebuilds.
  apply_pragma(conn, "PRAGMA synchronous=NORMAL").await?;
  // Keep temporary data structures in memory where possible (index rebuilds can use temp tables).
  apply_pragma(conn, "PRAGMA temp_store=MEMORY").await?;
  apply_pragma(
    conn,
    &format!("PRAGMA busy_timeout={}", db_busy_timeout_ms()),
  )
  .await?;
  Ok(())
}

pub async fn open_db<P: AsRef<Path>>(db_path: P) -> Result<Db> {
  let path_ref = db_path.as_ref();
  if let Some(parent) = path_ref.parent() {
    fs::create_dir_all(parent)?;
  }
  let path = path_ref.to_string_lossy().to_string();
  let db = Builder::new_local(path).build().await?;

  let conn = db.connect()?;
  configure_local_connection(&conn).await?;
  init(&conn).await?;
  Ok(Db { db, conn })
}

pub async fn open_db_with_config(config: &DbConfig) -> Result<Db> {
  match config.kind {
    DbKind::Local => {
      let path_ref = config.path.as_path();
      if let Some(parent) = path_ref.parent() {
        fs::create_dir_all(parent)?;
      }
      let mut builder = Builder::new_local(path_ref);
      if let Some(key) = config.resolved_encryption_key() {
        let cipher = config.resolved_cipher()?.unwrap_or(Cipher::Aes256Cbc);
        builder = builder.encryption_config(EncryptionConfig {
          cipher,
          encryption_key: key.into(),
        });
      }
      let db = builder.build().await?;
      let conn = db.connect()?;
      configure_local_connection(&conn).await?;
      init(&conn).await?;
      Ok(Db { db, conn })
    }
    DbKind::Remote => {
      let url = config
        .url
        .clone()
        .ok_or_else(|| crate::ZageError::ConfigError("Missing remote db url".to_string()))?;
      let auth_token = config.resolved_auth_token().unwrap_or_default();
      let mut builder = Builder::new_remote(url, auth_token);
      if let Some(key) = config.resolved_remote_encryption_key() {
        builder = builder.remote_encryption(EncryptionContext {
          key: EncryptionKey::Base64Encoded(key),
        });
      }
      let db = builder.build().await?;
      let conn = db.connect()?;
      init(&conn).await?;
      Ok(Db { db, conn })
    }
    DbKind::RemoteReplica => {
      let url = config
        .url
        .clone()
        .ok_or_else(|| crate::ZageError::ConfigError("Missing remote db url".to_string()))?;
      let auth_token = config.resolved_auth_token().unwrap_or_default();
      let path_ref = config.path.as_path();
      if let Some(parent) = path_ref.parent() {
        fs::create_dir_all(parent)?;
      }
      let mut builder = Builder::new_remote_replica(path_ref, url, auth_token);
      if let Some(key) = config.resolved_encryption_key() {
        let cipher = config.resolved_cipher()?.unwrap_or(Cipher::Aes256Cbc);
        builder = builder.encryption_config(EncryptionConfig {
          cipher,
          encryption_key: key.into(),
        });
      }
      if let Some(key) = config.resolved_remote_encryption_key() {
        builder = builder.remote_encryption(EncryptionContext {
          key: EncryptionKey::Base64Encoded(key),
        });
      }
      if let Some(interval) = config.resolved_sync_interval_ms() {
        builder = builder.sync_interval(std::time::Duration::from_millis(interval));
      }
      let db = builder.build().await?;
      let conn = db.connect()?;
      init(&conn).await?;
      Ok(Db { db, conn })
    }
  }
}

pub async fn init(conn: &Connection) -> Result<()> {
  execute_batch(conn, include_str!("db/schema-v0.sql")).await?;
  ensure_sequence_stats_columns(conn).await?;
  Ok(())
}

async fn ensure_sequence_stats_columns(conn: &Connection) -> Result<()> {
  let mut rows = conn.query("PRAGMA table_info(sequence_stats)", ()).await?;
  let mut columns = HashSet::new();
  while let Some(row) = rows.next().await? {
    let name: String = row.get(1)?;
    columns.insert(name);
  }
  if !columns.contains("sequence_len") {
    conn
      .execute(
        "ALTER TABLE sequence_stats ADD COLUMN sequence_len INTEGER NOT NULL DEFAULT 0",
        (),
      )
      .await?;
  }
  if !columns.contains("prefix_json") {
    conn
      .execute("ALTER TABLE sequence_stats ADD COLUMN prefix_json TEXT", ())
      .await?;
  }
  if !columns.contains("last_command") {
    conn
      .execute(
        "ALTER TABLE sequence_stats ADD COLUMN last_command TEXT",
        (),
      )
      .await?;
  }
  Ok(())
}

pub async fn insert_invocation(conn: &Connection, invocation: &Invocation) -> Result<bool> {
  let id = uuid::Uuid::now_v7().to_string();
  let command = crate::tokenize::normalize_command_whitespace(&invocation.command);
  let expanded_command = if invocation.expanded_command.is_empty() {
    String::new()
  } else {
    crate::tokenize::normalize_command_whitespace(&invocation.expanded_command)
  };
  let changed = conn
    .execute(
      "INSERT OR IGNORE INTO shell_history (id, command, expanded_command, shellname, working_directory, hostname, username, exit_status, start_unix_timestamp, end_unix_timestamp, session_id)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      (
        id,
        command,
        expanded_command,
        invocation.shellname.clone(),
        invocation.working_directory.clone(),
        invocation.hostname.clone(),
        invocation.username.clone(),
        invocation.exit_status,
        invocation.start_unix_timestamp,
        invocation.end_unix_timestamp,
        invocation.session_id,
      ),
    )
    .await?;
  if changed > 0 {
    return Ok(true);
  }

  Ok(false)
}

pub async fn import_history<I>(conn: &Connection, invocations: I) -> Result<()>
where
  I: IntoIterator<Item = Invocation>,
{
  let mut processed: usize = 0;
  let mut inserted: usize = 0;
  let progress_interval = 10_000usize;

  // Keep inserts fast by batching many rows per statement.
  // SQLite default max variables is often 999; stay below it.
  const MAX_SQL_VARS: usize = 900;
  const COLS_PER_ROW: usize = 11;
  let batch_rows = (MAX_SQL_VARS / COLS_PER_ROW).max(1);

  conn.execute("BEGIN", ()).await?;

  let mut batch_params: Vec<libsql::Value> = Vec::with_capacity(batch_rows * COLS_PER_ROW);
  let mut batch_count: usize = 0;

  async fn flush_shell_history_batch(
    conn: &Connection,
    batch_params: &mut Vec<libsql::Value>,
    batch_count: &mut usize,
  ) -> Result<usize> {
    const COLS_PER_ROW: usize = 11;
    if *batch_count == 0 {
      return Ok(0);
    }

    let mut sql = String::from(
      "INSERT OR IGNORE INTO shell_history (id, command, expanded_command, shellname, working_directory, hostname, username, exit_status, start_unix_timestamp, end_unix_timestamp, session_id) VALUES ",
    );
    append_multi_row_placeholders(&mut sql, *batch_count, COLS_PER_ROW);

    let params = std::mem::take(batch_params);
    *batch_count = 0;

    let changed = conn.execute(&sql, params).await?;
    Ok(changed as usize)
  }

  let write_result: Result<()> = async {
    for mut invocation in invocations {
      if invocation.expanded_command.is_empty() {
        invocation.expanded_command = invocation.command.clone();
      }

      if batch_count >= batch_rows {
        inserted += flush_shell_history_batch(conn, &mut batch_params, &mut batch_count).await?;
      }

      let id = uuid::Uuid::now_v7().to_string();
      let command = crate::tokenize::normalize_command_whitespace(&invocation.command);
      let expanded = crate::tokenize::normalize_command_whitespace(&invocation.expanded_command);
      batch_params.push(libsql::Value::Text(id));
      batch_params.push(libsql::Value::Text(command));
      batch_params.push(libsql::Value::Text(expanded));
      batch_params.push(libsql::Value::Text(invocation.shellname));
      batch_params.push(match invocation.working_directory {
        Some(v) => libsql::Value::Text(v),
        None => libsql::Value::Null,
      });
      batch_params.push(match invocation.hostname {
        Some(v) => libsql::Value::Text(v),
        None => libsql::Value::Null,
      });
      batch_params.push(match invocation.username {
        Some(v) => libsql::Value::Text(v),
        None => libsql::Value::Null,
      });
      batch_params.push(match invocation.exit_status {
        Some(v) => libsql::Value::Integer(v),
        None => libsql::Value::Null,
      });
      batch_params.push(match invocation.start_unix_timestamp {
        Some(v) => libsql::Value::Integer(v),
        None => libsql::Value::Null,
      });
      batch_params.push(match invocation.end_unix_timestamp {
        Some(v) => libsql::Value::Integer(v),
        None => libsql::Value::Null,
      });
      batch_params.push(libsql::Value::Integer(invocation.session_id));

      batch_count += 1;
      processed += 1;
      if processed.is_multiple_of(progress_interval) {
        inserted += flush_shell_history_batch(conn, &mut batch_params, &mut batch_count).await?;
        info!(
          "Imported {} history entries ({} inserted)",
          processed, inserted
        );
      }
    }

    inserted += flush_shell_history_batch(conn, &mut batch_params, &mut batch_count).await?;
    Ok(())
  }
  .await;

  if let Err(err) = write_result {
    let _ = conn.execute("ROLLBACK", ()).await;
    return Err(err);
  }

  conn.execute("COMMIT", ()).await?;
  info!(
    "Imported {} history entries ({} inserted)",
    processed, inserted
  );
  Ok(())
}

pub async fn delete_history_by_command(
  conn: &Connection,
  command: &str,
  match_expanded: bool,
) -> Result<u64> {
  let affected = if match_expanded {
    conn
      .execute(
        "DELETE FROM shell_history WHERE command = ? OR expanded_command = ?",
        libsql::params![command.to_string(), command.to_string()],
      )
      .await?
  } else {
    conn
      .execute(
        "DELETE FROM shell_history WHERE command = ?",
        libsql::params![command.to_string()],
      )
      .await?
  };
  Ok(affected)
}

pub async fn get_recent_invocations(
  conn: &Connection,
  session_id: Option<i64>,
  limit: usize,
) -> Result<Vec<Invocation>> {
  let mut rows = conn
    .query(
      include_str!("db/get-recent-invocations.sql"),
      libsql::params![session_id, limit as i64],
    )
    .await?;

  let mut invs = Vec::new();
  while let Some(row) = rows.next().await? {
    invs.push(Invocation {
      command: row.get::<String>(1)?,
      expanded_command: row.get::<String>(2)?,
      shellname: row.get::<String>(3)?,
      working_directory: row.get::<Option<String>>(4)?,
      workspace: None,
      hostname: row.get::<Option<String>>(5)?,
      username: row.get::<Option<String>>(6)?,
      exit_status: row.get::<Option<i64>>(7)?,
      start_unix_timestamp: row.get::<Option<i64>>(8)?,
      end_unix_timestamp: row.get::<Option<i64>>(9)?,
      session_id: row.get::<i64>(10)?,
    });
  }
  invs.reverse();
  Ok(invs)
}

fn append_multi_row_placeholders(sql: &mut String, rows: usize, cols_per_row: usize) {
  for row_idx in 0..rows {
    if row_idx > 0 {
      sql.push(',');
    }
    sql.push('(');
    for col_idx in 0..cols_per_row {
      if col_idx > 0 {
        sql.push(',');
      }
      sql.push('?');
    }
    sql.push(')');
  }
}

async fn update_sequence_stats_for_invocation(
  conn: &Connection,
  invocation: &Invocation,
) -> Result<()> {
  let config = SequenceConfig::default();
  let max_len = config.max_len.max(2);
  let mut window: Vec<String> = Vec::new();

  let recent = recent_invocations_for_session(conn, invocation, max_len - 1).await?;
  for prev in &recent {
    let shellname = if prev.shellname.is_empty() {
      invocation.shellname.as_str()
    } else {
      prev.shellname.as_str()
    };
    let raw_cmd = if prev.expanded_command.is_empty() {
      prev.command.as_str()
    } else {
      prev.expanded_command.as_str()
    };
    window.push(normalize_sequence_command(shellname, raw_cmd));
  }

  let raw_cmd = if invocation.expanded_command.is_empty() {
    invocation.command.as_str()
  } else {
    invocation.expanded_command.as_str()
  };
  window.push(normalize_sequence_command(
    invocation.shellname.as_str(),
    raw_cmd,
  ));

  if window.is_empty() {
    return Ok(());
  }

  let mut updated_prefixes: BTreeSet<String> = BTreeSet::new();
  for n in 1..=max_len {
    if window.len() < n {
      continue;
    }
    let start = window.len() - n;
    let sequence = window[start..].to_vec();
    let sequence_json = serde_json::to_string(&sequence)?;
    let last_command = sequence.last().cloned().unwrap_or_default();
    let (sequence_len, prefix_json) = if n == 1 {
      (1usize, None)
    } else {
      let prefix = serde_json::to_string(&sequence[..n - 1])?;
      updated_prefixes.insert(prefix.clone());
      (n, Some(prefix))
    };

    conn
      .execute(
        "INSERT INTO sequence_stats (sequence_json, support, confidence, lift, sequence_len, prefix_json, last_command, context_json)
         VALUES (?, 1, 0.0, 0.0, ?, ?, ?, NULL)
         ON CONFLICT(sequence_json) DO UPDATE SET
           support = support + 1,
           sequence_len = excluded.sequence_len,
           prefix_json = excluded.prefix_json,
           last_command = excluded.last_command",
        (
          sequence_json,
          sequence_len as i64,
          prefix_json,
          last_command,
        ),
      )
      .await?;
  }

  if updated_prefixes.is_empty() {
    return Ok(());
  }

  let mut rows = conn
    .query(
      "SELECT SUM(support) FROM sequence_stats WHERE sequence_len = 1",
      (),
    )
    .await?;
  let row = rows.next().await?;
  let total_unigram_support: i64 = match row {
    Some(row) => row.get::<Option<i64>>(0)?.unwrap_or(0),
    None => 0,
  };

  for prefix_json in &updated_prefixes {
    conn
      .execute(
        "UPDATE sequence_stats
         SET confidence = CASE
           WHEN (SELECT support FROM sequence_stats WHERE sequence_json = ?) > 0
             THEN CAST(support AS REAL) / (SELECT support FROM sequence_stats WHERE sequence_json = ?)
           ELSE 0.0
         END
         WHERE prefix_json = ?",
        (prefix_json.clone(), prefix_json.clone(), prefix_json.clone()),
      )
      .await?;

    if total_unigram_support > 0 {
      conn
        .execute(
          "UPDATE sequence_stats
           SET lift = CASE
             WHEN (SELECT support FROM sequence_stats AS uni
                   WHERE uni.sequence_len = 1
                     AND uni.last_command = sequence_stats.last_command) > 0
               THEN confidence / (
                 (SELECT support FROM sequence_stats AS uni
                  WHERE uni.sequence_len = 1
                    AND uni.last_command = sequence_stats.last_command) * 1.0 / ?
               )
             ELSE 0.0
           END
           WHERE prefix_json = ?
             AND sequence_len >= 2",
          (total_unigram_support, prefix_json.clone()),
        )
        .await?;
    }
  }

  Ok(())
}

fn resolve_workspace_root(
  working_directory: Option<&str>,
  workspace: Option<&crate::workspace::WorkspaceInfo>,
) -> String {
  if let Some(info) = workspace
    && !info.root.trim().is_empty()
  {
    return info.root.clone();
  }
  let Some(cwd) = working_directory else {
    return String::new();
  };
  if let Ok(Some(info)) = crate::workspace::detect_workspace_for_cwd(cwd)
    && !info.root.trim().is_empty()
  {
    return info.root;
  }
  String::new()
}

pub async fn update_stats_for_invocation(conn: &Connection, invocation: &Invocation) -> Result<()> {
  let now = invocation
    .start_unix_timestamp
    .or(invocation.end_unix_timestamp)
    .unwrap_or_else(|| {
      std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
    });

  let workspace_root = resolve_workspace_root(
    invocation.working_directory.as_deref(),
    invocation.workspace.as_ref(),
  );
  let stats_command = if invocation.expanded_command.is_empty() {
    invocation.command.as_str()
  } else {
    invocation.expanded_command.as_str()
  };
  let stats_command = normalize_command_whitespace(stats_command);
  let tokens = tokenize_index(&invocation.shellname, &stats_command);
  let raw_tokens: Vec<String> = tokens.iter().map(|t| t.raw.clone()).collect();
  let norm_tokens: Vec<String> = tokens.iter().map(|t| t.normalized.clone()).collect();

  conn.execute("BEGIN", ()).await?;
  let write_result: Result<()> = async {
    conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen)
         VALUES (?, 1, ?)
         ON CONFLICT(command) DO UPDATE SET
           freq = freq + 1,
           last_seen = MAX(last_seen, excluded.last_seen)",
        (stats_command.to_string(), now),
      )
      .await?;

    conn
      .execute(
        "INSERT INTO workspace_command_stats (workspace_root, command, freq, last_seen)
         VALUES (?, ?, 1, ?)
         ON CONFLICT(workspace_root, command) DO UPDATE SET
           freq = freq + 1,
           last_seen = MAX(last_seen, excluded.last_seen)",
        (workspace_root.clone(), stats_command.to_string(), now),
      )
      .await?;

    conn
      .execute(
        "INSERT INTO context_stats (command, working_directory, hostname, username, freq, last_seen)
         VALUES (?, ?, ?, ?, 1, ?)
         ON CONFLICT(command, working_directory, hostname, username) DO UPDATE SET
           freq = freq + 1,
           last_seen = MAX(last_seen, excluded.last_seen)",
        (
          stats_command.to_string(),
          invocation.working_directory.clone(),
          invocation.hostname.clone(),
          invocation.username.clone(),
          now,
        ),
      )
      .await?;

    if let Some(prev) = previous_invocation_for_session(conn, invocation).await? {
      let prev_command = if prev.expanded_command.is_empty() {
        prev.command.as_str()
      } else {
        prev.expanded_command.as_str()
      };
      let prev_command = normalize_command_whitespace(prev_command);
      let prev_status = prev.exit_status;
      let prev_workspace_root =
        resolve_workspace_root(prev.working_directory.as_deref(), None);

      let prev_head = {
        let tokens = tokenize_index(prev.shellname.as_str(), &prev_command);
        extract_command_stats_parts(&prev_command, &tokens)
          .map(|parts| parts.head)
          .or_else(|| extract_command_parts(&prev_command, &tokens).map(|parts| parts.head))
      };

      conn
        .execute(
        "INSERT INTO transition_stats (prev_command, prev_exit_status, next_command, freq, last_seen)
           VALUES (?, ?, ?, 1, ?)
           ON CONFLICT(prev_command, prev_exit_status, next_command) DO UPDATE SET
             freq = freq + 1,
             last_seen = MAX(last_seen, excluded.last_seen)",
          (
            prev_command.to_string(),
            prev_status,
            stats_command.to_string(),
            now,
          ),
        )
        .await?;

      conn
        .execute(
        "INSERT INTO transition_stats (prev_command, prev_exit_status, next_command, freq, last_seen)
           VALUES (?, NULL, ?, 1, ?)
           ON CONFLICT(prev_command, prev_exit_status, next_command) DO UPDATE SET
             freq = freq + 1,
             last_seen = MAX(last_seen, excluded.last_seen)",
          (prev_command.to_string(), stats_command.to_string(), now),
        )
        .await?;

      if let Some(prev_head) = prev_head.as_deref() {
        conn
          .execute(
            "INSERT INTO transition_head_stats (prev_head, prev_exit_status, next_command, freq, last_seen)
             VALUES (?, ?, ?, 1, ?)
             ON CONFLICT(prev_head, prev_exit_status, next_command) DO UPDATE SET
               freq = freq + 1,
               last_seen = MAX(last_seen, excluded.last_seen)",
            (
              prev_head.to_string(),
              prev_status,
              stats_command.to_string(),
              now,
            ),
          )
          .await?;

        conn
          .execute(
            "INSERT INTO transition_head_stats (prev_head, prev_exit_status, next_command, freq, last_seen)
             VALUES (?, NULL, ?, 1, ?)
             ON CONFLICT(prev_head, prev_exit_status, next_command) DO UPDATE SET
               freq = freq + 1,
               last_seen = MAX(last_seen, excluded.last_seen)",
            (prev_head.to_string(), stats_command.to_string(), now),
          )
          .await?;
      }

      conn
        .execute(
        "INSERT INTO workspace_transition_stats (workspace_root, prev_command, prev_exit_status, next_command, freq, last_seen)
           VALUES (?, ?, ?, ?, 1, ?)
           ON CONFLICT(workspace_root, prev_command, prev_exit_status, next_command) DO UPDATE SET
             freq = freq + 1,
             last_seen = MAX(last_seen, excluded.last_seen)",
          (
            prev_workspace_root.clone(),
            prev_command.to_string(),
            prev_status,
            stats_command.to_string(),
            now,
          ),
        )
        .await?;

      conn
        .execute(
        "INSERT INTO workspace_transition_stats (workspace_root, prev_command, prev_exit_status, next_command, freq, last_seen)
           VALUES (?, ?, NULL, ?, 1, ?)
           ON CONFLICT(workspace_root, prev_command, prev_exit_status, next_command) DO UPDATE SET
             freq = freq + 1,
             last_seen = MAX(last_seen, excluded.last_seen)",
          (
            prev_workspace_root.clone(),
            prev_command.to_string(),
            stats_command.to_string(),
            now,
          ),
        )
        .await?;

      if let Some(prev_head) = prev_head.as_deref() {
        conn
          .execute(
            "INSERT INTO workspace_transition_head_stats (workspace_root, prev_head, prev_exit_status, next_command, freq, last_seen)
             VALUES (?, ?, ?, ?, 1, ?)
             ON CONFLICT(workspace_root, prev_head, prev_exit_status, next_command) DO UPDATE SET
               freq = freq + 1,
               last_seen = MAX(last_seen, excluded.last_seen)",
            (
              prev_workspace_root.clone(),
              prev_head.to_string(),
              prev_status,
              stats_command.to_string(),
              now,
            ),
          )
          .await?;

        conn
          .execute(
            "INSERT INTO workspace_transition_head_stats (workspace_root, prev_head, prev_exit_status, next_command, freq, last_seen)
             VALUES (?, ?, NULL, ?, 1, ?)
             ON CONFLICT(workspace_root, prev_head, prev_exit_status, next_command) DO UPDATE SET
               freq = freq + 1,
               last_seen = MAX(last_seen, excluded.last_seen)",
            (
              prev_workspace_root,
              prev_head.to_string(),
              stats_command.to_string(),
              now,
            ),
          )
          .await?;
      }
    }

    let raw_json = serde_json::to_string(&raw_tokens)?;
    let norm_json = serde_json::to_string(&norm_tokens)?;
    conn
      .execute(
        "INSERT INTO token_cache (command, tokens_json, normalized_json, updated_at)
         VALUES (?, ?, ?, ?)
         ON CONFLICT(command) DO UPDATE SET
           tokens_json = excluded.tokens_json,
           normalized_json = excluded.normalized_json,
           updated_at = excluded.updated_at",
        (stats_command.to_string(), raw_json, norm_json, now),
      )
      .await?;

    async fn upsert_parts_stats(
      conn: &Connection,
      workspace_root: &str,
      head: &str,
      env: &[crate::tokenize::Token],
      flags: &[String],
      args: &[crate::tokenize::Token],
      now: i64,
    ) -> Result<()> {
      let mut flags = flags.to_vec();
      flags.sort();
      let flags_json = serde_json::to_string(&flags)?;
      for flag in &flags {
        let flag_norm = normalize_token(flag);
        conn
          .execute(
            "INSERT INTO flag_stats (workspace_root, command_head, flag_raw, flag_norm, freq, last_seen)
             VALUES (?, ?, ?, ?, 1, ?)
             ON CONFLICT(workspace_root, command_head, flag_raw) DO UPDATE SET
               freq = freq + 1,
               last_seen = MAX(last_seen, excluded.last_seen)",
            (
              workspace_root.to_string(),
              head.to_string(),
              flag.clone(),
              flag_norm,
              now,
            ),
          )
          .await?;
      }
      for env in env {
        let env_key = env
          .raw
          .split('=')
          .next()
          .unwrap_or_default()
          .to_string();
        conn
          .execute(
            "INSERT INTO env_stats (workspace_root, command_head, env_key, env_raw, env_norm, freq, last_seen)
             VALUES (?, ?, ?, ?, ?, 1, ?)
             ON CONFLICT(workspace_root, command_head, env_raw) DO UPDATE SET
               freq = freq + 1,
               last_seen = MAX(last_seen, excluded.last_seen)",
            (
              workspace_root.to_string(),
              head.to_string(),
              env_key,
              env.raw.clone(),
              env.normalized.clone(),
              now,
            ),
          )
          .await?;
      }
      for (idx, arg) in args.iter().enumerate() {
        conn
          .execute(
            "INSERT INTO arg_stats (workspace_root, command_head, flags_json, arg_index, arg_raw, arg_norm, freq, last_seen)
             VALUES (?, ?, ?, ?, ?, ?, 1, ?)
             ON CONFLICT(workspace_root, command_head, flags_json, arg_index, arg_raw) DO UPDATE SET
               freq = freq + 1,
               last_seen = MAX(last_seen, excluded.last_seen)",
            (
              workspace_root.to_string(),
              head.to_string(),
              flags_json.clone(),
              idx as i64,
              arg.raw.clone(),
              arg.normalized.clone(),
              now,
            ),
          )
          .await?;

        conn
          .execute(
            "INSERT INTO arg_stats_any (workspace_root, command_head, flags_json, arg_raw, arg_norm, freq, last_seen)
             VALUES (?, ?, ?, ?, ?, 1, ?)
             ON CONFLICT(workspace_root, command_head, flags_json, arg_raw) DO UPDATE SET
               freq = freq + 1,
               last_seen = MAX(last_seen, excluded.last_seen)",
            (
              workspace_root.to_string(),
              head.to_string(),
              flags_json.clone(),
              arg.raw.clone(),
              arg.normalized.clone(),
              now,
            ),
          )
          .await?;
      }
      Ok(())
    }

    let mut base_head: Option<String> = None;
    if let Some(parts) = extract_command_parts(&stats_command, &tokens) {
      base_head = Some(parts.head.clone());
      upsert_parts_stats(
        conn,
        &workspace_root,
        &parts.head,
        &parts.env,
        &parts.flags,
        &parts.args,
        now,
      )
      .await?;
    }
    if let Some(parts) = extract_command_stats_parts(&stats_command, &tokens)
      && base_head.as_deref() != Some(parts.head.as_str()) {
        upsert_parts_stats(
          conn,
          &workspace_root,
          &parts.head,
          &parts.env,
          &parts.flags,
          &parts.args,
          now,
        )
        .await?;
      }

    update_sequence_stats_for_invocation(conn, invocation).await?;

    Ok(())
  }
  .await;

  if let Err(err) = write_result {
    let _ = conn.execute("ROLLBACK", ()).await;
    return Err(err);
  }

  conn.execute("COMMIT", ()).await?;
  Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OnlineModelStatus {
  pub meta_entries: u64,
  pub token_embeddings: u64,
  pub command_biases: u64,
  pub context_biases: u64,
  pub head_biases: u64,
  pub group_scalars: u64,
  pub replay_global: u64,
  pub replay_workspace: u64,
  pub feedback: u64,
}

pub async fn online_model_status(conn: &Connection) -> Result<OnlineModelStatus> {
  Ok(OnlineModelStatus {
    meta_entries: count_rows(conn, "online_model_meta").await?,
    token_embeddings: count_rows(conn, "online_token_embedding").await?,
    command_biases: count_rows(conn, "online_command_bias").await?,
    context_biases: count_rows(conn, "online_context_bias").await?,
    head_biases: count_rows(conn, "online_head_bias").await?,
    group_scalars: count_rows(conn, "online_group_scalar").await?,
    replay_global: count_rows(conn, "online_replay_global").await?,
    replay_workspace: count_rows(conn, "online_replay_workspace").await?,
    feedback: count_rows(conn, "online_feedback").await?,
  })
}

pub async fn online_model_last_updated_at(conn: &Connection) -> Result<Option<i64>> {
  let mut last: Option<i64> = None;
  for table in [
    "online_token_embedding",
    "online_command_bias",
    "online_context_bias",
    "online_head_bias",
    "online_group_scalar",
  ] {
    let sql = format!("SELECT MAX(updated_at) FROM {table}");
    let mut rows = conn.query(&sql, ()).await?;
    let row = rows.next().await?.ok_or_else(|| {
      crate::ZageError::ConfigError(format!("missing MAX(updated_at) row for table {table}"))
    })?;
    let updated_at: Option<i64> = row.get(0)?;
    match (last, updated_at) {
      (None, Some(ts)) => last = Some(ts),
      (Some(prev), Some(ts)) if ts > prev => last = Some(ts),
      _ => {}
    }
  }
  Ok(last)
}

const ONLINE_MODEL_UPDATE_COUNT_KEY: &str = "online_model_update_count";

pub async fn online_model_update_count(conn: &Connection) -> Result<u64> {
  let Some(value) = online_model_meta_value(conn, ONLINE_MODEL_UPDATE_COUNT_KEY).await? else {
    return Ok(0);
  };
  value.parse::<u64>().map_err(|err| {
    crate::ZageError::ConfigError(format!(
      "invalid online model update count: {value} ({err})"
    ))
  })
}

pub async fn bump_online_model_update_count(conn: &Connection, delta: u64) -> Result<u64> {
  if delta == 0 {
    return online_model_update_count(conn).await;
  }
  let current = online_model_update_count(conn).await?;
  let next = current.saturating_add(delta);
  online_model_meta_set(conn, ONLINE_MODEL_UPDATE_COUNT_KEY, &next.to_string()).await?;
  Ok(next)
}

pub async fn online_model_group_scalars(conn: &Connection) -> Result<Vec<(String, f64)>> {
  let mut rows = conn
    .query(
      "SELECT group_name, value FROM online_group_scalar ORDER BY group_name",
      (),
    )
    .await?;
  let mut out = Vec::new();
  while let Some(row) = rows.next().await? {
    let group: String = row.get(0)?;
    let value: f64 = row.get(1)?;
    out.push((group, value));
  }
  Ok(out)
}

pub async fn online_model_head_biases(
  conn: &Connection,
  limit: usize,
) -> Result<Vec<(String, f64)>> {
  let mut rows = conn
    .query(
      "SELECT head, bias FROM online_head_bias ORDER BY ABS(bias) DESC LIMIT ?",
      libsql::params![limit as i64],
    )
    .await?;
  let mut out = Vec::new();
  while let Some(row) = rows.next().await? {
    let head: String = row.get(0)?;
    let bias: f64 = row.get(1)?;
    out.push((head, bias));
  }
  Ok(out)
}

pub async fn online_replay_workspace_roots(conn: &Connection) -> Result<u64> {
  let mut rows = conn
    .query(
      "SELECT COUNT(DISTINCT workspace_root) FROM online_replay_workspace",
      (),
    )
    .await?;
  let row = rows.next().await?.ok_or_else(|| {
    crate::ZageError::ConfigError("missing COUNT(DISTINCT workspace_root) row".to_string())
  })?;
  let count: i64 = row.get(0)?;
  Ok(count.max(0) as u64)
}

pub async fn reset_online_model(conn: &Connection) -> Result<()> {
  conn.execute("BEGIN", ()).await?;

  let write_result: Result<()> = async {
    // Keep this list explicit so we never accidentally delete non-model data.
    for stmt in [
      "DELETE FROM online_model_meta",
      "DELETE FROM online_token_embedding",
      "DELETE FROM online_command_bias",
      "DELETE FROM online_context_bias",
      "DELETE FROM online_head_bias",
      "DELETE FROM online_group_scalar",
      "DELETE FROM online_replay_global",
      "DELETE FROM online_replay_workspace",
      "DELETE FROM online_feedback",
    ] {
      conn.execute(stmt, ()).await?;
    }
    Ok(())
  }
  .await;

  if let Err(err) = write_result {
    let _ = conn.execute("ROLLBACK", ()).await;
    return Err(err);
  }

  conn.execute("COMMIT", ()).await?;
  Ok(())
}

pub(crate) async fn online_model_meta_value(
  conn: &Connection,
  key: &str,
) -> Result<Option<String>> {
  let mut rows = conn
    .query(
      "SELECT value FROM online_model_meta WHERE key = ?",
      libsql::params![key.to_string()],
    )
    .await?;
  let Some(row) = rows.next().await? else {
    return Ok(None);
  };
  let value: String = row.get(0)?;
  Ok(Some(value))
}

pub(crate) async fn online_model_meta_set(conn: &Connection, key: &str, value: &str) -> Result<()> {
  conn
    .execute(
      "INSERT INTO online_model_meta (key, value) VALUES (?, ?)
       ON CONFLICT(key) DO UPDATE SET value = excluded.value",
      libsql::params![key.to_string(), value.to_string()],
    )
    .await?;
  Ok(())
}

#[derive(Debug, Clone)]
pub struct OnlineFeedbackEvent {
  pub shown_id: String,
  pub shown_at: i64,
  pub cwd: Option<String>,
  pub suggestion: String,
  pub accepted_command: Option<String>,
  pub accepted_at: Option<i64>,
  pub outcome: Option<String>,
}

pub async fn upsert_online_feedback(conn: &Connection, event: OnlineFeedbackEvent) -> Result<()> {
  if event.shown_id.trim().is_empty() {
    return Err(crate::ZageError::ConfigError(
      "feedback shown_id is required".to_string(),
    ));
  }
  if event.shown_at <= 0 {
    return Err(crate::ZageError::ConfigError(
      "feedback shown_at must be > 0".to_string(),
    ));
  }
  if event.suggestion.trim().is_empty() {
    return Err(crate::ZageError::ConfigError(
      "feedback suggestion is required".to_string(),
    ));
  }
  if let Some(at) = event.accepted_at
    && at <= 0
  {
    return Err(crate::ZageError::ConfigError(
      "feedback accepted_at must be > 0 when present".to_string(),
    ));
  }
  if let Some(outcome) = event.outcome.as_deref()
    && outcome.trim().is_empty()
  {
    return Err(crate::ZageError::ConfigError(
      "feedback outcome must be non-empty when present".to_string(),
    ));
  }

  let workspace_root = event
    .cwd
    .as_deref()
    .and_then(|cwd| {
      crate::workspace::detect_workspace_for_cwd(cwd)
        .ok()
        .flatten()
    })
    .map(|w| w.root);

  conn
    .execute(
      "INSERT INTO online_feedback (
         shown_id, shown_at, workspace_root, cwd, prefix, suggestion, accepted_command, accepted_at, outcome
       ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?)
       ON CONFLICT(shown_id) DO UPDATE SET
         shown_at = MIN(shown_at, excluded.shown_at),
         workspace_root = COALESCE(online_feedback.workspace_root, excluded.workspace_root),
         cwd = COALESCE(online_feedback.cwd, excluded.cwd),
         suggestion = COALESCE(online_feedback.suggestion, excluded.suggestion),
         accepted_command = COALESCE(online_feedback.accepted_command, excluded.accepted_command),
         accepted_at = COALESCE(online_feedback.accepted_at, excluded.accepted_at),
         outcome = COALESCE(online_feedback.outcome, excluded.outcome)",
      libsql::params![
        event.shown_id,
        event.shown_at,
        workspace_root,
        event.cwd,
        event.suggestion,
        event.accepted_command,
        event.accepted_at,
        event.outcome
      ],
    )
    .await?;

  Ok(())
}

async fn count_rows(conn: &Connection, table: &str) -> Result<u64> {
  let sql = format!("SELECT COUNT(*) FROM {table}");
  let mut rows = conn.query(&sql, ()).await?;
  let row = rows.next().await?.ok_or_else(|| {
    crate::ZageError::ConfigError(format!("missing COUNT(*) row for table {table}"))
  })?;
  let count: i64 = row.get(0)?;
  Ok(count.try_into().unwrap_or(0))
}

#[cfg(test)]
mod import_tests {
  use super::*;
  use crate::shell_history;
  use std::io::Write;
  use tempfile::{NamedTempFile, tempdir};

  #[tokio::test]
  async fn test_import_history_basic() -> Result<()> {
    let tmp_db = NamedTempFile::new()?;
    let db = open_db(tmp_db.path()).await?;
    init(&db.conn).await?;

    let mut tmp = NamedTempFile::new()?;
    let content = ":1610000000:2;echo hello
:1610000002:3;ls -la
:1610000005:1;echo hello
";
    tmp.write_all(content.as_bytes())?;
    tmp.flush()?;

    let invocations = shell_history::parse_zsh_history(tmp.path(), None, None)?;
    import_history(&db.conn, invocations).await?;

    let mut rows = db
      .conn
      .query("SELECT COUNT(*) FROM shell_history", ())
      .await?;
    let row = rows.next().await?.expect("expected row");
    let count: i64 = row.get(0)?;
    assert_eq!(count, 3);
    Ok(())
  }

  #[tokio::test]
  async fn test_history_import() -> Result<()> {
    let temp_dir = tempdir()?;
    let db_path = temp_dir.path().join("test.db");

    let db = open_db(&db_path).await?;
    init(&db.conn).await?;

    let bash_history_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("tests")
      .join("data")
      .join("bash.history");
    let bash_invocations = shell_history::parse_bash_history(&bash_history_path, None, None)?;

    let zsh_history_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
      .join("tests")
      .join("data")
      .join("zsh.history");
    let zsh_invocations = shell_history::parse_zsh_history(&zsh_history_path, None, None)?;

    for invocation in bash_invocations {
      let inserted = insert_invocation(&db.conn, &invocation).await?;
      assert!(inserted);
    }

    for invocation in zsh_invocations {
      let inserted = insert_invocation(&db.conn, &invocation).await?;
      assert!(inserted);
    }

    let count = count_history_entries(&db.conn).await?;
    assert!(count > 0, "No history entries were imported");

    Ok(())
  }

  async fn count_history_entries(conn: &Connection) -> Result<usize> {
    let mut rows = conn.query("SELECT COUNT(*) FROM shell_history", ()).await?;
    let row = rows.next().await?.expect("expected row");
    let count: i64 = row.get(0)?;
    Ok(count as usize)
  }
}

struct PrevInvocation {
  command: String,
  expanded_command: String,
  shellname: String,
  exit_status: Option<i64>,
  working_directory: Option<String>,
}

async fn previous_invocation_for_session(
  conn: &Connection,
  invocation: &Invocation,
) -> Result<Option<PrevInvocation>> {
  let mut rows = conn
    .query(
      "SELECT command, expanded_command, shellname, exit_status, working_directory,
              start_unix_timestamp, end_unix_timestamp
       FROM shell_history
       WHERE session_id = ?
       ORDER BY COALESCE(end_unix_timestamp, start_unix_timestamp, 0) DESC, id DESC
       LIMIT 5",
      libsql::params![invocation.session_id],
    )
    .await?;
  while let Some(row) = rows.next().await? {
    let command: String = row.get(0)?;
    let expanded_command: String = row.get(1)?;
    let shellname: String = row.get(2)?;
    let exit_status: Option<i64> = row.get(3)?;
    let working_directory: Option<String> = row.get(4)?;
    let start_ts: Option<i64> = row.get(5)?;
    let end_ts: Option<i64> = row.get(6)?;
    let is_same = command == invocation.command
      && expanded_command == invocation.expanded_command
      && start_ts == invocation.start_unix_timestamp
      && end_ts == invocation.end_unix_timestamp;
    if is_same {
      continue;
    }
    return Ok(Some(PrevInvocation {
      command,
      expanded_command,
      shellname,
      exit_status,
      working_directory,
    }));
  }
  Ok(None)
}

async fn recent_invocations_for_session(
  conn: &Connection,
  invocation: &Invocation,
  limit: usize,
) -> Result<Vec<PrevInvocation>> {
  if limit == 0 {
    return Ok(Vec::new());
  }
  let mut rows = conn
    .query(
      "SELECT command, expanded_command, shellname, exit_status, working_directory,
              start_unix_timestamp, end_unix_timestamp
       FROM shell_history
       WHERE session_id = ?
       ORDER BY COALESCE(end_unix_timestamp, start_unix_timestamp, 0) DESC, id DESC
       LIMIT ?",
      libsql::params![invocation.session_id, (limit + 1) as i64],
    )
    .await?;

  let mut out = Vec::new();
  while let Some(row) = rows.next().await? {
    let command: String = row.get(0)?;
    let expanded_command: String = row.get(1)?;
    let shellname: String = row.get(2)?;
    let exit_status: Option<i64> = row.get(3)?;
    let working_directory: Option<String> = row.get(4)?;
    let start_ts: Option<i64> = row.get(5)?;
    let end_ts: Option<i64> = row.get(6)?;
    let is_same = command == invocation.command
      && expanded_command == invocation.expanded_command
      && start_ts == invocation.start_unix_timestamp
      && end_ts == invocation.end_unix_timestamp;
    if is_same {
      continue;
    }
    out.push(PrevInvocation {
      command,
      expanded_command,
      shellname,
      exit_status,
      working_directory,
    });
    if out.len() >= limit {
      break;
    }
  }
  out.reverse();
  Ok(out)
}

async fn execute_batch(conn: &Connection, sql: &str) -> Result<()> {
  let mut statements: Vec<&str> = Vec::new();
  let mut start = 0usize;
  for (idx, ch) in sql.char_indices() {
    if ch == ';' {
      let stmt = sql[start..idx].trim();
      if !stmt.is_empty() {
        statements.push(stmt);
      }
      start = idx + 1;
    }
  }
  if start < sql.len() {
    let stmt = sql[start..].trim();
    if !stmt.is_empty() {
      statements.push(stmt);
    }
  }

  for stmt in statements {
    conn.execute(stmt, ()).await?;
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[tokio::test]
  async fn test_init_table() -> Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    let mut rows = db
      .conn
      .query(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='shell_history'",
        (),
      )
      .await?;
    let row = rows.next().await?.expect("expected row");
    let count = row.get::<i64>(0)?;
    assert_eq!(count, 1);
    Ok(())
  }

  #[tokio::test]
  async fn init_creates_online_model_tables() -> Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    for table in [
      "online_model_meta",
      "online_token_embedding",
      "online_command_bias",
      "online_head_bias",
      "online_group_scalar",
      "online_replay_global",
      "online_replay_workspace",
      "online_feedback",
    ] {
      let mut rows = db
        .conn
        .query(
          "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
          libsql::params![table.to_string()],
        )
        .await?;
      let row = rows.next().await?.expect("expected row");
      let count = row.get::<i64>(0)?;
      assert_eq!(count, 1, "missing table: {table}");
    }

    Ok(())
  }

  #[tokio::test]
  async fn reset_online_model_clears_only_online_tables() -> Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    let invocation = Invocation {
      command: "ls -la".to_string(),
      expanded_command: "ls -la".to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/tmp".to_string()),
      workspace: None,
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(123),
      end_unix_timestamp: Some(124),
      session_id: 42,
    };
    assert!(insert_invocation(&db.conn, &invocation).await?);

    db.conn
      .execute(
        "INSERT INTO online_model_meta (key, value) VALUES (?, ?)",
        ("test".to_string(), "1".to_string()),
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO online_token_embedding (bucket, vec, opt_state, updated_at) VALUES (?, X'00', NULL, ?)",
        (1i64, 1i64),
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO online_command_bias (command, bias, updated_at) VALUES (?, ?, ?)",
        ("echo hi".to_string(), 0.5f64, 1i64),
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO online_context_bias (bucket, bias, updated_at) VALUES (?, ?, ?)",
        (1i64, 0.1f64, 1i64),
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO online_head_bias (head, bias, updated_at) VALUES (?, ?, ?)",
        ("echo".to_string(), 0.25f64, 1i64),
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO online_group_scalar (group_name, value, updated_at) VALUES (?, ?, ?)",
        ("global".to_string(), 1.0f64, 1i64),
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO online_replay_global (event_id, payload, sampled_at) VALUES (?, X'00', ?)",
        (1i64, 1i64),
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO online_replay_workspace (workspace_root, seq, payload) VALUES (?, ?, X'00')",
        ("/tmp".to_string(), 1i64),
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO online_feedback (shown_id, shown_at, suggestion) VALUES (?, ?, ?)",
        ("s1".to_string(), 1i64, "echo hi".to_string()),
      )
      .await?;

    let before = online_model_status(&db.conn).await?;
    assert_eq!(before.meta_entries, 1);
    assert_eq!(before.token_embeddings, 1);
    assert_eq!(before.command_biases, 1);
    assert_eq!(before.context_biases, 1);
    assert_eq!(before.head_biases, 1);
    assert_eq!(before.group_scalars, 1);
    assert_eq!(before.replay_global, 1);
    assert_eq!(before.replay_workspace, 1);
    assert_eq!(before.feedback, 1);

    let last_updated = online_model_last_updated_at(&db.conn).await?;
    assert_eq!(last_updated, Some(1));

    reset_online_model(&db.conn).await?;

    let after = online_model_status(&db.conn).await?;
    assert_eq!(
      after,
      OnlineModelStatus {
        meta_entries: 0,
        token_embeddings: 0,
        command_biases: 0,
        context_biases: 0,
        head_biases: 0,
        group_scalars: 0,
        replay_global: 0,
        replay_workspace: 0,
        feedback: 0,
      }
    );

    let last_updated = online_model_last_updated_at(&db.conn).await?;
    assert_eq!(last_updated, None);

    let mut rows = db
      .conn
      .query("SELECT COUNT(*) FROM shell_history", ())
      .await?;
    let row = rows.next().await?.expect("expected row");
    let count: i64 = row.get(0)?;
    assert_eq!(count, 1, "reset should not delete shell_history");

    Ok(())
  }

  #[tokio::test]
  async fn test_insert_invocation() -> Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    let invocation = Invocation {
      command: "ls -la".to_string(),
      expanded_command: "ls -la".to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/tmp".to_string()),
      workspace: None,
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(123),
      end_unix_timestamp: Some(124),
      session_id: 42,
    };

    let inserted = insert_invocation(&db.conn, &invocation).await?;
    assert!(inserted);

    let mut rows = db
      .conn
      .query("SELECT COUNT(*) FROM shell_history", ())
      .await?;
    let row = rows.next().await?.expect("expected row");
    let count = row.get::<i64>(0)?;
    assert_eq!(count, 1);
    Ok(())
  }

  #[tokio::test]
  async fn test_open_db_creates_parent_and_schema() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let db_path = temp.path().join("nested/dir/zage.db");
    let parent = db_path.parent().expect("expected parent path");
    assert!(!parent.exists());

    let db = open_db(&db_path).await?;
    assert!(parent.exists());
    assert!(db_path.exists());

    let mut rows = db
      .conn
      .query(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='shell_history'",
        (),
      )
      .await?;
    let row = rows.next().await?.expect("expected row");
    let count = row.get::<i64>(0)?;
    assert_eq!(count, 1);
    Ok(())
  }

  // workspace_json is intentionally not persisted.

  #[tokio::test]
  async fn get_recent_invocations_orders_by_end_timestamp() -> Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    let inv_a = Invocation {
      command: "cmd-a".to_string(),
      expanded_command: "cmd-a".to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/tmp".to_string()),
      workspace: None,
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(10),
      end_unix_timestamp: Some(11),
      session_id: 1,
    };
    let inv_b = Invocation {
      command: "cmd-b".to_string(),
      expanded_command: "cmd-b".to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/tmp".to_string()),
      workspace: None,
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(10),
      end_unix_timestamp: Some(12),
      session_id: 1,
    };

    assert!(insert_invocation(&db.conn, &inv_a).await?);
    assert!(insert_invocation(&db.conn, &inv_b).await?);

    let recent = get_recent_invocations(&db.conn, None, 2).await?;
    assert_eq!(recent.len(), 2);
    assert_eq!(recent[0].command, "cmd-a");
    assert_eq!(recent[1].command, "cmd-b");
    Ok(())
  }

  #[tokio::test]
  async fn transition_stats_use_previous_even_with_same_timestamp() -> Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    let inv_a = Invocation {
      command: "cmd-a".to_string(),
      expanded_command: "cmd-a".to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/tmp".to_string()),
      workspace: None,
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(10),
      end_unix_timestamp: Some(10),
      session_id: 1,
    };
    let inv_b = Invocation {
      command: "cmd-b".to_string(),
      expanded_command: "cmd-b".to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/tmp".to_string()),
      workspace: None,
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(10),
      end_unix_timestamp: Some(10),
      session_id: 1,
    };

    assert!(insert_invocation(&db.conn, &inv_a).await?);
    update_stats_for_invocation(&db.conn, &inv_a).await?;
    assert!(insert_invocation(&db.conn, &inv_b).await?);
    update_stats_for_invocation(&db.conn, &inv_b).await?;

    let mut rows = db
      .conn
      .query(
        "SELECT freq FROM transition_stats
         WHERE prev_command = ? AND next_command = ? AND prev_exit_status IS ?",
        ("cmd-a".to_string(), "cmd-b".to_string(), Some(0i64)),
      )
      .await?;
    let row = rows.next().await?;
    assert!(row.is_some());
    Ok(())
  }

  #[tokio::test]
  async fn upsert_online_feedback_validates_required_fields() -> Result<()> {
    let tmp = tempfile::NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    let err = upsert_online_feedback(
      &db.conn,
      OnlineFeedbackEvent {
        shown_id: "".to_string(),
        shown_at: 1,
        cwd: Some("/tmp".to_string()),
        suggestion: "git status".to_string(),
        accepted_command: None,
        accepted_at: None,
        outcome: None,
      },
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("shown_id"));

    let err = upsert_online_feedback(
      &db.conn,
      OnlineFeedbackEvent {
        shown_id: "s1".to_string(),
        shown_at: 0,
        cwd: Some("/tmp".to_string()),
        suggestion: "git status".to_string(),
        accepted_command: None,
        accepted_at: None,
        outcome: None,
      },
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("shown_at"));

    let err = upsert_online_feedback(
      &db.conn,
      OnlineFeedbackEvent {
        shown_id: "s1".to_string(),
        shown_at: 1,
        cwd: Some("/tmp".to_string()),
        suggestion: "".to_string(),
        accepted_command: None,
        accepted_at: None,
        outcome: None,
      },
    )
    .await
    .unwrap_err();
    assert!(err.to_string().contains("suggestion"));

    Ok(())
  }
}
