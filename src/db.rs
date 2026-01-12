use libsql::{
  Builder, Cipher, Connection, Database, EncryptionConfig, EncryptionContext, EncryptionKey,
};
use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::config::{DbConfig, DbKind};
use crate::repo::find_repo_root;
use crate::tokenize::{extract_command_parts, normalize_token, tokenize_index};
use crate::{Result, shell_history::Invocation};
use serde_json;
use tracing::info;

pub struct Db {
  pub db: Database,
  pub conn: Connection,
}

pub async fn open_db<P: AsRef<Path>>(db_path: P) -> Result<Db> {
  let path_ref = db_path.as_ref();
  if let Some(parent) = path_ref.parent() {
    fs::create_dir_all(parent)?;
  }
  let path = path_ref.to_string_lossy().to_string();
  let db = Builder::new_local(path).build().await?;

  let conn = db.connect()?;
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
  ensure_shell_history_columns(conn).await?;
  Ok(())
}

async fn ensure_shell_history_columns(conn: &Connection) -> Result<()> {
  let mut rows = conn.query("PRAGMA table_info(shell_history)", ()).await?;
  let mut columns = HashSet::new();
  while let Some(row) = rows.next().await? {
    let name: String = row.get(1)?;
    columns.insert(name);
  }
  if !columns.contains("workspace_json") {
    conn
      .execute(
        "ALTER TABLE shell_history ADD COLUMN workspace_json TEXT",
        (),
      )
      .await?;
  }
  Ok(())
}

pub async fn insert_invocation(conn: &Connection, invocation: &Invocation) -> Result<bool> {
  let id = uuid::Uuid::now_v7().to_string();
  let workspace_json = invocation
    .workspace
    .as_ref()
    .map(serde_json::to_string)
    .transpose()?;
  let changed = conn
    .execute(
      "INSERT OR IGNORE INTO shell_history (id, command, expanded_command, shellname, working_directory, workspace_json, hostname, username, exit_status, start_unix_timestamp, end_unix_timestamp, session_id)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      (
        id,
        invocation.command.clone(),
        invocation.expanded_command.clone(),
        invocation.shellname.clone(),
        invocation.working_directory.clone(),
        workspace_json,
        invocation.hostname.clone(),
        invocation.username.clone(),
        invocation.exit_status,
        invocation.start_unix_timestamp,
        invocation.end_unix_timestamp,
        invocation.session_id,
      ),
    )
    .await?;
  Ok(changed > 0)
}

pub async fn import_history<I>(conn: &Connection, invocations: I) -> Result<()>
where
  I: IntoIterator<Item = Invocation>,
{
  let mut processed: usize = 0;
  let mut inserted: usize = 0;
  let progress_interval = 10_000usize;

  conn.execute("BEGIN", ()).await?;

  for mut invocation in invocations {
    if invocation.expanded_command.is_empty() {
      invocation.expanded_command = invocation.command.clone();
    }
    processed += 1;
    let did_insert = insert_invocation(conn, &invocation).await?;
    if !did_insert {
      continue;
    }
    inserted += 1;
    if processed.is_multiple_of(progress_interval) {
      info!(
        "Imported {} history entries ({} inserted)",
        processed, inserted
      );
    }
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

pub async fn get_recent_invocations(conn: &Connection, limit: usize) -> Result<Vec<Invocation>> {
  let mut rows = conn
    .query(
      include_str!("db/get-recent-invocations.sql"),
      libsql::params![limit as i64],
    )
    .await?;

  let mut invs = Vec::new();
  while let Some(row) = rows.next().await? {
    let workspace_json = row.get::<Option<String>>(5)?;
    let workspace = match workspace_json {
      Some(raw) => Some(serde_json::from_str(&raw)?),
      None => None,
    };
    invs.push(Invocation {
      command: row.get::<String>(1)?,
      expanded_command: row.get::<String>(2)?,
      shellname: row.get::<String>(3)?,
      working_directory: row.get::<Option<String>>(4)?,
      workspace,
      hostname: row.get::<Option<String>>(6)?,
      username: row.get::<Option<String>>(7)?,
      exit_status: row.get::<Option<i64>>(8)?,
      start_unix_timestamp: row.get::<Option<i64>>(9)?,
      end_unix_timestamp: row.get::<Option<i64>>(10)?,
      session_id: row.get::<i64>(11)?,
    });
  }
  invs.reverse();
  Ok(invs)
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

  let repo_root = invocation
    .working_directory
    .as_deref()
    .and_then(find_repo_root)
    .unwrap_or_default();
  let stats_command = if invocation.expanded_command.is_empty() {
    invocation.command.as_str()
  } else {
    invocation.expanded_command.as_str()
  };
  let tokens = tokenize_index(&invocation.shellname, stats_command);
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
        "INSERT INTO repo_command_stats (repo_root, command, freq, last_seen)
         VALUES (?, ?, 1, ?)
         ON CONFLICT(repo_root, command) DO UPDATE SET
           freq = freq + 1,
           last_seen = MAX(last_seen, excluded.last_seen)",
        (repo_root.clone(), stats_command.to_string(), now),
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

    if let Some(prev) =
      previous_invocation_for_session(conn, invocation.session_id, now).await?
    {
      let prev_command = if prev.expanded_command.is_empty() {
        prev.command.as_str()
      } else {
        prev.expanded_command.as_str()
      };
      let prev_status = prev.exit_status;
      let prev_repo_root = prev
        .working_directory
        .as_deref()
        .and_then(find_repo_root)
        .unwrap_or_default();

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

      conn
        .execute(
        "INSERT INTO repo_transition_stats (repo_root, prev_command, prev_exit_status, next_command, freq, last_seen)
           VALUES (?, ?, ?, ?, 1, ?)
           ON CONFLICT(repo_root, prev_command, prev_exit_status, next_command) DO UPDATE SET
             freq = freq + 1,
             last_seen = MAX(last_seen, excluded.last_seen)",
          (
            prev_repo_root.clone(),
            prev_command.to_string(),
            prev_status,
            stats_command.to_string(),
            now,
          ),
        )
        .await?;

      conn
        .execute(
        "INSERT INTO repo_transition_stats (repo_root, prev_command, prev_exit_status, next_command, freq, last_seen)
           VALUES (?, ?, NULL, ?, 1, ?)
           ON CONFLICT(repo_root, prev_command, prev_exit_status, next_command) DO UPDATE SET
             freq = freq + 1,
             last_seen = MAX(last_seen, excluded.last_seen)",
          (
            prev_repo_root,
            prev_command.to_string(),
            stats_command.to_string(),
            now,
          ),
        )
        .await?;
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

    if let Some(parts) = extract_command_parts(stats_command, &tokens) {
      let mut flags = parts.flags;
      flags.sort();
      let flags_json = serde_json::to_string(&flags)?;
      for flag in &flags {
        let flag_norm = normalize_token(flag);
        conn
          .execute(
            "INSERT INTO flag_stats (repo_root, command_head, flag_raw, flag_norm, freq, last_seen)
             VALUES (?, ?, ?, ?, 1, ?)
             ON CONFLICT(repo_root, command_head, flag_raw) DO UPDATE SET
               freq = freq + 1,
               last_seen = MAX(last_seen, excluded.last_seen)",
            (
              repo_root.clone(),
              parts.head.clone(),
              flag.clone(),
              flag_norm,
              now,
            ),
          )
          .await?;
      }
      for env in &parts.env {
        let env_key = env
          .raw
          .split('=')
          .next()
          .unwrap_or_default()
          .to_string();
        conn
          .execute(
            "INSERT INTO env_stats (repo_root, command_head, env_key, env_raw, env_norm, freq, last_seen)
             VALUES (?, ?, ?, ?, ?, 1, ?)
             ON CONFLICT(repo_root, command_head, env_raw) DO UPDATE SET
               freq = freq + 1,
               last_seen = MAX(last_seen, excluded.last_seen)",
            (
              repo_root.clone(),
              parts.head.clone(),
              env_key,
              env.raw.clone(),
              env.normalized.clone(),
              now,
            ),
          )
          .await?;
      }
      for (idx, arg) in parts.args.iter().enumerate() {
        conn
          .execute(
            "INSERT INTO arg_stats (repo_root, command_head, flags_json, arg_index, arg_raw, arg_norm, freq, last_seen)
             VALUES (?, ?, ?, ?, ?, ?, 1, ?)
             ON CONFLICT(repo_root, command_head, flags_json, arg_index, arg_raw) DO UPDATE SET
               freq = freq + 1,
               last_seen = MAX(last_seen, excluded.last_seen)",
            (
              repo_root.clone(),
              parts.head.clone(),
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
            "INSERT INTO arg_stats_any (repo_root, command_head, flags_json, arg_raw, arg_norm, freq, last_seen)
             VALUES (?, ?, ?, ?, ?, 1, ?)
             ON CONFLICT(repo_root, command_head, flags_json, arg_raw) DO UPDATE SET
               freq = freq + 1,
               last_seen = MAX(last_seen, excluded.last_seen)",
            (
              repo_root.clone(),
              parts.head.clone(),
              flags_json.clone(),
              arg.raw.clone(),
              arg.normalized.clone(),
              now,
            ),
          )
          .await?;
      }
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
  exit_status: Option<i64>,
  working_directory: Option<String>,
}

async fn previous_invocation_for_session(
  conn: &Connection,
  session_id: i64,
  before_ts: i64,
) -> Result<Option<PrevInvocation>> {
  let mut rows = conn
    .query(
      "SELECT command, expanded_command, exit_status, working_directory FROM shell_history
       WHERE session_id = ? AND COALESCE(start_unix_timestamp, 0) < ?
       ORDER BY COALESCE(start_unix_timestamp, 0) DESC, id DESC
       LIMIT 1",
      (session_id, before_ts),
    )
    .await?;
  if let Some(row) = rows.next().await? {
    Ok(Some(PrevInvocation {
      command: row.get(0)?,
      expanded_command: row.get(1)?,
      exit_status: row.get::<Option<i64>>(2)?,
      working_directory: row.get::<Option<String>>(3)?,
    }))
  } else {
    Ok(None)
  }
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
}
