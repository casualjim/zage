use std::env;
use std::path::Path;
use libsql::{Builder, Connection, Database};

use crate::{Result, shell_history::Invocation};
use crate::repo::find_repo_root;
use crate::tokenize::{extract_command_parts, normalize_token, tokenize_index};
use serde_json;
use tracing::info;

pub struct Db {
  pub db: Database,
  pub conn: Connection,
}

pub async fn open_db<P: AsRef<Path>>(db_path: P) -> Result<Db> {
  let url = env::var("TURSO_DATABASE_URL").ok().or_else(|| env::var("LIBSQL_URL").ok());
  let token = env::var("TURSO_AUTH_TOKEN")
    .ok()
    .or_else(|| env::var("LIBSQL_AUTH_TOKEN").ok());
  let replica_path = env::var("TURSO_LOCAL_REPLICA_PATH").ok();

  let db = if let Some(url) = url {
    let token = token.unwrap_or_default();
    if let Some(replica_path) = replica_path {
      Builder::new_remote_replica(replica_path, url, token).build().await?
    } else {
      Builder::new_remote(url, token).build().await?
    }
  } else {
    let path = db_path.as_ref().to_string_lossy().to_string();
    Builder::new_local(path).build().await?
  };

  let conn = db.connect()?;
  Ok(Db { db, conn })
}

pub async fn init(conn: &Connection) -> Result<()> {
  execute_batch(conn, include_str!("db/schema-v0.sql")).await?;
  Ok(())
}

pub async fn insert_invocation(conn: &Connection, invocation: &Invocation) -> Result<bool> {
  let id = uuid::Uuid::now_v7().to_string();
  let changed = conn
    .execute(
      "INSERT OR IGNORE INTO shell_history (id, command, shellname, working_directory, hostname, username, exit_status, start_unix_timestamp, end_unix_timestamp, session_id)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
      (
        id,
        invocation.command.clone(),
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

  for invocation in invocations {
    processed += 1;
    let did_insert = insert_invocation(conn, &invocation).await?;
    if !did_insert {
      continue;
    }
    inserted += 1;
    if processed % progress_interval == 0 {
      info!(
        "Imported {} history entries ({} inserted)",
        processed, inserted
      );
    }
  }
  conn.execute("COMMIT", ()).await?;
  info!("Imported {} history entries ({} inserted)", processed, inserted);
  Ok(())
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
    invs.push(Invocation {
      command: row.get::<String>(1)?,
      shellname: row.get::<String>(2)?,
      working_directory: row.get::<Option<String>>(3)?,
      hostname: row.get::<Option<String>>(4)?,
      username: row.get::<Option<String>>(5)?,
      exit_status: row.get::<Option<i64>>(6)?,
      start_unix_timestamp: row.get::<Option<i64>>(7)?,
      end_unix_timestamp: row.get::<Option<i64>>(8)?,
      session_id: row.get::<i64>(9)?,
    });
  }
  invs.reverse();
  Ok(invs)
}

pub async fn update_stats_for_invocation(
  conn: &Connection,
  invocation: &Invocation,
) -> Result<()> {
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
  let tokens = tokenize_index(&invocation.shellname, &invocation.command);
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
        (invocation.command.clone(), now),
      )
      .await?;

    conn
      .execute(
        "INSERT INTO repo_command_stats (repo_root, command, freq, last_seen)
         VALUES (?, ?, 1, ?)
         ON CONFLICT(repo_root, command) DO UPDATE SET
           freq = freq + 1,
           last_seen = MAX(last_seen, excluded.last_seen)",
        (repo_root.clone(), invocation.command.clone(), now),
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
          invocation.command.clone(),
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
      let prev_status = prev.exit_status;
      let prev_repo_root = prev
        .working_directory
        .as_deref()
        .and_then(find_repo_root)
        .unwrap_or_default();

      conn
        .execute(
          "INSERT INTO transition_stats_v2 (prev_command, prev_exit_status, next_command, freq, last_seen)
           VALUES (?, ?, ?, 1, ?)
           ON CONFLICT(prev_command, prev_exit_status, next_command) DO UPDATE SET
             freq = freq + 1,
             last_seen = MAX(last_seen, excluded.last_seen)",
          (prev.command.clone(), prev_status, invocation.command.clone(), now),
        )
        .await?;

      conn
        .execute(
          "INSERT INTO transition_stats_v2 (prev_command, prev_exit_status, next_command, freq, last_seen)
           VALUES (?, NULL, ?, 1, ?)
           ON CONFLICT(prev_command, prev_exit_status, next_command) DO UPDATE SET
             freq = freq + 1,
             last_seen = MAX(last_seen, excluded.last_seen)",
          (prev.command.clone(), invocation.command.clone(), now),
        )
        .await?;

      conn
        .execute(
          "INSERT INTO repo_transition_stats_v2 (repo_root, prev_command, prev_exit_status, next_command, freq, last_seen)
           VALUES (?, ?, ?, ?, 1, ?)
           ON CONFLICT(repo_root, prev_command, prev_exit_status, next_command) DO UPDATE SET
             freq = freq + 1,
             last_seen = MAX(last_seen, excluded.last_seen)",
          (
            prev_repo_root.clone(),
            prev.command.clone(),
            prev_status,
            invocation.command.clone(),
            now,
          ),
        )
        .await?;

      conn
        .execute(
          "INSERT INTO repo_transition_stats_v2 (repo_root, prev_command, prev_exit_status, next_command, freq, last_seen)
           VALUES (?, ?, NULL, ?, 1, ?)
           ON CONFLICT(repo_root, prev_command, prev_exit_status, next_command) DO UPDATE SET
             freq = freq + 1,
             last_seen = MAX(last_seen, excluded.last_seen)",
          (
            prev_repo_root,
            prev.command.clone(),
            invocation.command.clone(),
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
        (invocation.command.clone(), raw_json, norm_json, now),
      )
      .await?;

    if let Some(parts) = extract_command_parts(&invocation.command, &tokens) {
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

    let mut rows = db.conn.query("SELECT COUNT(*) FROM shell_history", ()).await?;
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
      "SELECT command, exit_status, working_directory FROM shell_history
       WHERE session_id = ? AND COALESCE(start_unix_timestamp, 0) < ?
       ORDER BY COALESCE(start_unix_timestamp, 0) DESC, id DESC
       LIMIT 1",
      (session_id, before_ts),
    )
    .await?;
  if let Some(row) = rows.next().await? {
    Ok(Some(PrevInvocation {
      command: row.get(0)?,
      exit_status: row.get::<Option<i64>>(1)?,
      working_directory: row.get::<Option<String>>(2)?,
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
      shellname: "zsh".to_string(),
      working_directory: Some("/tmp".to_string()),
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
}
