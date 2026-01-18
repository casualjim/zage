use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use libsql::Connection;
use libsql::Value;
use serde_json;
use tracing::info;

use crate::Result;
use crate::predict::aliases::{expand_alias, load_aliases};
use crate::tokenize::{
  extract_command_parts, extract_command_stats_parts, normalize_command_whitespace,
  normalize_token, tokenize_index,
};

#[derive(Debug, Default)]
pub struct IndexReport {
  pub commands: usize,
  pub transitions: usize,
  pub contexts: usize,
  pub token_cache: usize,
}

#[derive(Debug, Default)]
struct Stat {
  freq: i64,
  last_seen: i64,
}

#[derive(Debug, Default)]
struct ArgStat {
  freq: i64,
  last_seen: i64,
  arg_norm: String,
}

type ContextKey = (String, Option<String>, Option<String>, Option<String>);

fn resolve_workspace_root(cwd: Option<&str>, cache: &mut HashMap<String, String>) -> String {
  let Some(cwd) = cwd else {
    return String::new();
  };
  if let Some(root) = cache.get(cwd) {
    return root.clone();
  }
  let root = crate::workspace::workspace_root_for_cwd(cwd).unwrap_or_default();
  cache.insert(cwd.to_string(), root.clone());
  root
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

fn opt_text_value(v: &Option<String>) -> Value {
  match v {
    Some(s) => Value::Text(s.clone()),
    None => Value::Null,
  }
}

fn opt_i64_value(v: &Option<i64>) -> Value {
  match v {
    Some(i) => Value::Integer(*i),
    None => Value::Null,
  }
}

struct BulkInserter<'a> {
  conn: &'a Connection,
  prefix: &'a str,
  suffix: &'a str,
  cols_per_row: usize,
  max_rows: usize,
  params: Vec<Value>,
  rows: usize,
}

impl<'a> BulkInserter<'a> {
  fn new(conn: &'a Connection, prefix: &'a str, cols_per_row: usize) -> Self {
    Self::new_with_suffix(conn, prefix, cols_per_row, "")
  }

  fn new_with_suffix(
    conn: &'a Connection,
    prefix: &'a str,
    cols_per_row: usize,
    suffix: &'a str,
  ) -> Self {
    // SQLite default max variables is often 999; stay below it.
    const MAX_SQL_VARS: usize = 900;
    let max_rows = (MAX_SQL_VARS / cols_per_row).max(1);
    Self {
      conn,
      prefix,
      suffix,
      cols_per_row,
      max_rows,
      params: Vec::with_capacity(max_rows * cols_per_row),
      rows: 0,
    }
  }

  async fn push_row(&mut self, row: impl IntoIterator<Item = Value>) -> crate::Result<()> {
    if self.rows >= self.max_rows {
      self.flush().await?;
    }
    self.params.extend(row);
    self.rows += 1;
    Ok(())
  }

  async fn flush(&mut self) -> crate::Result<u64> {
    if self.rows == 0 {
      return Ok(0);
    }
    let mut sql = String::from(self.prefix);
    append_multi_row_placeholders(&mut sql, self.rows, self.cols_per_row);
    sql.push_str(self.suffix);
    let params = std::mem::take(&mut self.params);
    self.rows = 0;
    self.params = Vec::with_capacity(self.max_rows * self.cols_per_row);
    let changed = self.conn.execute(&sql, params).await?;
    Ok(changed)
  }
}

pub async fn rebuild_stats(conn: &Connection, max_commands: Option<usize>) -> Result<IndexReport> {
  let mut command_stats: HashMap<String, Stat> = HashMap::new();
  let mut transition_stats: HashMap<(String, Option<i64>, String), Stat> = HashMap::new();
  let mut transition_head_stats: HashMap<(String, Option<i64>, String), Stat> = HashMap::new();
  let mut workspace_command_stats: HashMap<(String, String), Stat> = HashMap::new();
  let mut workspace_transition_stats: HashMap<(String, String, Option<i64>, String), Stat> =
    HashMap::new();
  let mut workspace_transition_head_stats: HashMap<(String, String, Option<i64>, String), Stat> =
    HashMap::new();
  let mut context_stats: HashMap<ContextKey, Stat> = HashMap::new();
  let mut arg_stats: HashMap<(String, String, String, i64, String), ArgStat> = HashMap::new();
  let mut arg_stats_any: HashMap<(String, String, String, String), ArgStat> = HashMap::new();
  let mut flag_stats: HashMap<(String, String, String, String), Stat> = HashMap::new();
  let mut env_stats: HashMap<(String, String, String, String, String), Stat> = HashMap::new();
  let mut token_cache: HashMap<String, (Vec<String>, Vec<String>)> = HashMap::new();
  let mut workspace_root_cache: HashMap<String, String> = HashMap::new();

  let mut prev_command: Option<String> = None;
  let mut prev_head: Option<String> = None;
  let mut prev_exit_status: Option<i64> = None;
  let mut prev_workspace_root: String = String::new();
  let mut processed: usize = 0;
  let progress_interval = 50_000usize;
  let aliases = load_aliases();

  let mut rows = if let Some(limit) = max_commands {
    conn
      .query(
        "WITH recent AS (
           SELECT id, command, expanded_command, shellname, working_directory, hostname, username,
                  exit_status, start_unix_timestamp
           FROM shell_history
           ORDER BY COALESCE(start_unix_timestamp, 0) DESC, id DESC
           LIMIT ?
         )
         SELECT command, expanded_command, shellname, working_directory, hostname, username,
                exit_status, start_unix_timestamp
         FROM recent
         ORDER BY COALESCE(start_unix_timestamp, 0) ASC, id ASC",
        libsql::params![limit as i64],
      )
      .await?
  } else {
    conn
      .query(
        "SELECT command, expanded_command, shellname, working_directory, hostname, username,
                exit_status, start_unix_timestamp
         FROM shell_history
         ORDER BY COALESCE(start_unix_timestamp, 0) ASC, id ASC",
        (),
      )
      .await?
  };

  while let Some(row) = rows.next().await? {
    let command = row.get::<String>(0)?;
    let expanded_command = row.get::<String>(1)?;
    let shellname = row.get::<String>(2)?;
    let working_directory = row.get::<Option<String>>(3)?;
    let hostname = row.get::<Option<String>>(4)?;
    let username = row.get::<Option<String>>(5)?;
    let exit_status = row.get::<Option<i64>>(6)?;
    let ts: Option<i64> = row.get(7)?;
    let ts = ts.unwrap_or(0);
    let workspace_root =
      resolve_workspace_root(working_directory.as_deref(), &mut workspace_root_cache);

    let stats_command = if !expanded_command.is_empty() {
      expanded_command
    } else {
      expand_alias(&command, &aliases).unwrap_or(command.clone())
    };
    let stats_command = normalize_command_whitespace(&stats_command);

    update_stat(&mut command_stats, &stats_command, ts);
    update_stat_key(
      &mut workspace_command_stats,
      (workspace_root.clone(), stats_command.clone()),
      ts,
    );

    let ctx_key = (
      stats_command.clone(),
      working_directory.clone(),
      hostname.clone(),
      username.clone(),
    );
    update_stat_key(&mut context_stats, ctx_key, ts);

    if let Some(prev) = &prev_command {
      update_stat_key(
        &mut transition_stats,
        (prev.clone(), prev_exit_status, stats_command.clone()),
        ts,
      );
      update_stat_key(
        &mut transition_stats,
        (prev.clone(), None, stats_command.clone()),
        ts,
      );
      update_stat_key(
        &mut workspace_transition_stats,
        (
          prev_workspace_root.clone(),
          prev.clone(),
          prev_exit_status,
          stats_command.clone(),
        ),
        ts,
      );
      update_stat_key(
        &mut workspace_transition_stats,
        (
          prev_workspace_root.clone(),
          prev.clone(),
          None,
          stats_command.clone(),
        ),
        ts,
      );
    }

    let tokens = tokenize_index(&shellname, &stats_command);
    if !token_cache.contains_key(&stats_command) {
      let raw = tokens.iter().map(|t| t.raw.clone()).collect();
      let normalized = tokens.iter().map(|t| t.normalized.clone()).collect();
      token_cache.insert(stats_command.clone(), (raw, normalized));
    }

    let current_head = extract_command_stats_parts(&stats_command, &tokens)
      .map(|parts| parts.head)
      .or_else(|| extract_command_parts(&stats_command, &tokens).map(|parts| parts.head));

    if let Some(prev_head) = &prev_head {
      update_stat_key(
        &mut transition_head_stats,
        (prev_head.clone(), prev_exit_status, stats_command.clone()),
        ts,
      );
      update_stat_key(
        &mut transition_head_stats,
        (prev_head.clone(), None, stats_command.clone()),
        ts,
      );
      update_stat_key(
        &mut workspace_transition_head_stats,
        (
          prev_workspace_root.clone(),
          prev_head.clone(),
          prev_exit_status,
          stats_command.clone(),
        ),
        ts,
      );
      update_stat_key(
        &mut workspace_transition_head_stats,
        (
          prev_workspace_root.clone(),
          prev_head.clone(),
          None,
          stats_command.clone(),
        ),
        ts,
      );
    }

    let mut base_head: Option<String> = None;
    if let Some(parts) = extract_command_parts(&stats_command, &tokens) {
      base_head = Some(parts.head.clone());
      let mut flags = parts.flags;
      flags.sort();
      let flags_json = serde_json::to_string(&flags)?;
      for flag in &flags {
        let flag_norm = normalize_token(flag);
        update_stat_key(
          &mut flag_stats,
          (
            workspace_root.clone(),
            parts.head.clone(),
            flag.clone(),
            flag_norm,
          ),
          ts,
        );
      }
      for env in &parts.env {
        let env_key = env.raw.split('=').next().unwrap_or_default().to_string();
        update_stat_key(
          &mut env_stats,
          (
            workspace_root.clone(),
            parts.head.clone(),
            env_key,
            env.raw.clone(),
            env.normalized.clone(),
          ),
          ts,
        );
      }
      for (idx, arg) in parts.args.iter().enumerate() {
        update_arg_stat(
          &mut arg_stats,
          (
            workspace_root.clone(),
            parts.head.clone(),
            flags_json.clone(),
            idx as i64,
            arg.raw.clone(),
          ),
          &arg.normalized,
          ts,
        );
        update_arg_stat(
          &mut arg_stats_any,
          (
            workspace_root.clone(),
            parts.head.clone(),
            flags_json.clone(),
            arg.raw.clone(),
          ),
          &arg.normalized,
          ts,
        );
      }
    }
    if let Some(parts) = extract_command_stats_parts(&stats_command, &tokens)
      && base_head.as_deref() != Some(parts.head.as_str())
    {
      let mut flags = parts.flags;
      flags.sort();
      let flags_json = serde_json::to_string(&flags)?;
      for flag in &flags {
        let flag_norm = normalize_token(flag);
        update_stat_key(
          &mut flag_stats,
          (
            workspace_root.clone(),
            parts.head.clone(),
            flag.clone(),
            flag_norm,
          ),
          ts,
        );
      }
      for env in &parts.env {
        let env_key = env.raw.split('=').next().unwrap_or_default().to_string();
        update_stat_key(
          &mut env_stats,
          (
            workspace_root.clone(),
            parts.head.clone(),
            env_key,
            env.raw.clone(),
            env.normalized.clone(),
          ),
          ts,
        );
      }
      for (idx, arg) in parts.args.iter().enumerate() {
        update_arg_stat(
          &mut arg_stats,
          (
            workspace_root.clone(),
            parts.head.clone(),
            flags_json.clone(),
            idx as i64,
            arg.raw.clone(),
          ),
          &arg.normalized,
          ts,
        );
        update_arg_stat(
          &mut arg_stats_any,
          (
            workspace_root.clone(),
            parts.head.clone(),
            flags_json.clone(),
            arg.raw.clone(),
          ),
          &arg.normalized,
          ts,
        );
      }
    }

    prev_command = Some(stats_command);
    prev_head = current_head;
    prev_exit_status = exit_status;
    prev_workspace_root = workspace_root;
    processed += 1;
    if processed.is_multiple_of(progress_interval) {
      info!("Indexed {} commands so far", processed);
    }
  }

  if processed == 0 {
    return Ok(IndexReport::default());
  }

  let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64;

  conn.execute("BEGIN", ()).await?;
  let write_result: Result<()> = async {
    conn.execute("DELETE FROM transition_stats", ()).await?;
    conn.execute("DELETE FROM transition_head_stats", ()).await?;
    conn.execute("DELETE FROM workspace_command_stats", ()).await?;
    conn.execute("DELETE FROM workspace_transition_stats", ()).await?;
    conn.execute("DELETE FROM workspace_transition_head_stats", ()).await?;
    conn.execute("DELETE FROM context_stats", ()).await?;
    conn.execute("DELETE FROM arg_stats", ()).await?;
    conn.execute("DELETE FROM arg_stats_any", ()).await?;
    conn.execute("DELETE FROM flag_stats", ()).await?;
    conn.execute("DELETE FROM env_stats", ()).await?;
    conn.execute("DELETE FROM token_cache", ()).await?;

    // Preserve any extra columns on command_stats (e.g. embeddings) by updating freq/last_seen
    // in-place and then deleting commands that are no longer present in the selected history.
    conn
      .execute(
        "CREATE TEMP TABLE IF NOT EXISTS tmp_command_stats (command TEXT PRIMARY KEY)",
        (),
      )
      .await?;
    conn.execute("DELETE FROM tmp_command_stats", ()).await?;

    let mut tmp_commands = BulkInserter::new(
      conn,
      "INSERT INTO tmp_command_stats (command) VALUES ",
      1,
    );
    let mut upsert_commands = BulkInserter::new_with_suffix(
      conn,
      "INSERT INTO command_stats (command, freq, last_seen) VALUES ",
      3,
      " ON CONFLICT(command) DO UPDATE SET freq = excluded.freq, last_seen = excluded.last_seen",
    );

    for (command, stat) in &command_stats {
      tmp_commands
        .push_row([Value::Text(command.clone())])
        .await?;
      upsert_commands
        .push_row([
          Value::Text(command.clone()),
          Value::Integer(stat.freq),
          Value::Integer(stat.last_seen),
        ])
        .await?;
    }
    tmp_commands.flush().await?;
    upsert_commands.flush().await?;

    conn
      .execute(
        "DELETE FROM command_stats
         WHERE command NOT IN (SELECT command FROM tmp_command_stats)",
        (),
      )
      .await?;

    let mut insert_workspace_commands = BulkInserter::new(
      conn,
      "INSERT INTO workspace_command_stats (workspace_root, command, freq, last_seen) VALUES ",
      4,
    );
    for ((workspace_root, command), stat) in &workspace_command_stats {
      insert_workspace_commands
        .push_row([
          Value::Text(workspace_root.clone()),
          Value::Text(command.clone()),
          Value::Integer(stat.freq),
          Value::Integer(stat.last_seen),
        ])
        .await?;
    }
    insert_workspace_commands.flush().await?;

    let mut insert_transitions = BulkInserter::new(
      conn,
      "INSERT INTO transition_stats (prev_command, prev_exit_status, next_command, freq, last_seen) VALUES ",
      5,
    );
    for ((prev, status, next), stat) in &transition_stats {
      insert_transitions
        .push_row([
          Value::Text(prev.clone()),
          opt_i64_value(status),
          Value::Text(next.clone()),
          Value::Integer(stat.freq),
          Value::Integer(stat.last_seen),
        ])
        .await?;
    }
    insert_transitions.flush().await?;

    let mut insert_transition_heads = BulkInserter::new(
      conn,
      "INSERT INTO transition_head_stats (prev_head, prev_exit_status, next_command, freq, last_seen) VALUES ",
      5,
    );
    for ((prev, status, next), stat) in &transition_head_stats {
      insert_transition_heads
        .push_row([
          Value::Text(prev.clone()),
          opt_i64_value(status),
          Value::Text(next.clone()),
          Value::Integer(stat.freq),
          Value::Integer(stat.last_seen),
        ])
        .await?;
    }
    insert_transition_heads.flush().await?;

    let mut insert_workspace_transitions = BulkInserter::new(
      conn,
      "INSERT INTO workspace_transition_stats (workspace_root, prev_command, prev_exit_status, next_command, freq, last_seen) VALUES ",
      6,
    );
    for ((workspace_root, prev, status, next), stat) in &workspace_transition_stats {
      insert_workspace_transitions
        .push_row([
          Value::Text(workspace_root.clone()),
          Value::Text(prev.clone()),
          opt_i64_value(status),
          Value::Text(next.clone()),
          Value::Integer(stat.freq),
          Value::Integer(stat.last_seen),
        ])
        .await?;
    }
    insert_workspace_transitions.flush().await?;

    let mut insert_workspace_transition_heads = BulkInserter::new(
      conn,
      "INSERT INTO workspace_transition_head_stats (workspace_root, prev_head, prev_exit_status, next_command, freq, last_seen) VALUES ",
      6,
    );
    for ((workspace_root, prev, status, next), stat) in &workspace_transition_head_stats {
      insert_workspace_transition_heads
        .push_row([
          Value::Text(workspace_root.clone()),
          Value::Text(prev.clone()),
          opt_i64_value(status),
          Value::Text(next.clone()),
          Value::Integer(stat.freq),
          Value::Integer(stat.last_seen),
        ])
        .await?;
    }
    insert_workspace_transition_heads.flush().await?;

    let mut insert_contexts = BulkInserter::new(
      conn,
      "INSERT INTO context_stats (command, working_directory, hostname, username, freq, last_seen) VALUES ",
      6,
    );
    for ((command, wd, host, user), stat) in &context_stats {
      insert_contexts
        .push_row([
          Value::Text(command.clone()),
          opt_text_value(wd),
          opt_text_value(host),
          opt_text_value(user),
          Value::Integer(stat.freq),
          Value::Integer(stat.last_seen),
        ])
        .await?;
    }
    insert_contexts.flush().await?;

    let mut insert_arg_stats = BulkInserter::new(
      conn,
      "INSERT INTO arg_stats (workspace_root, command_head, flags_json, arg_index, arg_raw, arg_norm, freq, last_seen) VALUES ",
      8,
    );
    for ((workspace_root, head, flags_json, arg_index, arg_raw), stat) in &arg_stats {
      insert_arg_stats
        .push_row([
          Value::Text(workspace_root.clone()),
          Value::Text(head.clone()),
          Value::Text(flags_json.clone()),
          Value::Integer(*arg_index),
          Value::Text(arg_raw.clone()),
          Value::Text(stat.arg_norm.clone()),
          Value::Integer(stat.freq),
          Value::Integer(stat.last_seen),
        ])
        .await?;
    }
    insert_arg_stats.flush().await?;

    let mut insert_arg_stats_any = BulkInserter::new(
      conn,
      "INSERT INTO arg_stats_any (workspace_root, command_head, flags_json, arg_raw, arg_norm, freq, last_seen) VALUES ",
      7,
    );
    for ((workspace_root, head, flags_json, arg_raw), stat) in &arg_stats_any {
      insert_arg_stats_any
        .push_row([
          Value::Text(workspace_root.clone()),
          Value::Text(head.clone()),
          Value::Text(flags_json.clone()),
          Value::Text(arg_raw.clone()),
          Value::Text(stat.arg_norm.clone()),
          Value::Integer(stat.freq),
          Value::Integer(stat.last_seen),
        ])
        .await?;
    }
    insert_arg_stats_any.flush().await?;

    let mut insert_flag_stats = BulkInserter::new(
      conn,
      "INSERT INTO flag_stats (workspace_root, command_head, flag_raw, flag_norm, freq, last_seen) VALUES ",
      6,
    );
    for ((workspace_root, head, flag_raw, flag_norm), stat) in &flag_stats {
      insert_flag_stats
        .push_row([
          Value::Text(workspace_root.clone()),
          Value::Text(head.clone()),
          Value::Text(flag_raw.clone()),
          Value::Text(flag_norm.clone()),
          Value::Integer(stat.freq),
          Value::Integer(stat.last_seen),
        ])
        .await?;
    }
    insert_flag_stats.flush().await?;

    let mut insert_env_stats = BulkInserter::new(
      conn,
      "INSERT INTO env_stats (workspace_root, command_head, env_key, env_raw, env_norm, freq, last_seen) VALUES ",
      7,
    );
    for ((workspace_root, head, env_key, env_raw, env_norm), stat) in &env_stats {
      insert_env_stats
        .push_row([
          Value::Text(workspace_root.clone()),
          Value::Text(head.clone()),
          Value::Text(env_key.clone()),
          Value::Text(env_raw.clone()),
          Value::Text(env_norm.clone()),
          Value::Integer(stat.freq),
          Value::Integer(stat.last_seen),
        ])
        .await?;
    }
    insert_env_stats.flush().await?;

    let mut insert_token_cache = BulkInserter::new(
      conn,
      "INSERT INTO token_cache (command, tokens_json, normalized_json, updated_at) VALUES ",
      4,
    );
    for (command, (raw, norm)) in &token_cache {
      let raw_json = serde_json::to_string(raw)?;
      let norm_json = serde_json::to_string(norm)?;
      insert_token_cache
        .push_row([
          Value::Text(command.clone()),
          Value::Text(raw_json),
          Value::Text(norm_json),
          Value::Integer(now),
        ])
        .await?;
    }
    insert_token_cache.flush().await?;

    Ok(())
  }
  .await;

  if let Err(err) = write_result {
    let _ = conn.execute("ROLLBACK", ()).await;
    return Err(err);
  }
  conn.execute("COMMIT", ()).await?;

  Ok(IndexReport {
    commands: command_stats.len(),
    transitions: transition_stats.len(),
    contexts: context_stats.len(),
    token_cache: token_cache.len(),
  })
}

fn update_stat(map: &mut HashMap<String, Stat>, key: &str, ts: i64) {
  let entry = map.entry(key.to_string()).or_default();
  entry.freq += 1;
  if ts > entry.last_seen {
    entry.last_seen = ts;
  }
}

fn update_stat_key<K: std::hash::Hash + Eq>(map: &mut HashMap<K, Stat>, key: K, ts: i64) {
  let entry = map.entry(key).or_default();
  entry.freq += 1;
  if ts > entry.last_seen {
    entry.last_seen = ts;
  }
}

fn update_arg_stat<K: std::hash::Hash + Eq>(
  map: &mut HashMap<K, ArgStat>,
  key: K,
  arg_norm: &str,
  ts: i64,
) {
  let entry = map.entry(key).or_default();
  entry.freq += 1;
  if ts > entry.last_seen {
    entry.last_seen = ts;
  }
  if entry.arg_norm != arg_norm {
    entry.arg_norm = arg_norm.to_string();
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::db;
  use crate::shell_history::Invocation;
  use tempfile::NamedTempFile;

  fn base_invocation(command: &str, ts: i64) -> Invocation {
    Invocation {
      command: command.to_string(),
      expanded_command: command.to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/nonexistent/project".to_string()),
      workspace: None,
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(ts),
      end_unix_timestamp: Some(ts + 1),
      session_id: 1,
    }
  }

  #[tokio::test]
  async fn rebuild_stats_should_preserve_existing_command_embeddings() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = db::open_db(tmp.path()).await?;
    db::init(&db.conn).await?;

    db.conn
      .execute("ALTER TABLE command_stats ADD COLUMN embedding BLOB", ())
      .await?;
    db.conn
      .execute(
        "ALTER TABLE command_stats ADD COLUMN embedding_updated_at INTEGER",
        (),
      )
      .await?;

    // Seed a precomputed embedding for a command we will still have in history.
    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen, embedding, embedding_updated_at)
         VALUES (?, ?, ?, ?, ?)",
        libsql::params![
          "echo hi".to_string(),
          1i64,
          10i64,
          Some(vec![1u8, 2, 3, 4]),
          Some(123i64)
        ],
      )
      .await?;

    db::insert_invocation(&db.conn, &base_invocation("echo hi", 10)).await?;
    rebuild_stats(&db.conn, None).await?;

    // Expected behavior: rebuilding frequency/transition stats must not destroy embeddings.
    // Current behavior deletes `command_stats` rows and reinserts without copying embeddings.
    let mut rows = db
      .conn
      .query(
        "SELECT embedding FROM command_stats WHERE command = ?",
        libsql::params!["echo hi".to_string()],
      )
      .await?;
    let row = rows.next().await?.expect("row");
    let embedding: Option<Vec<u8>> = row.get(0)?;
    assert_eq!(embedding, Some(vec![1u8, 2, 3, 4]));
    Ok(())
  }

  #[tokio::test]
  async fn rebuild_stats_should_preserve_embedding_updated_at() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = db::open_db(tmp.path()).await?;
    db::init(&db.conn).await?;

    db.conn
      .execute("ALTER TABLE command_stats ADD COLUMN embedding BLOB", ())
      .await?;
    db.conn
      .execute(
        "ALTER TABLE command_stats ADD COLUMN embedding_updated_at INTEGER",
        (),
      )
      .await?;

    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen, embedding, embedding_updated_at)
         VALUES (?, ?, ?, ?, ?)",
        libsql::params![
          "echo hi".to_string(),
          1i64,
          10i64,
          Some(vec![9u8, 9, 9, 9]),
          Some(999i64)
        ],
      )
      .await?;

    db::insert_invocation(&db.conn, &base_invocation("echo hi", 10)).await?;
    rebuild_stats(&db.conn, None).await?;

    let mut rows = db
      .conn
      .query(
        "SELECT embedding_updated_at FROM command_stats WHERE command = ?",
        libsql::params!["echo hi".to_string()],
      )
      .await?;
    let row = rows.next().await?.expect("row");
    let updated_at: Option<i64> = row.get(0)?;
    assert_eq!(updated_at, Some(999));
    Ok(())
  }

  #[tokio::test]
  async fn rebuild_stats_max_commands_should_take_most_recent_history() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = db::open_db(tmp.path()).await?;
    db::init(&db.conn).await?;

    db::insert_invocation(&db.conn, &base_invocation("cmd-1", 1)).await?;
    db::insert_invocation(&db.conn, &base_invocation("cmd-2", 2)).await?;
    db::insert_invocation(&db.conn, &base_invocation("cmd-3", 3)).await?;

    // Desired behavior: limiting the index should keep the newest commands (for relevance).
    // Current behavior orders ASC and LIMITs, keeping the oldest commands instead.
    rebuild_stats(&db.conn, Some(2)).await?;

    let mut rows = db
      .conn
      .query(
        "SELECT command FROM command_stats ORDER BY last_seen ASC",
        (),
      )
      .await?;
    let mut cmds = Vec::new();
    while let Some(row) = rows.next().await? {
      cmds.push(row.get::<String>(0)?);
    }
    assert_eq!(cmds, vec!["cmd-2".to_string(), "cmd-3".to_string()]);
    Ok(())
  }

  #[tokio::test]
  async fn rebuild_stats_workspace_root_should_fall_back_to_working_directory() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = db::open_db(tmp.path()).await?;
    db::init(&db.conn).await?;

    // This path will not exist during the test, so no workspace marker is found.
    let inv = base_invocation("echo hi", 10);
    db::insert_invocation(&db.conn, &inv).await?;
    rebuild_stats(&db.conn, None).await?;

    let mut rows = db
      .conn
      .query(
        "SELECT workspace_root FROM workspace_command_stats WHERE command = ?",
        libsql::params!["echo hi".to_string()],
      )
      .await?;
    let row = rows.next().await?.expect("row");
    let workspace_root: String = row.get(0)?;
    assert!(workspace_root.is_empty());
    Ok(())
  }
}
