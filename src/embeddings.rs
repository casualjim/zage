use libsql::{Connection, Value};

use crate::{Result, ZageError};

const META_KEY_COMMAND_EMBEDDING_DIM: &str = "command_embedding_dim";
const COMMAND_EMBEDDING_INDEX: &str = "command_stats_embedding_idx";
const DEFAULT_SHELLNAME: &str = "zsh";
const DEFAULT_EMBEDDING_DIM: usize = 128;
const MAX_EMBED_TOKENS: usize = 256;

#[derive(Debug, Clone, Copy)]
pub struct EmbedContextInput<'a> {
  pub workspace_root: Option<&'a str>,
  pub cwd: Option<&'a str>,
  pub hostname: Option<&'a str>,
  pub username: Option<&'a str>,
  pub exit_status: Option<i64>,
  pub session_id: Option<i64>,
  pub recent_commands: &'a [String],
  pub window: usize,
}

pub async fn ensure_command_embeddings_schema(
  conn: &Connection,
  embedding_dim: usize,
) -> Result<()> {
  conn
    .execute(
      "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
      (),
    )
    .await?;

  let existing_dim = load_meta_usize(conn, META_KEY_COMMAND_EMBEDDING_DIM).await?;
  match existing_dim {
    Some(dim) if dim != embedding_dim => {
      return Err(ZageError::ConfigError(format!(
        "embedding dimension mismatch: db has {dim}, requested {embedding_dim}"
      )));
    }
    Some(_) => {}
    None => {
      conn
        .execute(
          "INSERT INTO meta (key, value) VALUES (?, ?)",
          (META_KEY_COMMAND_EMBEDDING_DIM, embedding_dim.to_string()),
        )
        .await?;
    }
  }

  // Store embeddings on `command_stats` (one row per unique command).
  //
  // `shell_history` is per-invocation, so storing embeddings there would duplicate data massively.
  // `command_stats` is already our canonical "unique command" table.
  ensure_command_stats_embedding_columns(conn, embedding_dim).await?;

  conn
    .execute(
      &format!(
        "CREATE INDEX IF NOT EXISTS {COMMAND_EMBEDDING_INDEX}
         ON command_stats(libsql_vector_idx(embedding))"
      ),
      (),
    )
    .await?;

  Ok(())
}

pub async fn command_embedding_dim(conn: &Connection) -> Result<Option<usize>> {
  if !table_exists(conn, "command_stats").await? {
    return Ok(None);
  }
  if !table_exists(conn, "meta").await? {
    return Ok(None);
  }
  load_meta_usize(conn, META_KEY_COMMAND_EMBEDDING_DIM).await
}

pub async fn index_command_embeddings(
  conn: &Connection,
  max_commands: Option<usize>,
) -> Result<usize> {
  let embedding_dim = DEFAULT_EMBEDDING_DIM;
  ensure_command_embeddings_schema(conn, embedding_dim).await?;

  let now = unix_now();
  let limit = max_commands.map(|v| v as i64).unwrap_or(i64::MAX);
  let commands = list_commands_missing_embeddings(conn, limit).await?;

  let mut inserted = 0usize;
  for chunk in commands.chunks(512) {
    for (command, shellname) in chunk {
      let embedding = embed_command_hash(shellname, command, embedding_dim);
      upsert_command_embedding(conn, command, &embedding, now).await?;
      inserted += 1;
    }
  }

  Ok(inserted)
}

pub fn embed_context_hash(input: EmbedContextInput<'_>, embedding_dim: usize) -> Vec<f32> {
  let mut out = vec![0.0f32; embedding_dim];

  // Encode context fields with a fixed weighting scheme.
  // This is intentionally simple and deterministic; online learning will replace this later.
  add_token(&mut out, "ctx", 0.25);

  if let Some(root) = input.workspace_root.filter(|v| !v.is_empty()) {
    add_token(&mut out, &format!("ctx:workspace_root={root}"), 4.0);
  }
  if let Some(cwd) = input.cwd.filter(|v| !v.is_empty()) {
    add_token(&mut out, &format!("ctx:cwd={cwd}"), 2.0);
  }
  if let Some(exit) = input.exit_status {
    add_token(&mut out, &format!("ctx:exit={exit}"), 1.5);
  }
  if let Some(host) = input.hostname.filter(|v| !v.is_empty()) {
    add_token(&mut out, &format!("ctx:host={host}"), 0.25);
  }
  if let Some(user) = input.username.filter(|v| !v.is_empty()) {
    add_token(&mut out, &format!("ctx:user={user}"), 0.25);
  }
  if let Some(session) = input.session_id {
    add_token(&mut out, &format!("ctx:session={session}"), 0.10);
  }

  // Recent command window: add generalized command tokens with a positional prefix.
  // We bias towards later commands slightly by weighting closer-to-now higher.
  let mut remaining = MAX_EMBED_TOKENS;
  let window = input.window.min(input.recent_commands.len());
  let start = input.recent_commands.len().saturating_sub(window);
  for (pos, cmd) in input.recent_commands[start..].iter().enumerate() {
    if remaining == 0 {
      break;
    }
    // pos is 0..window-1; later commands should matter more.
    let age = (window - 1).saturating_sub(pos) as f32;
    let w = 1.0 / (1.0 + age * 0.25);

    let tokens = command_tokens(DEFAULT_SHELLNAME, cmd);
    for tok in tokens.into_iter().take(remaining) {
      add_token(&mut out, &format!("prev:{tok}"), w);
      remaining = remaining.saturating_sub(1);
      if remaining == 0 {
        break;
      }
    }
  }

  l2_normalize(&mut out);
  out
}

fn embed_command_hash(shellname: &str, command: &str, embedding_dim: usize) -> Vec<f32> {
  let mut out = vec![0.0f32; embedding_dim];
  for token in command_tokens(shellname, command)
    .into_iter()
    .take(MAX_EMBED_TOKENS)
  {
    add_token(&mut out, &token, 1.0);
  }
  l2_normalize(&mut out);
  out
}

fn command_tokens(shellname: &str, command: &str) -> Vec<String> {
  let tokens = crate::tokenize::tokenize_index(shellname, command);
  let Some(parts) = crate::tokenize::extract_command_parts(command, &tokens) else {
    return Vec::new();
  };

  let mut out = Vec::new();
  if !parts.head.is_empty() {
    out.push(format!("head:{}", parts.head));
  }

  // position-independent flags
  let mut flags = parts.flags;
  flags.sort();
  flags.dedup();
  for flag in flags {
    out.push(format!("flag:{flag}"));
  }

  // bounded normalized args
  for arg in parts.args.into_iter().take(8) {
    if !arg.normalized.is_empty() {
      out.push(format!("arg:{}", arg.normalized));
    }
  }

  out
}

fn add_token(dst: &mut [f32], token: &str, weight: f32) {
  if dst.is_empty() {
    return;
  }
  let h = crate::hash_util::stable_hash(token);
  let idx = (h % dst.len() as u64) as usize;
  let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
  dst[idx] += sign * weight;
}

fn l2_normalize(v: &mut [f32]) {
  let mut sum = 0.0f32;
  for x in v.iter() {
    sum += x * x;
  }
  if sum <= 0.0 {
    return;
  }
  let inv = sum.sqrt().recip();
  for x in v.iter_mut() {
    *x *= inv;
  }
}

async fn list_commands_missing_embeddings(
  conn: &Connection,
  limit: i64,
) -> Result<Vec<(String, String)>> {
  let mut rows = conn
    .query(
      "WITH missing AS (
         SELECT cs.command AS command, cs.last_seen AS last_seen
         FROM command_stats cs
         WHERE cs.embedding IS NULL
         ORDER BY cs.last_seen DESC
         LIMIT ?
       ),
       ranked AS (
         SELECT
           sh.expanded_command AS command,
           sh.shellname AS shellname,
           ROW_NUMBER() OVER (
             PARTITION BY sh.expanded_command
             ORDER BY COALESCE(sh.start_unix_timestamp, 0) DESC, sh.id DESC
           ) AS rn
         FROM shell_history sh
         JOIN missing m ON m.command = sh.expanded_command
       )
       SELECT
         m.command,
         COALESCE(r.shellname, ?) AS shellname
       FROM missing m
       LEFT JOIN ranked r ON r.command = m.command AND r.rn = 1
       ORDER BY m.last_seen DESC",
      libsql::params![limit, DEFAULT_SHELLNAME.to_string()],
    )
    .await?;

  let mut out = Vec::new();
  while let Some(row) = rows.next().await? {
    let command = row.get::<String>(0)?;
    let shellname = row
      .get::<String>(1)
      .unwrap_or_else(|_| DEFAULT_SHELLNAME.to_string());
    out.push((command, shellname));
  }
  Ok(out)
}

pub async fn load_command_embedding(conn: &Connection, command: &str) -> Result<Option<Vec<f32>>> {
  if !table_exists(conn, "command_stats").await? {
    return Ok(None);
  }

  let dim = command_embedding_dim(conn).await?.unwrap_or(0);
  if dim == 0 {
    return Ok(None);
  }

  let mut rows = conn
    .query(
      "SELECT embedding FROM command_stats WHERE command = ? LIMIT 1",
      libsql::params![command.to_string()],
    )
    .await?;
  let Some(row) = rows.next().await? else {
    return Ok(None);
  };
  let raw = row.get::<Option<Vec<u8>>>(0)?;
  Ok(raw.and_then(|v| decode_f32_blob(&v, dim)))
}

pub async fn mean_embedding_for_commands(
  conn: &Connection,
  commands: &[String],
  max_commands: usize,
) -> Result<Option<Vec<f32>>> {
  let dim = command_embedding_dim(conn).await?.unwrap_or(0);
  if dim == 0 {
    return Ok(None);
  }

  if max_commands == 0 {
    return Ok(None);
  }

  let mut weights: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
  for cmd in commands.iter().rev().take(max_commands) {
    *weights.entry(cmd.as_str()).or_insert(0) += 1;
  }
  if weights.is_empty() {
    return Ok(None);
  }

  let mut sql = String::from("SELECT command, embedding FROM command_stats WHERE command IN (");
  sql.push_str(&weights.keys().map(|_| "?").collect::<Vec<_>>().join(","));
  sql.push(')');

  let mut params: Vec<Value> = Vec::with_capacity(weights.len());
  for cmd in weights.keys() {
    params.push(Value::from((*cmd).to_string()));
  }

  let mut rows = conn.query(&sql, params).await?;
  let mut sum = vec![0.0f32; dim];
  let mut count = 0usize;
  while let Some(row) = rows.next().await? {
    let cmd = row.get::<String>(0)?;
    let raw = row.get::<Option<Vec<u8>>>(1)?;
    let Some(raw) = raw else {
      continue;
    };
    let Some(embedding) = decode_f32_blob(&raw, dim) else {
      continue;
    };
    let weight = *weights.get(cmd.as_str()).unwrap_or(&0);
    if weight == 0 {
      continue;
    }
    for (dst, src) in sum.iter_mut().zip(embedding.iter()) {
      *dst += *src * (weight as f32);
    }
    count += weight;
  }

  if count == 0 {
    return Ok(None);
  }

  let denom = count as f32;
  for v in sum.iter_mut() {
    *v /= denom;
  }
  Ok(Some(sum))
}

pub async fn search_similar_commands(
  conn: &Connection,
  query: &[f32],
  limit: usize,
) -> Result<Vec<String>> {
  if !table_exists(conn, "command_stats").await? {
    return Ok(Vec::new());
  }

  let dim = command_embedding_dim(conn).await?.unwrap_or(0);
  if dim == 0 || query.len() != dim {
    return Ok(Vec::new());
  }

  let query_blob = encode_f32_blob(query);
  let mut rows = conn
    .query(
      &format!(
        "SELECT cs.command
         FROM vector_top_k('{COMMAND_EMBEDDING_INDEX}', ?, ?) v
         JOIN command_stats cs ON cs.rowid = v.rowid"
      ),
      libsql::params![query_blob, limit as i64],
    )
    .await?;

  let mut out = Vec::new();
  while let Some(row) = rows.next().await? {
    out.push(row.get::<String>(0)?);
  }
  Ok(out)
}

async fn upsert_command_embedding(
  conn: &Connection,
  command: &str,
  embedding: &[f32],
  updated_at: i64,
) -> Result<()> {
  let embedding_blob = encode_f32_blob(embedding);
  conn
    .execute(
      "UPDATE command_stats
       SET embedding = ?,
           embedding_updated_at = ?
       WHERE command = ?",
      (embedding_blob, updated_at, command.to_string()),
    )
    .await?;
  Ok(())
}

async fn table_exists(conn: &Connection, name: &str) -> Result<bool> {
  let mut rows = conn
    .query(
      "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
      libsql::params![name.to_string()],
    )
    .await?;
  Ok(rows.next().await?.is_some())
}

async fn load_meta_usize(conn: &Connection, key: &str) -> Result<Option<usize>> {
  let mut rows = conn
    .query(
      "SELECT value FROM meta WHERE key = ? LIMIT 1",
      libsql::params![key.to_string()],
    )
    .await?;
  let Some(row) = rows.next().await? else {
    return Ok(None);
  };
  let value = row.get::<String>(0)?;
  Ok(value.parse::<usize>().ok())
}

async fn ensure_command_stats_embedding_columns(
  conn: &Connection,
  embedding_dim: usize,
) -> Result<()> {
  let mut rows = conn.query("PRAGMA table_info(command_stats)", ()).await?;
  let mut columns = std::collections::HashSet::new();
  while let Some(row) = rows.next().await? {
    let name: String = row.get(1)?;
    columns.insert(name);
  }

  if !columns.contains("embedding") {
    conn
      .execute(
        &format!("ALTER TABLE command_stats ADD COLUMN embedding F32_BLOB({embedding_dim})"),
        (),
      )
      .await?;
  }
  if !columns.contains("embedding_updated_at") {
    conn
      .execute(
        "ALTER TABLE command_stats ADD COLUMN embedding_updated_at INTEGER",
        (),
      )
      .await?;
  }

  Ok(())
}

fn encode_f32_blob(embedding: &[f32]) -> Vec<u8> {
  let mut out = Vec::with_capacity(embedding.len() * 4);
  for v in embedding {
    out.extend_from_slice(&v.to_le_bytes());
  }
  out
}

fn decode_f32_blob(blob: &[u8], dim: usize) -> Option<Vec<f32>> {
  let expected_len = dim.checked_mul(4)?;
  if blob.len() != expected_len {
    return None;
  }

  let mut out = Vec::with_capacity(dim);
  for chunk in blob.chunks_exact(4) {
    out.push(f32::from_le_bytes(chunk.try_into().ok()?));
  }
  Some(out)
}

fn unix_now() -> i64 {
  std::time::SystemTime::now()
    .duration_since(std::time::UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::db;
  use tempfile::NamedTempFile;

  #[tokio::test]
  async fn list_commands_missing_embeddings_filters_and_orders() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = db::open_db(tmp.path()).await?;

    db.conn
      .execute("ALTER TABLE command_stats ADD COLUMN embedding BLOB", ())
      .await?;

    // last_seen order: b (newest), a, c (oldest)
    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen, embedding) VALUES (?, ?, ?, ?)",
        libsql::params!["a".to_string(), 1i64, 200i64, Option::<Vec<u8>>::None],
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen, embedding) VALUES (?, ?, ?, ?)",
        libsql::params!["b".to_string(), 1i64, 300i64, Option::<Vec<u8>>::None],
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen, embedding) VALUES (?, ?, ?, ?)",
        libsql::params!["c".to_string(), 1i64, 100i64, Some(vec![0u8, 0, 0, 0])],
      )
      .await?;

    let out = list_commands_missing_embeddings(&db.conn, 10).await?;
    assert_eq!(
      out,
      vec![
        ("b".to_string(), DEFAULT_SHELLNAME.to_string()),
        ("a".to_string(), DEFAULT_SHELLNAME.to_string())
      ]
    );

    let out_limited = list_commands_missing_embeddings(&db.conn, 1).await?;
    assert_eq!(
      out_limited,
      vec![("b".to_string(), DEFAULT_SHELLNAME.to_string())]
    );
    Ok(())
  }

  #[tokio::test]
  async fn mean_embedding_for_commands_decodes_blob_and_averages() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = db::open_db(tmp.path()).await?;

    db.conn
      .execute(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        (),
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        libsql::params![META_KEY_COMMAND_EMBEDDING_DIM.to_string(), "3".to_string()],
      )
      .await?;

    db.conn
      .execute("ALTER TABLE command_stats ADD COLUMN embedding BLOB", ())
      .await?;

    fn blob3(a: f32, b: f32, c: f32) -> Vec<u8> {
      let mut out = Vec::with_capacity(12);
      out.extend_from_slice(&a.to_le_bytes());
      out.extend_from_slice(&b.to_le_bytes());
      out.extend_from_slice(&c.to_le_bytes());
      out
    }

    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen, embedding) VALUES (?, ?, ?, ?)",
        libsql::params!["x".to_string(), 1i64, 10i64, Some(blob3(1.0, 2.0, 3.0))],
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen, embedding) VALUES (?, ?, ?, ?)",
        libsql::params!["y".to_string(), 1i64, 20i64, Some(blob3(3.0, 2.0, 1.0))],
      )
      .await?;

    let commands = vec!["x".to_string(), "y".to_string()];
    let mean = mean_embedding_for_commands(&db.conn, &commands, 10)
      .await?
      .expect("mean embedding");
    assert_eq!(mean, vec![2.0, 2.0, 2.0]);
    Ok(())
  }

  #[tokio::test]
  async fn mean_embedding_for_commands_counts_duplicates_in_window() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = db::open_db(tmp.path()).await?;

    db.conn
      .execute(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        (),
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        libsql::params![META_KEY_COMMAND_EMBEDDING_DIM.to_string(), "1".to_string()],
      )
      .await?;

    db.conn
      .execute("ALTER TABLE command_stats ADD COLUMN embedding BLOB", ())
      .await?;

    fn blob1(v: f32) -> Vec<u8> {
      v.to_le_bytes().to_vec()
    }

    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen, embedding) VALUES (?, ?, ?, ?)",
        libsql::params!["x".to_string(), 1i64, 10i64, Some(blob1(0.0))],
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen, embedding) VALUES (?, ?, ?, ?)",
        libsql::params!["y".to_string(), 1i64, 20i64, Some(blob1(3.0))],
      )
      .await?;

    // Desired behavior: duplicates in the context window should contribute multiple times
    // to the mean (x, x, y) => (0 + 0 + 3) / 3 = 1.0.
    //
    // Current implementation uses `IN (...)` which collapses duplicates, yielding (0 + 3) / 2 = 1.5.
    let commands = vec!["x".to_string(), "x".to_string(), "y".to_string()];
    let mean = mean_embedding_for_commands(&db.conn, &commands, 3)
      .await?
      .expect("mean embedding");
    assert_eq!(mean, vec![1.0]);
    Ok(())
  }

  #[tokio::test]
  async fn mean_embedding_for_commands_zero_limit_returns_none() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = db::open_db(tmp.path()).await?;

    db.conn
      .execute(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
        (),
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        libsql::params![META_KEY_COMMAND_EMBEDDING_DIM.to_string(), "1".to_string()],
      )
      .await?;

    db.conn
      .execute("ALTER TABLE command_stats ADD COLUMN embedding BLOB", ())
      .await?;

    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen, embedding) VALUES (?, ?, ?, ?)",
        libsql::params![
          "x".to_string(),
          1i64,
          10i64,
          Some(0.0f32.to_le_bytes().to_vec())
        ],
      )
      .await?;

    // Desired behavior: max_commands=0 should mean "use zero commands", so return None.
    // Current code forces a minimum of 1 via `max_commands.max(1)`.
    let commands = vec!["x".to_string()];
    let mean = mean_embedding_for_commands(&db.conn, &commands, 0).await?;
    assert!(mean.is_none());
    Ok(())
  }

  #[tokio::test]
  async fn search_similar_commands_works_without_distance_column() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = db::open_db(tmp.path()).await?;

    ensure_command_embeddings_schema(&db.conn, 2).await?;

    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen, embedding) VALUES (?, ?, ?, ?)",
        libsql::params![
          "a".to_string(),
          1i64,
          10i64,
          Some(encode_f32_blob(&[1.0, 1.0]))
        ],
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen, embedding) VALUES (?, ?, ?, ?)",
        libsql::params![
          "b".to_string(),
          1i64,
          20i64,
          Some(encode_f32_blob(&[-1.0, -1.0]))
        ],
      )
      .await?;

    let out = search_similar_commands(&db.conn, &[1.0, 1.0], 2).await?;
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], "a".to_string());

    let out = search_similar_commands(&db.conn, &[1.0, 1.0], 1).await?;
    assert_eq!(out, vec!["a".to_string()]);
    Ok(())
  }
}
