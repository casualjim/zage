use libsql::Connection;
use rand::Rng;
use rand::rngs::StdRng;

use crate::{Result, ZageError};

use super::trainer::OnlineExample;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ReplayConfig {
  pub global_capacity: usize,
  pub workspace_capacity: usize,
  pub max_workspaces: usize,
}

impl Default for ReplayConfig {
  fn default() -> Self {
    Self {
      global_capacity: 20_000,
      workspace_capacity: 5_000,
      max_workspaces: 32,
    }
  }
}

const META_GLOBAL_SEEN: &str = "online_replay_global_seen";
const META_WS_SEQ_PREFIX: &str = "online_replay_ws_seq:";
const META_WS_LAST_PREFIX: &str = "online_replay_ws_last:";

pub(crate) async fn store_replay(
  conn: &Connection,
  example: &OnlineExample,
  config: &ReplayConfig,
  rng: &mut StdRng,
) -> Result<()> {
  let payload = encode_example(example)?;
  let now = example.now;

  store_global(conn, &payload, now, config.global_capacity, rng).await?;
  if let Some(workspace) = example.workspace_key.as_deref()
    && !workspace.is_empty()
  {
    store_workspace(
      conn,
      workspace,
      &payload,
      now,
      config.workspace_capacity,
      config.max_workspaces,
    )
    .await?;
  }

  Ok(())
}

pub(crate) async fn sample_global_replay(
  conn: &Connection,
  rng: &mut StdRng,
) -> Result<Option<OnlineExample>> {
  let count = count_rows(conn, "online_replay_global", None).await?;
  if count == 0 {
    return Ok(None);
  }
  let offset = rng.random_range(0..count) as i64;
  let mut rows = conn
    .query(
      "SELECT payload FROM online_replay_global ORDER BY event_id LIMIT 1 OFFSET ?",
      libsql::params![offset],
    )
    .await?;
  let Some(row) = rows.next().await? else {
    return Ok(None);
  };
  let payload: Vec<u8> = row.get(0)?;
  Ok(Some(decode_example(&payload)?))
}

pub(crate) async fn sample_workspace_replay(
  conn: &Connection,
  workspace_root: &str,
  rng: &mut StdRng,
) -> Result<Option<OnlineExample>> {
  let count = count_rows(
    conn,
    "online_replay_workspace",
    Some(("workspace_root", workspace_root)),
  )
  .await?;
  if count == 0 {
    return Ok(None);
  }
  let offset = rng.random_range(0..count) as i64;
  let mut rows = conn
    .query(
      "SELECT payload FROM online_replay_workspace
       WHERE workspace_root = ?
       ORDER BY seq
       LIMIT 1 OFFSET ?",
      libsql::params![workspace_root.to_string(), offset],
    )
    .await?;
  let Some(row) = rows.next().await? else {
    return Ok(None);
  };
  let payload: Vec<u8> = row.get(0)?;
  Ok(Some(decode_example(&payload)?))
}

async fn store_global(
  conn: &Connection,
  payload: &[u8],
  now: i64,
  capacity: usize,
  rng: &mut StdRng,
) -> Result<()> {
  if capacity == 0 {
    return Ok(());
  }
  let mut seen = get_meta_i64(conn, META_GLOBAL_SEEN).await?.unwrap_or(0);
  seen += 1;
  set_meta_i64(conn, META_GLOBAL_SEEN, seen).await?;

  let count = count_rows(conn, "online_replay_global", None).await?;
  if (count as usize) < capacity {
    conn
      .execute(
        "INSERT INTO online_replay_global (payload, sampled_at) VALUES (?, ?)",
        libsql::params![payload.to_vec(), now],
      )
      .await?;
    return Ok(());
  }

  // Reservoir: keep with prob capacity/seen.
  let keep_prob = (capacity as f64) / (seen as f64);
  if rng.random::<f64>() >= keep_prob {
    return Ok(());
  }

  let replace_offset = rng.random_range(0..capacity) as i64;
  let mut rows = conn
    .query(
      "SELECT event_id FROM online_replay_global ORDER BY event_id LIMIT 1 OFFSET ?",
      libsql::params![replace_offset],
    )
    .await?;
  let Some(row) = rows.next().await? else {
    return Ok(());
  };
  let event_id: i64 = row.get(0)?;
  conn
    .execute(
      "UPDATE online_replay_global SET payload = ?, sampled_at = ? WHERE event_id = ?",
      libsql::params![payload.to_vec(), now, event_id],
    )
    .await?;
  Ok(())
}

async fn store_workspace(
  conn: &Connection,
  workspace_root: &str,
  payload: &[u8],
  now: i64,
  capacity: usize,
  max_workspaces: usize,
) -> Result<()> {
  if capacity == 0 || max_workspaces == 0 {
    return Ok(());
  }

  let seq_key = format!("{META_WS_SEQ_PREFIX}{workspace_root}");
  let last_key = format!("{META_WS_LAST_PREFIX}{workspace_root}");

  let mut seq = get_meta_i64(conn, &seq_key).await?.unwrap_or(0);
  seq += 1;
  set_meta_i64(conn, &seq_key, seq).await?;
  set_meta_i64(conn, &last_key, now).await?;

  let slot = (seq.rem_euclid(capacity as i64)) as i64;
  conn
    .execute(
      "INSERT OR REPLACE INTO online_replay_workspace (workspace_root, seq, payload)
       VALUES (?, ?, ?)",
      libsql::params![workspace_root.to_string(), slot, payload.to_vec()],
    )
    .await?;

  enforce_workspace_lru(conn, max_workspaces).await?;
  Ok(())
}

async fn enforce_workspace_lru(conn: &Connection, max_workspaces: usize) -> Result<()> {
  let mut rows = conn
    .query(
      "SELECT key, value FROM online_model_meta
       WHERE key LIKE ?
       ORDER BY CAST(value AS INTEGER) ASC",
      libsql::params![format!("{META_WS_LAST_PREFIX}%")],
    )
    .await?;

  let mut entries: Vec<(String, i64)> = Vec::new();
  while let Some(row) = rows.next().await? {
    let key: String = row.get(0)?;
    let value: String = row.get(1)?;
    let ts = value.parse::<i64>().unwrap_or(0);
    entries.push((key, ts));
  }

  if entries.len() <= max_workspaces {
    return Ok(());
  }

  let evict_count = entries.len() - max_workspaces;
  for (key, _) in entries.into_iter().take(evict_count) {
    let Some(workspace_root) = key.strip_prefix(META_WS_LAST_PREFIX) else {
      continue;
    };
    conn
      .execute(
        "DELETE FROM online_replay_workspace WHERE workspace_root = ?",
        libsql::params![workspace_root.to_string()],
      )
      .await?;
    let seq_key = format!("{META_WS_SEQ_PREFIX}{workspace_root}");
    conn
      .execute(
        "DELETE FROM online_model_meta WHERE key = ?",
        libsql::params![key.clone()],
      )
      .await?;
    conn
      .execute(
        "DELETE FROM online_model_meta WHERE key = ?",
        libsql::params![seq_key],
      )
      .await?;
  }

  Ok(())
}

async fn count_rows(conn: &Connection, table: &str, filter: Option<(&str, &str)>) -> Result<usize> {
  let mut rows = if let Some((col, val)) = filter {
    let sql = format!("SELECT COUNT(*) FROM {table} WHERE {col} = ?");
    conn.query(&sql, libsql::params![val.to_string()]).await?
  } else {
    let sql = format!("SELECT COUNT(*) FROM {table}");
    conn.query(&sql, ()).await?
  };
  let row = rows
    .next()
    .await?
    .ok_or_else(|| ZageError::ConfigError(format!("missing COUNT(*) row for table {table}")))?;
  let count: i64 = row.get(0)?;
  Ok(count.try_into().unwrap_or(0))
}

async fn get_meta_i64(conn: &Connection, key: &str) -> Result<Option<i64>> {
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
  Ok(value.parse::<i64>().ok())
}

async fn set_meta_i64(conn: &Connection, key: &str, value: i64) -> Result<()> {
  conn
    .execute(
      "INSERT INTO online_model_meta (key, value) VALUES (?, ?)
       ON CONFLICT(key) DO UPDATE SET value = excluded.value",
      libsql::params![key.to_string(), value.to_string()],
    )
    .await?;
  Ok(())
}

fn encode_example(example: &OnlineExample) -> Result<Vec<u8>> {
  rkyv::to_bytes::<_, 4_096>(example)
    .map(|b| b.to_vec())
    .map_err(|err| ZageError::ConfigError(err.to_string()))
}

fn decode_example(payload: &[u8]) -> Result<OnlineExample> {
  rkyv::from_bytes::<OnlineExample>(payload).map_err(|err| ZageError::ConfigError(err.to_string()))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::db::{init, open_db};
  use rand::SeedableRng;
  use tempfile::NamedTempFile;

  fn example(now: i64, workspace: Option<&str>) -> OnlineExample {
    OnlineExample {
      shellname: "zsh".to_string(),
      workspace_root: workspace.unwrap_or_default().to_string(),
      cwd: workspace.map(|s| s.to_string()),
      workspace_key: workspace.map(|s| s.to_string()),
      positive_command: "echo test".to_string(),
      positive_head: Some("echo".to_string()),
      positive_command_hash: 1,
      positive_head_hash: 2,
      now,
      ctx_workspace: Vec::new(),
      ctx_cwd: Vec::new(),
      ctx_exit: Vec::new(),
      ctx_host: Vec::new(),
      ctx_user: Vec::new(),
      ctx_timebucket: Vec::new(),
      ctx_session: Vec::new(),
      recent_heads: Vec::new(),
      recent_flags: Vec::new(),
      recent_args: Vec::new(),
      cmd_buckets: vec![(1, 1.0)],
    }
  }

  #[tokio::test]
  async fn workspace_replay_ring_overwrites_and_lru_evicts() -> Result<()> {
    let tmp = NamedTempFile::new()?;
    let db = open_db(tmp.path()).await?;
    init(&db.conn).await?;

    let mut rng = StdRng::seed_from_u64(1);
    let cfg = ReplayConfig {
      global_capacity: 0,
      workspace_capacity: 2,
      max_workspaces: 2,
    };

    // Fill workspace A beyond ring capacity.
    store_replay(&db.conn, &example(10, Some("A")), &cfg, &mut rng).await?;
    store_replay(&db.conn, &example(11, Some("A")), &cfg, &mut rng).await?;
    store_replay(&db.conn, &example(12, Some("A")), &cfg, &mut rng).await?;

    let mut rows = db
      .conn
      .query(
        "SELECT COUNT(*) FROM online_replay_workspace WHERE workspace_root = ?",
        libsql::params!["A".to_string()],
      )
      .await?;
    let row = rows
      .next()
      .await?
      .ok_or_else(|| ZageError::ConfigError("missing row".to_string()))?;
    let count: i64 = row.get(0)?;
    assert_eq!(count, 2, "ring should cap workspace entries");

    // Touch A, then add B, then add C which should evict oldest (B).
    store_replay(&db.conn, &example(20, Some("A")), &cfg, &mut rng).await?;
    store_replay(&db.conn, &example(21, Some("B")), &cfg, &mut rng).await?;
    store_replay(&db.conn, &example(22, Some("C")), &cfg, &mut rng).await?;

    let mut rows = db
      .conn
      .query(
        "SELECT COUNT(DISTINCT workspace_root) FROM online_replay_workspace",
        (),
      )
      .await?;
    let row = rows
      .next()
      .await?
      .ok_or_else(|| ZageError::ConfigError("missing row".to_string()))?;
    let distinct: i64 = row.get(0)?;
    assert!(distinct <= 2, "should respect max_workspaces");

    Ok(())
  }
}
