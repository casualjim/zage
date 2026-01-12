use std::collections::HashSet;

use libsql::Connection;
use serde_json;

use crate::Result;
use crate::tokenize::normalized_tokens;

use super::sql::query_prepared;
pub(crate) const DEFAULT_RECENCY_HALF_LIFE_SECONDS: f64 = 60.0 * 60.0 * 24.0 * 7.0;

pub(crate) fn recency_score(now: i64, last_seen: i64, half_life_seconds: f64) -> f64 {
  if last_seen <= 0 || now <= last_seen {
    return 0.0;
  }
  let age = (now - last_seen) as f64;
  (-age / half_life_seconds).exp()
}

pub(crate) fn token_similarity(a: &[String], b: &[String]) -> f64 {
  if a.is_empty() || b.is_empty() {
    return 0.0;
  }
  let set_a: HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
  let set_b: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
  let intersection = set_a.intersection(&set_b).count() as f64;
  (2.0 * intersection) / (set_a.len() as f64 + set_b.len() as f64)
}

pub(crate) async fn load_normalized_tokens(
  conn: &Connection,
  command: &str,
) -> Result<Vec<String>> {
  let mut rows = query_prepared(
    conn,
    "SELECT normalized_json FROM token_cache WHERE command = ?",
    [command.to_string()],
  )
  .await?;
  if let Some(row) = rows.next().await? {
    let json = row.get::<String>(0)?;
    let tokens: Vec<String> = serde_json::from_str(&json)?;
    return Ok(tokens);
  }

  Ok(normalized_tokens(command))
}
