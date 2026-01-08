use std::collections::HashMap;

use libsql::Connection;
use serde_json;
use tracing::info;

use crate::Result;
use crate::tokenize::tokenize_index;

#[derive(Debug, Clone)]
pub struct SequenceConfig {
  pub min_support: usize,
  pub min_confidence: f64,
  pub min_lift: f64,
  pub max_len: usize,
}

impl Default for SequenceConfig {
  fn default() -> Self {
    Self {
      min_support: 2,
      min_confidence: 0.5,
      min_lift: 1.2,
      max_len: 3,
    }
  }
}

#[derive(Debug, Clone)]
pub struct SequenceReport {
  pub sequences: usize,
  pub bigrams: usize,
  pub trigrams: usize,
}

#[derive(Debug, Clone)]
pub struct TokenSequenceReport {
  pub sequences: usize,
  pub bigrams: usize,
  pub trigrams: usize,
}

#[derive(Debug, Clone)]
pub struct SequenceCandidate {
  pub command: String,
  pub confidence: f64,
  pub lift: f64,
  pub support: usize,
  pub prefix_len: usize,
}

pub async fn analyze_sequences(
  conn: &Connection,
  config: SequenceConfig,
) -> Result<SequenceReport> {
  let mut total: usize = 0;
  let mut unigram_counts: HashMap<String, usize> = HashMap::new();
  let mut bigram_counts: HashMap<(String, String), usize> = HashMap::new();
  let mut trigram_counts: HashMap<(String, String, String), usize> = HashMap::new();

  let mut rows = conn
    .query(
      "SELECT command FROM shell_history ORDER BY COALESCE(start_unix_timestamp, 0) ASC, id ASC",
      (),
    )
    .await?;
  let mut prev1: Option<String> = None;
  let mut prev2: Option<String> = None;
  let progress_interval = 50_000usize;

  while let Some(row) = rows.next().await? {
    let cmd = row.get::<String>(0)?;
    total += 1;
    *unigram_counts.entry(cmd.clone()).or_insert(0) += 1;

    if let Some(prev) = &prev1 {
      *bigram_counts
        .entry((prev.clone(), cmd.clone()))
        .or_insert(0) += 1;
    }

    if config.max_len >= 3
      && let (Some(p2), Some(p1)) = (&prev2, &prev1)
    {
      *trigram_counts
        .entry((p2.clone(), p1.clone(), cmd.clone()))
        .or_insert(0) += 1;
    }

    prev2 = prev1.take();
    prev1 = Some(cmd);

    if total.is_multiple_of(progress_interval) {
      info!("Scanned {} commands for sequences so far", total);
    }
  }

  if total < 2 {
    return Ok(SequenceReport {
      sequences: 0,
      bigrams: 0,
      trigrams: 0,
    });
  }

  let total_f = total as f64;

  conn.execute("BEGIN", ()).await?;

  let write_result: Result<(usize, usize, usize)> = async {
    let mut inserted = 0usize;
    let mut bigrams = 0usize;
    let mut trigrams = 0usize;
    conn.execute("DELETE FROM sequence_stats", ()).await?;

    for ((a, b), count) in &bigram_counts {
      let support = *count;
      if support < config.min_support {
        continue;
      }
      let prefix = *unigram_counts.get(a).unwrap_or(&0);
      if prefix == 0 {
        continue;
      }
      let confidence = support as f64 / prefix as f64;
      let base = *unigram_counts.get(b).unwrap_or(&0) as f64 / total_f;
      if base <= 0.0 {
        continue;
      }
      let lift = confidence / base;
      if confidence < config.min_confidence || lift < config.min_lift {
        continue;
      }
      let sequence_json = serde_json::to_string(&vec![a, b])?;
      conn
        .execute(
          "INSERT INTO sequence_stats (sequence_json, support, confidence, lift, context_json)
           VALUES (?, ?, ?, ?, NULL)",
          (sequence_json, support as i64, confidence, lift),
        )
        .await?;
      inserted += 1;
      bigrams += 1;
    }

    for ((a, b, c), count) in &trigram_counts {
      let support = *count;
      if support < config.min_support {
        continue;
      }
      let prefix = bigram_counts
        .get(&(a.clone(), b.clone()))
        .copied()
        .unwrap_or(0);
      if prefix == 0 {
        continue;
      }
      let confidence = support as f64 / prefix as f64;
      let base = *unigram_counts.get(c).unwrap_or(&0) as f64 / total_f;
      if base <= 0.0 {
        continue;
      }
      let lift = confidence / base;
      if confidence < config.min_confidence || lift < config.min_lift {
        continue;
      }
      let sequence_json = serde_json::to_string(&vec![a, b, c])?;
      conn
        .execute(
          "INSERT INTO sequence_stats (sequence_json, support, confidence, lift, context_json)
           VALUES (?, ?, ?, ?, NULL)",
          (sequence_json, support as i64, confidence, lift),
        )
        .await?;
      inserted += 1;
      trigrams += 1;
    }

    Ok((inserted, bigrams, trigrams))
  }
  .await;

  let (inserted, bigrams, trigrams) = match write_result {
    Ok(values) => values,
    Err(err) => {
      let _ = conn.execute("ROLLBACK", ()).await;
      return Err(err);
    }
  };

  conn.execute("COMMIT", ()).await?;

  Ok(SequenceReport {
    sequences: inserted,
    bigrams,
    trigrams,
  })
}

pub async fn analyze_token_sequences(
  conn: &Connection,
  config: SequenceConfig,
) -> Result<TokenSequenceReport> {
  let mut total: usize = 0;
  let mut unigram_counts: HashMap<String, usize> = HashMap::new();
  let mut bigram_counts: HashMap<(String, String), usize> = HashMap::new();
  let mut trigram_counts: HashMap<(String, String, String), usize> = HashMap::new();

  let mut rows = conn
    .query(
      "SELECT command, shellname FROM shell_history ORDER BY COALESCE(start_unix_timestamp, 0) ASC, id ASC",
      (),
    )
    .await?;
  let progress_interval = 50_000usize;

  while let Some(row) = rows.next().await? {
    let command = row.get::<String>(0)?;
    let shellname = row.get::<String>(1)?;
    let tokens = tokenize_index(&shellname, &command);
    if tokens.len() < 2 {
      continue;
    }
    let normalized: Vec<String> = tokens.into_iter().map(|t| t.normalized).collect();

    total += normalized.len();
    for tok in &normalized {
      *unigram_counts.entry(tok.clone()).or_insert(0) += 1;
    }

    for win in normalized.windows(2) {
      let key = (win[0].clone(), win[1].clone());
      *bigram_counts.entry(key).or_insert(0) += 1;
    }

    if config.max_len >= 3 {
      for win in normalized.windows(3) {
        let key = (win[0].clone(), win[1].clone(), win[2].clone());
        *trigram_counts.entry(key).or_insert(0) += 1;
      }
    }

    if total.is_multiple_of(progress_interval) {
      info!("Scanned {} tokens for token sequences so far", total);
    }
  }

  if total < 2 {
    return Ok(TokenSequenceReport {
      sequences: 0,
      bigrams: 0,
      trigrams: 0,
    });
  }

  let total_f = total as f64;

  conn.execute("BEGIN", ()).await?;
  let write_result: Result<(usize, usize, usize)> = async {
    let mut inserted = 0usize;
    let mut bigrams = 0usize;
    let mut trigrams = 0usize;
    conn.execute("DELETE FROM token_sequence_stats", ()).await?;

    for ((a, b), count) in &bigram_counts {
      let support = *count;
      if support < config.min_support {
        continue;
      }
      let prefix = *unigram_counts.get(a).unwrap_or(&0);
      if prefix == 0 {
        continue;
      }
      let confidence = support as f64 / prefix as f64;
      let base = *unigram_counts.get(b).unwrap_or(&0) as f64 / total_f;
      if base <= 0.0 {
        continue;
      }
      let lift = confidence / base;
      if confidence < config.min_confidence || lift < config.min_lift {
        continue;
      }
      let sequence_json = serde_json::to_string(&vec![a, b])?;
      conn
        .execute(
          "INSERT INTO token_sequence_stats (sequence_json, support, confidence, lift, prefix_len)
           VALUES (?, ?, ?, ?, 1)",
          (sequence_json, support as i64, confidence, lift),
        )
        .await?;
      inserted += 1;
      bigrams += 1;
    }

    for ((a, b, c), count) in &trigram_counts {
      let support = *count;
      if support < config.min_support {
        continue;
      }
      let prefix = bigram_counts
        .get(&(a.clone(), b.clone()))
        .copied()
        .unwrap_or(0);
      if prefix == 0 {
        continue;
      }
      let confidence = support as f64 / prefix as f64;
      let base = *unigram_counts.get(c).unwrap_or(&0) as f64 / total_f;
      if base <= 0.0 {
        continue;
      }
      let lift = confidence / base;
      if confidence < config.min_confidence || lift < config.min_lift {
        continue;
      }
      let sequence_json = serde_json::to_string(&vec![a, b, c])?;
      conn
        .execute(
          "INSERT INTO token_sequence_stats (sequence_json, support, confidence, lift, prefix_len)
           VALUES (?, ?, ?, ?, 2)",
          (sequence_json, support as i64, confidence, lift),
        )
        .await?;
      inserted += 1;
      trigrams += 1;
    }

    Ok((inserted, bigrams, trigrams))
  }
  .await;

  let (inserted, bigrams, trigrams) = match write_result {
    Ok(values) => values,
    Err(err) => {
      let _ = conn.execute("ROLLBACK", ()).await;
      return Err(err);
    }
  };

  conn.execute("COMMIT", ()).await?;

  Ok(TokenSequenceReport {
    sequences: inserted,
    bigrams,
    trigrams,
  })
}

pub async fn candidates_from_sequences(
  conn: &Connection,
  recent_commands: &[String],
  limit: usize,
) -> Result<Vec<SequenceCandidate>> {
  let mut rows = conn
    .query(
      "SELECT sequence_json, support, confidence, lift FROM sequence_stats ORDER BY lift DESC LIMIT ?",
      libsql::params![limit as i64],
    )
    .await?;

  let mut candidates = Vec::new();
  let recent_len = recent_commands.len();
  while let Some(row) = rows.next().await? {
    let sequence_json = row.get::<String>(0)?;
    let support = row.get::<i64>(1)? as usize;
    let confidence = row.get::<f64>(2)?;
    let lift = row.get::<f64>(3)?;

    let sequence: Vec<String> = serde_json::from_str(&sequence_json)?;
    if sequence.len() < 2 {
      continue;
    }
    let prefix_len = sequence.len() - 1;
    if prefix_len == 0 || recent_len < prefix_len {
      continue;
    }
    let recent_slice = &recent_commands[recent_len - prefix_len..];
    if sequence[..prefix_len] == *recent_slice {
      let next = sequence.last().cloned().unwrap_or_default();
      candidates.push(SequenceCandidate {
        command: next,
        confidence,
        lift,
        support,
        prefix_len,
      });
    }
  }

  Ok(candidates)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::db::{init, insert_invocation, open_db};
  use crate::shell_history::Invocation;

  async fn insert_cmd(conn: &libsql::Connection, command: &str, ts: i64) {
    let inv = Invocation {
      command: command.to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/tmp".to_string()),
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(ts),
      end_unix_timestamp: Some(ts + 1),
      session_id: 1,
    };
    let inserted = insert_invocation(conn, &inv).await.unwrap();
    assert!(inserted);
  }

  #[tokio::test]
  async fn test_sequence_mining_basic() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    insert_cmd(&db.conn, "git status", 1).await;
    insert_cmd(&db.conn, "git add .", 2).await;
    insert_cmd(&db.conn, "git status", 3).await;
    insert_cmd(&db.conn, "git add .", 4).await;

    let cfg = SequenceConfig {
      min_support: 1,
      min_confidence: 0.0,
      min_lift: 0.0,
      max_len: 3,
    };
    let report = analyze_sequences(&db.conn, cfg).await.unwrap();
    assert!(report.bigrams > 0);

    let mut rows = db
      .conn
      .query("SELECT COUNT(*) FROM sequence_stats", ())
      .await
      .unwrap();
    let row = rows.next().await.unwrap().expect("expected row");
    let count: i64 = row.get(0).unwrap();
    assert!(count > 0);
  }
}
