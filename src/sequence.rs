use std::collections::HashMap;
use std::collections::HashSet;

use libsql::Connection;
use rayon::prelude::*;
use serde_json;
use tracing::info;

use crate::Result;
use crate::tokenize::{extract_command_parts, tokenize_index};

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
      min_confidence: 0.0,
      min_lift: 1.0,
      max_len: 5,
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

fn command_signature(shellname: &str, command: &str) -> Option<String> {
  let tokens = tokenize_index(shellname, command);
  let parts = extract_command_parts(command, &tokens)?;
  let mut signature = parts.head;
  if let Some(first_arg) = parts.args.first() {
    signature.push(' ');
    signature.push_str(first_arg.raw.as_str());
  }
  if !parts.flags.is_empty() {
    let mut flags = parts.flags;
    flags.sort();
    signature.push(' ');
    signature.push_str(&flags.join(" "));
  }
  Some(signature)
}

pub(crate) fn normalize_sequence_command(shellname: &str, command: &str) -> String {
  command_signature(shellname, command).unwrap_or_else(|| command.to_string())
}

pub async fn analyze_sequences(
  conn: &Connection,
  config: SequenceConfig,
) -> Result<SequenceReport> {
  let max_len = config.max_len.max(2);

  let mut rows = conn
    .query(
      "SELECT expanded_command, shellname FROM shell_history ORDER BY COALESCE(start_unix_timestamp, 0) ASC, id ASC",
      (),
    )
    .await?;
  let mut commands: Vec<String> = Vec::new();
  let progress_interval = 50_000usize;
  let mut scanned: usize = 0;

  while let Some(row) = rows.next().await? {
    let raw_cmd = row.get::<String>(0)?;
    let shellname = row.get::<String>(1)?;
    let cmd = normalize_sequence_command(&shellname, &raw_cmd);
    commands.push(cmd);
    scanned += 1;
    if scanned.is_multiple_of(progress_interval) {
      info!("Scanned {} commands for sequences so far", scanned);
    }
  }

  let total = commands.len();
  if total == 0 {
    return Ok(SequenceReport {
      sequences: 0,
      bigrams: 0,
      trigrams: 0,
    });
  }

  let unigram_counts: HashMap<String, usize> = commands
    .par_iter()
    .fold(HashMap::new, |mut map, cmd| {
      *map.entry(cmd.clone()).or_insert(0) += 1;
      map
    })
    .reduce(HashMap::new, |mut left, right| {
      for (cmd, count) in right {
        *left.entry(cmd).or_insert(0) += count;
      }
      left
    });

  let ngram_counts: Vec<HashMap<Vec<String>, usize>> = (2..=max_len)
    .into_par_iter()
    .map(|n| {
      commands
        .par_iter()
        .enumerate()
        .fold(HashMap::new, |mut map, (idx, _)| {
          if idx + 1 < n {
            return map;
          }
          let start = idx + 1 - n;
          let seq = commands[start..=idx].to_vec();
          *map.entry(seq).or_insert(0) += 1;
          map
        })
        .reduce(HashMap::new, |mut left, right| {
          for (seq, count) in right {
            *left.entry(seq).or_insert(0) += count;
          }
          left
        })
    })
    .collect();

  let total_f = total as f64;

  conn.execute("BEGIN", ()).await?;

  let write_result: Result<(usize, usize, usize)> = async {
    let mut inserted = 0usize;
    let mut bigrams = 0usize;
    let mut trigrams = 0usize;
    conn.execute("DELETE FROM sequence_stats", ()).await?;
    for (command, support) in &unigram_counts {
      let sequence_json = serde_json::to_string(&vec![command.clone()])?;
      conn
        .execute(
          "INSERT INTO sequence_stats (sequence_json, support, confidence, lift, sequence_len, prefix_json, last_command, context_json)
           VALUES (?, ?, 0.0, 0.0, 1, NULL, ?, NULL)",
          (sequence_json, *support as i64, command.clone()),
        )
        .await?;
    }

    for (idx, counts) in ngram_counts.iter().enumerate() {
      let n = idx + 2;
      for (sequence, count) in counts {
        if sequence.is_empty() {
          continue;
        }
        let support = *count;
        if support < config.min_support {
          continue;
        }
        let prefix = if n == 2 {
          *unigram_counts.get(&sequence[0]).unwrap_or(&0)
        } else {
          let prefix_seq = sequence[..n - 1].to_vec();
          *ngram_counts
            .get(idx - 1)
            .and_then(|prefix_map| prefix_map.get(&prefix_seq))
            .unwrap_or(&0)
        };
        if prefix == 0 {
          continue;
        }
        let confidence = support as f64 / prefix as f64;
        let last = &sequence[sequence.len() - 1];
        let base = *unigram_counts.get(last).unwrap_or(&0) as f64 / total_f;
        if base <= 0.0 {
          continue;
        }
        let lift = confidence / base;
        if confidence < config.min_confidence || lift < config.min_lift {
          continue;
        }
        let sequence_json = serde_json::to_string(sequence)?;
        let prefix_json = serde_json::to_string(&sequence[..n - 1])?;
        conn
          .execute(
            "INSERT INTO sequence_stats (sequence_json, support, confidence, lift, sequence_len, prefix_json, last_command, context_json)
             VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
            (
              sequence_json,
              support as i64,
              confidence,
              lift,
              n as i64,
              prefix_json,
              last.clone(),
            ),
          )
          .await?;
        inserted += 1;
        if n == 2 {
          bigrams += 1;
        } else if n == 3 {
          trigrams += 1;
        }
      }
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
  let max_len = config.max_len.max(2);

  let mut rows = conn
    .query(
      "SELECT expanded_command, shellname FROM shell_history ORDER BY COALESCE(start_unix_timestamp, 0) ASC, id ASC",
      (),
    )
    .await?;
  let progress_interval = 50_000usize;
  let mut scanned: usize = 0;
  let mut history: Vec<(String, String)> = Vec::new();

  while let Some(row) = rows.next().await? {
    let command = row.get::<String>(0)?;
    let shellname = row.get::<String>(1)?;
    history.push((shellname, command));
    scanned += 1;
    if scanned.is_multiple_of(progress_interval) {
      info!("Scanned {} commands for token sequences so far", scanned);
    }
  }

  #[derive(Default)]
  struct TokenSeqAcc {
    total: usize,
    unigram_counts: HashMap<String, usize>,
    ngram_counts: Vec<HashMap<Vec<String>, usize>>,
  }

  let acc = history
    .par_iter()
    .fold(
      || TokenSeqAcc {
        total: 0,
        unigram_counts: HashMap::new(),
        ngram_counts: (0..max_len.saturating_sub(1))
          .map(|_| HashMap::new())
          .collect(),
      },
      |mut acc, (shellname, command)| {
        let tokens = tokenize_index(shellname, command);
        if tokens.len() < 2 {
          return acc;
        }
        let normalized: Vec<String> = tokens.into_iter().map(|t| t.normalized).collect();
        acc.total += normalized.len();
        for tok in &normalized {
          *acc.unigram_counts.entry(tok.clone()).or_insert(0) += 1;
        }
        for n in 2..=max_len {
          if normalized.len() < n {
            continue;
          }
          for win in normalized.windows(n) {
            let seq = win.to_vec();
            *acc.ngram_counts[n - 2].entry(seq).or_insert(0) += 1;
          }
        }
        acc
      },
    )
    .reduce(TokenSeqAcc::default, |mut left, right| {
      left.total += right.total;
      for (tok, count) in right.unigram_counts {
        *left.unigram_counts.entry(tok).or_insert(0) += count;
      }
      if left.ngram_counts.is_empty() {
        left.ngram_counts = right.ngram_counts;
      } else {
        for (idx, right_map) in right.ngram_counts.into_iter().enumerate() {
          if let Some(left_map) = left.ngram_counts.get_mut(idx) {
            for (seq, count) in right_map {
              *left_map.entry(seq).or_insert(0) += count;
            }
          }
        }
      }
      left
    });

  let total = acc.total;
  let unigram_counts = acc.unigram_counts;
  let ngram_counts = acc.ngram_counts;

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

    for (idx, counts) in ngram_counts.iter().enumerate() {
      let n = idx + 2;
      for (sequence, count) in counts {
        if sequence.is_empty() {
          continue;
        }
        let support = *count;
        if support < config.min_support {
          continue;
        }
        let prefix = if n == 2 {
          *unigram_counts.get(&sequence[0]).unwrap_or(&0)
        } else {
          let prefix_seq = sequence[..n - 1].to_vec();
          *ngram_counts
            .get(idx - 1)
            .and_then(|prefix_map| prefix_map.get(&prefix_seq))
            .unwrap_or(&0)
        };
        if prefix == 0 {
          continue;
        }
        let confidence = support as f64 / prefix as f64;
        let last = &sequence[sequence.len() - 1];
        let base = *unigram_counts.get(last).unwrap_or(&0) as f64 / total_f;
        if base <= 0.0 {
          continue;
        }
        let lift = confidence / base;
        if confidence < config.min_confidence || lift < config.min_lift {
          continue;
        }
        let sequence_json = serde_json::to_string(sequence)?;
        conn
          .execute(
            "INSERT INTO token_sequence_stats (sequence_json, support, confidence, lift, prefix_len)
             VALUES (?, ?, ?, ?, ?)",
            (sequence_json, support as i64, confidence, lift, (n - 1) as i64),
          )
          .await?;
        inserted += 1;
        if n == 2 {
          bigrams += 1;
        } else if n == 3 {
          trigrams += 1;
        }
      }
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
  shellname: &str,
  recent_commands: &[String],
  limit: usize,
) -> Result<Vec<SequenceCandidate>> {
  let config = SequenceConfig::default();
  let normalized_recent_commands = recent_commands
    .iter()
    .map(|cmd| normalize_sequence_command(shellname, cmd))
    .collect::<Vec<_>>();
  let recent_len = normalized_recent_commands.len();
  if recent_len == 0 {
    return Ok(Vec::new());
  }

  let max_prefix_len = config.max_len.saturating_sub(1).min(recent_len);
  if max_prefix_len == 0 {
    return Ok(Vec::new());
  }

  let mut candidates: Vec<SequenceCandidate> = Vec::new();
  let mut seen: HashSet<String> = HashSet::new();

  // Prefer the longest matching prefix, but back off to shorter prefixes if needed.
  for prefix_len in (1..=max_prefix_len).rev() {
    let prefix_slice = &normalized_recent_commands[recent_len - prefix_len..];
    let prefix_json = serde_json::to_string(prefix_slice)?;
    let seq_len = (prefix_len + 1) as i64;

    let mut rows = conn
      .query(
        "SELECT sequence_json, support, confidence, lift, sequence_len
         FROM sequence_stats
         WHERE prefix_json = ? AND sequence_len = ?
         ORDER BY lift DESC
         LIMIT ?",
        libsql::params![prefix_json, seq_len, limit as i64],
      )
      .await?;

    while let Some(row) = rows.next().await? {
      let sequence_json = row.get::<String>(0)?;
      let support = row.get::<i64>(1)? as usize;
      let confidence = row.get::<f64>(2)?;
      let lift = row.get::<f64>(3)?;
      let sequence_len = row.get::<i64>(4)? as usize;
      if support < config.min_support
        || confidence < config.min_confidence
        || lift < config.min_lift
      {
        continue;
      }

      let sequence: Vec<String> = serde_json::from_str(&sequence_json)?;
      if sequence_len != prefix_len + 1 || sequence.len() != sequence_len {
        continue;
      }
      let next = sequence.last().cloned().unwrap_or_default();
      if next.is_empty() || !seen.insert(next.clone()) {
        continue;
      }
      candidates.push(SequenceCandidate {
        command: next,
        confidence,
        lift,
        support,
        prefix_len,
      });
      if candidates.len() >= limit {
        return Ok(candidates);
      }
    }

    // If we found any candidates for the longest prefix, stop backing off.
    if !candidates.is_empty() {
      break;
    }
  }

  Ok(candidates)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::db::{init, insert_invocation, open_db, update_stats_for_invocation};
  use crate::shell_history::Invocation;

  async fn insert_cmd(conn: &libsql::Connection, command: &str, ts: i64) {
    let inv = Invocation {
      command: command.to_string(),
      expanded_command: command.to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/tmp".to_string()),
      workspace: None,
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

  #[tokio::test]
  async fn test_sequence_stats_update_online() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    let invs = [
      ("git status", 1),
      ("git add .", 2),
      ("git status", 3),
      ("git add .", 4),
    ];

    for (cmd, ts) in invs {
      let inv = Invocation {
        command: cmd.to_string(),
        expanded_command: cmd.to_string(),
        shellname: "zsh".to_string(),
        working_directory: Some("/tmp".to_string()),
        workspace: None,
        hostname: Some("host".to_string()),
        username: Some("user".to_string()),
        exit_status: Some(0),
        start_unix_timestamp: Some(ts),
        end_unix_timestamp: Some(ts + 1),
        session_id: 1,
      };
      assert!(insert_invocation(&db.conn, &inv).await.unwrap());
      update_stats_for_invocation(&db.conn, &inv).await.unwrap();
    }

    let expected_sequence = vec![
      normalize_sequence_command("zsh", "git status"),
      normalize_sequence_command("zsh", "git add ."),
    ];
    let sequence_json = serde_json::to_string(&expected_sequence).unwrap();
    let mut rows = db
      .conn
      .query(
        "SELECT support, sequence_len FROM sequence_stats WHERE sequence_json = ?",
        libsql::params![sequence_json],
      )
      .await
      .unwrap();
    let row = rows.next().await.unwrap().expect("expected row");
    let support = row.get::<i64>(0).unwrap();
    let sequence_len = row.get::<i64>(1).unwrap();
    assert!(support >= 2);
    assert_eq!(sequence_len, 2);

    let recent = vec!["git status".to_string()];
    let candidates = candidates_from_sequences(&db.conn, "zsh", &recent, 10)
      .await
      .unwrap();
    let expected_next = normalize_sequence_command("zsh", "git add .");
    assert!(candidates.iter().any(|cand| cand.command == expected_next));
  }
}
