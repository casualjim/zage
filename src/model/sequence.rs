//! # Sequence Detection and Scoring

//! This module focuses on identifying and scoring recurring sequences of commands
//! within the user's shell history. Unlike N-gram or Markov models that predict
//! the *next* command based on a fixed-size context, sequence detection looks
//! for meaningful, longer patterns of usage.

//! ## Key Concepts:

//! - **Sequence:** An ordered list of commands (e.g., `["git fetch", "git rebase", "git push"]`).
//! - **Support:** How often a specific sequence appears in the history.
//! - **Confidence:** The conditional probability of the sequence occurring given a preceding context (e.g., Confidence("B" follows "A") = Count("A", "B") / Count("A")).
//! - **Lift:** How much more likely the sequence is compared to its baseline frequency, indicating how "surprising" or correlated the pattern is.
//! - **Scoring:** Combining metrics like support, confidence, and lift into a single score to rank sequences.
//! - **Thresholds:** Minimum criteria (e.g., minimum support, minimum confidence) a sequence must meet to be considered significant.

use crate::{Result, db, err::ZageError};
use rusqlite::Connection;

/// Represents a scored sequence of commands.
#[derive(Debug, Clone, PartialEq)]
pub struct SequenceScore {
  pub sequence: Vec<String>,
  pub support: usize,
  pub confidence: f64,
  pub lift: f64,
  pub length: usize,
  pub context_json: Option<String>,
  pub commands: Vec<String>,
}

impl SequenceScore {
  /// Parse the context_json into a SequenceContext object
  pub fn parse_context(&self) -> Option<super::sequence_context::SequenceContext> {
    self.context_json
      .as_ref()
      .and_then(|json| Self::parse_context_from_json(json))
  }

  /// Parse context information from JSON string into a SequenceContext object
  pub fn parse_context_from_json(json_str: &str) -> Option<super::sequence_context::SequenceContext> {
    // Attempt to parse as JSON
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_str) {
      if !json.is_object() {
        return None;
      }

      // Extract working directory
      let cwd = json
        .get("working_directory")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

      // Extract hostname
      let hostname = json
        .get("hostname")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

      // Extract username
      let username = json
        .get("username")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

      // Extract exit status pattern
      let exit_status_pattern = json
        .get("exit_status")
        .and_then(|v| {
          if v.is_array() {
            let arr = v.as_array()?;
            let mut pattern = Vec::new();
            for item in arr {
              pattern.push(item.as_i64());
            }
            Some(pattern)
          } else {
            None
          }
        })
        .unwrap_or_default();

      // Extract session ID
      let session_id = json
        .get("session_id")
        .and_then(|v| v.as_i64());

      // Extract time info
      let (_start_time, _end_time) = if let Some(time_info) = json.get("time_info") {
        if time_info.is_object() {
          let start = time_info
            .get("start_time")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
          let end = time_info
            .get("end_time")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
          (start, end)
        } else {
          (0, 0)
        }
      } else {
        (0, 0)
      };

      // Create a SequenceContext
      let ctx = super::sequence_context::SequenceContext {
        base: super::context::Context {
          cwd: cwd.unwrap_or_default(),
          hostname,
          username,
          exit_status: None, // Not relevant for sequence context
        },
        temporal: super::sequence_context::TemporalContext {
          time_of_day: super::sequence_context::TimeOfDay::Morning, // Default to Morning
          day_of_week: 1, // Default to Monday (1)
          is_weekend: false, // Would need to calculate from timestamp
        },
        execution: super::sequence_context::ExecutionContext {
          duration_pattern: Vec::new(), // Would need more data
          exit_status_pattern,
          output_characteristics: Vec::new(), // Would need more data
        },
        session_id,
        metadata: std::collections::HashMap::new(),
        commands: Vec::new(), // Would need to be populated elsewhere
      };

      Some(ctx)
    } else {
      None
    }
  }

  /// Get the context similarity threshold to use for matching
  pub fn context_similarity_threshold(&self) -> f64 {
    0.7 // Default threshold
  }

  /// Create a SequenceScore from a RawSequenceScore
  pub fn from_raw(raw: &crate::db::RawSequenceScore) -> Result<Self> {
    // Parse the sequence JSON
    let sequence: Vec<String> = serde_json::from_str(&raw.sequence_json)
      .map_err(|e| ZageError::SerializationError(e))?;
    let size = sequence.len();
    
    Ok(SequenceScore {
      sequence: sequence.clone(),
      support: raw.support,
      confidence: raw.confidence,
      lift: raw.lift,
      length: size,
      context_json: raw.context_json.clone(),
      commands: sequence, // Use sequence as commands for now
    })
  }
}

/// Analyzes sequences via SQL and stores the results in the database.
pub fn analyze_and_store_sequences(
  conn: &mut Connection,
  min_support: usize,
  min_confidence: f64,
  min_lift: f64,
) -> Result<()> {
  let mut tx = conn.transaction()?;
  db::clear_sequence_scores_table(&mut tx)?;
  db::analyze_sequences(&mut tx, min_support, min_confidence, min_lift)?;
  tx.commit()?;
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::db;
  use crate::db::import_history;
  use crate::shell_history::Invocation;
  use rusqlite::Connection;
  use std::time::{SystemTime, UNIX_EPOCH};

  fn create_invocations(commands: Vec<&str>) -> Vec<Invocation> {
    let now = SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap()
      .as_secs() as i64;
    commands
      .into_iter()
      .enumerate()
      .map(|(i, cmd)| Invocation {
        command: cmd.to_string(),
        shellname: "test_shell".to_string(),
        working_directory: None,
        hostname: None,
        username: None,
        exit_status: Some(0),
        start_unix_timestamp: Some(now + i as i64),
        end_unix_timestamp: Some(now + i as i64 + 1),
        session_id: 1,
      })
      .collect()
  }

  #[test]
  fn test_sql_sequence_analysis() {
    let mut conn = Connection::open_in_memory().unwrap();
    let mut tx = conn.transaction().unwrap();
    db::init_table(&mut tx).unwrap();
    tx.commit().unwrap();

    let invs = create_invocations(vec!["X", "Y", "Z", "X", "Y"]);
    import_history(&mut conn, invs).unwrap();
    analyze_and_store_sequences(&mut conn, 2, 0.0, 0.0).unwrap();

    let raw_scores = db::get_sequence_scores(&mut conn, 10).unwrap();
    assert_eq!(raw_scores.len(), 1);
    
    let scores: Vec<SequenceScore> = raw_scores.iter()
      .filter_map(|raw| SequenceScore::from_raw(raw).ok())
      .collect();
    assert_eq!(scores.len(), 1);
    
    let s = &scores[0];
    assert_eq!(s.sequence, vec!["X".to_string(), "Y".to_string()]);
    assert_eq!(s.support, 2);
  }

  #[test]
  fn test_sql_sequence_exhaustive() {
    let mut conn = Connection::open_in_memory().unwrap();
    let mut tx = conn.transaction().unwrap();
    db::init_table(&mut tx).unwrap();
    tx.commit().unwrap();

    let invs = create_invocations(vec!["X", "Y", "Z", "X", "Y", "Z", "X", "Y", "Z"]);
    import_history(&mut conn, invs).unwrap();
    analyze_and_store_sequences(&mut conn, 2, 0.0, 0.0).unwrap();

    // Limit test: top 2 sequences
    let raw_top2 = db::get_sequence_scores(&mut conn, 2).unwrap();
    assert_eq!(raw_top2.len(), 2);
    
    let top2: Vec<SequenceScore> = raw_top2.iter()
      .filter_map(|raw| SequenceScore::from_raw(raw).ok())
      .collect();
    assert_eq!(top2.len(), 2);

    // Full result: expect 3 bigrams + 3 trigrams
    let raw_all = db::get_sequence_scores(&mut conn, 10).unwrap();
    assert_eq!(raw_all.len(), 6);
    
    let all: Vec<SequenceScore> = raw_all.iter()
      .filter_map(|raw| SequenceScore::from_raw(raw).ok())
      .collect();
    assert_eq!(all.len(), 6);
    
    let sequences: Vec<Vec<String>> = all.iter().map(|s| s.sequence.clone()).collect();
    let expected = vec![
      vec!["X".to_string(), "Y".to_string()],
      vec!["Y".to_string(), "Z".to_string()],
      vec!["Z".to_string(), "X".to_string()],
      vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
      vec!["Y".to_string(), "Z".to_string(), "X".to_string()],
      vec!["Z".to_string(), "X".to_string(), "Y".to_string()],
    ];
    for seq in expected {
      assert!(sequences.contains(&seq));
    }
  }

  #[test]
  fn test_sql_sequence_filters() {
    let mut conn = Connection::open_in_memory().unwrap();
    let mut tx = conn.transaction().unwrap();
    db::init_table(&mut tx).unwrap();
    tx.commit().unwrap();

    let invs = create_invocations(vec!["A", "B", "A", "B", "A"]);
    import_history(&mut conn, invs).unwrap();

    // No sequence meets support >= 3
    analyze_and_store_sequences(&mut conn, 3, 0.0, 0.0).unwrap();
    assert!(db::get_sequence_scores(&mut conn, 10).unwrap().is_empty());

    // No sequence meets confidence > 1.0
    analyze_and_store_sequences(&mut conn, 1, 1.1, 0.0).unwrap();
    assert!(db::get_sequence_scores(&mut conn, 10).unwrap().is_empty());

    // No sequence meets lift > 10.0
    analyze_and_store_sequences(&mut conn, 1, 0.0, 10.0).unwrap();
    assert!(db::get_sequence_scores(&mut conn, 10).unwrap().is_empty());
  }
}
