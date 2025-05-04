//! # MarkovChain: Predicting Shell Commands with First-Order Markov Chain
//!
//! This module implements a first-order Markov chain model for predicting the next shell command
//! a user might run, based solely on the previous command. The model learns transition frequencies
//! between commands and ranks likely next commands accordingly.
//!
//! ## What is a Markov Chain?
//! A first-order Markov chain predicts the next state based only on the current state. Here, each
//! state is a shell command, and transitions represent the frequency of one command following another.
//!
//! ## How it Works
//! 1. **Training**: For every consecutive command pair in history, the model increments the transition count.
//! 2. **Transition Table**: Maps each command (state) to a map of possible next commands and their counts.
//! 3. **Prediction**: Looks up the most recent command (or falls back to earlier history) and sorts
//!    possible next commands by frequency, returning the top N.
//!
//! ## Example
//! If history: ["echo foo", "cd project", "ls", "cd project", "ls"], then transitions:
//! - "echo foo" -> "cd project" (1)
//! - "cd project" -> "ls" (2)
//! - "ls" -> "cd project" (1)
//!
//! Requesting predictions after "cd project" yields ["ls"].

use super::PredictionModel;
use crate::Result;
use crate::db::{load_model, save_model};
use crate::shell_history::Invocation;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::{from_slice, to_vec};
use std::collections::HashMap;

/// MarkovChain: maintains transition counts from one command to the next
#[derive(Serialize, Deserialize, Clone)]
pub struct MarkovChain {
  transitions: HashMap<String, HashMap<String, usize>>,
}

impl MarkovChain {
  /// Create an empty chain
  pub fn new() -> Self {
    Self {
      transitions: HashMap::new(),
    }
  }

  /// Train on a sequence of Invocations (chronological order)
  pub fn train<I>(&mut self, invocations: I) -> Result<()>
  where
    I: IntoIterator<Item = Invocation>,
  {
    let mut prev: Option<String> = None;
    for inv in invocations {
      let cmd = String::from_utf8_lossy(&inv.command).to_string();
      if let Some(prev_cmd) = prev.take() {
        let entry = self
          .transitions
          .entry(prev_cmd)
          .or_insert_with(HashMap::new);
        *entry.entry(cmd.clone()).or_insert(0) += 1;
        prev = Some(cmd);
      } else {
        prev = Some(cmd);
      }
    }
    Ok(())
  }

  /// Predict next commands based on last invocation
  pub fn predict(
    &self,
    recent_history: &[Invocation],
    max_predictions: usize,
  ) -> Result<Vec<String>> {
    // Find the most recent command in history with outgoing transitions
    for inv in recent_history.iter().rev() {
      let key = String::from_utf8_lossy(&inv.command).to_string();
      if let Some(next_map) = self.transitions.get(&key) {
        let mut pairs: Vec<(&String, &usize)> = next_map.iter().collect();
        pairs.sort_by(|a, b| b.1.cmp(a.1));
        let preds = pairs.into_iter()
          .take(max_predictions)
          .map(|(cmd, _)| cmd.clone())
          .collect();
        return Ok(preds);
      }
    }
    Ok(Vec::new())
  }

  /// Serialize and save to the `models` table under key "markov"
  pub fn save_to_db(&self, conn: &mut Connection) -> Result<()> {
    let data = to_vec(self)?;
    save_model(conn, "markov", &data)?;
    Ok(())
  }

  /// Load from DB or return empty chain
  pub fn load_from_db(conn: &mut Connection) -> Result<Self> {
    if let Some(data) = load_model(conn, "markov")? {
      let chain: MarkovChain = from_slice(&data)?;
      Ok(chain)
    } else {
      Ok(MarkovChain::new())
    }
  }
}

impl PredictionModel for MarkovChain {
  fn train<I>(&mut self, invocations: I) -> Result<()>
  where
    I: IntoIterator<Item = Invocation>,
  {
    MarkovChain::train(self, invocations)
  }

  fn predict(&self, recent_history: &[Invocation], max_predictions: usize) -> Result<Vec<String>> {
    MarkovChain::predict(self, recent_history, max_predictions)
  }

  fn update<I>(&mut self, invocations: I) -> Result<()>
  where
    I: IntoIterator<Item = Invocation>,
  {
    MarkovChain::train(self, invocations)
  }
}

#[cfg(test)]
mod tests {
  use super::MarkovChain;
  use crate::db::{connect, init_table};
  use crate::shell_history::Invocation;
  use bstr::BString;

  fn make_invocation(cmd: &str) -> Invocation {
    Invocation {
      command: BString::from(cmd),
      shellname: "bash".to_string(),
      working_directory: None,
      hostname: None,
      username: None,
      exit_status: Some(0),
      start_unix_timestamp: Some(0),
      end_unix_timestamp: Some(0),
      session_id: 0,
    }
  }

  #[test]
  fn test_predict_empty() {
    let chain = MarkovChain::new();
    let preds = chain.predict(&[], 3).unwrap();
    assert!(preds.is_empty());
  }

  #[test]
  fn test_single_invocation_no_transition() {
    let mut chain = MarkovChain::new();
    chain.train(vec![make_invocation("a")]).unwrap();
    let preds = chain.predict(&[make_invocation("a")], 3).unwrap();
    assert!(preds.is_empty());
  }

  #[test]
  fn test_two_invocations_transition() {
    let invs = vec![make_invocation("a"), make_invocation("b")];
    let mut chain = MarkovChain::new();
    chain.train(invs.clone()).unwrap();
    let preds = chain.predict(&invs, 3).unwrap();
    assert_eq!(preds, vec!["b".to_string()]);
  }

  #[test]
  fn test_transition_counts_and_limit() {
    let invs = vec![
      make_invocation("a"),
      make_invocation("b"),
      make_invocation("a"),
      make_invocation("b"),
      make_invocation("a"),
      make_invocation("c"),
    ];
    let mut chain = MarkovChain::new();
    chain.train(invs.clone()).unwrap();
    let preds = chain.predict(&invs, 2).unwrap();
    assert_eq!(preds, vec!["b".to_string(), "c".to_string()]);
  }

  #[test]
  fn test_save_load_roundtrip() {
    // setup in-memory DB with models table
    let mut conn = connect(":memory:").unwrap();
    let mut tx = conn.transaction().unwrap();
    init_table(&mut tx).unwrap();
    tx.commit().unwrap();

    let invs = vec![make_invocation("x"), make_invocation("y")];
    let mut chain = MarkovChain::new();
    chain.train(invs.clone()).unwrap();
    chain.save_to_db(&mut conn).unwrap();

    let loaded = MarkovChain::load_from_db(&mut conn).unwrap();
    let preds = loaded.predict(&invs, 5).unwrap();
    assert_eq!(preds, vec!["y".to_string()]);
  }
}
