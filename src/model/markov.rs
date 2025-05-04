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
use crate::model::context::Context;
use crate::shell_history::Invocation;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::{from_slice, to_vec};
use std::collections::HashMap;

/// MarkovChain: maintains transition counts from one command to the next
#[derive(Serialize, Deserialize, Clone)]
pub struct MarkovChain {
  transitions: HashMap<String, HashMap<String, usize>>,
  /// Context-specific transition counts
  /// Maps (Context, prev_command) -> { next_command: count }
  context_transitions: HashMap<(Context, String), HashMap<String, usize>>,
  /// Whether to use context for predictions
  use_context: bool,
}

impl MarkovChain {
  /// Create an empty chain
  pub fn new() -> Self {
    Self {
      transitions: HashMap::new(),
      context_transitions: HashMap::new(),
      use_context: true,
    }
  }

  /// Enable or disable context for predictions
  pub fn set_use_context(&mut self, use_context: bool) {
    self.use_context = use_context;
  }

  /// Train on a sequence of Invocations (chronological order)
  pub fn train<I>(&mut self, invocations: I) -> Result<()>
  where
    I: IntoIterator<Item = Invocation>,
  {
    let mut prev: Option<String> = None;
    for inv in invocations {
      let cmd = String::from_utf8_lossy(&inv.command).to_string();
      let context_value = Context::from_invocation(&inv); // Returns Context

      if let Some(prev_cmd) = prev.take() {
        // Update global transitions
        let entry = self
          .transitions
          .entry(prev_cmd.clone())
          .or_insert_with(HashMap::new);
        *entry.entry(cmd.clone()).or_insert(0) += 1;

        // Update context-specific transitions if enabled
        if self.use_context {
          // No need for `if let Some(ctx) = context_value` anymore
          let key = (context_value, prev_cmd); // Use context_value directly
          let context_entry = self
            .context_transitions
            .entry(key)
            .or_insert_with(HashMap::new);
          *context_entry.entry(cmd.clone()).or_insert(0) += 1;
        }
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
      let key_cmd = String::from_utf8_lossy(&inv.command).to_string();
      let context_value = Context::from_invocation(inv);

      // 1. Try context-specific prediction if enabled
      if self.use_context {
        let context_key = (context_value, key_cmd.clone());
        if let Some(next_map) = self.context_transitions.get(&context_key) {
          if !next_map.is_empty() {
            let mut pairs: Vec<(&String, &usize)> = next_map.iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(a.1));
            let preds = pairs
              .into_iter()
              .take(max_predictions)
              .map(|(cmd, _)| cmd.clone())
              .collect();
            return Ok(preds);
          }
        }
      }

      // 2. Fallback to global prediction
      if let Some(next_map) = self.transitions.get(&key_cmd) {
        if !next_map.is_empty() {
          // Check if the map is not empty
          let mut pairs: Vec<(&String, &usize)> = next_map.iter().collect();
          pairs.sort_by(|a, b| b.1.cmp(a.1));
          let preds = pairs
            .into_iter()
            .take(max_predictions)
            .map(|(cmd, _)| cmd.clone())
            .collect();
          return Ok(preds);
        }
      }
    }
    // No predictions found for any command in recent history
    Ok(Vec::new())
  }

  /// Serialize and save to the `models` table under key "markov"
  pub fn save_to_db(&self, conn: &mut Connection) -> Result<()> {
    // Convert to serializable format
    let serializable = SerializableMarkovChain {
      use_context: self.use_context,
      transitions: self
        .transitions
        .iter()
        .map(|(prev, next_map)| SerializableTransitionEntry {
          prev_command: prev.clone(),
          next_commands: next_map.clone(),
        })
        .collect(),
      context_transitions: self
        .context_transitions
        .iter()
        .map(
          |((ctx, prev), next_map)| SerializableContextTransitionEntry {
            context: ctx.clone(),
            prev_command: prev.clone(),
            next_commands: next_map.clone(),
          },
        )
        .collect(),
    };

    let data = to_vec(&serializable)?; // Serialize the helper struct
    save_model(conn, "markov", &data)?;
    Ok(())
  }

  /// Load from DB or return empty chain
  pub fn load_from_db(conn: &mut Connection) -> Result<Self> {
    if let Some(data) = load_model(conn, "markov")? {
      // Deserialize into the helper struct
      let serializable: SerializableMarkovChain = from_slice(&data)?;

      // Rebuild the MarkovChain from the serializable format
      let mut chain = MarkovChain::new();
      chain.use_context = serializable.use_context;

      chain.transitions = serializable
        .transitions
        .into_iter()
        .map(|entry| (entry.prev_command, entry.next_commands))
        .collect();

      chain.context_transitions = serializable
        .context_transitions
        .into_iter()
        .map(|entry| ((entry.context, entry.prev_command), entry.next_commands))
        .collect();

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

// --- Helper structs for serialization ---

/// Entry for transitions serialization
#[derive(Serialize, Deserialize)]
struct SerializableTransitionEntry {
  prev_command: String,
  next_commands: HashMap<String, usize>,
}

/// Entry for context-specific transitions serialization
#[derive(Serialize, Deserialize)]
struct SerializableContextTransitionEntry {
  context: Context,
  prev_command: String,
  next_commands: HashMap<String, usize>,
}

/// Serializable version of the MarkovChain
#[derive(Serialize, Deserialize)]
struct SerializableMarkovChain {
  use_context: bool,
  transitions: Vec<SerializableTransitionEntry>,
  context_transitions: Vec<SerializableContextTransitionEntry>,
}

// --- End helper structs ---

#[cfg(test)]
mod tests {
  use super::MarkovChain;
  use crate::db::{connect, init_table};
  use crate::shell_history::Invocation;
  use bstr::BString;

  // Updated make_invocation to include optional context
  fn make_invocation(cmd: &str, cwd: Option<&str>, hostname: Option<&str>) -> Invocation {
    Invocation {
      command: BString::from(cmd),
      shellname: "bash".to_string(),
      working_directory: cwd.map(BString::from),
      hostname: hostname.map(BString::from),
      username: None, // Keep username None for simplicity for now
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
    chain.train(vec![make_invocation("a", None, None)]).unwrap(); // Updated call
    let preds = chain
      .predict(&[make_invocation("a", None, None)], 3)
      .unwrap(); // Updated call
    assert!(preds.is_empty());
  }

  #[test]
  fn test_two_invocations_transition() {
    let invs = vec![
      make_invocation("a", None, None),
      make_invocation("b", None, None),
    ]; // Updated calls
    let mut chain = MarkovChain::new();
    chain.train(invs.clone()).unwrap();
    let preds = chain.predict(&invs, 3).unwrap();
    assert_eq!(preds, vec!["b".to_string()]);
  }

  #[test]
  fn test_transition_counts_and_limit() {
    let invs = vec![
      make_invocation("a", None, None), // Updated calls
      make_invocation("b", None, None),
      make_invocation("a", None, None),
      make_invocation("b", None, None),
      make_invocation("a", None, None),
      make_invocation("c", None, None),
    ];
    let mut chain = MarkovChain::new();
    chain.train(invs.clone()).unwrap();
    // Predict based on the last 'a'
    let history_slice = &[invs[4].clone()];
    let preds = chain.predict(history_slice, 2).unwrap();
    // Expect 'b' (count 2) then 'c' (count 1)
    assert_eq!(preds, vec!["b".to_string(), "c".to_string()]);
  }

  #[test]
  fn test_save_load_roundtrip() {
    // setup in-memory DB with models table
    let mut conn = connect(":memory:").unwrap();
    let mut tx = conn.transaction().unwrap();
    init_table(&mut tx).unwrap();
    tx.commit().unwrap();

    // Invocations with context
    let invs = vec![
      make_invocation("x", Some("/dir1"), Some("host1")),
      make_invocation("y", Some("/dir1"), Some("host1")),
      make_invocation("x", Some("/dir2"), None),
      make_invocation("z", Some("/dir2"), None),
    ];
    let mut chain = MarkovChain::new();
    chain.train(invs.clone()).unwrap();
    chain.save_to_db(&mut conn).unwrap();

    let loaded = MarkovChain::load_from_db(&mut conn).unwrap();

    // Verify global transitions
    let preds_global_x = loaded.predict(&[invs[0].clone()], 5).unwrap(); // Predict after x in /dir1
    assert!(preds_global_x.contains(&"y".to_string()));
    let preds_global_x_no_context = loaded.predict(&[invs[2].clone()], 5).unwrap(); // Predict after x in /dir2
    assert!(preds_global_x_no_context.contains(&"z".to_string()));

    // Verify context transitions were loaded (implicitly checked by predict if use_context is true)
    assert_eq!(loaded.use_context, true); // Default should be true
    // There should be transitions for both 'x' and 'y'
    assert_eq!(loaded.transitions.len(), 2);
    assert!(loaded.transitions.contains_key("x"));
    assert!(loaded.transitions.contains_key("y"));
    // 'x' transitions to 'y' and 'z'
    assert_eq!(loaded.transitions.get("x").unwrap().len(), 2);
    // 'y' transitions to 'x'
    assert_eq!(loaded.transitions.get("y").unwrap().len(), 1);

    // Check context transitions count (should include 'y' context too)
    assert_eq!(loaded.context_transitions.len(), 3);

    // Explicitly check context prediction after load
    let history_ctx1 = &[make_invocation("x", Some("/dir1"), Some("host1"))];
    let preds_ctx1 = loaded.predict(history_ctx1, 3).unwrap();
    assert_eq!(preds_ctx1, vec!["y".to_string()]); // Context1: x -> y

    let history_ctx2 = &[make_invocation("x", Some("/dir2"), None)];
    let preds_ctx2 = loaded.predict(history_ctx2, 3).unwrap();
    assert_eq!(preds_ctx2, vec!["z".to_string()]); // Context2: x -> z
  }

  // New test for context-aware prediction
  #[test]
  fn test_context_aware_prediction() {
    let mut chain = MarkovChain::new();

    // Train with context
    let invs = vec![
      // Global transition a -> b
      make_invocation("a", Some("/home/user"), Some("host1")),
      make_invocation("b", Some("/home/user"), Some("host1")),
      // Context1 transition a -> c
      make_invocation("a", Some("/project1"), Some("host1")),
      make_invocation("c", Some("/project1"), Some("host1")),
      // Context2 transition a -> d
      make_invocation("a", Some("/home/user"), Some("host2")),
      make_invocation("d", Some("/home/user"), Some("host2")),
      // Another global a -> b to increase its count
      make_invocation("a", Some("/other"), None),
      make_invocation("b", Some("/other"), None),
    ];
    chain.train(invs.clone()).unwrap();

    // Predict in Context1: Expect 'c'
    let history_ctx1 = &[make_invocation("a", Some("/project1"), Some("host1"))];
    let preds_ctx1 = chain.predict(history_ctx1, 3).unwrap();
    assert_eq!(preds_ctx1, vec!["c".to_string()]);

    // Predict in Context2: Expect 'd'
    let history_ctx2 = &[make_invocation("a", Some("/home/user"), Some("host2"))];
    let preds_ctx2 = chain.predict(history_ctx2, 3).unwrap();
    assert_eq!(preds_ctx2, vec!["d".to_string()]);

    // Predict in a different context (should use global): Expect 'b'
    let history_global = &[make_invocation("a", Some("/another/dir"), Some("host3"))];
    let preds_global = chain.predict(history_global, 1).unwrap();
    assert_eq!(preds_global, vec!["b".to_string()]); // Global 'a' -> 'b' has count 2

    // Disable context and predict: Expect 'b'
    chain.set_use_context(false);
    let preds_no_context = chain.predict(history_ctx1, 1).unwrap(); // Use Context1 history
    assert_eq!(preds_no_context, vec!["b".to_string()]);
  }
}
