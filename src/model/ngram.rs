//!
//! # NGramModel: Predicting Shell Commands with N-grams
//!
//! This module implements an N-gram model for predicting the next shell command a user might run, based on their command history.
//! It is designed to be accessible for those new to Rust, N-grams, or command prediction systems.
//!
//! ## What is an N-gram Model?
//!
//! An N-gram model is a probabilistic model that predicts the next item in a sequence based on the previous N-1 items.
//! In this context, each "item" is a shell command (as a string), and the model learns which commands tend to follow others.
//!
//! ## How it Works
//!
//! 1. **Tokenization & Extraction**: The model processes command history as a sequence of strings, extracting overlapping groups of N commands (n-grams).
//!    - Each n-gram consists of a context (the previous N-1 commands) and a next command.
//!    - Optionally, the working directory (cwd) is recorded for each n-gram.
//!
//! 2. **Frequency Tables**:
//!    - The model maintains two tables:
//!      - **Global Frequency Table**: Maps each context to a map of possible next commands and how often each occurred.
//!      - **Context-specific Frequency Table**: Like the global table, but also keyed by the context, to capture context-sensitive command habits.
//!
//! 3. **Prediction**:
//!    - Given the most recent N-1 commands (and optionally the current context), the model ranks possible next commands by their observed frequency.
//!    - Context-specific matches are weighted higher, so predictions are more relevant to the user's current context.
//!
//! ## Concrete Example
//!
//! Suppose n = 3 and the user runs:
//!
//! 1. "echo foo" (wd: "/home/user")
//! 2. "cd project" (wd: "/home/user")
//! 3. "ls" (wd: "/home/user/project")
//! 4. "cargo build" (wd: "/home/user/project")
//!
//! The model extracts these n-grams:
//!
//! ```ignore
//!     (["echo foo", "cd project"], "ls", Some("/home/user/project"))
//!     (["cd project", "ls"], "cargo build", Some("/home/user/project"))
//! ```
//!
//! These update the tables as follows:
//!
//! **Global frequencies:**
//!   ["echo foo", "cd project"] => { "ls": 1 }
//!   ["cd project", "ls"]      => { "cargo build": 1 }
//!
//! **Context-specific frequencies:**
//!   ("/home/user/project", ["echo foo", "cd project"]) => { "ls": 1 }
//!   ("/home/user/project", ["cd project", "ls"])      => { "cargo build": 1 }
//!
//! ## How Predictions are Made
//!
//! - When predicting, the model first checks for context-specific matches (if enabled), giving them higher weight.
//! - If not enough matches are found, it falls back to the global table.
//! - Results are sorted by probability and the top-N are returned.
//!
//! ## Why Context?
//!
//! Many commands are only relevant in certain contexts (e.g., `cargo build` in a Rust project). By tracking context-specific habits, the model gives more accurate, context-aware predictions.
//!
//! ## Intended Audience
//!
//! This module is designed for learners and those new to Rust or predictive modeling. It aims for clarity, idiomatic Rust, and thorough documentation to help you understand and extend the code.
//!
//! See the tests at the bottom of this file for more concrete usage examples.
//!
use std::collections::{BTreeMap, HashMap};

use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use super::PredictionModel;
use crate::Result;
use crate::db::{load_model, save_model};
use crate::model::context::Context;
use crate::shell_history::Invocation;

/// N-gram model for predicting shell commands based on command history
#[derive(Serialize, Deserialize, Clone)]
pub struct NGramModel {
  /// The maximum length of n-grams to consider
  n: usize,
  /// Frequency table for n-grams
  /// Maps from a sequence of n-1 commands to a map of possible next commands and their frequencies
  frequencies: BTreeMap<Vec<String>, HashMap<String, usize>>,
  /// Context-specific frequency table
  /// Maps from (Context, context slice) to next commands and their frequencies
  context_frequencies: BTreeMap<(Context, Vec<String>), HashMap<String, usize>>,
  /// Total number of commands seen
  total_commands: usize,
  /// Whether to use context for predictions
  use_context: bool,
}

/// Entry for frequency table serialization
#[derive(Serialize, Deserialize)]
struct FrequencyEntry {
  context: Vec<String>,
  next_commands: HashMap<String, usize>,
}

/// Entry for context-specific frequency table serialization
#[derive(Serialize, Deserialize)]
struct SerializableContextFrequencyEntry {
  context: Context,
  context_slice: Vec<String>,
  next_commands: HashMap<String, usize>,
}

/// Serializable version of the model using entry lists
#[derive(Serialize, Deserialize)]
pub struct SerializableNGramModel {
  n: usize,
  frequencies: Vec<FrequencyEntry>,
  total_commands: usize,
  use_context: bool,
  context_frequencies: Vec<SerializableContextFrequencyEntry>,
}

impl NGramModel {
  /// Create a new N-gram model with the specified n value
  pub fn new(n: usize) -> Self {
    if n < 2 {
      panic!("N-gram model requires n >= 2");
    }

    Self {
      n,
      frequencies: BTreeMap::new(),
      context_frequencies: BTreeMap::new(),
      total_commands: 0,
      use_context: true,
    }
  }

  /// Get the n value for this model
  pub fn n(&self) -> usize {
    self.n
  }

  /// Enable or disable context for predictions
  pub fn set_use_context(&mut self, use_context: bool) {
    self.use_context = use_context;
  }

  /// Save the N-gram model to the database
  pub fn save_to_db(&self, conn: &mut Connection) -> Result<()> {
    let model_type = format!("ngram_n{}", self.n);

    // Convert to serializable entry lists
    let serializable = SerializableNGramModel {
      n: self.n,
      frequencies: self
        .frequencies
        .iter()
        .map(|(context, next_commands)| FrequencyEntry {
          context: context.clone(),
          next_commands: next_commands.clone(),
        })
        .collect(),
      total_commands: self.total_commands,
      use_context: self.use_context,
      context_frequencies: self
        .context_frequencies
        .iter()
        .map(
          |((ctx, slice), next_commands)| SerializableContextFrequencyEntry {
            context: ctx.clone(),
            context_slice: slice.clone(),
            next_commands: next_commands.clone(),
          },
        )
        .collect(),
    };

    // Serialize to JSON
    let model_data = serde_json::to_vec(&serializable)?;

    // Save to database
    save_model(conn, &model_type, &model_data)?;
    Ok(())
  }

  /// Load an N-gram model from the database
  pub fn load_from_db(conn: &mut Connection, n: usize) -> Result<Self> {
    let model_type = format!("ngram_n{}", n);

    if let Some(model_data) = load_model(conn, &model_type)? {
      // Deserialize entry lists and rebuild maps
      let serializable: SerializableNGramModel = serde_json::from_slice(&model_data)?;

      let mut model = NGramModel::new(serializable.n);
      // Frequencies
      model.frequencies = serializable
        .frequencies
        .into_iter()
        .map(|e| (e.context, e.next_commands))
        .collect();
      model.total_commands = serializable.total_commands;
      model.use_context = serializable.use_context;
      // Context-specific frequencies
      model.context_frequencies = serializable
        .context_frequencies
        .into_iter()
        .map(|e| ((e.context, e.context_slice), e.next_commands))
        .collect();

      Ok(model)
    } else {
      // No model found, create a new one
      Ok(Self::new(n))
    }
  }

  /// Convert an invocation to a command string for the model
  fn invocation_to_command(invocation: &Invocation) -> String {
    String::from_utf8_lossy(&invocation.command).to_string()
  }

  /// Find a context frequency entry or create a new one
  fn find_or_create_context_entry(
    &mut self,
    context: Context,
    context_slice: Vec<String>,
  ) -> &mut HashMap<String, usize> {
    // Clone the keys for lookup
    let key = (context.clone(), context_slice.clone());

    // Check if the entry exists and insert if not
    if !self.context_frequencies.contains_key(&key) {
      self.context_frequencies.insert(key.clone(), HashMap::new());
    }

    // Return a mutable reference to the entry
    self.context_frequencies.get_mut(&key).unwrap()
  }

  /// Find context frequency entries that match the given context and context slice
  fn find_context_entries(
    &self,
    context: &Context,
    context_slice: &[String],
  ) -> Vec<&HashMap<String, usize>> {
    self
      .context_frequencies
      .iter()
      .filter(|((context_entry, context_slice_entry), _)| {
        context_entry == context && context_slice_entry == context_slice
      })
      .map(|(_, next_commands)| next_commands)
      .collect()
  }

  /// Extract n-grams from a sequence of commands
  fn extract_ngrams<I>(&self, invocations: I) -> Vec<(Vec<String>, String, Context)>
  where
    I: IntoIterator<Item = Invocation>,
  {
    let invocations_vec: Vec<_> = invocations.into_iter().collect();
    let commands: Vec<String> = invocations_vec
      .iter()
      .map(|inv| Self::invocation_to_command(inv))
      .collect();

    let mut ngrams = Vec::new();

    if commands.len() < self.n {
      return ngrams;
    }

    for i in 0..=(commands.len() - self.n) {
      let context = commands[i..(i + self.n - 1)].to_vec();
      let next = commands[i + self.n - 1].clone();
      let ctx = Context::from_invocation(&invocations_vec[i + self.n - 1]);
      ngrams.push((context, next, ctx));
    }

    ngrams
  }

  /// Update the frequency table with a single n-gram
  fn update_frequency(
    &mut self,
    context_slice: Vec<String>,
    next: String,
    context_value: Option<Context>,
  ) {
    // Update global frequency table
    let entry = self
      .frequencies
      .entry(context_slice.clone())
      .or_insert_with(HashMap::new);
    *entry.entry(next.clone()).or_insert(0) += 1;

    // Update context-specific frequency table if provided
    if let Some(ctx) = context_value {
      let context_entry = self.find_or_create_context_entry(ctx, context_slice.clone());
      *context_entry.entry(next).or_insert(0) += 1;
    }
  }

  /// Get the most likely next commands given a context and optional context
  fn get_predictions(
    &self,
    context_slice: &[String],
    context_value: Option<&Context>,
    max_predictions: usize,
  ) -> Vec<(String, f64)> {
    let mut predictions = Vec::new();

    // If we have context-specific predictions and context is enabled, use them
    if self.use_context {
      if let Some(ctx) = context_value {
        let context_entries = self.find_context_entries(ctx, context_slice);

        for entry in context_entries {
          for (cmd, count) in entry {
            let probability = *count as f64 / self.total_commands as f64 * 2.0;
            predictions.push((cmd.clone(), probability));
          }
        }
      }
    }

    // Add global predictions
    if let Some(next_commands) = self.frequencies.get(&context_slice.to_vec()) {
      for (cmd, count) in next_commands {
        // If we already have this command from context-specific predictions, skip it
        if predictions.iter().any(|(c, _)| c == cmd) {
          continue;
        }

        let probability = *count as f64 / self.total_commands as f64;
        predictions.push((cmd.clone(), probability));
      }
    }

    // Sort by probability (highest first)
    predictions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Return top predictions
    predictions.truncate(max_predictions);
    predictions
  }

  /// Get model statistics
  pub fn stats(&self) -> ModelStats {
    let mut context_count = 0;
    let mut command_count = 0;
    let context_frequency_count = self.context_frequencies.len();

    for (_, commands) in &self.frequencies {
      context_count += 1;
      command_count += commands.len();
    }

    ModelStats {
      n_value: self.n,
      total_commands: self.total_commands,
      context_count,
      command_count,
      context_frequency_count,
      dir_context_count: context_frequency_count,
    }
  }
}

/// Statistics about the N-gram model
#[derive(Debug)]
pub struct ModelStats {
  /// The n value used in the model
  pub n_value: usize,
  /// Total number of commands processed
  pub total_commands: usize,
  /// Number of unique contexts
  pub context_count: usize,
  /// Number of unique commands
  pub command_count: usize,
  /// Number of context-specific frequencies
  pub context_frequency_count: usize,
  /// Number of directory contexts (alias for context_frequency_count)
  pub dir_context_count: usize,
}

impl PredictionModel for NGramModel {
  fn train<I>(&mut self, invocations: I) -> Result<()>
  where
    I: IntoIterator<Item = Invocation>,
  {
    // Collect invocations and update total_commands (number of commands processed)
    let inv_vec: Vec<Invocation> = invocations.into_iter().collect();
    let count = inv_vec.len();
    self.total_commands += count;
    // Generate n-grams from collected invocations
    let ngrams = self.extract_ngrams(inv_vec.into_iter());

    info!("Training N-gram model with {} n-grams", ngrams.len());

    for (context, next, ctx) in ngrams {
      self.update_frequency(context, next, Some(ctx));
    }

    Ok(())
  }

  fn predict(&self, recent_history: &[Invocation], max_predictions: usize) -> Result<Vec<String>> {
    if recent_history.len() < self.n - 1 {
      debug!(
        "Not enough history for prediction (need {} commands)",
        self.n - 1
      );
      return Ok(Vec::new());
    }

    // Extract the most recent n-1 commands as context
    let context_slice: Vec<String> = recent_history
      .iter()
      .skip(recent_history.len() - (self.n - 1))
      .map(|inv| Self::invocation_to_command(inv))
      .collect();

    // Get the context of the most recent command
    let context_val: Option<Context> = recent_history
      .last()
      .map(|inv| Context::from_invocation(inv));
    let context_value_ref = context_val.as_ref();

    debug!(
      "Predicting based on context: {:?}, context_value: {:?}",
      context_slice, context_value_ref
    );

    let predictions = self.get_predictions(&context_slice, context_value_ref, max_predictions);

    Ok(predictions.into_iter().map(|(cmd, _)| cmd).collect())
  }

  fn update<I>(&mut self, invocations: I) -> Result<()>
  where
    I: IntoIterator<Item = Invocation>,
  {
    let ngrams = self.extract_ngrams(invocations);

    debug!("Updating N-gram model with {} new n-grams", ngrams.len());

    for (context, next, ctx) in ngrams {
      self.update_frequency(context, next, Some(ctx));
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::model::context::Context;
  use bstr::BString;
  use rusqlite::Connection;

  fn create_test_invocation(command: &str, working_dir: Option<&str>) -> Invocation {
    Invocation {
      command: BString::from(command.as_bytes()),
      shellname: "zsh".to_string(),
      working_directory: working_dir.map(|wd| BString::from(wd.as_bytes())),
      hostname: None,
      username: None,
      exit_status: None,
      start_unix_timestamp: None,
      end_unix_timestamp: None,
      session_id: 0,
    }
  }

  #[test]
  fn test_ngram_training() -> Result<()> {
    let mut model = NGramModel::new(2);

    let invocations = vec![
      create_test_invocation("git status", Some("/project")),
      create_test_invocation("git add .", Some("/project")),
      create_test_invocation("git commit -m 'update'", Some("/project")),
      create_test_invocation("git push", Some("/project")),
      create_test_invocation("git status", Some("/project")),
      create_test_invocation("git pull", Some("/project")),
    ];

    model.train(invocations)?;

    let recent = vec![create_test_invocation("git status", Some("/project"))];

    let predictions = model.predict(&recent, 2)?;

    assert!(!predictions.is_empty());
    assert!(
      predictions.contains(&"git add .".to_string())
        || predictions.contains(&"git pull".to_string())
    );

    Ok(())
  }

  #[test]
  fn test_ngram_update() -> Result<()> {
    let mut model = NGramModel::new(2);

    // Initial training
    let invocations1 = vec![
      create_test_invocation("git status", Some("/project")),
      create_test_invocation("git add .", Some("/project")),
    ];

    model.train(invocations1)?;

    // Update with new data
    let invocations2 = vec![
      create_test_invocation("git status", Some("/project")),
      create_test_invocation("git pull", Some("/project")),
      create_test_invocation("git status", Some("/project")),
      create_test_invocation("git pull", Some("/project")),
    ];

    model.update(invocations2)?;

    let recent = vec![create_test_invocation("git status", Some("/project"))];

    let predictions = model.predict(&recent, 2)?;

    // "git pull" should be more frequent now
    assert!(!predictions.is_empty());
    assert_eq!(predictions[0], "git pull");

    Ok(())
  }

  #[test]
  fn test_context() -> Result<()> {
    let mut model = NGramModel::new(2);

    // Train with commands in different contexts
    let invocations = vec![
      // In /project1, "ls" is followed by "cd src"
      create_test_invocation("ls", Some("/project1")),
      create_test_invocation("cd src", Some("/project1")),
      create_test_invocation("ls", Some("/project1")),
      create_test_invocation("cd src", Some("/project1")),
      // In /project2, "ls" is followed by "make"
      create_test_invocation("ls", Some("/project2")),
      create_test_invocation("make", Some("/project2")),
      create_test_invocation("ls", Some("/project2")),
      create_test_invocation("make", Some("/project2")),
    ];

    model.train(invocations)?;

    // Test prediction in /project1
    let recent1 = vec![create_test_invocation("ls", Some("/project1"))];

    let predictions1 = model.predict(&recent1, 1)?;
    assert_eq!(predictions1[0], "cd src");

    // Test prediction in /project2
    let recent2 = vec![create_test_invocation("ls", Some("/project2"))];

    let predictions2 = model.predict(&recent2, 1)?;
    assert_eq!(predictions2[0], "make");

    Ok(())
  }

  #[test]
  fn test_serialization_round_trip() -> Result<()> {
    // Create a model with some data
    let mut model = NGramModel::new(2);

    // Add some frequency data
    let context1 = vec!["git".to_string(), "status".to_string()];
    let context2 = vec!["ls".to_string()];

    // Update global frequencies
    model.frequencies.insert(context1.clone(), {
      let mut map = HashMap::new();
      map.insert("push".to_string(), 5);
      map.insert("pull".to_string(), 3);
      map
    });

    model.frequencies.insert(context2.clone(), {
      let mut map = HashMap::new();
      map.insert("grep".to_string(), 2);
      map.insert("-la".to_string(), 7);
      map
    });

    // Update context-specific frequencies
    model.context_frequencies.insert(
      (
        Context {
          cwd: "/home".to_string(),
          hostname: None,
          username: None,
          exit_status: None,
        },
        context1,
      ),
      {
        let mut map = HashMap::new();
        map.insert("commit".to_string(), 4);
        map
      },
    );

    model.context_frequencies.insert(
      (
        Context {
          cwd: "/home".to_string(),
          hostname: None,
          username: None,
          exit_status: None,
        },
        context2,
      ),
      {
        let mut map = HashMap::new();
        map.insert("-la".to_string(), 3);
        map
      },
    );

    model.total_commands = 24;

    // Create in-memory database for testing
    let mut conn = Connection::open_in_memory()?;

    // Create the models table
    conn.execute(
      "CREATE TABLE IF NOT EXISTS models (
        model_type TEXT PRIMARY KEY,
        model_data BLOB NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      )",
      [],
    )?;

    // Save the model to the database
    model.save_to_db(&mut conn)?;

    // Load the model from the database
    let deserialized_model = NGramModel::load_from_db(&mut conn, 2)?;

    // Check that the deserialized model matches the original
    assert_eq!(model.n, deserialized_model.n);
    assert_eq!(model.total_commands, deserialized_model.total_commands);
    assert_eq!(model.use_context, deserialized_model.use_context);

    // Check frequencies
    assert_eq!(
      model.frequencies.len(),
      deserialized_model.frequencies.len()
    );
    for (context, commands) in &model.frequencies {
      let deserialized_commands = deserialized_model.frequencies.get(context).unwrap();
      assert_eq!(commands.len(), deserialized_commands.len());

      for (cmd, count) in commands {
        assert_eq!(deserialized_commands.get(cmd).unwrap(), count);
      }
    }

    // Check context-specific frequencies
    assert_eq!(
      model.context_frequencies.len(),
      deserialized_model.context_frequencies.len()
    );
    for ((context, context_slice), commands) in &model.context_frequencies {
      let deserialized_commands = deserialized_model
        .context_frequencies
        .get(&(context.clone(), context_slice.clone()))
        .unwrap();
      assert_eq!(commands.len(), deserialized_commands.len());

      for (cmd, count) in commands {
        assert_eq!(deserialized_commands.get(cmd).unwrap(), count);
      }
    }

    Ok(())
  }

  #[test]
  fn test_db_save_and_load() -> Result<()> {
    // Create a model with some data
    let mut model = NGramModel::new(2);

    // Train the model
    let invocations = vec![
      create_test_invocation("git status", Some("/project")),
      create_test_invocation("git add .", Some("/project")),
      create_test_invocation("git commit -m 'update'", Some("/project")),
      create_test_invocation("git push", Some("/project")),
    ];

    model.train(invocations)?;

    // Create in-memory database for testing
    let mut conn = Connection::open_in_memory()?;

    // Create the models table
    conn.execute(
      "CREATE TABLE IF NOT EXISTS models (
        model_type TEXT PRIMARY KEY,
        model_data BLOB NOT NULL,
        created_at INTEGER NOT NULL,
        updated_at INTEGER NOT NULL
      )",
      [],
    )?;

    // Save the model to the database
    model.save_to_db(&mut conn)?;

    // Load the model from the database
    let loaded_model = NGramModel::load_from_db(&mut conn, 2)?;

    // Check that the loaded model has the same predictions
    let recent = vec![create_test_invocation("git status", Some("/project"))];

    let original_predictions = model.predict(&recent, 2)?;
    let loaded_predictions = loaded_model.predict(&recent, 2)?;

    assert_eq!(original_predictions, loaded_predictions);

    Ok(())
  }

  #[test]
  fn test_predict_simple() -> Result<()> {
    let mut model = NGramModel::new(2);
    let invocations = vec![
      create_test_invocation("ls", Some("/home")),
      create_test_invocation("pwd", Some("/home")),
      create_test_invocation("ls", Some("/home")),
    ];
    model.train(invocations)?;
    let recent = vec![create_test_invocation("pwd", Some("/home"))];
    let preds = model.predict(&recent, 1)?;
    assert_eq!(preds, vec!["ls".to_string()]);
    Ok(())
  }

  #[test]
  fn test_predict_insufficient_history() -> Result<()> {
    let model = NGramModel::new(3);
    let recent = vec![create_test_invocation("ls", Some("/home"))];
    let preds = model.predict(&recent, 5)?;
    assert!(preds.is_empty());
    Ok(())
  }

  #[test]
  fn test_stats() -> Result<()> {
    let mut model = NGramModel::new(2);
    let invocations = vec![
      create_test_invocation("ls", Some("/home")),
      create_test_invocation("pwd", Some("/home")),
      create_test_invocation("ls", Some("/home")),
    ];
    model.train(invocations)?;
    let stats = model.stats();
    assert_eq!(stats.n_value, 2);
    assert_eq!(stats.total_commands, 3);
    assert_eq!(stats.context_count, 2);
    assert_eq!(stats.command_count, 2);
    assert_eq!(stats.context_frequency_count, 2);
    assert_eq!(stats.dir_context_count, 2);
    Ok(())
  }

  #[test]
  fn test_context_aware_predictions() -> Result<()> {
    let mut model = NGramModel::new(2);
    model.set_use_context(true);
    let invocations = vec![
      create_test_invocation("ls", Some("/dir1")),
      create_test_invocation("cmd1", Some("/dir1")),
      create_test_invocation("ls", Some("/dir2")),
      create_test_invocation("cmd2", Some("/dir2")),
    ];
    model.train(invocations)?;
    let recent1 = vec![create_test_invocation("ls", Some("/dir1"))];

    let predictions = model.predict(&recent1, 1)?;

    assert!(!predictions.is_empty());
    assert_eq!(predictions[0], "cmd1".to_string());

    let recent2 = vec![create_test_invocation("ls", Some("/dir2"))];

    let predictions = model.predict(&recent2, 1)?;

    assert!(!predictions.is_empty());
    assert_eq!(predictions[0], "cmd2".to_string());

    Ok(())
  }

  #[test]
  fn test_multi_dimensional_context_and_fallback() -> Result<()> {
    let mut model = NGramModel::new(2);
    model.set_use_context(true);
    // Context A: dir=/proj, host=h1, user=u1, exit=0 -> next = cmdA
    let mut inv1 = create_test_invocation("ls", Some("/proj"));
    inv1.hostname = Some(BString::from("h1".as_bytes()));
    inv1.username = Some(BString::from("u1".as_bytes()));
    inv1.exit_status = Some(0);
    let mut inv2 = create_test_invocation("cmdA", Some("/proj"));
    inv2.hostname = inv1.hostname.clone();
    inv2.username = inv1.username.clone();
    inv2.exit_status = inv1.exit_status;
    // Context B: same dir, host=h2, user=u2, exit=1 -> next = cmdB
    let mut inv3 = create_test_invocation("ls", Some("/proj"));
    inv3.hostname = Some(BString::from("h2".as_bytes()));
    inv3.username = Some(BString::from("u2".as_bytes()));
    inv3.exit_status = Some(1);
    let mut inv4 = create_test_invocation("cmdB", Some("/proj"));
    inv4.hostname = inv3.hostname.clone();
    inv4.username = inv3.username.clone();
    inv4.exit_status = inv3.exit_status;
    // Train both contexts
    model.train(vec![inv1.clone(), inv2.clone(), inv3.clone(), inv4.clone()])?;
    // Predict for Context A
    let preds_a = model.predict(&[inv1.clone()], 1)?;
    assert_eq!(preds_a, vec!["cmdA".to_string()]);
    // Predict for Context B
    let preds_b = model.predict(&[inv3.clone()], 1)?;
    assert_eq!(preds_b, vec!["cmdB".to_string()]);
    // Fallback: context not seen (e.g. missing hostname)
    let unknown = create_test_invocation("ls", Some("/proj"));
    let preds_fallback = model.predict(&[unknown], 2)?;
    // Should include both cmdA and cmdB from global frequencies
    assert!(preds_fallback.contains(&"cmdA".to_string()));
    assert!(preds_fallback.contains(&"cmdB".to_string()));
    Ok(())
  }

  #[test]
  fn test_username_exit_status_context_and_fallback() -> Result<()> {
    let mut model = NGramModel::new(2);
    model.set_use_context(true);
    // Context X: host=h1, user=u1, exit=0 -> next=out1
    let mut invx = create_test_invocation("run", Some("/proj"));
    invx.hostname = Some(BString::from("h1".as_bytes()));
    invx.username = Some(BString::from("u1".as_bytes()));
    invx.exit_status = Some(0);
    let mut outx = create_test_invocation("out1", Some("/proj"));
    outx.hostname = invx.hostname.clone();
    outx.username = invx.username.clone();
    outx.exit_status = invx.exit_status;
    // Context Y: host=h1, user=u2, exit=1 -> next=out2
    let mut invy = create_test_invocation("run", Some("/proj"));
    invy.hostname = Some(BString::from("h1".as_bytes()));
    invy.username = Some(BString::from("u2".as_bytes()));
    invy.exit_status = Some(1);
    let mut outy = create_test_invocation("out2", Some("/proj"));
    outy.hostname = invy.hostname.clone();
    outy.username = invy.username.clone();
    outy.exit_status = invy.exit_status;
    model.train(vec![invx.clone(), outx.clone(), invy.clone(), outy.clone()])?;
    // Predict for X
    let predx = model.predict(&[invx.clone()], 1)?;
    assert_eq!(predx, vec!["out1".to_string()]);
    // Predict for Y
    let predy = model.predict(&[invy.clone()], 1)?;
    assert_eq!(predy, vec!["out2".to_string()]);
    // Partial context: same host/user but missing exit -> fallback to global
    let mut inv_partial = create_test_invocation("run", Some("/proj"));
    inv_partial.hostname = Some(BString::from("h1".as_bytes()));
    inv_partial.username = Some(BString::from("u1".as_bytes())); // no exit_status
    let preds_partial = model.predict(&[inv_partial], 2)?;
    // Should include both out1 and out2 globally
    assert!(preds_partial.contains(&"out1".to_string()));
    assert!(preds_partial.contains(&"out2".to_string()));
    Ok(())
  }
}
