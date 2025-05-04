use std::collections::{BTreeMap, HashMap};

use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use super::PredictionModel;
use crate::Result;
use crate::db::{load_model, save_model};
use crate::shell_history::Invocation;

/// N-gram model for predicting shell commands based on command history
#[derive(Serialize, Deserialize, Clone)]
pub struct NGramModel {
  /// The maximum length of n-grams to consider
  n: usize,
  /// Frequency table for n-grams
  /// Maps from a sequence of n-1 commands to a map of possible next commands and their frequencies
  frequencies: BTreeMap<Vec<String>, HashMap<String, usize>>,
  /// Directory-specific frequency table
  /// Maps from (working_directory, context) to next commands and their frequencies
  dir_frequencies: BTreeMap<(String, Vec<String>), HashMap<String, usize>>,
  /// Total number of commands seen
  total_commands: usize,
  /// Whether to use working directory context for predictions
  use_dir_context: bool,
}

// Serializable version of the model that uses only JSON-compatible types
#[derive(Serialize, Deserialize)]
struct SerializableNGramModel {
  n: usize,
  // Store frequencies as a list of entries to avoid complex keys
  frequencies: Vec<FrequencyEntry>,
  // Store directory context as a list of entries
  dir_contexts: Vec<SerializableDirContextEntry>,
  total_commands: usize,
  use_dir_context: bool,
}

// Entry for frequency table serialization
#[derive(Serialize, Deserialize)]
struct FrequencyEntry {
  context: Vec<String>,
  next_commands: HashMap<String, usize>,
}

// Entry for directory context serialization
#[derive(Serialize, Deserialize)]
struct SerializableDirContextEntry {
  dir: String,
  context: Vec<String>,
  next_commands: HashMap<String, usize>,
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
      dir_frequencies: BTreeMap::new(),
      total_commands: 0,
      use_dir_context: true,
    }
  }

  /// Get the n value for this model
  pub fn n(&self) -> usize {
    self.n
  }

  /// Enable or disable directory context for predictions
  pub fn set_use_dir_context(&mut self, use_dir_context: bool) {
    self.use_dir_context = use_dir_context;
  }

  /// Save the N-gram model to the database
  pub fn save_to_db(&self, conn: &mut Connection) -> Result<()> {
    let model_type = format!("ngram_n{}", self.n);

    // Convert to serializable format
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
      dir_contexts: self
        .dir_frequencies
        .iter()
        .map(
          |((dir, context), next_commands)| SerializableDirContextEntry {
            dir: dir.clone(),
            context: context.clone(),
            next_commands: next_commands.clone(),
          },
        )
        .collect(),
      total_commands: self.total_commands,
      use_dir_context: self.use_dir_context,
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
      // Deserialize from JSON
      let serializable: SerializableNGramModel = serde_json::from_slice(&model_data)?;

      // Convert back to NGramModel
      let mut frequencies = BTreeMap::new();
      for entry in serializable.frequencies {
        frequencies.insert(entry.context, entry.next_commands);
      }

      let mut dir_frequencies = BTreeMap::new();
      for entry in serializable.dir_contexts {
        dir_frequencies.insert((entry.dir, entry.context), entry.next_commands);
      }

      Ok(Self {
        n: serializable.n,
        frequencies,
        dir_frequencies,
        total_commands: serializable.total_commands,
        use_dir_context: serializable.use_dir_context,
      })
    } else {
      // No model found, create a new one
      Ok(Self::new(n))
    }
  }

  /// Convert an invocation to a command string for the model
  fn invocation_to_command(invocation: &Invocation) -> String {
    String::from_utf8_lossy(&invocation.command).to_string()
  }

  /// Get the working directory from an invocation, or a default value
  fn get_working_directory(invocation: &Invocation) -> String {
    if let Some(ref wd) = invocation.working_directory {
      String::from_utf8_lossy(wd).to_string()
    } else {
      "unknown".to_string()
    }
  }

  /// Find a directory context entry or create a new one
  fn find_or_create_dir_entry(
    &mut self,
    dir: String,
    context: Vec<String>,
  ) -> &mut HashMap<String, usize> {
    // Clone the keys for lookup
    let key = (dir.clone(), context.clone());

    // Check if the entry exists and insert if not
    if !self.dir_frequencies.contains_key(&key) {
      self.dir_frequencies.insert(key.clone(), HashMap::new());
    }

    // Return a mutable reference to the entry
    self.dir_frequencies.get_mut(&key).unwrap()
  }

  /// Find directory context entries that match the given directory and context
  fn find_dir_entries(&self, dir: &str, context: &[String]) -> Vec<&HashMap<String, usize>> {
    self
      .dir_frequencies
      .iter()
      .filter(|((dir_entry, context_entry), _)| dir_entry == dir && context_entry == context)
      .map(|(_, next_commands)| next_commands)
      .collect()
  }

  /// Extract n-grams from a sequence of commands
  fn extract_ngrams<I>(&self, invocations: I) -> Vec<(Vec<String>, String, Option<String>)>
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
      let working_dir = Self::get_working_directory(&invocations_vec[i + self.n - 1]);
      ngrams.push((context, next, Some(working_dir)));
    }

    ngrams
  }

  /// Update the frequency table with a single n-gram
  fn update_frequency(&mut self, context: Vec<String>, next: String, working_dir: Option<String>) {
    // Update global frequency table
    let entry = self
      .frequencies
      .entry(context.clone())
      .or_insert_with(HashMap::new);
    *entry.entry(next.clone()).or_insert(0) += 1;

    // Update directory-specific frequency table if working directory is provided
    if let Some(dir) = working_dir {
      let dir_entry = self.find_or_create_dir_entry(dir, context);
      *dir_entry.entry(next).or_insert(0) += 1;
    }

    self.total_commands += 1;
  }

  /// Get the most likely next commands given a context and optional working directory
  fn get_predictions(
    &self,
    context: &[String],
    working_dir: Option<&str>,
    max_predictions: usize,
  ) -> Vec<(String, f64)> {
    let mut predictions = Vec::new();

    // If we have directory-specific predictions and directory context is enabled, use them
    if self.use_dir_context && working_dir.is_some() {
      let dir = working_dir.unwrap();
      let dir_entries = self.find_dir_entries(dir, context);

      for entry in dir_entries {
        for (cmd, count) in entry {
          let probability = *count as f64 / self.total_commands as f64 * 2.0; // Give higher weight to directory-specific predictions
          predictions.push((cmd.clone(), probability));
        }
      }
    }

    // Add global predictions
    if let Some(next_commands) = self.frequencies.get(&context.to_vec()) {
      for (cmd, count) in next_commands {
        // If we already have this command from directory-specific predictions, skip it
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
    let dir_context_count = self.dir_frequencies.len();

    for (_, commands) in &self.frequencies {
      context_count += 1;
      command_count += commands.len();
    }

    ModelStats {
      n_value: self.n,
      total_commands: self.total_commands,
      context_count,
      command_count,
      dir_context_count,
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
  /// Number of directory-specific contexts
  pub dir_context_count: usize,
}

impl PredictionModel for NGramModel {
  fn train<I>(&mut self, invocations: I) -> Result<()>
  where
    I: IntoIterator<Item = Invocation>,
  {
    let ngrams = self.extract_ngrams(invocations);

    info!("Training N-gram model with {} n-grams", ngrams.len());

    for (context, next, working_dir) in ngrams {
      self.update_frequency(context, next, working_dir);
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
    let context: Vec<String> = recent_history
      .iter()
      .skip(recent_history.len() - (self.n - 1))
      .map(|inv| Self::invocation_to_command(inv))
      .collect();

    // Get the working directory of the most recent command
    let working_dir = recent_history.last().and_then(|inv| {
      inv
        .working_directory
        .as_ref()
        .map(|wd| String::from_utf8_lossy(wd).to_string())
    });

    debug!(
      "Predicting based on context: {:?}, working_dir: {:?}",
      context, working_dir
    );

    let predictions = self.get_predictions(&context, working_dir.as_deref(), max_predictions);

    Ok(predictions.into_iter().map(|(cmd, _)| cmd).collect())
  }

  fn update<I>(&mut self, invocations: I) -> Result<()>
  where
    I: IntoIterator<Item = Invocation>,
  {
    let ngrams = self.extract_ngrams(invocations);

    debug!("Updating N-gram model with {} new n-grams", ngrams.len());

    for (context, next, working_dir) in ngrams {
      self.update_frequency(context, next, working_dir);
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
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
  fn test_directory_context() -> Result<()> {
    let mut model = NGramModel::new(2);

    // Train with commands in different directories
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

    // Update directory-specific frequencies
    model
      .dir_frequencies
      .insert(("~/project".to_string(), context1), {
        let mut map = HashMap::new();
        map.insert("commit".to_string(), 4);
        map
      });

    model
      .dir_frequencies
      .insert(("~/home".to_string(), context2), {
        let mut map = HashMap::new();
        map.insert("-la".to_string(), 3);
        map
      });

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
    assert_eq!(model.use_dir_context, deserialized_model.use_dir_context);

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

    // Check directory frequencies
    assert_eq!(
      model.dir_frequencies.len(),
      deserialized_model.dir_frequencies.len()
    );
    for ((dir, context), commands) in &model.dir_frequencies {
      let deserialized_commands = deserialized_model
        .dir_frequencies
        .get(&(dir.clone(), context.clone()))
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
}
