//! # Multi-dimensional Context for Sequence Detection
//!
//! This module implements a rich, multi-dimensional context representation for command sequences.
//! Unlike the simpler Context used for individual commands, SequenceContext captures the
//! environmental and temporal patterns that influence entire command sequences.
//!
//! ## Key Concepts:
//!
//! - **Temporal Patterns**: Time-based patterns (time of day, day of week, etc.) when sequences occur
//! - **Environmental Context**: Working directory, hostname, username, etc.
//! - **Command State**: Exit status patterns, output characteristics
//! - **Sequence Boundaries**: Indicators of sequence start/end (e.g., session boundaries)
//! - **Dimensional Weights**: Importance of different context dimensions for sequence matching

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

use crate::model::context::Context as BaseContext;
use crate::shell_history::Invocation;

/// Represents the time of day, segmented into periods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeOfDay {
  EarlyMorning, // 5:00 - 8:59
  Morning,      // 9:00 - 11:59
  Afternoon,    // 12:00 - 16:59
  Evening,      // 17:00 - 20:59
  Night,        // 21:00 - 4:59
}

impl TimeOfDay {
  /// Convert an hour (0-23) to a TimeOfDay
  pub fn from_hour(hour: u8) -> Self {
    match hour {
      5..=8 => TimeOfDay::EarlyMorning,
      9..=11 => TimeOfDay::Morning,
      12..=16 => TimeOfDay::Afternoon,
      17..=20 => TimeOfDay::Evening,
      _ => TimeOfDay::Night,
    }
  }
}

/// Represents the duration of a command
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CommandDuration {
  Instant,  // < 1 second
  Quick,    // 1-5 seconds
  Medium,   // 5-30 seconds
  Long,     // 30 seconds - 5 minutes
  VeryLong, // > 5 minutes
}

impl CommandDuration {
  /// Convert a duration to a CommandDuration category
  pub fn from_duration(duration: Duration) -> Self {
    let secs = duration.as_secs();
    match secs {
      0 => CommandDuration::Instant,
      1..=5 => CommandDuration::Quick,
      6..=30 => CommandDuration::Medium,
      31..=300 => CommandDuration::Long,
      _ => CommandDuration::VeryLong,
    }
  }
}

/// Represents the output characteristics of a command
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutputCharacteristic {
  NoOutput,    // No stdout/stderr
  TextOutput,  // Plain text output
  ErrorOutput, // Contains error messages
  DataOutput,  // Contains structured data (JSON, CSV, etc.)
  Interactive, // Interactive output (e.g., TUI)
}

/// Multi-dimensional context for sequence detection
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SequenceContext {
  /// Base context (working directory, hostname, username, exit status)
  pub base: BaseContext,

  /// Time-based context dimensions
  pub temporal: TemporalContext,

  /// Command execution characteristics
  pub execution: ExecutionContext,

  /// Session information
  pub session_id: Option<i64>,

  /// Additional metadata for sequence context
  pub metadata: HashMap<String, String>,

  /// Commands in the sequence
  pub commands: Vec<String>,
}

// Manual implementation of Hash that excludes metadata
impl std::hash::Hash for SequenceContext {
  fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
    self.base.hash(state);
    self.temporal.hash(state);
    self.execution.hash(state);
    self.session_id.hash(state);
    self.commands.hash(state);
    // Intentionally skip metadata as HashMap doesn't implement Hash
  }
}

/// Time-based context dimensions
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TemporalContext {
  /// Time of day category
  pub time_of_day: TimeOfDay,

  /// Day of week (1 = Monday, 7 = Sunday)
  pub day_of_week: u32,

  /// Whether it's a weekend
  pub is_weekend: bool,
}

/// Command execution characteristics
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExecutionContext {
  /// Duration category of commands in the sequence
  pub duration_pattern: Vec<CommandDuration>,

  /// Exit status pattern of commands in the sequence
  pub exit_status_pattern: Vec<Option<i64>>,

  /// Output characteristics of commands in the sequence
  pub output_characteristics: Vec<OutputCharacteristic>,
}

impl SequenceContext {
  /// Create a new SequenceContext from a sequence of invocations
  pub fn from_invocations(invocations: &[Invocation]) -> Self {
    if invocations.is_empty() {
      return Self::default();
    }

    // Extract base context from first invocation
    let first = &invocations[0];
    let working_dir = first
      .working_directory
      .as_ref()
      .map(|wd| String::from_utf8_lossy(wd.as_slice()).to_string())
      .unwrap_or_default();
    let hostname = first
      .hostname
      .as_ref()
      .map(|h| String::from_utf8_lossy(h.as_slice()).to_string())
      .unwrap_or_default();
    let username = first
      .username
      .as_ref()
      .map(|u| String::from_utf8_lossy(u.as_slice()).to_string())
      .unwrap_or_default();

    // Calculate temporal context from first invocation's timestamp
    let time_of_day = if let Some(timestamp) = first.start_unix_timestamp {
      let hour = (timestamp % 86400) / 3600; // Convert to hour of day (0-23)
      TimeOfDay::from_hour(hour as u8)
    } else {
      TimeOfDay::Morning
    };

    let day_of_week = if let Some(timestamp) = first.start_unix_timestamp {
      // Calculate day of week (0 = Sunday, 6 = Saturday)
      // Unix epoch (Jan 1, 1970) was a Thursday (4)
      let days_since_epoch = timestamp / 86400;
      let day_of_week = (days_since_epoch + 4) % 7;
      match day_of_week {
        0 => 7,
        1 => 1,
        2 => 2,
        3 => 3,
        4 => 4,
        5 => 5,
        6 => 6,
        _ => 1,
      }
    } else {
      1
    };

    let is_weekend = matches!(day_of_week, 6 | 7);

    // Collect execution patterns across all invocations
    let mut duration_pattern = Vec::new();
    let mut exit_status_pattern = Vec::new();
    let mut output_characteristics = Vec::new();
    let mut commands = Vec::new();

    for inv in invocations {
      // Calculate duration if both timestamps are available
      if let (Some(start), Some(end)) = (inv.start_unix_timestamp, inv.end_unix_timestamp) {
        let duration = std::time::Duration::from_secs((end - start) as u64);
        duration_pattern.push(CommandDuration::from_duration(duration));
      } else {
        duration_pattern.push(CommandDuration::Instant);
      }

      // Record exit status
      exit_status_pattern.push(inv.exit_status);

      // Determine output characteristic (simplified - would need actual output data)
      // This is a placeholder - in a real implementation, you'd analyze stdout/stderr
      output_characteristics.push(OutputCharacteristic::NoOutput);

      // Record command
      commands.push(String::from_utf8_lossy(inv.command.as_slice()).to_string());
    }

    SequenceContext {
      base: BaseContext {
        cwd: working_dir,
        hostname: Some(hostname),
        username: Some(username),
        exit_status: None,
      },
      temporal: TemporalContext {
        time_of_day,
        day_of_week,
        is_weekend,
      },
      execution: ExecutionContext {
        duration_pattern,
        exit_status_pattern,
        output_characteristics,
      },
      session_id: Some(invocations[0].session_id),
      metadata: HashMap::new(),
      commands,
    }
  }

  /// Calculate similarity score between two sequence contexts (0.0 to 1.0)
  pub fn similarity(&self, other: &SequenceContext) -> f64 {
    let mut score = 0.0;
    let mut weight_sum = 0.0;

    // Base context comparison
    let base_weight = 0.6; // Increased weight further to emphasize directory differences
    weight_sum += base_weight;

    // Working directory similarity (exact match or partial path match)
    if self.base.cwd == other.base.cwd {
      score += base_weight * 0.7; // Increased importance of exact directory match
    } else if self.base.cwd.starts_with(&other.base.cwd)
      || other.base.cwd.starts_with(&self.base.cwd)
    {
      // Partial path match - reduced significance
      score += base_weight * 0.1;
    }

    // Host/user similarity
    if self.base.hostname == other.base.hostname {
      score += base_weight * 0.15;
    }
    if self.base.username == other.base.username {
      score += base_weight * 0.15;
    }

    // Temporal context comparison
    let temporal_weight = 0.1; // Reduced weight
    weight_sum += temporal_weight;

    if self.temporal.time_of_day == other.temporal.time_of_day {
      score += temporal_weight * 0.5;
    }
    if self.temporal.day_of_week == other.temporal.day_of_week {
      score += temporal_weight * 0.3;
    }
    if self.temporal.is_weekend == other.temporal.is_weekend {
      score += temporal_weight * 0.2;
    }

    // Execution context comparison
    let execution_weight = 0.2; // Reduced weight
    weight_sum += execution_weight;

    // Compare duration patterns
    let duration_similarity = self.pattern_similarity(
      &self.execution.duration_pattern,
      &other.execution.duration_pattern,
    );
    score += execution_weight * 0.4 * duration_similarity;

    // Compare exit status patterns
    let exit_status_similarity = self.exit_status_pattern_similarity(
      &self.execution.exit_status_pattern,
      &other.execution.exit_status_pattern,
    );
    score += execution_weight * 0.4 * exit_status_similarity;

    // Compare output characteristics
    let output_similarity = self.pattern_similarity(
      &self.execution.output_characteristics,
      &other.execution.output_characteristics,
    );
    score += execution_weight * 0.2 * output_similarity;

    // Session comparison - increased weight and importance
    let session_weight = 0.1;
    weight_sum += session_weight;

    if self.session_id == other.session_id && self.session_id.is_some() {
      score += session_weight;
    } else if self.session_id.is_some() && other.session_id.is_some() {
      // Different sessions should strongly reduce similarity
      score -= session_weight * 0.8; // Increased penalty
    }

    // Command content comparison - new factor
    // If the commands are completely different, reduce similarity further
    if !self.commands.is_empty() && !other.commands.is_empty() {
      let command_weight = 0.2; // New weight for command comparison
      weight_sum += command_weight;

      if self.commands.is_empty() && other.commands.is_empty() {
        // Both empty, neutral contribution
      } else if self.commands.is_empty() || other.commands.is_empty() {
        // One is empty, penalty
        score -= command_weight * 0.5;
      } else {
        // Calculate command set similarity (e.g., Jaccard index)
        use std::collections::HashSet;
        let set1: HashSet<_> = self.commands.iter().collect();
        let set2: HashSet<_> = other.commands.iter().collect();

        let intersection = set1.intersection(&set2).count();
        let union = set1.union(&set2).count();

        if union > 0 {
          let jaccard_sim = intersection as f64 / union as f64;
          // Apply weight to the Jaccard similarity score.
          // A full match (jaccard_sim = 1.0) gets full positive weight contribution.
          // No match (jaccard_sim = 0.0) gets zero contribution from this part.
          // Partial match gets proportional contribution.
          score += command_weight * jaccard_sim;

          // Add a smaller penalty if there's absolutely no overlap, compared to partial overlap
          if intersection == 0 {
            score -= command_weight * 0.25; // Smaller penalty for no overlap vs one empty
          }
        } else {
          // Both sets were non-empty but union is 0? Should not happen if logic above is correct.
          // Treat as no match / neutral or slight penalty just in case.
        }
      }
    }

    // Normalize score
    if weight_sum > 0.0 {
      score / weight_sum
    } else {
      0.0
    }
  }

  /// Calculate similarity between two patterns (vectors of comparable items)
  fn pattern_similarity<T: PartialEq>(&self, a: &[T], b: &[T]) -> f64 {
    if a.is_empty() || b.is_empty() {
      return 0.0;
    }

    let min_len = a.len().min(b.len());
    let max_len = a.len().max(b.len());

    let mut matches = 0;
    for i in 0..min_len {
      if a[i] == b[i] {
        matches += 1;
      }
    }

    matches as f64 / max_len as f64
  }

  /// Calculate similarity between exit status patterns, handling None values
  fn exit_status_pattern_similarity(&self, a: &[Option<i64>], b: &[Option<i64>]) -> f64 {
    if a.is_empty() || b.is_empty() {
      return 0.0;
    }

    let min_len = a.len().min(b.len());
    let max_len = a.len().max(b.len());

    let mut matches = 0;
    for i in 0..min_len {
      match (a[i], b[i]) {
        (Some(x), Some(y)) if x == y => matches += 1,
        (None, None) => matches += 1,
        _ => {}
      }
    }

    matches as f64 / max_len as f64
  }
}

impl Default for SequenceContext {
  fn default() -> Self {
    SequenceContext {
      base: BaseContext {
        cwd: String::new(),
        hostname: None,
        username: None,
        exit_status: None,
      },
      temporal: TemporalContext::default(),
      execution: ExecutionContext::default(),
      session_id: None,
      metadata: HashMap::new(),
      commands: Vec::new(),
    }
  }
}

impl Default for TemporalContext {
  fn default() -> Self {
    TemporalContext {
      time_of_day: TimeOfDay::Morning,
      day_of_week: 1,
      is_weekend: false,
    }
  }
}

impl Default for ExecutionContext {
  fn default() -> Self {
    ExecutionContext {
      duration_pattern: Vec::new(),
      exit_status_pattern: Vec::new(),
      output_characteristics: Vec::new(),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::shell_history::Invocation;
  use bstr::BString;

  fn create_test_invocation(
    cmd: &str,
    dir: Option<&str>,
    exit: Option<i64>,
    start_time: Option<i64>,
    end_time: Option<i64>,
    session: i64,
  ) -> Invocation {
    Invocation {
      command: BString::from(cmd),
      shellname: "test_shell".to_string(),
      working_directory: dir.map(BString::from),
      hostname: Some(BString::from("test-host")),
      username: Some(BString::from("test-user")),
      exit_status: exit,
      start_unix_timestamp: start_time,
      end_unix_timestamp: end_time,
      session_id: session,
    }
  }

  #[test]
  fn test_sequence_context_creation() {
    let invocations = vec![
      create_test_invocation(
        "git fetch",
        Some("/home/user/project"),
        Some(0),
        Some(1643723400),
        Some(1643723401),
        1,
      ),
      create_test_invocation(
        "git status",
        Some("/home/user/project"),
        Some(0),
        Some(1643723402),
        Some(1643723403),
        1,
      ),
      create_test_invocation(
        "git pull",
        Some("/home/user/project"),
        Some(0),
        Some(1643723404),
        Some(1643723410),
        1,
      ),
    ];

    let context = SequenceContext::from_invocations(&invocations);

    // Verify base context
    assert_eq!(context.base.cwd, "/home/user/project");
    assert_eq!(context.base.hostname, Some("test-host".to_string()));
    assert_eq!(context.base.username, Some("test-user".to_string()));

    // Verify execution context
    assert_eq!(context.execution.duration_pattern.len(), 3);
    assert_eq!(
      context.execution.exit_status_pattern,
      vec![Some(0), Some(0), Some(0)]
    );

    // Verify session ID
    assert_eq!(context.session_id, Some(1));

    // Verify commands
    assert_eq!(context.commands.len(), 3);
    assert_eq!(context.commands[0], "git fetch");
    assert_eq!(context.commands[1], "git status");
    assert_eq!(context.commands[2], "git pull");
  }

  #[test]
  fn test_context_similarity() {
    // Create two similar sequences
    let seq1 = vec![
      create_test_invocation(
        "git fetch",
        Some("/home/user/project"),
        Some(0),
        Some(1643723400),
        Some(1643723401),
        1,
      ),
      create_test_invocation(
        "git status",
        Some("/home/user/project"),
        Some(0),
        Some(1643723402),
        Some(1643723403),
        1,
      ),
    ];

    let seq2 = vec![
      create_test_invocation(
        "git fetch",
        Some("/home/user/project"),
        Some(0),
        Some(1643723400),
        Some(1643723401),
        1,
      ),
      create_test_invocation(
        "git status",
        Some("/home/user/project"),
        Some(0),
        Some(1643723402),
        Some(1643723403),
        1,
      ),
    ];

    // Create a different sequence
    let seq3 = vec![
      create_test_invocation(
        "cd /tmp",
        Some("/home"),
        Some(0),
        Some(1643723400),
        Some(1643723401),
        2,
      ),
      create_test_invocation(
        "ls -la",
        Some("/tmp"),
        Some(0),
        Some(1643723402),
        Some(1643723403),
        2,
      ),
    ];

    let ctx1 = SequenceContext::from_invocations(&seq1);
    let ctx2 = SequenceContext::from_invocations(&seq2);
    let ctx3 = SequenceContext::from_invocations(&seq3);

    // Similar contexts should have high similarity
    let sim12 = ctx1.similarity(&ctx2);
    assert!(
      sim12 > 0.9,
      "Similar contexts should have high similarity: {}",
      sim12
    );

    // Different contexts should have lower similarity
    let sim13 = ctx1.similarity(&ctx3);
    assert!(
      sim13 < 0.5,
      "Different contexts should have low similarity: {}",
      sim13
    );
  }

  #[test]
  fn test_command_similarity_logic() {
    let mut ctx1 = SequenceContext::default();
    ctx1.commands = vec!["ls".to_string(), "cd".to_string(), "pwd".to_string()];

    let mut ctx2 = SequenceContext::default();
    ctx2.commands = vec!["ls".to_string(), "cd".to_string(), "pwd".to_string()];

    // 1. Exact Match
    // Keep everything else default/identical
    let sim_exact = ctx1.similarity(&ctx2);
    // Expect high similarity as only commands differ, and they match exactly.
    // The exact value depends on weights, but should be close to 1.0 if command weight is high.
    // Let's assert it's significantly higher than the partial match case.

    // 2. Partial Match (at least one command matches)
    ctx2.commands = vec!["ls".to_string(), "grep".to_string(), "cat".to_string()];
    let sim_partial = ctx1.similarity(&ctx2);
    // Should be lower than exact match, but still positive due to the partial match logic.

    // 3. No Match
    ctx2.commands = vec!["grep".to_string(), "cat".to_string(), "vim".to_string()];
    let sim_none = ctx1.similarity(&ctx2);
    // Should be significantly lower, potentially negative contribution from commands
    // depending on weights, definitely lower than partial match.

    // 4. Different lengths, partial match
    ctx2.commands = vec!["ls".to_string(), "cd".to_string()];
    let sim_diff_len_partial = ctx1.similarity(&ctx2);
    // Should still register as partial match.

    // 5. Different lengths, no match
    ctx2.commands = vec!["grep".to_string(), "cat".to_string()];
    let sim_diff_len_none = ctx1.similarity(&ctx2);
    // Should register as no match.

    println!("Exact: {}, Partial: {}, None: {}, DiffLenPartial: {}, DiffLenNone: {}", sim_exact, sim_partial, sim_none, sim_diff_len_partial, sim_diff_len_none);

    // Assert the ordering based on command similarity contribution
    assert!(sim_exact > sim_partial, "Exact match should score higher than partial");
    assert!(sim_partial > sim_none, "Partial match should score higher than no match");
    assert!(sim_exact > sim_diff_len_partial, "Exact match should score higher than diff length partial");
    assert!(sim_diff_len_partial > sim_diff_len_none, "Diff length partial match should score higher than diff length no match");
    assert!(sim_partial > sim_diff_len_none, "Partial match should score higher than diff length no match"); // Comparing full no match vs diff length no match

    // We can't easily assert absolute values without knowing weights, but relative order is key.
  }

  #[test]
  fn test_pattern_similarity_helper() {
      let ctx = SequenceContext::default(); // Need an instance to call the method

      // Test cases for pattern_similarity<T>
      assert_eq!(ctx.pattern_similarity::<u32>(&[], &[]), 0.0, "Both empty");
      assert_eq!(ctx.pattern_similarity(&[1, 2, 3], &[]), 0.0, "One empty");
      assert_eq!(ctx.pattern_similarity(&[], &[1, 2, 3]), 0.0, "Other empty");
      assert_eq!(ctx.pattern_similarity(&[1, 2, 3], &[1, 2, 3]), 1.0, "Identical");
      assert_eq!(ctx.pattern_similarity(&[1, 2, 3], &[1, 2, 4]), 2.0 / 3.0, "Partial match");
      assert_eq!(ctx.pattern_similarity(&[1, 2, 3], &[4, 5, 6]), 0.0, "No match");
      assert_eq!(ctx.pattern_similarity(&[1, 2], &[1, 2, 3]), 2.0 / 3.0, "Different lengths, partial");
      assert_eq!(ctx.pattern_similarity(&[1, 2, 3], &[1, 2]), 2.0 / 3.0, "Different lengths, partial other way");
      assert_eq!(ctx.pattern_similarity(&[1, 2], &[3, 4, 5]), 0.0 / 3.0, "Different lengths, no match");
  }

  #[test]
  fn test_exit_status_pattern_similarity_helper() {
      let ctx = SequenceContext::default(); // Need an instance to call the method

      // Test cases for exit_status_pattern_similarity
      assert_eq!(ctx.exit_status_pattern_similarity(&[], &[]), 0.0, "Both empty");
      assert_eq!(ctx.exit_status_pattern_similarity(&[Some(0)], &[]), 0.0, "One empty");
      assert_eq!(ctx.exit_status_pattern_similarity(&[], &[Some(0)]), 0.0, "Other empty");
      assert_eq!(ctx.exit_status_pattern_similarity(&[Some(0), Some(1)], &[Some(0), Some(1)]), 1.0, "Identical Some");
      assert_eq!(ctx.exit_status_pattern_similarity(&[None, Some(1)], &[None, Some(1)]), 1.0, "Identical with None");
      assert_eq!(ctx.exit_status_pattern_similarity(&[Some(0), None], &[Some(0), Some(1)]), 1.0 / 2.0, "Partial match, None vs Some");
      assert_eq!(ctx.exit_status_pattern_similarity(&[Some(0), Some(1)], &[Some(0), Some(127)]), 1.0 / 2.0, "Partial match, Some vs Some different");
      assert_eq!(ctx.exit_status_pattern_similarity(&[Some(0), Some(1)], &[None, None]), 0.0 / 2.0, "No match, Some vs None");
      assert_eq!(ctx.exit_status_pattern_similarity(&[Some(0)], &[Some(0), Some(1)]), 1.0 / 2.0, "Different lengths, partial");
      assert_eq!(ctx.exit_status_pattern_similarity(&[Some(0), Some(1)], &[Some(0)]), 1.0 / 2.0, "Different lengths, partial other way");
      assert_eq!(ctx.exit_status_pattern_similarity(&[Some(0)], &[Some(1), Some(127)]), 0.0 / 2.0, "Different lengths, no match");

  }

  #[test]
  fn test_individual_dimension_similarity() {
      // Base context setup (can be simple, using defaults and adding some commands)
      let base_invocations = vec![
          create_test_invocation("cmd1", Some("/home/user"), Some(0), Some(1643723400), None, 1),
          create_test_invocation("cmd2", Some("/home/user"), Some(0), Some(1643723402), None, 1),
      ];
      let ctx_base = SequenceContext::from_invocations(&base_invocations);

      // 1. Identical context
      let ctx_identical = SequenceContext::from_invocations(&base_invocations);
      let sim_identical = ctx_base.similarity(&ctx_identical);
      const TOLERANCE: f64 = 1e-12;
      assert!((sim_identical - 1.0).abs() < TOLERANCE, "Identical contexts should have similarity close to 1.0, got {}", sim_identical);

      // 2. Different CWD
      let mut ctx_diff_cwd = ctx_base.clone();
      ctx_diff_cwd.base.cwd = "/home/other".to_string();
      let sim_diff_cwd = ctx_base.similarity(&ctx_diff_cwd);
      assert!(sim_diff_cwd < 1.0, "Different CWD should reduce similarity below 1.0");
      // Check it's reasonably high as only one dimension changed
      assert!(sim_diff_cwd > 0.5, "Different CWD similarity ({}) should still be significant", sim_diff_cwd);

      // 3. Different Hostname
      let mut ctx_diff_host = ctx_base.clone();
      ctx_diff_host.base.hostname = Some("other-host".to_string());
      let sim_diff_host = ctx_base.similarity(&ctx_diff_host);
      assert!(sim_diff_host < 1.0, "Different Hostname should reduce similarity below 1.0");
      assert!(sim_diff_host > 0.5, "Different Hostname similarity ({}) should still be significant", sim_diff_host);

       // 4. Different Username
      let mut ctx_diff_user = ctx_base.clone();
      ctx_diff_user.base.username = Some("other-user".to_string());
      let sim_diff_user = ctx_base.similarity(&ctx_diff_user);
      assert!(sim_diff_user < 1.0, "Different Username should reduce similarity below 1.0");
      assert!(sim_diff_user > 0.5, "Different Username similarity ({}) should still be significant", sim_diff_user);

      // 5. Different Session ID
      let mut ctx_diff_session = ctx_base.clone();
      ctx_diff_session.session_id = Some(2); // Base session_id is 1
      let sim_diff_session = ctx_base.similarity(&ctx_diff_session);
      assert!(sim_diff_session < 1.0, "Different Session ID should reduce similarity below 1.0");
      // Session ID difference has a specific penalty, check it's lower than others
      assert!(sim_diff_cwd < sim_diff_session, "Different CWD similarity ({}) should be lower than different Session ID ({}) due to penalties", sim_diff_cwd, sim_diff_session);

      // 6. Different TimeOfDay
      let mut ctx_diff_time = ctx_base.clone();
      // Base is Afternoon (from timestamp 13:50 GMT), set this one to Morning to test difference.
      ctx_diff_time.temporal.time_of_day = TimeOfDay::Morning;
      let sim_diff_time = ctx_base.similarity(&ctx_diff_time);
      // Use the same tolerance as the identical check
      assert!(sim_diff_time < 1.0 - TOLERANCE, "Different TimeOfDay should reduce similarity below 1.0 (accounting for tolerance), got {}", sim_diff_time);
      assert!(sim_diff_time > 0.5, "Different TimeOfDay similarity ({}) should still be significant", sim_diff_time);

      // Add more checks for other dimensions like day_of_week, is_weekend, patterns if needed
  }
}
