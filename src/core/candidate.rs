#[derive(Debug, Clone)]
pub struct Candidate {
  pub command: String,
  pub freq: i64,
  pub last_seen: i64,
  pub transition_freq: i64,
  pub repo_transition_freq: i64,
  pub(crate) repo_freq: i64,
  pub context_freq: i64,
  pub(crate) session_freq: i64,
  pub session_last_seen: i64,
  pub(crate) sequence_confidence: f64,
  pub(crate) sequence_lift: f64,
  pub(crate) sequence_prefix_len: usize,
}

impl Candidate {
  pub(crate) fn new(command: &str) -> Self {
    Self {
      command: command.to_string(),
      freq: 0,
      last_seen: 0,
      transition_freq: 0,
      repo_transition_freq: 0,
      repo_freq: 0,
      context_freq: 0,
      session_freq: 0,
      session_last_seen: 0,
      sequence_confidence: 0.0,
      sequence_lift: 0.0,
      sequence_prefix_len: 0,
    }
  }
}
