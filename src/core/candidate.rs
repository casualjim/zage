#[derive(Debug, Clone)]
pub struct Candidate {
  pub command: String,
  pub freq: i64,
  pub last_seen: i64,
  pub transition_freq: i64,
  pub workspace_transition_freq: i64,
  pub(crate) transition_exit_status_match: bool,
  pub(crate) workspace_freq: i64,
  pub context_freq: i64,
  pub(crate) context_cwd_match: bool,
  pub(crate) context_host_match: bool,
  pub(crate) context_user_match: bool,
  pub(crate) session_freq: i64,
  pub session_last_seen: i64,
  pub(crate) from_embedding: bool,
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
      workspace_transition_freq: 0,
      transition_exit_status_match: false,
      workspace_freq: 0,
      context_freq: 0,
      context_cwd_match: false,
      context_host_match: false,
      context_user_match: false,
      session_freq: 0,
      session_last_seen: 0,
      from_embedding: false,
      sequence_confidence: 0.0,
      sequence_lift: 0.0,
      sequence_prefix_len: 0,
    }
  }
}
