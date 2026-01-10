#[derive(Debug, Clone)]
pub struct SuggestConfig {
  pub max_results: usize,
  pub recent_limit: usize,
  pub prefix: Option<String>,
  pub cwd: Option<String>,
  pub hostname: Option<String>,
  pub username: Option<String>,
  pub session_id: Option<i64>,
  pub use_sequences: bool,
  pub prefer_full_line: bool,
}

impl Default for SuggestConfig {
  fn default() -> Self {
    Self {
      max_results: 5,
      recent_limit: 10,
      prefix: None,
      cwd: None,
      hostname: None,
      username: None,
      session_id: None,
      use_sequences: true,
      prefer_full_line: false,
    }
  }
}

#[derive(Debug, Clone)]
pub struct RankingWeights {
  pub recency: f64,
  pub frequency: f64,
  pub transition: f64,
  pub context: f64,
  pub sequence: f64,
  pub similarity: f64,
}

impl Default for RankingWeights {
  fn default() -> Self {
    Self {
      recency: 0.05,
      frequency: 0.15,
      transition: 0.25,
      context: 0.25,
      sequence: 0.25,
      similarity: 0.05,
    }
  }
}
