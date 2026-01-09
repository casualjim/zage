#[derive(Debug, Clone)]
pub struct Suggestion {
  pub command: String,
  pub score: f64,
  pub breakdown: ScoreBreakdown,
}

#[derive(Debug, Clone, Default)]
pub struct ScoreBreakdown {
  pub recency: f64,
  pub frequency: f64,
  pub transition: f64,
  pub context: f64,
  pub sequence: f64,
  pub similarity: f64,
}
