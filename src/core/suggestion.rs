#[derive(Debug, Clone)]
pub struct Suggestion {
  pub command: String,
  pub score: f64,
  pub breakdown: ScoreBreakdown,
  pub debug: Option<SuggestionDebug>,
}

#[derive(Debug, Clone, Default)]
pub struct ScoreBreakdown {
  pub recency: f64,
  pub session_recency: f64,
  pub frequency: f64,
  pub transition: f64,
  pub context: f64,
  pub sequence: f64,
  pub similarity: f64,
  pub embedding_retrieval: f64,
  pub online_model: f64,
}

#[derive(Debug, Clone, Default)]
pub struct SuggestionDebug {
  pub blend: BlendDebug,
  pub candidate: CandidateDebug,
  pub pipeline: PipelineDebug,
}

#[derive(Debug, Clone, Default)]
pub struct PipelineDebug {
  pub added_transition: usize,
  pub added_session: usize,
  pub added_embedding: usize,
  pub added_context: usize,
  pub added_workspace: usize,
  pub added_head: usize,
  pub added_sequence: usize,
  pub added_template: usize,
  pub added_recent: usize,
  pub added_global: usize,

  pub total_candidates: usize,
  pub conditional_candidates: usize,

  pub pruned_before: usize,
  pub pruned_after: usize,
  pub pruned_kept_conditional: usize,
}

#[derive(Debug, Clone, Default)]
pub struct BlendDebug {
  pub model_gate: f64,
  pub model_alpha: f64,
  pub blend_model_weight: f64,
  pub blend_frecency_weight: f64,
  pub blend_sequence_weight: f64,
  pub blend_tier1_weight: f64,

  pub model_feature: f64,
  pub frecency_feature: f64,
  pub sequence_feature: f64,
  pub tier1_feature: f64,

  pub model_contrib: f64,
  pub frecency_contrib: f64,
  pub sequence_contrib: f64,
  pub tier1_contrib: f64,
}

#[derive(Debug, Clone, Default)]
pub struct CandidateDebug {
  pub freq: i64,
  pub workspace_freq: i64,
  pub last_seen: i64,

  pub transition_freq: i64,
  pub workspace_transition_freq: i64,
  pub transition_exit_status_match: bool,

  pub context_freq: i64,
  pub context_cwd_match: bool,
  pub context_host_match: bool,
  pub context_user_match: bool,

  pub session_freq: i64,
  pub session_last_seen: i64,

  pub from_embedding: bool,
  pub sequence_confidence: f64,
  pub sequence_lift: f64,
  pub sequence_prefix_len: usize,
}
