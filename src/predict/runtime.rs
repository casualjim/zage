use std::collections::HashMap;

use crate::core::RankingWeights;
use crate::phase::PhaseConfig;

#[derive(Debug, Clone)]
pub(crate) struct SuggestRuntime {
  pub(crate) aliases: HashMap<String, String>,
  pub(crate) weights: RankingWeights,
  pub(crate) recency_half_life: f64,
  pub(crate) now: i64,
  pub(crate) phase_config: Option<PhaseConfig>,
}
