pub(crate) const MODEL_NAME: &str = "rerank";
pub(crate) const HASH_FEATURES: usize = 64;
pub(crate) const BASE_FEATURES: usize = 18;
pub(crate) const FEATURE_COUNT: usize = BASE_FEATURES + HASH_FEATURES;

mod calibration;
mod config;
mod features;
mod model;
mod training;

pub use calibration::CalibrationParams;
pub use config::{RerankContext, TrainConfig, TrainReport};
pub use model::{
  ModelStatus, clear_model_cache, model_status, reset_model, runtime_context, warm_model_cache,
};
pub use training::train_model;

#[cfg(test)]
pub(crate) use features::{
  add_hash, build_feature_matrix, feature_names, features_from_suggestion,
};
pub(crate) use model::rerank_suggestions;

#[cfg(test)]
pub(crate) use model::load_model;
#[cfg(test)]
mod tests {
  use super::*;
  use crate::core::{Candidate, Suggestion};
  use crate::predict::{ScoreBreakdown, candidate_for_test};
  use gbrt_rs::boosting::GBRTConfig;
  use gbrt_rs::{Dataset, GBRTModel, ModelIO};
  use std::collections::{HashMap, HashSet};
  use std::sync::Mutex;
  use tempfile::tempdir;

  static ENV_LOCK: Mutex<()> = Mutex::new(());

  #[test]
  fn feature_matrix_is_deterministic() {
    let a = vec![vec![1.0; FEATURE_COUNT], vec![0.5; FEATURE_COUNT]];
    let b = vec![vec![1.0; FEATURE_COUNT], vec![0.5; FEATURE_COUNT]];
    let left = build_feature_matrix(&a).unwrap();
    let right = build_feature_matrix(&b).unwrap();
    assert_eq!(left.data(), right.data());
  }

  #[test]
  fn hash_features_stable() {
    let mut values = vec![0.0; FEATURE_COUNT];
    add_hash(&mut values, "head:git");
    let mut values2 = vec![0.0; FEATURE_COUNT];
    add_hash(&mut values2, "head:git");
    assert_eq!(values, values2);
  }

  #[test]
  fn rerank_prefers_trained_positive() {
    let _guard = ENV_LOCK.lock().unwrap();
    let temp = tempdir().unwrap();
    let prev_env = std::env::var("ZAGE_MODEL_PATH").ok();
    unsafe {
      std::env::set_var("ZAGE_MODEL_PATH", temp.path());
    }

    let good_cmd = "git status";
    let bad_cmd = "cargo build";
    let suggestions = [
      Suggestion {
        command: bad_cmd.to_string(),
        score: 0.1,
        breakdown: ScoreBreakdown::default(),
      },
      Suggestion {
        command: good_cmd.to_string(),
        score: 2.0,
        breakdown: ScoreBreakdown::default(),
      },
    ];

    let mut candidates: HashMap<String, Candidate> = HashMap::new();
    candidates.insert(good_cmd.to_string(), candidate_for_test(good_cmd));
    candidates.insert(bad_cmd.to_string(), candidate_for_test(bad_cmd));

    let context = RerankContext {
      repo_root: String::new(),
      recent_heads: vec!["git".to_string()],
      session_tokens: Vec::new(),
      session_phase: None,
      shellname: "sh".to_string(),
      branch: None,
      time_bucket: 0,
      working_directory: None,
      hostname: None,
      username: None,
      session_id: None,
      prev_exit_status: None,
      now: 0,
    };

    let recent_heads: HashSet<String> = context.recent_heads.iter().cloned().collect();
    let good_features = features_from_suggestion(
      &suggestions[1],
      candidates.get(good_cmd).unwrap(),
      &context,
      &recent_heads,
    )
    .unwrap();
    let bad_features = features_from_suggestion(
      &suggestions[0],
      candidates.get(bad_cmd).unwrap(),
      &context,
      &recent_heads,
    )
    .unwrap();

    let feature_matrix =
      build_feature_matrix(&[good_features.values.clone(), bad_features.values.clone()]).unwrap();
    let dataset = Dataset::new(feature_matrix, vec![1.0, 0.0]).unwrap();
    let mut gbrt_config = GBRTConfig::for_binary_classification();
    gbrt_config.n_estimators = 32;
    let mut model = GBRTModel::with_config(gbrt_config).unwrap();
    model.set_feature_names(feature_names());
    model.fit(&dataset).unwrap();

    let model_io = ModelIO::new().unwrap();
    model_io
      .save_model(model.booster(), temp.path(), MODEL_NAME)
      .unwrap();

    let loaded = load_model().unwrap().expect("model should load");
    let matrix =
      build_feature_matrix(&[good_features.values.clone(), bad_features.values.clone()]).unwrap();
    let scores = loaded.predict(&matrix).unwrap();
    assert!(scores.len() >= 2);
    assert!(scores[0] > scores[1]);

    unsafe {
      if let Some(prev) = prev_env {
        std::env::set_var("ZAGE_MODEL_PATH", prev);
      } else {
        std::env::remove_var("ZAGE_MODEL_PATH");
      }
    }
  }
}
