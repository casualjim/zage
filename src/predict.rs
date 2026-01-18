use std::collections::{HashMap, HashSet};

use libsql::{Connection, Value};
use serde::{Deserialize, Serialize};

use crate::Result;
use crate::config::{OnlineModelBlendConfig, OnlineModelConfig};
use crate::core::{Candidate, SystemTimeProvider, TimeProvider};
pub use crate::core::{ScoreBreakdown, Suggestion};
use crate::db::{
  OnlineFeedbackEvent, get_recent_invocations, online_model_meta_set, online_model_meta_value,
  online_model_status,
};
use crate::err::ZageError;
use crate::online_model::trainer::{
  OnlineScoreContext, context_embedding_for_retrieval, score_commands as online_score_commands,
};
use crate::phase::PhaseConfig;
use crate::shell_history::{Invocation, normalize_shellname};
use crate::tokenize::{TokenKind, extract_command_parts, normalized_tokens, tokenize_index};
pub use config::{RankingWeights, SuggestConfig};

pub mod aliases;
mod candidates;
mod config;
mod phase_support;
pub(crate) mod ranking;
mod runtime;
mod sql;
mod templates;
#[cfg(any(test, feature = "tier1-tests"))]
pub mod verifier;

use aliases::{
  add_alias_candidates, alias_for_command, build_prefix_variants, expand_alias, load_aliases,
};
use candidates::{
  add_context_candidates, add_global_candidates, add_head_candidates, add_phase_candidates,
  add_recent_candidates, add_sequence_candidates, add_session_candidates, add_template_candidates,
  add_transition_candidates, add_workspace_candidates, hydrate_candidate_stats, load_session_stats,
  push_opt_i64, push_opt_string,
};
use phase_support::{
  PhaseSignal, command_head_for_phase, detect_session_phase, detect_session_phase_from_commands,
  load_phase_for_heads, phase_match_boost,
};
use ranking::{
  DEFAULT_RECENCY_HALF_LIFE_SECONDS, load_normalized_tokens, recency_score, token_similarity,
};
use runtime::SuggestRuntime;
use sql::query_prepared;
use templates::{
  arg_template_candidates, env_template_candidates, split_env_prefix, token_sequence_predictions,
};

const GLOBAL_CANDIDATE_LIMIT: usize = 50;
const RECENT_CANDIDATE_LIMIT: usize = 200;
const EMBEDDING_CANDIDATE_LIMIT: usize = 150;
const FULL_LINE_POOL_LIMIT: usize = 50;
const CONTEXT_WORKSPACE_WEIGHT: f64 = 1.4;
const CONTEXT_CWD_WEIGHT: f64 = 1.1;
const CONTEXT_EXIT_WEIGHT: f64 = 0.9;
const CONTEXT_HOST_WEIGHT: f64 = 0.2;
const CONTEXT_USER_WEIGHT: f64 = 0.3;
const CONTEXT_TIME_WEIGHT: f64 = 0.05;
const CONTEXT_SESSION_WEIGHT: f64 = 0.03;
const CONTEXT_SESSION_MISS_PENALTY: f64 = 0.6;
const CONTEXT_SESSION_BOOST: f64 = 0.5;
const BLEND_META_KEY: &str = "online_blend_weights_v1";
const BLEND_MIN_WEIGHT: f64 = 0.0;
const BLEND_MAX_WEIGHT: f64 = 5.0;
const BLEND_LR: f64 = 0.05;
const BLEND_L2: f64 = 0.01;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BlendWeights {
  model: f64,
  frecency: f64,
  sequence: f64,
  tier1: f64,
}

impl BlendWeights {
  fn defaults() -> Self {
    Self {
      model: 0.1,
      frecency: 0.5,
      sequence: 1.0,
      tier1: 1.0,
    }
  }

  fn clamp(self) -> Self {
    Self {
      model: self.model.clamp(BLEND_MIN_WEIGHT, BLEND_MAX_WEIGHT),
      frecency: self.frecency.clamp(BLEND_MIN_WEIGHT, BLEND_MAX_WEIGHT),
      sequence: self.sequence.clamp(BLEND_MIN_WEIGHT, BLEND_MAX_WEIGHT),
      tier1: self.tier1.clamp(BLEND_MIN_WEIGHT, BLEND_MAX_WEIGHT),
    }
  }
}

fn expanded_command_for(
  invocation: &crate::shell_history::Invocation,
  aliases: &HashMap<String, String>,
) -> String {
  if !invocation.expanded_command.is_empty() {
    invocation.expanded_command.clone()
  } else {
    expand_alias(&invocation.command, aliases).unwrap_or_else(|| invocation.command.clone())
  }
}

fn normalize_prefix_for_match(prefix: &str) -> String {
  let trimmed = prefix.trim_start();
  let mut out = String::with_capacity(trimmed.len());
  let mut last_was_space = false;
  for ch in trimmed.chars() {
    if ch.is_whitespace() {
      if !last_was_space {
        out.push(' ');
        last_was_space = true;
      }
    } else {
      out.push(ch);
      last_was_space = false;
    }
  }
  out
}

fn apply_prefix_spacing(prefix: &str, normalized_prefix: &str, suggestions: &mut [Suggestion]) {
  if prefix.is_empty() || normalized_prefix.is_empty() {
    return;
  }
  for suggestion in suggestions.iter_mut() {
    let normalized_command = normalize_prefix_for_match(&suggestion.command);
    let Some(rest) = normalized_command.strip_prefix(normalized_prefix) else {
      continue;
    };
    if rest.is_empty() {
      continue;
    }
    suggestion.command = format!("{prefix}{rest}");
  }
}

async fn fetch_command_counts(
  conn: &Connection,
  commands: &[String],
) -> Result<HashMap<String, i64>> {
  let mut out = HashMap::new();
  if commands.is_empty() {
    return Ok(out);
  }

  let mut placeholders = String::new();
  for (idx, _) in commands.iter().enumerate() {
    if idx > 0 {
      placeholders.push(',');
    }
    placeholders.push('?');
  }
  let sql = format!(
    "SELECT command, COUNT(*) FROM shell_history WHERE command IN ({}) GROUP BY command",
    placeholders
  );
  let params = commands
    .iter()
    .map(|cmd| Value::from(cmd.clone()))
    .collect::<Vec<_>>();
  let mut rows = query_prepared(conn, &sql, libsql::params_from_iter(params)).await?;
  while let Some(row) = rows.next().await? {
    let cmd = row.get::<String>(0)?;
    let count = row.get::<i64>(1)?;
    out.insert(cmd, count);
  }
  Ok(out)
}

async fn apply_alias_preference(
  conn: &Connection,
  suggestions: &mut Vec<Suggestion>,
  aliases: &HashMap<String, String>,
) -> Result<()> {
  if suggestions.is_empty() || aliases.is_empty() {
    return Ok(());
  }

  let mut mappings: Vec<(usize, String, String)> = Vec::new();
  let mut command_set: HashSet<String> = HashSet::new();

  for (idx, suggestion) in suggestions.iter().enumerate() {
    for (alias, expansion) in aliases {
      if let Some(alias_command) = alias_for_command(alias, expansion, &suggestion.command) {
        mappings.push((idx, alias_command.clone(), suggestion.command.clone()));
        command_set.insert(alias_command);
        command_set.insert(suggestion.command.clone());
        break;
      }
    }
  }

  if mappings.is_empty() {
    return Ok(());
  }

  let commands = command_set.into_iter().collect::<Vec<_>>();
  let counts = fetch_command_counts(conn, &commands).await?;

  for (idx, alias_command, expanded_command) in mappings {
    let alias_count = counts.get(&alias_command).copied().unwrap_or(0);
    let expanded_count = counts.get(&expanded_command).copied().unwrap_or(0);
    if alias_count > expanded_count
      && let Some(suggestion) = suggestions.get_mut(idx)
    {
      suggestion.command = alias_command;
    }
  }

  let mut seen = HashSet::new();
  suggestions.retain(|suggestion| seen.insert(suggestion.command.clone()));
  Ok(())
}

fn resolve_shellname(config: &SuggestConfig) -> Result<String> {
  let Some(shellname) = config.shellname.as_deref() else {
    return Err(ZageError::ConfigError(
      "shellname is required for suggestions".to_string(),
    ));
  };
  let normalized = normalize_shellname(shellname);
  if normalized.is_empty() {
    return Err(ZageError::ConfigError(
      "shellname is required for suggestions".to_string(),
    ));
  }
  Ok(normalized)
}

fn resolve_workspace_root(cwd: Option<&str>) -> String {
  let Some(cwd) = cwd else {
    return String::new();
  };
  if let Ok(Some(info)) = crate::workspace::detect_workspace_for_cwd(cwd)
    && !info.root.trim().is_empty()
  {
    return info.root.trim_end_matches('/').to_string();
  }
  String::new()
}

async fn load_recent_invocations(
  conn: &Connection,
  config: &SuggestConfig,
) -> Result<Vec<Invocation>> {
  let mut recent = get_recent_invocations(conn, config.session_id, config.recent_limit).await?;
  if recent.is_empty() && config.session_id.is_some() {
    recent = get_recent_invocations(conn, None, config.recent_limit).await?;
  }
  Ok(recent)
}

pub async fn suggest(conn: &Connection, config: SuggestConfig) -> Result<Vec<Suggestion>> {
  suggest_with_aliases(conn, config, load_aliases()).await
}

pub async fn suggest_with_aliases(
  conn: &Connection,
  config: SuggestConfig,
  aliases: HashMap<String, String>,
) -> Result<Vec<Suggestion>> {
  let time_provider = SystemTimeProvider;
  let runtime = SuggestRuntime {
    aliases,
    weights: RankingWeights::default(),
    recency_half_life: DEFAULT_RECENCY_HALF_LIFE_SECONDS,
    now: time_provider.now(),
    phase_config: None,
  };
  suggest_with_runtime(conn, config, &runtime, None).await
}

pub(crate) async fn suggest_with_runtime(
  conn: &Connection,
  config: SuggestConfig,
  runtime: &SuggestRuntime,
  override_prev: Option<(String, Option<i64>)>,
) -> Result<Vec<Suggestion>> {
  let prefix = config.prefix.clone().unwrap_or_default();
  let prefix_norm = normalized_tokens(&prefix);
  let has_prefix = !prefix.is_empty();

  if has_prefix {
    return suggest_completions(conn, &config, runtime, &prefix, &prefix_norm).await;
  }

  let aliases = &runtime.aliases;
  let Some(context) =
    build_pipeline_context(conn, &config, runtime, override_prev.as_ref(), aliases).await?
  else {
    return Ok(Vec::new());
  };

  let collected = collect_candidates(conn, &config, &context, aliases, runtime.now).await?;
  let feature_args = FeatureMatrixArgs {
    conn,
    context: &context,
    candidates: &collected.candidates,
    prefix_norm: &prefix_norm,
    weights: &runtime.weights,
    now: runtime.now,
    recency_half_life: runtime.recency_half_life,
    session_id: config.session_id,
    cwd: config.cwd.as_deref(),
    hostname: config.hostname.as_deref(),
    username: config.username.as_deref(),
  };
  let feature_context = build_feature_matrix(&feature_args);
  let scored = model_score(&feature_context).await?;

  let mut filtered = final_filter(
    scored,
    &config,
    &runtime.weights,
    context.last_command.as_ref(),
  );
  apply_alias_preference(conn, &mut filtered, aliases).await?;
  Ok(filtered)
}

struct PipelineContext {
  sequence_commands: Vec<String>,
  recent_heads: Vec<String>,
  last_command: Option<String>,
  last_exit_status: Option<i64>,
  workspace_root: String,
  shellname: String,
  phase_config: Option<PhaseConfig>,
  session_phase: Option<PhaseSignal>,
}

struct CollectedCandidates {
  candidates: HashMap<String, Candidate>,
}

async fn build_pipeline_context(
  conn: &Connection,
  config: &SuggestConfig,
  runtime: &SuggestRuntime,
  override_prev: Option<&(String, Option<i64>)>,
  aliases: &HashMap<String, String>,
) -> Result<Option<PipelineContext>> {
  let recent = load_recent_invocations(conn, config).await?;
  if recent.is_empty() {
    return Ok(None);
  }

  let recent_commands: Vec<String> = recent
    .iter()
    .map(|inv| expanded_command_for(inv, aliases))
    .collect();
  let mut sequence_commands = recent_commands.clone();
  if let Some((cmd, _)) = override_prev {
    if let Some(last) = sequence_commands.last_mut() {
      *last = cmd.clone();
    } else {
      sequence_commands.push(cmd.clone());
    }
  }
  let recent_heads: Vec<String> = recent
    .iter()
    .map(|inv| (inv, expanded_command_for(inv, aliases)))
    .filter_map(|(inv, command)| command_head_for_phase(&inv.shellname, &command))
    .collect();
  let last_command = override_prev
    .map(|(cmd, _)| cmd.clone())
    .or_else(|| recent_commands.last().cloned());
  let last_exit_status = override_prev
    .map(|(_, exit)| *exit)
    .unwrap_or_else(|| recent.last().and_then(|inv| inv.exit_status));
  let shellname = resolve_shellname(config)?;
  let workspace_root = resolve_workspace_root(config.cwd.as_deref());

  let recent_head_set: HashSet<String> = recent_heads.iter().cloned().collect();
  let phase_for_recent = load_phase_for_heads(conn, &recent_head_set).await?;
  let phase_config = if let Some(override_config) = runtime.phase_config.as_ref() {
    override_config.clone()
  } else {
    PhaseConfig::load()?
  };
  let phase_config = if phase_config.labels().len() > 1 {
    Some(phase_config)
  } else {
    None
  };
  let session_phase = phase_config
    .as_ref()
    .and_then(|config| detect_session_phase_from_commands(&recent_commands, config))
    .or_else(|| detect_session_phase(&recent_heads, &phase_for_recent));

  Ok(Some(PipelineContext {
    sequence_commands,
    recent_heads,
    last_command,
    last_exit_status,
    workspace_root,
    shellname,
    phase_config,
    session_phase,
  }))
}

async fn collect_candidates(
  conn: &Connection,
  config: &SuggestConfig,
  context: &PipelineContext,
  aliases: &HashMap<String, String>,
  now: i64,
) -> Result<CollectedCandidates> {
  let mut candidates: HashMap<String, Candidate> = HashMap::new();

  if let Some(last) = &context.last_command {
    add_transition_candidates(
      conn,
      last,
      context.last_exit_status,
      &context.workspace_root,
      &mut candidates,
    )
    .await?;
  }

  if let Some(session_id) = config.session_id {
    add_session_candidates(conn, session_id, &mut candidates).await?;
  }

  let online_cfg = OnlineModelConfig::load()?;
  let online_status = online_model_status(conn).await?;
  let warmed_up = online_status.token_embeddings > 0 || online_status.group_scalars > 0;
  if warmed_up {
    let query = context_embedding_for_retrieval(
      conn,
      OnlineScoreContext {
        shellname: context.shellname.as_str(),
        workspace_root: context.workspace_root.as_str(),
        cwd: config.cwd.as_deref(),
        hostname: config.hostname.as_deref(),
        username: config.username.as_deref(),
        phase: context
          .session_phase
          .as_ref()
          .map(|phase| phase.phase.as_str()),
        exit_status: context.last_exit_status,
        session_id: config.session_id,
        unix_timestamp: now,
        recent_commands: &context.sequence_commands,
        window: online_cfg.window,
      },
      &online_cfg,
    )
    .await?;
    if let Some(query) = query {
      let similar =
        crate::embeddings::search_similar_commands(conn, &query, EMBEDDING_CANDIDATE_LIMIT).await?;
      if !similar.is_empty() {
        let model_scores = online_score_commands(
          conn,
          OnlineScoreContext {
            shellname: context.shellname.as_str(),
            workspace_root: context.workspace_root.as_str(),
            cwd: config.cwd.as_deref(),
            hostname: config.hostname.as_deref(),
            username: config.username.as_deref(),
            phase: context
              .session_phase
              .as_ref()
              .map(|phase| phase.phase.as_str()),
            exit_status: context.last_exit_status,
            session_id: config.session_id,
            unix_timestamp: now,
            recent_commands: &context.sequence_commands,
            window: online_cfg.window,
          },
          &similar,
          &online_cfg,
        )
        .await?;
        let gate = online_model_gate(&model_scores, online_cfg.blend).unwrap_or(0.0);
        if gate > 0.0 {
          for cmd in similar {
            let entry = candidates
              .entry(cmd.clone())
              .or_insert_with(|| Candidate::new(&cmd));
            entry.from_embedding = true;
          }
        }
      }
    }
  }

  add_phase_candidates(
    conn,
    context.session_phase.as_ref(),
    &context.workspace_root,
    &mut candidates,
  )
  .await?;

  add_context_candidates(conn, config, &mut candidates).await?;

  if !context.workspace_root.is_empty() {
    add_workspace_candidates(conn, &context.workspace_root, &mut candidates).await?;
  }

  if !context.recent_heads.is_empty() {
    add_head_candidates(
      conn,
      &context.recent_heads,
      &context.workspace_root,
      &mut candidates,
    )
    .await?;
  }

  if config.use_sequences {
    add_sequence_candidates(
      conn,
      &context.sequence_commands,
      context.shellname.as_str(),
      &mut candidates,
    )
    .await?;
  }

  if !candidates.is_empty() {
    add_template_candidates(
      conn,
      &context.workspace_root,
      context.shellname.as_str(),
      &mut candidates,
    )
    .await?;
  }

  if candidates.is_empty() {
    add_global_candidates(conn, &mut candidates, GLOBAL_CANDIDATE_LIMIT).await?;
    add_template_candidates(
      conn,
      &context.workspace_root,
      context.shellname.as_str(),
      &mut candidates,
    )
    .await?;
  }

  if candidates.len() < 25 {
    add_recent_candidates(conn, &mut candidates, RECENT_CANDIDATE_LIMIT).await?;
    add_global_candidates(conn, &mut candidates, GLOBAL_CANDIDATE_LIMIT).await?;
    add_template_candidates(
      conn,
      &context.workspace_root,
      context.shellname.as_str(),
      &mut candidates,
    )
    .await?;
  }

  let session_stats = if let Some(session_id) = config.session_id {
    load_session_stats(conn, session_id, config.recent_limit).await?
  } else {
    HashMap::new()
  };

  if !session_stats.is_empty() {
    apply_session_stats(&session_stats, &mut candidates);
  }

  if !candidates.is_empty() {
    hydrate_candidate_stats(conn, &context.workspace_root, &mut candidates).await?;
  }

  if !aliases.is_empty() {
    add_alias_candidates(aliases, &mut candidates);
    if !session_stats.is_empty() {
      apply_session_stats(&session_stats, &mut candidates);
    }
  }

  Ok(CollectedCandidates { candidates })
}

fn apply_session_stats(
  session_stats: &HashMap<String, (i64, i64)>,
  candidates: &mut HashMap<String, Candidate>,
) {
  for (cmd, (freq, last_seen)) in session_stats {
    let entry = candidates
      .entry(cmd.clone())
      .or_insert_with(|| Candidate::new(cmd));
    entry.session_freq = entry.session_freq.max(*freq);
    entry.session_last_seen = entry.session_last_seen.max(*last_seen);
    entry.last_seen = entry.last_seen.max(*last_seen);
  }
}

struct FeatureMatrixArgs<'a> {
  conn: &'a Connection,
  context: &'a PipelineContext,
  candidates: &'a HashMap<String, Candidate>,
  prefix_norm: &'a [String],
  weights: &'a RankingWeights,
  now: i64,
  recency_half_life: f64,
  session_id: Option<i64>,
  cwd: Option<&'a str>,
  hostname: Option<&'a str>,
  username: Option<&'a str>,
}

fn build_feature_matrix<'a>(args: &'a FeatureMatrixArgs<'a>) -> ScoreContext<'a> {
  ScoreContext {
    conn: args.conn,
    candidates: args.candidates,
    prefix_norm: args.prefix_norm,
    shellname: args.context.shellname.as_str(),
    sequence_commands: &args.context.sequence_commands,
    cwd: args.cwd,
    hostname: args.hostname,
    username: args.username,
    exit_status: args.context.last_exit_status,
    session_phase: args.context.session_phase.as_ref(),
    session_id: args.session_id,
    recent_heads: &args.context.recent_heads,
    weights: args.weights,
    now: args.now,
    recency_half_life: args.recency_half_life,
    phase_config: args.context.phase_config.as_ref(),
    workspace_root: &args.context.workspace_root,
  }
}

async fn model_score(context: &ScoreContext<'_>) -> Result<Vec<Suggestion>> {
  score_candidates(context).await
}

fn final_filter(
  mut scored: Vec<Suggestion>,
  config: &SuggestConfig,
  weights: &RankingWeights,
  last_command: Option<&String>,
) -> Vec<Suggestion> {
  let transition_only = weights.transition > 0.0
    && weights.recency.abs() <= f64::EPSILON
    && weights.frequency.abs() <= f64::EPSILON
    && weights.context.abs() <= f64::EPSILON
    && weights.sequence.abs() <= f64::EPSILON
    && weights.similarity.abs() <= f64::EPSILON;
  if transition_only && last_command.is_some() {
    let has_transition = scored.iter().any(|s| s.breakdown.transition > 0.0);
    if has_transition {
      scored.retain(|s| s.breakdown.transition > 0.0);
    }
  }

  scored.truncate(config.max_results);
  scored
}

struct ScoreContext<'a> {
  conn: &'a Connection,
  candidates: &'a HashMap<String, Candidate>,
  prefix_norm: &'a [String],
  shellname: &'a str,
  sequence_commands: &'a [String],
  cwd: Option<&'a str>,
  hostname: Option<&'a str>,
  username: Option<&'a str>,
  exit_status: Option<i64>,
  session_phase: Option<&'a PhaseSignal>,
  session_id: Option<i64>,
  recent_heads: &'a [String],
  weights: &'a RankingWeights,
  now: i64,
  recency_half_life: f64,
  phase_config: Option<&'a PhaseConfig>,
  workspace_root: &'a str,
}

fn time_bucket(ts: i64) -> u8 {
  if ts <= 0 {
    return 0;
  }
  let hour = ((ts / 3600) % 24) as u8;
  match hour {
    0..=5 => 1,
    6..=11 => 2,
    12..=17 => 3,
    _ => 4,
  }
}

async fn load_blend_weights(conn: &Connection) -> Result<BlendWeights> {
  let Some(raw) = online_model_meta_value(conn, BLEND_META_KEY).await? else {
    return Ok(BlendWeights::defaults());
  };
  let weights = serde_json::from_str(&raw)
    .map_err(|err| ZageError::ConfigError(format!("invalid online blend weights: {err}")))?;
  Ok(weights)
}

async fn store_blend_weights(conn: &Connection, weights: &BlendWeights) -> Result<()> {
  let json = serde_json::to_string(weights).map_err(|err| {
    ZageError::ConfigError(format!("failed to encode online blend weights: {err}"))
  })?;
  online_model_meta_set(conn, BLEND_META_KEY, &json).await?;
  Ok(())
}

fn sigmoid(value: f64) -> f64 {
  1.0 / (1.0 + (-value).exp())
}

fn frecency_prior(weights: &RankingWeights, suggestion: &Suggestion) -> f64 {
  weights.recency * suggestion.breakdown.recency
    + weights.frequency * suggestion.breakdown.frequency
    + 0.1 * suggestion.breakdown.session_recency
}

fn log_frecency_prior(value: f64) -> f64 {
  if value > 0.0 { value.ln_1p() } else { 0.0 }
}

fn sequence_score(weights: &RankingWeights, suggestion: &Suggestion) -> f64 {
  weights.sequence * suggestion.breakdown.sequence
}

fn tier1_score(weights: &RankingWeights, suggestion: &Suggestion) -> f64 {
  weights.transition * suggestion.breakdown.transition
    + weights.context * suggestion.breakdown.context
    + weights.similarity * suggestion.breakdown.similarity
}

fn online_model_gate(model_scores: &[f32], cfg: OnlineModelBlendConfig) -> Option<f64> {
  if model_scores.is_empty() {
    return None;
  }

  let mut top1 = f32::NEG_INFINITY;
  let mut top2 = f32::NEG_INFINITY;
  for &score in model_scores {
    if !score.is_finite() {
      continue;
    }
    if score > top1 {
      top2 = top1;
      top1 = score;
    } else if score > top2 {
      top2 = score;
    }
  }

  if !top1.is_finite() {
    return None;
  }

  let gate = if (top1 as f64) < cfg.min_score_gate {
    0.0
  } else if cfg.margin_gate <= f64::EPSILON || !top2.is_finite() {
    1.0
  } else {
    let margin = (top1 - top2).max(0.0) as f64;
    if margin >= cfg.margin_gate {
      1.0
    } else {
      (margin / cfg.margin_gate).clamp(0.0, 1.0)
    }
  };

  Some(gate)
}

async fn score_candidates(context: &ScoreContext<'_>) -> Result<Vec<Suggestion>> {
  let mut candidate_heads: HashMap<String, String> = HashMap::new();
  let mut phase_heads: HashSet<String> = context.recent_heads.iter().cloned().collect();
  for candidate in context.candidates.values() {
    if let Some(head) = command_head_for_phase(context.shellname, &candidate.command) {
      phase_heads.insert(head.clone());
      candidate_heads.insert(candidate.command.clone(), head);
    }
  }
  let phase_for_head = load_phase_for_heads(context.conn, &phase_heads).await?;

  let mut scored: Vec<Suggestion> = Vec::new();
  for candidate in context.candidates.values() {
    let recency = recency_score(context.now, candidate.last_seen, context.recency_half_life);
    let workspace_weight = if !context.workspace_root.is_empty() {
      1.0
    } else {
      0.5
    };
    let frequency = (candidate.freq as f64).ln_1p()
      + workspace_weight * (candidate.workspace_freq as f64).ln_1p();
    let mut transition = (candidate.transition_freq as f64).ln_1p()
      + 0.7 * (candidate.workspace_transition_freq as f64).ln_1p();
    if candidate.transition_exit_status_match {
      transition *= 1.3;
    }
    let workspace_context = (candidate.workspace_freq as f64).ln_1p();
    let cwd_context = if candidate.context_cwd_match {
      (candidate.context_freq as f64).ln_1p()
    } else {
      0.0
    };
    let exit_context = if candidate.transition_exit_status_match {
      1.0
    } else {
      0.0
    };
    let host_context = if candidate.context_host_match {
      1.0
    } else {
      0.0
    };
    let user_context = if candidate.context_user_match {
      1.0
    } else {
      0.0
    };
    let time_context = if time_bucket(candidate.last_seen) == time_bucket(context.now) {
      1.0
    } else {
      0.0
    };
    let session_context = (candidate.session_freq as f64).ln_1p();

    let mut context_score = CONTEXT_WORKSPACE_WEIGHT * workspace_context
      + CONTEXT_CWD_WEIGHT * cwd_context
      + CONTEXT_EXIT_WEIGHT * exit_context
      + CONTEXT_HOST_WEIGHT * host_context
      + CONTEXT_USER_WEIGHT * user_context
      + CONTEXT_TIME_WEIGHT * time_context
      + CONTEXT_SESSION_WEIGHT * session_context;
    let head_phase = candidate_heads
      .get(&candidate.command)
      .and_then(|head| phase_for_head.get(head));
    let pattern_phase = context.phase_config.and_then(|config| {
      config
        .match_label(&candidate.command)
        .and_then(|idx| config.labels().get(idx).cloned())
        .map(|phase| PhaseSignal {
          phase,
          confidence: 1.0,
        })
    });
    let candidate_phase = pattern_phase.as_ref().or(head_phase);
    context_score += phase_match_boost(context.session_phase, candidate_phase);
    if context.session_id.is_some() {
      if candidate.session_freq > 0 {
        context_score += CONTEXT_SESSION_BOOST;
      } else {
        context_score *= CONTEXT_SESSION_MISS_PENALTY;
      }
    }
    let mut sequence = if candidate.sequence_confidence > 0.0 {
      let order_weight = 0.8 + 0.1 * (candidate.sequence_prefix_len as f64).min(4.0);
      candidate.sequence_confidence * candidate.sequence_lift.max(1.0) * order_weight
    } else {
      0.0
    };
    if !context.workspace_root.is_empty() && candidate.workspace_freq == 0 {
      sequence *= 0.5;
    }
    let similarity = if context.prefix_norm.is_empty() {
      0.0
    } else {
      token_similarity(
        context.prefix_norm,
        &load_normalized_tokens(context.conn, &candidate.command).await?,
      )
    };
    let session_recency = if candidate.session_last_seen > 0 {
      recency_score(
        context.now,
        candidate.session_last_seen,
        context.recency_half_life,
      )
    } else {
      0.0
    };

    let score = context.weights.recency * recency
      + context.weights.frequency * frequency
      + context.weights.transition * transition
      + context.weights.context * context_score
      + context.weights.sequence * sequence
      + 0.1 * session_recency
      + context.weights.similarity * similarity;

    if !score.is_finite() || score <= 0.0 {
      continue;
    }

    scored.push(Suggestion {
      command: candidate.command.clone(),
      score,
      breakdown: ScoreBreakdown {
        recency,
        session_recency,
        frequency,
        transition,
        context: context_score,
        sequence,
        similarity,
        embedding_retrieval: if candidate.from_embedding { 1.0 } else { 0.0 },
        online_model: 0.0,
      },
    });
  }

  let weights = context.weights;
  let online_cfg = OnlineModelConfig::load()?;
  let online_status = online_model_status(context.conn).await?;
  let warmed_up = online_status.token_embeddings > 0 || online_status.group_scalars > 0;
  let blend_weights = load_blend_weights(context.conn).await?;
  if warmed_up && !scored.is_empty() {
    let commands = scored
      .iter()
      .map(|suggestion| suggestion.command.clone())
      .collect::<Vec<_>>();
    let model_scores = online_score_commands(
      context.conn,
      OnlineScoreContext {
        shellname: context.shellname,
        workspace_root: context.workspace_root,
        cwd: context.cwd,
        hostname: context.hostname,
        username: context.username,
        phase: context
          .session_phase
          .as_ref()
          .map(|phase| phase.phase.as_str()),
        exit_status: context.exit_status,
        session_id: context.session_id,
        unix_timestamp: context.now,
        recent_commands: context.sequence_commands,
        window: online_cfg.window,
      },
      &commands,
      &online_cfg,
    )
    .await?;
    if model_scores.len() == scored.len() {
      let gate = online_model_gate(&model_scores, online_cfg.blend).unwrap_or(0.0);
      let model_scale = online_cfg.blend.alpha * gate;
      if model_scale > 0.0 {
        for (idx, suggestion) in scored.iter_mut().enumerate() {
          let model = model_scores[idx];
          if model.is_finite() {
            let model_feature = (model as f64).tanh();
            suggestion.breakdown.online_model = model_feature * model_scale;
          }
        }
      }
    }
  }

  for suggestion in scored.iter_mut() {
    let frecency = log_frecency_prior(frecency_prior(weights, suggestion));
    let sequence = sequence_score(weights, suggestion);
    let tier1 = tier1_score(weights, suggestion);
    let base = blend_weights.model * suggestion.breakdown.online_model
      + blend_weights.sequence * sequence
      + blend_weights.tier1 * tier1;
    let score = if base.is_finite() && base > 0.0 {
      base
    } else {
      blend_weights.frecency * frecency
    };
    if score.is_finite() {
      suggestion.score = score;
    }
  }

  scored.sort_by(|a, b| {
    let score_cmp = b.score.total_cmp(&a.score);
    if score_cmp != std::cmp::Ordering::Equal {
      return score_cmp;
    }
    let recency_cmp = b.breakdown.recency.total_cmp(&a.breakdown.recency);
    if recency_cmp != std::cmp::Ordering::Equal {
      return recency_cmp;
    }
    a.command.cmp(&b.command)
  });
  Ok(scored)
}

pub(crate) async fn update_blend_weights_for_feedback(
  conn: &Connection,
  event: &OnlineFeedbackEvent,
) -> Result<()> {
  if let Some(outcome) = event.outcome.as_deref()
    && !outcome.eq_ignore_ascii_case("accepted")
  {
    return Ok(());
  }
  let command = event
    .accepted_command
    .as_deref()
    .unwrap_or(event.suggestion.as_str())
    .trim();
  if command.is_empty() {
    return Ok(());
  }
  let now = event.accepted_at.unwrap_or(event.shown_at);
  if now <= 0 {
    return Ok(());
  }

  let cwd = event.cwd.as_deref();
  let workspace_root = resolve_workspace_root(cwd);

  let mut candidates = HashMap::new();
  candidates.insert(command.to_string(), Candidate::new(command));
  hydrate_candidate_stats(conn, &workspace_root, &mut candidates).await?;
  if let (Some(cwd), Some(candidate)) = (cwd, candidates.get_mut(command)) {
    let mut rows = query_prepared(
      conn,
      "SELECT freq, last_seen FROM context_stats
       WHERE command = ? AND working_directory = ?
       ORDER BY freq DESC LIMIT 1",
      libsql::params![command.to_string(), cwd.to_string()],
    )
    .await?;
    if let Some(row) = rows.next().await? {
      let freq = row.get::<i64>(0)?;
      let last_seen = row.get::<i64>(1)?;
      candidate.context_freq = candidate.context_freq.max(freq);
      candidate.context_cwd_match = true;
      candidate.last_seen = candidate.last_seen.max(last_seen);
    }
  }

  let prefix_norm: Vec<String> = Vec::new();
  let sequence_commands: Vec<String> = Vec::new();
  let recent_heads: Vec<String> = Vec::new();
  let weights = RankingWeights::default();
  let context = ScoreContext {
    conn,
    candidates: &candidates,
    prefix_norm: &prefix_norm,
    shellname: "zsh",
    sequence_commands: &sequence_commands,
    cwd,
    hostname: None,
    username: None,
    exit_status: None,
    session_phase: None,
    session_id: None,
    recent_heads: &recent_heads,
    weights: &weights,
    now,
    recency_half_life: DEFAULT_RECENCY_HALF_LIFE_SECONDS,
    phase_config: None,
    workspace_root: &workspace_root,
  };

  let scored = score_candidates(&context).await?;
  let Some(suggestion) = scored.iter().find(|s| s.command == command) else {
    return Ok(());
  };

  let frecency = log_frecency_prior(frecency_prior(&weights, suggestion));
  let sequence = sequence_score(&weights, suggestion);
  let tier1 = tier1_score(&weights, suggestion);
  let model = suggestion.breakdown.online_model;
  if !model.is_finite() || !frecency.is_finite() || !sequence.is_finite() || !tier1.is_finite() {
    return Ok(());
  }
  if model == 0.0 && frecency == 0.0 && sequence == 0.0 && tier1 == 0.0 {
    return Ok(());
  }

  let defaults = BlendWeights::defaults();
  let mut blend = load_blend_weights(conn).await?;
  let base = blend.model * model + blend.sequence * sequence + blend.tier1 * tier1;
  let effective_frecency = if base.is_finite() && base > 0.0 {
    0.0
  } else {
    frecency
  };
  let logit = base + blend.frecency * effective_frecency;
  let prob = sigmoid(logit);
  let error = prob - 1.0;

  blend.model -= BLEND_LR * (error * model + BLEND_L2 * (blend.model - defaults.model));
  blend.frecency -=
    BLEND_LR * (error * effective_frecency + BLEND_L2 * (blend.frecency - defaults.frecency));
  blend.sequence -= BLEND_LR * (error * sequence + BLEND_L2 * (blend.sequence - defaults.sequence));
  blend.tier1 -= BLEND_LR * (error * tier1 + BLEND_L2 * (blend.tier1 - defaults.tier1));
  let blend = blend.clamp();
  store_blend_weights(conn, &blend).await?;
  Ok(())
}

async fn suggest_completions(
  conn: &Connection,
  config: &SuggestConfig,
  runtime: &SuggestRuntime,
  prefix: &str,
  prefix_norm: &[String],
) -> Result<Vec<Suggestion>> {
  let aliases = &runtime.aliases;
  if let Some(session_id) = config.session_id {
    let session_scored = completion_candidates(
      conn,
      config,
      runtime,
      prefix,
      prefix_norm,
      aliases,
      Some(session_id),
    )
    .await?;
    if !session_scored.is_empty() {
      return Ok(session_scored);
    }
  }

  completion_candidates(conn, config, runtime, prefix, prefix_norm, aliases, None).await
}

async fn completion_candidates(
  conn: &Connection,
  config: &SuggestConfig,
  runtime: &SuggestRuntime,
  prefix: &str,
  prefix_norm: &[String],
  aliases: &HashMap<String, String>,
  session_filter: Option<i64>,
) -> Result<Vec<Suggestion>> {
  let normalized_prefix = normalize_prefix_for_match(prefix);
  let workspace_root = resolve_workspace_root(config.cwd.as_deref());
  let project_root = if workspace_root.is_empty() {
    None
  } else {
    Some(workspace_root.clone())
  };
  let project_like = project_root.as_ref().map(|root| format!("{root}/%"));

  let shellname = resolve_shellname(config)?;
  let prefer_full_line = config.prefer_full_line;
  let pool_limit = if prefer_full_line {
    config.max_results.max(FULL_LINE_POOL_LIMIT)
  } else {
    config.max_results
  };
  let token_priors = token_sequence_predictions(conn, prefix_norm).await?;
  let (prefix_flags, prefix_args) = {
    let tokens = tokenize_index(shellname.as_str(), prefix);
    let ends_with_space = prefix
      .chars()
      .last()
      .map(|c| c.is_whitespace())
      .unwrap_or(false);
    extract_command_parts(prefix, &tokens)
      .map(|parts| {
        let mut flags = parts.flags;
        let mut args = parts
          .args
          .iter()
          .map(|arg| arg.normalized.clone())
          .collect::<Vec<_>>();
        if !ends_with_space
          && let Some(last) = tokens.last()
          && matches!(
            last.kind,
            TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment
          )
        {
          let partial_norm = last.normalized.clone();
          if last.raw.starts_with('-') {
            if last.raw.len() == 1 {
              if let Some(pos) = flags.iter().position(|flag| flag == &last.raw) {
                flags.remove(pos);
              } else if let Some(pos) = args.iter().position(|arg| arg == &partial_norm) {
                args.remove(pos);
              }
            }
          } else if let Some(pos) = args.iter().position(|arg| arg == &partial_norm) {
            args.remove(pos);
          }
        }
        (flags, args)
      })
      .unwrap_or_default()
  };

  let mut env_suggestions = None;
  if let Some(mut suggestions) = env_template_candidates(
    conn,
    prefix,
    &workspace_root,
    &token_priors,
    runtime.now,
    runtime.recency_half_life,
  )
  .await?
  {
    suggestions.sort_by(|a, b| b.score.total_cmp(&a.score));
    suggestions.truncate(pool_limit);
    env_suggestions = Some(suggestions);
  }

  let mut arg_suggestions_for_merge = None;
  if let Some(mut arg_suggestions) = arg_template_candidates(
    conn,
    prefix,
    &workspace_root,
    shellname.as_str(),
    &token_priors,
    runtime.now,
    runtime.recency_half_life,
  )
  .await?
  {
    let has_prefix_match = arg_suggestions.iter().any(|suggestion| {
      normalize_prefix_for_match(&suggestion.command).starts_with(&normalized_prefix)
    });
    if !has_prefix_match {
      // fall through to normal completion candidates
    } else {
      let trimmed_prefix = normalized_prefix.trim_end();
      arg_suggestions.retain(|suggestion| {
        if normalize_prefix_for_match(&suggestion.command).trim_end() == trimmed_prefix {
          return false;
        }
        if prefix_flags.is_empty() && prefix_args.is_empty() {
          return true;
        }
        let tokens = tokenize_index(shellname.as_str(), &suggestion.command);
        let Some(parts) = extract_command_parts(&suggestion.command, &tokens) else {
          return false;
        };
        if !prefix_flags
          .iter()
          .all(|flag| parts.flags.iter().any(|cand| cand == flag))
        {
          return false;
        }
        if !prefix_args.is_empty() {
          let candidate_args = parts
            .args
            .iter()
            .map(|arg| arg.normalized.clone())
            .collect::<Vec<_>>();
          if !prefix_args
            .iter()
            .all(|arg| candidate_args.iter().any(|cand| cand == arg))
          {
            return false;
          }
        }
        true
      });
      if arg_suggestions.is_empty() {
        // fall through to normal completion candidates
      } else {
        arg_suggestions.sort_by(|a, b| b.score.total_cmp(&a.score));
        arg_suggestions.truncate(pool_limit);
        let last_char = prefix.chars().last();
        let ends_with_space = last_char.map(|c| c.is_whitespace()).unwrap_or(false);
        let ends_with_quote = matches!(last_char, Some('"') | Some('\''));
        if prefer_full_line || (ends_with_space && !prefix_flags.is_empty()) || ends_with_quote {
          arg_suggestions_for_merge = Some(arg_suggestions);
        } else {
          apply_prefix_spacing(prefix, &normalized_prefix, &mut arg_suggestions);
          return Ok(arg_suggestions);
        }
      }
    }
  }

  let (env_prefix, mut match_prefix) = split_env_prefix(prefix);
  let mut apply_env_prefix = env_prefix.clone();
  if !env_prefix.is_empty() && match_prefix.trim().is_empty() {
    match_prefix = prefix.to_string();
    apply_env_prefix.clear();
  }
  let normalized_match_prefix = normalize_prefix_for_match(&match_prefix);
  let match_prefixes = build_prefix_variants(&normalized_match_prefix, aliases);
  if match_prefixes.is_empty() {
    let mut suggestions = env_suggestions.unwrap_or_default();
    apply_prefix_spacing(prefix, &normalized_prefix, &mut suggestions);
    return Ok(suggestions);
  }

  let mut sql = String::from(
    "SELECT command,
            MAX(COALESCE(start_unix_timestamp, 0)) as last_seen,
            COUNT(*) as freq,
            SUM(CASE WHEN ? IS NOT NULL AND working_directory = ? THEN 1 ELSE 0 END) as cwd_hits,
            SUM(CASE WHEN ? IS NOT NULL AND (working_directory = ? OR working_directory LIKE ?) THEN 1 ELSE 0 END) as project_hits,
            SUM(CASE WHEN ? IS NOT NULL AND hostname = ? THEN 1 ELSE 0 END) as host_hits,
            SUM(CASE WHEN ? IS NOT NULL AND username = ? THEN 1 ELSE 0 END) as user_hits,
            SUM(CASE WHEN ? IS NOT NULL AND session_id = ? THEN 1 ELSE 0 END) as session_hits
     FROM shell_history
     WHERE ",
  );

  let mut where_parts: Vec<String> = Vec::new();
  let mut params: Vec<Value> = Vec::new();

  push_opt_string(&mut params, &config.cwd);
  push_opt_string(&mut params, &config.cwd);
  push_opt_string(&mut params, &project_root);
  push_opt_string(&mut params, &project_root);
  push_opt_string(&mut params, &project_like);
  push_opt_string(&mut params, &config.hostname);
  push_opt_string(&mut params, &config.hostname);
  push_opt_string(&mut params, &config.username);
  push_opt_string(&mut params, &config.username);
  push_opt_i64(&mut params, config.session_id);
  push_opt_i64(&mut params, config.session_id);

  if let Some(session_id) = session_filter {
    where_parts.push("session_id = ?".to_string());
    params.push(Value::from(session_id));
  }

  let like_clause = match_prefixes
    .iter()
    .map(|_| "command LIKE ?")
    .collect::<Vec<_>>()
    .join(" OR ");
  where_parts.push(format!("({like_clause})"));

  sql.push_str(&where_parts.join(" AND "));
  sql.push_str(" GROUP BY command ORDER BY last_seen DESC LIMIT 200");

  for prefix_value in &match_prefixes {
    params.push(Value::from(format!("{prefix_value}%")));
  }

  let mut rows = query_prepared(conn, &sql, libsql::params_from_iter(params)).await?;

  let mut scored: Vec<Suggestion> = Vec::new();

  while let Some(row) = rows.next().await? {
    let command = row.get::<String>(0)?;
    let last_seen = row.get::<i64>(1)?;
    let freq = row.get::<i64>(2)?;
    let cwd_hits = row.get::<i64>(3)?;
    let project_hits = row.get::<i64>(4)?;
    let host_hits = row.get::<i64>(5)?;
    let user_hits = row.get::<i64>(6)?;
    let session_hits = row.get::<i64>(7)?;

    let expanded = expand_alias(&command, aliases);
    let expanded_for_score = expanded.as_deref().unwrap_or(&command);
    let matches_prefix = match_prefixes
      .iter()
      .any(|variant| command.starts_with(variant) || expanded_for_score.starts_with(variant));
    if !matches_prefix {
      continue;
    }
    let norm_tokens = normalized_tokens(expanded_for_score);
    let similarity = token_similarity(prefix_norm, &norm_tokens);
    let prefix_score = if command.starts_with(&normalized_match_prefix) {
      1.0
    } else if expanded_for_score.starts_with(&normalized_match_prefix) {
      0.8
    } else {
      0.0
    };

    let recency = recency_score(runtime.now, last_seen, runtime.recency_half_life);
    let frequency = (freq as f64).ln_1p();
    let mut context = 0.0;
    if session_hits > 0 {
      context += 1.0;
    }
    if cwd_hits > 0 {
      context += 0.9;
    } else if project_hits > 0 {
      context += 0.6;
    }
    if host_hits > 0 {
      context += 0.3;
    }
    if user_hits > 0 {
      context += 0.3;
    }

    let score =
      0.45 * recency + 0.3 * context + 0.15 * frequency + 0.07 * prefix_score + 0.03 * similarity;

    let mut suggestion_command = command.clone();
    for (alias, expansion) in aliases {
      if let Some(alias_command) = alias_for_command(alias, expansion, &command)
        && alias_command.starts_with(&normalized_match_prefix)
      {
        suggestion_command = alias_command;
        break;
      }
    }
    if !apply_env_prefix.is_empty() {
      let prefix = if apply_env_prefix
        .chars()
        .last()
        .map(|c| c.is_whitespace())
        .unwrap_or(false)
      {
        apply_env_prefix.clone()
      } else {
        format!("{apply_env_prefix} ")
      };
      suggestion_command = format!("{prefix}{suggestion_command}");
    }

    scored.push(Suggestion {
      command: suggestion_command,
      score,
      breakdown: ScoreBreakdown {
        recency,
        session_recency: 0.0,
        frequency,
        transition: 0.0,
        context,
        sequence: 0.0,
        similarity,
        embedding_retrieval: 0.0,
        online_model: 0.0,
      },
    });
  }

  if scored.is_empty() {
    if prefix_flags.is_empty() && prefix_args.is_empty() {
      let mut merged = env_suggestions.unwrap_or_default();
      if let Some(mut arg_suggestions) = arg_suggestions_for_merge.take() {
        merged.append(&mut arg_suggestions);
      }
      apply_prefix_spacing(prefix, &normalized_prefix, &mut merged);
      return Ok(merged);
    }
    let mut rows = query_prepared(
      conn,
      "SELECT command, freq, last_seen FROM command_stats ORDER BY last_seen DESC LIMIT 200",
      (),
    )
    .await?;
    while let Some(row) = rows.next().await? {
      let command = row.get::<String>(0)?;
      let freq = row.get::<i64>(1)?;
      let last_seen = row.get::<i64>(2)?;
      if !prefix_flags.is_empty() {
        let tokens = tokenize_index(shellname.as_str(), &command);
        let Some(parts) = extract_command_parts(&command, &tokens) else {
          continue;
        };
        if !prefix_flags
          .iter()
          .all(|flag| parts.flags.iter().any(|cand| cand == flag))
        {
          continue;
        }
        if !prefix_args.is_empty() {
          let candidate_args = parts
            .args
            .iter()
            .map(|arg| arg.normalized.clone())
            .collect::<Vec<_>>();
          if !prefix_args
            .iter()
            .all(|arg| candidate_args.iter().any(|cand| cand == arg))
          {
            continue;
          }
        }
      } else if !prefix_args.is_empty() {
        let tokens = tokenize_index(shellname.as_str(), &command);
        let Some(parts) = extract_command_parts(&command, &tokens) else {
          continue;
        };
        let candidate_args = parts
          .args
          .iter()
          .map(|arg| arg.normalized.clone())
          .collect::<Vec<_>>();
        if !prefix_args
          .iter()
          .all(|arg| candidate_args.iter().any(|cand| cand == arg))
        {
          continue;
        }
      }
      let expanded = expand_alias(&command, aliases);
      let expanded_for_score = expanded.as_deref().unwrap_or(&command);
      let norm_tokens = normalized_tokens(expanded_for_score);
      let similarity = token_similarity(prefix_norm, &norm_tokens);
      if similarity <= 0.0 {
        continue;
      }
      let recency = recency_score(runtime.now, last_seen, runtime.recency_half_life);
      let frequency = (freq as f64).ln_1p();
      let score = 0.7 * similarity + 0.2 * recency + 0.1 * frequency;

      let mut suggestion_command = command.clone();
      for (alias, expansion) in aliases {
        if let Some(alias_command) = alias_for_command(alias, expansion, &command)
          && alias_command.starts_with(&normalized_match_prefix)
        {
          suggestion_command = alias_command;
          break;
        }
      }
      if !apply_env_prefix.is_empty() {
        let prefix = if apply_env_prefix
          .chars()
          .last()
          .map(|c| c.is_whitespace())
          .unwrap_or(false)
        {
          apply_env_prefix.clone()
        } else {
          format!("{apply_env_prefix} ")
        };
        suggestion_command = format!("{prefix}{suggestion_command}");
      }

      scored.push(Suggestion {
        command: suggestion_command,
        score,
        breakdown: ScoreBreakdown {
          recency,
          session_recency: 0.0,
          frequency,
          transition: 0.0,
          context: 0.0,
          sequence: 0.0,
          similarity,
          embedding_retrieval: 0.0,
          online_model: 0.0,
        },
      });
    }
  }

  if !prefix.is_empty() {
    let ends_with_space = normalized_prefix
      .chars()
      .last()
      .map(|ch| ch.is_whitespace())
      .unwrap_or(false);
    let match_prefix = if ends_with_space {
      normalized_prefix.trim_end()
    } else {
      normalized_prefix.as_str()
    };
    if match_prefix.is_empty() {
      scored.clear();
    } else {
      let has_prefix_match = scored.iter().any(|suggestion| {
        normalize_prefix_for_match(&suggestion.command).starts_with(match_prefix)
      });
      if has_prefix_match {
        if ends_with_space {
          scored.retain(|suggestion| {
            let normalized_command = normalize_prefix_for_match(&suggestion.command);
            if !normalized_command.starts_with(match_prefix) {
              return false;
            }
            let rest = &normalized_command[match_prefix.len()..];
            rest
              .chars()
              .next()
              .map(|ch| ch.is_whitespace())
              .unwrap_or(true)
          });
        } else {
          scored.retain(|suggestion| {
            normalize_prefix_for_match(&suggestion.command).starts_with(match_prefix)
          });
        }
      }
    }
  }

  apply_online_model_for_completions(conn, config, runtime, aliases, &workspace_root, &mut scored)
    .await?;

  if prefer_full_line && !scored.is_empty() {
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    scored.truncate(config.max_results);
    apply_prefix_spacing(prefix, &normalized_prefix, &mut scored);
    return Ok(scored);
  }

  if let Some(mut suggestions) = env_suggestions {
    if !scored.is_empty() {
      let weight: f64 = if apply_env_prefix.is_empty() {
        0.8
      } else {
        1.0
      };
      if (weight - 1.0).abs() > f64::EPSILON {
        for suggestion in suggestions.iter_mut() {
          suggestion.score *= weight;
          suggestion.breakdown.recency *= weight;
          suggestion.breakdown.frequency *= weight;
          suggestion.breakdown.context *= weight;
          suggestion.breakdown.sequence *= weight;
          suggestion.breakdown.similarity *= weight;
          suggestion.breakdown.online_model *= weight;
        }
      }
    }
    scored.extend(suggestions);
  }

  if let Some(mut suggestions) = arg_suggestions_for_merge {
    scored.append(&mut suggestions);
  }

  scored.sort_by(|a, b| b.score.total_cmp(&a.score));
  scored.truncate(config.max_results);
  apply_prefix_spacing(prefix, &normalized_prefix, &mut scored);
  Ok(scored)
}

async fn apply_online_model_for_completions(
  conn: &Connection,
  config: &SuggestConfig,
  runtime: &SuggestRuntime,
  aliases: &HashMap<String, String>,
  workspace_root: &str,
  scored: &mut [Suggestion],
) -> Result<()> {
  if scored.is_empty() {
    return Ok(());
  }

  let online_cfg = OnlineModelConfig::load()?;
  let online_status = online_model_status(conn).await?;
  let warmed_up = online_status.token_embeddings > 0 || online_status.group_scalars > 0;
  if !warmed_up {
    return Ok(());
  }

  let recent = load_recent_invocations(conn, config).await?;
  let recent_commands: Vec<String> = recent
    .iter()
    .map(|inv| expanded_command_for(inv, aliases))
    .collect();
  let last_exit_status = recent.last().and_then(|inv| inv.exit_status);
  let shellname = resolve_shellname(config)?;

  let commands = scored
    .iter()
    .map(|suggestion| suggestion.command.clone())
    .collect::<Vec<_>>();
  let model_scores = online_score_commands(
    conn,
    OnlineScoreContext {
      shellname: shellname.as_str(),
      workspace_root,
      cwd: config.cwd.as_deref(),
      hostname: config.hostname.as_deref(),
      username: config.username.as_deref(),
      phase: None,
      exit_status: last_exit_status,
      session_id: config.session_id,
      unix_timestamp: runtime.now,
      recent_commands: &recent_commands,
      window: online_cfg.window,
    },
    &commands,
    &online_cfg,
  )
  .await?;

  if model_scores.len() == scored.len() {
    let gate = online_model_gate(&model_scores, online_cfg.blend).unwrap_or(0.0);
    let model_scale = online_cfg.blend.alpha * gate;
    if model_scale > 0.0 {
      for (idx, suggestion) in scored.iter_mut().enumerate() {
        let model = model_scores[idx];
        if model.is_finite() {
          let model_feature = (model as f64).tanh();
          let contrib = model_feature * model_scale;
          suggestion.score += contrib;
          suggestion.breakdown.online_model = contrib;
        }
      }
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::config::OnlineModelBlendMode;
  use crate::db::{OnlineFeedbackEvent, init, insert_invocation, open_db};
  use crate::shell_history::Invocation;

  use super::aliases::{add_alias_candidates, alias_for_command};
  use super::candidates::{add_head_candidates, add_phase_candidates};
  use super::phase_support::{PhaseSignal, detect_session_phase, phase_match_boost};

  async fn insert_command(conn: &Connection, command: &str, expanded: &str, ts: i64) -> Result<()> {
    let inv = Invocation {
      command: command.to_string(),
      expanded_command: expanded.to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/tmp".to_string()),
      workspace: None,
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(ts),
      end_unix_timestamp: Some(ts + 1),
      session_id: 1,
    };
    insert_invocation(conn, &inv).await?;
    Ok(())
  }

  #[test]
  fn detects_session_phase_from_phase_stats() {
    let recent = vec!["cargo build".to_string(), "pytest".to_string()];
    let mut map = HashMap::new();
    map.insert(
      "cargo build".to_string(),
      PhaseSignal {
        phase: "build".to_string(),
        confidence: 0.9,
      },
    );
    map.insert(
      "pytest".to_string(),
      PhaseSignal {
        phase: "test".to_string(),
        confidence: 0.8,
      },
    );
    let phase = detect_session_phase(&recent, &map).unwrap();
    assert_eq!(phase.phase, "test");
  }

  #[test]
  fn phase_boost_requires_match() {
    let session = PhaseSignal {
      phase: "test".to_string(),
      confidence: 0.9,
    };
    let candidate = PhaseSignal {
      phase: "test".to_string(),
      confidence: 0.8,
    };
    let boost = phase_match_boost(Some(&session), Some(&candidate));
    assert!(boost > 0.0);
    let other = PhaseSignal {
      phase: "build".to_string(),
      confidence: 0.9,
    };
    assert_eq!(phase_match_boost(Some(&session), Some(&other)), 0.0);
  }

  #[test]
  fn online_model_gate_blocks_ambiguous_scores() {
    let model_scores = vec![0.10, 0.11];
    let gate = online_model_gate(
      &model_scores,
      OnlineModelBlendConfig {
        mode: OnlineModelBlendMode::Additive,
        alpha: 5.0,
        margin_gate: 0.05,
        min_score_gate: 0.0,
      },
    )
    .unwrap_or(0.0);
    assert!(gate < 1.0);
  }

  #[test]
  fn online_model_gate_allows_confident_scores() {
    let model_scores = vec![-0.5, 0.5];
    let gate = online_model_gate(
      &model_scores,
      OnlineModelBlendConfig {
        mode: OnlineModelBlendMode::Additive,
        alpha: 5.0,
        margin_gate: 0.05,
        min_score_gate: 0.0,
      },
    )
    .unwrap_or(0.0);
    assert_eq!(gate, 1.0);
  }

  #[tokio::test]
  async fn blend_weights_update_increases_frecency_weight() -> Result<()> {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen) VALUES (?, ?, ?)",
        libsql::params!["git status".to_string(), 10i64, 3600i64],
      )
      .await?;

    let before = load_blend_weights(&db.conn).await?;
    let event = OnlineFeedbackEvent {
      shown_id: "fb-1".to_string(),
      shown_at: 32400,
      cwd: Some("/tmp".to_string()),
      suggestion: "git status".to_string(),
      accepted_command: Some("git status".to_string()),
      accepted_at: Some(32400),
      outcome: Some("accepted".to_string()),
    };
    update_blend_weights_for_feedback(&db.conn, &event).await?;
    let after = load_blend_weights(&db.conn).await?;
    assert!(after.frecency > before.frecency);
    Ok(())
  }

  #[tokio::test]
  async fn feedback_increases_frecency_score_for_accepted() -> Result<()> {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen) VALUES (?, ?, ?)",
        libsql::params!["A".to_string(), 1000i64, 100i64],
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen) VALUES (?, ?, ?)",
        libsql::params!["B".to_string(), 1i64, 100i64],
      )
      .await?;
    db.conn
      .execute(
        "INSERT INTO context_stats (command, working_directory, hostname, username, freq, last_seen)
         VALUES (?, ?, NULL, NULL, ?, ?)",
        libsql::params!["B".to_string(), "/tmp".to_string(), 100i64, 100i64],
      )
      .await?;

    let mut candidates = HashMap::new();
    let mut cand_a = Candidate::new("A");
    cand_a.freq = 1000;
    cand_a.last_seen = 100;
    let mut cand_b = Candidate::new("B");
    cand_b.freq = 1;
    cand_b.last_seen = 100;
    cand_b.context_freq = 100;
    cand_b.context_cwd_match = true;
    candidates.insert("A".to_string(), cand_a);
    candidates.insert("B".to_string(), cand_b);

    let prefix_norm: Vec<String> = Vec::new();
    let sequence_commands: Vec<String> = Vec::new();
    let recent_heads: Vec<String> = Vec::new();
    let weights = RankingWeights::default();
    let ctx = ScoreContext {
      conn: &db.conn,
      candidates: &candidates,
      prefix_norm: &prefix_norm,
      shellname: "zsh",
      sequence_commands: &sequence_commands,
      cwd: Some("/tmp"),
      hostname: None,
      username: None,
      exit_status: None,
      session_phase: None,
      session_id: None,
      recent_heads: &recent_heads,
      weights: &weights,
      now: 200,
      recency_half_life: DEFAULT_RECENCY_HALF_LIFE_SECONDS,
      phase_config: None,
      workspace_root: "/tmp",
    };

    let before = score_candidates(&ctx).await?;
    let before_score = before
      .iter()
      .find(|s| s.command == "A")
      .map(|s| s.score)
      .unwrap_or(0.0);

    let event = OnlineFeedbackEvent {
      shown_id: "fb-2".to_string(),
      shown_at: 200,
      cwd: Some("/tmp".to_string()),
      suggestion: "A".to_string(),
      accepted_command: Some("A".to_string()),
      accepted_at: Some(200),
      outcome: Some("accepted".to_string()),
    };
    for _ in 0..120 {
      update_blend_weights_for_feedback(&db.conn, &event).await?;
    }

    let after = score_candidates(&ctx).await?;
    let after_score = after
      .iter()
      .find(|s| s.command == "A")
      .map(|s| s.score)
      .unwrap_or(0.0);
    assert!(after_score > before_score);
    Ok(())
  }

  #[test]
  fn alias_for_command_matches_exact() {
    let alias = alias_for_command("gst", "git status", "git status");
    assert_eq!(alias.as_deref(), Some("gst"));
  }

  #[test]
  fn alias_for_command_matches_prefix() {
    let alias = alias_for_command("gst", "git status", "git status -sb");
    assert_eq!(alias.as_deref(), Some("gst -sb"));
  }

  #[test]
  fn alias_for_command_rejects_mismatch() {
    let alias = alias_for_command("gst", "git status", "git diff");
    assert!(alias.is_none());
  }

  #[tokio::test]
  async fn apply_alias_preference_prefers_alias_when_more_frequent() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    insert_command(&db.conn, "gst", "git status", 1)
      .await
      .unwrap();
    insert_command(&db.conn, "gst", "git status", 3)
      .await
      .unwrap();
    insert_command(&db.conn, "gst", "git status", 5)
      .await
      .unwrap();
    insert_command(&db.conn, "git status", "git status", 7)
      .await
      .unwrap();

    let mut suggestions = vec![Suggestion {
      command: "git status".to_string(),
      score: 1.0,
      breakdown: ScoreBreakdown::default(),
    }];
    let mut aliases = HashMap::new();
    aliases.insert("gst".to_string(), "git status".to_string());

    apply_alias_preference(&db.conn, &mut suggestions, &aliases)
      .await
      .unwrap();

    assert_eq!(suggestions[0].command, "gst");
  }

  #[tokio::test]
  async fn apply_alias_preference_keeps_expanded_when_more_frequent() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    insert_command(&db.conn, "gst", "git status", 1)
      .await
      .unwrap();
    insert_command(&db.conn, "git status", "git status", 3)
      .await
      .unwrap();
    insert_command(&db.conn, "git status", "git status", 5)
      .await
      .unwrap();

    let mut suggestions = vec![Suggestion {
      command: "git status".to_string(),
      score: 1.0,
      breakdown: ScoreBreakdown::default(),
    }];
    let mut aliases = HashMap::new();
    aliases.insert("gst".to_string(), "git status".to_string());

    apply_alias_preference(&db.conn, &mut suggestions, &aliases)
      .await
      .unwrap();

    assert_eq!(suggestions[0].command, "git status");
  }

  #[test]
  fn add_alias_candidates_clones_stats() {
    let mut candidates = HashMap::new();
    let mut base = Candidate::new("git status");
    base.freq = 4;
    base.last_seen = 123;
    candidates.insert(base.command.clone(), base.clone());

    let mut aliases = HashMap::new();
    aliases.insert("gst".to_string(), "git status".to_string());
    add_alias_candidates(&aliases, &mut candidates);

    let alias_candidate = candidates.get("gst").expect("alias candidate");
    assert_eq!(alias_candidate.freq, 4);
    assert_eq!(alias_candidate.last_seen, 123);
  }

  #[tokio::test]
  async fn embedding_retrieval_adds_candidates_when_tier1_omits() -> Result<()> {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    let online_config = OnlineModelConfig::default();
    crate::embeddings::ensure_command_embeddings_schema(&db.conn, online_config.embedding_dim)
      .await
      .unwrap();

    db.conn
      .execute(
        "INSERT INTO online_group_scalar (group_name, value, updated_at)
         VALUES (?, ?, ?)",
        ("cwd", 1.0f64, 1i64),
      )
      .await
      .unwrap();

    let target = "target-command".to_string();
    let mut commands = vec![target.clone()];
    for idx in 0..59 {
      commands.push(format!("cmd-{idx}"));
    }

    for (idx, command) in commands.iter().enumerate() {
      let freq = if command == &target {
        1i64
      } else {
        1_000i64 - idx as i64
      };
      db.conn
        .execute(
          "INSERT INTO command_stats (command, freq, last_seen, embedding)
           VALUES (?, ?, ?, ?)",
          libsql::params![
            command.clone(),
            freq,
            100i64 + idx as i64,
            Option::<Vec<u8>>::None
          ],
        )
        .await
        .unwrap();
    }

    insert_command(&db.conn, "ls", "ls", 10).await?;
    insert_command(&db.conn, "echo hi", "echo hi", 20).await?;

    crate::embeddings::index_command_embeddings(&db.conn, None)
      .await
      .unwrap();

    let runtime = SuggestRuntime {
      aliases: HashMap::new(),
      weights: RankingWeights::default(),
      recency_half_life: ranking::DEFAULT_RECENCY_HALF_LIFE_SECONDS,
      now: 200,
      phase_config: None,
    };
    let config = SuggestConfig {
      max_results: 100,
      recent_limit: 5,
      prefix: None,
      cwd: Some("/tmp".to_string()),
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
      session_id: Some(1),
      shellname: Some("zsh".to_string()),
      use_sequences: false,
      prefer_full_line: true,
    };

    let suggestions = suggest_with_runtime(&db.conn, config, &runtime, None).await?;
    let suggestion = suggestions
      .iter()
      .find(|suggestion| suggestion.command == target)
      .expect("expected embedding candidate");
    assert!(suggestion.breakdown.embedding_retrieval > 0.0);
    Ok(())
  }

  #[tokio::test]
  async fn completion_suggests_double_dash_flags() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    db.conn
      .execute(
        "INSERT INTO flag_stats (workspace_root, command_head, flag_raw, flag_norm, freq, last_seen)
         VALUES (?, ?, ?, ?, ?, ?)",
        ("", "cargo install", "--locked", "--locked", 5i64, 10i64),
      )
      .await
      .unwrap();

    let cwd = tempfile::tempdir().unwrap();
    let runtime = SuggestRuntime {
      aliases: HashMap::new(),
      weights: RankingWeights::default(),
      recency_half_life: ranking::DEFAULT_RECENCY_HALF_LIFE_SECONDS,
      now: 1_000,
      phase_config: None,
    };
    let prefix = "cargo install --path . --force --";
    let config = SuggestConfig {
      max_results: 5,
      recent_limit: 10,
      prefix: Some(prefix.to_string()),
      cwd: Some(cwd.path().to_string_lossy().into_owned()),
      hostname: None,
      username: None,
      session_id: None,
      shellname: Some("zsh".to_string()),
      use_sequences: false,
      prefer_full_line: true,
    };

    let suggestions = suggest_with_runtime(&db.conn, config, &runtime, None)
      .await
      .unwrap();

    assert!(
      suggestions
        .iter()
        .any(|s| s.command == format!("{prefix}locked"))
    );
    assert!(suggestions.iter().all(|s| s.command.starts_with(prefix)));
  }

  #[tokio::test]
  async fn completion_scopes_flags_to_subcommand_head() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    db.conn
      .execute(
        "INSERT INTO flag_stats (workspace_root, command_head, flag_raw, flag_norm, freq, last_seen)
         VALUES (?, ?, ?, ?, ?, ?)",
        ("", "zage import", "--shell", "--shell", 5i64, 10i64),
      )
      .await
      .unwrap();
    db.conn
      .execute(
        "INSERT INTO flag_stats (workspace_root, command_head, flag_raw, flag_norm, freq, last_seen)
         VALUES (?, ?, ?, ?, ?, ?)",
        ("", "zage suggest", "--current-line", "--current-line", 5i64, 10i64),
      )
      .await
      .unwrap();

    let cwd = tempfile::tempdir().unwrap();
    let runtime = SuggestRuntime {
      aliases: HashMap::new(),
      weights: RankingWeights::default(),
      recency_half_life: ranking::DEFAULT_RECENCY_HALF_LIFE_SECONDS,
      now: 1_000,
      phase_config: None,
    };
    let prefix = "zage import --";
    let config = SuggestConfig {
      max_results: 10,
      recent_limit: 10,
      prefix: Some(prefix.to_string()),
      cwd: Some(cwd.path().to_string_lossy().into_owned()),
      hostname: None,
      username: None,
      session_id: None,
      shellname: Some("zsh".to_string()),
      use_sequences: false,
      prefer_full_line: true,
    };

    let suggestions = suggest_with_runtime(&db.conn, config, &runtime, None)
      .await
      .unwrap();

    assert!(
      suggestions
        .iter()
        .any(|s| s.command == format!("{prefix}shell"))
    );
    assert!(
      !suggestions
        .iter()
        .any(|s| s.command == format!("{prefix}current-line"))
    );
    assert!(suggestions.iter().all(|s| s.command.starts_with(prefix)));
  }

  #[tokio::test]
  async fn adds_head_candidates_from_stats() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen) VALUES (?, ?, ?)",
        ("git status", 5, 10),
      )
      .await
      .unwrap();
    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen) VALUES (?, ?, ?)",
        ("git diff", 3, 8),
      )
      .await
      .unwrap();
    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen) VALUES (?, ?, ?)",
        ("cargo build", 4, 9),
      )
      .await
      .unwrap();

    let mut candidates = HashMap::new();
    add_head_candidates(&db.conn, &["git".to_string()], "", &mut candidates)
      .await
      .unwrap();

    assert!(candidates.contains_key("git status"));
    assert!(candidates.contains_key("git diff"));
    assert!(!candidates.contains_key("cargo build"));
  }

  #[tokio::test]
  async fn adds_phase_candidates_from_stats() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    db.conn
      .execute(
        "INSERT INTO command_stats (command, freq, last_seen) VALUES (?, ?, ?)",
        ("git status", 5, 10),
      )
      .await
      .unwrap();
    db.conn
      .execute(
        "INSERT INTO phase_stats (command_head, phase, confidence, freq, last_seen)
         VALUES (?, ?, ?, ?, ?)",
        ("git", "build", 0.9, 5, 10),
      )
      .await
      .unwrap();

    let mut candidates = HashMap::new();
    let phase = PhaseSignal {
      phase: "build".to_string(),
      confidence: 0.9,
    };
    add_phase_candidates(&db.conn, Some(&phase), "", &mut candidates)
      .await
      .unwrap();

    assert!(candidates.contains_key("git status"));
  }
}
