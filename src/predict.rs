use std::collections::{HashMap, HashSet};

use libsql::{Connection, Value};

use crate::Result;
use crate::core::{Candidate, SystemTimeProvider, TimeProvider};
pub use crate::core::{ScoreBreakdown, Suggestion};
use crate::db::get_recent_invocations;
use crate::phase::PhaseConfig;
use crate::repo::find_repo_root;
use crate::rerank;
use crate::shell_history::detect_shellname;
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

use crate::rerank_config::RerankConfig;
use aliases::{
  add_alias_candidates, alias_for_command, build_prefix_variants, expand_alias, load_aliases,
};
use candidates::{
  add_context_candidates, add_global_candidates, add_head_candidates, add_phase_candidates,
  add_recent_candidates, add_repo_candidates, add_sequence_candidates, add_session_candidates,
  add_template_candidates, add_transition_candidates, hydrate_candidate_stats, load_session_stats,
  push_opt_i64, push_opt_string,
};
use phase_support::{
  PhaseSignal, command_head_for_phase, detect_session_phase, detect_session_phase_from_commands,
  load_phase_for_heads, phase_match_boost,
};
use ranking::{
  DEFAULT_RECENCY_HALF_LIFE_SECONDS, load_normalized_tokens, low_confidence, recency_score,
  token_similarity,
};
use runtime::SuggestRuntime;
use sql::query_prepared;
use templates::{
  arg_template_candidates, env_template_candidates, split_env_prefix, token_sequence_predictions,
};

const GLOBAL_CANDIDATE_LIMIT: usize = 50;
const GLOBAL_CANDIDATE_LIMIT_FALLBACK: usize = 200;
const RECENT_CANDIDATE_LIMIT: usize = 200;
const RECENT_CANDIDATE_LIMIT_FALLBACK: usize = 500;
const FULL_LINE_POOL_LIMIT: usize = 50;

#[cfg(test)]
pub(crate) fn candidate_for_test(command: &str) -> Candidate {
  Candidate::new(command)
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

pub async fn suggest(conn: &Connection, config: SuggestConfig) -> Result<Vec<Suggestion>> {
  let time_provider = SystemTimeProvider;
  let runtime = SuggestRuntime {
    aliases: load_aliases(),
    weights: RankingWeights::default(),
    recency_half_life: DEFAULT_RECENCY_HALF_LIFE_SECONDS,
    now: time_provider.now(),
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
  let recent = get_recent_invocations(conn, config.recent_limit).await?;
  if recent.is_empty() {
    return Ok(Vec::new());
  }

  let recent_commands: Vec<String> = recent
    .iter()
    .map(|inv| expanded_command_for(inv, aliases))
    .collect();
  let mut sequence_commands = recent_commands.clone();
  if let Some((cmd, _)) = override_prev.as_ref() {
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
  let session_tokens = recent_commands
    .iter()
    .flat_map(|cmd| normalized_tokens(cmd))
    .collect::<Vec<_>>();
  let last_command = override_prev
    .as_ref()
    .map(|(cmd, _)| cmd.clone())
    .or_else(|| recent_commands.last().cloned());
  let last_exit_status = override_prev
    .as_ref()
    .map(|(_, exit)| *exit)
    .unwrap_or_else(|| recent.last().and_then(|inv| inv.exit_status));
  let repo_root = config
    .cwd
    .as_deref()
    .and_then(find_repo_root)
    .unwrap_or_default();

  let mut candidates: HashMap<String, Candidate> = HashMap::new();

  if let Some(last) = &last_command {
    add_transition_candidates(conn, last, last_exit_status, &repo_root, &mut candidates).await?;
  }

  if let Some(session_id) = config.session_id {
    add_session_candidates(conn, session_id, &mut candidates).await?;
  }

  let recent_head_set: HashSet<String> = recent_heads.iter().cloned().collect();
  let phase_for_recent = load_phase_for_heads(conn, &recent_head_set).await?;
  let phase_config = PhaseConfig::load()?;
  let phase_config = if phase_config.labels().len() > 1 {
    Some(phase_config)
  } else {
    None
  };
  let session_phase = phase_config
    .as_ref()
    .and_then(|config| detect_session_phase_from_commands(&recent_commands, config))
    .or_else(|| detect_session_phase(&recent_heads, &phase_for_recent));
  add_phase_candidates(conn, session_phase.as_ref(), &repo_root, &mut candidates).await?;

  add_context_candidates(conn, &config, &mut candidates).await?;

  if !repo_root.is_empty() {
    add_repo_candidates(conn, &repo_root, &mut candidates).await?;
  }

  if !recent_heads.is_empty() {
    add_head_candidates(conn, &recent_heads, &repo_root, &mut candidates).await?;
  }

  if config.use_sequences {
    add_sequence_candidates(conn, &sequence_commands, &mut candidates).await?;
  }

  if !candidates.is_empty() {
    add_template_candidates(conn, &repo_root, &mut candidates).await?;
  }

  if candidates.is_empty() {
    add_global_candidates(conn, &mut candidates, GLOBAL_CANDIDATE_LIMIT).await?;
    add_template_candidates(conn, &repo_root, &mut candidates).await?;
  }

  if candidates.len() < 25 {
    add_recent_candidates(conn, &mut candidates, RECENT_CANDIDATE_LIMIT).await?;
    add_global_candidates(conn, &mut candidates, GLOBAL_CANDIDATE_LIMIT).await?;
    add_template_candidates(conn, &repo_root, &mut candidates).await?;
  }

  let session_stats = if let Some(session_id) = config.session_id {
    load_session_stats(conn, session_id, config.recent_limit).await?
  } else {
    HashMap::new()
  };

  if !session_stats.is_empty() {
    for (cmd, (freq, last_seen)) in session_stats {
      let entry = candidates
        .entry(cmd.clone())
        .or_insert_with(|| Candidate::new(&cmd));
      entry.session_freq = entry.session_freq.max(freq);
      entry.session_last_seen = entry.session_last_seen.max(last_seen);
      entry.last_seen = entry.last_seen.max(last_seen);
    }
  }

  if !candidates.is_empty() {
    hydrate_candidate_stats(conn, &repo_root, &mut candidates).await?;
  }

  if !aliases.is_empty() {
    add_alias_candidates(aliases, &mut candidates);
  }

  let mut scored = score_candidates(&ScoreContext {
    conn,
    candidates: &candidates,
    prefix_norm: &prefix_norm,
    session_phase: session_phase.as_ref(),
    recent_heads: &recent_heads,
    weights: &runtime.weights,
    now: runtime.now,
    recency_half_life: runtime.recency_half_life,
    phase_config: phase_config.as_ref(),
  })
  .await?;

  let rerank_config = RerankConfig::load()?;
  if low_confidence(&scored, &rerank_config) {
    let before = candidates.len();
    expand_low_confidence_candidates(
      conn,
      &repo_root,
      &recent_heads,
      &recent_commands,
      config.use_sequences,
      &mut candidates,
    )
    .await?;
    if candidates.len() > before {
      scored = score_candidates(&ScoreContext {
        conn,
        candidates: &candidates,
        prefix_norm: &prefix_norm,
        session_phase: session_phase.as_ref(),
        recent_heads: &recent_heads,
        weights: &runtime.weights,
        now: runtime.now,
        recency_half_life: runtime.recency_half_life,
        phase_config: phase_config.as_ref(),
      })
      .await?;
    }
  }

  let shellname = detect_shellname();
  let context = rerank::runtime_context(
    &repo_root,
    &recent_heads,
    session_tokens,
    session_phase.as_ref().map(|phase| phase.phase.as_str()),
    &shellname,
  );
  let _ = rerank::rerank_suggestions(&mut scored, &candidates, &context, &rerank_config);

  let transition_only = runtime.weights.transition > 0.0
    && runtime.weights.recency.abs() <= f64::EPSILON
    && runtime.weights.frequency.abs() <= f64::EPSILON
    && runtime.weights.context.abs() <= f64::EPSILON
    && runtime.weights.sequence.abs() <= f64::EPSILON
    && runtime.weights.similarity.abs() <= f64::EPSILON;
  if transition_only && last_command.is_some() {
    let has_transition = scored.iter().any(|s| s.breakdown.transition > 0.0);
    if has_transition {
      scored.retain(|s| s.breakdown.transition > 0.0);
    }
  }

  scored.truncate(config.max_results);
  Ok(scored)
}

struct ScoreContext<'a> {
  conn: &'a Connection,
  candidates: &'a HashMap<String, Candidate>,
  prefix_norm: &'a [String],
  session_phase: Option<&'a PhaseSignal>,
  recent_heads: &'a [String],
  weights: &'a RankingWeights,
  now: i64,
  recency_half_life: f64,
  phase_config: Option<&'a PhaseConfig>,
}

async fn score_candidates(context: &ScoreContext<'_>) -> Result<Vec<Suggestion>> {
  let mut candidate_heads: HashMap<String, String> = HashMap::new();
  let mut phase_heads: HashSet<String> = context.recent_heads.iter().cloned().collect();
  for candidate in context.candidates.values() {
    if let Some(head) = command_head_for_phase("sh", &candidate.command) {
      phase_heads.insert(head.clone());
      candidate_heads.insert(candidate.command.clone(), head);
    }
  }
  let phase_for_head = load_phase_for_heads(context.conn, &phase_heads).await?;

  let mut scored: Vec<Suggestion> = Vec::new();
  for candidate in context.candidates.values() {
    let recency = recency_score(context.now, candidate.last_seen, context.recency_half_life);
    let frequency = (candidate.freq as f64).ln_1p() + 0.5 * (candidate.repo_freq as f64).ln_1p();
    let transition = (candidate.transition_freq as f64).ln_1p()
      + 0.7 * (candidate.repo_transition_freq as f64).ln_1p();
    let mut context_score =
      (candidate.context_freq as f64).ln_1p() + 0.8 * (candidate.session_freq as f64).ln_1p();
    let pattern_phase = context.phase_config.and_then(|config| {
      config
        .match_label(&candidate.command)
        .and_then(|idx| config.labels().get(idx).cloned())
        .map(|phase| PhaseSignal {
          phase,
          confidence: 1.0,
        })
    });
    let candidate_phase = pattern_phase.as_ref().or_else(|| {
      candidate_heads
        .get(&candidate.command)
        .and_then(|head| phase_for_head.get(head))
    });
    context_score += phase_match_boost(context.session_phase, candidate_phase);
    let sequence = if candidate.sequence_confidence > 0.0 {
      let order_weight = if candidate.sequence_prefix_len >= 2 {
        1.0
      } else {
        0.7
      };
      candidate.sequence_confidence * candidate.sequence_lift.max(1.0) * order_weight
    } else {
      0.0
    };
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

    if score <= 0.0 {
      continue;
    }

    scored.push(Suggestion {
      command: candidate.command.clone(),
      score,
      breakdown: ScoreBreakdown {
        recency,
        frequency,
        transition,
        context: context_score,
        sequence,
        similarity,
      },
    });
  }

  scored.sort_by(|a, b| {
    let diff = (b.score - a.score).abs();
    if diff < 1e-4 {
      let recency_cmp = a
        .breakdown
        .recency
        .partial_cmp(&b.breakdown.recency)
        .unwrap_or(std::cmp::Ordering::Equal);
      if recency_cmp != std::cmp::Ordering::Equal {
        return recency_cmp;
      }
      return a.command.cmp(&b.command);
    }
    b.score
      .partial_cmp(&a.score)
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  Ok(scored)
}

async fn expand_low_confidence_candidates(
  conn: &Connection,
  repo_root: &str,
  recent_heads: &[String],
  recent_commands: &[String],
  use_sequences: bool,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  add_recent_candidates(conn, candidates, RECENT_CANDIDATE_LIMIT_FALLBACK).await?;
  add_global_candidates(conn, candidates, GLOBAL_CANDIDATE_LIMIT_FALLBACK).await?;
  if !repo_root.is_empty() {
    add_repo_candidates(conn, repo_root, candidates).await?;
  }
  if !recent_heads.is_empty() {
    add_head_candidates(conn, recent_heads, repo_root, candidates).await?;
  }
  if use_sequences {
    add_sequence_candidates(conn, recent_commands, candidates).await?;
  }
  add_template_candidates(conn, repo_root, candidates).await?;
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
  let project_root = config
    .cwd
    .as_deref()
    .and_then(find_repo_root)
    .map(|root| root.trim_end_matches('/').to_string());
  let project_like = project_root.as_ref().map(|root| format!("{root}/%"));

  let repo_root = config
    .cwd
    .as_deref()
    .and_then(find_repo_root)
    .unwrap_or_default();
  let prefer_full_line = config.prefer_full_line;
  let pool_limit = if prefer_full_line {
    config.max_results.max(FULL_LINE_POOL_LIMIT)
  } else {
    config.max_results
  };
  let token_priors = token_sequence_predictions(conn, prefix_norm).await?;
  let (prefix_flags, prefix_args) = {
    let tokens = tokenize_index("sh", prefix);
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
    &repo_root,
    &token_priors,
    runtime.now,
    runtime.recency_half_life,
  )
  .await?
  {
    suggestions.sort_by(|a, b| {
      b.score
        .partial_cmp(&a.score)
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    suggestions.truncate(pool_limit);
    env_suggestions = Some(suggestions);
  }

  let mut arg_suggestions_for_merge = None;
  if let Some(mut arg_suggestions) = arg_template_candidates(
    conn,
    prefix,
    &repo_root,
    &token_priors,
    runtime.now,
    runtime.recency_half_life,
  )
  .await?
  {
    let has_prefix_match = arg_suggestions
      .iter()
      .any(|suggestion| suggestion.command.starts_with(prefix));
    if !has_prefix_match {
      // fall through to normal completion candidates
    } else {
      let trimmed_prefix = prefix.trim_end();
      arg_suggestions.retain(|suggestion| {
        if suggestion.command.trim_end() == trimmed_prefix {
          return false;
        }
        if prefix_flags.is_empty() && prefix_args.is_empty() {
          return true;
        }
        let tokens = tokenize_index("sh", &suggestion.command);
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
        arg_suggestions.sort_by(|a, b| {
          b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
        });
        arg_suggestions.truncate(pool_limit);
        let last_char = prefix.chars().last();
        let ends_with_space = last_char.map(|c| c.is_whitespace()).unwrap_or(false);
        let ends_with_quote = matches!(last_char, Some('"') | Some('\''));
        if prefer_full_line || (ends_with_space && !prefix_flags.is_empty()) || ends_with_quote {
          arg_suggestions_for_merge = Some(arg_suggestions);
        } else {
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
  let match_prefixes = build_prefix_variants(&match_prefix, aliases);
  if match_prefixes.is_empty() {
    return Ok(env_suggestions.unwrap_or_default());
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
    let prefix_score = if command.starts_with(prefix) {
      1.0
    } else if expanded_for_score.starts_with(prefix) {
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
        && alias_command.starts_with(&match_prefix)
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
        frequency,
        transition: 0.0,
        context,
        sequence: 0.0,
        similarity,
      },
    });
  }

  if scored.is_empty() {
    if prefix_flags.is_empty() && prefix_args.is_empty() {
      let mut merged = env_suggestions.unwrap_or_default();
      if let Some(mut arg_suggestions) = arg_suggestions_for_merge.take() {
        merged.append(&mut arg_suggestions);
      }
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
        let tokens = tokenize_index("sh", &command);
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
        let tokens = tokenize_index("sh", &command);
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
          && alias_command.starts_with(&match_prefix)
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
          frequency,
          transition: 0.0,
          context: 0.0,
          sequence: 0.0,
          similarity,
        },
      });
    }
  }

  if prefer_full_line && !scored.is_empty() {
    scored.sort_by(|a, b| {
      b.score
        .partial_cmp(&a.score)
        .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(config.max_results);
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
        }
      }
    }
    scored.extend(suggestions);
  }

  if let Some(mut suggestions) = arg_suggestions_for_merge {
    scored.append(&mut suggestions);
  }

  scored.sort_by(|a, b| {
    b.score
      .partial_cmp(&a.score)
      .unwrap_or(std::cmp::Ordering::Equal)
  });
  scored.truncate(config.max_results);
  Ok(scored)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::db::{init, open_db};

  use super::aliases::{add_alias_candidates, alias_for_command};
  use super::candidates::{add_head_candidates, add_phase_candidates};
  use super::phase_support::{PhaseSignal, detect_session_phase, phase_match_boost};

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
