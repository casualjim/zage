use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use libsql::{Connection, Value};

use crate::Result;
use crate::db::get_recent_invocations;
use crate::repo::find_repo_root;
use crate::tokenize::normalized_tokens;

mod aliases;
mod candidates;
mod phase_support;
mod ranking;
mod sql;
mod templates;

use aliases::{add_alias_candidates, build_prefix_variants, expand_alias, load_aliases};
use candidates::{
  add_context_candidates, add_global_candidates, add_head_candidates, add_phase_candidates,
  add_recent_candidates, add_repo_candidates, add_sequence_candidates, add_session_candidates,
  add_template_candidates, add_transition_candidates, load_session_stats, push_opt_i64,
  push_opt_string,
};
use phase_support::{
  PhaseSignal, command_head_for_phase, detect_session_phase, load_phase_for_heads,
  phase_match_boost,
};
use ranking::{load_normalized_tokens, low_confidence, recency_score, token_similarity};
use sql::query_prepared;
use templates::{
  arg_template_candidates, env_template_candidates, split_env_prefix, token_sequence_predictions,
};

const GLOBAL_CANDIDATE_LIMIT: usize = 50;
const GLOBAL_CANDIDATE_LIMIT_FALLBACK: usize = 200;
const RECENT_CANDIDATE_LIMIT: usize = 200;
const RECENT_CANDIDATE_LIMIT_FALLBACK: usize = 500;

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
    }
  }
}

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

#[derive(Debug, Clone)]
struct Candidate {
  command: String,
  freq: i64,
  last_seen: i64,
  transition_freq: i64,
  repo_transition_freq: i64,
  repo_freq: i64,
  context_freq: i64,
  session_freq: i64,
  session_last_seen: i64,
  sequence_confidence: f64,
  sequence_lift: f64,
  sequence_prefix_len: usize,
}

#[derive(Debug, Clone)]
struct RankingWeights {
  recency: f64,
  frequency: f64,
  transition: f64,
  context: f64,
  sequence: f64,
  similarity: f64,
}

impl Default for RankingWeights {
  fn default() -> Self {
    Self {
      recency: 0.25,
      frequency: 0.25,
      transition: 0.2,
      context: 0.15,
      sequence: 0.1,
      similarity: 0.05,
    }
  }
}

fn new_candidate(command: &str) -> Candidate {
  Candidate {
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

pub async fn suggest(conn: &Connection, config: SuggestConfig) -> Result<Vec<Suggestion>> {
  let prefix = config.prefix.clone().unwrap_or_default();
  let prefix_norm = normalized_tokens(&prefix);
  let has_prefix = !prefix.is_empty();

  if has_prefix {
    return suggest_completions(conn, &config, &prefix, &prefix_norm).await;
  }

  let aliases = load_aliases();
  let recent = get_recent_invocations(conn, config.recent_limit).await?;
  if recent.is_empty() {
    return Ok(Vec::new());
  }

  let recent_commands: Vec<String> = recent.iter().map(|inv| inv.command.clone()).collect();
  let recent_heads: Vec<String> = recent
    .iter()
    .filter_map(|inv| command_head_for_phase(&inv.shellname, &inv.command))
    .collect();
  let last_command = recent_commands.last().cloned();
  let last_exit_status = recent.last().and_then(|inv| inv.exit_status);
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
  let session_phase = detect_session_phase(&recent_heads, &phase_for_recent);
  add_phase_candidates(conn, session_phase.as_ref(), &repo_root, &mut candidates).await?;

  add_context_candidates(conn, &config, &mut candidates).await?;

  if !repo_root.is_empty() {
    add_repo_candidates(conn, &repo_root, &mut candidates).await?;
  }

  if !recent_heads.is_empty() {
    add_head_candidates(conn, &recent_heads, &repo_root, &mut candidates).await?;
  }

  if config.use_sequences {
    add_sequence_candidates(conn, &recent_commands, &mut candidates).await?;
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

  if !aliases.is_empty() {
    add_alias_candidates(&aliases, &mut candidates);
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
        .or_insert_with(|| new_candidate(&cmd));
      entry.session_freq = entry.session_freq.max(freq);
      entry.session_last_seen = entry.session_last_seen.max(last_seen);
      entry.last_seen = entry.last_seen.max(last_seen);
    }
  }

  let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64;

  let weights = RankingWeights::default();
  let mut scored = score_candidates(
    conn,
    &candidates,
    &prefix_norm,
    session_phase.as_ref(),
    &recent_heads,
    &weights,
    now,
  )
  .await?;

  if low_confidence(&scored) {
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
      scored = score_candidates(
        conn,
        &candidates,
        &prefix_norm,
        session_phase.as_ref(),
        &recent_heads,
        &weights,
        now,
      )
      .await?;
    }
  }

  scored.truncate(config.max_results);
  Ok(scored)
}

async fn score_candidates(
  conn: &Connection,
  candidates: &HashMap<String, Candidate>,
  prefix_norm: &[String],
  session_phase: Option<&PhaseSignal>,
  recent_heads: &[String],
  weights: &RankingWeights,
  now: i64,
) -> Result<Vec<Suggestion>> {
  let mut candidate_heads: HashMap<String, String> = HashMap::new();
  let mut phase_heads: HashSet<String> = recent_heads.iter().cloned().collect();
  for candidate in candidates.values() {
    if let Some(head) = command_head_for_phase("sh", &candidate.command) {
      phase_heads.insert(head.clone());
      candidate_heads.insert(candidate.command.clone(), head);
    }
  }
  let phase_for_head = load_phase_for_heads(conn, &phase_heads).await?;

  let mut scored: Vec<Suggestion> = Vec::new();
  for candidate in candidates.values() {
    let recency = recency_score(now, candidate.last_seen);
    let frequency = (candidate.freq as f64).ln_1p() + 0.5 * (candidate.repo_freq as f64).ln_1p();
    let transition = (candidate.transition_freq as f64).ln_1p()
      + 0.7 * (candidate.repo_transition_freq as f64).ln_1p();
    let mut context =
      (candidate.context_freq as f64).ln_1p() + 0.8 * (candidate.session_freq as f64).ln_1p();
    let candidate_phase = candidate_heads
      .get(&candidate.command)
      .and_then(|head| phase_for_head.get(head));
    context += phase_match_boost(session_phase, candidate_phase);
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
    let similarity = if prefix_norm.is_empty() {
      0.0
    } else {
      token_similarity(
        prefix_norm,
        &load_normalized_tokens(conn, &candidate.command).await?,
      )
    };
    let session_recency = if candidate.session_last_seen > 0 {
      recency_score(now, candidate.session_last_seen)
    } else {
      0.0
    };

    let score = weights.recency * recency
      + weights.frequency * frequency
      + weights.transition * transition
      + weights.context * context
      + weights.sequence * sequence
      + 0.1 * session_recency
      + weights.similarity * similarity;

    scored.push(Suggestion {
      command: candidate.command.clone(),
      score,
      breakdown: ScoreBreakdown {
        recency,
        frequency,
        transition,
        context,
        sequence,
        similarity,
      },
    });
  }

  scored.sort_by(|a, b| {
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
  prefix: &str,
  prefix_norm: &[String],
) -> Result<Vec<Suggestion>> {
  let aliases = load_aliases();
  if let Some(session_id) = config.session_id {
    let session_scored = completion_candidates(
      conn,
      config,
      prefix,
      prefix_norm,
      &aliases,
      Some(session_id),
    )
    .await?;
    if !session_scored.is_empty() {
      return Ok(session_scored);
    }
  }

  completion_candidates(conn, config, prefix, prefix_norm, &aliases, None).await
}

async fn completion_candidates(
  conn: &Connection,
  config: &SuggestConfig,
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
  let project_like = project_root.as_ref().map(|root| format!("{}/%", root));

  let (env_prefix, match_prefix) = split_env_prefix(prefix);
  let match_prefixes = build_prefix_variants(&match_prefix, aliases);
  if match_prefixes.is_empty() {
    return Ok(Vec::new());
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
  where_parts.push(format!("({})", like_clause));

  sql.push_str(&where_parts.join(" AND "));
  sql.push_str(" GROUP BY command ORDER BY last_seen DESC LIMIT 200");

  for prefix_value in match_prefixes {
    params.push(Value::from(format!("{prefix_value}%")));
  }

  let mut rows = query_prepared(conn, &sql, libsql::params_from_iter(params)).await?;

  let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64;

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
    let norm_tokens = normalized_tokens(expanded_for_score);
    let similarity = token_similarity(prefix_norm, &norm_tokens);
    let prefix_score = if command.starts_with(prefix) {
      1.0
    } else if expanded_for_score.starts_with(prefix) {
      0.8
    } else {
      0.0
    };

    let recency = recency_score(now, last_seen);
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

    let mut suggestion_command = command;
    if !env_prefix.is_empty() {
      let prefix = if env_prefix
        .chars()
        .last()
        .map(|c| c.is_whitespace())
        .unwrap_or(false)
      {
        env_prefix.clone()
      } else {
        format!("{env_prefix} ")
      };
      suggestion_command = format!("{}{}", prefix, suggestion_command);
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

  let repo_root = config
    .cwd
    .as_deref()
    .and_then(find_repo_root)
    .unwrap_or_default();
  let token_priors = token_sequence_predictions(conn, prefix_norm).await?;
  if let Some(mut env_suggestions) =
    env_template_candidates(conn, prefix, &repo_root, &token_priors).await?
  {
    let mut merged: HashMap<String, Suggestion> = HashMap::new();
    for suggestion in scored.into_iter().chain(env_suggestions.drain(..)) {
      match merged.get(&suggestion.command) {
        Some(existing) if existing.score >= suggestion.score => {}
        _ => {
          merged.insert(suggestion.command.clone(), suggestion);
        }
      }
    }
    scored = merged.into_values().collect();
  }
  if let Some(mut arg_suggestions) =
    arg_template_candidates(conn, prefix, &repo_root, &token_priors).await?
  {
    let mut merged: HashMap<String, Suggestion> = HashMap::new();
    for suggestion in scored.into_iter().chain(arg_suggestions.drain(..)) {
      match merged.get(&suggestion.command) {
        Some(existing) if existing.score >= suggestion.score => {}
        _ => {
          merged.insert(suggestion.command.clone(), suggestion);
        }
      }
    }
    scored = merged.into_values().collect();
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
    let mut base = new_candidate("git status");
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
