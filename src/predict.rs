use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, UNIX_EPOCH};

use libsql::{Connection, Value};
use serde_json;

use crate::Result;
use crate::db::get_recent_invocations;
use crate::repo::find_repo_root;
use crate::sequence::candidates_from_sequences;
use crate::tokenize::{extract_command_parts, normalized_tokens, token_strings, tokenize, Token, TokenKind};

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

  let recent = get_recent_invocations(conn, config.recent_limit).await?;
  if recent.is_empty() {
    return Ok(Vec::new());
  }

  let recent_commands: Vec<String> = recent.iter().map(|inv| inv.command.clone()).collect();
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

  add_context_candidates(conn, &config, &mut candidates).await?;

  if !repo_root.is_empty() {
    add_repo_candidates(conn, &repo_root, &mut candidates).await?;
  }

  if config.use_sequences {
    add_sequence_candidates(conn, &recent_commands, &mut candidates).await?;
  }

  if candidates.is_empty() {
    add_global_candidates(conn, &mut candidates).await?;
  }

  let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64;

  let weights = RankingWeights::default();
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
  let mut scored: Vec<Suggestion> = Vec::new();

  for candidate in candidates.values() {
    let recency = recency_score(now, candidate.last_seen);
    let frequency = (candidate.freq as f64).ln_1p()
      + 0.5 * (candidate.repo_freq as f64).ln_1p();
    let transition = (candidate.transition_freq as f64).ln_1p()
      + 0.7 * (candidate.repo_transition_freq as f64).ln_1p();
    let context = (candidate.context_freq as f64).ln_1p()
      + 0.8 * (candidate.session_freq as f64).ln_1p();
    let sequence = if candidate.sequence_confidence > 0.0 {
      let order_weight = if candidate.sequence_prefix_len >= 2 { 1.0 } else { 0.7 };
      candidate.sequence_confidence * candidate.sequence_lift.max(1.0) * order_weight
    } else {
      0.0
    };
    let similarity = token_similarity(
      &prefix_norm,
      &load_normalized_tokens(conn, &candidate.command).await?,
    );
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

  scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
  scored.truncate(config.max_results);
  Ok(scored)
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
  let project_like = project_root
    .as_ref()
    .map(|root| format!("{}/%", root));

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

  let mut rows = conn
    .query(&sql, libsql::params_from_iter(params))
    .await?;

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
      let prefix = if env_prefix.chars().last().map(|c| c.is_whitespace()).unwrap_or(false) {
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

  scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
  scored.truncate(config.max_results);
  Ok(scored)
}

async fn add_transition_candidates(
  conn: &Connection,
  last_command: &str,
  last_exit_status: Option<i64>,
  repo_root: &str,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  if !repo_root.is_empty() {
    let found = fetch_transition_candidates(
      conn,
      "repo_transition_stats_v2",
      Some(repo_root),
      last_command,
      last_exit_status,
      true,
      candidates,
    )
    .await?;
    if !found {
      let _ = fetch_transition_candidates(
        conn,
        "repo_transition_stats_v2",
        Some(repo_root),
        last_command,
        None,
        true,
        candidates,
      )
      .await?;
    }
  }

  let found = fetch_transition_candidates(
    conn,
    "transition_stats_v2",
    None,
    last_command,
    last_exit_status,
    false,
    candidates,
  )
  .await?;
  if !found {
    let _ = fetch_transition_candidates(
      conn,
      "transition_stats_v2",
      None,
      last_command,
      None,
      false,
      candidates,
    )
    .await?;
  }
  Ok(())
}

async fn fetch_transition_candidates(
  conn: &Connection,
  table: &str,
  repo_root: Option<&str>,
  last_command: &str,
  last_exit_status: Option<i64>,
  is_repo: bool,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<bool> {
  let mut sql = String::from("SELECT next_command, freq, last_seen FROM ");
  sql.push_str(table);
  sql.push_str(" WHERE ");
  let mut clauses: Vec<String> = Vec::new();
  let mut params: Vec<Value> = Vec::new();

  if let Some(root) = repo_root {
    clauses.push("repo_root = ?".to_string());
    params.push(Value::from(root.to_string()));
  }
  clauses.push("prev_command = ?".to_string());
  params.push(Value::from(last_command.to_string()));

  if let Some(status) = last_exit_status {
    clauses.push("prev_exit_status = ?".to_string());
    params.push(Value::from(status));
  } else {
    clauses.push("prev_exit_status IS NULL".to_string());
  }

  sql.push_str(&clauses.join(" AND "));
  sql.push_str(" ORDER BY freq DESC LIMIT 50");

  let mut rows = conn
    .query(&sql, libsql::params_from_iter(params))
    .await?;
  let mut found = false;
  while let Some(row) = rows.next().await? {
    let cmd = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let last_seen = row.get::<i64>(2)?;
    let entry = candidates
      .entry(cmd.clone())
      .or_insert_with(|| new_candidate(&cmd));
    if is_repo {
      entry.repo_transition_freq = entry.repo_transition_freq.max(freq);
    } else {
      entry.transition_freq = entry.transition_freq.max(freq);
    }
    entry.last_seen = entry.last_seen.max(last_seen);
    found = true;
  }
  Ok(found)
}

async fn add_context_candidates(
  conn: &Connection,
  config: &SuggestConfig,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  let mut rows_vec: Vec<(String, i64, i64)> = Vec::new();

  if let (Some(cwd), Some(host), Some(user)) =
    (&config.cwd, &config.hostname, &config.username)
  {
    let mut rows = conn
      .query(
        "SELECT command, freq, last_seen FROM context_stats
         WHERE working_directory = ? AND hostname = ? AND username = ?
         ORDER BY freq DESC LIMIT 50",
        (cwd.clone(), host.clone(), user.clone()),
      )
      .await?;
    while let Some(row) = rows.next().await? {
      rows_vec.push((row.get(0)?, row.get(1)?, row.get(2)?));
    }
  }

  if rows_vec.is_empty() {
    if let Some(cwd) = &config.cwd {
      let mut rows = conn
        .query(
          "SELECT command, freq, last_seen FROM context_stats WHERE working_directory = ? ORDER BY freq DESC LIMIT 50",
          libsql::params![cwd.clone()],
        )
        .await?;
      while let Some(row) = rows.next().await? {
        rows_vec.push((row.get(0)?, row.get(1)?, row.get(2)?));
      }
    }
  }

  for (cmd, freq, last_seen) in rows_vec {
    let entry = candidates
      .entry(cmd.clone())
      .or_insert_with(|| new_candidate(&cmd));
    entry.context_freq = entry.context_freq.max(freq);
    entry.last_seen = entry.last_seen.max(last_seen);
  }
  Ok(())
}

async fn add_repo_candidates(
  conn: &Connection,
  repo_root: &str,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  let mut rows = conn
    .query(
      "SELECT command, freq, last_seen FROM repo_command_stats WHERE repo_root = ? ORDER BY freq DESC LIMIT 50",
      libsql::params![repo_root.to_string()],
    )
    .await?;
  while let Some(row) = rows.next().await? {
    let cmd = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let last_seen = row.get::<i64>(2)?;
    let entry = candidates
      .entry(cmd.clone())
      .or_insert_with(|| new_candidate(&cmd));
    entry.repo_freq = entry.repo_freq.max(freq);
    entry.last_seen = entry.last_seen.max(last_seen);
  }
  Ok(())
}

async fn add_global_candidates(
  conn: &Connection,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  let mut rows = conn
    .query(
      "SELECT command, freq, last_seen FROM command_stats ORDER BY freq DESC LIMIT 50",
      (),
    )
    .await?;
  while let Some(row) = rows.next().await? {
    let cmd = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let last_seen = row.get::<i64>(2)?;
    let entry = candidates
      .entry(cmd.clone())
      .or_insert_with(|| new_candidate(&cmd));
    entry.freq = entry.freq.max(freq);
    entry.last_seen = entry.last_seen.max(last_seen);
  }
  Ok(())
}

async fn add_sequence_candidates(
  conn: &Connection,
  recent_commands: &[String],
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  let sequences = candidates_from_sequences(conn, recent_commands, 200).await?;
  for seq in sequences {
    let entry = candidates
      .entry(seq.command.clone())
      .or_insert_with(|| new_candidate(&seq.command));
    if seq.prefix_len >= entry.sequence_prefix_len {
      entry.sequence_confidence = entry.sequence_confidence.max(seq.confidence);
      entry.sequence_lift = entry.sequence_lift.max(seq.lift);
      entry.sequence_prefix_len = entry.sequence_prefix_len.max(seq.prefix_len);
    }
  }
  Ok(())
}

fn push_opt_string(params: &mut Vec<Value>, value: &Option<String>) {
  if let Some(val) = value {
    params.push(Value::from(val.clone()));
  } else {
    params.push(Value::Null);
  }
}

fn push_opt_i64(params: &mut Vec<Value>, value: Option<i64>) {
  if let Some(val) = value {
    params.push(Value::from(val));
  } else {
    params.push(Value::Null);
  }
}

fn build_prefix_variants(prefix: &str, aliases: &HashMap<String, String>) -> Vec<String> {
  let mut variants = Vec::new();
  let trimmed = prefix.trim_start();
  if !trimmed.is_empty() {
    variants.push(trimmed.to_string());
  }
  for (alias, expansion) in aliases {
    if expansion.starts_with(trimmed) {
      variants.push(alias.clone());
    }
  }
  variants.sort();
  variants.dedup();
  variants
}

fn load_aliases() -> HashMap<String, String> {
  let mut map = HashMap::new();
  if let Ok(value) = std::env::var("ZAGE_ALIASES") {
    parse_aliases_into(&value, &mut map);
  }
  if let Ok(path) = std::env::var("ZAGE_ALIAS_FILE") {
    if let Ok(contents) = std::fs::read_to_string(path) {
      parse_aliases_into(&contents, &mut map);
    }
  }
  map
}

fn parse_aliases_into(input: &str, map: &mut HashMap<String, String>) {
  for raw in input.split(|ch| ch == '\n' || ch == ';') {
    if let Some((name, value)) = parse_alias_line(raw) {
      map.insert(name, value);
    }
  }
}

fn parse_alias_line(raw: &str) -> Option<(String, String)> {
  let mut line = raw.trim();
  if line.is_empty() {
    return None;
  }
  if let Some(rest) = line.strip_prefix("alias ") {
    line = rest.trim();
  }
  if line.starts_with("-") {
    return None;
  }
  let (name, value) = line.split_once('=')?;
  let name = name.trim();
  if name.is_empty() {
    return None;
  }
  let mut value = value.trim().to_string();
  if (value.starts_with('\'') && value.ends_with('\'')) || (value.starts_with('"') && value.ends_with('"')) {
    value = value[1..value.len() - 1].to_string();
  }
  if value.is_empty() {
    return None;
  }
  Some((name.to_string(), value))
}

struct PrefixContext {
  base: String,
  head: String,
  flags_json: String,
  flags: Vec<String>,
  arg_index: i64,
  partial: Option<String>,
  partial_is_flag: bool,
  repo_root: String,
}

struct EnvPrefixContext {
  base: String,
  partial: Option<String>,
  repo_root: String,
  match_on_key: bool,
}

async fn arg_template_candidates(
  conn: &Connection,
  prefix: &str,
  repo_root: &str,
  token_priors: &HashMap<String, f64>,
) -> Result<Option<Vec<Suggestion>>> {
  let ctx = match analyze_prefix(prefix, repo_root) {
    Some(ctx) => ctx,
    None => return Ok(None),
  };

  if ctx.partial_is_flag {
    return flag_candidates(conn, &ctx, token_priors).await;
  }

  let like = ctx
    .partial
    .as_ref()
    .map(|p| format!("{p}%"))
    .unwrap_or_else(|| "%".to_string());

  let repo_positional = fetch_arg_candidates(
    conn,
    &ctx.repo_root,
    &ctx.head,
    &ctx.flags_json,
    ctx.arg_index,
    &like,
    &ctx.base,
    token_priors,
  )
  .await?;

  let mut repo_any = fetch_arg_candidates_any(
    conn,
    &ctx.repo_root,
    &ctx.head,
    &ctx.flags_json,
    &like,
    &ctx.base,
    token_priors,
  )
  .await?;

  let mut global_positional = Vec::new();
  let mut global_any = Vec::new();
  if !ctx.repo_root.is_empty() {
    global_positional = fetch_arg_candidates(
      conn,
      "",
      &ctx.head,
      &ctx.flags_json,
      ctx.arg_index,
      &like,
      &ctx.base,
      token_priors,
    )
    .await?;

    global_any = fetch_arg_candidates_any(
      conn,
      "",
      &ctx.head,
      &ctx.flags_json,
      &like,
      &ctx.base,
      token_priors,
    )
    .await?;
  }

  if repo_positional.is_empty()
    && repo_any.is_empty()
    && global_positional.is_empty()
    && global_any.is_empty()
  {
    Ok(None)
  } else {
    let mut merged: HashMap<String, Suggestion> = HashMap::new();
    scale_suggestions(&mut repo_any, 0.6);
    scale_suggestions(&mut global_positional, 0.75);
    scale_suggestions(&mut global_any, 0.45);

    for list in [repo_positional, repo_any, global_positional, global_any] {
      for suggestion in list {
        match merged.get(&suggestion.command) {
          Some(existing) if existing.score >= suggestion.score => {}
          _ => {
            merged.insert(suggestion.command.clone(), suggestion);
          }
        }
      }
    }
    Ok(Some(merged.into_values().collect()))
  }
}

async fn env_template_candidates(
  conn: &Connection,
  prefix: &str,
  repo_root: &str,
  token_priors: &HashMap<String, f64>,
) -> Result<Option<Vec<Suggestion>>> {
  let ctx = match analyze_env_prefix(prefix, repo_root) {
    Some(ctx) => ctx,
    None => return Ok(None),
  };

  let like = ctx
    .partial
    .as_ref()
    .map(|p| format!("{p}%"))
    .unwrap_or_else(|| "%".to_string());

  let repo_env = if ctx.match_on_key {
    fetch_env_key_candidates(conn, &ctx.repo_root, &like, &ctx.base).await?
  } else {
    fetch_env_candidates(
      conn,
      &ctx.repo_root,
      &like,
      &ctx.base,
      token_priors,
    )
    .await?
  };

  let mut global_env = Vec::new();
  if !ctx.repo_root.is_empty() {
    global_env = if ctx.match_on_key {
      fetch_env_key_candidates(conn, "", &like, &ctx.base).await?
    } else {
      fetch_env_candidates(conn, "", &like, &ctx.base, token_priors).await?
    };
  }

  if repo_env.is_empty() && global_env.is_empty() {
    Ok(None)
  } else {
    let mut merged: HashMap<String, Suggestion> = HashMap::new();
    scale_suggestions(&mut global_env, 0.75);
    for suggestion in repo_env.into_iter().chain(global_env.into_iter()) {
      match merged.get(&suggestion.command) {
        Some(existing) if existing.score >= suggestion.score => {}
        _ => {
          merged.insert(suggestion.command.clone(), suggestion);
        }
      }
    }
    Ok(Some(merged.into_values().collect()))
  }
}

async fn flag_candidates(
  conn: &Connection,
  ctx: &PrefixContext,
  token_priors: &HashMap<String, f64>,
) -> Result<Option<Vec<Suggestion>>> {
  let like = ctx
    .partial
    .as_ref()
    .map(|p| format!("{p}%"))
    .unwrap_or_else(|| "%".to_string());

  let exclude: HashSet<String> = ctx.flags.iter().cloned().collect();

  let repo_flags = fetch_flag_candidates(
    conn,
    &ctx.repo_root,
    &ctx.head,
    &like,
    &ctx.base,
    token_priors,
    &exclude,
  )
  .await?;

  let mut global_flags = Vec::new();
  if !ctx.repo_root.is_empty() {
    global_flags = fetch_flag_candidates(
      conn,
      "",
      &ctx.head,
      &like,
      &ctx.base,
      token_priors,
      &exclude,
    )
    .await?;
  }

  if repo_flags.is_empty() && global_flags.is_empty() {
    Ok(None)
  } else {
    let mut merged: HashMap<String, Suggestion> = HashMap::new();
    scale_suggestions(&mut global_flags, 0.75);
    for suggestion in repo_flags.into_iter().chain(global_flags.into_iter()) {
      match merged.get(&suggestion.command) {
        Some(existing) if existing.score >= suggestion.score => {}
        _ => {
          merged.insert(suggestion.command.clone(), suggestion);
        }
      }
    }
    Ok(Some(merged.into_values().collect()))
  }
}

async fn fetch_arg_candidates(
  conn: &Connection,
  repo_root: &str,
  head: &str,
  flags_json: &str,
  arg_index: i64,
  like: &str,
  base_prefix: &str,
  token_priors: &HashMap<String, f64>,
) -> Result<Vec<Suggestion>> {
  let mut rows = conn
    .query(
      "SELECT arg_raw, arg_norm, freq, last_seen
       FROM arg_stats
       WHERE repo_root = ? AND command_head = ? AND flags_json = ? AND arg_index = ? AND arg_raw LIKE ?
       ORDER BY freq DESC, last_seen DESC
       LIMIT 50",
      libsql::params![repo_root, head, flags_json, arg_index, like],
    )
    .await?;

  let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64;

  let mut results = Vec::new();
  while let Some(row) = rows.next().await? {
    let arg_raw = row.get::<String>(0)?;
    let arg_norm = row.get::<String>(1)?;
    let freq = row.get::<i64>(2)?;
    let last_seen = row.get::<i64>(3)?;

    let recency = recency_score(now, last_seen);
    let frequency = (freq as f64).ln_1p();
    let token_prior = token_priors.get(&arg_norm).copied().unwrap_or(0.0);
    let context = if repo_root.is_empty() { 0.0 } else { 1.0 };
    let score = 0.45 * recency + 0.3 * frequency + 0.2 * token_prior + 0.05 * context;

    let mut base = base_prefix.to_string();
    if !base.is_empty() && !base.chars().last().map(|c| c.is_whitespace()).unwrap_or(false) {
      base.push(' ');
    }
    let suggestion = format!("{}{}", base, arg_raw);

    results.push(Suggestion {
      command: suggestion,
      score,
      breakdown: ScoreBreakdown {
        recency,
        frequency,
        transition: 0.0,
        context,
        sequence: token_prior,
        similarity: 0.0,
      },
    });
  }

  Ok(results)
}

async fn fetch_arg_candidates_any(
  conn: &Connection,
  repo_root: &str,
  head: &str,
  flags_json: &str,
  like: &str,
  base_prefix: &str,
  token_priors: &HashMap<String, f64>,
) -> Result<Vec<Suggestion>> {
  let mut rows = conn
    .query(
      "SELECT arg_raw, arg_norm, freq, last_seen
       FROM arg_stats_any
       WHERE repo_root = ? AND command_head = ? AND flags_json = ? AND arg_raw LIKE ?
       ORDER BY freq DESC, last_seen DESC
       LIMIT 50",
      libsql::params![repo_root, head, flags_json, like],
    )
    .await?;

  let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64;

  let mut results = Vec::new();
  while let Some(row) = rows.next().await? {
    let arg_raw = row.get::<String>(0)?;
    let arg_norm = row.get::<String>(1)?;
    let freq = row.get::<i64>(2)?;
    let last_seen = row.get::<i64>(3)?;

    let recency = recency_score(now, last_seen);
    let frequency = (freq as f64).ln_1p();
    let token_prior = token_priors.get(&arg_norm).copied().unwrap_or(0.0);
    let context = if repo_root.is_empty() { 0.0 } else { 1.0 };
    let score = 0.4 * recency + 0.3 * frequency + 0.25 * token_prior + 0.05 * context;

    let mut base = base_prefix.to_string();
    if !base.is_empty() && !base.chars().last().map(|c| c.is_whitespace()).unwrap_or(false) {
      base.push(' ');
    }
    let suggestion = format!("{}{}", base, arg_raw);

    results.push(Suggestion {
      command: suggestion,
      score,
      breakdown: ScoreBreakdown {
        recency,
        frequency,
        transition: 0.0,
        context,
        sequence: token_prior,
        similarity: 0.0,
      },
    });
  }

  Ok(results)
}

async fn fetch_env_candidates(
  conn: &Connection,
  repo_root: &str,
  like: &str,
  base_prefix: &str,
  token_priors: &HashMap<String, f64>,
) -> Result<Vec<Suggestion>> {
  let (sql, params): (&str, Vec<Value>) = if repo_root.is_empty() {
    (
      "SELECT env_raw, env_norm, freq, last_seen
       FROM env_stats
       WHERE env_raw LIKE ?
       ORDER BY freq DESC, last_seen DESC
       LIMIT 50",
      vec![Value::from(like.to_string())],
    )
  } else {
    (
      "SELECT env_raw, env_norm, freq, last_seen
       FROM env_stats
       WHERE repo_root = ? AND env_raw LIKE ?
       ORDER BY freq DESC, last_seen DESC
       LIMIT 50",
      vec![Value::from(repo_root.to_string()), Value::from(like.to_string())],
    )
  };

  let mut rows = conn.query(sql, libsql::params_from_iter(params)).await?;
  let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64;

  let mut results = Vec::new();
  while let Some(row) = rows.next().await? {
    let env_raw = row.get::<String>(0)?;
    let env_norm = row.get::<String>(1)?;
    let freq = row.get::<i64>(2)?;
    let last_seen = row.get::<i64>(3)?;

    let recency = recency_score(now, last_seen);
    let frequency = (freq as f64).ln_1p();
    let token_prior = token_priors.get(&env_norm).copied().unwrap_or(0.0);
    let context = if repo_root.is_empty() { 0.0 } else { 1.0 };
    let score = 0.45 * recency + 0.3 * frequency + 0.2 * token_prior + 0.05 * context;

    let mut base = base_prefix.to_string();
    if !base.is_empty() && !base.chars().last().map(|c| c.is_whitespace()).unwrap_or(false) {
      base.push(' ');
    }
    let suggestion = format!("{}{}", base, env_raw);

    results.push(Suggestion {
      command: suggestion,
      score,
      breakdown: ScoreBreakdown {
        recency,
        frequency,
        transition: 0.0,
        context,
        sequence: token_prior,
        similarity: 0.0,
      },
    });
  }

  Ok(results)
}

async fn fetch_env_key_candidates(
  conn: &Connection,
  repo_root: &str,
  like: &str,
  base_prefix: &str,
) -> Result<Vec<Suggestion>> {
  let (sql, params): (&str, Vec<Value>) = if repo_root.is_empty() {
    (
      "SELECT env_key, SUM(freq) as freq, MAX(last_seen) as last_seen
       FROM env_stats
       WHERE env_key LIKE ?
       GROUP BY env_key
       ORDER BY freq DESC, last_seen DESC
       LIMIT 50",
      vec![Value::from(like.to_string())],
    )
  } else {
    (
      "SELECT env_key, SUM(freq) as freq, MAX(last_seen) as last_seen
       FROM env_stats
       WHERE repo_root = ? AND env_key LIKE ?
       GROUP BY env_key
       ORDER BY freq DESC, last_seen DESC
       LIMIT 50",
      vec![Value::from(repo_root.to_string()), Value::from(like.to_string())],
    )
  };

  let mut rows = conn.query(sql, libsql::params_from_iter(params)).await?;
  let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64;

  let mut results = Vec::new();
  while let Some(row) = rows.next().await? {
    let env_key = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let last_seen = row.get::<i64>(2)?;

    let recency = recency_score(now, last_seen);
    let frequency = (freq as f64).ln_1p();
    let context = if repo_root.is_empty() { 0.0 } else { 1.0 };
    let score = 0.55 * recency + 0.35 * frequency + 0.1 * context;

    let mut base = base_prefix.to_string();
    if !base.is_empty() && !base.chars().last().map(|c| c.is_whitespace()).unwrap_or(false) {
      base.push(' ');
    }
    let suggestion = format!("{}{}=", base, env_key);

    results.push(Suggestion {
      command: suggestion,
      score,
      breakdown: ScoreBreakdown {
        recency,
        frequency,
        transition: 0.0,
        context,
        sequence: 0.0,
        similarity: 0.0,
      },
    });
  }

  Ok(results)
}

async fn fetch_flag_candidates(
  conn: &Connection,
  repo_root: &str,
  head: &str,
  like: &str,
  base_prefix: &str,
  token_priors: &HashMap<String, f64>,
  exclude: &HashSet<String>,
) -> Result<Vec<Suggestion>> {
  let mut rows = conn
    .query(
      "SELECT flag_raw, flag_norm, freq, last_seen
       FROM flag_stats
       WHERE repo_root = ? AND command_head = ? AND flag_raw LIKE ?
       ORDER BY freq DESC, last_seen DESC
       LIMIT 50",
      libsql::params![repo_root, head, like],
    )
    .await?;

  let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64;

  let mut results = Vec::new();
  while let Some(row) = rows.next().await? {
    let flag_raw = row.get::<String>(0)?;
    let flag_norm = row.get::<String>(1)?;
    if exclude.contains(&flag_raw) {
      continue;
    }
    let freq = row.get::<i64>(2)?;
    let last_seen = row.get::<i64>(3)?;

    let recency = recency_score(now, last_seen);
    let frequency = (freq as f64).ln_1p();
    let token_prior = token_priors.get(&flag_norm).copied().unwrap_or(0.0);
    let context = if repo_root.is_empty() { 0.0 } else { 1.0 };
    let score = 0.45 * recency + 0.3 * frequency + 0.2 * token_prior + 0.05 * context;

    let mut base = base_prefix.to_string();
    if !base.is_empty() && !base.chars().last().map(|c| c.is_whitespace()).unwrap_or(false) {
      base.push(' ');
    }
    let suggestion = format!("{}{}", base, flag_raw);

    results.push(Suggestion {
      command: suggestion,
      score,
      breakdown: ScoreBreakdown {
        recency,
        frequency,
        transition: 0.0,
        context,
        sequence: token_prior,
        similarity: 0.0,
      },
    });
  }

  Ok(results)
}

fn analyze_prefix(prefix: &str, repo_root: &str) -> Option<PrefixContext> {
  let tokens = tokenize(prefix);
  if tokens.is_empty() {
    return None;
  }
  let parts = extract_command_parts(prefix, &tokens)?;
  let mut flags = parts.flags;
  flags.sort();
  let flags_json = serde_json::to_string(&flags).ok()?;

  let ends_with_space = prefix
    .chars()
    .last()
    .map(|c| c.is_whitespace())
    .unwrap_or(false);

  let mut arg_index = parts.args.len() as i64;
  let mut partial = None;

  if !ends_with_space {
    if let Some(last) = tokens.last() {
      if matches!(last.kind, TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment) {
        partial = Some(last.raw.clone());
      } else {
        return None;
      }
    }
  }

  let partial_is_flag = partial.as_deref().map(|p| p.starts_with('-')).unwrap_or(false);
  if partial_is_flag {
    if let Some(ref part) = partial {
      flags.retain(|flag| flag != part);
    }
  } else if partial.is_some() {
    if arg_index > 0 {
      arg_index -= 1;
    }
  }

  let mut base = if let Some(ref part) = partial {
    if let Some(pos) = prefix.rfind(part) {
      prefix[..pos].to_string()
    } else {
      prefix.to_string()
    }
  } else {
    prefix.to_string()
  };

  if !base.is_empty() && !base.chars().last().map(|c| c.is_whitespace()).unwrap_or(false) {
    base.push(' ');
  }

  Some(PrefixContext {
    base,
    head: parts.head,
    flags_json,
    flags,
    arg_index,
    partial,
    partial_is_flag,
    repo_root: repo_root.to_string(),
  })
}

fn analyze_env_prefix(prefix: &str, repo_root: &str) -> Option<EnvPrefixContext> {
  let tokens = tokenize(prefix);
  if tokens.is_empty() {
    return None;
  }

  let ends_with_space = prefix
    .chars()
    .last()
    .map(|c| c.is_whitespace())
    .unwrap_or(false);

  let env_count = leading_env_count(&tokens);
  let last_idx = tokens.len().saturating_sub(1);

  let (partial, match_on_key) = if ends_with_space {
    (None, true)
  } else if env_count == tokens.len() {
    let last = tokens.last()?;
    if let Some(eq_idx) = last.raw.find('=') {
      if looks_like_env_lhs(&last.raw[..eq_idx]) {
        (Some(last.raw.clone()), false)
      } else {
        return None;
      }
    } else if looks_like_env_lhs(&last.raw) {
      (Some(last.raw.clone()), true)
    } else {
      return None;
    }
  } else if env_count == last_idx {
    let last = tokens.last()?;
    if looks_like_env_lhs(&last.raw) {
      (Some(last.raw.clone()), true)
    } else {
      return None;
    }
  } else {
    return None;
  };

  let mut base = if let Some(ref part) = partial {
    if let Some(pos) = prefix.rfind(part) {
      prefix[..pos].to_string()
    } else {
      prefix.to_string()
    }
  } else {
    prefix.to_string()
  };

  if !base.is_empty() && !base.chars().last().map(|c| c.is_whitespace()).unwrap_or(false) {
    base.push(' ');
  }

  Some(EnvPrefixContext {
    base,
    partial,
    repo_root: repo_root.to_string(),
    match_on_key,
  })
}

fn scale_suggestions(list: &mut [Suggestion], weight: f64) {
  if (weight - 1.0).abs() < f64::EPSILON {
    return;
  }
  for suggestion in list.iter_mut() {
    suggestion.score *= weight;
    suggestion.breakdown.recency *= weight;
    suggestion.breakdown.frequency *= weight;
    suggestion.breakdown.context *= weight;
    suggestion.breakdown.sequence *= weight;
    suggestion.breakdown.similarity *= weight;
  }
}

fn split_env_prefix(prefix: &str) -> (String, String) {
  let tokens = tokenize(prefix);
  if tokens.is_empty() {
    return (String::new(), prefix.to_string());
  }
  let env_count = leading_env_count(&tokens);
  if env_count == 0 {
    return (String::new(), prefix.to_string());
  }

  let mut search_start = 0usize;
  let mut end = 0usize;
  for idx in 0..env_count {
    if let Some(found) = prefix[search_start..].find(&tokens[idx].raw) {
      let start = search_start + found;
      end = start + tokens[idx].raw.len();
      search_start = end;
    } else {
      return (String::new(), prefix.to_string());
    }
  }

  let bytes = prefix.as_bytes();
  let mut end_with_space = end;
  while end_with_space < bytes.len() && bytes[end_with_space].is_ascii_whitespace() {
    end_with_space += 1;
  }

  let env_prefix = prefix[..end_with_space].to_string();
  let command_prefix = prefix[end_with_space..].to_string();
  (env_prefix, command_prefix)
}

fn leading_env_count(tokens: &[Token]) -> usize {
  let mut idx = 0usize;
  while idx < tokens.len() {
    let token = &tokens[idx];
    if matches!(token.kind, TokenKind::Assignment) {
      idx += 1;
      continue;
    }
    if looks_like_env_lhs(&token.raw) {
      if let Some(next) = tokens.get(idx + 1) {
        if next.raw == "=" {
          idx += if tokens.get(idx + 2).is_some() { 3 } else { 2 };
          continue;
        }
        if next.raw.starts_with('=') {
          idx += 2;
          continue;
        }
      }
    }
    break;
  }
  idx
}

fn looks_like_env_lhs(raw: &str) -> bool {
  if raw.is_empty() || raw.starts_with('-') {
    return false;
  }
  let mut chars = raw.chars();
  let first = match chars.next() {
    Some(c) => c,
    None => return false,
  };
  if !(first.is_ascii_alphabetic() || first == '_') {
    return false;
  }
  chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

async fn token_sequence_predictions(
  conn: &Connection,
  prefix_norm: &[String],
) -> Result<HashMap<String, f64>> {
  if prefix_norm.is_empty() {
    return Ok(HashMap::new());
  }
  let mut rows = conn
    .query(
      "SELECT sequence_json, confidence, lift, prefix_len
       FROM token_sequence_stats
       ORDER BY lift DESC
       LIMIT 500",
      (),
    )
    .await?;

  let mut scores: HashMap<String, f64> = HashMap::new();
  while let Some(row) = rows.next().await? {
    let sequence_json = row.get::<String>(0)?;
    let confidence = row.get::<f64>(1)?;
    let lift = row.get::<f64>(2)?;
    let prefix_len = row.get::<i64>(3)? as usize;

    let sequence: Vec<String> = serde_json::from_str(&sequence_json)?;
    if sequence.len() < 2 || prefix_len == 0 {
      continue;
    }
    if prefix_norm.len() < prefix_len {
      continue;
    }
    let recent_slice = &prefix_norm[prefix_norm.len() - prefix_len..];
    if sequence[..prefix_len] == *recent_slice {
      if let Some(next_token) = sequence.last() {
        let score = confidence * lift;
        let entry = scores.entry(next_token.clone()).or_insert(0.0);
        if score > *entry {
          *entry = score;
        }
      }
    }
  }
  Ok(scores)
}

fn expand_alias(command: &str, aliases: &HashMap<String, String>) -> Option<String> {
  let mut parts = command.splitn(2, char::is_whitespace);
  let head = parts.next()?;
  let tail = parts.next().unwrap_or("");
  let expansion = aliases.get(head)?;
  if tail.is_empty() {
    Some(expansion.clone())
  } else {
    Some(format!("{} {}", expansion, tail.trim_start()))
  }
}

async fn load_session_stats(
  conn: &Connection,
  session_id: i64,
  recent_limit: usize,
) -> Result<HashMap<String, (i64, i64)>> {
  let mut rows = conn
    .query(
      "SELECT command, COUNT(*) as freq, MAX(COALESCE(start_unix_timestamp, 0)) as last_seen
       FROM (
         SELECT command, start_unix_timestamp
         FROM shell_history
         WHERE session_id = ?
         ORDER BY COALESCE(start_unix_timestamp, 0) DESC, id DESC
         LIMIT ?
       )
       GROUP BY command",
      libsql::params![session_id, recent_limit as i64],
    )
    .await?;

  let mut stats = HashMap::new();
  while let Some(row) = rows.next().await? {
    let command = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let last_seen = row.get::<i64>(2)?;
    stats.insert(command, (freq, last_seen));
  }
  Ok(stats)
}

fn recency_score(now: i64, last_seen: i64) -> f64 {
  if last_seen <= 0 || now <= last_seen {
    return 0.0;
  }
  let half_life = 60.0 * 60.0 * 24.0 * 7.0;
  let age = (now - last_seen) as f64;
  (-age / half_life).exp()
}

fn token_similarity(a: &[String], b: &[String]) -> f64 {
  if a.is_empty() || b.is_empty() {
    return 0.0;
  }
  let set_a: HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
  let set_b: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
  let intersection = set_a.intersection(&set_b).count() as f64;
  (2.0 * intersection) / (set_a.len() as f64 + set_b.len() as f64)
}

async fn load_normalized_tokens(conn: &Connection, command: &str) -> Result<Vec<String>> {
  let mut rows = conn
    .query(
      "SELECT normalized_json FROM token_cache WHERE command = ?",
      libsql::params![command.to_string()],
    )
    .await?;
  if let Some(row) = rows.next().await? {
    let json = row.get::<String>(0)?;
    let tokens: Vec<String> = serde_json::from_str(&json)?;
    return Ok(tokens);
  }

  let (raw_tokens, norm_tokens) = token_strings(command);
  let raw_json = serde_json::to_string(&raw_tokens)?;
  let norm_json = serde_json::to_string(&norm_tokens)?;
  let now = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs() as i64;
  conn
    .execute(
      "INSERT OR REPLACE INTO token_cache (command, tokens_json, normalized_json, updated_at)
       VALUES (?, ?, ?, ?)",
      (command.to_string(), raw_json, norm_json, now),
    )
    .await?;
  Ok(norm_tokens)
}
