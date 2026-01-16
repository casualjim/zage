use std::collections::{HashMap, HashSet};

use libsql::{Connection, Value};
use serde_json;

use crate::Result;
use crate::tokenize::{Token, TokenKind, extract_command_parts, tokenize, tokenize_index};

use super::ranking::recency_score;
use super::sql::query_prepared;
use super::{ScoreBreakdown, Suggestion};

#[derive(Debug)]
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

#[derive(Debug)]
struct EnvPrefixContext {
  base: String,
  partial: Option<String>,
  repo_root: String,
  match_on_key: bool,
}

struct ArgCandidateQuery<'a> {
  repo_root: &'a str,
  head: &'a str,
  flags_json: &'a str,
  base_prefix: &'a str,
  token_priors: &'a HashMap<String, f64>,
}

pub(crate) async fn arg_template_candidates(
  conn: &Connection,
  prefix: &str,
  repo_root: &str,
  shellname: &str,
  token_priors: &HashMap<String, f64>,
  now: i64,
  recency_half_life: f64,
) -> Result<Option<Vec<Suggestion>>> {
  let ctx = match analyze_prefix(prefix, repo_root, shellname) {
    Some(ctx) => ctx,
    None => return Ok(None),
  };

  if ctx.partial_is_flag {
    return flag_candidates(conn, &ctx, token_priors, now, recency_half_life).await;
  }

  let like = ctx
    .partial
    .as_ref()
    .map(|p| format!("{p}%"))
    .unwrap_or_else(|| "%".to_string());

  let repo_query = ArgCandidateQuery {
    repo_root: &ctx.repo_root,
    head: &ctx.head,
    flags_json: &ctx.flags_json,
    base_prefix: &ctx.base,
    token_priors,
  };

  let repo_positional = fetch_arg_candidates(
    conn,
    &repo_query,
    ctx.arg_index,
    &like,
    now,
    recency_half_life,
  )
  .await?;

  let repo_any = fetch_arg_candidates_any(conn, &repo_query, &like, now, recency_half_life).await?;

  let mut global_positional = Vec::new();
  let mut global_any = Vec::new();
  if !ctx.repo_root.is_empty() {
    let global_query = ArgCandidateQuery {
      repo_root: "",
      head: &ctx.head,
      flags_json: &ctx.flags_json,
      base_prefix: &ctx.base,
      token_priors,
    };

    global_positional = fetch_arg_candidates(
      conn,
      &global_query,
      ctx.arg_index,
      &like,
      now,
      recency_half_life,
    )
    .await?;

    global_any =
      fetch_arg_candidates_any(conn, &global_query, &like, now, recency_half_life).await?;
  }

  let has_positional = !(repo_positional.is_empty() && global_positional.is_empty());
  if has_positional {
    let mut merged: HashMap<String, Suggestion> = HashMap::new();
    scale_suggestions(&mut global_positional, 0.75);
    for list in [repo_positional, global_positional] {
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
  } else if repo_any.is_empty() && global_any.is_empty() {
    Ok(None)
  } else {
    let mut merged: HashMap<String, Suggestion> = HashMap::new();
    scale_suggestions(&mut global_any, 0.75);
    for suggestion in repo_any.into_iter().chain(global_any.into_iter()) {
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

pub(crate) async fn env_template_candidates(
  conn: &Connection,
  prefix: &str,
  repo_root: &str,
  token_priors: &HashMap<String, f64>,
  now: i64,
  recency_half_life: f64,
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
    fetch_env_key_candidates(
      conn,
      &ctx.repo_root,
      &like,
      &ctx.base,
      now,
      recency_half_life,
    )
    .await?
  } else {
    fetch_env_candidates(
      conn,
      &ctx.repo_root,
      &like,
      &ctx.base,
      token_priors,
      now,
      recency_half_life,
    )
    .await?
  };

  let mut global_env = Vec::new();
  if !ctx.repo_root.is_empty() {
    global_env = if ctx.match_on_key {
      fetch_env_key_candidates(conn, "", &like, &ctx.base, now, recency_half_life).await?
    } else {
      fetch_env_candidates(
        conn,
        "",
        &like,
        &ctx.base,
        token_priors,
        now,
        recency_half_life,
      )
      .await?
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
  now: i64,
  recency_half_life: f64,
) -> Result<Option<Vec<Suggestion>>> {
  let like = ctx
    .partial
    .as_ref()
    .map(|p| format!("{p}%"))
    .unwrap_or_else(|| "%".to_string());

  let exclude: HashSet<String> = ctx.flags.iter().cloned().collect();

  let repo_context = FlagCandidateContext {
    repo_root: &ctx.repo_root,
    head: &ctx.head,
    like: &like,
    base_prefix: &ctx.base,
    token_priors,
    exclude: &exclude,
    now,
    recency_half_life,
  };
  let mut repo_flags = fetch_flag_candidates(conn, &repo_context).await?;

  let mut global_flags = Vec::new();
  if !ctx.repo_root.is_empty() {
    let global_context = FlagCandidateContext {
      repo_root: "",
      head: &ctx.head,
      like: &like,
      base_prefix: &ctx.base,
      token_priors,
      exclude: &exclude,
      now,
      recency_half_life,
    };
    global_flags = fetch_flag_candidates(conn, &global_context).await?;
  }

  if matches!(ctx.partial.as_deref(), Some("-")) {
    let is_long_flag = |command: &str| {
      command
        .split_whitespace()
        .last()
        .map(|flag| flag.starts_with("--"))
        .unwrap_or(false)
    };
    let has_short_flag = repo_flags
      .iter()
      .chain(global_flags.iter())
      .any(|suggestion| !is_long_flag(&suggestion.command));
    if has_short_flag {
      repo_flags.retain(|suggestion| !is_long_flag(&suggestion.command));
      global_flags.retain(|suggestion| !is_long_flag(&suggestion.command));
    }
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
  query: &ArgCandidateQuery<'_>,
  arg_index: i64,
  like: &str,
  now: i64,
  recency_half_life: f64,
) -> Result<Vec<Suggestion>> {
  let mut rows = query_prepared(
    conn,
    "SELECT arg_raw, arg_norm, freq, last_seen
     FROM arg_stats
     WHERE repo_root = ? AND command_head = ? AND flags_json = ? AND arg_index = ? AND arg_raw LIKE ?
     ORDER BY freq DESC, last_seen DESC
     LIMIT 50",
    libsql::params![query.repo_root, query.head, query.flags_json, arg_index, like],
  )
  .await?;

  let mut results = Vec::new();
  while let Some(row) = rows.next().await? {
    let arg_raw = row.get::<String>(0)?;
    let arg_norm = row.get::<String>(1)?;
    let freq = row.get::<i64>(2)?;
    let last_seen = row.get::<i64>(3)?;

    let recency = recency_score(now, last_seen, recency_half_life);
    let frequency = (freq as f64).ln_1p();
    let token_prior = query.token_priors.get(&arg_norm).copied().unwrap_or(0.0);
    let context = if query.repo_root.is_empty() { 0.0 } else { 1.0 };
    let score = 0.45 * recency + 0.3 * frequency + 0.2 * token_prior + 0.05 * context;

    let mut base = query.base_prefix.to_string();
    if !base.is_empty()
      && !base
        .chars()
        .last()
        .map(|c| c.is_whitespace())
        .unwrap_or(false)
    {
      base.push(' ');
    }
    let suggestion = format!("{base}{arg_raw}");

    results.push(Suggestion {
      command: suggestion,
      score,
      breakdown: ScoreBreakdown {
        recency,
        session_recency: 0.0,
        frequency,
        transition: 0.0,
        context,
        sequence: token_prior,
        similarity: 0.0,
        online_model: 0.0,
      },
    });
  }

  Ok(results)
}

async fn fetch_arg_candidates_any(
  conn: &Connection,
  query: &ArgCandidateQuery<'_>,
  like: &str,
  now: i64,
  recency_half_life: f64,
) -> Result<Vec<Suggestion>> {
  let mut rows = query_prepared(
    conn,
    "SELECT arg_raw, arg_norm, freq, last_seen
     FROM arg_stats_any
     WHERE repo_root = ? AND command_head = ? AND flags_json = ? AND arg_raw LIKE ?
     ORDER BY freq DESC, last_seen DESC
     LIMIT 50",
    libsql::params![query.repo_root, query.head, query.flags_json, like],
  )
  .await?;

  let mut results = Vec::new();
  while let Some(row) = rows.next().await? {
    let arg_raw = row.get::<String>(0)?;
    let arg_norm = row.get::<String>(1)?;
    let freq = row.get::<i64>(2)?;
    let last_seen = row.get::<i64>(3)?;

    let recency = recency_score(now, last_seen, recency_half_life);
    let frequency = (freq as f64).ln_1p();
    let token_prior = query.token_priors.get(&arg_norm).copied().unwrap_or(0.0);
    let context = if query.repo_root.is_empty() { 0.0 } else { 1.0 };
    let score = 0.4 * recency + 0.3 * frequency + 0.25 * token_prior + 0.05 * context;

    let mut base = query.base_prefix.to_string();
    if !base.is_empty()
      && !base
        .chars()
        .last()
        .map(|c| c.is_whitespace())
        .unwrap_or(false)
    {
      base.push(' ');
    }
    let suggestion = format!("{base}{arg_raw}");

    results.push(Suggestion {
      command: suggestion,
      score,
      breakdown: ScoreBreakdown {
        recency,
        session_recency: 0.0,
        frequency,
        transition: 0.0,
        context,
        sequence: token_prior,
        similarity: 0.0,
        online_model: 0.0,
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
  now: i64,
  recency_half_life: f64,
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
      vec![
        Value::from(repo_root.to_string()),
        Value::from(like.to_string()),
      ],
    )
  };

  let mut rows = query_prepared(conn, sql, libsql::params_from_iter(params)).await?;

  let mut results = Vec::new();
  while let Some(row) = rows.next().await? {
    let env_raw = row.get::<String>(0)?;
    let env_norm = row.get::<String>(1)?;
    let freq = row.get::<i64>(2)?;
    let last_seen = row.get::<i64>(3)?;

    let recency = recency_score(now, last_seen, recency_half_life);
    let frequency = (freq as f64).ln_1p();
    let token_prior = token_priors.get(&env_norm).copied().unwrap_or(0.0);
    let context = if repo_root.is_empty() { 0.0 } else { 1.0 };
    let score = 0.45 * recency + 0.3 * frequency + 0.2 * token_prior + 0.05 * context;

    let mut base = base_prefix.to_string();
    if !base.is_empty()
      && !base
        .chars()
        .last()
        .map(|c| c.is_whitespace())
        .unwrap_or(false)
    {
      base.push(' ');
    }
    let suggestion = format!("{base}{env_raw}");

    results.push(Suggestion {
      command: suggestion,
      score,
      breakdown: ScoreBreakdown {
        recency,
        session_recency: 0.0,
        frequency,
        transition: 0.0,
        context,
        sequence: token_prior,
        similarity: 0.0,
        online_model: 0.0,
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
  now: i64,
  recency_half_life: f64,
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
      vec![
        Value::from(repo_root.to_string()),
        Value::from(like.to_string()),
      ],
    )
  };

  let mut rows = query_prepared(conn, sql, libsql::params_from_iter(params)).await?;

  let mut results = Vec::new();
  while let Some(row) = rows.next().await? {
    let env_key = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let last_seen = row.get::<i64>(2)?;

    let recency = recency_score(now, last_seen, recency_half_life);
    let frequency = (freq as f64).ln_1p();
    let context = if repo_root.is_empty() { 0.0 } else { 1.0 };
    let score = 0.55 * recency + 0.35 * frequency + 0.1 * context;

    let mut base = base_prefix.to_string();
    if !base.is_empty()
      && !base
        .chars()
        .last()
        .map(|c| c.is_whitespace())
        .unwrap_or(false)
    {
      base.push(' ');
    }
    let suggestion = format!("{base}{env_key}=");

    results.push(Suggestion {
      command: suggestion,
      score,
      breakdown: ScoreBreakdown {
        recency,
        session_recency: 0.0,
        frequency,
        transition: 0.0,
        context,
        sequence: 0.0,
        similarity: 0.0,
        online_model: 0.0,
      },
    });
  }

  Ok(results)
}

struct FlagCandidateContext<'a> {
  repo_root: &'a str,
  head: &'a str,
  like: &'a str,
  base_prefix: &'a str,
  token_priors: &'a HashMap<String, f64>,
  exclude: &'a HashSet<String>,
  now: i64,
  recency_half_life: f64,
}

async fn fetch_flag_candidates(
  conn: &Connection,
  context: &FlagCandidateContext<'_>,
) -> Result<Vec<Suggestion>> {
  let mut rows = query_prepared(
    conn,
    "SELECT flag_raw, flag_norm, freq, last_seen
     FROM flag_stats
     WHERE repo_root = ? AND command_head = ? AND flag_raw LIKE ?
     ORDER BY freq DESC, last_seen DESC
     LIMIT 50",
    libsql::params![context.repo_root, context.head, context.like],
  )
  .await?;

  let mut results = Vec::new();
  while let Some(row) = rows.next().await? {
    let flag_raw = row.get::<String>(0)?;
    let flag_norm = row.get::<String>(1)?;
    if context.exclude.contains(&flag_raw) {
      continue;
    }
    let freq = row.get::<i64>(2)?;
    let last_seen = row.get::<i64>(3)?;

    let recency = recency_score(context.now, last_seen, context.recency_half_life);
    let frequency = (freq as f64).ln_1p();
    let token_prior = context.token_priors.get(&flag_norm).copied().unwrap_or(0.0);
    let context_score = if context.repo_root.is_empty() {
      0.0
    } else {
      1.0
    };
    let score = 0.45 * recency + 0.3 * frequency + 0.2 * token_prior + 0.05 * context_score;

    let mut base = context.base_prefix.to_string();
    if !base.is_empty()
      && !base
        .chars()
        .last()
        .map(|c| c.is_whitespace())
        .unwrap_or(false)
    {
      base.push(' ');
    }
    let suggestion = format!("{base}{flag_raw}");

    results.push(Suggestion {
      command: suggestion,
      score,
      breakdown: ScoreBreakdown {
        recency,
        session_recency: 0.0,
        frequency,
        transition: 0.0,
        context: context_score,
        sequence: token_prior,
        similarity: 0.0,
        online_model: 0.0,
      },
    });
  }

  Ok(results)
}

fn analyze_prefix(prefix: &str, repo_root: &str, shellname: &str) -> Option<PrefixContext> {
  let tokens = tokenize_index(shellname, prefix);
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

  if !ends_with_space && let Some(last) = tokens.last() {
    if matches!(
      last.kind,
      TokenKind::Word | TokenKind::Quoted | TokenKind::Variable | TokenKind::Assignment
    ) {
      partial = Some(last.raw.clone());
    } else {
      return None;
    }
  }

  let partial_is_flag = partial
    .as_deref()
    .map(|p| p.starts_with('-'))
    .unwrap_or(false);
  if partial_is_flag {
    if let Some(ref part) = partial {
      flags.retain(|flag| flag != part);
    }
  } else if partial.is_some() && arg_index > 0 {
    arg_index -= 1;
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

  base = base.trim_end().to_string();
  if !base.is_empty()
    && !base
      .chars()
      .last()
      .map(|c| c.is_whitespace())
      .unwrap_or(false)
  {
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
    if env_count == 0 {
      return None;
    }
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

  if !base.is_empty()
    && !base
      .chars()
      .last()
      .map(|c| c.is_whitespace())
      .unwrap_or(false)
  {
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

pub(crate) fn split_env_prefix(prefix: &str) -> (String, String) {
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
  for token in tokens.iter().take(env_count) {
    if let Some(found) = prefix[search_start..].find(&token.raw) {
      let start = search_start + found;
      end = start + token.raw.len();
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
    if looks_like_env_lhs(&token.raw)
      && let Some(next) = tokens.get(idx + 1)
    {
      if next.raw == "=" {
        idx += if tokens.get(idx + 2).is_some() { 3 } else { 2 };
        continue;
      }
      if next.raw.starts_with('=') {
        idx += 2;
        continue;
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

pub(crate) async fn token_sequence_predictions(
  conn: &Connection,
  prefix_norm: &[String],
) -> Result<HashMap<String, f64>> {
  if prefix_norm.is_empty() {
    return Ok(HashMap::new());
  }
  let mut rows = query_prepared(
    conn,
    "SELECT sequence_json, confidence, lift, prefix_len
     FROM token_sequence_stats
     WHERE prefix_len <= ?
     ORDER BY lift DESC
     LIMIT 500",
    libsql::params![prefix_norm.len() as i64],
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
    if sequence[..prefix_len] == *recent_slice
      && let Some(next_token) = sequence.last()
    {
      let score = confidence * lift;
      let entry = scores.entry(next_token.clone()).or_insert(0.0);
      if score > *entry {
        *entry = score;
      }
    }
  }
  Ok(scores)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::db::{init, open_db};

  #[tokio::test]
  async fn arg_templates_prefer_positional_matches() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    db.conn
      .execute(
        "INSERT INTO arg_stats (repo_root, command_head, flags_json, arg_index, arg_raw, arg_norm, freq, last_seen)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("", "git", "[]", 0i64, "status", "status", 5i64, 10i64),
      )
      .await
      .unwrap();
    db.conn
      .execute(
        "INSERT INTO arg_stats_any (repo_root, command_head, flags_json, arg_raw, arg_norm, freq, last_seen)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("", "git", "[]", "commit", "commit", 4i64, 9i64),
      )
      .await
      .unwrap();

    let token_priors = HashMap::new();
    let suggestions = arg_template_candidates(
      &db.conn,
      "git ",
      "",
      "zsh",
      &token_priors,
      1_000,
      crate::predict::ranking::DEFAULT_RECENCY_HALF_LIFE_SECONDS,
    )
    .await
    .unwrap()
    .unwrap();
    let commands: Vec<String> = suggestions.into_iter().map(|s| s.command).collect();

    assert!(commands.iter().any(|c| c == "git status"));
    assert!(!commands.iter().any(|c| c == "git commit"));
  }

  #[tokio::test]
  async fn arg_templates_fall_back_to_any_position() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let db = open_db(tmp.path()).await.unwrap();
    init(&db.conn).await.unwrap();

    db.conn
      .execute(
        "INSERT INTO arg_stats_any (repo_root, command_head, flags_json, arg_raw, arg_norm, freq, last_seen)
         VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("", "git", "[]", "commit", "commit", 4i64, 9i64),
      )
      .await
      .unwrap();

    let token_priors = HashMap::new();
    let suggestions = arg_template_candidates(
      &db.conn,
      "git ",
      "",
      "zsh",
      &token_priors,
      1_000,
      crate::predict::ranking::DEFAULT_RECENCY_HALF_LIFE_SECONDS,
    )
    .await
    .unwrap()
    .unwrap();
    let commands: Vec<String> = suggestions.into_iter().map(|s| s.command).collect();

    assert!(commands.iter().any(|c| c == "git commit"));
  }
}
