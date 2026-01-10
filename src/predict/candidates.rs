use std::collections::{HashMap, HashSet};

use libsql::{Connection, Value};

use crate::Result;
use crate::sequence::candidates_from_sequences;

use super::phase_support::{PhaseSignal, command_head_for_phase};
use super::sql::query_prepared;
use crate::core::Candidate;

pub(crate) async fn add_transition_candidates(
  conn: &Connection,
  last_command: &str,
  last_exit_status: Option<i64>,
  repo_root: &str,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  if !repo_root.is_empty() {
    let found = fetch_transition_candidates(
      conn,
      TransitionQuery::new(
        "repo_transition_stats",
        Some(repo_root),
        last_command,
        last_exit_status,
        true,
        last_exit_status.is_some(),
      ),
      candidates,
    )
    .await?;
    if !found {
      let _ = fetch_transition_candidates(
        conn,
        TransitionQuery::new(
          "repo_transition_stats",
          Some(repo_root),
          last_command,
          None,
          true,
          false,
        ),
        candidates,
      )
      .await?;
    }
  }

  let found = fetch_transition_candidates(
    conn,
    TransitionQuery::new(
      "transition_stats",
      None,
      last_command,
      last_exit_status,
      false,
      last_exit_status.is_some(),
    ),
    candidates,
  )
  .await?;
  if !found {
    let _ = fetch_transition_candidates(
      conn,
      TransitionQuery::new("transition_stats", None, last_command, None, false, false),
      candidates,
    )
    .await?;
  }
  Ok(())
}

struct TransitionQuery<'a> {
  table: &'a str,
  repo_root: Option<&'a str>,
  last_command: &'a str,
  last_exit_status: Option<i64>,
  is_repo: bool,
  status_specific: bool,
}

impl<'a> TransitionQuery<'a> {
  fn new(
    table: &'a str,
    repo_root: Option<&'a str>,
    last_command: &'a str,
    last_exit_status: Option<i64>,
    is_repo: bool,
    status_specific: bool,
  ) -> Self {
    Self {
      table,
      repo_root,
      last_command,
      last_exit_status,
      is_repo,
      status_specific,
    }
  }
}

async fn fetch_transition_candidates(
  conn: &Connection,
  query: TransitionQuery<'_>,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<bool> {
  let mut sql = format!(
    "SELECT next_command, freq FROM {} WHERE prev_command = ?",
    query.table
  );
  let mut params: Vec<Value> = vec![Value::from(query.last_command.to_string())];

  if let Some(status) = query.last_exit_status {
    sql.push_str(" AND prev_exit_status = ?");
    params.push(Value::from(status));
  }
  if let Some(repo_root) = query.repo_root {
    sql.push_str(" AND repo_root = ?");
    params.push(Value::from(repo_root.to_string()));
  }

  sql.push_str(" ORDER BY freq DESC LIMIT 50");

  let mut rows = query_prepared(conn, &sql, libsql::params_from_iter(params)).await?;
  let mut found = false;
  while let Some(row) = rows.next().await? {
    found = true;
    let cmd = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let entry = candidates
      .entry(cmd.clone())
      .or_insert_with(|| Candidate::new(&cmd));
    if query.is_repo {
      entry.repo_transition_freq = entry.repo_transition_freq.max(freq);
    } else {
      entry.transition_freq = entry.transition_freq.max(freq);
    }
    if query.status_specific {
      entry.transition_exit_status_match = true;
    }
  }
  Ok(found)
}

async fn fetch_context_candidates(
  conn: &Connection,
  cwd: Option<&str>,
  hostname: Option<&str>,
  username: Option<&str>,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<bool> {
  let mut sql = String::from("SELECT command, freq, last_seen FROM context_stats WHERE 1=1");
  let mut params: Vec<Value> = Vec::new();

  if let Some(cwd) = cwd {
    sql.push_str(" AND working_directory = ?");
    params.push(Value::from(cwd.to_string()));
  }
  if let Some(hostname) = hostname {
    sql.push_str(" AND hostname = ?");
    params.push(Value::from(hostname.to_string()));
  }
  if let Some(username) = username {
    sql.push_str(" AND username = ?");
    params.push(Value::from(username.to_string()));
  }
  // context_stats is keyed by (command, cwd, hostname, username); session_id is tracked elsewhere
  sql.push_str(" ORDER BY freq DESC LIMIT 50");

  let mut rows = query_prepared(conn, &sql, libsql::params_from_iter(params)).await?;
  let mut found = false;
  while let Some(row) = rows.next().await? {
    found = true;
    let cmd = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let last_seen = row.get::<i64>(2)?;
    let entry = candidates
      .entry(cmd.clone())
      .or_insert_with(|| Candidate::new(&cmd));
    entry.context_freq = entry.context_freq.max(freq);
    entry.context_cwd_match |= cwd.is_some();
    entry.context_host_match |= hostname.is_some();
    entry.context_user_match |= username.is_some();
    entry.last_seen = entry.last_seen.max(last_seen);
  }
  Ok(found)
}

pub(crate) async fn add_context_candidates(
  conn: &Connection,
  config: &super::SuggestConfig,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  let cwd = config.cwd.as_deref();
  let hostname = config.hostname.as_deref();
  let username = config.username.as_deref();

  if fetch_context_candidates(conn, cwd, hostname, username, candidates).await? {
    return Ok(());
  }

  if hostname.is_some() || username.is_some() {
    if hostname.is_some() && fetch_context_candidates(conn, cwd, hostname, None, candidates).await?
    {
      return Ok(());
    }
    if username.is_some() && fetch_context_candidates(conn, cwd, None, username, candidates).await?
    {
      return Ok(());
    }
    let _ = fetch_context_candidates(conn, cwd, None, None, candidates).await?;
  }

  Ok(())
}

pub(crate) async fn add_session_candidates(
  conn: &Connection,
  session_id: i64,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  let mut rows = query_prepared(
    conn,
    "SELECT expanded_command, COUNT(*) as freq, MAX(COALESCE(start_unix_timestamp, 0)) as last_seen
     FROM shell_history
     WHERE session_id = ?
     GROUP BY expanded_command
     ORDER BY last_seen DESC
     LIMIT 200",
    libsql::params![session_id],
  )
  .await?;
  while let Some(row) = rows.next().await? {
    let cmd = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let last_seen = row.get::<i64>(2)?;
    let entry = candidates
      .entry(cmd.clone())
      .or_insert_with(|| Candidate::new(&cmd));
    entry.session_freq = entry.session_freq.max(freq);
    entry.session_last_seen = entry.session_last_seen.max(last_seen);
    entry.last_seen = entry.last_seen.max(last_seen);
  }
  Ok(())
}

pub(crate) async fn add_repo_candidates(
  conn: &Connection,
  repo_root: &str,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  let mut rows = query_prepared(
    conn,
    "SELECT command, freq, last_seen FROM repo_command_stats
     WHERE repo_root = ?
     ORDER BY freq DESC LIMIT 50",
    libsql::params![repo_root.to_string()],
  )
  .await?;
  while let Some(row) = rows.next().await? {
    let cmd = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let last_seen = row.get::<i64>(2)?;
    let entry = candidates
      .entry(cmd.clone())
      .or_insert_with(|| Candidate::new(&cmd));
    entry.repo_freq = entry.repo_freq.max(freq);
    entry.last_seen = entry.last_seen.max(last_seen);
  }
  Ok(())
}

pub(crate) async fn add_phase_candidates(
  conn: &Connection,
  session_phase: Option<&PhaseSignal>,
  repo_root: &str,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  let session_phase = match session_phase {
    Some(phase) if phase.confidence >= 0.25 => phase,
    _ => return Ok(()),
  };

  let mut rows = query_prepared(
    conn,
    "SELECT command_head, freq, last_seen FROM phase_stats
     WHERE phase = ?
     ORDER BY freq DESC
     LIMIT 30",
    libsql::params![session_phase.phase.clone()],
  )
  .await?;

  let mut seen_heads: HashSet<String> = HashSet::new();
  while let Some(row) = rows.next().await? {
    let head = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let last_seen = row.get::<i64>(2)?;
    if !seen_heads.insert(head.clone()) {
      continue;
    }
    add_head_candidates(conn, &[head], repo_root, candidates).await?;
    for candidate in candidates.values_mut() {
      candidate.context_freq += (freq as f64 * session_phase.confidence) as i64;
      candidate.last_seen = candidate.last_seen.max(last_seen);
    }
  }

  Ok(())
}

pub(crate) async fn add_head_candidates(
  conn: &Connection,
  recent_heads: &[String],
  repo_root: &str,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  for head in recent_heads {
    let like = format!("{head} %");
    if !repo_root.is_empty() {
      let mut rows = query_prepared(
        conn,
        "SELECT command, freq, last_seen FROM repo_command_stats
         WHERE repo_root = ? AND (command = ? OR command LIKE ?)
         ORDER BY freq DESC LIMIT 20",
        libsql::params![repo_root.to_string(), head.clone(), like.clone()],
      )
      .await?;
      while let Some(row) = rows.next().await? {
        let cmd = row.get::<String>(0)?;
        let freq = row.get::<i64>(1)?;
        let last_seen = row.get::<i64>(2)?;
        let entry = candidates
          .entry(cmd.clone())
          .or_insert_with(|| Candidate::new(&cmd));
        entry.repo_freq = entry.repo_freq.max(freq);
        entry.last_seen = entry.last_seen.max(last_seen);
      }
    }

    let mut rows = query_prepared(
      conn,
      "SELECT command, freq, last_seen FROM command_stats
       WHERE command = ? OR command LIKE ?
       ORDER BY freq DESC LIMIT 20",
      libsql::params![head.clone(), like.clone()],
    )
    .await?;
    while let Some(row) = rows.next().await? {
      let cmd = row.get::<String>(0)?;
      let freq = row.get::<i64>(1)?;
      let last_seen = row.get::<i64>(2)?;
      let entry = candidates
        .entry(cmd.clone())
        .or_insert_with(|| Candidate::new(&cmd));
      entry.freq = entry.freq.max(freq);
      entry.last_seen = entry.last_seen.max(last_seen);
    }
  }

  Ok(())
}

pub(crate) async fn add_global_candidates(
  conn: &Connection,
  candidates: &mut HashMap<String, Candidate>,
  limit: usize,
) -> Result<()> {
  let mut rows = query_prepared(
    conn,
    "SELECT command, freq, last_seen FROM command_stats ORDER BY freq DESC LIMIT ?",
    libsql::params![limit as i64],
  )
  .await?;
  while let Some(row) = rows.next().await? {
    let cmd = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let last_seen = row.get::<i64>(2)?;
    let entry = candidates
      .entry(cmd.clone())
      .or_insert_with(|| Candidate::new(&cmd));
    entry.freq = entry.freq.max(freq);
    entry.last_seen = entry.last_seen.max(last_seen);
  }
  Ok(())
}

pub(crate) async fn add_recent_candidates(
  conn: &Connection,
  candidates: &mut HashMap<String, Candidate>,
  limit: usize,
) -> Result<()> {
  let mut rows = query_prepared(
    conn,
    "SELECT expanded_command, COUNT(*) as freq, MAX(COALESCE(start_unix_timestamp, 0)) as last_seen
     FROM shell_history
     GROUP BY expanded_command
     ORDER BY last_seen DESC
     LIMIT ?",
    libsql::params![limit as i64],
  )
  .await?;
  while let Some(row) = rows.next().await? {
    let cmd = row.get::<String>(0)?;
    let freq = row.get::<i64>(1)?;
    let last_seen = row.get::<i64>(2)?;
    let entry = candidates
      .entry(cmd.clone())
      .or_insert_with(|| Candidate::new(&cmd));
    entry.freq = entry.freq.max(freq);
    entry.last_seen = entry.last_seen.max(last_seen);
  }
  Ok(())
}

pub(crate) async fn hydrate_candidate_stats(
  conn: &Connection,
  repo_root: &str,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  if candidates.is_empty() {
    return Ok(());
  }

  let commands: Vec<String> = candidates.keys().cloned().collect();
  let chunk_size = 200usize;

  for chunk in commands.chunks(chunk_size) {
    let mut placeholders = String::new();
    for idx in 0..chunk.len() {
      if idx > 0 {
        placeholders.push(',');
      }
      placeholders.push('?');
    }
    let sql = format!(
      "SELECT command, freq, last_seen FROM command_stats WHERE command IN ({})",
      placeholders
    );
    let params = chunk
      .iter()
      .map(|cmd| Value::from(cmd.clone()))
      .collect::<Vec<_>>();
    let mut rows = query_prepared(conn, &sql, libsql::params_from_iter(params)).await?;
    while let Some(row) = rows.next().await? {
      let cmd = row.get::<String>(0)?;
      let freq = row.get::<i64>(1)?;
      let last_seen = row.get::<i64>(2)?;
      let entry = candidates
        .entry(cmd.clone())
        .or_insert_with(|| Candidate::new(&cmd));
      entry.freq = entry.freq.max(freq);
      entry.last_seen = entry.last_seen.max(last_seen);
    }
  }

  if !repo_root.is_empty() {
    for chunk in commands.chunks(chunk_size) {
      let mut placeholders = String::new();
      for idx in 0..chunk.len() {
        if idx > 0 {
          placeholders.push(',');
        }
        placeholders.push('?');
      }
      let sql = format!(
        "SELECT command, freq, last_seen FROM repo_command_stats
         WHERE repo_root = ? AND command IN ({})",
        placeholders
      );
      let mut params: Vec<Value> = Vec::with_capacity(chunk.len() + 1);
      params.push(Value::from(repo_root.to_string()));
      for cmd in chunk {
        params.push(Value::from(cmd.clone()));
      }
      let mut rows = query_prepared(conn, &sql, libsql::params_from_iter(params)).await?;
      while let Some(row) = rows.next().await? {
        let cmd = row.get::<String>(0)?;
        let freq = row.get::<i64>(1)?;
        let last_seen = row.get::<i64>(2)?;
        let entry = candidates
          .entry(cmd.clone())
          .or_insert_with(|| Candidate::new(&cmd));
        entry.repo_freq = entry.repo_freq.max(freq);
        entry.last_seen = entry.last_seen.max(last_seen);
      }
    }
  }

  Ok(())
}

pub(crate) async fn add_sequence_candidates(
  conn: &Connection,
  recent_commands: &[String],
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  let sequences = candidates_from_sequences(conn, recent_commands, 200).await?;
  for seq in sequences {
    let entry = candidates
      .entry(seq.command.clone())
      .or_insert_with(|| Candidate::new(&seq.command));
    if seq.prefix_len >= entry.sequence_prefix_len {
      entry.sequence_confidence = entry.sequence_confidence.max(seq.confidence);
      entry.sequence_lift = entry.sequence_lift.max(seq.lift);
      entry.sequence_prefix_len = entry.sequence_prefix_len.max(seq.prefix_len);
    }
  }
  Ok(())
}

#[derive(Debug)]
struct TemplateStat {
  value: String,
  freq: i64,
  last_seen: i64,
}

pub(crate) async fn add_template_candidates(
  conn: &Connection,
  repo_root: &str,
  candidates: &mut HashMap<String, Candidate>,
) -> Result<()> {
  let mut heads: HashSet<String> = HashSet::new();
  for cmd in candidates.keys() {
    if let Some(head) = command_head_for_phase("sh", cmd) {
      heads.insert(head);
    }
  }
  if heads.is_empty() {
    return Ok(());
  }

  let mut added = 0usize;
  for head in heads {
    let (flags, flags_repo) = fetch_template_flags(conn, repo_root, &head, 3).await?;
    let (args0, args0_repo) = fetch_template_args(conn, repo_root, &head, 0, 3).await?;
    let (args1, args1_repo) = fetch_template_args(conn, repo_root, &head, 1, 2).await?;

    let mut base = head.clone();
    let mut flags_freq = 0i64;
    let mut flags_last_seen = 0i64;
    if !flags.is_empty() {
      base.push(' ');
      base.push_str(
        &flags
          .iter()
          .map(|stat| stat.value.clone())
          .collect::<Vec<_>>()
          .join(" "),
      );
      for stat in &flags {
        flags_freq += stat.freq;
        flags_last_seen = flags_last_seen.max(stat.last_seen);
      }
      add_template_candidate(candidates, &base, flags_freq, flags_last_seen, flags_repo);
    }

    for arg0 in &args0 {
      let mut cmd = base.clone();
      if !cmd.is_empty() {
        cmd.push(' ');
      }
      cmd.push_str(&arg0.value);
      let freq = flags_freq + arg0.freq;
      let last_seen = flags_last_seen.max(arg0.last_seen);
      add_template_candidate(candidates, &cmd, freq, last_seen, flags_repo || args0_repo);

      for arg1 in &args1 {
        let mut cmd = cmd.clone();
        cmd.push(' ');
        cmd.push_str(&arg1.value);
        let freq = freq + arg1.freq;
        let last_seen = last_seen.max(arg1.last_seen);
        add_template_candidate(
          candidates,
          &cmd,
          freq,
          last_seen,
          flags_repo || args0_repo || args1_repo,
        );
        added += 1;
        if added > 50 {
          return Ok(());
        }
      }
    }
  }

  Ok(())
}

fn add_template_candidate(
  candidates: &mut HashMap<String, Candidate>,
  command: &str,
  freq: i64,
  last_seen: i64,
  is_repo: bool,
) {
  if candidates.contains_key(command) {
    return;
  }
  let mut entry = Candidate::new(command);
  if is_repo {
    entry.repo_freq = freq;
  } else {
    entry.freq = freq;
  }
  entry.last_seen = last_seen;
  candidates.insert(command.to_string(), entry);
}

async fn fetch_template_flags(
  conn: &Connection,
  repo_root: &str,
  head: &str,
  limit: usize,
) -> Result<(Vec<TemplateStat>, bool)> {
  let repo_flags = fetch_flag_stats(conn, repo_root, head, limit).await?;
  if !repo_flags.is_empty() {
    return Ok((repo_flags, true));
  }
  let global_flags = fetch_flag_stats(conn, "", head, limit).await?;
  Ok((global_flags, false))
}

async fn fetch_flag_stats(
  conn: &Connection,
  repo_root: &str,
  head: &str,
  limit: usize,
) -> Result<Vec<TemplateStat>> {
  let mut rows = query_prepared(
    conn,
    "SELECT flag_raw, freq, last_seen FROM flag_stats
     WHERE repo_root = ? AND command_head = ?
     ORDER BY freq DESC LIMIT ?",
    libsql::params![repo_root.to_string(), head.to_string(), limit as i64],
  )
  .await?;
  let mut stats = Vec::new();
  while let Some(row) = rows.next().await? {
    stats.push(TemplateStat {
      value: row.get::<String>(0)?,
      freq: row.get::<i64>(1)?,
      last_seen: row.get::<i64>(2)?,
    });
  }
  Ok(stats)
}

async fn fetch_template_args(
  conn: &Connection,
  repo_root: &str,
  head: &str,
  index: i64,
  limit: usize,
) -> Result<(Vec<TemplateStat>, bool)> {
  let repo_args = fetch_arg_stats(conn, repo_root, head, index, limit).await?;
  if !repo_args.is_empty() {
    return Ok((repo_args, true));
  }
  let global_args = fetch_arg_stats(conn, "", head, index, limit).await?;
  if !global_args.is_empty() {
    return Ok((global_args, false));
  }
  let repo_any = fetch_arg_stats_any(conn, repo_root, head, limit).await?;
  if !repo_any.is_empty() {
    return Ok((repo_any, true));
  }
  let global_any = fetch_arg_stats_any(conn, "", head, limit).await?;
  Ok((global_any, false))
}

async fn fetch_arg_stats(
  conn: &Connection,
  repo_root: &str,
  head: &str,
  index: i64,
  limit: usize,
) -> Result<Vec<TemplateStat>> {
  let mut rows = query_prepared(
    conn,
    "SELECT arg_raw, freq, last_seen FROM arg_stats
     WHERE repo_root = ? AND command_head = ? AND arg_index = ?
     ORDER BY freq DESC LIMIT ?",
    libsql::params![repo_root.to_string(), head.to_string(), index, limit as i64],
  )
  .await?;
  let mut stats = Vec::new();
  while let Some(row) = rows.next().await? {
    stats.push(TemplateStat {
      value: row.get::<String>(0)?,
      freq: row.get::<i64>(1)?,
      last_seen: row.get::<i64>(2)?,
    });
  }
  Ok(stats)
}

async fn fetch_arg_stats_any(
  conn: &Connection,
  repo_root: &str,
  head: &str,
  limit: usize,
) -> Result<Vec<TemplateStat>> {
  let mut rows = query_prepared(
    conn,
    "SELECT arg_raw, freq, last_seen FROM arg_stats_any
     WHERE repo_root = ? AND command_head = ?
     ORDER BY freq DESC LIMIT ?",
    libsql::params![repo_root.to_string(), head.to_string(), limit as i64],
  )
  .await?;
  let mut stats = Vec::new();
  while let Some(row) = rows.next().await? {
    stats.push(TemplateStat {
      value: row.get::<String>(0)?,
      freq: row.get::<i64>(1)?,
      last_seen: row.get::<i64>(2)?,
    });
  }
  Ok(stats)
}

pub(crate) async fn load_session_stats(
  conn: &Connection,
  session_id: i64,
  recent_limit: usize,
) -> Result<HashMap<String, (i64, i64)>> {
  let mut rows = query_prepared(
    conn,
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

pub(crate) fn push_opt_string(params: &mut Vec<Value>, value: &Option<String>) {
  if let Some(val) = value {
    params.push(Value::from(val.clone()));
  } else {
    params.push(Value::Null);
  }
}

pub(crate) fn push_opt_i64(params: &mut Vec<Value>, value: Option<i64>) {
  if let Some(val) = value {
    params.push(Value::from(val));
  } else {
    params.push(Value::Null);
  }
}
