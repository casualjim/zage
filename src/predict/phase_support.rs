use std::collections::{HashMap, HashSet};

use libsql::{Connection, Value};

use crate::Result;
use crate::phase::{PhaseConfig, detect_phase_from_commands};
use crate::tokenize::{extract_command_parts, tokenize_index};

use super::sql::query_prepared;

#[derive(Clone, Debug)]
pub(crate) struct PhaseSignal {
  pub(crate) phase: String,
  pub(crate) confidence: f64,
}

pub(crate) fn command_head_for_phase(shellname: &str, command: &str) -> Option<String> {
  let tokens = tokenize_index(shellname, command);
  if let Some(parts) = extract_command_parts(command, &tokens) {
    return Some(parts.head);
  }
  tokens.first().map(|token| token.raw.clone())
}

pub(crate) async fn load_phase_for_heads(
  conn: &Connection,
  heads: &HashSet<String>,
) -> Result<HashMap<String, PhaseSignal>> {
  if heads.is_empty() {
    return Ok(HashMap::new());
  }

  let mut placeholders = String::new();
  for idx in 0..heads.len() {
    if idx > 0 {
      placeholders.push(',');
    }
    placeholders.push('?');
  }
  let mut params: Vec<Value> = Vec::with_capacity(heads.len());
  for head in heads {
    params.push(Value::from(head.clone()));
  }

  let sql = format!(
    "SELECT command_head, phase, confidence, freq
     FROM phase_stats
     WHERE command_head IN ({})",
    placeholders
  );
  let mut rows = query_prepared(conn, &sql, params).await?;

  let mut map: HashMap<String, PhaseSignal> = HashMap::new();
  while let Some(row) = rows.next().await? {
    let head = row.get::<String>(0)?;
    let phase = row.get::<String>(1)?;
    let confidence = row.get::<f64>(2)?;
    let freq = row.get::<i64>(3)?;
    let score = confidence * (freq as f64).ln_1p();
    match map.get(&head) {
      Some(existing) if existing.confidence >= score => {}
      _ => {
        map.insert(
          head,
          PhaseSignal {
            phase,
            confidence: score,
          },
        );
      }
    }
  }
  Ok(map)
}

pub(crate) fn detect_session_phase(
  recent_heads: &[String],
  phase_for_head: &HashMap<String, PhaseSignal>,
) -> Option<PhaseSignal> {
  let mut scores: HashMap<String, f64> = HashMap::new();
  let mut total = 0.0f64;
  for (idx, head) in recent_heads.iter().rev().take(6).enumerate() {
    if let Some(phase) = phase_for_head.get(head) {
      let weight = 0.5_f64.powi(idx as i32);
      let confidence = phase.confidence.min(1.0);
      let score = confidence * weight;
      *scores.entry(phase.phase.clone()).or_insert(0.0) += score;
      total += score;
    }
  }
  let (phase, score) = scores
    .into_iter()
    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))?;
  let confidence = if total > 0.0 { score / total } else { 0.0 };
  Some(PhaseSignal { phase, confidence })
}

pub(crate) fn detect_session_phase_from_commands(
  recent_commands: &[String],
  phase_config: &PhaseConfig,
) -> Option<PhaseSignal> {
  detect_phase_from_commands(recent_commands, phase_config)
    .map(|(phase, confidence)| PhaseSignal { phase, confidence })
}

pub(crate) fn phase_match_boost(
  session: Option<&PhaseSignal>,
  candidate: Option<&PhaseSignal>,
) -> f64 {
  let (session, candidate) = match (session, candidate) {
    (Some(session), Some(candidate)) => (session, candidate),
    _ => return 0.0,
  };
  if session.phase != candidate.phase {
    return 0.0;
  }
  let session_confidence = session.confidence.min(1.0);
  let candidate_confidence = candidate.confidence.min(1.0);
  6.0 * session_confidence * candidate_confidence
}
