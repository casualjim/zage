use std::collections::HashSet;
use std::env;
use std::fs;
use std::time::Duration;

use color_eyre::eyre::{Result, eyre};

use crate::cli::{BackendRef, CompletionFormat, SuggestArgs};
use crate::predict::{ScoreBreakdown, SuggestConfig, Suggestion, suggest};
use crate::server::{self, Request, Response};
use crate::shell_history::{get_hostname, normalize_shellname};
use crate::tokenize::tokenize;

const DEFAULT_AUTOSUGGEST_TIMEOUT_MS: u64 = 150;

pub async fn run(backend: BackendRef<'_>, args: SuggestArgs) -> Result<()> {
  let SuggestArgs {
    count,
    current_line,
    recent_limit,
    cwd,
    hostname,
    username,
    session_id,
    shellname,
    no_sequences,
    completion_format,
    show_scores,
    show_breakdown,
    autosuggest,
    timeout,
  } = args;

  let current_line = current_line.filter(|value| !value.is_empty());

  let cwd = match cwd {
    Some(val) => Some(val),
    None => std::env::current_dir()
      .ok()
      .and_then(|p| p.to_str().map(|s| s.to_string())),
  };

  let session_id = session_id.or_else(|| {
    env::var("ZAGE_SESSION_ID")
      .ok()
      .and_then(|val| val.parse::<i64>().ok())
  });

  let hostname = hostname.or_else(|| Some(get_hostname()));
  let username =
    username.or_else(|| uzers::get_current_username().map(|v| v.to_string_lossy().into_owned()));

  let shellname = normalize_shellname(&shellname);

  let aliases = env::var("ZAGE_ALIASES")
    .ok()
    .filter(|value| !value.trim().is_empty())
    .or_else(|| {
      env::var("ZAGE_ALIAS_FILE")
        .ok()
        .and_then(|path| fs::read_to_string(path).ok())
        .filter(|value| !value.trim().is_empty())
    });

  let has_prefix = current_line
    .as_ref()
    .map(|s| !s.is_empty())
    .unwrap_or(false);

  let prefix = current_line.as_ref().filter(|value| !value.is_empty());
  let server_suggestions = match &backend {
    BackendRef::Server => {
      let timeout_ms = resolve_timeout_ms(timeout, autosuggest);
      let request = Request::Suggest {
        current_line: prefix.cloned(),
        working_directory: cwd.clone(),
        hostname: hostname.clone(),
        username: username.clone(),
        session_id,
        shellname: Some(shellname.clone()),
        aliases: aliases.clone(),
        limit: count as u32,
        recent_limit,
        use_sequences: !no_sequences,
        prefer_full_line: autosuggest,
        include_debug: show_breakdown,
        timeout_ms,
      };
      match server::try_request(request).await? {
        Some(Response::Suggestions { items }) => Some(map_server_suggestions(items)),
        Some(Response::Error { message }) => return Err(eyre!(message)),
        Some(_) => return Err(eyre!("Unexpected response from server")),
        None => return Err(eyre!("Suggest server unavailable")),
      }
    }
    BackendRef::Embedded(_) => None,
  };

  if has_prefix {
    let completions = match backend {
      BackendRef::Server => {
        server_suggestions.ok_or_else(|| eyre!("Suggest server unavailable"))?
      }
      BackendRef::Embedded(db) => {
        let base_config = SuggestConfig {
          max_results: count,
          recent_limit,
          prefix: prefix.cloned(),
          cwd: cwd.clone(),
          hostname: hostname.clone(),
          username: username.clone(),
          session_id,
          shellname: Some(shellname.clone()),
          use_sequences: !no_sequences,
          prefer_full_line: autosuggest,
          include_debug: show_breakdown,
        };
        suggest(&db.conn, base_config).await?
      }
    };
    if completions.is_empty() {
      return Ok(());
    }

    if autosuggest {
      if let Some(first) = completions.first() {
        println!("{}", first.command);
      }
      return Ok(());
    }

    let prefix_str = current_line.unwrap_or_default();
    let prefix_tokens = tokenize(&prefix_str);
    let ends_with_space = prefix_str
      .chars()
      .last()
      .map(|c| c.is_whitespace())
      .unwrap_or(false);
    let target_index = if prefix_tokens.is_empty() {
      0
    } else if ends_with_space {
      prefix_tokens.len()
    } else {
      prefix_tokens.len() - 1
    };

    let mut seen = HashSet::new();
    for suggestion in completions {
      let candidate_tokens = tokenize(&suggestion.command);
      if let Some(tok) = candidate_tokens.get(target_index)
        && seen.insert(tok.raw.clone())
      {
        match completion_format {
          CompletionFormat::Plain => {
            if show_breakdown {
              println!(
                "{}\t{}",
                tok.raw,
                format_suggestion_debug(&suggestion, show_scores)
              );
            } else if show_scores {
              println!("{}\t{:.4}", tok.raw, suggestion.score);
            } else {
              println!("{}", tok.raw);
            }
          }
          CompletionFormat::Zsh => {
            let desc = if show_breakdown {
              Some(format_suggestion_debug(&suggestion, show_scores))
            } else if show_scores {
              Some(format!("{:.4}", suggestion.score))
            } else {
              None
            };
            println!("{}", format_zsh_item(&tok.raw, desc.as_deref()));
          }
        }
      }
    }
  } else {
    let suggestions = match backend {
      BackendRef::Server => {
        server_suggestions.ok_or_else(|| eyre!("Suggest server unavailable"))?
      }
      BackendRef::Embedded(db) => {
        let config = SuggestConfig {
          max_results: count,
          recent_limit,
          prefix: None,
          cwd,
          hostname,
          username,
          session_id,
          shellname: Some(shellname.clone()),
          use_sequences: !no_sequences,
          prefer_full_line: autosuggest,
          include_debug: show_breakdown,
        };
        suggest(&db.conn, config).await?
      }
    };
    if autosuggest {
      if let Some(first) = suggestions.first() {
        println!("{}", first.command);
      }
      return Ok(());
    }
    for suggestion in suggestions {
      match completion_format {
        CompletionFormat::Plain => {
          if show_breakdown {
            println!(
              "{}\t{}",
              suggestion.command,
              format_suggestion_debug(&suggestion, show_scores)
            );
          } else if show_scores {
            println!("{}\t{:.4}", suggestion.command, suggestion.score);
          } else {
            println!("{}", suggestion.command);
          }
        }
        CompletionFormat::Zsh => {
          let desc = if show_breakdown {
            Some(format_suggestion_debug(&suggestion, show_scores))
          } else if show_scores {
            Some(format!("{:.4}", suggestion.score))
          } else {
            None
          };
          println!("{}", format_zsh_item(&suggestion.command, desc.as_deref()));
        }
      }
    }
  }

  Ok(())
}

fn resolve_timeout_ms(timeout: Option<humantime::Duration>, autosuggest: bool) -> Option<u64> {
  timeout
    .map(|duration| Duration::from(duration).as_millis())
    .map(|millis| u64::try_from(millis).unwrap_or(u64::MAX))
    .or_else(|| autosuggest.then_some(DEFAULT_AUTOSUGGEST_TIMEOUT_MS))
}

fn format_zsh_item(word: &str, desc: Option<&str>) -> String {
  let mut escaped = String::new();
  for ch in word.chars() {
    match ch {
      '\\' => escaped.push_str("\\\\"),
      ':' => escaped.push_str("\\:"),
      _ => escaped.push(ch),
    }
  }
  match desc {
    Some(d) => {
      let mut d_esc = String::new();
      for ch in d.chars() {
        match ch {
          '\\' => d_esc.push_str("\\\\"),
          ':' => d_esc.push_str("\\:"),
          _ => d_esc.push(ch),
        }
      }
      format!("{escaped}:{d_esc}")
    }
    None => format!("{escaped}:"),
  }
}

fn map_server_suggestions(items: Vec<server::Suggestion>) -> Vec<Suggestion> {
  items
    .into_iter()
    .map(|item| Suggestion {
      command: item.command,
      score: item.score as f64,
      breakdown: ScoreBreakdown {
        recency: item.breakdown.recency as f64,
        session_recency: item.breakdown.session_recency as f64,
        frequency: item.breakdown.frequency as f64,
        transition: item.breakdown.transition as f64,
        context: item.breakdown.context as f64,
        sequence: item.breakdown.sequence as f64,
        similarity: item.breakdown.similarity as f64,
        embedding_retrieval: item.breakdown.embedding_retrieval as f64,
        online_model: item.breakdown.online_model as f64,
      },
      debug: item.debug.map(|debug| crate::core::SuggestionDebug {
        blend: crate::core::BlendDebug {
          model_gate: debug.blend.model_gate as f64,
          model_alpha: debug.blend.model_alpha as f64,
          blend_model_weight: debug.blend.blend_model_weight as f64,
          blend_frecency_weight: debug.blend.blend_frecency_weight as f64,
          blend_sequence_weight: debug.blend.blend_sequence_weight as f64,
          blend_tier1_weight: debug.blend.blend_tier1_weight as f64,
          model_feature: debug.blend.model_feature as f64,
          frecency_feature: debug.blend.frecency_feature as f64,
          sequence_feature: debug.blend.sequence_feature as f64,
          tier1_feature: debug.blend.tier1_feature as f64,
          model_contrib: debug.blend.model_contrib as f64,
          frecency_contrib: debug.blend.frecency_contrib as f64,
          sequence_contrib: debug.blend.sequence_contrib as f64,
          tier1_contrib: debug.blend.tier1_contrib as f64,
        },
        candidate: crate::core::CandidateDebug {
          freq: debug.candidate.freq,
          workspace_freq: debug.candidate.workspace_freq,
          last_seen: debug.candidate.last_seen,
          transition_freq: debug.candidate.transition_freq,
          workspace_transition_freq: debug.candidate.workspace_transition_freq,
          transition_exit_status_match: debug.candidate.transition_exit_status_match,
          context_freq: debug.candidate.context_freq,
          context_cwd_match: debug.candidate.context_cwd_match,
          context_host_match: debug.candidate.context_host_match,
          context_user_match: debug.candidate.context_user_match,
          session_freq: debug.candidate.session_freq,
          session_last_seen: debug.candidate.session_last_seen,
          from_embedding: debug.candidate.from_embedding,
          sequence_confidence: debug.candidate.sequence_confidence as f64,
          sequence_lift: debug.candidate.sequence_lift as f64,
          sequence_prefix_len: debug.candidate.sequence_prefix_len as usize,
        },
        pipeline: crate::core::PipelineDebug {
          added_transition: debug.pipeline.added_transition as usize,
          added_session: debug.pipeline.added_session as usize,
          added_embedding: debug.pipeline.added_embedding as usize,
          added_context: debug.pipeline.added_context as usize,
          added_workspace: debug.pipeline.added_workspace as usize,
          added_head: debug.pipeline.added_head as usize,
          added_sequence: debug.pipeline.added_sequence as usize,
          added_template: debug.pipeline.added_template as usize,
          added_recent: debug.pipeline.added_recent as usize,
          added_global: debug.pipeline.added_global as usize,
          total_candidates: debug.pipeline.total_candidates as usize,
          conditional_candidates: debug.pipeline.conditional_candidates as usize,
          pruned_before: debug.pipeline.pruned_before as usize,
          pruned_after: debug.pipeline.pruned_after as usize,
          pruned_kept_conditional: debug.pipeline.pruned_kept_conditional as usize,
        },
      }),
    })
    .collect()
}

fn format_suggestion_debug(suggestion: &Suggestion, always_show_score: bool) -> String {
  let score = if always_show_score {
    format!("score={:.4}", suggestion.score)
  } else {
    format!("{:.4}", suggestion.score)
  };

  if let Some(debug) = suggestion.debug.as_ref() {
    let b = &debug.blend;
    let c = &debug.candidate;
    let p = &debug.pipeline;
    let (dom, dom_val) = {
      let mut best = ("m", b.model_contrib.abs());
      for cand in [
        ("f", b.frecency_contrib.abs()),
        ("s", b.sequence_contrib.abs()),
        ("t", b.tier1_contrib.abs()),
      ] {
        if cand.1 > best.1 {
          best = cand;
        }
      }
      best
    };
    return format!(
      "{score} dom={dom}={dom_val:.3} contrib[m={:.3} f={:.3} s={:.3} t={:.3}] gate={:.2} a={:.2} w[m={:.2} f={:.2} s={:.2} t={:.2}] feat[mdl={:.3} fre={:.3} seq={:.3} t1={:.3}] cand[tr={} wtr={} ctx={} freq={} wf={} seqc={:.3} lift={:.2} pref={} emb={}] pool[total={} cond={} add[tr={} ctx={} ws={} head={} seq={} tpl={} rec={} glb={} emb={} ses={}] prune[{}→{} keep_cond={}]",
      b.model_contrib,
      b.frecency_contrib,
      b.sequence_contrib,
      b.tier1_contrib,
      b.model_gate,
      b.model_alpha,
      b.blend_model_weight,
      b.blend_frecency_weight,
      b.blend_sequence_weight,
      b.blend_tier1_weight,
      b.model_feature,
      b.frecency_feature,
      b.sequence_feature,
      b.tier1_feature,
      c.transition_freq,
      c.workspace_transition_freq,
      c.context_freq,
      c.freq,
      c.workspace_freq,
      c.sequence_confidence,
      c.sequence_lift,
      c.sequence_prefix_len,
      if c.from_embedding { 1 } else { 0 },
      p.total_candidates,
      p.conditional_candidates,
      p.added_transition,
      p.added_context,
      p.added_workspace,
      p.added_head,
      p.added_sequence,
      p.added_template,
      p.added_recent,
      p.added_global,
      p.added_embedding,
      p.added_session,
      p.pruned_before,
      p.pruned_after,
      p.pruned_kept_conditional
    );
  }

  format!(
    "{score} feats[rec={:.3} freq={:.3} tr={:.3} ctx={:.3} seq={:.3} sim={:.3} mdl={:.3}]",
    suggestion.breakdown.recency,
    suggestion.breakdown.frequency,
    suggestion.breakdown.transition,
    suggestion.breakdown.context,
    suggestion.breakdown.sequence,
    suggestion.breakdown.similarity,
    suggestion.breakdown.online_model
  )
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn autosuggest_uses_short_default_timeout() {
    assert_eq!(
      resolve_timeout_ms(None, true),
      Some(DEFAULT_AUTOSUGGEST_TIMEOUT_MS)
    );
  }

  #[test]
  fn explicit_timeout_overrides_autosuggest_default() {
    assert_eq!(
      resolve_timeout_ms(
        Some(humantime::Duration::from(Duration::from_secs(2))),
        true
      ),
      Some(2_000)
    );
  }

  #[test]
  fn non_autosuggest_keeps_existing_default() {
    assert_eq!(resolve_timeout_ms(None, false), None);
  }
}
