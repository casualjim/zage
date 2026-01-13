use std::collections::HashSet;
use std::env;
use std::time::Duration;

use color_eyre::eyre::{Result, eyre};

use crate::cli::{BackendRef, CompletionFormat, SuggestArgs};
use crate::predict::{ScoreBreakdown, SuggestConfig, Suggestion, suggest};
use crate::server::{self, Request, Response};
use crate::shell_history::{get_hostname, normalize_shellname};
use crate::tokenize::tokenize;

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
    autosuggest,
    timeout,
  } = args;

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

  let shellname = shellname
    .as_deref()
    .map(normalize_shellname)
    .unwrap_or_else(|| "sh".to_string());

  let has_prefix = current_line
    .as_ref()
    .map(|s| !s.is_empty())
    .unwrap_or(false);

  let prefix = current_line.as_ref().filter(|value| !value.is_empty());
  let server_suggestions = match &backend {
    BackendRef::Server => {
      let timeout_ms = timeout
        .map(|duration| Duration::from(duration).as_millis())
        .map(|millis| u64::try_from(millis).unwrap_or(u64::MAX));
      let request = Request::Suggest {
        current_line: prefix.cloned(),
        working_directory: cwd.clone(),
        hostname: hostname.clone(),
        username: username.clone(),
        session_id,
        shellname: Some(shellname.clone()),
        limit: count as u32,
        recent_limit,
        use_sequences: !no_sequences,
        prefer_full_line: autosuggest,
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
            if show_scores {
              println!("{}\t{:.4}", tok.raw, suggestion.score);
            } else {
              println!("{}", tok.raw);
            }
          }
          CompletionFormat::Zsh => {
            let desc = if show_scores {
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
          if show_scores {
            println!("{}\t{:.4}", suggestion.command, suggestion.score);
          } else {
            println!("{}", suggestion.command);
          }
        }
        CompletionFormat::Zsh => {
          let desc = if show_scores {
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
      breakdown: ScoreBreakdown::default(),
    })
    .collect()
}
