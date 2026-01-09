use std::collections::HashSet;

use color_eyre::eyre::Result;

use crate::cli::CompletionFormat;
use crate::db::Db;
use crate::predict::{ScoreBreakdown, SuggestConfig, Suggestion, suggest};
use crate::server::{self, Request, Response};
use crate::shell_history::get_hostname;
use crate::tokenize::tokenize;

pub async fn run(
  db: &Db,
  count: usize,
  current_line: Option<String>,
  recent_limit: usize,
  cwd: Option<String>,
  hostname: Option<String>,
  username: Option<String>,
  session_id: Option<i64>,
  no_sequences: bool,
  completion_format: CompletionFormat,
  show_scores: bool,
  autosuggest: bool,
) -> Result<()> {
  let cwd = match cwd {
    Some(val) => Some(val),
    None => std::env::current_dir()
      .ok()
      .and_then(|p| p.to_str().map(|s| s.to_string())),
  };

  let hostname = hostname.or_else(|| Some(get_hostname()));
  let username = username
    .or_else(|| uzers::get_current_username().map(|v| v.to_string_lossy().into_owned()));

  let has_prefix = current_line
    .as_ref()
    .map(|s| !s.is_empty())
    .unwrap_or(false);

  let mut server_suggestions = None;
  let request = Request::Suggest {
    current_line: current_line.clone().unwrap_or_default(),
    working_directory: cwd.clone().unwrap_or_else(|| "".to_string()),
    session_id: session_id.unwrap_or_default() as u64,
    limit: count as u32,
    prefer_full_line: autosuggest,
  };
  if let Ok(Some(response)) = server::try_request(request).await
    && let Response::Suggestions { items } = response
  {
    server_suggestions = Some(map_server_suggestions(items));
  }

  if has_prefix {
    let base_config = SuggestConfig {
      max_results: count,
      recent_limit,
      prefix: current_line.clone(),
      cwd: cwd.clone(),
      hostname: hostname.clone(),
      username: username.clone(),
      session_id,
      use_sequences: !no_sequences,
      prefer_full_line: autosuggest,
    };

    let completions = if let Some(items) = server_suggestions {
      items
    } else {
      suggest(&db.conn, base_config).await?
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
    let config = SuggestConfig {
      max_results: count,
      recent_limit,
      prefix: None,
      cwd,
      hostname,
      username,
      session_id,
      use_sequences: !no_sequences,
      prefer_full_line: autosuggest,
    };

    let suggestions = if let Some(items) = server_suggestions {
      items
    } else {
      suggest(&db.conn, config).await?
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
