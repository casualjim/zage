use std::env;
use std::path::PathBuf;

use clap::{CommandFactory, Parser, Subcommand};
use color_eyre::eyre::{Result, eyre};
use dirs::home_dir;
use human_panic::setup_panic;
use tracing::debug;
use tracing::info;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use zage::db::{import_history, init, insert_invocation, open_db, update_stats_for_invocation};
use zage::indexer::rebuild_stats;
use zage::predict::aliases::{expand_alias, load_aliases};
use zage::predict::{ScoreBreakdown, SuggestConfig, Suggestion, suggest};
use zage::rerank::{TrainConfig, model_status, reset_model, train_model};
use zage::sequence::{SequenceConfig, analyze_sequences, analyze_token_sequences};
use zage::server::{self, Request, Response};
use zage::service;
use zage::shell_history::{
  Invocation, Shell, detect_shellname, get_hostname, parse_bash_history, parse_zsh_history,
};
use zage::tokenize::tokenize;

/// CLI for Zage
#[derive(Parser)]
#[clap(author, version, about)]
struct Cli {
  #[command(subcommand)]
  command: Option<Commands>,
  /// Path to the SQLite database file (overrides default location)
  #[clap(long, env = "ZAGE_DB_PATH")]
  db_path: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Commands {
  /// Import shell history
  Import {
    /// Path to history file (defaults to $HISTFILE env var)
    #[arg(value_name = "FILE", env = "HISTFILE")]
    file: Option<PathBuf>,
    /// Override hostname for import
    #[arg(long)]
    hostname: Option<String>,
    /// Override username for import
    #[arg(long)]
    username: Option<String>,
    /// Shell type (bash or zsh); defaults to $SHELL env var basename
    #[arg(long, env = "SHELL")]
    shell: Shell,
    /// Skip rebuilding stats and sequences after import (for bulk imports)
    #[arg(long)]
    no_index: bool,
  },

  /// Build or rebuild index tables for prediction
  Index {
    /// Maximum number of commands to index (oldest first)
    #[arg(long)]
    max_commands: Option<usize>,

    /// Also recompute sequence statistics
    #[arg(long)]
    with_sequences: bool,
  },

  /// Suggest next command
  Suggest {
    /// Number of suggestions to return
    #[arg(short, long, default_value = "5")]
    count: usize,

    /// Current input line (prefix)
    #[arg(long)]
    current_line: Option<String>,

    /// Number of recent commands to consider as context
    #[arg(long, default_value = "10")]
    recent_limit: usize,

    /// Override current working directory
    #[arg(long)]
    cwd: Option<String>,

    /// Override hostname
    #[arg(long)]
    hostname: Option<String>,

    /// Override username
    #[arg(long)]
    username: Option<String>,

    /// Override session id
    #[arg(long, env = "ZAGE_SESSION_ID")]
    session_id: Option<i64>,

    /// Disable sequence-based candidates
    #[arg(long)]
    no_sequences: bool,

    /// Output format for completion (plain or zsh)
    #[arg(long, env = "ZAGE_COMPLETION_FORMAT", default_value = "plain")]
    completion_format: CompletionFormat,

    /// Show per-suggestion scores
    #[arg(long)]
    show_scores: bool,

    /// Return full-line suggestions for autosuggest backends
    #[arg(long)]
    autosuggest: bool,
  },

  /// Analyze and store frequent command sequences
  AnalyzeSequences {
    /// Minimum support count
    #[arg(long, default_value = "2")]
    min_support: usize,
    /// Minimum confidence threshold
    #[arg(long, default_value = "0.5")]
    min_confidence: f64,
    /// Minimum lift threshold
    #[arg(long, default_value = "1.2")]
    min_lift: f64,
    /// Maximum sequence length (2 or 3)
    #[arg(long, default_value = "3")]
    max_len: usize,
  },

  /// Run the suggestion server (foreground)
  Server {},

  /// Install or uninstall the background suggestion service
  Service {
    #[command(subcommand)]
    action: ServiceAction,
  },

  /// Train the lightweight reranker model
  Train {
    /// Number of training epochs
    #[arg(long, default_value = "3")]
    epochs: usize,
    /// Number of negatives per positive example
    #[arg(long, default_value = "6")]
    negatives: usize,
    /// Minimum history size required to train
    #[arg(long, default_value = "1000")]
    min_history: usize,
    /// Maximum number of history entries to use
    #[arg(long, default_value = "25000")]
    max_samples: usize,
  },

  /// Show reranker model status
  ModelStatus,

  /// Reset (delete) the reranker model
  ModelReset,

  /// (Internal) Record a single command invocation (used by shell hooks)
  Record {
    /// The command string that was executed
    #[arg(long)]
    command: String,
    /// The working directory where the command was executed
    #[arg(long)]
    working_directory: String,
    /// The exit status of the command
    #[arg(long)]
    exit_status: i64,
    /// The timestamp when the command started (Unix epoch seconds)
    #[arg(long)]
    start_timestamp: i64,
    /// The timestamp when the command finished (Unix epoch seconds)
    #[arg(long)]
    end_timestamp: i64,
    /// The shell session ID
    #[arg(long)]
    session_id: Option<i64>, // Optional for now
  },
}

#[derive(Subcommand, Debug)]
enum ServiceAction {
  /// Install the user service (systemd or launchd)
  Install,
  /// Uninstall the user service (systemd or launchd)
  Uninstall,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CompletionFormat {
  Plain,
  Zsh,
}

#[tokio::main]
async fn main() -> Result<()> {
  let cli = Cli::parse();
  setup_panic!();
  color_eyre::install()?;

  // Initialize tracing
  fmt()
    .with_env_filter(EnvFilter::from_env("ZAGE_LOG"))
    .init();

  // Debug environment variables
  let histfile = std::env::var("HISTFILE").unwrap_or_else(|_| "NOT SET".to_string());
  let shell = std::env::var("SHELL").unwrap_or_else(|_| "NOT SET".to_string());
  info!(
    "Environment variables: HISTFILE={}, SHELL={}",
    histfile, shell
  );

  info!("Starting application");
  // Determine DB path
  let db_path = if let Some(path) = &cli.db_path {
    path.clone()
  } else if let Ok(env_path) = std::env::var("ZAGE_DB_PATH") {
    PathBuf::from(env_path)
  } else {
    dirs::data_dir()
      .map(|v| v.join("zage/zage.db"))
      .ok_or_else(|| eyre!("Could not determine data directory"))?
  };
  debug!("Initializing db at {}", db_path.display());
  if let Some(parent) = db_path.parent() {
    std::fs::create_dir_all(parent)?;
  }
  let db = open_db(&db_path).await?;
  init(&db.conn).await?;

  match &cli.command {
    Some(Commands::Import {
      file,
      hostname,
      username,
      shell,
      no_index,
    }) => {
      // Import history and exit
      debug!("File argument value: {:?}", file);

      let history_file = if let Some(path) = file.clone() {
        path
      } else {
        debug!("No file specified, falling back to default path");
        let mut path = home_dir().ok_or_else(|| eyre!("Cannot find home directory"))?;
        let filename = match shell {
          Shell::Zsh => ".zsh_history",
          Shell::Bash => ".bash_history",
        };
        path.push(filename);
        debug!("Using default history file path: {:?}", path);
        path
      };

      let hostname_s = hostname.clone();
      let username_s = username.clone();
      let aliases = load_aliases();
      let mut invocations = match shell {
        Shell::Zsh => parse_zsh_history(&history_file, hostname_s.clone(), username_s.clone())?,
        Shell::Bash => parse_bash_history(&history_file, hostname_s, username_s)?,
      };
      for invocation in invocations.iter_mut() {
        if invocation.expanded_command.is_empty() {
          invocation.expanded_command = expand_alias(&invocation.command, &aliases)
            .unwrap_or_else(|| invocation.command.clone());
        }
      }
      import_history(&db.conn, invocations).await?;
      eprintln!("Imported history from {:?}", history_file);
      if *no_index {
        eprintln!("Index rebuild skipped (requested via --no-index)");
      } else {
        let report = rebuild_stats(&db.conn, None).await?;
        let seq_report = analyze_sequences(&db.conn, SequenceConfig::default()).await?;
        let token_seq_report = analyze_token_sequences(&db.conn, SequenceConfig::default()).await?;
        eprintln!(
          "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}, phase_stats={}",
          report.commands,
          report.transitions,
          report.contexts,
          report.token_cache,
          report.phase_stats
        );
        eprintln!(
          "Command sequence stats: sequences={}, bigrams={}, trigrams={}",
          seq_report.sequences, seq_report.bigrams, seq_report.trigrams
        );
        eprintln!(
          "Token sequence stats: sequences={}, bigrams={}, trigrams={}",
          token_seq_report.sequences, token_seq_report.bigrams, token_seq_report.trigrams
        );
      }
      info!("Imported history from {:?}", history_file);
    }

    Some(Commands::Index {
      max_commands,
      with_sequences,
    }) => {
      let report = rebuild_stats(&db.conn, *max_commands).await?;
      eprintln!(
        "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}, phase_stats={}",
        report.commands,
        report.transitions,
        report.contexts,
        report.token_cache,
        report.phase_stats
      );

      if *with_sequences {
        let seq_report = analyze_sequences(&db.conn, SequenceConfig::default()).await?;
        eprintln!(
          "Command sequence stats: sequences={}, bigrams={}, trigrams={}",
          seq_report.sequences, seq_report.bigrams, seq_report.trigrams
        );
        let token_report = analyze_token_sequences(&db.conn, SequenceConfig::default()).await?;
        eprintln!(
          "Token sequence stats: sequences={}, bigrams={}, trigrams={}",
          token_report.sequences, token_report.bigrams, token_report.trigrams
        );
      }
    }

    Some(Commands::Suggest {
      count,
      current_line,
      recent_limit,
      cwd,
      hostname,
      username,
      session_id,
      no_sequences,
      completion_format,
      show_scores,
      autosuggest,
    }) => {
      let cwd = match cwd.clone() {
        Some(val) => Some(val),
        None => std::env::current_dir()
          .ok()
          .and_then(|p| p.to_str().map(|s| s.to_string())),
      };

      let hostname = hostname.clone().or_else(|| Some(get_hostname()));
      let username = username
        .clone()
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
        limit: *count as u32,
        prefer_full_line: *autosuggest,
      };
      if let Ok(Some(response)) = server::try_request(request).await
        && let Response::Suggestions { items } = response
      {
        server_suggestions = Some(map_server_suggestions(items));
      }

      if has_prefix {
        let base_config = SuggestConfig {
          max_results: *count,
          recent_limit: *recent_limit,
          prefix: current_line.clone(),
          cwd: cwd.clone(),
          hostname: hostname.clone(),
          username: username.clone(),
          session_id: *session_id,
          use_sequences: !*no_sequences,
          prefer_full_line: *autosuggest,
        };

        let completions = if let Some(items) = server_suggestions {
          items
        } else {
          suggest(&db.conn, base_config).await?
        };
        if completions.is_empty() {
          return Ok(());
        }

        if *autosuggest {
          if let Some(first) = completions.first() {
            println!("{}", first.command);
          }
          return Ok(());
        }

        let prefix_str = current_line.clone().unwrap_or_default();
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

        let mut seen = std::collections::HashSet::new();
        for suggestion in completions {
          let candidate_tokens = tokenize(&suggestion.command);
          if let Some(tok) = candidate_tokens.get(target_index)
            && seen.insert(tok.raw.clone())
          {
            match completion_format {
              CompletionFormat::Plain => {
                if *show_scores {
                  println!("{}\t{:.4}", tok.raw, suggestion.score);
                } else {
                  println!("{}", tok.raw);
                }
              }
              CompletionFormat::Zsh => {
                let desc = if *show_scores {
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
          max_results: *count,
          recent_limit: *recent_limit,
          prefix: None,
          cwd,
          hostname,
          username,
          session_id: *session_id,
          use_sequences: !*no_sequences,
          prefer_full_line: *autosuggest,
        };

        let suggestions = if let Some(items) = server_suggestions {
          items
        } else {
          suggest(&db.conn, config).await?
        };
        if *autosuggest {
          if let Some(first) = suggestions.first() {
            println!("{}", first.command);
          }
          return Ok(());
        }
        for suggestion in suggestions {
          match completion_format {
            CompletionFormat::Plain => {
              if *show_scores {
                println!("{}\t{:.4}", suggestion.command, suggestion.score);
              } else {
                println!("{}", suggestion.command);
              }
            }
            CompletionFormat::Zsh => {
              let desc = if *show_scores {
                Some(format!("{:.4}", suggestion.score))
              } else {
                None
              };
              println!("{}", format_zsh_item(&suggestion.command, desc.as_deref()));
            }
          }
        }
      }
    }

    Some(Commands::AnalyzeSequences {
      min_support,
      min_confidence,
      min_lift,
      max_len,
    }) => {
      let config = SequenceConfig {
        min_support: *min_support,
        min_confidence: *min_confidence,
        min_lift: *min_lift,
        max_len: *max_len,
      };
      let report = analyze_sequences(&db.conn, config.clone()).await?;
      eprintln!(
        "Command sequence stats: sequences={}, bigrams={}, trigrams={}",
        report.sequences, report.bigrams, report.trigrams
      );
      let token_report = analyze_token_sequences(&db.conn, config).await?;
      eprintln!(
        "Token sequence stats: sequences={}, bigrams={}, trigrams={}",
        token_report.sequences, token_report.bigrams, token_report.trigrams
      );
    }
    Some(Commands::Train {
      epochs,
      negatives,
      min_history,
      max_samples,
    }) => {
      let report = train_model(
        &db.conn,
        TrainConfig {
          epochs: *epochs,
          negatives_per_pos: *negatives,
          min_history: *min_history,
          max_samples: *max_samples,
        },
      )
      .await?;
      eprintln!(
        "Trained reranker: samples={}, pairs={}, validation_accuracy={:.2}, model={}",
        report.samples,
        report.pairs,
        report.validation_accuracy,
        report.model_path.display()
      );
    }
    Some(Commands::ModelStatus) => {
      if let Some(model) = model_status()? {
        eprintln!(
          "Reranker model (trees={}, objective={}, loss={}, created_at={}, path={})",
          model.n_trees,
          model.objective,
          model.loss,
          model.created_at,
          model.model_path.display()
        );
      } else {
        eprintln!("Reranker model not found");
      }
    }
    Some(Commands::ModelReset) => {
      reset_model()?;
      eprintln!("Reranker model reset");
    }

    Some(Commands::Server { .. }) => {
      server::run_server(db_path.as_path()).await?;
    }

    Some(Commands::Service { action }) => match action {
      ServiceAction::Install => {
        service::install()?;
        eprintln!("Service installed");
      }
      ServiceAction::Uninstall => {
        service::uninstall()?;
        eprintln!("Service uninstalled");
      }
    },

    Some(Commands::Record {
      command,
      working_directory,
      exit_status,
      start_timestamp,
      end_timestamp,
      session_id,
    }) => {
      info!("Recording command invocation");
      let aliases = load_aliases();
      let expanded_command = expand_alias(command, &aliases).unwrap_or_else(|| command.clone());

      let server_req = Request::Record {
        command: command.clone(),
        expanded_command: expanded_command.clone(),
        working_directory: working_directory.clone(),
        exit_status: *exit_status as i32,
        start_timestamp: *start_timestamp,
        end_timestamp: *end_timestamp,
        session_id: session_id.unwrap_or_else(|| std::process::id() as i64) as u64,
      };
      if let Ok(Some(Response::Ack)) = server::try_request(server_req).await {
        return Ok(());
      }

      // Get hostname and username (best effort)
      let hostname = get_hostname();
      let username = uzers::get_current_username()
        .map(|v| v.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_string());

      // Generate a session ID if none provided (e.g., using PID)
      // For now, use a placeholder or default.
      // A better approach might involve Zsh sending a unique session ID.
      let session_id = session_id.unwrap_or_else(|| std::process::id() as i64);

      // Create Invocation struct
      let invocation = Invocation {
        command: command.clone(),
        expanded_command: expanded_command.clone(),
        shellname: detect_shellname(),
        working_directory: Some(working_directory.clone()),
        hostname: Some(hostname.clone()),
        username: Some(username.clone()),
        exit_status: Some(*exit_status),
        start_unix_timestamp: Some(*start_timestamp),
        end_unix_timestamp: Some(*end_timestamp),
        session_id,
      };

      // Record the command invocation
      debug!("Inserting invocation: {:?}", invocation);
      let inserted = insert_invocation(&db.conn, &invocation).await?;
      if inserted {
        update_stats_for_invocation(&db.conn, &invocation).await?;
        info!("Invocation recorded successfully.");
      } else {
        info!("Duplicate invocation skipped: {:?}", invocation.command);
      }
    }

    None => {
      let mut cmd = Cli::command();
      cmd.print_help()?;
      println!();
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
