use std::path::PathBuf;

use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;
use dirs::home_dir;
use human_panic::setup_panic;
use tracing::debug;
use tracing::info;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use uzers;
use zage::db::{import_history, init, insert_invocation, open_db, update_stats_for_invocation};
use zage::indexer::rebuild_stats;
use zage::predict::{SuggestConfig, suggest};
use zage::sequence::{SequenceConfig, analyze_sequences, analyze_token_sequences};
use zage::shell_history::{Invocation, Shell, get_hostname, parse_bash_history, parse_zsh_history};
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
  fmt().with_env_filter(EnvFilter::from_default_env()).init();

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
      .expect("Could not determine data directory")
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
    }) => {
      // Import history and exit
      debug!("File argument value: {:?}", file);

      let history_file = file.clone().unwrap_or_else(|| {
        debug!("No file specified, falling back to default path");
        let mut path = home_dir().expect("Cannot find home directory");
        let filename = match shell {
          Shell::Zsh => ".zsh_history",
          Shell::Bash => ".bash_history",
        };
        path.push(filename);
        debug!("Using default history file path: {:?}", path);
        path
      });

      let hostname_s = hostname.clone();
      let username_s = username.clone();
      let invocations = match shell {
        Shell::Zsh => parse_zsh_history(&history_file, hostname_s.clone(), username_s.clone())?,
        Shell::Bash => parse_bash_history(&history_file, hostname_s, username_s)?,
      };
      import_history(&db.conn, invocations).await?;
      println!("Imported history from {:?}", history_file);
      info!("Imported history from {:?}", history_file);
    }

    Some(Commands::Index {
      max_commands,
      with_sequences,
    }) => {
      let report = rebuild_stats(&db.conn, *max_commands).await?;
      println!(
        "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}",
        report.commands, report.transitions, report.contexts, report.token_cache
      );

      if *with_sequences {
        let seq_report = analyze_sequences(&db.conn, SequenceConfig::default()).await?;
        println!(
          "Command sequence stats: sequences={}, bigrams={}, trigrams={}",
          seq_report.sequences, seq_report.bigrams, seq_report.trigrams
        );
        let token_report = analyze_token_sequences(&db.conn, SequenceConfig::default()).await?;
        println!(
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
      let username = username.clone().or_else(|| {
        uzers::get_current_username().map(|v| v.to_string_lossy().into_owned())
      });

      let has_prefix = current_line.as_ref().map(|s| !s.is_empty()).unwrap_or(false);

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
        };

        let completions = suggest(&db.conn, base_config).await?;
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
          if let Some(tok) = candidate_tokens.get(target_index) {
            if seen.insert(tok.raw.clone()) {
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
        };

        let suggestions = suggest(&db.conn, config).await?;
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
      println!(
        "Command sequence stats: sequences={}, bigrams={}, trigrams={}",
        report.sequences, report.bigrams, report.trigrams
      );
      let token_report = analyze_token_sequences(&db.conn, config).await?;
      println!(
        "Token sequence stats: sequences={}, bigrams={}, trigrams={}",
        token_report.sequences, token_report.bigrams, token_report.trigrams
      );
    }

    Some(Commands::Record {
      command,
      working_directory,
      exit_status,
      start_timestamp,
      end_timestamp,
      session_id,
    }) => {
      info!("Recording command invocation");

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
        shellname: "zsh".to_string(), // Assume zsh for now
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
      // Default application behavior
      println!("Starting Zage application...");
      println!("Use --help to see available commands");
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
      format!("{}:{}", escaped, d_esc)
    }
    None => format!("{}:", escaped),
  }
}
