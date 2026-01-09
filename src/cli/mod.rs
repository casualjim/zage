use std::env;
use std::path::PathBuf;

use clap::{CommandFactory, Parser, Subcommand};
use color_eyre::eyre::{Result, eyre};
use human_panic::setup_panic;
use tracing::{debug, info};
use tracing_subscriber::{EnvFilter, fmt};

use crate::db::{Db, init, open_db};
use crate::shell_history::Shell;

mod import;
mod index;
mod record;
mod sequences;
mod server;
mod service;
mod suggest;
mod train;

/// CLI for Zage
#[derive(Parser)]
#[clap(author, version, about)]
pub struct Cli {
  #[command(subcommand)]
  command: Option<Commands>,
  /// Path to the SQLite database file (overrides default location)
  #[clap(long, env = "ZAGE_DB_PATH")]
  db_path: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
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

#[derive(Subcommand, Debug, Clone, Copy)]
pub enum ServiceAction {
  /// Install the user service (systemd or launchd)
  Install,
  /// Uninstall the user service (systemd or launchd)
  Uninstall,
}

#[derive(Clone, Copy, Debug, clap::ValueEnum)]
pub enum CompletionFormat {
  Plain,
  Zsh,
}

pub async fn run(cli: Cli) -> Result<()> {
  setup_panic!();
  color_eyre::install()?;

  fmt()
    .with_env_filter(EnvFilter::from_env("ZAGE_LOG"))
    .init();

  let histfile = env::var("HISTFILE").unwrap_or_else(|_| "NOT SET".to_string());
  let shell = env::var("SHELL").unwrap_or_else(|_| "NOT SET".to_string());
  info!(
    "Environment variables: HISTFILE={}, SHELL={}",
    histfile, shell
  );
  info!("Starting application");

  let db_path = resolve_db_path(&cli.db_path)?;
  let db = open_db_for_cli(&db_path).await?;

  match cli.command {
    Some(Commands::Import {
      file,
      hostname,
      username,
      shell,
      no_index,
    }) => {
      import::run(&db, file, hostname, username, shell, no_index).await?;
    }
    Some(Commands::Index {
      max_commands,
      with_sequences,
    }) => {
      index::run(&db, max_commands, with_sequences).await?;
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
      suggest::run(
        &db,
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
      )
      .await?;
    }
    Some(Commands::AnalyzeSequences {
      min_support,
      min_confidence,
      min_lift,
      max_len,
    }) => {
      sequences::run(
        &db,
        min_support,
        min_confidence,
        min_lift,
        max_len,
      )
      .await?;
    }
    Some(Commands::Train {
      epochs,
      negatives,
      min_history,
      max_samples,
    }) => {
      train::run(&db, epochs, negatives, min_history, max_samples).await?;
    }
    Some(Commands::ModelStatus) => {
      train::model_status()?;
    }
    Some(Commands::ModelReset) => {
      train::model_reset()?;
    }
    Some(Commands::Server {}) => {
      server::run(&db_path).await?;
    }
    Some(Commands::Service { action }) => {
      service::run(action)?;
    }
    Some(Commands::Record {
      command,
      working_directory,
      exit_status,
      start_timestamp,
      end_timestamp,
      session_id,
    }) => {
      record::run(
        &db,
        command,
        working_directory,
        exit_status,
        start_timestamp,
        end_timestamp,
        session_id,
      )
      .await?;
    }
    None => {
      let mut cmd = Cli::command();
      cmd.print_help()?;
      println!();
    }
  }

  Ok(())
}

fn resolve_db_path(cli_db_path: &Option<PathBuf>) -> Result<PathBuf> {
  if let Some(path) = cli_db_path {
    Ok(path.clone())
  } else if let Ok(env_path) = env::var("ZAGE_DB_PATH") {
    Ok(PathBuf::from(env_path))
  } else {
    dirs::data_dir()
      .map(|v| v.join("zage/zage.db"))
      .ok_or_else(|| eyre!("Could not determine data directory"))
  }
}

async fn open_db_for_cli(db_path: &PathBuf) -> Result<Db> {
  debug!("Initializing db at {}", db_path.display());
  if let Some(parent) = db_path.parent() {
    std::fs::create_dir_all(parent)?;
  }
  let db = open_db(db_path).await?;
  init(&db.conn).await?;
  Ok(db)
}
