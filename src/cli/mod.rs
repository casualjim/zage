use std::env;
use std::path::PathBuf;

use clap::{CommandFactory, Parser, Subcommand};
use color_eyre::eyre::{Result, eyre};
use human_panic::setup_panic;
use humantime::Duration as HumanDuration;
use tracing::{debug, info};
use tracing_subscriber::{EnvFilter, fmt};

use crate::config::{AppConfig, DbConfig};
use crate::db::{Db, open_db_with_config};
use crate::shell_history::Shell;

mod feedback;
mod import;
mod index;
#[cfg(feature = "pprof")]
mod pprof;
mod record;
mod sequences;
mod server;
mod service;
mod suggest;
mod train;
mod yank;

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
    /// Use the embedded SQLite database
    #[arg(long)]
    embedded_db: bool,
  },

  /// Build or rebuild index tables for prediction
  Index {
    /// Maximum number of commands to index (oldest first)
    #[arg(long)]
    max_commands: Option<usize>,

    /// Also recompute sequence statistics
    #[arg(long)]
    with_sequences: bool,

    /// Also generate embeddings for commands
    #[arg(long)]
    with_embeddings: bool,
    /// Use the embedded SQLite database
    #[arg(long)]
    embedded_db: bool,
  },

  /// (Internal) Record a single command invocation (used by shell hooks)
  #[command(hide = true)]
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
    /// Shell name (defaults to $SHELL basename)
    #[arg(long = "shell", env = "SHELL")]
    shellname: Option<String>,
    /// The shell session ID
    #[arg(long)]
    session_id: Option<i64>, // Optional for now
    /// Use the embedded SQLite database
    #[arg(long)]
    embedded_db: bool,
  },

  /// (Internal) Record a suggestion feedback event (used by shell hooks)
  #[command(hide = true)]
  Feedback {
    /// Unique id for the suggestion that was shown
    #[arg(long)]
    shown_id: String,
    /// Unix epoch seconds when the suggestion was shown
    #[arg(long)]
    shown_at: i64,
    /// Current working directory at time of feedback
    #[arg(long)]
    working_directory: Option<String>,
    /// The suggestion string that was shown
    #[arg(long)]
    suggestion: String,
    /// The command that was executed (if known)
    #[arg(long)]
    accepted_command: Option<String>,
    /// Unix epoch seconds when the execution happened (if known)
    #[arg(long)]
    accepted_at: Option<i64>,
    /// Outcome label (e.g. accepted, rejected)
    #[arg(long)]
    outcome: Option<String>,
    /// Use the embedded SQLite database
    #[arg(long)]
    embedded_db: bool,
  },

  /// Remove matching commands from history
  Yank {
    /// Command line to remove (exact match)
    #[arg(value_name = "COMMAND")]
    command: String,
    /// Also match expanded_command entries
    #[arg(long)]
    match_expanded: bool,
    /// Skip recomputing sequence statistics
    #[arg(long)]
    no_sequences: bool,
    /// Use the embedded SQLite database
    #[arg(long)]
    embedded_db: bool,
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

    /// Override shell name (defaults to $SHELL basename)
    #[arg(long = "shell", env = "SHELL")]
    shellname: Option<String>,

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

    /// Request timeout for suggestions when using the server (e.g. 2s, 500ms)
    #[arg(long, env = "ZAGE_SUGGEST_TIMEOUT")]
    timeout: Option<HumanDuration>,
    /// Use the embedded SQLite database
    #[arg(long)]
    embedded_db: bool,
  },

  /// Sequence analysis and stats
  Sequences {
    #[command(subcommand)]
    action: SequencesAction,
  },

  /// Manage the online model
  Model {
    #[command(subcommand)]
    action: ModelAction,
  },

  /// Run the suggestion server (foreground)
  Server {},

  /// Install or uninstall the background suggestion service
  Service {
    #[command(subcommand)]
    action: ServiceAction,
  },

  /// Capture a CPU profile using pprof
  #[cfg(feature = "pprof")]
  Pprof {
    /// Duration to sample (e.g. 30s, 2m)
    #[arg(long, default_value = "30s")]
    duration: HumanDuration,
    /// Sampling frequency in Hz
    #[arg(long, default_value = "100")]
    frequency: u32,
    /// Output file path for the profile
    #[arg(long, default_value = "zage.pprof")]
    output: PathBuf,
    /// Use the embedded SQLite database
    #[arg(long)]
    embedded_db: bool,
  },
}

#[derive(Subcommand, Debug)]
pub enum SequencesAction {
  /// Analyze and store frequent command sequences
  Analyze {
    /// Minimum support count
    #[arg(long, default_value = "2")]
    min_support: usize,
    /// Minimum confidence threshold
    #[arg(long, default_value = "0.5")]
    min_confidence: f64,
    /// Minimum lift threshold
    #[arg(long, default_value = "1.2")]
    min_lift: f64,
    /// Maximum sequence length (2-5)
    #[arg(long, default_value = "5")]
    max_len: usize,
    /// Use the embedded SQLite database
    #[arg(long)]
    embedded_db: bool,
  },
}

#[derive(Subcommand, Debug)]
pub enum ModelAction {
  /// Train the legacy reranker model
  Train {
    /// Number of training epochs
    #[arg(long, default_value = "150")]
    epochs: usize,
    /// Number of negatives per positive example
    #[arg(long, default_value = "6")]
    negatives: usize,
    /// Minimum history size required to train (0 = auto)
    #[arg(long, default_value = "0")]
    min_history: usize,
    /// Maximum number of history entries to use (0 = no limit)
    #[arg(long, default_value = "0")]
    max_samples: usize,
    /// Request timeout for training when using the server (e.g. 5m, 30s)
    #[arg(long)]
    timeout: Option<HumanDuration>,
    /// Use the embedded SQLite database
    #[arg(long)]
    embedded_db: bool,
  },

  /// Show online model status
  Status {
    /// Use the embedded SQLite database
    #[arg(long)]
    embedded_db: bool,
  },

  /// Reset (delete) the online model
  Reset {
    /// Use the embedded SQLite database
    #[arg(long)]
    embedded_db: bool,
  },
}

#[derive(Clone)]
pub struct SuggestArgs {
  pub count: usize,
  pub current_line: Option<String>,
  pub recent_limit: usize,
  pub cwd: Option<String>,
  pub hostname: Option<String>,
  pub username: Option<String>,
  pub session_id: Option<i64>,
  pub shellname: Option<String>,
  pub no_sequences: bool,
  pub completion_format: CompletionFormat,
  pub show_scores: bool,
  pub autosuggest: bool,
  pub timeout: Option<HumanDuration>,
}

pub enum Backend {
  Server,
  Embedded(Box<Db>),
}

pub enum BackendRef<'a> {
  Server,
  Embedded(&'a Db),
}

impl Backend {
  pub fn as_ref(&self) -> BackendRef<'_> {
    match self {
      Self::Server => BackendRef::Server,
      Self::Embedded(db) => BackendRef::Embedded(db.as_ref()),
    }
  }
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
  let app_config = AppConfig::load()?;
  let db_config = app_config.db.with_cli_path(cli.db_path.as_ref());

  let backend = resolve_backend(&app_config, &db_config, cli.command.as_ref()).await?;

  match cli.command {
    Some(Commands::Import {
      file,
      hostname,
      username,
      shell,
      no_index,
      embedded_db: _,
    }) => {
      let backend = require_backend(backend.as_ref())?;
      import::run(backend, file, hostname, username, shell, no_index).await?;
    }
    Some(Commands::Index {
      max_commands,
      with_sequences,
      with_embeddings,
      embedded_db: _,
    }) => {
      let backend = require_backend(backend.as_ref())?;
      index::run(backend, max_commands, with_sequences, with_embeddings).await?;
    }
    Some(Commands::Record {
      command,
      working_directory,
      exit_status,
      start_timestamp,
      end_timestamp,
      shellname,
      session_id,
      embedded_db: _,
    }) => {
      let backend = require_backend(backend.as_ref())?;
      record::run(
        backend,
        command,
        working_directory,
        exit_status,
        start_timestamp,
        end_timestamp,
        shellname,
        session_id,
      )
      .await?;
    }
    Some(Commands::Feedback {
      shown_id,
      shown_at,
      working_directory,
      suggestion,
      accepted_command,
      accepted_at,
      outcome,
      embedded_db: _,
    }) => {
      let backend = require_backend(backend.as_ref())?;
      feedback::run(
        backend,
        shown_id,
        shown_at,
        working_directory,
        suggestion,
        accepted_command,
        accepted_at,
        outcome,
      )
      .await?;
    }
    Some(Commands::Yank {
      command,
      match_expanded,
      no_sequences,
      embedded_db: _,
    }) => {
      let backend = require_backend(backend.as_ref())?;
      yank::run(backend, command, match_expanded, no_sequences).await?;
    }
    Some(Commands::Suggest {
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
      embedded_db: _,
    }) => {
      let args = SuggestArgs {
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
      };
      let backend = require_backend(backend.as_ref())?;
      suggest::run(backend, args).await?;
    }
    Some(Commands::Sequences { action }) => match action {
      SequencesAction::Analyze {
        min_support,
        min_confidence,
        min_lift,
        max_len,
        embedded_db: _,
      } => {
        let backend = require_backend(backend.as_ref())?;
        sequences::run(backend, min_support, min_confidence, min_lift, max_len).await?;
      }
    },
    Some(Commands::Model { action }) => match action {
      ModelAction::Train {
        epochs,
        negatives,
        min_history,
        max_samples,
        timeout,
        embedded_db: _,
      } => {
        let backend = require_backend(backend.as_ref())?;
        train::run(
          backend,
          epochs,
          negatives,
          min_history,
          max_samples,
          timeout,
        )
        .await?;
      }
      ModelAction::Status { embedded_db: _ } => {
        let backend = require_backend(backend.as_ref())?;
        train::model_status(backend).await?;
      }
      ModelAction::Reset { embedded_db: _ } => {
        let backend = require_backend(backend.as_ref())?;
        train::model_reset(backend).await?;
      }
    },
    Some(Commands::Server {}) => {
      server::run(&db_config).await?;
    }
    Some(Commands::Service { action }) => {
      service::run(action)?;
    }
    #[cfg(feature = "pprof")]
    Some(Commands::Pprof {
      duration,
      frequency,
      output,
      embedded_db: _,
    }) => {
      let backend = require_backend(backend.as_ref())?;
      pprof::run(backend, duration, frequency, output).await?;
    }
    None => {
      let mut cmd = Cli::command();
      cmd.print_help()?;
      println!();
    }
  }

  Ok(())
}

async fn open_db_for_cli(db_config: &DbConfig) -> Result<Db> {
  match db_config.kind {
    crate::config::DbKind::Local => {
      debug!("Initializing db at {}", db_config.path.display());
    }
    crate::config::DbKind::Remote => {
      debug!("Initializing remote db");
    }
    crate::config::DbKind::RemoteReplica => {
      debug!(
        "Initializing remote replica db at {}",
        db_config.path.display()
      );
    }
  }
  open_db_with_config(db_config)
    .await
    .map_err(|err| eyre!(err))
}

fn command_uses_backend(command: &Commands) -> bool {
  !matches!(command, Commands::Server { .. } | Commands::Service { .. })
}

fn command_embedded_override(command: &Commands) -> bool {
  match command {
    Commands::Import { embedded_db, .. }
    | Commands::Index { embedded_db, .. }
    | Commands::Record { embedded_db, .. }
    | Commands::Feedback { embedded_db, .. }
    | Commands::Yank { embedded_db, .. } => *embedded_db,
    Commands::Suggest { embedded_db, .. } => *embedded_db,
    Commands::Sequences { action } => match action {
      SequencesAction::Analyze { embedded_db, .. } => *embedded_db,
    },
    Commands::Model { action } => match action {
      ModelAction::Train { embedded_db, .. }
      | ModelAction::Status { embedded_db, .. }
      | ModelAction::Reset { embedded_db, .. } => *embedded_db,
    },
    #[cfg(feature = "pprof")]
    Commands::Pprof { embedded_db, .. } => *embedded_db,
    Commands::Server { .. } | Commands::Service { .. } => false,
  }
}

async fn resolve_backend(
  app_config: &AppConfig,
  db_config: &DbConfig,
  command: Option<&Commands>,
) -> Result<Option<Backend>> {
  let Some(command) = command else {
    return Ok(None);
  };
  if !command_uses_backend(command) {
    return Ok(None);
  }
  let use_embedded = command_embedded_override(command) || app_config.backend.is_embedded();
  if use_embedded {
    let db = open_db_for_cli(db_config).await?;
    Ok(Some(Backend::Embedded(Box::new(db))))
  } else {
    Ok(Some(Backend::Server))
  }
}

fn require_backend(backend: Option<&Backend>) -> Result<BackendRef<'_>> {
  backend
    .map(|value| value.as_ref())
    .ok_or_else(|| eyre!("Backend unavailable"))
}
