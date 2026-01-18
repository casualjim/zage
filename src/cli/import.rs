use std::env;
use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{Result, eyre};
use dirs::home_dir;
use tracing::{debug, info};

use crate::cli::BackendRef;
use crate::db::{Db, import_history, reset_online_model};
use crate::indexer::rebuild_stats;
use crate::online_model::trainer::train_on_invocations_bulk;
use crate::predict::aliases::{expand_alias, load_aliases};
use crate::sequence::{SequenceConfig, analyze_sequences, analyze_token_sequences};
use crate::server::{self, Request, Response};
use crate::shell_history::{Shell, parse_bash_history, parse_zsh_history};

pub async fn run(
  backend: BackendRef<'_>,
  file: Option<PathBuf>,
  hostname: Option<String>,
  username: Option<String>,
  shell: Shell,
  no_index: bool,
  reset_model: bool,
) -> Result<()> {
  match backend {
    BackendRef::Server => run_server(file, hostname, username, shell, no_index, reset_model).await,
    BackendRef::Embedded(db) => {
      run_embedded(db, file, hostname, username, shell, no_index, reset_model).await
    }
  }
}

fn normalize_input_path(path: PathBuf) -> Result<PathBuf> {
  if path.is_relative() {
    let cwd = env::current_dir()?;
    Ok(cwd.join(path))
  } else {
    Ok(path)
  }
}

async fn run_server(
  file: Option<PathBuf>,
  hostname: Option<String>,
  username: Option<String>,
  shell: Shell,
  no_index: bool,
  reset_model: bool,
) -> Result<()> {
  let base_dir = std::env::current_dir()
    .ok()
    .map(|path| path.to_string_lossy().into_owned());
  let shell_name = match shell {
    Shell::Zsh => "zsh".to_string(),
    Shell::Bash => "bash".to_string(),
  };

  let request = Request::Import {
    file: file.map(|p| p.to_string_lossy().into_owned()),
    base_dir,
    hostname,
    username,
    shell: shell_name,
    no_index,
    reset_model,
  };
  match server::try_request(request).await? {
    Some(Response::Text { lines }) => {
      for line in lines {
        eprintln!("{line}");
      }
      Ok(())
    }
    Some(Response::Error { message }) => Err(eyre!(message)),
    Some(_) => Err(eyre!("Unexpected response from server")),
    None => Err(eyre!("Import server unavailable")),
  }
}

async fn run_embedded(
  db: &Db,
  file: Option<PathBuf>,
  hostname: Option<String>,
  username: Option<String>,
  shell: Shell,
  no_index: bool,
  reset_model: bool,
) -> Result<()> {
  debug!("File argument value: {:?}", file);

  let import_start = Instant::now();

  let history_file = if let Some(path) = file {
    normalize_input_path(path)?
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

  let parse_start = Instant::now();
  let aliases = load_aliases();
  let mut invocations = match shell {
    Shell::Zsh => parse_zsh_history(&history_file, hostname.clone(), username.clone())?,
    Shell::Bash => parse_bash_history(&history_file, hostname.clone(), username.clone())?,
  };
  eprintln!(
    "Parsed history in {:.2}s",
    parse_start.elapsed().as_secs_f64()
  );

  let expand_start = Instant::now();
  for invocation in invocations.iter_mut() {
    if invocation.expanded_command.is_empty() {
      invocation.expanded_command =
        expand_alias(&invocation.command, &aliases).unwrap_or_else(|| invocation.command.clone());
    }
  }
  eprintln!(
    "Expanded aliases in {:.2}s",
    expand_start.elapsed().as_secs_f64()
  );

  let db_insert_start = Instant::now();
  import_history(&db.conn, invocations.iter().cloned()).await?;
  eprintln!(
    "Inserted history into DB in {:.2}s",
    db_insert_start.elapsed().as_secs_f64()
  );
  eprintln!("Imported history from {:?}", history_file);
  if reset_model {
    let reset_start = Instant::now();
    reset_online_model(&db.conn).await?;
    eprintln!("Online model reset");
    eprintln!(
      "Reset online model in {:.2}s",
      reset_start.elapsed().as_secs_f64()
    );
  }
  let train_start = Instant::now();
  train_on_invocations_bulk(&db.conn, &invocations).await?;
  eprintln!(
    "Trained online model in {:.2}s",
    train_start.elapsed().as_secs_f64()
  );
  eprintln!("Online model trained on {} invocations", invocations.len());
  if no_index {
    eprintln!("Index rebuild skipped (requested via --no-index)");
  } else {
    let index_start = Instant::now();
    let report = rebuild_stats(&db.conn, None).await?;
    eprintln!(
      "Rebuilt stats in {:.2}s",
      index_start.elapsed().as_secs_f64()
    );
    let sequences_start = Instant::now();
    let seq_report = analyze_sequences(&db.conn, SequenceConfig::default()).await?;
    eprintln!(
      "Analyzed command sequences in {:.2}s",
      sequences_start.elapsed().as_secs_f64()
    );
    let token_sequences_start = Instant::now();
    let token_seq_report = analyze_token_sequences(&db.conn, SequenceConfig::default()).await?;
    eprintln!(
      "Analyzed token sequences in {:.2}s",
      token_sequences_start.elapsed().as_secs_f64()
    );
    eprintln!(
      "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}, phase_stats={}",
      report.commands, report.transitions, report.contexts, report.token_cache, report.phase_stats
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
  eprintln!(
    "Total import pipeline: {:.2}s",
    import_start.elapsed().as_secs_f64()
  );
  info!("Imported history from {:?}", history_file);
  Ok(())
}
