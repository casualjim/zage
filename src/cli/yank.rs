use color_eyre::eyre::{Result, eyre};
use tracing::info;

use crate::cli::BackendRef;
use crate::db::{Db, delete_history_by_command};
use crate::indexer::rebuild_stats;
use crate::sequence::{SequenceConfig, analyze_sequences, analyze_token_sequences};
use crate::server::{self, Request, Response};

pub async fn run(
  backend: BackendRef<'_>,
  command: String,
  match_expanded: bool,
  no_sequences: bool,
) -> Result<()> {
  match backend {
    BackendRef::Server => run_server(command, match_expanded, no_sequences).await,
    BackendRef::Embedded(db) => run_embedded(db, command, match_expanded, no_sequences).await,
  }
}

async fn run_server(command: String, match_expanded: bool, no_sequences: bool) -> Result<()> {
  let request = Request::Yank {
    command,
    match_expanded,
    with_sequences: !no_sequences,
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
    None => Err(eyre!("Yank server unavailable")),
  }
}

async fn run_embedded(
  db: &Db,
  command: String,
  match_expanded: bool,
  no_sequences: bool,
) -> Result<()> {
  let removed = delete_history_by_command(&db.conn, &command, match_expanded).await?;
  if removed == 0 {
    eprintln!("No history entries matched {command:?}");
    return Ok(());
  }

  if match_expanded {
    eprintln!(
      "Removed {} history entries matching command or expanded_command: {:?}",
      removed, command
    );
  } else {
    eprintln!(
      "Removed {} history entries matching command: {:?}",
      removed, command
    );
  }

  let report = rebuild_stats(&db.conn, None).await?;
  eprintln!(
    "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}, phase_stats={}",
    report.commands, report.transitions, report.contexts, report.token_cache, report.phase_stats
  );

  if !no_sequences {
    let seq_report = analyze_sequences(&db.conn, SequenceConfig::default()).await?;
    eprintln!(
      "Command sequence stats: sequences={}, bigrams={}, trigrams={}",
      seq_report.sequences, seq_report.bigrams, seq_report.trigrams
    );
    let token_seq_report = analyze_token_sequences(&db.conn, SequenceConfig::default()).await?;
    eprintln!(
      "Token sequence stats: sequences={}, bigrams={}, trigrams={}",
      token_seq_report.sequences, token_seq_report.bigrams, token_seq_report.trigrams
    );
  }

  info!("Yanked {} history entries", removed);
  Ok(())
}
