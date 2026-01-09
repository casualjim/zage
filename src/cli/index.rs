use color_eyre::eyre::{Result, eyre};

use crate::cli::Backend;
use crate::db::Db;
use crate::indexer::rebuild_stats;
use crate::sequence::{SequenceConfig, analyze_sequences, analyze_token_sequences};
use crate::server::{self, Request, Response};

pub async fn run(
  backend: Backend<'_>,
  max_commands: Option<usize>,
  with_sequences: bool,
) -> Result<()> {
  match backend {
    Backend::Server => run_server(max_commands, with_sequences).await,
    Backend::Embedded(db) => run_embedded(db, max_commands, with_sequences).await,
  }
}

async fn run_server(max_commands: Option<usize>, with_sequences: bool) -> Result<()> {
  let request = Request::Index {
    max_commands,
    with_sequences,
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
    None => Err(eyre!("Index server unavailable")),
  }
}

async fn run_embedded(db: &Db, max_commands: Option<usize>, with_sequences: bool) -> Result<()> {
  let report = rebuild_stats(&db.conn, max_commands).await?;
  eprintln!(
    "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}, phase_stats={}",
    report.commands, report.transitions, report.contexts, report.token_cache, report.phase_stats
  );

  if with_sequences {
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

  Ok(())
}
