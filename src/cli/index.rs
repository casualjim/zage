use color_eyre::eyre::{Result, eyre};

use crate::cli::BackendRef;
use crate::db::Db;
use crate::indexer::rebuild_stats;
use crate::sequence::{SequenceConfig, analyze_sequences, analyze_token_sequences};
use crate::server::{self, Request, Response};

pub async fn run(
  backend: BackendRef<'_>,
  max_commands: Option<usize>,
  with_sequences: bool,
  with_embeddings: bool,
) -> Result<()> {
  match backend {
    BackendRef::Server => run_server(max_commands, with_sequences, with_embeddings).await,
    BackendRef::Embedded(db) => {
      run_embedded(db, max_commands, with_sequences, with_embeddings).await
    }
  }
}

async fn run_server(
  max_commands: Option<usize>,
  with_sequences: bool,
  with_embeddings: bool,
) -> Result<()> {
  let request = Request::Index {
    max_commands,
    with_sequences,
    with_embeddings,
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

async fn run_embedded(
  db: &Db,
  max_commands: Option<usize>,
  with_sequences: bool,
  with_embeddings: bool,
) -> Result<()> {
  let report = rebuild_stats(&db.conn, max_commands).await?;
  eprintln!(
    "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}",
    report.commands, report.transitions, report.contexts, report.token_cache
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

  if with_embeddings {
    let count = crate::embeddings::index_command_embeddings(&db.conn, max_commands)
      .await
      .map_err(|err| eyre!(err))?;
    eprintln!("Command embeddings: embedded={count}");
  }

  Ok(())
}
