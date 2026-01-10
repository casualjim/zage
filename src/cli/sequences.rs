use color_eyre::eyre::{Result, eyre};

use crate::cli::BackendRef;
use crate::db::Db;
use crate::sequence::{SequenceConfig, analyze_sequences, analyze_token_sequences};
use crate::server::{self, Request, Response};

pub async fn run(
  backend: BackendRef<'_>,
  min_support: usize,
  min_confidence: f64,
  min_lift: f64,
  max_len: usize,
) -> Result<()> {
  match backend {
    BackendRef::Server => run_server(min_support, min_confidence, min_lift, max_len).await,
    BackendRef::Embedded(db) => {
      run_embedded(db, min_support, min_confidence, min_lift, max_len).await
    }
  }
}

async fn run_server(
  min_support: usize,
  min_confidence: f64,
  min_lift: f64,
  max_len: usize,
) -> Result<()> {
  let request = Request::AnalyzeSequences {
    min_support,
    min_confidence,
    min_lift,
    max_len,
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
    None => Err(eyre!("Sequence server unavailable")),
  }
}

async fn run_embedded(
  db: &Db,
  min_support: usize,
  min_confidence: f64,
  min_lift: f64,
  max_len: usize,
) -> Result<()> {
  let config = SequenceConfig {
    min_support,
    min_confidence,
    min_lift,
    max_len,
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
  Ok(())
}
