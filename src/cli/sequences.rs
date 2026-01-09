use color_eyre::eyre::Result;

use crate::db::Db;
use crate::sequence::{SequenceConfig, analyze_sequences, analyze_token_sequences};

pub async fn run(
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
