use color_eyre::eyre::Result;

use crate::db::Db;
use crate::rerank::{TrainConfig, model_status as rerank_model_status, reset_model, train_model};

pub async fn run(
  db: &Db,
  epochs: usize,
  negatives: usize,
  min_history: usize,
  max_samples: usize,
) -> Result<()> {
  let report = train_model(
    &db.conn,
    TrainConfig {
      epochs,
      negatives_per_pos: negatives,
      min_history,
      max_samples,
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
  Ok(())
}

pub fn model_status() -> Result<()> {
  if let Some(model) = rerank_model_status()? {
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
  Ok(())
}

pub fn model_reset() -> Result<()> {
  reset_model()?;
  eprintln!("Reranker model reset");
  Ok(())
}
