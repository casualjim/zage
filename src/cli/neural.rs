use std::time::Duration;

use color_eyre::eyre::{Result, eyre};
use humantime::Duration as HumanDuration;

use crate::cli::BackendRef;
use crate::db::Db;
use crate::neural::NeuralTrainConfig;
use crate::server::{self, Request, Response};

pub async fn train(
  backend: BackendRef<'_>,
  cfg: NeuralTrainConfig,
  timeout: Option<HumanDuration>,
) -> Result<()> {
  match backend {
    BackendRef::Server => train_server(cfg, timeout).await,
    BackendRef::Embedded(db) => train_embedded(db, cfg).await,
  }
}

async fn train_server(cfg: NeuralTrainConfig, timeout: Option<HumanDuration>) -> Result<()> {
  let timeout_ms = timeout
    .map(|duration| Duration::from(duration).as_millis())
    .map(|millis| u64::try_from(millis).unwrap_or(u64::MAX));
  let request = Request::NeuralTrain {
    epochs: cfg.epochs,
    batch_size: cfg.batch_size,
    learning_rate: cfg.learning_rate,
    window: cfg.window,
    vocab_size: cfg.vocab_size,
    max_seq_len: cfg.max_seq_len,
    embed_dim: cfg.embed_dim,
    projection_dim: cfg.projection_dim,
    temperature: cfg.temperature,
    seed: cfg.seed,
    timeout_ms,
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
    None => Err(eyre!("Neural train server unavailable")),
  }
}

async fn train_embedded(db: &Db, cfg: NeuralTrainConfig) -> Result<()> {
  let path = crate::neural::train_biencoder_wgpu(&db.conn, cfg).await?;
  eprintln!("Trained neural bi-encoder: model={}", path.display());
  Ok(())
}
