use std::time::Duration;

use color_eyre::eyre::{Result, eyre};
use humantime::Duration as HumanDuration;

use crate::cli::BackendRef;
use crate::db::Db;
use crate::rerank::{TrainConfig, model_status as rerank_model_status, reset_model, train_model};
use crate::server::{self, Request, Response};

pub async fn run(
  backend: BackendRef<'_>,
  epochs: usize,
  negatives: usize,
  min_history: usize,
  max_samples: usize,
  timeout: Option<HumanDuration>,
) -> Result<()> {
  match backend {
    BackendRef::Server => run_server(epochs, negatives, min_history, max_samples, timeout).await,
    BackendRef::Embedded(db) => run_embedded(db, epochs, negatives, min_history, max_samples).await,
  }
}

pub async fn model_status(backend: BackendRef<'_>) -> Result<()> {
  match backend {
    BackendRef::Server => model_status_server().await,
    BackendRef::Embedded(_) => model_status_embedded(),
  }
}

pub async fn model_reset(backend: BackendRef<'_>) -> Result<()> {
  match backend {
    BackendRef::Server => model_reset_server().await,
    BackendRef::Embedded(_) => model_reset_embedded(),
  }
}

async fn run_server(
  epochs: usize,
  negatives: usize,
  min_history: usize,
  max_samples: usize,
  timeout: Option<HumanDuration>,
) -> Result<()> {
  let timeout_ms = timeout
    .map(|duration| Duration::from(duration).as_millis())
    .map(|millis| u64::try_from(millis).unwrap_or(u64::MAX));
  let request = Request::Train {
    epochs,
    negatives,
    min_history,
    max_samples,
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
    None => Err(eyre!("Train server unavailable")),
  }
}

async fn run_embedded(
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
    "Trained reranker: samples={}, pairs={}, validation_accuracy={:.2}, validation_top1={:.2}, model={}",
    report.samples,
    report.pairs,
    report.validation_accuracy,
    report.validation_top1,
    report.model_path.display()
  );
  Ok(())
}

async fn model_status_server() -> Result<()> {
  match server::try_request(Request::ModelStatus).await? {
    Some(Response::Text { lines }) => {
      for line in lines {
        eprintln!("{line}");
      }
      Ok(())
    }
    Some(Response::Error { message }) => Err(eyre!(message)),
    Some(_) => Err(eyre!("Unexpected response from server")),
    None => Err(eyre!("Model status server unavailable")),
  }
}

fn model_status_embedded() -> Result<()> {
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

async fn model_reset_server() -> Result<()> {
  match server::try_request(Request::ModelReset).await? {
    Some(Response::Text { lines }) => {
      for line in lines {
        eprintln!("{line}");
      }
      Ok(())
    }
    Some(Response::Error { message }) => Err(eyre!(message)),
    Some(_) => Err(eyre!("Unexpected response from server")),
    None => Err(eyre!("Model reset server unavailable")),
  }
}

fn model_reset_embedded() -> Result<()> {
  reset_model()?;
  eprintln!("Reranker model reset");
  Ok(())
}
