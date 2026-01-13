use std::time::Duration;

use color_eyre::eyre::{Result, eyre};
use humantime::Duration as HumanDuration;

use crate::cli::BackendRef;
use crate::db::{Db, online_model_status, reset_online_model};
use crate::rerank::{TrainConfig, train_model};
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
    BackendRef::Embedded(db) => model_status_embedded(db).await,
  }
}

pub async fn model_reset(backend: BackendRef<'_>) -> Result<()> {
  match backend {
    BackendRef::Server => model_reset_server().await,
    BackendRef::Embedded(db) => model_reset_embedded(db).await,
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

async fn model_status_embedded(db: &Db) -> Result<()> {
  let status = online_model_status(&db.conn).await?;
  let config = crate::config::OnlineModelConfig::load()
    .unwrap_or(crate::config::OnlineModelConfig::default());
  let update_count = crate::db::online_model_update_count(&db.conn).await?;
  let last_update = crate::db::online_model_last_updated_at(&db.conn).await?;
  let replay_workspaces = crate::db::online_replay_workspace_roots(&db.conn).await?;
  let group_scalars = crate::db::online_model_group_scalars(&db.conn).await?;
  let head_biases = crate::db::online_model_head_biases(&db.conn, 8).await?;
  let warmed_up = status.token_embeddings > 0 || status.group_scalars > 0;

  eprintln!(
    "Online model: version={}, warmed_up={}, update_count={}, last_update={:?}",
    config.model_version(),
    warmed_up,
    update_count,
    last_update
  );
  eprintln!(
    "Replay: global={}, workspace={}, workspaces={}",
    status.replay_global, status.replay_workspace, replay_workspaces
  );
  eprintln!(
    "Tables: meta={}, token_embeddings={}, command_biases={}, head_biases={}, group_scalars={}, feedback={}",
    status.meta_entries,
    status.token_embeddings,
    status.command_biases,
    status.head_biases,
    status.group_scalars,
    status.feedback
  );
  if !group_scalars.is_empty() {
    let rendered = group_scalars
      .iter()
      .map(|(name, value)| format!("{name}={value:.3}"))
      .collect::<Vec<_>>()
      .join(", ");
    eprintln!("Group scalars: {rendered}");
  }
  if !head_biases.is_empty() {
    let rendered = head_biases
      .iter()
      .map(|(head, bias)| format!("{head}={bias:.3}"))
      .collect::<Vec<_>>()
      .join(", ");
    eprintln!("Top head biases: {rendered}");
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

async fn model_reset_embedded(db: &Db) -> Result<()> {
  reset_online_model(&db.conn).await?;
  eprintln!("Online model reset");
  Ok(())
}
