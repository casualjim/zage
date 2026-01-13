use color_eyre::eyre::{Result, eyre};
use tracing::info;

use crate::cli::BackendRef;
use crate::db::{OnlineFeedbackEvent, upsert_online_feedback};
use crate::server::{self, Request, Response};

pub async fn run(
  backend: BackendRef<'_>,
  shown_id: String,
  shown_at: i64,
  working_directory: Option<String>,
  suggestion: String,
  accepted_command: Option<String>,
  accepted_at: Option<i64>,
  outcome: Option<String>,
) -> Result<()> {
  info!("Recording feedback event");
  match backend {
    BackendRef::Server => {
      let req = Request::Feedback {
        shown_id,
        shown_at,
        working_directory,
        suggestion,
        accepted_command,
        accepted_at,
        outcome,
      };
      match server::try_request(req).await? {
        Some(Response::Ack) => Ok(()),
        Some(Response::Error { message }) => Err(eyre!(message)),
        Some(_) => Err(eyre!("Unexpected response from server")),
        None => Err(eyre!("Feedback server unavailable")),
      }
    }
    BackendRef::Embedded(db) => {
      upsert_online_feedback(
        &db.conn,
        OnlineFeedbackEvent {
          shown_id,
          shown_at,
          cwd: working_directory,
          suggestion,
          accepted_command,
          accepted_at,
          outcome,
        },
      )
      .await?;
      Ok(())
    }
  }
}
