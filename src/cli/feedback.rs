use color_eyre::eyre::{Result, eyre};
use tracing::info;

use crate::cli::BackendRef;
use crate::db::{OnlineFeedbackEvent, upsert_online_feedback};
use crate::predict::update_blend_weights_for_feedback;
use crate::server::{self, Request, Response};

pub struct FeedbackArgs {
  pub shown_id: String,
  pub shown_at: i64,
  pub working_directory: Option<String>,
  pub suggestion: String,
  pub accepted_command: Option<String>,
  pub accepted_at: Option<i64>,
  pub outcome: Option<String>,
}

pub async fn run(backend: BackendRef<'_>, args: FeedbackArgs) -> Result<()> {
  let FeedbackArgs {
    shown_id,
    shown_at,
    working_directory,
    suggestion,
    accepted_command,
    accepted_at,
    outcome,
  } = args;
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
      let event = OnlineFeedbackEvent {
        shown_id,
        shown_at,
        cwd: working_directory,
        suggestion,
        accepted_command,
        accepted_at,
        outcome,
      };
      upsert_online_feedback(&db.conn, event.clone()).await?;
      update_blend_weights_for_feedback(&db.conn, &event).await?;
      Ok(())
    }
  }
}
