use color_eyre::eyre::{Result, eyre};
use tracing::{debug, info};

use crate::cli::BackendRef;
use crate::db::{insert_invocation, update_stats_for_invocation};
use crate::predict::aliases::{expand_alias, load_aliases};
use crate::server::{self, Request, Response};
use crate::shell_history::{Invocation, detect_shellname, get_hostname};
use crate::workspace::detect_workspace_for_cwd;

pub async fn run(
  backend: BackendRef<'_>,
  command: String,
  working_directory: String,
  exit_status: i64,
  start_timestamp: i64,
  end_timestamp: i64,
  session_id: Option<i64>,
) -> Result<()> {
  info!("Recording command invocation");
  let aliases = load_aliases();
  let expanded_command = expand_alias(&command, &aliases).unwrap_or_else(|| command.clone());

  let server_req = Request::Record {
    command: command.clone(),
    expanded_command: expanded_command.clone(),
    working_directory: working_directory.clone(),
    exit_status: exit_status as i32,
    start_timestamp,
    end_timestamp,
    session_id: session_id.unwrap_or_else(|| std::process::id() as i64) as u64,
  };
  match backend {
    BackendRef::Server => match server::try_request(server_req).await? {
      Some(Response::Ack) => Ok(()),
      Some(Response::Error { message }) => Err(eyre!(message)),
      Some(_) => Err(eyre!("Unexpected response from server")),
      None => Err(eyre!("Record server unavailable")),
    },
    BackendRef::Embedded(db) => {
      let hostname = get_hostname();
      let username = uzers::get_current_username()
        .map(|v| v.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_string());
      let session_id = session_id.unwrap_or_else(|| std::process::id() as i64);
      let workspace =
        detect_workspace_for_cwd(&working_directory).map_err(|err| eyre!(err.to_string()))?;

      let invocation = Invocation {
        command: command.clone(),
        expanded_command: expanded_command.clone(),
        shellname: detect_shellname(),
        working_directory: Some(working_directory.clone()),
        workspace,
        hostname: Some(hostname.clone()),
        username: Some(username.clone()),
        exit_status: Some(exit_status),
        start_unix_timestamp: Some(start_timestamp),
        end_unix_timestamp: Some(end_timestamp),
        session_id,
      };

      debug!("Inserting invocation: {:?}", invocation);
      let inserted = insert_invocation(&db.conn, &invocation).await?;
      if inserted {
        update_stats_for_invocation(&db.conn, &invocation).await?;
        info!("Invocation recorded successfully.");
      } else {
        info!("Duplicate invocation skipped: {:?}", invocation.command);
      }

      Ok(())
    }
  }
}
