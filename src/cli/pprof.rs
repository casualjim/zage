use std::path::PathBuf;
use std::time::Duration;

use color_eyre::eyre::{Result, eyre};
use humantime::Duration as HumanDuration;

use crate::capture_profile;
use crate::cli::BackendRef;
use crate::server::{self, Request, Response};

pub async fn run(
  backend: BackendRef<'_>,
  duration: HumanDuration,
  frequency: u32,
  output: PathBuf,
) -> Result<()> {
  if frequency == 0 {
    return Err(eyre!("frequency must be greater than 0"));
  }
  let duration = Duration::from(duration);

  match backend {
    BackendRef::Server => run_server(duration, frequency, output).await,
    BackendRef::Embedded(_) => run_embedded(duration, frequency, output).await,
  }
}

async fn run_server(duration: Duration, frequency: u32, output: PathBuf) -> Result<()> {
  let duration_ms = u64::try_from(duration.as_millis()).unwrap_or(u64::MAX);
  let request = Request::Pprof {
    duration_ms,
    frequency,
    output: output.to_string_lossy().into_owned(),
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
    None => Err(eyre!("Pprof server unavailable")),
  }
}

async fn run_embedded(duration: Duration, frequency: u32, output: PathBuf) -> Result<()> {
  capture_profile(duration, frequency, &output).await?;
  eprintln!("Wrote profile to {}", output.display());
  Ok(())
}
