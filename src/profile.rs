use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

use pprof::ProfilerGuard;
use pprof::protos::Message;

use crate::{Result, ZageError};

pub async fn capture_profile(duration: Duration, frequency: u32, output: &Path) -> Result<()> {
  let frequency = i32::try_from(frequency).map_err(|err| ZageError::GenericError(Box::new(err)))?;
  let guard =
    ProfilerGuard::new(frequency).map_err(|err| ZageError::GenericError(Box::new(err)))?;
  tokio::time::sleep(duration).await;

  let report = guard
    .report()
    .build()
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;
  let profile = report
    .pprof()
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;

  if let Some(parent) = output.parent()
    && !parent.as_os_str().is_empty()
  {
    fs::create_dir_all(parent)?;
  }

  let buf = profile.encode_to_vec();

  let mut file = fs::File::create(output)?;
  file.write_all(&buf)?;
  Ok(())
}
