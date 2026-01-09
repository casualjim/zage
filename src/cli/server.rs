use std::path::Path;

use color_eyre::eyre::Result;

use crate::server;

pub async fn run(db_path: &Path) -> Result<()> {
  server::run_server(db_path)
    .await
    .map_err(|err| color_eyre::eyre::eyre!(err))
}
