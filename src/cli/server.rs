use color_eyre::eyre::Result;

use crate::config::DbConfig;
use crate::server;

pub async fn run(db_config: &DbConfig) -> Result<()> {
  server::run_server(db_config)
    .await
    .map_err(|err| color_eyre::eyre::eyre!(err))
}
