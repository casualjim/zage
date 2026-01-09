use clap::Parser;
use color_eyre::eyre::Result;

use zage::cli::{Cli, run};

#[tokio::main]
async fn main() -> Result<()> {
  let cli = Cli::parse();
  run(cli).await
}
