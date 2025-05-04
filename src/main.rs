use color_eyre::eyre::Result;
use human_panic::setup_panic;
use tracing::info;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use zage::db;

fn main() -> Result<()> {
  setup_panic!();
  color_eyre::install()?;

  // Initialize tracing
  fmt().with_env_filter(EnvFilter::from_default_env()).init();

  info!("Starting application");
  db::init(
    dirs::data_dir()
      .map(|v| v.join("zage/zage.db"))
      .unwrap_or(".zage/zage.db".into()),
  )?;

  println!("Hello, world!");

  Ok(())
}
