use std::path::PathBuf;

use bstr::BString;
use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;
use dirs::home_dir;
use human_panic::setup_panic;
use tracing::debug;
use tracing::info;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use zage::db::{connect, import_history, init};
use zage::shell_history::{Shell, parse_bash_history, parse_zsh_history};

/// CLI for Zage
#[derive(Parser)]
#[clap(author, version, about)]
struct Cli {
  #[command(subcommand)]
  command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
  /// Import shell history
  Import {
    /// Path to history file (defaults to $HISTFILE env var)
    #[arg(value_name = "FILE", env = "HISTFILE")]
    file: Option<PathBuf>,
    /// Override hostname for import
    #[arg(long)]
    hostname: Option<String>,
    /// Override username for import
    #[arg(long)]
    username: Option<String>,
    /// Shell type (bash or zsh); defaults to $SHELL env var basename
    #[arg(long, env = "SHELL")]
    shell: Shell,
  },
}

fn main() -> Result<()> {
  let cli = Cli::parse();
  setup_panic!();
  color_eyre::install()?;

  // Initialize tracing
  fmt().with_env_filter(EnvFilter::from_default_env()).init();

  // Debug environment variables
  let histfile = std::env::var("HISTFILE").unwrap_or_else(|_| "NOT SET".to_string());
  let shell = std::env::var("SHELL").unwrap_or_else(|_| "NOT SET".to_string());
  info!(
    "Environment variables: HISTFILE={}, SHELL={}",
    histfile, shell
  );

  info!("Starting application");
  // Determine DB path
  let db_path = dirs::data_dir()
    .map(|v| v.join("zage/zage.db"))
    .unwrap_or_else(|| ".zage/zage.db".into());
  debug!("Initializing db at {}", db_path.display());
  std::fs::create_dir_all(db_path.parent().unwrap())?;
  init(&db_path)?;

  match &cli.command {
    Some(Commands::Import {
      file,
      hostname,
      username,
      shell,
    }) => {
      // Import history and exit
      debug!("File argument value: {:?}", file);

      let history_file = file.clone().unwrap_or_else(|| {
        debug!("No file specified, falling back to default path");
        let mut path = home_dir().expect("Cannot find home directory");
        let filename = match shell {
          Shell::Zsh => ".zsh_history",
          Shell::Bash => ".bash_history",
        };
        path.push(filename);
        debug!("Using default history file path: {:?}", path);
        path
      });

      let mut conn = connect(&db_path)?;
      let hostname_bs = hostname.clone().map(|h| BString::from(h.into_bytes()));
      let username_bs = username.clone().map(|u| BString::from(u.into_bytes()));
      let invocations = match shell {
        Shell::Zsh => parse_zsh_history(&history_file, hostname_bs.clone(), username_bs.clone())?,
        Shell::Bash => parse_bash_history(&history_file, hostname_bs, username_bs)?,
      };
      import_history(&mut conn, invocations)?;
      info!("Imported history from {:?}", history_file);
    }
    None => {
      // Default application behavior
      println!("Starting Zage application...");
    }
  }

  Ok(())
}
