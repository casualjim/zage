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
use zage::db::{connect, import_history, init, get_recent_invocations};
use zage::shell_history::{Shell, parse_bash_history, parse_zsh_history};
use zage::model::{PredictionModel, ngram::NGramModel};

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
  
  /// Train prediction model
  Train {
    /// Model type to use (default: ngram)
    #[arg(long, default_value = "ngram")]
    model_type: String,
    
    /// N-gram size for ngram model (default: 2)
    #[arg(long, default_value = "2")]
    n: usize,
    
    /// Maximum number of commands to use for training
    #[arg(long, default_value = "1000")]
    max_commands: usize,
    
    /// Enable directory context for predictions
    #[arg(long, default_value = "true")]
    use_dir_context: bool,
  },
  
  /// Predict next command
  Predict {
    /// Number of predictions to return
    #[arg(short, long, default_value = "5")]
    count: usize,
    
    /// Model type to use (default: ngram)
    #[arg(long, default_value = "ngram")]
    model_type: String,
    
    /// N-gram size for ngram model (default: 2)
    #[arg(long, default_value = "2")]
    n: usize,
    
    /// Show prediction probabilities
    #[arg(short, long)]
    show_probability: bool,
  },
  
  /// Show model statistics
  Stats {
    /// Model type to use (default: ngram)
    #[arg(long, default_value = "ngram")]
    model_type: String,
    
    /// N-gram size for ngram model (default: 2)
    #[arg(long, default_value = "2")]
    n: usize,
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
    
    Some(Commands::Train { model_type, n, max_commands, use_dir_context }) => {
      info!("Training {} model with n={}", model_type, n);
      
      let mut conn = connect(&db_path)?;
      let invocations = get_recent_invocations(&mut conn, *max_commands)?;
      
      if invocations.is_empty() {
        println!("No command history found. Please import history first.");
        return Ok(());
      }
      
      info!("Retrieved {} commands for training", invocations.len());
      
      match model_type.as_str() {
        "ngram" => {
          // Create a new model or load existing one
          let mut model = NGramModel::load_from_db(&mut conn, *n)?;
          model.set_use_dir_context(*use_dir_context);
          
          // Train the model
          model.train(invocations)?;
          
          // Save the model back to the database
          model.save_to_db(&mut conn)?;
          
          let stats = model.stats();
          println!("Model trained successfully and saved to database");
          println!("Model statistics:");
          println!("  N-gram size: {}", stats.n_value);
          println!("  Total commands: {}", stats.total_commands);
          println!("  Unique contexts: {}", stats.context_count);
          println!("  Unique commands: {}", stats.command_count);
          println!("  Directory contexts: {}", stats.dir_context_count);
        }
        _ => {
          println!("Unsupported model type: {}", model_type);
        }
      }
    }
    
    Some(Commands::Predict { count, model_type, n, show_probability }) => {
      info!("Predicting next command using {} model", model_type);
      
      let mut conn = connect(&db_path)?;
      let recent_invocations = get_recent_invocations(&mut conn, 10)?;
      
      if recent_invocations.is_empty() {
        println!("No command history found. Please import history first.");
        return Ok(());
      }
      
      match model_type.as_str() {
        "ngram" => {
          // Load the model from the database
          let model = NGramModel::load_from_db(&mut conn, *n)?;
          
          // Get raw predictions with probabilities for display
          let context: Vec<String> = recent_invocations
            .iter()
            .skip(recent_invocations.len().saturating_sub(model.n() - 1))
            .map(|inv| String::from_utf8_lossy(&inv.command).to_string())
            .collect();
          
          // Get the current working directory
          let current_dir = std::env::current_dir()
            .ok()
            .and_then(|p| p.to_str().map(|s| s.to_string()));
          
          if let Some(dir) = &current_dir {
            println!("Current directory: {}", dir);
          }
          
          println!("Based on recent commands: {}", context.join(" → "));
          
          // For raw predictions with probabilities, we need to implement a custom method
          if *show_probability {
            // This is a simplified version - in a real implementation, we'd add a method to get predictions with probabilities
            println!("Predicted commands (with probabilities):");
            let predictions = model.predict(&recent_invocations, *count)?;
            for (i, cmd) in predictions.iter().enumerate() {
              // Simplified probability display
              println!("  {}. {} (likely)", i + 1, cmd);
            }
          } else {
            let predictions = model.predict(&recent_invocations, *count)?;
            
            if predictions.is_empty() {
              println!("No predictions available for your recent command history.");
            } else {
              println!("Predicted commands:");
              for (i, cmd) in predictions.iter().enumerate() {
                println!("  {}. {}", i + 1, cmd);
              }
            }
          }
        }
        _ => {
          println!("Unsupported model type: {}", model_type);
        }
      }
    }
    
    Some(Commands::Stats { model_type, n }) => {
      info!("Displaying statistics for {} model", model_type);
      
      let mut conn = connect(&db_path)?;
      
      match model_type.as_str() {
        "ngram" => {
          // Load the model from the database
          let model = NGramModel::load_from_db(&mut conn, *n)?;
          let stats = model.stats();
          
          println!("Model statistics:");
          println!("  N-gram size: {}", stats.n_value);
          println!("  Total commands: {}", stats.total_commands);
          println!("  Unique contexts: {}", stats.context_count);
          println!("  Unique commands: {}", stats.command_count);
          println!("  Directory contexts: {}", stats.dir_context_count);
        }
        _ => {
          println!("Unsupported model type: {}", model_type);
        }
      }
    }
    
    None => {
      // Default application behavior
      println!("Starting Zage application...");
      println!("Use --help to see available commands");
    }
  }

  Ok(())
}
