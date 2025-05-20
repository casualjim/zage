use std::path::PathBuf;

use candle_core::Device;
use candle_core::backend::BackendDevice;
use clap::{Parser, Subcommand};
use color_eyre::eyre::Result;
use dirs::home_dir;
use human_panic::setup_panic;
use tracing::debug;
use tracing::info;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::fmt;
use uzers;
use zage::db::{
  connect, get_recent_invocations, get_sequence_scores, import_history, init, insert_invocation,
};
use zage::model::sequence::SequenceScore;
use zage::model::sequence::analyze_and_store_sequences;
use zage::model::{PredictionModel, markov::MarkovChain, ngram::NGramModel};
use zage::shell_history::{Invocation, Shell, get_hostname, parse_bash_history, parse_zsh_history};
use zage::socket_server::ServerConfig;
use zage::socket_server::SocketServer;

/// CLI for Zage
#[derive(Parser)]
#[clap(author, version, about)]
struct Cli {
  #[command(subcommand)]
  command: Option<Commands>,
  /// Path to the SQLite database file (overrides default location)
  #[clap(long, env = "ZAGE_DB_PATH")]
  db_path: Option<PathBuf>,
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
    #[arg(long, default_value = "markov")]
    model_type: String,

    /// N-gram size for ngram model (default: 2)
    #[arg(long, default_value = "2")]
    n: usize,

    /// Maximum number of commands to use for training
    #[arg(long, default_value = "1000")]
    max_commands: usize,

    /// Enable context (directory, hostname, etc.) for predictions
    #[arg(long, default_value = "true")]
    use_context: bool,
  },

  /// Predict next command
  Predict {
    /// Number of predictions to return
    #[arg(short, long, default_value = "5")]
    count: usize,

    /// Model type to use (default: markov)
    #[arg(long, default_value = "markov")]
    model_type: String,

    /// N-gram size for ngram model (default: 2)
    #[arg(long, default_value = "2")]
    n: usize,

    /// Show prediction probabilities
    #[arg(short, long)]
    show_probability: bool,

    /// Skip sequence detection in predict
    #[arg(long)]
    skip_sequence: bool,
  },

  /// Show model statistics
  Stats {
    /// Model type to use (default: markov)
    #[arg(long, default_value = "markov")]
    model_type: String,

    /// N-gram size for ngram model (default: 2)
    #[arg(long, default_value = "2")]
    n: usize,
  },

  /// Analyze and store frequent command sequences
  AnalyzeSequences {
    /// Minimum support count
    #[arg(long, default_value = "2")]
    min_support: usize,
    /// Minimum confidence threshold
    #[arg(long, default_value = "0.5")]
    min_confidence: f64,
    /// Minimum lift threshold
    #[arg(long, default_value = "1.5")]
    min_lift: f64,
  },

  /// (Internal) Record a single command invocation (used by shell hooks)
  Record {
    /// The command string that was executed
    #[arg(long)]
    command: String,
    /// The working directory where the command was executed
    #[arg(long)]
    working_directory: String,
    /// The exit status of the command
    #[arg(long)]
    exit_status: i64,
    /// The timestamp when the command started (Unix epoch seconds)
    #[arg(long)]
    start_timestamp: i64,
    /// The timestamp when the command finished (Unix epoch seconds)
    #[arg(long)]
    end_timestamp: i64,
    /// The shell session ID
    #[arg(long)]
    session_id: Option<i64>, // Optional for now
  },

  /// Run as a Unix domain socket server for embedding requests
  Server {
    /// Path to the Unix domain socket
    #[arg(long, default_value = "/tmp/zage_embedder.sock")]
    socket_path: String,

    /// Number of worker threads
    #[arg(long, default_value_os = num_cpus::get().to_string())]
    num_threads: usize,

    /// Connection timeout in seconds
    #[arg(long, default_value = "30")]
    timeout_secs: u64,

    /// Device to use for embeddings (cpu, metal, cuda)
    #[arg(long, default_value = "cpu")]
    device: String,
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
  let db_path = if let Some(path) = &cli.db_path {
    path.clone()
  } else if let Ok(env_path) = std::env::var("ZAGE_DB_PATH") {
    PathBuf::from(env_path)
  } else {
    dirs::data_dir()
      .map(|v| v.join("zage/zage.db"))
      .expect("Could not determine data directory")
  };
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
      let hostname_s = hostname.clone();
      let username_s = username.clone();
      let invocations = match shell {
        Shell::Zsh => parse_zsh_history(&history_file, hostname_s.clone(), username_s.clone())?,
        Shell::Bash => parse_bash_history(&history_file, hostname_s, username_s)?,
      };
      import_history(&mut conn, invocations)?;
      info!("Imported history from {:?}", history_file);
    }

    Some(Commands::Train {
      model_type,
      n,
      max_commands,
      use_context,
    }) => {
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
          model.set_use_context(*use_context);

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
        "markov" => {
          let mut model = MarkovChain::load_from_db(&mut conn)?;
          model.train(invocations)?;
          model.save_to_db(&mut conn)?;
          println!("Markov model trained and saved to database");
        }
        _ => {
          println!("Unsupported model type: {}", model_type);
        }
      }
    }

    Some(Commands::Predict {
      count,
      model_type,
      n,
      show_probability,
      skip_sequence,
    }) => {
      info!("Predicting next command using {} model", model_type);

      let mut conn = connect(&db_path)?;
      let recent_invocations = get_recent_invocations(&mut conn, 10)?;

      if recent_invocations.is_empty() {
        println!("No command history found. Please import history first.");
        std::process::exit(1);
      }

      match model_type.as_str() {
        "ngram" => {
          // Load the model from the database
          let model = NGramModel::load_from_db(&mut conn, *n)?;

          // Get raw predictions with probabilities for display
          let context: Vec<String> = recent_invocations
            .iter()
            .skip(recent_invocations.len().saturating_sub(model.n() - 1))
            .map(|inv| inv.command.clone())
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
        "markov" => {
          let model = MarkovChain::load_from_db(&mut conn)?;
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
        _ => {
          println!("Unsupported model type: {}", model_type);
        }
      }

      if !*skip_sequence {
        // Fetch a larger history for sequence detection
        let all_invocations = get_recent_invocations(&mut conn, 10000)?;
        let total_invocations = all_invocations.len();

        // --- Sequence Detection ---
        if total_invocations > 0 {
          info!("Running sequence detection...");

          // Define sequence detection parameters (use defaults for now)
          let seq_min_support = 2;
          let seq_min_confidence = 0.5;
          let seq_min_lift = 1.5;
          let top_n_sequences = 5;

          // Run SQL-based sequence analysis
          analyze_and_store_sequences(
            &mut conn,
            seq_min_support,
            seq_min_confidence,
            seq_min_lift,
          )?;
          let raw_scores = get_sequence_scores(&mut conn, top_n_sequences)?;

          // Convert raw scores to model scores
          let scores: Vec<SequenceScore> = raw_scores
            .iter()
            .filter_map(|raw| SequenceScore::from_raw(raw).ok())
            .collect();

          if !scores.is_empty() {
            println!(
              "\n--- Detected Command Sequences (Top {} by Lift) ---",
              top_n_sequences
            );
            for (i, s) in scores.iter().enumerate() {
              let seq_str = s.sequence.join(" → ");
              println!(
                "  {}. {} (S: {}, C: {:.2}, L: {:.2})",
                i + 1,
                seq_str,
                s.support,
                s.confidence,
                s.lift
              );
            }
          } else {
            println!("\n--- No command sequences detected with current thresholds ---");
          }
        } else {
          println!("\n--- Skipping sequence detection (no history) ---");
        }
      } else {
        info!("Skipping sequence detection per user request");
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

    Some(Commands::AnalyzeSequences {
      min_support,
      min_confidence,
      min_lift,
    }) => {
      info!(
        "Analyzing command sequences with support ≥ {}, confidence ≥ {}, lift ≥ {}",
        min_support, min_confidence, min_lift
      );
      let mut conn = connect(&db_path)?;
      analyze_and_store_sequences(&mut conn, *min_support, *min_confidence, *min_lift)?;
      println!("Sequence analysis complete and stored in sequence_scores table.");
    }

    Some(Commands::Record {
      command,
      working_directory,
      exit_status,
      start_timestamp,
      end_timestamp,
      session_id,
    }) => {
      info!("Recording command invocation");

      let mut conn = connect(&db_path)?;
      let mut tx = conn.transaction()?;

      // Get hostname and username (best effort)
      let hostname = get_hostname();
      let username = uzers::get_current_username()
        .map(|v| v.to_string_lossy().into_owned())
        .unwrap_or_else(|| "unknown".to_string());

      // Generate a session ID if none provided (e.g., using PID)
      // For now, use a placeholder or default.
      // A better approach might involve Zsh sending a unique session ID.
      let session_id = session_id.unwrap_or_else(|| std::process::id() as i64);

      // Create Invocation struct
      let invocation = Invocation {
        command: command.clone(),
        shellname: "zsh".to_string(), // Assume zsh for now
        working_directory: Some(working_directory.clone()),
        hostname: Some(hostname.clone()),
        username: Some(username.clone()),
        exit_status: Some(*exit_status),
        start_unix_timestamp: Some(*start_timestamp),
        end_unix_timestamp: Some(*end_timestamp),
        session_id,
      };

      // Record the command invocation
      debug!("Inserting invocation: {:?}", invocation);
      if let Err(e) = insert_invocation(&mut tx, &invocation) {
        // Handle potential duplicate errors gracefully if needed
        if e.to_string().contains("UNIQUE constraint failed") {
          info!("Duplicate invocation skipped: {:?}", invocation.command);
        } else {
          // Re-throw other errors
          return Err(e.into());
        }
      } else {
        info!("Invocation recorded successfully.");
      }

      tx.commit()?;
    }

    Some(Commands::Server {
      socket_path,
      num_threads,
      timeout_secs,
      device,
    }) => {
      info!("Starting socket server on {}", socket_path);

      // Create the device based on the user's choice
      let device = match device.to_lowercase().as_str() {
        "cpu" => Device::Cpu,
        "metal" => {
          #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
          {
            Device::Metal(candle_core::MetalDevice::new(0).expect("Failed to create Metal device"))
          }
          #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
          {
            println!("Metal device is only supported on macOS with Apple Silicon.");
            println!("Falling back to CPU device.");
            Device::Cpu
          }
        }
        "cuda" => {
          #[cfg(feature = "cuda")]
          {
            Device::Cuda(candle_core::CudaDevice::new(0).expect("Failed to create CUDA device"))
          }
          #[cfg(not(feature = "cuda"))]
          {
            println!("CUDA device is not supported in this build.");
            println!("Falling back to CPU device.");
            Device::Cpu
          }
        }
        _ => {
          println!("Unknown device: {}. Falling back to CPU device.", device);
          Device::Cpu
        }
      };

      // Initialize the embedder from the model module
      // This ensures we're using the intended abstraction (Embedder trait)
      let embedder = zage::model::create_embedder(device.clone())?;

      // Initialize and start the socket server in a new thread
      let server_config = ServerConfig {
        socket_path: socket_path.clone(),
        num_threads: *num_threads,
        timeout_secs: *timeout_secs,
      };

      // Create and start the server
      let server = SocketServer::new(server_config, embedder);
      server.start()?;
    }

    None => {
      // Default application behavior
      println!("Starting Zage application...");
      println!("Use --help to see available commands");
    }
  }

  Ok(())
}
