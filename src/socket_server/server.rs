use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use threadpool::ThreadPool;
use tracing::{error, info};

use crate::{Result, ZageError};

use crate::embedding::Embedder;
use crate::embedding::ProtocolMessage;

/// Socket server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
  /// Path to the Unix domain socket
  pub socket_path: String,
  /// Number of worker threads
  pub num_threads: usize,
  /// Connection timeout in seconds
  pub timeout_secs: u64,
}

impl Default for ServerConfig {
  fn default() -> Self {
    Self {
      socket_path: "/tmp/zage_embedder.sock".into(),
      num_threads: num_cpus::get(),
      timeout_secs: 30,
    }
  }
}

/// Socket server for handling embedding requests
pub struct SocketServer {
  config: ServerConfig,
  embedder: Arc<dyn Embedder>,
  pool: ThreadPool,
}

impl SocketServer {
  /// Create a new socket server with the given configuration
  pub fn new(config: ServerConfig, embedder: Arc<dyn Embedder>) -> Self {
    let pool = ThreadPool::new(config.num_threads);

    Self {
      config,
      embedder,
      pool,
    }
  }

  /// Start the server and listen for connections
  pub fn start(&self) -> Result<()> {
    // Remove socket file if it already exists
    let socket_path = Path::new(&self.config.socket_path);
    if socket_path.exists() {
      std::fs::remove_file(socket_path)?;
    }

    // Create the listener
    let listener = UnixListener::bind(&self.config.socket_path)?;
    info!("Server listening on {}", self.config.socket_path);

    // Set socket permissions to allow other users to connect
    #[cfg(unix)]
    {
      use std::os::unix::fs::PermissionsExt;
      let perms = std::fs::Permissions::from_mode(0o666);
      std::fs::set_permissions(&self.config.socket_path, perms)?;
    }

    // Accept connections
    for stream in listener.incoming() {
      match stream {
        Ok(stream) => {
          // Set read timeout
          stream.set_read_timeout(Some(Duration::from_secs(self.config.timeout_secs)))?;
          stream.set_write_timeout(Some(Duration::from_secs(self.config.timeout_secs)))?;

          // Clone the embedder for the worker thread
          let embedder = Arc::clone(&self.embedder);

          // Process the connection in a worker thread
          self.pool.execute(move || {
            if let Err(e) = handle_client(stream, embedder) {
              error!("Error handling client: {}", e);
            }
          });
        }
        Err(e) => {
          error!("Error accepting connection: {}", e);
        }
      }
    }

    Ok(())
  }
}

/// Handle a client connection
fn handle_client(mut stream: UnixStream, embedder: Arc<dyn Embedder>) -> Result<()> {
  // Read the request message
  let request = ProtocolMessage::read_from(&mut stream)?;

  // Process based on message type
  match request {
    ProtocolMessage::EmbedRequest(text) => {
      // Extract the text to embed
      let text = text;

      // Process the embedding request
      match embedder.embed(&text) {
        Ok(embedding) => {
          // Create and send the successful response
          let response = ProtocolMessage::EmbedResponse(embedding);
          response.write_to(&mut stream)?
        }
        Err(e) => {
          // Create and send the error response
          let error_msg = format!("Embedding error: {}", e);
          let response = ProtocolMessage::ErrorResponse(error_msg);
          response.write_to(&mut stream)?
        }
      }
    }
    _ => {
      return Err(ZageError::ConfigError(format!(
        "Unexpected message type: {:?}",
        request
      )));
    }
  }

  Ok(())
}
