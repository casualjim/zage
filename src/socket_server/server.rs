use std::io::{Read as _, Write as _};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use threadpool::ThreadPool;
use tracing::{error, info};

use crate::model::pretrained_embedder::PretrainedEmbedder;
use crate::{Result, ZageError};

use super::MessageType;
use super::encoder::{LengthDelimitedDecoder, LengthDelimitedEncoder};

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
      socket_path: "/tmp/zage_embedder.sock".to_string(),
      num_threads: 4,
      timeout_secs: 30,
    }
  }
}

/// Socket server for handling embedding requests
pub struct SocketServer {
  config: ServerConfig,
  embedder: Arc<PretrainedEmbedder>,
  pool: ThreadPool,
}

impl SocketServer {
  /// Create a new socket server with the given configuration
  pub fn new(config: ServerConfig, embedder: PretrainedEmbedder) -> Self {
    let embedder = Arc::new(embedder);
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
fn handle_client(mut stream: UnixStream, embedder: Arc<PretrainedEmbedder>) -> Result<()> {
  // Read the message type
  let mut type_buf = [0u8; 1];
  stream.read_exact(&mut type_buf)?;
  let msg_type = MessageType::from_byte(type_buf[0])?;

  match msg_type {
    MessageType::EmbedRequest => {
      // Read the message length (4 bytes, little endian)
      let mut len_buf = [0u8; 4];
      stream.read_exact(&mut len_buf)?;
      let msg_len = u32::from_le_bytes(len_buf) as usize;

      // Read the RLE-encoded message
      let mut rle_buf = vec![0u8; msg_len];
      stream.read_exact(&mut rle_buf)?;

      // Decode the message
      let mut decoder = LengthDelimitedDecoder::new(&rle_buf);
      let text = decoder.decode_string()?;

      // Process the embedding request
      match embedder.embed(&text) {
        Ok(embedding) => {
          // Encode the response
          let mut encoder = LengthDelimitedEncoder::new();
          encoder.encode_f32_vec(&embedding);
          let encoded = encoder.finish();

          // Write the response type
          stream.write_all(&[MessageType::EmbedResponse.to_byte()])?;

          // Write the response length
          let len_bytes = (encoded.len() as u32).to_le_bytes();
          stream.write_all(&len_bytes)?;

          // Write the RLE-encoded response
          stream.write_all(&encoded)?;
          stream.flush()?;
        }
        Err(e) => {
          // Send error response
          let error_msg = format!("Embedding error: {}", e);
          let mut encoder = LengthDelimitedEncoder::new();
          encoder.encode_string(&error_msg);
          let encoded = encoder.finish();

          // Write the error response type
          stream.write_all(&[MessageType::ErrorResponse.to_byte()])?;

          // Write the error response length
          let len_bytes = (encoded.len() as u32).to_le_bytes();
          stream.write_all(&len_bytes)?;

          // Write the RLE-encoded error response
          stream.write_all(&encoded)?;
          stream.flush()?;
        }
      }
    }
    _ => {
      return Err(ZageError::ConfigError(format!(
        "Unexpected message type: {:?}",
        msg_type
      )));
    }
  }

  Ok(())
}
