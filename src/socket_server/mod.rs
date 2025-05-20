//! Unix domain socket server for the embedding model.
//!
//! This module implements a server that listens on a Unix domain socket
//! and processes embedding requests using a thread pool.

mod server;

// Re-export what's needed for main.rs to manage the server
pub use self::server::ServerConfig;
pub use self::server::SocketServer;
