//! Unix domain socket server for the embedding model.
//!
//! This module implements a server that listens on a Unix domain socket
//! and processes embedding requests using a thread pool.

pub mod encoder;
mod messages;
mod server;

pub use messages::*;
pub use server::*;
