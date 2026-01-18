mod err;
pub use err::*;
mod config;
mod hash_util;
pub use config::*;
pub mod cli;
pub mod core;
pub mod db;
pub mod embeddings;
pub mod indexer;
pub mod online_model;
pub mod predict;
#[cfg(feature = "pprof")]
mod profile;
pub mod sequence;
pub mod server;
pub mod service;
pub mod shell_history;
pub mod tokenize;
pub mod workspace;

#[cfg(feature = "pprof")]
pub(crate) use profile::capture_profile;

#[cfg(test)]
mod corpus_tests;

#[cfg(test)]
mod tests {
  use ctor::ctor;
  use tracing_subscriber::prelude::*;

  #[ctor]
  fn init_color_backtrace() {
    // let console_layer = console_subscriber::spawn();

    let env_filter = tracing_subscriber::EnvFilter::from_default_env();
    let subscriber = tracing_subscriber::fmt::layer()
      .pretty()
      .with_test_writer()
      .with_filter(env_filter);

    tracing_subscriber::registry()
      .with(subscriber)
      // .with(console_layer)
      .init();
    color_backtrace::install();
  }
}
