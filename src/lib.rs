mod err;
pub use err::*;
mod config;
pub use config::*;
pub mod db;
pub mod model;
pub mod shell_history;

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
