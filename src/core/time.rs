use std::time::{SystemTime, UNIX_EPOCH};

pub trait TimeProvider: Send + Sync {
  fn now(&self) -> i64;
}

#[derive(Debug, Clone, Default)]
pub struct SystemTimeProvider;

impl TimeProvider for SystemTimeProvider {
  fn now(&self) -> i64 {
    SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap_or_default()
      .as_secs() as i64
  }
}

#[derive(Debug, Clone)]
pub struct FixedTimeProvider {
  now: i64,
}

impl FixedTimeProvider {
  pub fn new(now: i64) -> Self {
    Self { now }
  }
}

impl TimeProvider for FixedTimeProvider {
  fn now(&self) -> i64 {
    self.now
  }
}
