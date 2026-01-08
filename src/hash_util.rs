use std::hash::BuildHasher;

pub(crate) fn stable_hash(input: &str) -> u64 {
  let state = foldhash::fast::FixedState::default();
  state.hash_one(input)
}
