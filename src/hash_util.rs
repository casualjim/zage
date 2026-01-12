use std::hash::BuildHasher;

pub(crate) const SUBWORD_BUCKETS: u32 = 1 << 17;
pub(crate) const SUBWORD_NGRAM_MIN: usize = 3;
pub(crate) const SUBWORD_NGRAM_MAX: usize = 6;

pub(crate) fn stable_hash(input: &str) -> u64 {
  let state = foldhash::fast::FixedState::default();
  state.hash_one(input)
}

pub(crate) fn stable_bucket_and_sign(input: &str, buckets_pow2: u32) -> (u32, f32) {
  debug_assert!(buckets_pow2.is_power_of_two());
  let h = stable_hash(input);
  let bucket = (h as u32) & (buckets_pow2 - 1);
  let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
  (bucket, sign)
}

pub(crate) fn stable_char_ngrams_buckets(
  token: &str,
  buckets_pow2: u32,
  scratch_indices: &mut Vec<usize>,
  out: &mut Vec<(u32, f32)>,
) {
  debug_assert!(buckets_pow2.is_power_of_two());
  out.clear();
  scratch_indices.clear();

  if token.is_empty() {
    return;
  }

  scratch_indices.extend(token.char_indices().map(|(idx, _)| idx));
  scratch_indices.push(token.len());
  let char_len = scratch_indices.len().saturating_sub(1);

  if char_len < SUBWORD_NGRAM_MIN {
    return;
  }

  for start in 0..char_len {
    for n in SUBWORD_NGRAM_MIN..=SUBWORD_NGRAM_MAX {
      let end = start + n;
      if end > char_len {
        break;
      }
      let slice = &token[scratch_indices[start]..scratch_indices[end]];
      out.push(stable_bucket_and_sign(slice, buckets_pow2));
    }
  }
}
