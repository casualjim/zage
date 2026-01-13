use std::collections::{HashMap, HashSet};

use libsql::Connection;
use rand::Rng;
use rand::rngs::StdRng;

use crate::Result;
use crate::hash_util::stable_hash;
use crate::tokenize::{extract_command_parts, tokenize_index};

#[derive(Debug, Clone)]
pub(crate) struct GlobalCommandPool {
  commands: Vec<String>,
  hashes: Vec<u64>,
  cum_weights: Vec<f64>,
  total_weight: f64,
  weight_by_hash: HashMap<u64, f64>,
}

impl GlobalCommandPool {
  pub(crate) async fn load(conn: &Connection) -> Result<Self> {
    // Global pool: top frequent commands.
    let mut rows = conn
      .query(
        "SELECT command, freq FROM command_stats ORDER BY freq DESC LIMIT 5000",
        (),
      )
      .await?;
    let mut commands = Vec::new();
    let mut hashes = Vec::new();
    let mut weights = Vec::new();

    while let Some(row) = rows.next().await? {
      let cmd: String = row.get(0)?;
      let freq: i64 = row.get(1)?;
      if cmd.trim().is_empty() {
        continue;
      }
      let w = (freq.max(1) as f64).powf(0.75);
      commands.push(cmd.clone());
      hashes.push(stable_hash(&cmd));
      weights.push(w);
    }

    let mut cum_weights = Vec::with_capacity(weights.len());
    let mut total = 0.0f64;
    let mut weight_by_hash = HashMap::new();
    for (idx, w) in weights.into_iter().enumerate() {
      total += w;
      cum_weights.push(total);
      if let Some(h) = hashes.get(idx) {
        weight_by_hash.insert(*h, w);
      }
    }

    Ok(Self {
      commands,
      hashes,
      cum_weights,
      total_weight: total,
      weight_by_hash,
    })
  }

  fn is_empty(&self) -> bool {
    self.commands.is_empty() || self.total_weight <= 0.0
  }

  fn sample_weighted(&self, rng: &mut StdRng) -> Option<(String, u64)> {
    if self.is_empty() {
      return None;
    }
    let r = rng.random_range(0.0..self.total_weight);
    let idx = match self.cum_weights.binary_search_by(|w| w.total_cmp(&r)) {
      Ok(i) => i,
      Err(i) => i,
    };
    let cmd = self.commands.get(idx)?.clone();
    let hash = *self.hashes.get(idx)?;
    Some((cmd, hash))
  }

  fn prob(&self, hash: u64) -> f64 {
    if self.total_weight <= 0.0 {
      return 0.0;
    }
    let Some(w) = self.weight_by_hash.get(&hash) else {
      return 0.0;
    };
    *w / self.total_weight
  }
}

pub(crate) struct SamplerPools {
  pub workspace: Vec<String>,
  pub workspace_hashes: HashSet<u64>,
  pub head: Vec<String>,
  pub head_hashes: HashSet<u64>,
  pub recent: Vec<String>,
  pub recent_hashes: HashSet<u64>,
}

pub(crate) struct NegativeSampler<'a> {
  global: &'a GlobalCommandPool,
}

pub(crate) struct NegativeSample {
  pub cmd_buckets: Vec<(u32, f32)>,
  pub log_q: f32,
}

impl<'a> NegativeSampler<'a> {
  pub(crate) fn new(global: &'a GlobalCommandPool) -> Self {
    Self { global }
  }

  pub(crate) async fn build_pools(
    &mut self,
    conn: &Connection,
    shellname: &str,
    repo_root: &str,
    cwd: Option<&str>,
    recent_commands: &[String],
    positive_head_hash: u64,
  ) -> Result<SamplerPools> {
    let workspace = if !repo_root.is_empty() {
      load_workspace_repo_commands(conn, repo_root).await?
    } else if let Some(cwd) = cwd.filter(|v| !v.is_empty()) {
      load_workspace_cwd_commands(conn, cwd).await?
    } else {
      Vec::new()
    };
    let workspace_hashes = workspace
      .iter()
      .map(|c| stable_hash(c))
      .collect::<HashSet<_>>();

    // Head pool: filter a bounded candidate set by head hash.
    let mut head = Vec::new();
    for cmd in workspace.iter().take(500) {
      if head_hash_for_command(shellname, cmd) == Some(positive_head_hash) {
        head.push(cmd.clone());
      }
    }
    if head.len() < 50 {
      // Backfill from global, but keep bounded.
      for cmd in self.global.commands.iter().take(1000) {
        if head_hash_for_command(shellname, cmd) == Some(positive_head_hash) {
          head.push(cmd.clone());
          if head.len() >= 500 {
            break;
          }
        }
      }
    }
    head.sort();
    head.dedup();
    let head_hashes = head.iter().map(|c| stable_hash(c)).collect::<HashSet<_>>();

    let mut recent = recent_commands.to_vec();
    recent.sort();
    recent.dedup();
    let recent_hashes = recent
      .iter()
      .map(|c| stable_hash(c))
      .collect::<HashSet<_>>();

    Ok(SamplerPools {
      workspace,
      workspace_hashes,
      head,
      head_hashes,
      recent,
      recent_hashes,
    })
  }

  pub(crate) fn sample_with_logq(
    &mut self,
    pools: &SamplerPools,
    shellname: &str,
    positive_command_hash: u64,
    negatives: usize,
    rng: &mut StdRng,
  ) -> Result<(Vec<NegativeSample>, f32)> {
    let mut weights = Vec::new();
    let mut components = Vec::new();

    // Mixture weights from docs/online_next_command_prediction.md.
    if !pools.workspace.is_empty() {
      weights.push(0.4);
      components.push(Component::Workspace);
    }
    if !pools.head.is_empty() {
      weights.push(0.3);
      components.push(Component::Head);
    }
    if !pools.recent.is_empty() {
      weights.push(0.2);
      components.push(Component::Recent);
    }
    if !self.global.is_empty() {
      weights.push(0.1);
      components.push(Component::Global);
    }

    if components.is_empty() {
      return Ok((Vec::new(), (1e-12f64).ln() as f32));
    }

    // Normalize weights.
    let sum: f64 = weights.iter().sum();
    let weights: Vec<f64> = weights.into_iter().map(|w| w / sum).collect();
    let mut cum = Vec::with_capacity(weights.len());
    let mut acc = 0.0f64;
    for w in &weights {
      acc += *w;
      cum.push(acc);
    }

    let log_q_pos = log_q_for_hash(
      positive_command_hash,
      pools,
      &weights,
      &components,
      self.global,
    );

    let mut selected: HashSet<u64> = HashSet::new();
    selected.insert(positive_command_hash);
    let mut out = Vec::new();
    let mut attempts = 0usize;
    let max_attempts = negatives.saturating_mul(50).max(100);

    while out.len() < negatives && attempts < max_attempts {
      attempts += 1;
      let r = rng.random::<f64>();
      let idx = cum.iter().position(|p| r <= *p).unwrap_or(cum.len() - 1);
      let component = components[idx];

      let (cmd, hash) = match component {
        Component::Workspace => {
          sample_uniform(&pools.workspace, rng).map(|c| (c.clone(), stable_hash(c)))
        }
        Component::Head => sample_uniform(&pools.head, rng).map(|c| (c.clone(), stable_hash(c))),
        Component::Recent => {
          sample_uniform(&pools.recent, rng).map(|c| (c.clone(), stable_hash(c)))
        }
        Component::Global => self.global.sample_weighted(rng),
      }
      .unwrap_or_else(|| (String::new(), 0));

      if cmd.is_empty() || hash == 0 || selected.contains(&hash) {
        continue;
      }
      selected.insert(hash);

      let cmd_buckets = command_buckets(shellname, &cmd);
      if cmd_buckets.is_empty() {
        continue;
      }
      let log_q = log_q_for_hash(hash, pools, &weights, &components, self.global);
      out.push(NegativeSample { cmd_buckets, log_q });
    }

    Ok((out, log_q_pos))
  }
}

#[derive(Clone, Copy, Debug)]
enum Component {
  Workspace,
  Head,
  Recent,
  Global,
}

fn log_q_for_hash(
  hash: u64,
  pools: &SamplerPools,
  weights: &[f64],
  components: &[Component],
  global: &GlobalCommandPool,
) -> f32 {
  let mut q = 0.0f64;
  for (idx, component) in components.iter().enumerate() {
    let w = weights.get(idx).copied().unwrap_or(0.0);
    let p = match component {
      Component::Workspace => {
        if pools.workspace.is_empty() || !pools.workspace_hashes.contains(&hash) {
          0.0
        } else {
          1.0 / (pools.workspace.len() as f64)
        }
      }
      Component::Head => {
        if pools.head.is_empty() || !pools.head_hashes.contains(&hash) {
          0.0
        } else {
          1.0 / (pools.head.len() as f64)
        }
      }
      Component::Recent => {
        if pools.recent.is_empty() || !pools.recent_hashes.contains(&hash) {
          0.0
        } else {
          1.0 / (pools.recent.len() as f64)
        }
      }
      Component::Global => global.prob(hash),
    };
    q += w * p;
  }
  (q.max(1e-12)).ln() as f32
}

fn sample_uniform<'a>(list: &'a [String], rng: &mut StdRng) -> Option<&'a String> {
  if list.is_empty() {
    return None;
  }
  let idx = rng.random_range(0..list.len());
  list.get(idx)
}

fn head_hash_for_command(shellname: &str, command: &str) -> Option<u64> {
  let tokens = tokenize_index(shellname, command);
  let parts = extract_command_parts(command, &tokens)?;
  if parts.head.trim().is_empty() {
    return None;
  }
  Some(stable_hash(parts.head.trim()))
}

fn command_buckets(shellname: &str, command: &str) -> Vec<(u32, f32)> {
  let mut map = HashMap::<u32, f32>::new();
  let mut scratch_indices = Vec::new();
  let mut scratch = Vec::new();
  for tok in super::command_tokens(shellname, command) {
    crate::hash_util::stable_char_ngrams_buckets(
      &tok,
      crate::hash_util::SUBWORD_BUCKETS,
      &mut scratch_indices,
      &mut scratch,
    );
    for (bucket, sign) in scratch.iter().copied() {
      *map.entry(bucket).or_insert(0.0) += sign;
    }
  }
  let mut out: Vec<(u32, f32)> = map.into_iter().collect();
  out.sort_by(|a, b| a.0.cmp(&b.0));
  out
}

async fn load_workspace_repo_commands(conn: &Connection, repo_root: &str) -> Result<Vec<String>> {
  let mut rows = conn
    .query(
      "SELECT command FROM repo_command_stats WHERE repo_root = ? ORDER BY freq DESC LIMIT 2000",
      libsql::params![repo_root.to_string()],
    )
    .await?;
  let mut out = Vec::new();
  while let Some(row) = rows.next().await? {
    let cmd: String = row.get(0)?;
    out.push(cmd);
  }
  Ok(out)
}

async fn load_workspace_cwd_commands(conn: &Connection, cwd: &str) -> Result<Vec<String>> {
  let mut rows = conn
    .query(
      "SELECT command FROM context_stats WHERE working_directory = ? ORDER BY freq DESC LIMIT 2000",
      libsql::params![cwd.to_string()],
    )
    .await?;
  let mut out = Vec::new();
  while let Some(row) = rows.next().await? {
    let cmd: String = row.get(0)?;
    out.push(cmd);
  }
  Ok(out)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn logq_is_finite_and_negative() {
    let pools = SamplerPools {
      workspace: vec!["a".to_string()],
      workspace_hashes: vec![stable_hash("a")].into_iter().collect(),
      head: vec!["b".to_string()],
      head_hashes: vec![stable_hash("b")].into_iter().collect(),
      recent: vec!["c".to_string()],
      recent_hashes: vec![stable_hash("c")].into_iter().collect(),
    };
    let global = GlobalCommandPool {
      commands: vec!["d".to_string()],
      hashes: vec![stable_hash("d")],
      cum_weights: vec![1.0],
      total_weight: 1.0,
      weight_by_hash: vec![(stable_hash("d"), 1.0)].into_iter().collect(),
    };
    let weights = vec![0.4, 0.3, 0.2, 0.1];
    let comps = vec![
      Component::Workspace,
      Component::Head,
      Component::Recent,
      Component::Global,
    ];
    let logq = log_q_for_hash(stable_hash("a"), &pools, &weights, &comps, &global);
    assert!(logq.is_finite());
    assert!(logq <= 0.0);
  }
}
