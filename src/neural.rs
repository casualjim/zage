use std::path::PathBuf;

use burn::backend::{Autodiff, Wgpu};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, Relu};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use libsql::Connection;
use serde::{Deserialize, Serialize};

use crate::hash_util::stable_hash;
use crate::tokenize::{extract_command_parts, tokenize_index};
use crate::{Result, ZageError};

pub const MODEL_NAME: &str = "neural_command_biencoder";
const MODEL_CONFIG_NAME: &str = "neural_command_biencoder_config.json";
const DEFAULT_SHELLNAME: &str = "zsh";

pub type LoadedBiEncoder = (BiEncoder<Wgpu<f32, i32>>, NeuralTrainConfig);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralTrainConfig {
  pub epochs: usize,
  pub batch_size: usize,
  pub learning_rate: f64,
  pub window: usize,
  pub vocab_size: usize,
  pub max_seq_len: usize,
  pub embed_dim: usize,
  pub projection_dim: usize,
  pub temperature: f64,
  pub seed: u64,
}

impl Default for NeuralTrainConfig {
  fn default() -> Self {
    Self {
      epochs: 5,
      batch_size: 256,
      learning_rate: 1e-3,
      window: 10,
      vocab_size: 65_536,
      max_seq_len: 256,
      embed_dim: 128,
      projection_dim: 128,
      temperature: 0.07,
      seed: 42,
    }
  }
}

#[derive(Module, Debug)]
pub struct TokenEncoder<B: Backend> {
  tok: Embedding<B>,
  pos: Embedding<B>,
  dropout: Dropout,
  proj1: Linear<B>,
  act: Relu,
  proj2: Linear<B>,
}

impl<B: Backend> TokenEncoder<B> {
  pub fn init(config: &NeuralTrainConfig, device: &B::Device) -> Self {
    Self {
      tok: EmbeddingConfig::new(config.vocab_size, config.embed_dim).init(device),
      pos: EmbeddingConfig::new(config.max_seq_len, config.embed_dim).init(device),
      dropout: DropoutConfig::new(0.1).init(),
      proj1: LinearConfig::new(config.embed_dim, config.projection_dim).init(device),
      act: Relu::new(),
      proj2: LinearConfig::new(config.projection_dim, config.projection_dim).init(device),
    }
  }

  pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
    let [batch, seq_len] = token_ids.dims();
    let mut pos_raw = Vec::with_capacity(batch * seq_len);
    for _ in 0..batch {
      for idx in 0..seq_len {
        pos_raw.push(idx as i64);
      }
    }
    let pos_ids = Tensor::<B, 2, Int>::from_ints(pos_raw.as_slice(), &token_ids.device())
      .reshape([batch, seq_len]);

    let x = self.tok.forward(token_ids) + self.pos.forward(pos_ids);
    let x = self.dropout.forward(x);
    let x = x.mean_dim(1).squeeze_dim::<2>(1);
    let x = self.proj1.forward(x);
    let x = self.act.forward(x);
    self.proj2.forward(x)
  }
}

#[derive(Module, Debug)]
pub struct BiEncoder<B: Backend> {
  encoder: TokenEncoder<B>,
}

impl<B: Backend> BiEncoder<B> {
  pub fn init(config: &NeuralTrainConfig, device: &B::Device) -> Self {
    Self {
      encoder: TokenEncoder::init(config, device),
    }
  }

  pub fn encode(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 2> {
    self.encoder.forward(token_ids)
  }
}

#[derive(Debug, Clone)]
struct HistoryRow {
  expanded_command: String,
  shellname: String,
  working_directory: Option<String>,
  workspace_root: Option<String>,
  hostname: Option<String>,
  username: Option<String>,
  exit_status: Option<i64>,
  start_ts: Option<i64>,
  session_id: Option<i64>,
}

pub async fn train_biencoder_wgpu(conn: &Connection, config: NeuralTrainConfig) -> Result<PathBuf> {
  type Base = Wgpu<f32, i32>;
  type Backend = Autodiff<Base>;

  let device = burn::backend::wgpu::WgpuDevice::default();
  Backend::seed(&device, config.seed);

  let rows = load_history(conn).await?;
  if rows.len() < config.window.saturating_add(2) {
    return Err(ZageError::ConfigError(format!(
      "not enough history to train: have {}, need > {}",
      rows.len(),
      config.window.saturating_add(2)
    )));
  }

  let mut model = BiEncoder::<Backend>::init(&config, &device);
  let mut optimizer = AdamConfig::new().init();
  let loss_fn = CrossEntropyLossConfig::new().init(&device);

  let batches = build_batches(&rows, &config)?;
  for epoch in 0..config.epochs {
    let mut total_loss = 0.0f32;
    let mut steps = 0usize;
    for batch in batches.iter() {
      let ctx_ids = Tensor::<Backend, 2, Int>::from_ints(batch.ctx_ids.as_slice(), &device)
        .reshape([batch.batch_size, config.max_seq_len]);
      let tgt_ids = Tensor::<Backend, 2, Int>::from_ints(batch.tgt_ids.as_slice(), &device)
        .reshape([batch.batch_size, config.max_seq_len]);

      let ctx_emb = model.encode(ctx_ids);
      let tgt_emb = model.encode(tgt_ids);

      let logits = ctx_emb.matmul(tgt_emb.transpose());
      let logits = logits / (config.temperature as f32);

      let targets = Tensor::<Backend, 1, Int>::from_ints(
        (0..batch.batch_size as i64).collect::<Vec<_>>().as_slice(),
        &device,
      );

      let loss = loss_fn.forward(logits, targets);
      total_loss += loss.clone().into_scalar();
      steps += 1;

      let grads = loss.backward();
      let grads = GradientsParams::from_grads(grads, &model);
      model = optimizer.step(config.learning_rate, model, grads);
    }
    let avg = if steps == 0 {
      0.0
    } else {
      total_loss / steps as f32
    };
    tracing::info!(
      "neural epoch {}/{} loss={:.4}",
      epoch + 1,
      config.epochs,
      avg
    );
  }

  let model_dir = model_dir()?;
  std::fs::create_dir_all(&model_dir)?;
  let model_path = model_dir.join(MODEL_NAME);

  let config_path = model_dir.join(MODEL_CONFIG_NAME);
  let config_json =
    serde_json::to_string_pretty(&config).map_err(|err| ZageError::GenericError(Box::new(err)))?;
  std::fs::write(&config_path, config_json)?;

  model
    .save_file(
      model_path.to_string_lossy().to_string(),
      &CompactRecorder::new(),
    )
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;

  Ok(model_path)
}

pub fn load_biencoder_wgpu() -> Result<Option<LoadedBiEncoder>> {
  type Backend = Wgpu<f32, i32>;

  let model_dir = model_dir()?;
  let model_path = model_dir.join(MODEL_NAME);
  if !model_path.exists() {
    return Ok(None);
  }

  let device = burn::backend::wgpu::WgpuDevice::default();
  let config_path = model_dir.join(MODEL_CONFIG_NAME);
  let config = if config_path.exists() {
    let raw = std::fs::read_to_string(&config_path)?;
    serde_json::from_str::<NeuralTrainConfig>(&raw)
      .map_err(|err| ZageError::ConfigError(err.to_string()))?
  } else {
    NeuralTrainConfig::default()
  };
  let record = CompactRecorder::new()
    .load(model_path.to_string_lossy().to_string().into(), &device)
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;

  let model = BiEncoder::<Backend>::init(&config, &device).load_record(record);
  Ok(Some((model, config)))
}

fn model_dir() -> Result<PathBuf> {
  if let Ok(path) = std::env::var("ZAGE_MODEL_PATH") {
    return Ok(PathBuf::from(path));
  }
  let base =
    dirs::data_dir().ok_or_else(|| ZageError::ConfigError("missing data dir".to_string()))?;
  Ok(base.join("zage/model"))
}

async fn load_history(conn: &Connection) -> Result<Vec<HistoryRow>> {
  let mut rows = conn
    .query(
      "SELECT expanded_command, shellname, working_directory, workspace_json, hostname, username, exit_status, start_unix_timestamp, session_id
       FROM shell_history
       WHERE expanded_command != ''
       ORDER BY COALESCE(start_unix_timestamp, 0) ASC, id ASC",
      (),
    )
    .await?;

  let mut out = Vec::new();
  while let Some(row) = rows.next().await? {
    let expanded_command = row.get::<String>(0)?;
    let shellname = row.get::<String>(1)?;
    let working_directory = row.get::<Option<String>>(2)?;
    let workspace_json = row.get::<Option<String>>(3)?;
    let hostname = row.get::<Option<String>>(4)?;
    let username = row.get::<Option<String>>(5)?;
    let exit_status = row.get::<Option<i64>>(6)?;
    let start_ts = row.get::<Option<i64>>(7)?;
    let session_id = row.get::<Option<i64>>(8)?;

    let workspace_root = workspace_json
      .as_deref()
      .and_then(|raw| serde_json::from_str::<crate::workspace::WorkspaceInfo>(raw).ok())
      .map(|ws| ws.root);

    out.push(HistoryRow {
      expanded_command,
      shellname,
      working_directory,
      workspace_root,
      hostname,
      username,
      exit_status,
      start_ts,
      session_id,
    });
  }
  Ok(out)
}

struct Batch {
  ctx_ids: Vec<i64>,
  tgt_ids: Vec<i64>,
  batch_size: usize,
}

fn build_batches(rows: &[HistoryRow], config: &NeuralTrainConfig) -> Result<Vec<Batch>> {
  let mut ctx_flat: Vec<i64> = Vec::new();
  let mut tgt_flat: Vec<i64> = Vec::new();
  let mut out: Vec<Batch> = Vec::new();

  let mut examples = 0usize;
  for idx in config.window..rows.len() {
    let prev = &rows[idx - 1];
    let context_tokens = build_context_tokens(rows, idx, config.window, prev);
    let target_tokens = command_tokens(&rows[idx].shellname, &rows[idx].expanded_command);

    let ctx_ids = tokens_to_ids(&context_tokens, config);
    let tgt_ids = tokens_to_ids(&target_tokens, config);

    ctx_flat.extend(ctx_ids.into_iter().map(|v| v as i64));
    tgt_flat.extend(tgt_ids.into_iter().map(|v| v as i64));
    examples += 1;

    if examples == config.batch_size {
      out.push(Batch {
        ctx_ids: std::mem::take(&mut ctx_flat),
        tgt_ids: std::mem::take(&mut tgt_flat),
        batch_size: examples,
      });
      examples = 0;
    }
  }

  if examples > 0 {
    out.push(Batch {
      ctx_ids: ctx_flat,
      tgt_ids: tgt_flat,
      batch_size: examples,
    });
  }

  Ok(out)
}

fn build_context_tokens(
  rows: &[HistoryRow],
  idx: usize,
  window: usize,
  prev: &HistoryRow,
) -> Vec<String> {
  let start = idx.saturating_sub(window);
  let mut out = Vec::new();
  out.extend(context_field_tokens(prev));
  out.push("__CTX__".to_string());

  for row in &rows[start..idx] {
    out.push("__CMD__".to_string());
    out.extend(command_tokens(&row.shellname, &row.expanded_command));
  }
  out
}

fn context_field_tokens(row: &HistoryRow) -> Vec<String> {
  let mut out = Vec::new();
  if let Some(root) = row.workspace_root.as_deref().filter(|v| !v.is_empty()) {
    out.push(format!("ctx:workspace_root={root}"));
  }
  if let Some(cwd) = row.working_directory.as_deref().filter(|v| !v.is_empty()) {
    out.push(format!("ctx:cwd={cwd}"));
  }
  if let Some(exit) = row.exit_status {
    out.push(format!("ctx:exit={exit}"));
  }
  if let Some(host) = row.hostname.as_deref().filter(|v| !v.is_empty()) {
    out.push(format!("ctx:host={host}"));
  }
  if let Some(user) = row.username.as_deref().filter(|v| !v.is_empty()) {
    out.push(format!("ctx:user={user}"));
  }
  if let Some(ts) = row.start_ts {
    out.push(format!("ctx:timebucket={}", time_bucket(ts)));
  }
  if let Some(session) = row.session_id {
    out.push(format!("ctx:session={session}"));
  }
  out
}

fn command_tokens(shellname: &str, command: &str) -> Vec<String> {
  let tokens = tokenize_index(shellname, command);
  let Some(parts) = extract_command_parts(command, &tokens) else {
    return Vec::new();
  };

  let mut out = Vec::new();
  if !parts.head.is_empty() {
    out.push(format!("head:{}", parts.head));
  }

  let mut flags = parts.flags;
  flags.sort();
  flags.dedup();
  for flag in flags {
    out.push(format!("flag:{flag}"));
  }

  for arg in parts.args.into_iter().take(8) {
    if !arg.normalized.is_empty() {
      out.push(format!("arg:{}", arg.normalized));
    }
  }

  out
}

fn tokens_to_ids(tokens: &[String], config: &NeuralTrainConfig) -> Vec<u32> {
  let mut out = vec![0u32; config.max_seq_len];
  let take = tokens.len().min(config.max_seq_len);
  for (dst, tok) in out.iter_mut().take(take).zip(tokens.iter()) {
    // Reserve 0 for PAD.
    let bucket = (stable_hash(tok) % (config.vocab_size as u64 - 1)) as u32 + 1;
    *dst = bucket;
  }
  out
}

fn time_bucket(ts: i64) -> i64 {
  ts / (60 * 60)
}

pub async fn embed_command_with_model(
  model: &BiEncoder<Wgpu<f32, i32>>,
  shellname: &str,
  command: &str,
  config: &NeuralTrainConfig,
) -> Result<Vec<f32>> {
  type Backend = Wgpu<f32, i32>;
  let device = burn::backend::wgpu::WgpuDevice::default();

  let tokens = command_tokens(shellname, command);
  let ids = tokens_to_ids(&tokens, config);
  let raw = ids.iter().map(|v| *v as i64).collect::<Vec<_>>();
  let ids =
    Tensor::<Backend, 2, Int>::from_ints(raw.as_slice(), &device).reshape([1, config.max_seq_len]);

  let embedding = model.encode(ids);
  let data = embedding.into_data();
  let values = data
    .to_vec::<f32>()
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;
  Ok(values)
}

pub fn embed_commands_batch_with_model(
  model: &BiEncoder<Wgpu<f32, i32>>,
  commands: &[(String, String)],
  config: &NeuralTrainConfig,
) -> Result<Vec<Vec<f32>>> {
  type Backend = Wgpu<f32, i32>;
  let device = burn::backend::wgpu::WgpuDevice::default();

  if commands.is_empty() {
    return Ok(Vec::new());
  }

  let batch = commands.len();
  let mut flat: Vec<i64> = Vec::with_capacity(batch * config.max_seq_len);
  for (command, shellname) in commands {
    let tokens = command_tokens(shellname, command);
    let ids = tokens_to_ids(&tokens, config);
    flat.extend(ids.into_iter().map(|v| v as i64));
  }

  let ids = Tensor::<Backend, 2, Int>::from_ints(flat.as_slice(), &device)
    .reshape([batch, config.max_seq_len]);
  let embedding = model.encode(ids);
  let [_batch, dim] = embedding.dims();
  let values = embedding
    .into_data()
    .to_vec::<f32>()
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;

  let mut out: Vec<Vec<f32>> = Vec::with_capacity(batch);
  for chunk in values.chunks_exact(dim) {
    out.push(chunk.to_vec());
  }
  Ok(out)
}

#[derive(Debug, Clone, Copy)]
pub struct EmbedContextInput<'a> {
  pub workspace_root: Option<&'a str>,
  pub cwd: Option<&'a str>,
  pub hostname: Option<&'a str>,
  pub username: Option<&'a str>,
  pub exit_status: Option<i64>,
  pub session_id: Option<i64>,
  pub recent_commands: &'a [String],
}

pub fn embed_context_with_model(
  model: &BiEncoder<Wgpu<f32, i32>>,
  train_config: &NeuralTrainConfig,
  input: EmbedContextInput<'_>,
) -> Result<Vec<f32>> {
  type Backend = Wgpu<f32, i32>;
  let device = burn::backend::wgpu::WgpuDevice::default();

  let mut tokens = Vec::new();
  if let Some(root) = input.workspace_root.filter(|v| !v.is_empty()) {
    tokens.push(format!("ctx:workspace_root={root}"));
  }
  if let Some(cwd) = input.cwd.filter(|v| !v.is_empty()) {
    tokens.push(format!("ctx:cwd={cwd}"));
  }
  if let Some(exit) = input.exit_status {
    tokens.push(format!("ctx:exit={exit}"));
  }
  if let Some(host) = input.hostname.filter(|v| !v.is_empty()) {
    tokens.push(format!("ctx:host={host}"));
  }
  if let Some(user) = input.username.filter(|v| !v.is_empty()) {
    tokens.push(format!("ctx:user={user}"));
  }
  if let Some(session) = input.session_id {
    tokens.push(format!("ctx:session={session}"));
  }

  tokens.push("__CTX__".to_string());
  for cmd in input
    .recent_commands
    .iter()
    .rev()
    .take(train_config.window)
    .rev()
  {
    tokens.push("__CMD__".to_string());
    tokens.extend(command_tokens(DEFAULT_SHELLNAME, cmd));
  }

  let ids = tokens_to_ids(&tokens, train_config);
  let raw = ids.iter().map(|v| *v as i64).collect::<Vec<_>>();
  let ids = Tensor::<Backend, 2, Int>::from_ints(raw.as_slice(), &device)
    .reshape([1, train_config.max_seq_len]);

  let embedding = model.encode(ids);
  let values = embedding
    .into_data()
    .to_vec::<f32>()
    .map_err(|err| ZageError::GenericError(Box::new(err)))?;
  Ok(values)
}

pub fn assert_build_config() -> Result<()> {
  if cfg!(feature = "pprof") {
    // noop; keep function around for symmetry with other subsystems.
  }
  Ok(())
}
