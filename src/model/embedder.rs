use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, VarMap, embedding};
use hf_hub::api::sync::Api;
use serde_json::Value;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;
use tokenizers::Tokenizer;

use crate::protocol::ProtocolMessage;
use crate::{Result, ZageError};

/// Trait defining the embedding interface
pub trait Embedder: Send + Sync {
  /// Embed a single text into a vector
  fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

/// Pretrained embedder using HF model 'nomic-ai/CodeRankEmbed'.
pub struct InProcessEmbedder {
  embedding: Embedding,
  tokenizer: Tokenizer,
  device: Device,
}

impl InProcessEmbedder {
  /// Download and initialize the pretrained embedder.
  pub fn new(device: Device) -> Result<Self> {
    let api = Api::new()?;
    let repo = api.model("nomic-ai/CodeRankEmbed".to_string());
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let model_path = repo.get("model.safetensors")?;
    let mut varmap = VarMap::new();
    varmap.load(model_path)?;
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let config_file = std::fs::File::open(config_path)?;
    let config: Value = serde_json::from_reader(config_file)?;
    let vocab_size = config["vocab_size"].as_u64().unwrap_or_default() as usize;
    // The HF config uses `n_embd` for the embedding dimension
    let emb_dim = config["n_embd"].as_u64().unwrap() as usize;
    let embedding = embedding(vocab_size, emb_dim, vb.pp("embedding"))?;
    let tokenizer =
      Tokenizer::from_file(tokenizer_path).map_err(|e| crate::ZageError::GenericError(e.into()))?;
    Ok(Self {
      embedding,
      tokenizer,
      device,
    })
  }
}

impl Embedder for InProcessEmbedder {
  /// Embed a single text into a vector by averaging token embeddings.
  fn embed(&self, text: &str) -> Result<Vec<f32>> {
    let encoding = self
      .tokenizer
      .encode(text, false)
      .map_err(|e| crate::ZageError::GenericError(e.into()))?;
    let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    // Fallback for empty tokenization to avoid NaN embeddings
    if ids.is_empty() {
      ids.push(0);
    }
    let tensor = Tensor::new(ids.as_slice(), &self.device)?;
    let embeds = self.embedding.forward(&tensor)?;
    let pooled = embeds.mean(0)?;
    Ok(pooled.to_vec1::<f32>()?)
  }
}

/// Client for the embedding socket server
pub struct RemoteEmbedder {
  socket_path: PathBuf,
  timeout_secs: u64,
}

impl Default for RemoteEmbedder {
  fn default() -> Self {
    Self {
      socket_path: "/tmp/zage_embedder.sock".into(),
      timeout_secs: 30,
    }
  }
}

impl RemoteEmbedder {
  /// Create a new client with custom settings
  pub fn new<P: Into<PathBuf>>(socket_path: P, timeout_secs: u64) -> Self {
    Self {
      socket_path: socket_path.into(),
      timeout_secs,
    }
  }

  /// Embed a text string
  pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
    // Connect to the socket
    let mut stream = UnixStream::connect(&self.socket_path)
      .map_err(|e| ZageError::ConfigError(format!("Failed to connect to socket: {}", e)))?;

    // Set timeouts
    stream.set_read_timeout(Some(Duration::from_secs(self.timeout_secs)))?;
    stream.set_write_timeout(Some(Duration::from_secs(self.timeout_secs)))?;

    // Create and send the embedding request message
    let request = ProtocolMessage::EmbedRequest(text.to_string());
    request.write_to(&mut stream)?;

    // Read the response message
    let response = ProtocolMessage::read_from(&mut stream)?;

    // Process the response based on its type
    match response {
      ProtocolMessage::EmbedResponse(embedding) => Ok(embedding),
      ProtocolMessage::ErrorResponse(error_msg) => Err(ZageError::ConfigError(format!(
        "Server error: {}",
        error_msg
      ))),
      _ => Err(ZageError::ConfigError(format!(
        "Unexpected response type: {:?}",
        response
      ))),
    }
  }
}

impl Embedder for RemoteEmbedder {
  fn embed(&self, text: &str) -> Result<Vec<f32>> {
    self.embed(text)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{Result, model::create_embedder};
  use std::time::{Duration, Instant};

  #[test]
  fn test_embed_length() -> Result<()> {
    let embedder = create_embedder(Device::Cpu)?;
    let vec = embedder.embed("hello world")?;
    assert!(!vec.is_empty(), "Embedding output should not be empty");
    Ok(())
  }

  #[test]
  fn test_embed_difference() -> Result<()> {
    let embedder = create_embedder(Device::Cpu)?;
    let v1 = embedder.embed("foo bar")?;
    let v2 = embedder.embed("baz qux")?;
    assert_ne!(v1, v2, "Embeddings for different inputs should differ");
    Ok(())
  }

  #[test]
  fn test_embed_latency() -> Result<()> {
    // load embedder once
    let embedder = create_embedder(Device::Cpu)?;
    // prepare 50 sample commands by cycling a base list
    let base_cmds = vec![
      "ls",
      "pwd",
      "git status",
      "echo hi",
      "cd ..",
      "cargo build",
      "cargo test",
      "grep foo .",
      "sed 's/a/b/' file",
      "cat Cargo.toml",
    ];
    let commands: Vec<&str> = base_cmds.iter().cycle().take(50).cloned().collect();
    // warm up
    for cmd in &commands {
      embedder.embed(cmd)?;
    }
    let mut durations = Vec::new();
    // measure 10 rounds
    for _ in 0..10 {
      for cmd in &commands {
        let start = Instant::now();
        embedder.embed(cmd)?;
        durations.push(start.elapsed());
      }
    }
    durations.sort();
    let len = durations.len();
    let total = durations
      .iter()
      .copied()
      .fold(Duration::ZERO, |acc, d| acc + d);
    let avg = total / (len as u32);
    let min = durations.first().copied().unwrap();
    let max = durations.last().copied().unwrap();
    let p90 = durations[(len * 90 / 100).saturating_sub(1)];
    let p99 = durations[(len * 99 / 100).saturating_sub(1)];
    println!("latency over {} runs:", len);
    println!(
      "min: {:?}, p90: {:?}, p99: {:?}, max: {:?}, avg: {:?}",
      min, p90, p99, max, avg
    );
    Ok(())
  }
}
