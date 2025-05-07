use crate::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, VarMap, embedding};
use hf_hub::api::sync::Api;
use serde_json::Value;
use tokenizers::Tokenizer;

/// Pretrained embedder using HF model 'Alibaba-NLP/gte-modernbert-base'.
pub struct PretrainedEmbedder {
  embedding: Embedding,
  tokenizer: Tokenizer,
  device: Device,
}

impl PretrainedEmbedder {
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
    // The HF config uses `hidden_size` for the embedding dimension
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

  /// Embed a single text into a vector by averaging token embeddings.
  pub fn embed(&self, text: &str) -> Result<Vec<f32>> {
    let encoding = self
      .tokenizer
      .encode(text, false)
      .map_err(|e| crate::ZageError::GenericError(e.into()))?;
    let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
    let tensor = Tensor::new(ids.as_slice(), &self.device)?;
    let embeds = self.embedding.forward(&tensor)?;
    let pooled = embeds.mean(0)?;
    Ok(pooled.to_vec1::<f32>()?)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Result;
  use candle_core::{Device, MetalDevice, backend::BackendDevice};
  use std::time::{Duration, Instant};

  #[test]
  fn test_embed_length() -> Result<()> {
    // let embedder = PretrainedEmbedder::new(Device::Cpu)?;
    let embedder = PretrainedEmbedder::new(Device::Metal(MetalDevice::new(0).unwrap()))?;
    let vec = embedder.embed("hello world")?;
    assert!(!vec.is_empty(), "Embedding output should not be empty");
    Ok(())
  }

  #[test]
  fn test_embed_difference() -> Result<()> {
    // let embedder = PretrainedEmbedder::new(Device::Cpu)?;
    let embedder = PretrainedEmbedder::new(Device::Metal(MetalDevice::new(0).unwrap()))?;
    let v1 = embedder.embed("foo bar")?;
    let v2 = embedder.embed("baz qux")?;
    assert_ne!(v1, v2, "Embeddings for different inputs should differ");
    Ok(())
  }

  #[test]
  fn test_embed_latency() -> Result<()> {
    // load embedder once
    let embedder = PretrainedEmbedder::new(Device::Cpu)?;
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
