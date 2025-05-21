use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, VarMap};
use hf_hub::api::sync::Api;
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::Result;
use crate::embedding::Embedder;

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

    let embedding = candle_nn::embedding(vocab_size, emb_dim, vb.pp("embedding"))?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| {
      std::io::Error::new(std::io::ErrorKind::Other, format!("Tokenizer error: {}", e))
    })?;

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
