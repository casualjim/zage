use crate::model::PredictionModel;
use crate::model::contextual_features::ContextualFeatures;
use crate::embedding::Embedder;
use crate::model::feature_integration::FeatureIntegrator;
use crate::{Result, shell_history::Invocation};
use candle_core::{Device, IndexOp};
use candle_nn::{LSTM, LSTMConfig, Linear, Module, RNN, VarBuilder, ops};
use std::sync::Arc;

/// Configuration for the NeuralModel
#[derive(Debug, Clone, Copy)]
pub struct ModelConfig {
  pub embedding_dim: usize, // Dimension of the combined input (command embedding + context features)
  pub lstm_hidden_size: usize,
  pub output_size: usize, // e.g., size of vocabulary or number of predictable commands
  pub lstm_num_layers: usize,
  pub dropout: f32,
}

/// Neural network based prediction model using Candle
pub struct NeuralModel {
  embedder: Arc<dyn Embedder>,
  context_features: Arc<ContextualFeatures>,
  #[allow(dead_code)]
  device: Device,
  config: ModelConfig,
  lstm: LSTM,
  output_layer: Linear,
  integrator: FeatureIntegrator,
}

impl NeuralModel {
  /// Create a new NeuralModel instance with default configuration
  pub fn new(
    embedder: Arc<dyn Embedder>,
    context_features_arc: Arc<ContextualFeatures>,
    device: Device,
  ) -> Result<Self> {
    // Initialize FeatureIntegrator
    let integrator = FeatureIntegrator::new(
      embedder.clone(),
      context_features_arc.clone(),
      device.clone(),
    )?;

    // Determine embedding_dim using the integrator
    // Use a default invocation to get the dimension
    let default_invocation = Invocation::default();
    let embedding_dim = integrator.get_feature_dimension(&default_invocation)?;

    let config = ModelConfig {
      embedding_dim,
      lstm_hidden_size: 128, // Example value
      output_size: 100,      // Example value (e.g., predict top 100 commands)
      lstm_num_layers: 1,
      dropout: 0.0,
    };

    let vb = VarBuilder::zeros(candle_core::DType::F32, &device); // Placeholder VarBuilder

    let lstm_config = LSTMConfig::default();
    let lstm = candle_nn::lstm(
      config.embedding_dim,
      config.lstm_hidden_size,
      lstm_config,
      vb.pp("lstm"),
    )?;
    let output_layer =
      candle_nn::linear(config.lstm_hidden_size, config.output_size, vb.pp("output"))?;

    Ok(NeuralModel {
      embedder: embedder,
      context_features: context_features_arc,
      device,
      config,
      lstm,
      output_layer,
      integrator,
    })
  }

  /// Get embeddings for a command
  pub fn embed_command(&self, command: &str) -> Result<Vec<f32>> {
    self.embedder.embed(command)
  }

  /// Extract and embed contextual features
  pub fn extract_context_features(&self, invocation: &Invocation) -> Result<Vec<f32>> {
    self.context_features.encode_all_features(invocation)
  }
}

impl PredictionModel for NeuralModel {
  fn train<I>(&mut self, _invocations: I) -> Result<()>
  where
    I: IntoIterator<Item = Invocation>,
  {
    // TODO: implement training logic
    Ok(())
  }

  fn predict(&self, recent_history: &[Invocation], _max_predictions: usize) -> Result<Vec<String>> {
    if recent_history.is_empty() || self.config.embedding_dim == 0 {
      return Ok(Vec::new()); // Cannot predict if no history or zero embedding dim
    }

    // 1. Create integrated features tensor from recent_history
    // The create_integrated_tensor method returns a tensor of shape [batch_size, feature_dim]
    // For LSTM input, we often need [sequence_length, batch_size, feature_dim] or [batch_size, sequence_length, feature_dim]
    // Here, recent_history is a sequence, so it becomes [sequence_length, embedding_dim].
    // We need to add a batch dimension: [1, sequence_length, embedding_dim]
    let features_tensor_flat = self.integrator.create_integrated_tensor(recent_history)?;
    // Reshape for LSTM: [batch_size=1, sequence_length, embedding_dim]
    let sequence_length = recent_history.len();
    let features_tensor =
      features_tensor_flat.reshape((1, sequence_length, self.config.embedding_dim))?;

    // 2. Forward pass through LSTM
    // sequence_of_hidden_states has shape [batch_size, sequence_length, hidden_size]
    let batch_size = features_tensor.dim(0)?;
    let initial_state = self.lstm.zero_state(batch_size)?;
    let lstm_states = self.lstm.seq_init(&features_tensor, &initial_state)?;
    let lstm_outputs = self.lstm.states_to_tensor(&lstm_states)?;

    // Get the output from the last time step: shape [batch_size, hidden_size]
    let last_output = lstm_outputs.i((.., sequence_length - 1, ..))?;

    // 3. Forward pass through output layer
    let logits = self.output_layer.forward(&last_output)?;
    let _probabilities = ops::softmax(&logits, candle_core::D::Minus1)?; // Probabilities for each output unit

    // 4. Decode probabilities to command strings
    // This part is complex and depends on how `output_size` maps to actual commands.
    // For now, returning an empty vec as a placeholder.
    // In a real scenario, you'd map the top probability indices to command strings.
    Ok(Vec::new())
  }

  fn update<I>(&mut self, _invocations: I) -> Result<()>
  where
    I: IntoIterator<Item = Invocation>,
  {
    // TODO: implement incremental update logic
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::embedding::create_embedder;
  use candle_core::Device;

  #[test]
  fn test_neural_model_trait_impl() -> Result<()> {
    let device = Device::Cpu;
    let embedder = create_embedder(device.clone())?;
    let context_features_arc = Arc::new(ContextualFeatures::new(device.clone(), embedder.clone())?);
    let mut model = NeuralModel::new(embedder, context_features_arc, device)?;
    // training on empty data should succeed
    model.train(Vec::<Invocation>::new())?;
    // prediction on empty history should return empty list
    let preds = model.predict(&[], 5)?;
    assert!(preds.is_empty());
    // update on empty data should succeed
    model.update(Vec::<Invocation>::new())?;
    Ok(())
  }

  #[test]
  fn test_predict_shapes() -> Result<()> {
    // Use a history of length 3
    let history = vec![Invocation::default(); 3];
    let device = Device::Cpu;
    let embedder = create_embedder(device.clone())?;
    let context_features_arc = Arc::new(ContextualFeatures::new(device.clone(), embedder.clone())?);
    let model = NeuralModel::new(embedder, context_features_arc, device)?;

    // 1. Integrated tensor shape: [seq_len, embedding_dim]
    let flat = model.integrator.create_integrated_tensor(&history)?;
    let (b, f) = flat.dims2()?;
    assert_eq!((b, f), (3, model.config.embedding_dim));

    // 2. Reshape for LSTM: [batch=1, seq_len=3, embedding_dim]
    let features = flat.reshape((1, 3, model.config.embedding_dim))?;
    let (b2, s, f2) = features.dims3()?;
    assert_eq!((b2, s, f2), (1, 3, model.config.embedding_dim));

    // 3. LSTM states and outputs
    let init = model.lstm.zero_state(1)?;
    let states = model.lstm.seq_init(&features, &init)?;
    assert_eq!(states.len(), 3);
    let outputs = model.lstm.states_to_tensor(&states)?;
    let (b3, s2, h) = outputs.dims3()?;
    assert_eq!((b3, s2, h), (1, 3, model.config.lstm_hidden_size));

    // 4. Last output and logits shape
    let last = outputs.i((.., 2, ..))?;
    let (b4, h2) = last.dims2()?;
    assert_eq!((b4, h2), (1, model.config.lstm_hidden_size));
    let logits = model.output_layer.forward(&last)?;
    let (b5, o) = logits.dims2()?;
    assert_eq!((b5, o), (1, model.config.output_size));

    Ok(())
  }

  #[test]
  fn bench_predict_latency() -> Result<()> {
    use std::time::Instant;
    let device = Device::Cpu;
    let embedder = create_embedder(device.clone())?;
    let context_features_arc = Arc::new(ContextualFeatures::new(device.clone(), embedder.clone())?);
    let model = NeuralModel::new(embedder, context_features_arc, device)?;
    // create a sample history of length 5
    let history = vec![Invocation::default(); 5];
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
      model.predict(&history, 10)?;
    }
    let elapsed = start.elapsed();
    println!("Ran {} predictions in {:?}", iterations, elapsed);
    Ok(())
  }

  #[test]
  fn test_softmax_sum_to_one() -> Result<()> {
    // Softmax of zero logits should yield uniform distribution summing to 1
    use candle_core::{Device, Tensor};
    use candle_nn::ops;
    let device = Device::Cpu;
    // dummy logits [0, 0]
    let logits = Tensor::from_vec(vec![0f32, 0f32], (1, 2), &device)?;
    let probs = ops::softmax(&logits, candle_core::D::Minus1)?;
    let sums = probs.sum_keepdim(candle_core::D::Minus1)?;
    let sums2d = sums.to_vec2::<f32>()?;
    assert!((sums2d[0][0] - 1.0).abs() < 1e-6);
    Ok(())
  }
}
