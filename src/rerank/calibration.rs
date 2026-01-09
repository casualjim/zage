use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize, serde::Serialize)]
pub struct CalibrationParams {
  pub tier1_a: f64,
  pub tier1_b: f64,
  pub model_a: f64,
  pub model_b: f64,
  pub stack_w0: f64,
  pub stack_w1: f64,
  pub stack_w2: f64,
}

pub(crate) fn sigmoid(value: f64) -> f64 {
  let clamped = value.clamp(-40.0, 40.0);
  1.0 / (1.0 + (-clamped).exp())
}

pub(crate) fn fit_platt(scores: &[f64], labels: &[f64]) -> (f64, f64) {
  if scores.is_empty() {
    return (1.0, 0.0);
  }
  let mut a = 1.0;
  let mut b = 0.0;
  let n = scores.len() as f64;
  let lr = 0.1;
  for _ in 0..200 {
    let mut grad_a = 0.0;
    let mut grad_b = 0.0;
    for (s, y) in scores.iter().zip(labels.iter()) {
      let p = sigmoid(a * s + b);
      let diff = p - y;
      grad_a += diff * s;
      grad_b += diff;
    }
    a -= lr * (grad_a / n);
    b -= lr * (grad_b / n);
  }
  (a, b)
}

pub(crate) fn fit_stack(p_tier1: &[f64], p_model: &[f64], labels: &[f64]) -> (f64, f64, f64) {
  if p_tier1.is_empty() {
    return (0.0, 1.0, 1.0);
  }
  let mut w0 = 0.0;
  let mut w1 = 1.0;
  let mut w2 = 1.0;
  let n = p_tier1.len() as f64;
  let lr = 0.2;
  for _ in 0..200 {
    let mut g0 = 0.0;
    let mut g1 = 0.0;
    let mut g2 = 0.0;
    for ((p1, p2), y) in p_tier1.iter().zip(p_model.iter()).zip(labels.iter()) {
      let p = sigmoid(w0 + w1 * p1 + w2 * p2);
      let diff = p - y;
      g0 += diff;
      g1 += diff * p1;
      g2 += diff * p2;
    }
    w0 -= lr * (g0 / n);
    w1 -= lr * (g1 / n);
    w2 -= lr * (g2 / n);
  }
  (w0, w1, w2)
}
