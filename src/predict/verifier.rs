use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

use jiff::Timestamp;
use libsql::{Connection, Value};
use serde::Deserialize;
use tempfile::TempDir;

use crate::Result;
use crate::ZageError;
use crate::db::{init, insert_invocation, open_db};
use crate::hash_util::stable_hash;
use crate::indexer::rebuild_stats;
use crate::sequence::{SequenceConfig, analyze_sequences};
use crate::shell_history::Invocation;

use super::RankingWeights;
use super::ScoreBreakdown;
use super::SuggestConfig;
use super::SuggestRuntime;
use super::Suggestion;
use super::SystemTimeProvider;
use super::TimeProvider;
use super::aliases::expand_alias;
use super::ranking::DEFAULT_RECENCY_HALF_LIFE_SECONDS;
use super::suggest_with_runtime;

const DEFAULT_HOSTNAME: &str = "testhost";
const DEFAULT_USERNAME: &str = "testuser";
const DEFAULT_SESSION: &str = "testsession";

#[derive(Debug, Clone)]
pub struct TestConfig {
  pub now: Option<i64>,
  pub weights: Option<RankingWeights>,
  pub recency_half_life: Option<f64>,
  pub debug: bool,
}

#[derive(Debug, Clone)]
pub struct TestSuggestion {
  pub command: String,
  pub score: f64,
  pub breakdown: ScoreBreakdown,
  pub rank: usize,
}

#[derive(Debug, Clone)]
pub struct Tier1Case {
  pub name: String,
  pub index: usize,
}

#[derive(Debug, Deserialize, Default)]
struct Meta {
  #[allow(dead_code)]
  description: Option<String>,
  #[allow(dead_code)]
  tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum AtValue {
  String(String),
  Integer(i64),
}

#[derive(Debug, Deserialize, Default)]
struct Physics {
  now: Option<AtValue>,
  w_recency: Option<f64>,
  w_frequency: Option<f64>,
  w_transition: Option<f64>,
  w_context: Option<f64>,
  w_sequence: Option<f64>,
  w_similarity: Option<f64>,
  recency_half_life_seconds: Option<f64>,
}

#[derive(Debug, Deserialize, Default)]
struct Options {
  use_sequences: Option<bool>,
  run_sequence_analysis: Option<bool>,
  min_sequence_support: Option<usize>,
  min_sequence_confidence: Option<f64>,
  min_sequence_lift: Option<f64>,
  run_phase_indexing: Option<bool>,
}

#[derive(Debug, Deserialize, Default)]
struct HistoryEntry {
  cmd: String,
  expanded: Option<String>,
  shell: Option<String>,
  at: Option<AtValue>,
  cwd: Option<String>,
  hostname: Option<String>,
  username: Option<String>,
  exit: Option<i64>,
  session: Option<String>,
  count: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
struct ScenarioContext {
  cwd: Option<String>,
  hostname: Option<String>,
  username: Option<String>,
  session: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct CandidateExpect {
  cmd: String,
  min_score: Option<f64>,
  max_score: Option<f64>,
  min_recency: Option<f64>,
  max_recency: Option<f64>,
  min_frequency: Option<f64>,
  max_frequency: Option<f64>,
  min_transition: Option<f64>,
  max_transition: Option<f64>,
  min_context: Option<f64>,
  max_context: Option<f64>,
  min_sequence: Option<f64>,
  max_sequence: Option<f64>,
  min_similarity: Option<f64>,
  max_similarity: Option<f64>,
  min_rank: Option<usize>,
  max_rank: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ExpectedValue {
  Integer(i64),
  Float(f64),
  String(String),
  Bool(bool),
}

#[derive(Debug, Deserialize)]
struct DbExpect {
  description: Option<String>,
  sql: String,
  params: Option<Vec<ExpectedValue>>,
  operator: String,
  value: ExpectedValue,
}

#[derive(Debug, Deserialize, Default)]
struct ScenarioExpect {
  top: Option<Vec<String>>,
  contains: Option<Vec<String>>,
  absent: Option<Vec<String>>,
  empty: Option<bool>,
  min_results: Option<usize>,
  max_results: Option<usize>,
  candidate: Option<Vec<CandidateExpect>>,
  db: Option<Vec<DbExpect>>,
}

#[derive(Debug, Deserialize, Default)]
struct Scenario {
  name: String,
  mode: String,
  input: Option<String>,
  cursor: Option<usize>,
  prev_command: Option<String>,
  prev_exit: Option<i64>,
  context: Option<ScenarioContext>,
  expect: ScenarioExpect,
}

#[derive(Debug, Deserialize, Default)]
struct TestSpec {
  #[allow(dead_code)]
  meta: Option<Meta>,
  physics: Option<Physics>,
  fs: Option<HashMap<String, String>>,
  aliases: Option<HashMap<String, String>>,
  #[serde(default)]
  history: Vec<HistoryEntry>,
  #[serde(default)]
  scenario: Vec<Scenario>,
  options: Option<Options>,
  phases: Option<HashMap<String, Vec<String>>>,
}

pub async fn suggest_for_test(
  conn: &Connection,
  config: SuggestConfig,
  test_config: TestConfig,
) -> Result<Vec<TestSuggestion>> {
  let runtime = build_runtime(test_config, HashMap::new())?;
  let suggestions = suggest_with_runtime(conn, config, &runtime, None).await?;
  Ok(rank_suggestions(suggestions))
}

async fn suggest_for_test_with_aliases(
  conn: &Connection,
  config: SuggestConfig,
  test_config: TestConfig,
  aliases: HashMap<String, String>,
  override_prev: Option<(String, Option<i64>)>,
) -> Result<Vec<TestSuggestion>> {
  let runtime = build_runtime(test_config, aliases)?;
  let suggestions = suggest_with_runtime(conn, config, &runtime, override_prev).await?;
  Ok(rank_suggestions(suggestions))
}

fn build_runtime(
  test_config: TestConfig,
  aliases: HashMap<String, String>,
) -> Result<SuggestRuntime> {
  let now = if let Some(now) = test_config.now {
    now
  } else {
    let provider = SystemTimeProvider;
    provider.now()
  };
  let weights = test_config.weights.unwrap_or_default();
  let recency_half_life = test_config
    .recency_half_life
    .unwrap_or(DEFAULT_RECENCY_HALF_LIFE_SECONDS);
  Ok(SuggestRuntime {
    aliases,
    weights,
    recency_half_life,
    now,
  })
}

fn rank_suggestions(suggestions: Vec<Suggestion>) -> Vec<TestSuggestion> {
  suggestions
    .into_iter()
    .enumerate()
    .map(|(idx, suggestion)| TestSuggestion {
      command: suggestion.command,
      score: suggestion.score,
      breakdown: suggestion.breakdown,
      rank: idx + 1,
    })
    .collect()
}

fn parse_timestamp(value: &AtValue, base: Option<i64>) -> Result<i64> {
  match value {
    AtValue::Integer(ts) => Ok(*ts),
    AtValue::String(raw) => parse_timestamp_string(raw, base),
  }
}

fn parse_timestamp_string(raw: &str, base: Option<i64>) -> Result<i64> {
  let trimmed = raw.trim();
  if trimmed.is_empty() {
    return Err(ZageError::ConfigError("empty timestamp".to_string()));
  }
  let is_relative = trimmed.starts_with('-') || trimmed.starts_with('+');
  if is_relative {
    let base = base.ok_or_else(|| {
      ZageError::ConfigError("relative timestamp requires [physics].now".to_string())
    })?;
    let delta = parse_relative_offset(trimmed)?;
    return Ok(base + delta);
  }
  if trimmed.chars().all(|c| c.is_ascii_digit()) {
    let ts: i64 = trimmed.parse()?;
    return Ok(ts);
  }
  let parsed = trimmed
    .parse::<Timestamp>()
    .map_err(|err| ZageError::ConfigError(err.to_string()))?;
  Ok(parsed.as_second())
}

fn parse_relative_offset(raw: &str) -> Result<i64> {
  let (sign, rest) = if let Some(value) = raw.strip_prefix('-') {
    (-1i64, value)
  } else if let Some(value) = raw.strip_prefix('+') {
    (1i64, value)
  } else {
    (1i64, raw)
  };
  if rest.is_empty() {
    return Err(ZageError::ConfigError(
      "relative offset missing value".to_string(),
    ));
  }
  let last = rest.chars().last().unwrap_or('s');
  if last.is_ascii_digit() {
    let value: i64 = rest.parse()?;
    return Ok(sign * value);
  }
  let number = &rest[..rest.len() - last.len_utf8()];
  let magnitude: i64 = number.parse()?;
  let seconds = match last {
    's' => 1,
    'm' => 60,
    'h' => 60 * 60,
    'd' => 60 * 60 * 24,
    'w' => 60 * 60 * 24 * 7,
    'M' => 60 * 60 * 24 * 30,
    'y' => 60 * 60 * 24 * 365,
    _ => {
      return Err(ZageError::ConfigError(format!(
        "unsupported relative unit: {last}"
      )));
    }
  };
  Ok(sign * magnitude * seconds)
}

fn resolve_session_id(value: Option<&String>) -> i64 {
  let raw = value.map(String::as_str).unwrap_or(DEFAULT_SESSION);
  stable_hash(raw) as i64
}

fn materialize_filesystem(fs_map: Option<&HashMap<String, String>>) -> Result<TempDir> {
  let temp = TempDir::new()?;
  if let Some(entries) = fs_map {
    for (path, kind) in entries {
      let trimmed = path.trim_end_matches('/');
      let absolute = temp.path().join(trimmed);
      match kind.as_str() {
        "dir" => {
          fs::create_dir_all(&absolute)?;
        }
        "file" => {
          if let Some(parent) = absolute.parent() {
            fs::create_dir_all(parent)?;
          }
          fs::write(&absolute, b"")?;
        }
        other => {
          return Err(ZageError::ConfigError(format!(
            "unsupported fs entry type: {other}"
          )));
        }
      }
    }
  }
  Ok(temp)
}

fn resolve_path(root: &Path, value: Option<&String>) -> Option<String> {
  let value = value?;
  let path = Path::new(value);
  if path.is_absolute() {
    return Some(path.to_string_lossy().to_string());
  }
  Some(root.join(path).to_string_lossy().to_string())
}

async fn seed_history(
  conn: &Connection,
  entries: &[HistoryEntry],
  root: &TempDir,
  physics_now: Option<i64>,
  aliases: &HashMap<String, String>,
) -> Result<()> {
  let base_now = physics_now.unwrap_or_else(|| {
    let provider = SystemTimeProvider;
    provider.now()
  });
  for entry in entries {
    let count = entry.count.unwrap_or(1).max(1);
    let base_ts = if let Some(at) = entry.at.as_ref() {
      parse_timestamp(at, Some(base_now))?
    } else {
      base_now
    };
    let working_directory = resolve_path(root.path(), entry.cwd.as_ref());
    let hostname = entry
      .hostname
      .clone()
      .or_else(|| Some(DEFAULT_HOSTNAME.to_string()));
    let username = entry
      .username
      .clone()
      .or_else(|| Some(DEFAULT_USERNAME.to_string()));
    let session_id = resolve_session_id(entry.session.as_ref());
    let shellname = entry.shell.clone().unwrap_or_else(|| "zsh".to_string());
    let expanded = entry
      .expanded
      .clone()
      .or_else(|| expand_alias(&entry.cmd, aliases))
      .unwrap_or_else(|| entry.cmd.clone());
    for idx in 0..count {
      let ts = base_ts + idx as i64;
      let invocation = Invocation {
        command: entry.cmd.clone(),
        expanded_command: expanded.clone(),
        shellname: shellname.clone(),
        working_directory: working_directory.clone(),
        hostname: hostname.clone(),
        username: username.clone(),
        exit_status: entry.exit.or(Some(0)),
        start_unix_timestamp: Some(ts),
        end_unix_timestamp: Some(ts + 1),
        session_id,
      };
      insert_invocation(conn, &invocation).await?;
    }
  }
  Ok(())
}

fn build_weights(physics: Option<&Physics>) -> Option<RankingWeights> {
  let physics = physics?;
  let mut weights = RankingWeights::default();
  let mut changed = false;
  if let Some(value) = physics.w_recency {
    weights.recency = value;
    changed = true;
  }
  if let Some(value) = physics.w_frequency {
    weights.frequency = value;
    changed = true;
  }
  if let Some(value) = physics.w_transition {
    weights.transition = value;
    changed = true;
  }
  if let Some(value) = physics.w_context {
    weights.context = value;
    changed = true;
  }
  if let Some(value) = physics.w_sequence {
    weights.sequence = value;
    changed = true;
  }
  if let Some(value) = physics.w_similarity {
    weights.similarity = value;
    changed = true;
  }
  if changed { Some(weights) } else { None }
}

fn build_test_config(physics: Option<&Physics>) -> Result<TestConfig> {
  let now = if let Some(physics) = physics {
    if let Some(value) = physics.now.as_ref() {
      Some(parse_timestamp(value, None)?)
    } else {
      None
    }
  } else {
    None
  };
  Ok(TestConfig {
    now,
    weights: build_weights(physics),
    recency_half_life: physics.and_then(|p| p.recency_half_life_seconds),
    debug: false,
  })
}

fn scenario_prefix(scenario: &Scenario) -> Option<String> {
  if scenario.mode != "completion" {
    return None;
  }
  let input = scenario.input.clone().unwrap_or_default();
  let cursor = scenario.cursor.unwrap_or(input.len());
  let end = cursor.min(input.len());
  Some(input[..end].to_string())
}

fn build_suggest_config(
  scenario: &Scenario,
  options: Option<&Options>,
  root: &TempDir,
) -> SuggestConfig {
  let mut config = SuggestConfig {
    recent_limit: 50,
    ..SuggestConfig::default()
  };
  let expect = &scenario.expect;
  let top_items = expect.top.as_deref().unwrap_or(&[]);
  let contains_items = expect.contains.as_deref().unwrap_or(&[]);
  let mut required = top_items.len();
  if !contains_items.is_empty() {
    let mut extras = 0usize;
    for item in contains_items {
      if !top_items.contains(item) {
        extras += 1;
      }
    }
    required = required.max(top_items.len() + extras);
  }
  if top_items.is_empty() && !contains_items.is_empty() {
    required = required.max(contains_items.len() + 1);
  }
  if let Some(candidates) = expect.candidate.as_ref() {
    required = required.max(candidates.len());
  }
  required = required.max(expect.min_results.unwrap_or(0));
  let max_results = expect.max_results.unwrap_or(required.max(1));
  config.max_results = max_results;
  config.use_sequences = options.and_then(|opt| opt.use_sequences).unwrap_or(true);
  config.prefix = scenario_prefix(scenario);

  let ctx = scenario.context.as_ref();
  config.cwd = resolve_path(root.path(), ctx.and_then(|ctx| ctx.cwd.as_ref()))
    .or_else(|| Some(root.path().to_string_lossy().to_string()));
  config.hostname = ctx
    .and_then(|ctx| ctx.hostname.clone())
    .or_else(|| Some(DEFAULT_HOSTNAME.to_string()));
  config.username = ctx
    .and_then(|ctx| ctx.username.clone())
    .or_else(|| Some(DEFAULT_USERNAME.to_string()));
  config.session_id = ctx
    .and_then(|ctx| ctx.session.as_ref())
    .map(|value| stable_hash(value) as i64);

  config
}

fn build_sequence_config(options: Option<&Options>) -> SequenceConfig {
  let mut config = SequenceConfig::default();
  if let Some(options) = options {
    if let Some(value) = options.min_sequence_support {
      config.min_support = value;
    }
    if let Some(value) = options.min_sequence_confidence {
      config.min_confidence = value;
    }
    if let Some(value) = options.min_sequence_lift {
      config.min_lift = value;
    }
  }
  config
}

fn assert_bounds(
  scenario: &Scenario,
  label: &str,
  command: &str,
  actual: f64,
  min: Option<f64>,
  max: Option<f64>,
) {
  if let Some(min) = min
    && actual < min
  {
    panic!(
      "scenario {}: {} for {} below min {} (got {})",
      scenario.name, label, command, min, actual
    );
  }
  if let Some(max) = max
    && actual > max
  {
    panic!(
      "scenario {}: {} for {} above max {} (got {})",
      scenario.name, label, command, max, actual
    );
  }
}

fn assert_candidate_expectations(scenario: &Scenario, results: &[TestSuggestion]) {
  let expectations = match scenario.expect.candidate.as_ref() {
    Some(list) => list,
    None => return,
  };

  for expected in expectations {
    let candidate = results
      .iter()
      .find(|suggestion| suggestion.command == expected.cmd)
      .unwrap_or_else(|| {
        panic!(
          "scenario {}: candidate {} not found",
          scenario.name, expected.cmd
        )
      });

    assert_bounds(
      scenario,
      "score",
      &expected.cmd,
      candidate.score,
      expected.min_score,
      expected.max_score,
    );
    assert_bounds(
      scenario,
      "recency",
      &expected.cmd,
      candidate.breakdown.recency,
      expected.min_recency,
      expected.max_recency,
    );
    assert_bounds(
      scenario,
      "frequency",
      &expected.cmd,
      candidate.breakdown.frequency,
      expected.min_frequency,
      expected.max_frequency,
    );
    assert_bounds(
      scenario,
      "transition",
      &expected.cmd,
      candidate.breakdown.transition,
      expected.min_transition,
      expected.max_transition,
    );
    assert_bounds(
      scenario,
      "context",
      &expected.cmd,
      candidate.breakdown.context,
      expected.min_context,
      expected.max_context,
    );
    assert_bounds(
      scenario,
      "sequence",
      &expected.cmd,
      candidate.breakdown.sequence,
      expected.min_sequence,
      expected.max_sequence,
    );
    assert_bounds(
      scenario,
      "similarity",
      &expected.cmd,
      candidate.breakdown.similarity,
      expected.min_similarity,
      expected.max_similarity,
    );
    if let Some(min_rank) = expected.min_rank
      && candidate.rank < min_rank
    {
      panic!(
        "scenario {}: {} ranked below min rank {} (got {})",
        scenario.name, expected.cmd, min_rank, candidate.rank
      );
    }
    if let Some(max_rank) = expected.max_rank
      && candidate.rank > max_rank
    {
      panic!(
        "scenario {}: {} ranked above max rank {} (got {})",
        scenario.name, expected.cmd, max_rank, candidate.rank
      );
    }
  }
}

fn assert_expectations(scenario: &Scenario, results: &[TestSuggestion]) {
  let expect = &scenario.expect;
  if expect.empty.unwrap_or(false) {
    if !results.is_empty() {
      panic!(
        "scenario {}: expected empty results, got {}",
        scenario.name,
        results.len()
      );
    }
    return;
  }

  if let Some(min) = expect.min_results
    && results.len() < min
  {
    panic!(
      "scenario {}: expected at least {} results, got {}",
      scenario.name,
      min,
      results.len()
    );
  }

  if let Some(max) = expect.max_results
    && results.len() > max
  {
    panic!(
      "scenario {}: expected at most {} results, got {}",
      scenario.name,
      max,
      results.len()
    );
  }

  if let Some(top) = expect.top.as_ref() {
    let got = results
      .iter()
      .take(top.len())
      .map(|suggestion| suggestion.command.clone())
      .collect::<Vec<_>>();
    if &got != top {
      panic!(
        "scenario {}: expected top {:?}, got {:?}",
        scenario.name, top, got
      );
    }
  }

  if let Some(contains) = expect.contains.as_ref() {
    for expected in contains {
      let found = results.iter().any(|s| s.command == *expected);
      if !found {
        panic!(
          "scenario {}: expected suggestion {} in results",
          scenario.name, expected
        );
      }
    }
  }

  if let Some(absent) = expect.absent.as_ref() {
    for expected in absent {
      let found = results
        .iter()
        .any(|s| s.command == *expected || s.command.starts_with(expected));
      if found {
        panic!(
          "scenario {}: unexpected suggestion {} in results",
          scenario.name, expected
        );
      }
    }
  }

  assert_candidate_expectations(scenario, results);
}

fn expect_value_to_libsql(value: &ExpectedValue) -> Value {
  match value {
    ExpectedValue::Integer(val) => Value::from(*val),
    ExpectedValue::Float(val) => Value::from(*val),
    ExpectedValue::String(val) => Value::from(val.clone()),
    ExpectedValue::Bool(val) => Value::from(if *val { 1 } else { 0 }),
  }
}

fn compare_numeric(actual: f64, operator: &str, expected: f64) -> bool {
  match operator {
    "eq" => (actual - expected).abs() < f64::EPSILON,
    "gt" => actual > expected,
    "gte" => actual >= expected,
    "lt" => actual < expected,
    "lte" => actual <= expected,
    "ne" => (actual - expected).abs() >= f64::EPSILON,
    _ => false,
  }
}

async fn assert_db_expectations(conn: &Connection, scenario: &Scenario) -> Result<()> {
  let expects = match scenario.expect.db.as_ref() {
    Some(list) => list,
    None => return Ok(()),
  };
  for expect in expects {
    let params = expect
      .params
      .as_ref()
      .map(|values| {
        values
          .iter()
          .map(expect_value_to_libsql)
          .collect::<Vec<_>>()
      })
      .unwrap_or_default();
    let mut rows = conn
      .query(&expect.sql, libsql::params_from_iter(params))
      .await?;
    let row = rows.next().await?.ok_or_else(|| {
      ZageError::ConfigError(format!(
        "scenario {}: db assertion returned no rows",
        scenario.name
      ))
    })?;

    let ok = match &expect.value {
      ExpectedValue::Integer(expected) => {
        let actual = row.get::<i64>(0)?;
        compare_numeric(actual as f64, &expect.operator, *expected as f64)
      }
      ExpectedValue::Float(expected) => {
        let actual = row
          .get::<f64>(0)
          .or_else(|_| row.get::<i64>(0).map(|v| v as f64))?;
        compare_numeric(actual, &expect.operator, *expected)
      }
      ExpectedValue::String(expected) => {
        let actual = row.get::<String>(0)?;
        match expect.operator.as_str() {
          "eq" => actual == *expected,
          "ne" => actual != *expected,
          _ => false,
        }
      }
      ExpectedValue::Bool(expected) => {
        let actual = row.get::<i64>(0)?;
        let actual = actual != 0;
        match expect.operator.as_str() {
          "eq" => actual == *expected,
          "ne" => actual != *expected,
          _ => false,
        }
      }
    };
    if !ok {
      let description = expect
        .description
        .as_deref()
        .unwrap_or("db assertion failed");
      return Err(ZageError::ConfigError(format!(
        "scenario {}: {}",
        scenario.name, description
      )));
    }
  }
  Ok(())
}

struct EnvGuard {
  key: &'static str,
  previous: Option<String>,
}

impl Drop for EnvGuard {
  fn drop(&mut self) {
    if let Some(value) = self.previous.as_ref() {
      unsafe {
        std::env::set_var(self.key, value);
      }
    } else {
      unsafe {
        std::env::remove_var(self.key);
      }
    }
  }
}

fn set_env_guard(key: &'static str, value: Option<String>) -> EnvGuard {
  let previous = std::env::var(key).ok();
  if let Some(value) = value {
    unsafe {
      std::env::set_var(key, value);
    }
  } else {
    unsafe {
      std::env::remove_var(key);
    }
  }
  EnvGuard { key, previous }
}

fn env_lock() -> std::sync::MutexGuard<'static, ()> {
  static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
  LOCK
    .get_or_init(|| Mutex::new(()))
    .lock()
    .expect("env lock")
}

fn build_phases_config(phases: &HashMap<String, Vec<String>>) -> String {
  let mut labels: Vec<&String> = phases.keys().collect();
  labels.sort();
  let mut output = String::new();
  for label in labels {
    output.push_str(&format!("[phases.{label}]\npatterns = [\n"));
    if let Some(entries) = phases.get(label) {
      for entry in entries {
        output.push_str(&format!("  \"{entry}\",\n"));
      }
    }
    output.push_str("]\n\n");
  }
  output
}

fn load_tier1_spec(path: &Path) -> Result<TestSpec> {
  let contents = fs::read_to_string(path)?;
  toml::from_str(&contents).map_err(|err| ZageError::ConfigError(err.to_string()))
}

pub fn list_tier1_cases(path: &Path) -> Result<Vec<Tier1Case>> {
  let spec = load_tier1_spec(path)?;
  Ok(
    spec
      .scenario
      .iter()
      .enumerate()
      .map(|(index, scenario)| Tier1Case {
        name: scenario.name.clone(),
        index,
      })
      .collect(),
  )
}

pub async fn run_tier1_case(path: &Path, scenario_index: usize) -> Result<()> {
  run_tier1_spec(path, Some(scenario_index)).await
}

#[allow(clippy::await_holding_lock)]
async fn run_tier1_spec(path: &Path, scenario_index: Option<usize>) -> Result<()> {
  let spec = load_tier1_spec(path)?;

  let temp_dir = materialize_filesystem(spec.fs.as_ref())?;
  let db = open_db(temp_dir.path().join("test.db")).await?;
  init(&db.conn).await?;

  let aliases = spec.aliases.clone().unwrap_or_default();
  let test_config = build_test_config(spec.physics.as_ref())?;
  let physics_now = test_config.now;

  seed_history(&db.conn, &spec.history, &temp_dir, physics_now, &aliases).await?;

  let _env_lock = env_lock();
  let model_dir = temp_dir.path().join("model");
  fs::create_dir_all(&model_dir)?;
  let _model_guard = set_env_guard(
    "ZAGE_MODEL_PATH",
    Some(model_dir.to_string_lossy().to_string()),
  );
  crate::rerank::clear_model_cache();
  let run_phase_indexing = spec
    .options
    .as_ref()
    .and_then(|opt| opt.run_phase_indexing)
    .unwrap_or(false);
  let _phase_guard = if run_phase_indexing {
    if let Some(phases) = spec.phases.as_ref() {
      let phases_path = temp_dir.path().join("phases.toml");
      let contents = build_phases_config(phases);
      fs::write(&phases_path, contents)?;
      Some(set_env_guard(
        "ZAGE_PHASES_CONFIG",
        Some(phases_path.to_string_lossy().to_string()),
      ))
    } else {
      Some(set_env_guard("ZAGE_PHASES_CONFIG", None))
    }
  } else {
    let phases_path = temp_dir.path().join("phases_disabled.toml");
    fs::write(&phases_path, "[phases.default]\npatterns = []\n")?;
    Some(set_env_guard(
      "ZAGE_PHASES_CONFIG",
      Some(phases_path.to_string_lossy().to_string()),
    ))
  };

  rebuild_stats(&db.conn, None).await?;

  if spec
    .options
    .as_ref()
    .and_then(|opt| opt.run_sequence_analysis)
    .unwrap_or(false)
  {
    let sequence_config = build_sequence_config(spec.options.as_ref());
    let _ = analyze_sequences(&db.conn, sequence_config).await?;
  }

  let scenarios: Vec<(usize, &Scenario)> = match scenario_index {
    Some(index) => {
      let scenario = spec.scenario.get(index).ok_or_else(|| {
        ZageError::ConfigError(format!(
          "scenario index {index} out of bounds for {}",
          path.display()
        ))
      })?;
      vec![(index, scenario)]
    }
    None => spec.scenario.iter().enumerate().collect(),
  };

  for (_, scenario) in scenarios {
    let config = build_suggest_config(scenario, spec.options.as_ref(), &temp_dir);
    let override_prev = scenario
      .prev_command
      .clone()
      .map(|cmd| (cmd, scenario.prev_exit));
    let suggestions = suggest_for_test_with_aliases(
      &db.conn,
      config,
      test_config.clone(),
      aliases.clone(),
      override_prev,
    )
    .await?;

    assert_expectations(scenario, &suggestions);
    assert_db_expectations(&db.conn, scenario).await?;
  }

  Ok(())
}
