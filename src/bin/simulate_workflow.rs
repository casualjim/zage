use clap::Parser;
use color_eyre::eyre::Result;
use csv;
use rand::seq::IndexedRandom;
use serde::{Deserialize, Serialize};
use serde_json;
use serde_yaml;
use std::fs;
use std::io;
use zage::model::markov::MarkovChain;
use zage::shell_history::Invocation;

/// Simulate workflows from scenario templates and validate predictions
#[derive(Parser, Debug)]
#[clap(
  name = "simulate_workflow",
  about = "Simulate and validate command predictions using scenario definitions"
)]
struct Cli {
  /// Path to scenarios YAML file
  #[clap(long, default_value = "docs/dev/scenarios.yaml")]
  scenarios: String,
  /// Target history length per scenario
  #[clap(long, default_value_t = 100)]
  history_length: usize,
  /// Output format: json, csv, or mermaid
  #[clap(long, default_value = "json")]
  format: String,
}

#[derive(Debug, Deserialize)]
struct ScenariosFile {
  scenarios: Vec<ScenarioDef>,
}

#[derive(Debug, Deserialize)]
struct ScenarioDef {
  id: usize,
  name: String,
  context: ContextDef,
  command_history: Vec<String>,
  #[serde(default)]
  expectations: Vec<ExpectationDef>,
}

#[derive(Debug, Deserialize)]
struct ContextDef {
  cwd: String,
  hostname: String,
  username: String,
  exit_status: i64,
}

#[derive(Debug, Deserialize)]
struct ExpectationDef {
  #[serde(default)]
  prev_command: String,
  expected: String,
}

#[derive(Serialize)]
struct Row {
  scenario_id: usize,
  scenario_name: String,
  prev_command: String,
  expected: String,
  predicted: String,
  success: bool,
}

fn main() -> Result<()> {
  let cli = Cli::parse();
  let data = fs::read_to_string(&cli.scenarios)?;
  let scenarios_file: ScenariosFile = serde_yaml::from_str(&data)?;
  let mut rows: Vec<Row> = Vec::new();

  let noise_pool = vec![
    "ls",
    "pwd",
    "echo $PWD",
    "date",
    "whoami",
    "cd .",
    "echo Hello",
  ];
  let mut rng = rand::rng();

  for scenario in scenarios_file.scenarios {
    // Build core invocations
    let core: Vec<Invocation> = scenario
      .command_history
      .iter()
      .map(|cmd| Invocation {
        command: cmd.clone(),
        shellname: "zsh".to_string(),
        working_directory: Some(scenario.context.cwd.clone()),
        hostname: Some(scenario.context.hostname.clone()),
        username: Some(scenario.context.username.clone()),
        exit_status: Some(scenario.context.exit_status),
        start_unix_timestamp: None,
        end_unix_timestamp: None,
        session_id: 0,
      })
      .collect();
    // Expand with noise
    let total = cli.history_length.max(core.len());
    let mut expanded: Vec<Invocation> = Vec::new();
    let noise_each = if core.is_empty() {
      0
    } else {
      (total - core.len()) / core.len()
    };
    for inv in &core {
      for _ in 0..noise_each {
        if let Some(cmd) = noise_pool.choose(&mut rng) {
          expanded.push(Invocation {
            command: cmd.to_string(),
            shellname: "zsh".to_string(),
            working_directory: Some(scenario.context.cwd.clone()),
            hostname: Some(scenario.context.hostname.clone()),
            username: Some(scenario.context.username.clone()),
            exit_status: Some(scenario.context.exit_status),
            start_unix_timestamp: None,
            end_unix_timestamp: None,
            session_id: 0,
          });
        }
      }
      expanded.push(inv.clone());
    }
    while expanded.len() < total {
      if let Some(cmd) = noise_pool.choose(&mut rng) {
        expanded.push(Invocation {
          command: cmd.to_string(),
          shellname: "zsh".to_string(),
          working_directory: Some(scenario.context.cwd.clone()),
          hostname: Some(scenario.context.hostname.clone()),
          username: Some(scenario.context.username.clone()),
          exit_status: Some(scenario.context.exit_status),
          start_unix_timestamp: None,
          end_unix_timestamp: None,
          session_id: 0,
        });
      }
    }
    // Train model
    let mut model = MarkovChain::new();
    model.set_use_context(true);
    model.train(expanded.clone())?;
    // Validate
    for exp in scenario.expectations {
      if let Some(pos) = expanded
        .iter()
        .position(|inv| inv.command == exp.prev_command)
      {
        let inv_prev = expanded[pos].clone();
        let preds = model.predict(&[inv_prev], 1)?;
        let pred = preds.get(0).cloned().unwrap_or_default();
        let ok = pred == exp.expected;
        rows.push(Row {
          scenario_id: scenario.id,
          scenario_name: scenario.name.clone(),
          prev_command: exp.prev_command.clone(),
          expected: exp.expected.clone(),
          predicted: pred,
          success: ok,
        });
      } else {
        rows.push(Row {
          scenario_id: scenario.id,
          scenario_name: scenario.name.clone(),
          prev_command: exp.prev_command.clone(),
          expected: exp.expected.clone(),
          predicted: String::new(),
          success: false,
        });
      }
    }
  }
  // Output
  match cli.format.as_str() {
    "csv" => {
      let mut wtr = csv::Writer::from_writer(io::stdout());
      for row in rows {
        wtr.serialize(row)?;
      }
      wtr.flush()?;
    }
    "mermaid" => {
      println!("```mermaid\nflowchart LR");
      for row in rows {
        println!(
          "  subgraph scenario{st}[{nm}]",
          st = row.scenario_id,
          nm = row.scenario_name
        );
        println!(
          "    \"{prev}\" --> \"{pred}\"",
          prev = row.prev_command,
          pred = row.predicted
        );
        println!("  end");
      }
      println!("```");
    }
    _ => {
      println!("{}", serde_json::to_string_pretty(&rows)?);
    }
  }
  Ok(())
}
