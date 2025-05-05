use clap::Parser;
use color_eyre::eyre::Result;
use serde::Serialize;
use csv;
use itertools::iproduct;
use std::io;
use zage::model::markov::MarkovChain;
use zage::shell_history::Invocation;

#[derive(Parser, Debug)]
#[clap(name = "simulate_context", about = "Scenario-based context simulation for Zage predictions")]
struct Cli {
    /// Comma-separated list of directories
    #[clap(long, default_value = "projA,projB")]
    dirs: String,
    /// Comma-separated list of hostnames
    #[clap(long, default_value = "host1,host2")]
    hosts: String,
    /// Comma-separated list of usernames
    #[clap(long, default_value = "alice,bob")]
    users: String,
    /// Comma-separated list of exit statuses
    #[clap(long, default_value = "0,1")]
    statuses: String,
    /// Output format: 'json', 'csv', or 'mermaid'
    #[clap(long, default_value = "json")]
    format: String,
}

#[derive(Serialize)]
struct ScenarioResult {
    scenario_id: usize,
    cwd: String,
    hostname: String,
    username: String,
    exit_status: i64,
    predicted: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let dirs: Vec<String> = cli.dirs.split(',').map(|s| s.to_string()).collect();
    let hosts: Vec<String> = cli.hosts.split(',').map(|s| s.to_string()).collect();
    let users: Vec<String> = cli.users.split(',').map(|s| s.to_string()).collect();
    let statuses: Vec<i64> = cli.statuses.split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    let mut results: Vec<ScenarioResult> = Vec::new();

    let mut scenario_id = 0;
    for (dir, host, user, status) in iproduct!(dirs.iter(), hosts.iter(), users.iter(), statuses.iter()) {
        let prev_cmd = "start";
        let next_cmd = format!("next_{}", scenario_id);

        let inv_prev = Invocation {
            command: prev_cmd.to_string(),
            shellname: "zsh".to_string(),
            working_directory: Some(dir.clone()),
            hostname: Some(host.clone()),
            username: Some(user.clone()),
            exit_status: Some(*status),
            start_unix_timestamp: None,
            end_unix_timestamp: None,
            session_id: 0,
        };
        let inv_next = Invocation {
            command: next_cmd.clone(),
            shellname: "zsh".to_string(),
            working_directory: Some(dir.clone()),
            hostname: Some(host.clone()),
            username: Some(user.clone()),
            exit_status: Some(*status),
            start_unix_timestamp: None,
            end_unix_timestamp: None,
            session_id: 0,
        };

        let mut model = MarkovChain::new();
        model.set_use_context(true);
        model.train(vec![inv_prev.clone(), inv_next.clone()])?;

        let predicted = model.predict(&[inv_prev.clone()], 1)?;

        results.push(ScenarioResult {
            scenario_id,
            cwd: dir.clone(),
            hostname: host.clone(),
            username: user.clone(),
            exit_status: *status,
            predicted,
        });

        scenario_id += 1;
    }

    match cli.format.as_str() {
        "csv" => {
            let mut wtr = csv::Writer::from_writer(io::stdout());
            for r in &results {
                wtr.serialize(r)?;
            }
            wtr.flush()?;
        }
        "mermaid" => {
            println!("```mermaid\nflowchart LR");
            for r in &results {
                let pred = r.predicted.get(0).cloned().unwrap_or_default();
                println!(
                    "  subgraph scenario{}[{}, {}, {}, status={}]",
                    r.scenario_id, r.cwd, r.hostname, r.username, r.exit_status
                );
                println!(
                    "    start{}(start) --> next{}[\"{}\"]",
                    r.scenario_id, r.scenario_id, pred
                );
                println!("  end");
            }
            println!("```");
        }
        _ => {
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
    }
    Ok(())
}
