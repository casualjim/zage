use clap::Parser;
use color_eyre::eyre::Result;
use serde::Serialize;
use serde_json;
use csv;
use std::time::Instant;
use zage::model::markov::MarkovChain;
use zage::model::PredictionModel;
use zage::shell_history::Invocation;

#[derive(Parser, Debug)]
#[clap(name = "simulate", about = "Synthetic simulation for Zage predictions")]
struct Cli {
    /// Synthetic pattern depth (max directory levels)
    #[clap(long, default_value_t = 10)]
    depth: usize,
    /// Number of simulations to run
    #[clap(long, default_value_t = 100)]
    count: usize,
    /// Output format: 'json' or 'csv'
    #[clap(long, default_value = "json")]
    format: String,
}

#[derive(Serialize)]
struct SimulationResult {
    pattern_id: usize,
    depth: usize,
    latency_ms: u128,
    suggestion: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let model = MarkovChain::new();
    let mut results: Vec<SimulationResult> = Vec::new();

    for i in 0..cli.count {
        // Generate synthetic history: cd level1 ... levelN
        let mut history: Vec<Invocation> = Vec::new();
        for lvl in 1..=cli.depth {
            history.push(Invocation {
                command: format!("cd level{}", lvl),
                shellname: "zsh".to_string(),
                working_directory: Some(format!("/tmp/sim/level{}", lvl)),
                hostname: None,
                username: None,
                exit_status: None,
                start_unix_timestamp: None,
                end_unix_timestamp: None,
                session_id: 0,
            });
        }

        let start = Instant::now();
        let preds = model.predict(&history, 1)?;
        let latency = start.elapsed().as_millis();

        results.push(SimulationResult {
            pattern_id: i,
            depth: cli.depth,
            latency_ms: latency,
            suggestion: preds,
        });
    }

    match cli.format.as_str() {
        "csv" => {
            let mut wtr = csv::Writer::from_writer(std::io::stdout());
            for res in &results {
                wtr.serialize(res)?;
            }
            wtr.flush()?;
        }
        _ => {
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
    }

    Ok(())
}
