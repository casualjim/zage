use std::path::PathBuf;

use color_eyre::eyre::{Result, eyre};
use dirs::home_dir;
use tracing::{debug, info};

use crate::db::{Db, import_history};
use crate::indexer::rebuild_stats;
use crate::predict::aliases::{expand_alias, load_aliases};
use crate::sequence::{SequenceConfig, analyze_sequences, analyze_token_sequences};
use crate::shell_history::{Shell, parse_bash_history, parse_zsh_history};

pub async fn run(
  db: &Db,
  file: Option<PathBuf>,
  hostname: Option<String>,
  username: Option<String>,
  shell: Shell,
  no_index: bool,
) -> Result<()> {
  debug!("File argument value: {:?}", file);

  let history_file = if let Some(path) = file {
    path
  } else {
    debug!("No file specified, falling back to default path");
    let mut path = home_dir().ok_or_else(|| eyre!("Cannot find home directory"))?;
    let filename = match shell {
      Shell::Zsh => ".zsh_history",
      Shell::Bash => ".bash_history",
    };
    path.push(filename);
    debug!("Using default history file path: {:?}", path);
    path
  };

  let aliases = load_aliases();
  let mut invocations = match shell {
    Shell::Zsh => parse_zsh_history(&history_file, hostname.clone(), username.clone())?,
    Shell::Bash => parse_bash_history(&history_file, hostname.clone(), username.clone())?,
  };
  for invocation in invocations.iter_mut() {
    if invocation.expanded_command.is_empty() {
      invocation.expanded_command = expand_alias(&invocation.command, &aliases)
        .unwrap_or_else(|| invocation.command.clone());
    }
  }
  import_history(&db.conn, invocations).await?;
  eprintln!("Imported history from {:?}", history_file);
  if no_index {
    eprintln!("Index rebuild skipped (requested via --no-index)");
  } else {
    let report = rebuild_stats(&db.conn, None).await?;
    let seq_report = analyze_sequences(&db.conn, SequenceConfig::default()).await?;
    let token_seq_report = analyze_token_sequences(&db.conn, SequenceConfig::default()).await?;
    eprintln!(
      "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}, phase_stats={}",
      report.commands,
      report.transitions,
      report.contexts,
      report.token_cache,
      report.phase_stats
    );
    eprintln!(
      "Command sequence stats: sequences={}, bigrams={}, trigrams={}",
      seq_report.sequences, seq_report.bigrams, seq_report.trigrams
    );
    eprintln!(
      "Token sequence stats: sequences={}, bigrams={}, trigrams={}",
      token_seq_report.sequences, token_seq_report.bigrams, token_seq_report.trigrams
    );
  }
  info!("Imported history from {:?}", history_file);
  Ok(())
}
