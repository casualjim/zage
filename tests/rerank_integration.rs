use tempfile::tempdir;
use zage::db::{init, insert_invocation, open_db};
use zage::indexer::rebuild_stats;
use zage::predict::{SuggestConfig, suggest};
use zage::rerank::{TrainConfig, model_status, train_model};
use zage::shell_history::Invocation;

#[tokio::test]
async fn rerank_training_integrates_with_suggest() -> zage::Result<()> {
  let dir = tempdir()?;
  let db_path = dir.path().join("zage.db");
  let model_path = dir.path().join("model");
  unsafe {
    std::env::set_var("ZAGE_MODEL_PATH", &model_path);
  }

  let db = open_db(&db_path).await?;
  init(&db.conn).await?;

  let commands = ["git status", "git diff", "git status", "git log"];
  for (idx, cmd) in commands.iter().enumerate() {
    let invocation = Invocation {
      command: cmd.to_string(),
      expanded_command: cmd.to_string(),
      shellname: "zsh".to_string(),
      working_directory: Some("/tmp".to_string()),
      hostname: Some("host".to_string()),
      username: Some("user".to_string()),
      exit_status: Some(0),
      start_unix_timestamp: Some(1 + idx as i64),
      end_unix_timestamp: Some(2 + idx as i64),
      session_id: 1,
    };
    let _ = insert_invocation(&db.conn, &invocation).await?;
  }

  let _ = rebuild_stats(&db.conn, None).await?;
  let _ = train_model(
    &db.conn,
    TrainConfig {
      epochs: 20,
      negatives_per_pos: 2,
      min_history: 1,
      max_samples: 100,
    },
  )
  .await?;

  assert!(model_status()?.is_some(), "expected trained model status");

  let suggestions = suggest(
    &db.conn,
    SuggestConfig {
      max_results: 5,
      recent_limit: 4,
      prefix: None,
      cwd: Some("/tmp".to_string()),
      hostname: None,
      username: None,
      session_id: Some(1),
      use_sequences: false,
    },
  )
  .await?;

  assert!(!suggestions.is_empty(), "expected suggestions");

  unsafe {
    std::env::remove_var("ZAGE_MODEL_PATH");
  }
  Ok(())
}
