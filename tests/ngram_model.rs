use rusqlite::Connection;
use zage::Result;
use zage::db::init_table;
use zage::model::PredictionModel;
use zage::model::ngram::NGramModel;
use zage::shell_history::Invocation;

/// Helper to create a test Invocation
fn create_test_invocation(command: &str, working_dir: Option<&str>) -> Invocation {
  Invocation {
    command: command.to_string(),
    shellname: "zsh".to_string(),
    working_directory: working_dir.map(|wd| wd.to_string()),
    hostname: None,
    username: None,
    exit_status: None,
    start_unix_timestamp: None,
    end_unix_timestamp: None,
    session_id: 0,
  }
}

#[test]
fn test_ngram_training() -> Result<()> {
  let mut model = NGramModel::new(2);
  let invocations = vec![
    create_test_invocation("git status", Some("/project")),
    create_test_invocation("git add .", Some("/project")),
    create_test_invocation("git commit -m 'update'", Some("/project")),
    create_test_invocation("git push", Some("/project")),
    create_test_invocation("git status", Some("/project")),
    create_test_invocation("git pull", Some("/project")),
  ];
  model.train(invocations)?;
  let recent = vec![create_test_invocation("git status", Some("/project"))];
  let predictions = model.predict(&recent, 2)?;
  assert!(!predictions.is_empty());
  assert!(
    predictions.contains(&"git add .".to_string()) || predictions.contains(&"git pull".to_string())
  );
  Ok(())
}

#[test]
fn test_ngram_update() -> Result<()> {
  let mut model = NGramModel::new(2);
  let inv1 = vec![
    create_test_invocation("git status", Some("/project")),
    create_test_invocation("git add .", Some("/project")),
  ];
  model.train(inv1)?;
  let inv2 = vec![
    create_test_invocation("git status", Some("/project")),
    create_test_invocation("git pull", Some("/project")),
    create_test_invocation("git status", Some("/project")),
    create_test_invocation("git pull", Some("/project")),
  ];
  model.update(inv2)?;
  let recent = vec![create_test_invocation("git status", Some("/project"))];
  let predictions = model.predict(&recent, 2)?;
  assert_eq!(predictions[0], "git pull");
  Ok(())
}

#[test]
fn test_directory_context() -> Result<()> {
  let mut model = NGramModel::new(2);
  let invocations = vec![
    create_test_invocation("ls", Some("/project1")),
    create_test_invocation("cd src", Some("/project1")),
    create_test_invocation("ls", Some("/project1")),
    create_test_invocation("cd src", Some("/project1")),
    create_test_invocation("ls", Some("/project2")),
    create_test_invocation("make", Some("/project2")),
  ];
  model.train(invocations)?;
  let recent1 = vec![create_test_invocation("ls", Some("/project1"))];
  let preds1 = model.predict(&recent1, 1)?;
  assert_eq!(preds1, vec!["cd src".to_string()]);
  let recent2 = vec![create_test_invocation("ls", Some("/project2"))];
  let preds2 = model.predict(&recent2, 1)?;
  assert_eq!(preds2, vec!["make".to_string()]);
  Ok(())
}

#[test]
fn test_predict_simple() -> Result<()> {
  let mut model = NGramModel::new(2);
  let invocations = vec![
    create_test_invocation("ls", Some("/home")),
    create_test_invocation("pwd", Some("/home")),
    create_test_invocation("ls", Some("/home")),
  ];
  model.train(invocations)?;
  let recent = vec![create_test_invocation("pwd", Some("/home"))];
  let preds = model.predict(&recent, 1)?;
  assert_eq!(preds, vec!["ls".to_string()]);
  Ok(())
}

#[test]
fn test_predict_insufficient_history() -> Result<()> {
  let model = NGramModel::new(3);
  let recent = vec![create_test_invocation("ls", Some("/home"))];
  let preds = model.predict(&recent, 5)?;
  assert!(preds.is_empty());
  Ok(())
}

#[test]
fn test_stats() -> Result<()> {
  let mut model = NGramModel::new(2);
  let invocations = vec![
    create_test_invocation("ls", Some("/home")),
    create_test_invocation("pwd", Some("/home")),
    create_test_invocation("ls", Some("/home")),
  ];
  model.train(invocations)?;
  let stats = model.stats();
  assert_eq!(stats.n_value, 2);
  assert_eq!(stats.total_commands, 3);
  assert_eq!(stats.context_count, 2);
  assert_eq!(stats.command_count, 2);
  assert_eq!(stats.dir_context_count, 2);
  Ok(())
}

#[test]
fn test_db_save_and_load() -> Result<()> {
  let mut conn = Connection::open_in_memory()?;
  {
    let mut tx = conn.transaction()?;
    init_table(&mut tx)?;
    tx.commit()?;
  }
  let mut model = NGramModel::new(2);
  let invocations = vec![
    create_test_invocation("ls", Some("/home")),
    create_test_invocation("pwd", Some("/home")),
    create_test_invocation("ls", Some("/home")),
  ];
  model.train(invocations)?;
  model.save_to_db(&mut conn)?;
  let loaded = NGramModel::load_from_db(&mut conn, 2)?;
  assert_eq!(model.stats().total_commands, loaded.stats().total_commands);
  Ok(())
}
