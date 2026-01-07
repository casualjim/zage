use zage::db::{open_db, init, insert_invocation};
use zage::sequence::{SequenceConfig, analyze_sequences};
use zage::shell_history::Invocation;

async fn insert_cmd(conn: &libsql::Connection, command: &str, ts: i64) {
  let inv = Invocation {
    command: command.to_string(),
    shellname: "zsh".to_string(),
    working_directory: Some("/tmp".to_string()),
    hostname: Some("host".to_string()),
    username: Some("user".to_string()),
    exit_status: Some(0),
    start_unix_timestamp: Some(ts),
    end_unix_timestamp: Some(ts + 1),
    session_id: 1,
  };
  let inserted = insert_invocation(conn, &inv).await.unwrap();
  assert!(inserted);
}

#[tokio::test]
async fn test_sequence_mining_basic() {
  let tmp = tempfile::NamedTempFile::new().unwrap();
  let db = open_db(tmp.path()).await.unwrap();
  init(&db.conn).await.unwrap();

  insert_cmd(&db.conn, "git status", 1).await;
  insert_cmd(&db.conn, "git add .", 2).await;
  insert_cmd(&db.conn, "git status", 3).await;
  insert_cmd(&db.conn, "git add .", 4).await;

  let cfg = SequenceConfig {
    min_support: 1,
    min_confidence: 0.0,
    min_lift: 0.0,
    max_len: 3,
  };
  let report = analyze_sequences(&db.conn, cfg).await.unwrap();
  assert!(report.bigrams > 0);

  let mut rows = db
    .conn
    .query("SELECT COUNT(*) FROM sequence_stats", ())
    .await
    .unwrap();
  let row = rows.next().await.unwrap().expect("expected row");
  let count: i64 = row.get(0).unwrap();
  assert!(count > 0);
}
