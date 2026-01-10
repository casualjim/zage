use std::time::Duration;

use tempfile::tempdir;
use zage::db::{init, open_db};
use zage::server::{Request, Response};
use zage::{DbConfig, DbKind};

#[tokio::test]
async fn daemon_roundtrip_ping_record_suggest() -> zage::Result<()> {
  let dir = tempdir()?;
  let db_path = dir.path().join("zage.db");
  let socket_path = dir.path().join("zage.sock");
  unsafe {
    std::env::set_var("ZAGE_SOCKET_PATH", &socket_path);
  }

  let db = open_db(&db_path).await?;
  init(&db.conn).await?;

  let db_config = DbConfig {
    kind: DbKind::Local,
    path: db_path,
    url: None,
    auth_token: None,
    encryption_key: None,
    encryption_cipher: None,
    remote_encryption_key: None,
    sync_interval_ms: None,
  };

  let server_handle = tokio::spawn(async move { zage::server::run_server(&db_config).await });

  let mut ready = false;
  for _ in 0..20 {
    if let Ok(Some(Response::Pong)) = zage::server::try_request(Request::Ping).await {
      ready = true;
      break;
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
  }
  assert!(ready, "daemon did not respond to ping");

  let record = Request::Record {
    command: "git status".to_string(),
    expanded_command: "git status".to_string(),
    working_directory: "/tmp".to_string(),
    exit_status: 0,
    start_timestamp: 1,
    end_timestamp: 1,
    session_id: 42,
  };
  match zage::server::try_request(record).await? {
    Some(Response::Ack) => {}
    other => panic!("unexpected record response: {other:?}"),
  }

  tokio::time::sleep(Duration::from_millis(300)).await;

  let suggest = Request::Suggest {
    current_line: Some("git ".to_string()),
    working_directory: Some("/tmp".to_string()),
    hostname: None,
    username: None,
    session_id: Some(42),
    limit: 5,
    recent_limit: 10,
    use_sequences: true,
    prefer_full_line: false,
    timeout_ms: None,
  };
  match zage::server::try_request(suggest).await? {
    Some(Response::Suggestions { .. }) => {}
    other => panic!("unexpected suggest response: {other:?}"),
  }

  server_handle.abort();
  let _ = server_handle.await;
  unsafe {
    std::env::remove_var("ZAGE_SOCKET_PATH");
  }
  Ok(())
}
