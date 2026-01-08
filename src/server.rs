use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use libsql::Connection;
use rkyv::{Archive, Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, info, warn};

use crate::db::{insert_invocation, open_db, update_stats_for_invocation};
use crate::predict::{SuggestConfig, Suggestion as InternalSuggestion, suggest};
use crate::rerank::{TrainConfig, model_status, train_model};
use crate::shell_history::Invocation;
use crate::{Result, ZageError};

const DEFAULT_TIMEOUT_MS: u64 = 200;

#[derive(Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub enum Request {
  Record {
    command: String,
    working_directory: String,
    exit_status: i32,
    start_timestamp: i64,
    end_timestamp: i64,
    session_id: u64,
  },
  Suggest {
    current_line: String,
    working_directory: String,
    session_id: u64,
    limit: u32,
  },
  Ping,
  Train,
  Status,
  Shutdown,
}

#[derive(Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub enum Response {
  Ack,
  Suggestions {
    items: Vec<Suggestion>,
  },
  Pong,
  Status {
    model_loaded: bool,
    history_count: u64,
    last_train: Option<i64>,
  },
  Error {
    message: String,
  },
}

#[derive(Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct Suggestion {
  pub command: String,
  pub score: f32,
  pub source: SuggestionSource,
}

#[derive(Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub enum SuggestionSource {
  Recency,
  Frequency,
  Transition,
  Sequence,
  Template,
  Reranker,
}

pub async fn run_server(db_path: &Path) -> Result<()> {
  let socket_path = socket_path()?;
  if socket_path.exists() {
    std::fs::remove_file(&socket_path)?;
  }
  if let Some(parent) = socket_path.parent() {
    std::fs::create_dir_all(parent)?;
  }

  let listener = UnixListener::bind(&socket_path)?;
  info!("zage server listening on {}", socket_path.display());

  let db = open_db(db_path).await?;
  let conn = Arc::new(tokio::sync::Mutex::new(db.conn));

  let (record_tx, mut record_rx) = mpsc::channel::<Invocation>(128);
  let conn_writer = conn.clone();
  tokio::spawn(async move {
    let mut buffer: Vec<Invocation> = Vec::new();
    let mut tick = tokio::time::interval(Duration::from_millis(200));
    loop {
      tokio::select! {
        Some(invocation) = record_rx.recv() => {
          buffer.push(invocation);
          if buffer.len() >= 64 {
            flush_records(&conn_writer, &mut buffer).await;
          }
        }
        _ = tick.tick() => {
          if !buffer.is_empty() {
            flush_records(&conn_writer, &mut buffer).await;
          }
        }
      }
    }
  });

  loop {
    let (stream, _) = listener.accept().await?;
    let conn = conn.clone();
    let record_tx = record_tx.clone();
    tokio::spawn(async move {
      if let Err(err) = handle_client(stream, conn, record_tx).await {
        warn!("server request error: {}", err);
      }
    });
  }
}

pub async fn try_request(request: Request) -> Result<Option<Response>> {
  let socket_path = socket_path()?;
  if !socket_path.exists() {
    return Ok(None);
  }

  let connect = timeout(
    Duration::from_millis(DEFAULT_TIMEOUT_MS),
    UnixStream::connect(&socket_path),
  )
  .await;
  let Ok(Ok(mut stream)) = connect else {
    return Ok(None);
  };

  let payload =
    rkyv::to_bytes::<_, 256>(&request).map_err(|err| ZageError::ConfigError(err.to_string()))?;
  let len = (payload.len() as u32).to_le_bytes();
  stream.write_all(&len).await?;
  stream.write_all(&payload).await?;
  stream.flush().await?;

  let mut size_buf = [0u8; 4];
  if timeout(
    Duration::from_millis(DEFAULT_TIMEOUT_MS),
    stream.read_exact(&mut size_buf),
  )
  .await
  .is_err()
  {
    return Ok(None);
  }
  let size = u32::from_le_bytes(size_buf) as usize;
  let mut buf = vec![0u8; size];
  stream.read_exact(&mut buf).await?;
  let response =
    rkyv::from_bytes::<Response>(&buf).map_err(|err| ZageError::ConfigError(err.to_string()))?;
  Ok(Some(response))
}

fn socket_path() -> Result<PathBuf> {
  if let Ok(path) = std::env::var("ZAGE_SOCKET_PATH") {
    return Ok(PathBuf::from(path));
  }
  if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
    return Ok(PathBuf::from(dir).join("zage.sock"));
  }
  if let Ok(tmp) = std::env::var("TMPDIR") {
    return Ok(PathBuf::from(tmp).join("zage.sock"));
  }
  Ok(PathBuf::from("/tmp/zage.sock"))
}

async fn handle_client(
  mut stream: UnixStream,
  conn: Arc<tokio::sync::Mutex<Connection>>,
  record_tx: mpsc::Sender<Invocation>,
) -> Result<()> {
  loop {
    let mut size_buf = [0u8; 4];
    if stream.read_exact(&mut size_buf).await.is_err() {
      return Ok(());
    }
    let size = u32::from_le_bytes(size_buf) as usize;
    let mut buf = vec![0u8; size];
    stream.read_exact(&mut buf).await?;
    let request =
      rkyv::from_bytes::<Request>(&buf).map_err(|err| ZageError::ConfigError(err.to_string()))?;

    let response = handle_request(request, &conn, &record_tx).await;
    let payload =
      rkyv::to_bytes::<_, 256>(&response).map_err(|err| ZageError::ConfigError(err.to_string()))?;
    let len = (payload.len() as u32).to_le_bytes();
    stream.write_all(&len).await?;
    stream.write_all(&payload).await?;
    stream.flush().await?;
  }
}

async fn handle_request(
  request: Request,
  conn: &Arc<tokio::sync::Mutex<Connection>>,
  record_tx: &mpsc::Sender<Invocation>,
) -> Response {
  match request {
    Request::Ping => Response::Pong,
    Request::Status => status_response(conn).await,
    Request::Train => {
      let conn = conn.lock().await;
      let _ = train_model(&conn, TrainConfig::default()).await;
      Response::Ack
    }
    Request::Shutdown => {
      info!("server shutdown requested");
      std::process::exit(0);
    }
    Request::Record {
      command,
      working_directory,
      exit_status,
      start_timestamp,
      end_timestamp,
      session_id,
    } => {
      let invocation = Invocation {
        command,
        shellname: detect_shellname(),
        working_directory: Some(working_directory),
        hostname: Some(crate::shell_history::get_hostname()),
        username: Some(
          uzers::get_current_username()
            .map(|v| v.to_string_lossy().into_owned())
            .unwrap_or_else(|| "unknown".to_string()),
        ),
        exit_status: Some(exit_status as i64),
        start_unix_timestamp: Some(start_timestamp),
        end_unix_timestamp: Some(end_timestamp),
        session_id: session_id as i64,
      };
      if record_tx.send(invocation).await.is_err() {
        return Response::Error {
          message: "ingestion queue closed".to_string(),
        };
      }
      Response::Ack
    }
    Request::Suggest {
      current_line,
      working_directory,
      session_id,
      limit,
    } => {
      let config = SuggestConfig {
        prefix: Some(current_line),
        cwd: Some(working_directory),
        hostname: None,
        username: None,
        session_id: Some(session_id as i64),
        max_results: limit as usize,
        use_sequences: true,
        recent_limit: 100,
      };
      let conn = conn.lock().await;
      match suggest(&conn, config).await {
        Ok(suggestions) => Response::Suggestions {
          items: map_suggestions(&suggestions),
        },
        Err(err) => Response::Error {
          message: err.to_string(),
        },
      }
    }
  }
}

async fn status_response(conn: &Arc<tokio::sync::Mutex<Connection>>) -> Response {
  let status = model_status().ok().flatten();
  let model_loaded = status.is_some();
  let last_train = status.and_then(|status| status.created_at.parse::<i64>().ok());
  let history_count = {
    let conn = conn.lock().await;
    let rows = conn
      .query("SELECT COUNT(*) FROM shell_history", ())
      .await
      .ok();
    if let Some(mut rows) = rows {
      if let Ok(Some(row)) = rows.next().await {
        row.get::<i64>(0).unwrap_or(0) as u64
      } else {
        0
      }
    } else {
      0
    }
  };
  Response::Status {
    model_loaded,
    history_count,
    last_train,
  }
}

fn map_suggestions(items: &[InternalSuggestion]) -> Vec<Suggestion> {
  items
    .iter()
    .map(|item| Suggestion {
      command: item.command.clone(),
      score: item.score as f32,
      source: suggestion_source(item),
    })
    .collect()
}

fn suggestion_source(item: &InternalSuggestion) -> SuggestionSource {
  if item.breakdown.sequence > 0.0 {
    return SuggestionSource::Sequence;
  }
  let mut best = SuggestionSource::Recency;
  let mut score = item.breakdown.recency;
  if item.breakdown.frequency > score {
    score = item.breakdown.frequency;
    best = SuggestionSource::Frequency;
  }
  if item.breakdown.transition > score {
    score = item.breakdown.transition;
    best = SuggestionSource::Transition;
  }
  if item.breakdown.similarity > score {
    best = SuggestionSource::Reranker;
  }
  best
}

fn detect_shellname() -> String {
  let Some(shell) = std::env::var("SHELL").ok().and_then(|value| {
    Path::new(&value)
      .file_name()
      .map(|name| name.to_string_lossy().to_string())
  }) else {
    return "sh".to_string();
  };
  let normalized = shell.to_lowercase();
  match normalized.as_str() {
    "zsh" | "bash" | "sh" | "fish" | "nushell" | "nu" => normalized,
    _ => shell,
  }
}

async fn flush_records(conn: &Arc<tokio::sync::Mutex<Connection>>, buffer: &mut Vec<Invocation>) {
  if buffer.is_empty() {
    return;
  }
  let mut pending = Vec::new();
  std::mem::swap(buffer, &mut pending);
  let conn = conn.lock().await;
  for invocation in pending {
    match insert_invocation(&conn, &invocation).await {
      Ok(true) => {
        let _ = update_stats_for_invocation(&conn, &invocation).await;
      }
      Ok(false) => {}
      Err(err) => {
        warn!("failed to record invocation: {}", err);
      }
    }
  }
  debug!("flushed ingestion batch");
}
