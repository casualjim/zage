use std::os::unix::io::FromRawFd;
use std::os::unix::net::UnixListener as StdUnixListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use deadpool_libsql::{Manager, Pool};
use libsql::Database;
use rkyv::{Archive, Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, info, warn};

use crate::db::{insert_invocation, open_db, update_stats_for_invocation};
use crate::predict::{SuggestConfig, Suggestion as InternalSuggestion, suggest};
use crate::rerank::{TrainConfig, model_status, train_model, warm_model_cache};
use crate::shell_history::{Invocation, detect_shellname};
use crate::{Result, ZageError};

const DEFAULT_TIMEOUT_MS: u64 = 200;

#[derive(Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub enum Request {
  Record {
    command: String,
    expanded_command: String,
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
    prefer_full_line: bool,
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

struct ConnectionPool {
  pool: Pool,
}

impl ConnectionPool {
  fn new(db: Database) -> Result<Self> {
    let manager = Manager::from_libsql_database(db);
    let default_size = std::thread::available_parallelism()
      .map(|n| n.get())
      .unwrap_or(4);
    let max_size = std::env::var("ZAGE_DB_POOL_SIZE")
      .ok()
      .and_then(|val| val.parse::<usize>().ok())
      .unwrap_or(default_size);
    let pool = Pool::builder(manager)
      .max_size(max_size)
      .build()
      .map_err(|err| ZageError::ConfigError(err.to_string()))?;
    Ok(Self { pool })
  }

  async fn get(&self) -> Result<deadpool_libsql::Object> {
    self
      .pool
      .get()
      .await
      .map_err(|err| ZageError::ConfigError(err.to_string()))
  }
}

pub async fn run_server(db_path: &Path) -> Result<()> {
  let listener = if let Some(listener) = activated_listener()? {
    info!("zage server listening on activated socket");
    listener
  } else {
    let socket_path = socket_path()?;
    if socket_path.exists() {
      std::fs::remove_file(&socket_path)?;
    }
    if let Some(parent) = socket_path.parent() {
      std::fs::create_dir_all(parent)?;
    }

    let listener = UnixListener::bind(&socket_path)?;
    info!("zage server listening on {}", socket_path.display());
    listener
  };

  let db = open_db(db_path).await?;
  let pool = Arc::new(ConnectionPool::new(db.db)?);
  let _ = warm_model_cache();

  let (record_tx, mut record_rx) = mpsc::channel::<Invocation>(128);
  let conn_writer = pool.clone();
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
    let conn = pool.clone();
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
  if cfg!(target_os = "macos") {
    return Ok(PathBuf::from("/tmp/zage.sock"));
  }
  if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
    return Ok(PathBuf::from(dir).join("zage.sock"));
  }
  if let Ok(tmp) = std::env::var("TMPDIR") {
    return Ok(PathBuf::from(tmp).join("zage.sock"));
  }
  Ok(PathBuf::from("/tmp/zage.sock"))
}

fn activated_listener() -> Result<Option<UnixListener>> {
  if let Some(listener) = systemd_listener()? {
    return Ok(Some(listener));
  }
  if let Some(listener) = launchd_listener()? {
    return Ok(Some(listener));
  }
  Ok(None)
}

fn systemd_listener() -> Result<Option<UnixListener>> {
  let listen_pid = std::env::var("LISTEN_PID")
    .ok()
    .and_then(|val| val.parse::<u32>().ok());
  let listen_fds = std::env::var("LISTEN_FDS")
    .ok()
    .and_then(|val| val.parse::<i32>().ok())
    .unwrap_or(0);
  let pid = std::process::id();
  if listen_pid != Some(pid) || listen_fds < 1 {
    return Ok(None);
  }
  unsafe {
    std::env::remove_var("LISTEN_PID");
    std::env::remove_var("LISTEN_FDS");
  }

  let fd = 3;
  let std_listener = unsafe { StdUnixListener::from_raw_fd(fd) };
  std_listener.set_nonblocking(true)?;
  Ok(Some(UnixListener::from_std(std_listener)?))
}

#[cfg(target_os = "macos")]
fn launchd_listener() -> Result<Option<UnixListener>> {
  use std::ffi::CString;
  use std::ptr;

  extern "C" {
    fn launch_activate_socket(
      name: *const libc::c_char,
      fds: *mut *mut libc::c_int,
      cnt: *mut libc::size_t,
    ) -> libc::c_int;
  }

  let name = CString::new("Listeners").map_err(|err| ZageError::ConfigError(err.to_string()))?;
  let mut fds: *mut libc::c_int = ptr::null_mut();
  let mut count: libc::size_t = 0;
  let status = unsafe { launch_activate_socket(name.as_ptr(), &mut fds, &mut count) };
  if status != 0 || fds.is_null() || count == 0 {
    if !fds.is_null() {
      unsafe { libc::free(fds as *mut libc::c_void) };
    }
    return Ok(None);
  }
  let fd = unsafe { *fds };
  unsafe { libc::free(fds as *mut libc::c_void) };

  let std_listener = unsafe { StdUnixListener::from_raw_fd(fd) };
  std_listener.set_nonblocking(true)?;
  Ok(Some(UnixListener::from_std(std_listener)?))
}

#[cfg(not(target_os = "macos"))]
fn launchd_listener() -> Result<Option<UnixListener>> {
  Ok(None)
}

async fn handle_client(
  mut stream: UnixStream,
  pool: Arc<ConnectionPool>,
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

    let response = handle_request(request, &pool, &record_tx).await;
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
  pool: &Arc<ConnectionPool>,
  record_tx: &mpsc::Sender<Invocation>,
) -> Response {
  match request {
    Request::Ping => Response::Pong,
    Request::Status => status_response(pool).await,
    Request::Train => match pool.get().await {
      Ok(conn) => {
        let _ = train_model(&conn, TrainConfig::default()).await;
        Response::Ack
      }
      Err(err) => Response::Error {
        message: err.to_string(),
      },
    },
    Request::Shutdown => {
      info!("server shutdown requested");
      std::process::exit(0);
    }
    Request::Record {
      command,
      expanded_command,
      working_directory,
      exit_status,
      start_timestamp,
      end_timestamp,
      session_id,
    } => {
      let expanded_command = if expanded_command.is_empty() {
        command.clone()
      } else {
        expanded_command
      };
      let invocation = Invocation {
        command,
        expanded_command,
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
      prefer_full_line,
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
        prefer_full_line,
      };
      match pool.get().await {
        Ok(conn) => match suggest(&conn, config).await {
          Ok(suggestions) => Response::Suggestions {
            items: map_suggestions(&suggestions),
          },
          Err(err) => Response::Error {
            message: err.to_string(),
          },
        },
        Err(err) => Response::Error {
          message: err.to_string(),
        },
      }
    }
  }
}

async fn status_response(pool: &Arc<ConnectionPool>) -> Response {
  let status = model_status().ok().flatten();
  let model_loaded = status.is_some();
  let last_train = status.and_then(|status| status.created_at.parse::<i64>().ok());
  let history_count = match pool.get().await {
    Ok(conn) => {
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
    }
    Err(_) => 0,
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

async fn flush_records(pool: &Arc<ConnectionPool>, buffer: &mut Vec<Invocation>) {
  if buffer.is_empty() {
    return;
  }
  let mut pending = Vec::new();
  std::mem::swap(buffer, &mut pending);
  let conn = match pool.get().await {
    Ok(conn) => conn,
    Err(err) => {
      warn!("failed to acquire connection: {}", err);
      return;
    }
  };
  for invocation in pending {
    match insert_invocation(&conn, &invocation).await {
      Ok(true) => {
        let _ = update_stats_for_invocation(&conn, &invocation).await;
      }
      Ok(false) => {}
      Err(err) => {
        warn!(error = ?err, "failed to record invocation");
      }
    }
  }
  debug!("flushed ingestion batch");
}
