use std::os::unix::io::FromRawFd;
use std::os::unix::net::UnixListener as StdUnixListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use deadpool::managed::{Manager as DeadpoolManager, Object, Pool, RecycleError, RecycleResult};
use libsql::{Connection, Database};
use rkyv::{Archive, Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, info, warn};

#[cfg(feature = "pprof")]
use crate::capture_profile;
use crate::config::DbConfig;
use crate::db::{
  OnlineFeedbackEvent, delete_history_by_command, import_history, insert_invocation,
  online_model_group_scalars, online_model_head_biases, online_model_last_updated_at,
  online_model_status, online_model_update_count, online_replay_workspace_roots,
  open_db_with_config, reset_online_model, update_stats_for_invocation, upsert_online_feedback,
};
use crate::indexer::rebuild_stats;
use crate::online_model::trainer::train_on_invocations as train_online_model;
use crate::predict::aliases::{expand_alias, load_aliases};
use crate::predict::{SuggestConfig, Suggestion as InternalSuggestion, suggest};
use crate::sequence::{SequenceConfig, analyze_sequences, analyze_token_sequences};
use crate::shell_history::{
  Invocation, Shell, normalize_shellname, parse_bash_history, parse_zsh_history,
};
use crate::workspace::detect_workspace_for_cwd;
use crate::{Result, ZageError};

const DEFAULT_TIMEOUT_MS: u64 = 1_000;
// We use the server for long-running operations (import/index) over a local UDS.
// Default to a very large timeout to avoid client disconnects during indexing.
const LONG_TIMEOUT_MS: u64 = 86_400_000;

fn response_timeout_ms(request: &Request) -> u64 {
  match request {
    Request::Suggest { timeout_ms, .. } => timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS),
    Request::Yank { .. } => LONG_TIMEOUT_MS,
    #[cfg(feature = "pprof")]
    Request::Pprof { duration_ms, .. } => duration_ms.saturating_add(5_000),
    Request::Import { .. } | Request::Index { .. } | Request::AnalyzeSequences { .. } => {
      LONG_TIMEOUT_MS
    }
    _ => DEFAULT_TIMEOUT_MS,
  }
}

#[derive(Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub enum Request {
  Record {
    command: String,
    expanded_command: String,
    shellname: String,
    working_directory: String,
    exit_status: i32,
    start_timestamp: i64,
    end_timestamp: i64,
    session_id: u64,
  },
  Feedback {
    shown_id: String,
    shown_at: i64,
    working_directory: Option<String>,
    suggestion: String,
    accepted_command: Option<String>,
    accepted_at: Option<i64>,
    outcome: Option<String>,
  },
  Suggest {
    current_line: Option<String>,
    working_directory: Option<String>,
    hostname: Option<String>,
    username: Option<String>,
    session_id: Option<i64>,
    shellname: Option<String>,
    limit: u32,
    recent_limit: usize,
    use_sequences: bool,
    prefer_full_line: bool,
    timeout_ms: Option<u64>,
  },
  Import {
    file: Option<String>,
    base_dir: Option<String>,
    hostname: Option<String>,
    username: Option<String>,
    shell: String,
    no_index: bool,
    reset_model: bool,
  },
  Index {
    max_commands: Option<usize>,
    with_sequences: bool,
    with_embeddings: bool,
  },
  AnalyzeSequences {
    min_support: usize,
    min_confidence: f64,
    min_lift: f64,
    max_len: usize,
  },
  Yank {
    command: String,
    match_expanded: bool,
    with_sequences: bool,
  },
  Ping,
  #[cfg(feature = "pprof")]
  Pprof {
    duration_ms: u64,
    frequency: u32,
    output: String,
  },
  ModelStatus,
  ModelReset,
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
    online_model_version: String,
    online_warmed_up: bool,
    online_update_count: u64,
    online_last_update: Option<i64>,
    online_replay_global: u64,
    online_replay_workspace: u64,
    online_replay_workspaces: u64,
    online_group_scalars: Vec<(String, f64)>,
    online_head_biases: Vec<(String, f64)>,
    online_blend_alpha: f64,
    online_blend_margin_gate: f64,
    online_blend_min_score_gate: f64,
  },
  Text {
    lines: Vec<String>,
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
  Similarity,
}

struct DbManager {
  db: Arc<Database>,
  test_query_count: AtomicU64,
}

impl DbManager {
  fn new(db: Arc<Database>) -> Self {
    Self {
      db,
      test_query_count: AtomicU64::new(0),
    }
  }

  async fn run_test_query(&self, conn: &libsql::Connection) -> Result<(), libsql::Error> {
    let test_query_count = self.test_query_count.fetch_add(1, Ordering::Relaxed);
    let mut rows = conn.query("SELECT ?", [test_query_count]).await?;
    let row = rows.next().await?.ok_or_else(|| {
      libsql::Error::ConnectionFailed("No rows returned from database for test query".into())
    })?;
    let value: u64 = row.get(0)?;
    if value == test_query_count {
      Ok(())
    } else {
      Err(libsql::Error::ConnectionFailed(
        "Unexpected value returned for test query".into(),
      ))
    }
  }
}

impl DeadpoolManager for DbManager {
  type Type = libsql::Connection;
  type Error = libsql::Error;

  async fn create(&self) -> Result<Self::Type, Self::Error> {
    let conn = self.db.connect()?;
    self.run_test_query(&conn).await?;
    Ok(conn)
  }

  async fn recycle(
    &self,
    conn: &mut Self::Type,
    _: &deadpool::managed::Metrics,
  ) -> RecycleResult<Self::Error> {
    self
      .run_test_query(conn)
      .await
      .map_err(RecycleError::Backend)
  }
}

struct ConnectionPool {
  pool: Pool<DbManager>,
  db: Arc<Database>,
  sync_enabled: bool,
}

impl ConnectionPool {
  fn new(db: Database, sync_enabled: bool) -> Result<Self> {
    let db = Arc::new(db);
    let manager = DbManager::new(db.clone());
    let default_size = 30;
    let max_size = std::env::var("ZAGE_DB_POOL_SIZE")
      .ok()
      .and_then(|val| val.parse::<usize>().ok())
      .unwrap_or(default_size);
    let pool = Pool::builder(manager)
      .max_size(max_size)
      .build()
      .map_err(|err| ZageError::ConfigError(err.to_string()))?;
    Ok(Self {
      pool,
      db,
      sync_enabled,
    })
  }

  async fn get(&self) -> Result<Object<DbManager>> {
    let conn = self
      .pool
      .get()
      .await
      .map_err(|err| ZageError::ConfigError(err.to_string()))?;
    let _ = apply_pragma(&conn, "PRAGMA busy_timeout=5000").await;
    Ok(conn)
  }

  async fn sync(&self) {
    if !self.sync_enabled {
      return;
    }
    if let Err(err) = self.db.sync().await {
      warn!("db sync failed: {}", err);
    }
  }
}

pub async fn run_server(db_config: &DbConfig) -> Result<()> {
  let db = open_db_with_config(db_config).await?;
  if let Err(err) = apply_pragma(&db.conn, "PRAGMA journal_mode=WAL").await {
    warn!("failed to enable WAL: {}", err);
  }
  if let Err(err) = apply_pragma(&db.conn, "PRAGMA busy_timeout=5000").await {
    warn!("failed to set busy_timeout: {}", err);
  }
  let sync_enabled = matches!(db_config.kind, crate::config::DbKind::RemoteReplica);
  let pool = Arc::new(ConnectionPool::new(db.db, sync_enabled)?);

  if sync_enabled {
    let pool = pool.clone();
    let interval_ms = db_config.resolved_sync_interval_ms().unwrap_or(1_000);
    tokio::spawn(async move {
      let mut tick = tokio::time::interval(Duration::from_millis(interval_ms));
      loop {
        tick.tick().await;
        pool.sync().await;
      }
    });
  }

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

async fn apply_pragma(conn: &Connection, sql: &str) -> Result<()> {
  let mut rows = conn.query(sql, ()).await?;
  let _ = rows.next().await?;
  Ok(())
}

pub async fn try_request(request: Request) -> Result<Option<Response>> {
  let socket_path = socket_path()?;
  if !socket_path.exists() {
    return Ok(None);
  }

  let request_kind = request_kind(&request);
  let response_timeout_ms = response_timeout_ms(&request);
  let response_timeout = Duration::from_millis(response_timeout_ms);
  let connect = timeout(
    Duration::from_millis(DEFAULT_TIMEOUT_MS),
    UnixStream::connect(&socket_path),
  )
  .await;
  let mut stream = match connect {
    Ok(Ok(stream)) => stream,
    Ok(Err(err)) => {
      return Err(ZageError::ServerRequestError(format!(
        "Failed to connect to server socket {} for {request_kind} request: {err}",
        socket_path.display()
      )));
    }
    Err(_) => {
      return Err(ZageError::ServerRequestTimeout {
        timeout_ms: DEFAULT_TIMEOUT_MS,
        context: format!(
          "connecting to server socket {} for {request_kind} request",
          socket_path.display()
        ),
      });
    }
  };

  let payload =
    rkyv::to_bytes::<_, 256>(&request).map_err(|err| ZageError::ConfigError(err.to_string()))?;
  let len = (payload.len() as u32).to_le_bytes();
  stream.write_all(&len).await?;
  stream.write_all(&payload).await?;
  stream.flush().await?;

  let mut size_buf = [0u8; 4];
  match timeout(response_timeout, stream.read_exact(&mut size_buf)).await {
    Ok(Ok(_)) => {}
    Ok(Err(err)) => {
      return Err(ZageError::ServerRequestError(format!(
        "Failed reading response header from server for {request_kind} request: {err}"
      )));
    }
    Err(_) => {
      return Err(ZageError::ServerRequestTimeout {
        timeout_ms: response_timeout_ms,
        context: format!("waiting for response header for {request_kind} request"),
      });
    }
  }
  let size = u32::from_le_bytes(size_buf) as usize;
  let mut buf = vec![0u8; size];
  match timeout(response_timeout, stream.read_exact(&mut buf)).await {
    Ok(Ok(_)) => {}
    Ok(Err(err)) => {
      return Err(ZageError::ServerRequestError(format!(
        "Failed reading response body from server for {request_kind} request: {err}"
      )));
    }
    Err(_) => {
      return Err(ZageError::ServerRequestTimeout {
        timeout_ms: response_timeout_ms,
        context: format!("waiting for response body for {request_kind} request"),
      });
    }
  }
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

    let request_kind = request_kind(&request);
    let response = handle_request(request, &pool, &record_tx).await;
    if let Response::Error { message } = &response {
      warn!("server request error ({}): {}", request_kind, message);
    }
    let payload =
      rkyv::to_bytes::<_, 256>(&response).map_err(|err| ZageError::ConfigError(err.to_string()))?;
    let len = (payload.len() as u32).to_le_bytes();
    stream.write_all(&len).await?;
    stream.write_all(&payload).await?;
    stream.flush().await?;
  }
}

fn request_kind(request: &Request) -> &'static str {
  match request {
    Request::Record { .. } => "record",
    Request::Feedback { .. } => "feedback",
    Request::Suggest { .. } => "suggest",
    Request::Import { .. } => "import",
    Request::Index { .. } => "index",
    Request::AnalyzeSequences { .. } => "sequences analyze",
    Request::Yank { .. } => "yank",
    Request::Ping => "ping",
    #[cfg(feature = "pprof")]
    Request::Pprof { .. } => "pprof",
    Request::ModelStatus => "model status",
    Request::ModelReset => "model reset",
    Request::Status => "status",
    Request::Shutdown => "shutdown",
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
    #[cfg(feature = "pprof")]
    Request::Pprof {
      duration_ms,
      frequency,
      output,
    } => {
      let output_path = PathBuf::from(output);
      match capture_profile(Duration::from_millis(duration_ms), frequency, &output_path).await {
        Ok(()) => Response::Text {
          lines: vec![format!("Wrote profile to {}", output_path.display())],
        },
        Err(err) => Response::Error {
          message: err.to_string(),
        },
      }
    }
    Request::ModelStatus => match pool.get().await {
      Ok(conn) => match online_model_status(&conn).await {
        Ok(status) => {
          let config = crate::config::OnlineModelConfig::load().unwrap_or_default();
          let update_count = online_model_update_count(&conn).await.unwrap_or(0);
          let last_update = online_model_last_updated_at(&conn).await.ok().flatten();
          let replay_workspaces = online_replay_workspace_roots(&conn).await.unwrap_or(0);
          let group_scalars = online_model_group_scalars(&conn).await.unwrap_or_default();
          let head_biases = online_model_head_biases(&conn, 8).await.unwrap_or_default();
          let warmed_up = status.token_embeddings > 0 || status.group_scalars > 0;

          let mut lines = Vec::new();
          lines.push(format!(
            "Online model: version={}, warmed_up={}, update_count={}, last_update={:?}",
            config.model_version(),
            warmed_up,
            update_count,
            last_update
          ));
          lines.push(format!(
            "Blend: alpha={:.3}, margin_gate={:.3}, min_score_gate={:.3}",
            config.blend.alpha,
            config.blend.margin_gate,
            config.blend.min_score_gate
          ));
          lines.push(format!(
            "Replay: global={}, workspace={}, workspaces={}",
            status.replay_global, status.replay_workspace, replay_workspaces
          ));
          lines.push(format!(
            "Tables: meta={}, token_embeddings={}, command_biases={}, head_biases={}, group_scalars={}, feedback={}",
            status.meta_entries,
            status.token_embeddings,
            status.command_biases,
            status.head_biases,
            status.group_scalars,
            status.feedback
          ));
          if !group_scalars.is_empty() {
            let rendered = group_scalars
              .iter()
              .map(|(name, value)| format!("{name}={value:.3}"))
              .collect::<Vec<_>>()
              .join(", ");
            lines.push(format!("Group scalars: {rendered}"));
          }
          if !head_biases.is_empty() {
            let rendered = head_biases
              .iter()
              .map(|(head, bias)| format!("{head}={bias:.3}"))
              .collect::<Vec<_>>()
              .join(", ");
            lines.push(format!("Top head biases: {rendered}"));
          }
          Response::Text { lines }
        }
        Err(err) => Response::Error {
          message: err.to_string(),
        },
      },
      Err(err) => Response::Error {
        message: err.to_string(),
      },
    },
    Request::ModelReset => match pool.get().await {
      Ok(conn) => match reset_online_model(&conn).await {
        Ok(()) => Response::Text {
          lines: vec!["Online model reset".to_string()],
        },
        Err(err) => Response::Error {
          message: err.to_string(),
        },
      },
      Err(err) => Response::Error {
        message: err.to_string(),
      },
    },
    Request::Import {
      file,
      base_dir,
      hostname,
      username,
      shell,
      no_index,
      reset_model,
    } => match pool.get().await {
      Ok(conn) => {
        let import = ImportRequest {
          file,
          base_dir,
          hostname,
          username,
          shell,
          no_index,
          reset_model,
        };
        match handle_import(&conn, import).await {
          Ok(lines) => Response::Text { lines },
          Err(err) => Response::Error {
            message: err.to_string(),
          },
        }
      }
      Err(err) => Response::Error {
        message: err.to_string(),
      },
    },
    Request::Index {
      max_commands,
      with_sequences,
      with_embeddings,
    } => match pool.get().await {
      Ok(conn) => match handle_index(&conn, max_commands, with_sequences, with_embeddings).await {
        Ok(lines) => Response::Text { lines },
        Err(err) => Response::Error {
          message: err.to_string(),
        },
      },
      Err(err) => Response::Error {
        message: err.to_string(),
      },
    },
    Request::AnalyzeSequences {
      min_support,
      min_confidence,
      min_lift,
      max_len,
    } => match pool.get().await {
      Ok(conn) => {
        match handle_sequences(&conn, min_support, min_confidence, min_lift, max_len).await {
          Ok(lines) => Response::Text { lines },
          Err(err) => Response::Error {
            message: err.to_string(),
          },
        }
      }
      Err(err) => Response::Error {
        message: err.to_string(),
      },
    },
    Request::Yank {
      command,
      match_expanded,
      with_sequences,
    } => match pool.get().await {
      Ok(conn) => match handle_yank(&conn, command, match_expanded, with_sequences).await {
        Ok(lines) => Response::Text { lines },
        Err(err) => Response::Error {
          message: err.to_string(),
        },
      },
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
      shellname,
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
      let workspace = match detect_workspace_for_cwd(&working_directory) {
        Ok(value) => value,
        Err(err) => {
          return Response::Error {
            message: err.to_string(),
          };
        }
      };
      let invocation = Invocation {
        command,
        expanded_command,
        shellname: normalize_shellname(&shellname),
        working_directory: Some(working_directory),
        workspace,
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
    Request::Feedback {
      shown_id,
      shown_at,
      working_directory,
      suggestion,
      accepted_command,
      accepted_at,
      outcome,
    } => match pool.get().await {
      Ok(conn) => match upsert_online_feedback(
        &conn,
        OnlineFeedbackEvent {
          shown_id,
          shown_at,
          cwd: working_directory,
          suggestion,
          accepted_command,
          accepted_at,
          outcome,
        },
      )
      .await
      {
        Ok(()) => Response::Ack,
        Err(err) => Response::Error {
          message: err.to_string(),
        },
      },
      Err(err) => Response::Error {
        message: err.to_string(),
      },
    },
    Request::Suggest {
      current_line,
      working_directory,
      hostname,
      username,
      session_id,
      shellname,
      limit,
      recent_limit,
      use_sequences,
      prefer_full_line,
      timeout_ms: _,
    } => {
      let config = SuggestConfig {
        prefix: current_line,
        cwd: working_directory,
        hostname,
        username,
        session_id,
        shellname,
        max_results: limit as usize,
        use_sequences,
        recent_limit,
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

struct ImportRequest {
  file: Option<String>,
  base_dir: Option<String>,
  hostname: Option<String>,
  username: Option<String>,
  shell: String,
  no_index: bool,
  reset_model: bool,
}

async fn handle_import(conn: &Connection, request: ImportRequest) -> Result<Vec<String>> {
  let history_file = if let Some(path) = request.file {
    let path = PathBuf::from(path);
    if path.is_relative() {
      if let Some(base) = request.base_dir {
        PathBuf::from(base).join(path)
      } else {
        std::env::current_dir()?.join(path)
      }
    } else {
      path
    }
  } else {
    default_history_path(&request.shell)?
  };

  let shell = parse_shell(&request.shell)?;
  let aliases = load_aliases();
  let mut invocations = match shell {
    Shell::Zsh => parse_zsh_history(
      &history_file,
      request.hostname.clone(),
      request.username.clone(),
    )?,
    Shell::Bash => parse_bash_history(
      &history_file,
      request.hostname.clone(),
      request.username.clone(),
    )?,
  };
  for invocation in invocations.iter_mut() {
    if invocation.expanded_command.is_empty() {
      invocation.expanded_command =
        expand_alias(&invocation.command, &aliases).unwrap_or_else(|| invocation.command.clone());
    }
  }
  import_history(conn, invocations.iter().cloned()).await?;

  let mut lines = vec![format!("Imported history from {:?}", history_file)];
  if request.reset_model {
    reset_online_model(conn).await?;
    lines.push("Online model reset".to_string());
  }
  train_online_model(conn, &invocations).await?;
  lines.push(format!(
    "Online model trained on {} invocations",
    invocations.len()
  ));
  if request.no_index {
    lines.push("Index rebuild skipped (requested via --no-index)".to_string());
    return Ok(lines);
  }

  let report = rebuild_stats(conn, None).await?;
  let seq_report = analyze_sequences(conn, SequenceConfig::default()).await?;
  let token_seq_report = analyze_token_sequences(conn, SequenceConfig::default()).await?;
  lines.push(format!(
    "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}, phase_stats={}",
    report.commands, report.transitions, report.contexts, report.token_cache, report.phase_stats
  ));
  lines.push(format!(
    "Command sequence stats: sequences={}, bigrams={}, trigrams={}",
    seq_report.sequences, seq_report.bigrams, seq_report.trigrams
  ));
  lines.push(format!(
    "Token sequence stats: sequences={}, bigrams={}, trigrams={}",
    token_seq_report.sequences, token_seq_report.bigrams, token_seq_report.trigrams
  ));
  Ok(lines)
}

async fn handle_index(
  conn: &Connection,
  max_commands: Option<usize>,
  with_sequences: bool,
  with_embeddings: bool,
) -> Result<Vec<String>> {
  let report = rebuild_stats(conn, max_commands).await?;
  let mut lines = vec![format!(
    "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}, phase_stats={}",
    report.commands, report.transitions, report.contexts, report.token_cache, report.phase_stats
  )];

  if with_sequences {
    let seq_report = analyze_sequences(conn, SequenceConfig::default()).await?;
    lines.push(format!(
      "Command sequence stats: sequences={}, bigrams={}, trigrams={}",
      seq_report.sequences, seq_report.bigrams, seq_report.trigrams
    ));
    let token_report = analyze_token_sequences(conn, SequenceConfig::default()).await?;
    lines.push(format!(
      "Token sequence stats: sequences={}, bigrams={}, trigrams={}",
      token_report.sequences, token_report.bigrams, token_report.trigrams
    ));
  }

  if with_embeddings {
    let count = crate::embeddings::index_command_embeddings(conn, max_commands).await?;
    lines.push(format!("Command embeddings: embedded={count}"));
  }

  Ok(lines)
}

async fn handle_sequences(
  conn: &Connection,
  min_support: usize,
  min_confidence: f64,
  min_lift: f64,
  max_len: usize,
) -> Result<Vec<String>> {
  let config = SequenceConfig {
    min_support,
    min_confidence,
    min_lift,
    max_len,
  };
  let report = analyze_sequences(conn, config.clone()).await?;
  let token_report = analyze_token_sequences(conn, config).await?;
  Ok(vec![
    format!(
      "Command sequence stats: sequences={}, bigrams={}, trigrams={}",
      report.sequences, report.bigrams, report.trigrams
    ),
    format!(
      "Token sequence stats: sequences={}, bigrams={}, trigrams={}",
      token_report.sequences, token_report.bigrams, token_report.trigrams
    ),
  ])
}

async fn handle_yank(
  conn: &Connection,
  command: String,
  match_expanded: bool,
  with_sequences: bool,
) -> Result<Vec<String>> {
  let removed = delete_history_by_command(conn, &command, match_expanded).await?;
  if removed == 0 {
    return Ok(vec![format!("No history entries matched {command:?}")]);
  }

  let mut lines = Vec::new();
  if match_expanded {
    lines.push(format!(
      "Removed {} history entries matching command or expanded_command: {:?}",
      removed, command
    ));
  } else {
    lines.push(format!(
      "Removed {} history entries matching command: {:?}",
      removed, command
    ));
  }

  let report = rebuild_stats(conn, None).await?;
  lines.push(format!(
    "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}, phase_stats={}",
    report.commands, report.transitions, report.contexts, report.token_cache, report.phase_stats
  ));

  if with_sequences {
    let seq_report = analyze_sequences(conn, SequenceConfig::default()).await?;
    lines.push(format!(
      "Command sequence stats: sequences={}, bigrams={}, trigrams={}",
      seq_report.sequences, seq_report.bigrams, seq_report.trigrams
    ));
    let token_report = analyze_token_sequences(conn, SequenceConfig::default()).await?;
    lines.push(format!(
      "Token sequence stats: sequences={}, bigrams={}, trigrams={}",
      token_report.sequences, token_report.bigrams, token_report.trigrams
    ));
  }

  Ok(lines)
}

fn parse_shell(shell: &str) -> Result<Shell> {
  match shell.to_lowercase().as_str() {
    "zsh" => Ok(Shell::Zsh),
    "bash" => Ok(Shell::Bash),
    other => Err(ZageError::ConfigError(format!(
      "unsupported shell: {other}"
    ))),
  }
}

fn default_history_path(shell: &str) -> Result<PathBuf> {
  let mut path =
    dirs::home_dir().ok_or_else(|| ZageError::ConfigError("missing home dir".to_string()))?;
  let filename = match parse_shell(shell)? {
    Shell::Zsh => ".zsh_history",
    Shell::Bash => ".bash_history",
  };
  path.push(filename);
  Ok(path)
}

async fn status_response(pool: &Arc<ConnectionPool>) -> Response {
  let default_config = crate::config::OnlineModelConfig::default();
  let (model_loaded, history_count, last_train, online) = match pool.get().await {
    Ok(conn) => {
      let model_status = online_model_status(&conn).await.ok();
      let model_loaded = model_status
        .as_ref()
        .map(|status| {
          status.meta_entries > 0
            || status.token_embeddings > 0
            || status.command_biases > 0
            || status.head_biases > 0
            || status.group_scalars > 0
            || status.replay_global > 0
            || status.replay_workspace > 0
            || status.feedback > 0
        })
        .unwrap_or(false);
      let history_count = match conn.query("SELECT COUNT(*) FROM shell_history", ()).await {
        Ok(mut rows) => match rows.next().await {
          Ok(Some(row)) => row.get::<i64>(0).unwrap_or(0) as u64,
          _ => 0,
        },
        Err(_) => 0,
      };
      let last_train = online_model_last_updated_at(&conn).await.ok().flatten();
      let config =
        crate::config::OnlineModelConfig::load().unwrap_or_else(|_| default_config.clone());
      let update_count = online_model_update_count(&conn).await.unwrap_or(0);
      let replay_workspaces = online_replay_workspace_roots(&conn).await.unwrap_or(0);
      let group_scalars = online_model_group_scalars(&conn).await.unwrap_or_default();
      let head_biases = online_model_head_biases(&conn, 8).await.unwrap_or_default();
      let warmed_up = model_status
        .as_ref()
        .map(|status| status.token_embeddings > 0 || status.group_scalars > 0)
        .unwrap_or(false);
      let blend_alpha = config.blend.alpha;
      let blend_margin_gate = config.blend.margin_gate;
      let blend_min_score_gate = config.blend.min_score_gate;
      let (replay_global, replay_workspace) = model_status
        .map(|status| (status.replay_global, status.replay_workspace))
        .unwrap_or((0, 0));
      (
        model_loaded,
        history_count,
        last_train,
        (
          config.model_version(),
          warmed_up,
          update_count,
          last_train,
          replay_global,
          replay_workspace,
          replay_workspaces,
          group_scalars,
          head_biases,
          blend_alpha,
          blend_margin_gate,
          blend_min_score_gate,
        ),
      )
    }
    Err(_) => (
      false,
      0,
      None,
      (
        default_config.model_version(),
        false,
        0,
        None,
        0,
        0,
        0,
        Vec::new(),
        Vec::new(),
        default_config.blend.alpha,
        default_config.blend.margin_gate,
        default_config.blend.min_score_gate,
      ),
    ),
  };
  let (
    online_model_version,
    online_warmed_up,
    online_update_count,
    online_last_update,
    online_replay_global,
    online_replay_workspace,
    online_replay_workspaces,
    online_group_scalars,
    online_head_biases,
    online_blend_alpha,
    online_blend_margin_gate,
    online_blend_min_score_gate,
  ) = online;
  Response::Status {
    model_loaded,
    history_count,
    last_train,
    online_model_version,
    online_warmed_up,
    online_update_count,
    online_last_update,
    online_replay_global,
    online_replay_workspace,
    online_replay_workspaces,
    online_group_scalars,
    online_head_biases,
    online_blend_alpha,
    online_blend_margin_gate,
    online_blend_min_score_gate,
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
    best = SuggestionSource::Similarity;
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
  let mut inserted = Vec::new();
  for invocation in pending {
    match insert_invocation(&conn, &invocation).await {
      Ok(true) => {
        let _ = update_stats_for_invocation(&conn, &invocation).await;
        inserted.push(invocation);
      }
      Ok(false) => {}
      Err(err) => {
        warn!(error = ?err, "failed to record invocation");
      }
    }
  }
  if !inserted.is_empty()
    && let Err(err) = train_online_model(&conn, &inserted).await
  {
    warn!(error = ?err, "online model training failed");
  }
  pool.sync().await;
  debug!("flushed ingestion batch");
}

#[cfg(test)]
mod tests {
  use super::DEFAULT_TIMEOUT_MS;

  #[test]
  fn suggest_default_timeout_should_be_human_reasonable() {
    // 200ms is too small for non-trivial histories and results in user-visible failures.
    let timeout = std::hint::black_box(DEFAULT_TIMEOUT_MS);
    assert!(timeout >= 1_000);
  }
}
