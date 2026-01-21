use std::os::unix::io::FromRawFd;
use std::os::unix::net::UnixListener as StdUnixListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use deadpool::managed::{Manager as DeadpoolManager, Object, Pool, RecycleError, RecycleResult};
use libsql::{Connection, Database};
use rkyv::{Archive, Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::mpsc;
use tokio::sync::oneshot;
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
use crate::online_model::trainer::train_on_invocations_bulk as train_online_model_bulk;
use crate::predict::aliases::{expand_alias, load_aliases};
use crate::predict::{
  SuggestConfig, Suggestion as InternalSuggestion, update_blend_weights_for_feedback,
};
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
const RECORD_TIMEOUT_MS: u64 = 300;

type RecordResult = std::result::Result<(), String>;

struct RecordMessage {
  invocation: Invocation,
  respond_to: oneshot::Sender<RecordResult>,
}

fn db_busy_timeout_ms() -> u64 {
  std::env::var("ZAGE_DB_BUSY_TIMEOUT_MS")
    .ok()
    .and_then(|val| val.parse::<u64>().ok())
    .unwrap_or(300_000)
}

fn response_timeout_ms(request: &Request) -> u64 {
  match request {
    Request::Record { .. } => RECORD_TIMEOUT_MS,
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
    aliases: Option<String>,
    limit: u32,
    recent_limit: usize,
    use_sequences: bool,
    prefer_full_line: bool,
    include_debug: bool,
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
  pub breakdown: ScoreBreakdown,
  pub debug: Option<SuggestionDebug>,
  pub source: SuggestionSource,
}

#[derive(Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct ScoreBreakdown {
  pub recency: f32,
  pub session_recency: f32,
  pub frequency: f32,
  pub transition: f32,
  pub context: f32,
  pub sequence: f32,
  pub similarity: f32,
  pub embedding_retrieval: f32,
  pub online_model: f32,
}

#[derive(Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct SuggestionDebug {
  pub blend: BlendDebug,
  pub candidate: CandidateDebug,
  pub pipeline: PipelineDebug,
}

#[derive(Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct PipelineDebug {
  pub added_transition: u32,
  pub added_session: u32,
  pub added_embedding: u32,
  pub added_context: u32,
  pub added_workspace: u32,
  pub added_head: u32,
  pub added_sequence: u32,
  pub added_template: u32,
  pub added_recent: u32,
  pub added_global: u32,

  pub total_candidates: u32,
  pub conditional_candidates: u32,

  pub pruned_before: u32,
  pub pruned_after: u32,
  pub pruned_kept_conditional: u32,
}

#[derive(Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct BlendDebug {
  pub model_gate: f32,
  pub model_alpha: f32,
  pub blend_model_weight: f32,
  pub blend_frecency_weight: f32,
  pub blend_sequence_weight: f32,
  pub blend_tier1_weight: f32,

  pub model_feature: f32,
  pub frecency_feature: f32,
  pub sequence_feature: f32,
  pub tier1_feature: f32,

  pub model_contrib: f32,
  pub frecency_contrib: f32,
  pub sequence_contrib: f32,
  pub tier1_contrib: f32,
}

#[derive(Debug, Archive, Serialize, Deserialize)]
#[archive(check_bytes)]
pub struct CandidateDebug {
  pub freq: i64,
  pub workspace_freq: i64,
  pub last_seen: i64,

  pub transition_freq: i64,
  pub workspace_transition_freq: i64,
  pub transition_exit_status_match: bool,

  pub context_freq: i64,
  pub context_cwd_match: bool,
  pub context_host_match: bool,
  pub context_user_match: bool,

  pub session_freq: i64,
  pub session_last_seen: i64,

  pub from_embedding: bool,
  pub sequence_confidence: f32,
  pub sequence_lift: f32,
  pub sequence_prefix_len: u32,
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
    let _ = apply_pragma(
      &conn,
      &format!("PRAGMA busy_timeout={}", db_busy_timeout_ms()),
    )
    .await;
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
  let available = std::thread::available_parallelism()
    .map(|n| n.get())
    .unwrap_or(1);
  info!(
    "server starting: db_kind={:?} available_parallelism={}",
    db_config.kind, available
  );
  let db = open_db_with_config(db_config).await?;
  if matches!(db_config.kind, crate::config::DbKind::Local)
    && let Err(err) = apply_pragma(&db.conn, "PRAGMA journal_mode=WAL").await
  {
    warn!("failed to enable WAL: {}", err);
  }
  if let Err(err) = apply_pragma(
    &db.conn,
    &format!("PRAGMA busy_timeout={}", db_busy_timeout_ms()),
  )
  .await
  {
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

  let (record_tx, mut record_rx) = mpsc::channel::<RecordMessage>(128);
  let conn_writer = pool.clone();
  tokio::spawn(async move {
    let mut train_buffer: Vec<Invocation> = Vec::new();
    let mut tick = tokio::time::interval(Duration::from_millis(200));
    loop {
      tokio::select! {
        Some(msg) = record_rx.recv() => {
          let result = process_record(&conn_writer, &mut train_buffer, msg.invocation).await;
          let _ = msg.respond_to.send(result);
        }
        _ = tick.tick() => {
          if !train_buffer.is_empty() {
            flush_training(&conn_writer, &mut train_buffer).await;
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

  unsafe extern "C" {
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
  record_tx: mpsc::Sender<RecordMessage>,
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
  record_tx: &mpsc::Sender<RecordMessage>,
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
            config.blend.alpha, config.blend.margin_gate, config.blend.min_score_gate
          ));
          lines.push(format!(
            "Replay: global={}, workspace={}, workspaces={}",
            status.replay_global, status.replay_workspace, replay_workspaces
          ));
          lines.push(format!(
            "Tables: meta={}, token_embeddings={}, command_biases={}, context_biases={}, head_biases={}, group_scalars={}, feedback={}",
            status.meta_entries,
            status.token_embeddings,
            status.command_biases,
            status.context_biases,
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
      let (tx, rx) = oneshot::channel::<RecordResult>();
      if record_tx
        .send(RecordMessage {
          invocation,
          respond_to: tx,
        })
        .await
        .is_err()
      {
        return Response::Error {
          message: "ingestion queue closed".to_string(),
        };
      }
      match rx.await {
        Ok(Ok(())) => Response::Ack,
        Ok(Err(message)) => Response::Error { message },
        Err(_) => Response::Error {
          message: "ingestion worker exited".to_string(),
        },
      }
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
      Ok(conn) => {
        let event = OnlineFeedbackEvent {
          shown_id,
          shown_at,
          cwd: working_directory,
          suggestion,
          accepted_command,
          accepted_at,
          outcome,
        };
        match upsert_online_feedback(&conn, event.clone()).await {
          Ok(()) => match update_blend_weights_for_feedback(&conn, &event).await {
            Ok(()) => Response::Ack,
            Err(err) => Response::Error {
              message: err.to_string(),
            },
          },
          Err(err) => Response::Error {
            message: err.to_string(),
          },
        }
      }
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
      aliases,
      limit,
      recent_limit,
      use_sequences,
      prefer_full_line,
      include_debug,
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
        include_debug,
      };
      match pool.get().await {
        Ok(conn) => {
          let runtime_aliases = aliases
            .as_deref()
            .map(crate::predict::aliases::parse_aliases)
            .unwrap_or_else(crate::predict::aliases::load_aliases);
          match crate::predict::suggest_with_aliases(&conn, config, runtime_aliases).await {
            Ok(suggestions) => Response::Suggestions {
              items: map_suggestions(&suggestions),
            },
            Err(err) => Response::Error {
              message: err.to_string(),
            },
          }
        }
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
  let total_start = Instant::now();
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
  let parse_start = Instant::now();
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
  let parse_dur = parse_start.elapsed();
  let expand_start = Instant::now();
  for invocation in invocations.iter_mut() {
    if invocation.expanded_command.is_empty() {
      invocation.expanded_command =
        expand_alias(&invocation.command, &aliases).unwrap_or_else(|| invocation.command.clone());
    }
  }
  let expand_dur = expand_start.elapsed();
  let insert_start = Instant::now();
  import_history(conn, invocations.iter().cloned()).await?;
  let insert_dur = insert_start.elapsed();

  let mut lines = vec![format!("Imported history from {:?}", history_file)];
  lines.push(format!(
    "Import timings: parse={:.2}s alias_expand={:.2}s db_insert={:.2}s",
    parse_dur.as_secs_f64(),
    expand_dur.as_secs_f64(),
    insert_dur.as_secs_f64()
  ));
  info!(
    "import timings: parse={:.2}s alias_expand={:.2}s db_insert={:.2}s",
    parse_dur.as_secs_f64(),
    expand_dur.as_secs_f64(),
    insert_dur.as_secs_f64()
  );
  if request.reset_model {
    let reset_start = Instant::now();
    reset_online_model(conn).await?;
    lines.push("Online model reset".to_string());
    lines.push(format!(
      "Import timings: model_reset={:.2}s",
      reset_start.elapsed().as_secs_f64()
    ));
    info!(
      "import timings: model_reset={:.2}s",
      reset_start.elapsed().as_secs_f64()
    );
  }
  let train_start = Instant::now();
  train_online_model_bulk(conn, &invocations).await?;
  let train_dur = train_start.elapsed();
  lines.push(format!(
    "Online model trained on {} invocations",
    invocations.len()
  ));
  lines.push(format!(
    "Import timings: online_train={:.2}s",
    train_dur.as_secs_f64()
  ));
  info!(
    "import timings: online_train={:.2}s",
    train_dur.as_secs_f64()
  );
  if request.no_index {
    lines.push("Index rebuild skipped (requested via --no-index)".to_string());
    lines.push(format!(
      "Import timings: total={:.2}s",
      total_start.elapsed().as_secs_f64()
    ));
    info!(
      "import timings: total={:.2}s",
      total_start.elapsed().as_secs_f64()
    );
    return Ok(lines);
  }

  let index_start = Instant::now();
  let report = rebuild_stats(conn, None).await?;
  let index_dur = index_start.elapsed();
  let seq_start = Instant::now();
  let seq_report = analyze_sequences(conn, SequenceConfig::default()).await?;
  let seq_dur = seq_start.elapsed();
  let token_seq_start = Instant::now();
  let token_seq_report = analyze_token_sequences(conn, SequenceConfig::default()).await?;
  let token_seq_dur = token_seq_start.elapsed();
  lines.push(format!(
    "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}",
    report.commands, report.transitions, report.contexts, report.token_cache
  ));
  lines.push(format!(
    "Command sequence stats: sequences={}, bigrams={}, trigrams={}",
    seq_report.sequences, seq_report.bigrams, seq_report.trigrams
  ));
  lines.push(format!(
    "Token sequence stats: sequences={}, bigrams={}, trigrams={}",
    token_seq_report.sequences, token_seq_report.bigrams, token_seq_report.trigrams
  ));
  lines.push(format!(
    "Import timings: rebuild_stats={:.2}s sequences={:.2}s token_sequences={:.2}s total={:.2}s",
    index_dur.as_secs_f64(),
    seq_dur.as_secs_f64(),
    token_seq_dur.as_secs_f64(),
    total_start.elapsed().as_secs_f64()
  ));
  info!(
    "import timings: rebuild_stats={:.2}s sequences={:.2}s token_sequences={:.2}s total={:.2}s",
    index_dur.as_secs_f64(),
    seq_dur.as_secs_f64(),
    token_seq_dur.as_secs_f64(),
    total_start.elapsed().as_secs_f64()
  );
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
    "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}",
    report.commands, report.transitions, report.contexts, report.token_cache
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
    "Indexed stats: commands={}, transitions={}, contexts={}, token_cache={}",
    report.commands, report.transitions, report.contexts, report.token_cache
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
            || status.context_biases > 0
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
      breakdown: ScoreBreakdown {
        recency: item.breakdown.recency as f32,
        session_recency: item.breakdown.session_recency as f32,
        frequency: item.breakdown.frequency as f32,
        transition: item.breakdown.transition as f32,
        context: item.breakdown.context as f32,
        sequence: item.breakdown.sequence as f32,
        similarity: item.breakdown.similarity as f32,
        embedding_retrieval: item.breakdown.embedding_retrieval as f32,
        online_model: item.breakdown.online_model as f32,
      },
      debug: item.debug.as_ref().map(|debug| SuggestionDebug {
        blend: BlendDebug {
          model_gate: debug.blend.model_gate as f32,
          model_alpha: debug.blend.model_alpha as f32,
          blend_model_weight: debug.blend.blend_model_weight as f32,
          blend_frecency_weight: debug.blend.blend_frecency_weight as f32,
          blend_sequence_weight: debug.blend.blend_sequence_weight as f32,
          blend_tier1_weight: debug.blend.blend_tier1_weight as f32,
          model_feature: debug.blend.model_feature as f32,
          frecency_feature: debug.blend.frecency_feature as f32,
          sequence_feature: debug.blend.sequence_feature as f32,
          tier1_feature: debug.blend.tier1_feature as f32,
          model_contrib: debug.blend.model_contrib as f32,
          frecency_contrib: debug.blend.frecency_contrib as f32,
          sequence_contrib: debug.blend.sequence_contrib as f32,
          tier1_contrib: debug.blend.tier1_contrib as f32,
        },
        candidate: CandidateDebug {
          freq: debug.candidate.freq,
          workspace_freq: debug.candidate.workspace_freq,
          last_seen: debug.candidate.last_seen,
          transition_freq: debug.candidate.transition_freq,
          workspace_transition_freq: debug.candidate.workspace_transition_freq,
          transition_exit_status_match: debug.candidate.transition_exit_status_match,
          context_freq: debug.candidate.context_freq,
          context_cwd_match: debug.candidate.context_cwd_match,
          context_host_match: debug.candidate.context_host_match,
          context_user_match: debug.candidate.context_user_match,
          session_freq: debug.candidate.session_freq,
          session_last_seen: debug.candidate.session_last_seen,
          from_embedding: debug.candidate.from_embedding,
          sequence_confidence: debug.candidate.sequence_confidence as f32,
          sequence_lift: debug.candidate.sequence_lift as f32,
          sequence_prefix_len: u32::try_from(debug.candidate.sequence_prefix_len)
            .unwrap_or(u32::MAX),
        },
        pipeline: PipelineDebug {
          added_transition: u32::try_from(debug.pipeline.added_transition).unwrap_or(u32::MAX),
          added_session: u32::try_from(debug.pipeline.added_session).unwrap_or(u32::MAX),
          added_embedding: u32::try_from(debug.pipeline.added_embedding).unwrap_or(u32::MAX),
          added_context: u32::try_from(debug.pipeline.added_context).unwrap_or(u32::MAX),
          added_workspace: u32::try_from(debug.pipeline.added_workspace).unwrap_or(u32::MAX),
          added_head: u32::try_from(debug.pipeline.added_head).unwrap_or(u32::MAX),
          added_sequence: u32::try_from(debug.pipeline.added_sequence).unwrap_or(u32::MAX),
          added_template: u32::try_from(debug.pipeline.added_template).unwrap_or(u32::MAX),
          added_recent: u32::try_from(debug.pipeline.added_recent).unwrap_or(u32::MAX),
          added_global: u32::try_from(debug.pipeline.added_global).unwrap_or(u32::MAX),
          total_candidates: u32::try_from(debug.pipeline.total_candidates).unwrap_or(u32::MAX),
          conditional_candidates: u32::try_from(debug.pipeline.conditional_candidates)
            .unwrap_or(u32::MAX),
          pruned_before: u32::try_from(debug.pipeline.pruned_before).unwrap_or(u32::MAX),
          pruned_after: u32::try_from(debug.pipeline.pruned_after).unwrap_or(u32::MAX),
          pruned_kept_conditional: u32::try_from(debug.pipeline.pruned_kept_conditional)
            .unwrap_or(u32::MAX),
        },
      }),
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

async fn process_record(
  pool: &Arc<ConnectionPool>,
  train_buffer: &mut Vec<Invocation>,
  invocation: Invocation,
) -> RecordResult {
  let conn = match pool.get().await {
    Ok(conn) => conn,
    Err(err) => return Err(format!("failed to acquire connection: {err}")),
  };
  match insert_invocation(&conn, &invocation).await {
    Ok(true) => {
      if let Err(err) = update_stats_for_invocation(&conn, &invocation).await {
        warn!(error = ?err, "failed updating stats for invocation");
      }
      train_buffer.push(invocation);
      Ok(())
    }
    Ok(false) => Ok(()),
    Err(err) => Err(format!("failed to record invocation: {err}")),
  }
}

async fn flush_training(pool: &Arc<ConnectionPool>, train_buffer: &mut Vec<Invocation>) {
  if train_buffer.is_empty() {
    return;
  }
  let mut pending = Vec::new();
  std::mem::swap(train_buffer, &mut pending);
  let conn = match pool.get().await {
    Ok(conn) => conn,
    Err(err) => {
      warn!("failed to acquire connection: {}", err);
      train_buffer.extend(pending);
      return;
    }
  };
  if let Err(err) = train_online_model(&conn, &pending).await {
    warn!(error = ?err, "online model training failed");
  }
  pool.sync().await;
  debug!("flushed training batch");
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
