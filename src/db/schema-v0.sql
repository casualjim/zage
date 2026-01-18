CREATE TABLE IF NOT EXISTS shell_history (
  id TEXT PRIMARY KEY,
  command TEXT NOT NULL,
  expanded_command TEXT NOT NULL,
  shellname TEXT NOT NULL,
  working_directory TEXT,
  hostname TEXT,
  username TEXT,
  exit_status INTEGER,
  start_unix_timestamp INTEGER,
  end_unix_timestamp INTEGER,
  session_id INTEGER
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_shell_history_unique ON shell_history (
  command,
  expanded_command,
  shellname,
  working_directory,
  hostname,
  username,
  exit_status,
  start_unix_timestamp,
  end_unix_timestamp,
  session_id
);

CREATE INDEX IF NOT EXISTS idx_shell_history_session_ts ON shell_history (
  session_id,
  COALESCE(start_unix_timestamp, 0),
  id
);

CREATE INDEX IF NOT EXISTS idx_shell_history_ts ON shell_history (
  COALESCE(start_unix_timestamp, 0),
  id
);

CREATE TABLE IF NOT EXISTS command_stats (
  command TEXT PRIMARY KEY,
  freq INTEGER NOT NULL,
  last_seen INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_command_stats_last_seen ON command_stats(last_seen);

CREATE TABLE IF NOT EXISTS context_stats (
  command TEXT NOT NULL,
  working_directory TEXT,
  hostname TEXT,
  username TEXT,
  freq INTEGER NOT NULL,
  last_seen INTEGER NOT NULL,
  PRIMARY KEY (command, working_directory, hostname, username)
);

CREATE INDEX IF NOT EXISTS idx_context_lookup ON context_stats(working_directory, hostname, username);

CREATE TABLE IF NOT EXISTS sequence_stats (
  sequence_json TEXT PRIMARY KEY,
  support INTEGER NOT NULL,
  confidence REAL NOT NULL,
  lift REAL NOT NULL,
  sequence_len INTEGER NOT NULL,
  prefix_json TEXT,
  last_command TEXT,
  context_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_sequence_lift ON sequence_stats(lift);

CREATE TABLE IF NOT EXISTS token_cache (
  command TEXT PRIMARY KEY,
  tokens_json TEXT NOT NULL,
  normalized_json TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS transition_stats (
  prev_command TEXT NOT NULL,
  prev_exit_status INTEGER,
  next_command TEXT NOT NULL,
  freq INTEGER NOT NULL,
  last_seen INTEGER NOT NULL,
  PRIMARY KEY (prev_command, prev_exit_status, next_command)
);

CREATE INDEX IF NOT EXISTS idx_transition_prev ON transition_stats(prev_command, prev_exit_status);
CREATE INDEX IF NOT EXISTS idx_transition_next ON transition_stats(next_command);

CREATE TABLE IF NOT EXISTS workspace_command_stats (
  workspace_root TEXT NOT NULL,
  command TEXT NOT NULL,
  freq INTEGER NOT NULL,
  last_seen INTEGER NOT NULL,
  PRIMARY KEY (workspace_root, command)
);

CREATE INDEX IF NOT EXISTS idx_workspace_command_root ON workspace_command_stats(workspace_root);

CREATE TABLE IF NOT EXISTS workspace_transition_stats (
  workspace_root TEXT NOT NULL,
  prev_command TEXT NOT NULL,
  prev_exit_status INTEGER,
  next_command TEXT NOT NULL,
  freq INTEGER NOT NULL,
  last_seen INTEGER NOT NULL,
  PRIMARY KEY (workspace_root, prev_command, prev_exit_status, next_command)
);

CREATE INDEX IF NOT EXISTS idx_workspace_transition_root_prev ON workspace_transition_stats(workspace_root, prev_command, prev_exit_status);

CREATE TABLE IF NOT EXISTS arg_stats (
  workspace_root TEXT NOT NULL,
  command_head TEXT NOT NULL,
  flags_json TEXT NOT NULL,
  arg_index INTEGER NOT NULL,
  arg_raw TEXT NOT NULL,
  arg_norm TEXT NOT NULL,
  freq INTEGER NOT NULL,
  last_seen INTEGER NOT NULL,
  PRIMARY KEY (workspace_root, command_head, flags_json, arg_index, arg_raw)
);

CREATE INDEX IF NOT EXISTS idx_arg_stats_lookup ON arg_stats(command_head, flags_json, arg_index);
CREATE INDEX IF NOT EXISTS idx_arg_stats_workspace ON arg_stats(workspace_root, command_head);

CREATE TABLE IF NOT EXISTS arg_stats_any (
  workspace_root TEXT NOT NULL,
  command_head TEXT NOT NULL,
  flags_json TEXT NOT NULL,
  arg_raw TEXT NOT NULL,
  arg_norm TEXT NOT NULL,
  freq INTEGER NOT NULL,
  last_seen INTEGER NOT NULL,
  PRIMARY KEY (workspace_root, command_head, flags_json, arg_raw)
);

CREATE INDEX IF NOT EXISTS idx_arg_stats_any_lookup ON arg_stats_any(command_head, flags_json);
CREATE INDEX IF NOT EXISTS idx_arg_stats_any_workspace ON arg_stats_any(workspace_root, command_head);

CREATE TABLE IF NOT EXISTS flag_stats (
  workspace_root TEXT NOT NULL,
  command_head TEXT NOT NULL,
  flag_raw TEXT NOT NULL,
  flag_norm TEXT NOT NULL,
  freq INTEGER NOT NULL,
  last_seen INTEGER NOT NULL,
  PRIMARY KEY (workspace_root, command_head, flag_raw)
);

CREATE INDEX IF NOT EXISTS idx_flag_stats_lookup ON flag_stats(command_head, flag_raw);
CREATE INDEX IF NOT EXISTS idx_flag_stats_workspace ON flag_stats(workspace_root, command_head);

CREATE TABLE IF NOT EXISTS env_stats (
  workspace_root TEXT NOT NULL,
  command_head TEXT NOT NULL,
  env_key TEXT NOT NULL,
  env_raw TEXT NOT NULL,
  env_norm TEXT NOT NULL,
  freq INTEGER NOT NULL,
  last_seen INTEGER NOT NULL,
  PRIMARY KEY (workspace_root, command_head, env_raw)
);

CREATE INDEX IF NOT EXISTS idx_env_stats_key ON env_stats(command_head, env_key);
CREATE INDEX IF NOT EXISTS idx_env_stats_workspace ON env_stats(workspace_root, command_head);

CREATE TABLE IF NOT EXISTS token_sequence_stats (
  sequence_json TEXT PRIMARY KEY,
  support INTEGER NOT NULL,
  confidence REAL NOT NULL,
  lift REAL NOT NULL,
  prefix_len INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_token_sequence_lift ON token_sequence_stats(lift);

CREATE TABLE IF NOT EXISTS phase_stats (
  command_head TEXT NOT NULL,
  phase TEXT NOT NULL,
  confidence REAL NOT NULL,
  freq INTEGER NOT NULL,
  last_seen INTEGER NOT NULL,
  PRIMARY KEY (command_head, phase)
);

CREATE INDEX IF NOT EXISTS idx_phase_stats_head ON phase_stats(command_head);
CREATE INDEX IF NOT EXISTS idx_phase_stats_phase ON phase_stats(phase);

-- Online next-command model (v1)
CREATE TABLE IF NOT EXISTS online_model_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS online_token_embedding (
  bucket INTEGER PRIMARY KEY,
  vec BLOB NOT NULL,
  opt_state BLOB,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS online_command_bias (
  command TEXT PRIMARY KEY,
  bias REAL NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS online_context_bias (
  bucket INTEGER PRIMARY KEY,
  bias REAL NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS online_head_bias (
  head TEXT PRIMARY KEY,
  bias REAL NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS online_group_scalar (
  group_name TEXT PRIMARY KEY,
  value REAL NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS online_replay_global (
  event_id INTEGER PRIMARY KEY,
  payload BLOB NOT NULL,
  sampled_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS online_replay_workspace (
  workspace_root TEXT NOT NULL,
  seq INTEGER NOT NULL,
  payload BLOB NOT NULL,
  PRIMARY KEY (workspace_root, seq)
);

CREATE TABLE IF NOT EXISTS online_feedback (
  shown_id TEXT PRIMARY KEY,
  shown_at INTEGER NOT NULL,
  workspace_root TEXT,
  cwd TEXT,
  prefix TEXT,
  suggestion TEXT NOT NULL,
  accepted_command TEXT,
  accepted_at INTEGER,
  outcome TEXT
);

CREATE INDEX IF NOT EXISTS idx_online_feedback_shown_at ON online_feedback(shown_at);
CREATE INDEX IF NOT EXISTS idx_online_feedback_accepted_at ON online_feedback(accepted_at);
