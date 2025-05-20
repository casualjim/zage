CREATE TABLE IF NOT EXISTS shell_history (
  id TEXT PRIMARY KEY,
  command TEXT NOT NULL,
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
  shellname,
  working_directory,
  hostname,
  username,
  exit_status,
  start_unix_timestamp,
  end_unix_timestamp,
  session_id
);

CREATE TABLE IF NOT EXISTS models (
  model_type TEXT PRIMARY KEY,
  model_data BLOB,
  created_at INTEGER,
  updated_at INTEGER
);

CREATE TABLE IF NOT EXISTS sequence_scores (
    sequence TEXT PRIMARY KEY,
    support INTEGER NOT NULL,
    confidence REAL NOT NULL,
    lift REAL NOT NULL,
    context TEXT
);

-- Runtime state for path component statistics used in path normalization
CREATE TABLE IF NOT EXISTS path_stats (
    id INTEGER PRIMARY KEY CHECK (id = 1),  -- Only one row allowed
    total_paths INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS path_components (
    component TEXT PRIMARY KEY,
    document_frequency INTEGER NOT NULL
);
