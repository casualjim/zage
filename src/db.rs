use std::time::Duration;

use crate::{Result, shell_history::Invocation};
use rusqlite::{Connection, Transaction};

pub fn connect(db_path: &str) -> Result<Connection> {
    let conn = Connection::open(db_path)?;
    conn.busy_timeout(Duration::from_millis(500))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    conn.pragma_update(None, "cache_size", "16777216")?;
    Ok(conn)
}

pub fn insert_invocation(tx: &mut Transaction, invocation: &Invocation) -> Result<()> {
    let working_directory = invocation.working_directory.as_ref().map(|wd| wd.as_ref());
    let hostname = invocation.hostname.as_ref().map(|hn| hn.as_ref());
    let username = invocation.username.as_ref().map(|un| un.as_ref());
    tx.execute(
        "INSERT INTO shell_history (
            id,
            command,
            shellname,
            working_directory,
            hostname,
            username,
            exit_status,
            start_unix_timestamp,
            end_unix_timestamp,
            session_id
        ) VALUES (
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?
        )",
        [
            uuid::Uuid::now_v7().as_bytes(),
            invocation.command.as_slice(),
            invocation.shellname.as_bytes(),
            working_directory.unwrap_or_default(),
            hostname.unwrap_or_default(),
            username.unwrap_or_default(),
            &invocation.exit_status.unwrap_or(0).to_be_bytes(),
            &invocation.start_unix_timestamp.unwrap_or(0).to_be_bytes(),
            &invocation.end_unix_timestamp.unwrap_or(0).to_be_bytes(),
            &invocation.session_id.to_be_bytes(),
        ],
    )?;
    Ok(())
}

pub fn init(db_path: &str) -> Result<()> {
    let mut conn = connect(db_path)?;
    let mut tx = conn.transaction()?;
    init_table(&mut tx)?;
    tx.commit()?;
    Ok(())
}

fn init_table(tx: &mut Transaction) -> Result<()> {
    tx.execute(
        "CREATE TABLE IF NOT EXISTS shell_history (
            id BLOB PRIMARY KEY,
            command TEXT NOT NULL,
            shellname TEXT NOT NULL,
            working_directory TEXT,
            hostname TEXT,
            username TEXT,
            exit_status INTEGER,
            start_unix_timestamp INTEGER,
            end_unix_timestamp INTEGER,
            session_id INTEGER
        )",
        [],
    )?;
    tx.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_shell_history_unique ON shell_history (
            command,
            shellname,
            working_directory,
            hostname,
            username,
            exit_status,
            start_unix_timestamp,
            end_unix_timestamp,
            session_id
        )",
        [],
    )?;
    Ok(())
}
