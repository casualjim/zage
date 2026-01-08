use libsql::params::IntoParams;
use libsql::{Connection, Rows};

use crate::Result;

pub(crate) async fn query_prepared(
  conn: &Connection,
  sql: &str,
  params: impl IntoParams,
) -> Result<Rows> {
  let mut stmt = conn.prepare(sql).await?;
  let rows = stmt.query(params).await?;
  Ok(rows)
}
