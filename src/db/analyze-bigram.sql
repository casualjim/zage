WITH bigrams AS (
  SELECT LAG(command) OVER (ORDER BY rowid) AS c1,
        command AS c2
  FROM shell_history
), counts AS (
  SELECT c1, c2, COUNT(*) AS support
  FROM bigrams
  WHERE c1 IS NOT NULL
  GROUP BY c1, c2
), prefix AS (
  SELECT c1, COUNT(*) AS sp
  FROM bigrams
  WHERE c1 IS NOT NULL
  GROUP BY c1
), suffix AS (
  SELECT c2, COUNT(*) AS ss
  FROM bigrams
  GROUP BY c2
), total AS (
  SELECT COUNT(*) AS tot FROM shell_history
)
INSERT OR REPLACE INTO sequence_scores(sequence, support, confidence, lift, context)
SELECT
  json_array(counts.c1, counts.c2),
  counts.support,
  counts.support * 1.0 / prefix.sp,
  (counts.support * 1.0 / prefix.sp) / (suffix.ss * 1.0 / total.tot),
  json_object(
    'working_directory', MAX(h1.working_directory),
    'hostname', MAX(h1.hostname),
    'username', MAX(h1.username),
    'exit_status', json_array(MAX(h1.exit_status), MAX(h2.exit_status)),
    'session_id', MAX(h1.session_id),
    'time_info', json_object(
      'start_time', MIN(h1.start_unix_timestamp),
      'end_time', MAX(h2.end_unix_timestamp)
    )
  )
FROM counts
JOIN prefix ON counts.c1 = prefix.c1
JOIN suffix ON counts.c2 = suffix.c2
CROSS JOIN total
JOIN shell_history h1 ON counts.c1 = h1.command
JOIN shell_history h2 ON counts.c2 = h2.command
WHERE
  counts.support    >= :min_support
  AND (counts.support * 1.0 / prefix.sp)             >= :min_confidence
  AND ((counts.support * 1.0 / prefix.sp) / (suffix.ss * 1.0 / total.tot)) >= :min_lift
GROUP BY counts.c1, counts.c2;
