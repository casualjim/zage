WITH trigrams AS (
  SELECT LAG(command,2) OVER (ORDER BY rowid) AS c1,
        LAG(command,1) OVER (ORDER BY rowid) AS c2,
        command AS c3
  FROM shell_history
), counts AS (
  SELECT c1, c2, c3, COUNT(*) AS support
  FROM trigrams
  WHERE c1 IS NOT NULL
  GROUP BY c1, c2, c3
), prefix AS (
  SELECT c1, c2, COUNT(*) AS sp
  FROM trigrams
  WHERE c1 IS NOT NULL
  GROUP BY c1, c2
), suffix AS (
  SELECT c3, COUNT(*) AS ss
  FROM trigrams
  GROUP BY c3
), total AS (
  SELECT COUNT(*) AS tot FROM shell_history
)
INSERT OR REPLACE INTO sequence_scores(sequence, support, confidence, lift, context)
SELECT
  json_array(counts.c1, counts.c2, counts.c3),
  counts.support,
  counts.support * 1.0 / prefix.sp,
  (counts.support * 1.0 / prefix.sp) / (suffix.ss * 1.0 / total.tot),
  json_object(
    'working_directory', MAX(h1.working_directory),
    'hostname', MAX(h1.hostname),
    'username', MAX(h1.username),
    'exit_status', json_array(MAX(h1.exit_status), MAX(h2.exit_status), MAX(h3.exit_status)),
    'session_id', MAX(h1.session_id),
    'time_info', json_object(
      'start_time', MIN(h1.start_unix_timestamp),
      'end_time', MAX(h3.end_unix_timestamp)
    )
  )
FROM counts
JOIN prefix ON counts.c1 = prefix.c1 AND counts.c2 = prefix.c2
JOIN suffix ON counts.c3 = suffix.c3
CROSS JOIN total
JOIN shell_history h1 ON counts.c1 = h1.command
JOIN shell_history h2 ON counts.c2 = h2.command
JOIN shell_history h3 ON counts.c3 = h3.command
WHERE
  counts.support    >= :min_support
  AND (counts.support * 1.0 / prefix.sp)             >= :min_confidence
  AND ((counts.support * 1.0 / prefix.sp) / (suffix.ss * 1.0 / total.tot)) >= :min_lift
GROUP BY counts.c1, counts.c2, counts.c3;
