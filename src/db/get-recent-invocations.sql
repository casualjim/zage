SELECT id, command, expanded_command, shellname, working_directory, workspace_json, hostname, username, exit_status, start_unix_timestamp, end_unix_timestamp, session_id
FROM shell_history
WHERE (?1 IS NULL OR session_id = ?1)
ORDER BY COALESCE(end_unix_timestamp, start_unix_timestamp, 0) DESC, id DESC
LIMIT ?2
