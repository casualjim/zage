SELECT id, command, expanded_command, shellname, working_directory, workspace_json, hostname, username, exit_status, start_unix_timestamp, end_unix_timestamp, session_id
FROM shell_history
ORDER BY start_unix_timestamp DESC
LIMIT ?
