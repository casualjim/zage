#!/usr/bin/env bash
# Bash integration for Zage using bash-preexec (https://github.com/rcaloras/bash-preexec)

# Ensure Zage binary is available
if ! command -v zage &> /dev/null; then
  echo "Zage: 'zage' command not found. Please ensure it's in your PATH." >&2
  return 1
fi

# Ensure bash-preexec loaded
if ! declare -F preexec_invoke_exec &>/dev/null; then
  echo "Zage: bash-preexec plugin not loaded. Please source bash-preexec.sh first." >&2
  return 1
fi

# Expose the current shell session id for completion scoring
export ZAGE_SESSION_ID=$$

# Capture aliases from the running shell for completion scoring
# Prefer a file when ZAGE_ALIAS_FILE is set (keeps the environment small);
# otherwise fall back to the ZAGE_ALIASES environment variable.
if [[ -n "$ZAGE_ALIAS_FILE" ]]; then
    alias -p > "$ZAGE_ALIAS_FILE"
elif [[ -z "$ZAGE_ALIASES" ]]; then
    ZAGE_ALIASES="$(alias -p)"
    export ZAGE_ALIASES
fi

# Optional debug log file: set ZAGE_BASH_DEBUG to a filepath to enable
: "${ZAGE_BASH_DEBUG:=""}"

# Variables to store command context
_zage_cmd_start_time=""
_zage_cmd_string=""
_zage_cmd_pwd=""

_zage_epoch_seconds() {
  if [[ -n "${EPOCHSECONDS-}" ]]; then
    printf "%s" "$EPOCHSECONDS"
  else
    date +%s
  fi
}

# Function to run before each command (preexec)
_zage_preexec() {
  _zage_cmd_start_time="$(_zage_epoch_seconds)"
  _zage_cmd_string="$1"
  _zage_cmd_pwd="$PWD"

  if [[ -n "$ZAGE_BASH_DEBUG" ]]; then
    echo "[zage-hook preexec] start=$_zage_cmd_start_time cmd=$_zage_cmd_string pwd=$_zage_cmd_pwd" >> "$ZAGE_BASH_DEBUG"
  fi
}

# Function to run before prompt (precmd)
_zage_precmd() {
  local exit_status=$?
  local end_time
  end_time="$(_zage_epoch_seconds)"

  # Skip empty or internal commands
  if [[ -z "$_zage_cmd_string" || "$_zage_cmd_string" =~ ^zage[[:space:]]+(record|feedback) ]]; then
    _zage_cmd_string=""
    _zage_cmd_start_time=""
    _zage_cmd_pwd=""
    return
  fi

  # Check if we have a start time (handles potential initial prompt)
  if [[ -z "$_zage_cmd_start_time" ]]; then
    _zage_cmd_string=""
    return
  fi

  if [[ -n "$ZAGE_BASH_DEBUG" ]]; then
    echo "[zage-hook precmd] exit=$exit_status start=$_zage_cmd_start_time end=$end_time cmd=$_zage_cmd_string pwd=$_zage_cmd_pwd" >> "$ZAGE_BASH_DEBUG"
    zage record \
      --command "$_zage_cmd_string" \
      --working-directory "$_zage_cmd_pwd" \
      --exit-status "$exit_status" \
      --start-timestamp "$_zage_cmd_start_time" \
      --end-timestamp "$end_time" \
      --session-id $$ >> "$ZAGE_BASH_DEBUG" 2>&1 &
  else
    # Invoke zage record silently
    zage record \
      --command "$_zage_cmd_string" \
      --working-directory "$_zage_cmd_pwd" \
      --exit-status "$exit_status" \
      --start-timestamp "$_zage_cmd_start_time" \
      --end-timestamp "$end_time" \
      --session-id $$ > /dev/null 2>&1 &
  fi
  _ZAGE_RECORD_PID=$!

  if command -v disown >/dev/null 2>&1; then
    disown
  fi

  # Clear for next command
  _zage_cmd_string=""
  _zage_cmd_start_time=""
  _zage_cmd_pwd=""
}

# Register hooks
preexec_functions+=( _zage_preexec )
precmd_functions+=( _zage_precmd )
