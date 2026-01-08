#!/usr/bin/env bash
# Bash integration for Zage using bash-preexec (https://github.com/rcaloras/bash-preexec)

# Ensure bash-preexec loaded
if ! declare -F preexec_invoke_exec &>/dev/null; then
    echo "Zage: bash-preexec plugin not loaded. Please source bash-preexec.sh first." >&2
    return 1
fi

# Expose the current shell session id for completion scoring
export ZAGE_SESSION_ID=$$

# Capture aliases from the running shell for completion scoring
if [[ -z "$ZAGE_ALIASES" ]]; then
    ZAGE_ALIASES="$(alias -p)"
    export ZAGE_ALIASES
fi

# Variables to store command context
_zage_cmd_start_time=""
_zage_cmd_string=""
_zage_cmd_pwd=""

# Function to run before each command (preexec)
_zage_preexec() {
    local start_time
    start_time=$(date +%s)
    _zage_cmd_start_time="$start_time"
    _zage_cmd_string="$1"
    _zage_cmd_pwd="$PWD"
}

# Function to run before prompt (precmd)
_zage_precmd() {
    local exit_status=$?
    local end_time
    end_time=$(date +%s)

    # Skip empty or record commands
    if [[ -z "$_zage_cmd_string" || "$_zage_cmd_string" =~ ^zage[[:space:]]+record ]]; then
        _zage_cmd_string=""
        return
    fi

    # Invoke zage record silently
    zage record \
        --command "$_zage_cmd_string" \
        --working-directory "$_zage_cmd_pwd" \
        --exit-status "$exit_status" \
        --start-timestamp "$_zage_cmd_start_time" \
        --end-timestamp "$end_time" \
        --session-id $$ > /dev/null 2>&1 &
    if command -v disown >/dev/null 2>&1; then
        disown
    fi

    # Clear for next command
    _zage_cmd_string=""
}

# Register hooks
preexec_functions+=( _zage_preexec )
precmd_functions+=( _zage_precmd )
