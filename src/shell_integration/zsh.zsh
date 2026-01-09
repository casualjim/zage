# Zsh integration for Zage
#
# This script hooks into Zsh's preexec and precmd functions to record
# command history directly into the Zage database.

# Ensure Zage binary is available
if ! command -v zage &> /dev/null; then
    echo "Zage: 'zage' command not found. Please ensure it's in your PATH." >&2
    return 1
fi

# Expose the current shell session id for completion scoring
export ZAGE_SESSION_ID=$$

# Capture aliases from the running shell for completion scoring
if [[ -z "$ZAGE_ALIASES" ]]; then
  ZAGE_ALIASES="$(alias -L)"
  export ZAGE_ALIASES
fi

# Use zsh completion format for zage suggestions
export ZAGE_COMPLETION_FORMAT="zsh"

# Enable zsh-autosuggestions backend unless explicitly disabled
: ${ZAGE_AUTOSUGGEST_DISABLE:="0"}
# Force zage to be the only autosuggest strategy when set to 1
: ${ZAGE_AUTOSUGGEST_ONLY:="0"}

# Provide a zsh-autosuggestions strategy backed by zage
_zsh_autosuggest_strategy_zage() {
    emulate -L zsh
    local prefix="$BUFFER"
    local output

    if [[ -z "$prefix" ]]; then
      output="$(zage suggest --autosuggest --count 5 2>/dev/null | head -n 1)"
    else
      output="$(zage suggest --autosuggest --count 5 --current-line "$prefix" 2>/dev/null | head -n 1)"
    fi

    if [[ -n "$output" && "$output" == "$prefix"* && "$output" != "$prefix" ]]; then
      suggestion="$output"
    else
      suggestion=""
    fi

    if [[ -n "$ZAGE_ZSH_DEBUG" ]]; then
      print -r -- "[zage-autosuggest] prefix=$prefix suggestion=$suggestion" >> "$ZAGE_ZSH_DEBUG"
    fi
}

if [[ "$ZAGE_AUTOSUGGEST_DISABLE" != "1" ]]; then
  if [[ "$ZAGE_AUTOSUGGEST_ONLY" == "1" ]]; then
    ZSH_AUTOSUGGEST_STRATEGY=(zage)
  else
    if [[ -z "${ZSH_AUTOSUGGEST_STRATEGY+x}" ]]; then
      ZSH_AUTOSUGGEST_STRATEGY=(zage)
    elif (( ${ZSH_AUTOSUGGEST_STRATEGY[(I)zage]} == 0 )); then
      ZSH_AUTOSUGGEST_STRATEGY=(zage $ZSH_AUTOSUGGEST_STRATEGY)
    fi
  fi
fi

# Optional debug log file: set ZAGE_ZSH_DEBUG to a filepath to enable
: ${ZAGE_ZSH_DEBUG:=""}

# Variables to store command context between preexec and precmd
_zage_cmd_start_time=""
_zage_cmd_string=""
_zage_cmd_pwd=""

# Function to run before command execution (preexec)
_zage_preexec() {
    # Store command details
    # $1 is the command string
    _zage_cmd_start_time=$(date +%s) # Capture start time (Unix epoch seconds)
    _zage_cmd_string=$1
    _zage_cmd_pwd=$PWD

    # Debug preexec
    if [[ -n "$ZAGE_ZSH_DEBUG" ]]; then
      print -r -- "[zage-hook preexec] start=$_zage_cmd_start_time cmd=$_zage_cmd_string pwd=$_zage_cmd_pwd" >> "$ZAGE_ZSH_DEBUG"
    fi
}

# Function to run before prompt (precmd)
_zage_precmd() {
    # Placeholder - Logic to capture end time, exit status, and call zage record goes here
    # Requires access to _zage_cmd_start_time, _zage_cmd_string, _zage_cmd_pwd and $?
    local exit_status=$?
    local end_time=$(date +%s)

    # Ensure we don't record empty commands or the recording command itself
    if [[ -z "$_zage_cmd_string" || "$_zage_cmd_string" =~ ^zage\s+record ]]; then
        _zage_cmd_string="" # Clear command string to avoid re-recording
        return
    fi

    # Check if we have a start time (handles potential initial prompt)
    if [[ -z "$_zage_cmd_start_time" ]]; then
      _zage_cmd_string=""
      return
    fi

    # Debug precmd
    if [[ -n "$ZAGE_ZSH_DEBUG" ]]; then
      print -r -- "[zage-hook precmd] exit=$exit_status start=$_zage_cmd_start_time end=$end_time cmd=$_zage_cmd_string pwd=$_zage_cmd_pwd" >> "$ZAGE_ZSH_DEBUG"
    fi

    # Invoke recorder, logging to debug if enabled
    if [[ -n "$ZAGE_ZSH_DEBUG" ]]; then
      zage record \
        --command "$_zage_cmd_string" \
        --working-directory "$_zage_cmd_pwd" \
        --exit-status "$exit_status" \
        --start-timestamp "$_zage_cmd_start_time" \
        --end-timestamp "$end_time" \
        --session-id "$$" >> "$ZAGE_ZSH_DEBUG" 2>&1 &
    else
      zage record \
        --command "$_zage_cmd_string" \
        --working-directory "$_zage_cmd_pwd" \
        --exit-status "$exit_status" \
        --start-timestamp "$_zage_cmd_start_time" \
        --end-timestamp "$end_time" \
        --session-id "$$" > /dev/null 2>&1 &!  # Use &! to disown and suppress job messages
    fi

    # Clear variables for the next command
    _zage_cmd_string=""
    _zage_cmd_start_time=""
    _zage_cmd_pwd=""
}

# Add hook functions to Zsh if not already present
autoload -Uz add-zsh-hook
add-zsh-hook preexec _zage_preexec
add-zsh-hook precmd _zage_precmd

# Optional: Initial message
# echo "Zage Zsh integration enabled."
