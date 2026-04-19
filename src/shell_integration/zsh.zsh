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
# Clear suggestions during bracketed paste to prevent accidental acceptance
# We need to handle this after zsh-autosuggestions loads, so use a deferred hook
_zage_setup_bracketed_paste_clear() {
  emulate -L zsh
  add-zsh-hook -d precmd _zage_setup_bracketed_paste_clear
  if (( ${+ZSH_AUTOSUGGEST_CLEAR_WIDGETS} )); then
    if (( ${ZSH_AUTOSUGGEST_CLEAR_WIDGETS[(I)bracketed-paste]} == 0 )); then
      ZSH_AUTOSUGGEST_CLEAR_WIDGETS+=(bracketed-paste)
      if [[ -n "$ZAGE_ZSH_DEBUG" ]]; then
        print -r -- "[zage-hook] added bracketed-paste to clear widgets" >> "$ZAGE_ZSH_DEBUG"
      fi
    fi
  fi
}
add-zsh-hook precmd _zage_setup_bracketed_paste_clear

# State for suggestion feedback.
_ZAGE_LAST_SUGGESTION=""
_ZAGE_LAST_SUGGESTION_AT=""
_ZAGE_LAST_SUGGESTION_ID=""
_ZAGE_LAST_SUGGESTION_PWD=""

# Provide a zsh-autosuggestions strategy backed by zage
_zsh_autosuggest_strategy_zage() {
    emulate -L zsh
    local prefix="$BUFFER"
    local output

    if [[ "$prefix" == :* ]]; then
      suggestion=""
      return
    fi

    if [[ -z "$prefix" ]]; then
      output="$(zage suggest --autosuggest --count 5 2>/dev/null)"
    else
      output="$(zage suggest --autosuggest --count 5 --current-line "$prefix" 2>/dev/null)"
    fi

    if [[ -n "$output" && "$output" == "$prefix"* && "$output" != "$prefix" ]]; then
      suggestion="$output"
    else
      suggestion=""
    fi

    if [[ -n "$ZAGE_ZSH_DEBUG" ]]; then
      print -r -- "[zage-autosuggest] prefix=$prefix suggestion=$suggestion" >> "$ZAGE_ZSH_DEBUG"
    fi

    if [[ -n "$suggestion" ]]; then
      _ZAGE_LAST_SUGGESTION="$suggestion"
      _ZAGE_LAST_SUGGESTION_AT="$EPOCHSECONDS"
      _ZAGE_LAST_SUGGESTION_ID="${ZAGE_SESSION_ID:-$$}-${EPOCHREALTIME}"
      _ZAGE_LAST_SUGGESTION_PWD="$PWD"
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

# Trigger autosuggestion fetch on a fresh prompt (empty buffer).
_zage_zle_line_init() {
    emulate -L zsh
    if (( ${+_ZAGE_RECORD_PID} )) && kill -0 "$_ZAGE_RECORD_PID" 2>/dev/null; then
      local start=$EPOCHREALTIME
      while kill -0 "$_ZAGE_RECORD_PID" 2>/dev/null; do
        if (( EPOCHREALTIME - start > 0.2 )); then
          break
        fi
        sleep 0.01
      done
    fi
    if [[ -n "$ZAGE_ZSH_DEBUG" ]]; then
      local has_fetch="0"
      if (( ${+functions[_zsh_autosuggest_fetch]} )); then
        has_fetch="1"
      fi
      print -r -- "[zage-hook line-init] fired has_fetch=$has_fetch buffer_len=${#BUFFER}" >> "$ZAGE_ZSH_DEBUG"
    fi
    if [[ -z "$BUFFER" ]] && (( ${+functions[_zsh_autosuggest_strategy_zage]} )); then
      _zsh_autosuggest_strategy_zage
      if [[ -n "$suggestion" ]]; then
        POSTDISPLAY="$suggestion"
        if (( ${+functions[_zsh_autosuggest_highlight_reset]} )); then
          _zsh_autosuggest_highlight_reset
        fi
        if (( ${+functions[_zsh_autosuggest_highlight_apply]} )); then
          _zsh_autosuggest_highlight_apply
        fi
        zle -R
        return
      fi
    fi
    if (( ${+functions[_zsh_autosuggest_fetch]} )); then
      _zsh_autosuggest_fetch
      if (( ${+functions[_zsh_autosuggest_display]} )); then
        _zsh_autosuggest_display
      fi
    fi
}

_zage_install_line_init_hook() {
  emulate -L zsh
  if [[ "$ZAGE_AUTOSUGGEST_DISABLE" == "1" ]]; then
    return
  fi
  if (( ${+_ZAGE_LINE_INIT_INSTALLED} )); then
    return
  fi
  zmodload zsh/zle 2>/dev/null || return
  autoload -Uz add-zle-hook-widget 2>/dev/null
  if (( ${+functions[add-zle-hook-widget]} )); then
    add-zle-hook-widget line-init _zage_zle_line_init
    _ZAGE_LINE_INIT_INSTALLED=1
    if [[ -n "$ZAGE_ZSH_DEBUG" ]]; then
      print -r -- "[zage-hook line-init] installed via add-zle-hook-widget" >> "$ZAGE_ZSH_DEBUG"
    fi
  else
    zle -N zle-line-init _zage_zle_line_init
    _ZAGE_LINE_INIT_INSTALLED=1
    if [[ -n "$ZAGE_ZSH_DEBUG" ]]; then
      print -r -- "[zage-hook line-init] installed via zle -N" >> "$ZAGE_ZSH_DEBUG"
    fi
  fi
}

_zage_install_line_init_hook

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
    _zage_cmd_start_time=$EPOCHSECONDS # Unix epoch seconds
    _zage_cmd_string=$1
    _zage_cmd_pwd=$PWD

    # Emit best-effort feedback if we recently showed a suggestion.
    if [[ "$ZAGE_FEEDBACK_DISABLE" != "1" && -n "$_ZAGE_LAST_SUGGESTION_ID" && -n "$_ZAGE_LAST_SUGGESTION_AT" ]]; then
      if [[ -n "$_zage_cmd_string" && "$_zage_cmd_string" != zage\ * ]]; then
        if [[ "$_zage_cmd_string" == "$_ZAGE_LAST_SUGGESTION" ]]; then
          local shown_at="$_ZAGE_LAST_SUGGESTION_AT"
          local now="$_zage_cmd_start_time"

          local cwd="$_ZAGE_LAST_SUGGESTION_PWD"
          if [[ -z "$cwd" ]]; then
            cwd="$_zage_cmd_pwd"
          fi

          if [[ -n "$ZAGE_ZSH_DEBUG" ]]; then
            zage feedback \
              --shown-id "$_ZAGE_LAST_SUGGESTION_ID" \
              --shown-at "$shown_at" \
              --working-directory "$cwd" \
              --suggestion "$_ZAGE_LAST_SUGGESTION" \
              --accepted-command "$_zage_cmd_string" \
              --accepted-at "$now" \
              --outcome "accepted" >> "$ZAGE_ZSH_DEBUG" 2>&1 &
          else
            zage feedback \
              --shown-id "$_ZAGE_LAST_SUGGESTION_ID" \
              --shown-at "$shown_at" \
              --working-directory "$cwd" \
              --suggestion "$_ZAGE_LAST_SUGGESTION" \
              --accepted-command "$_zage_cmd_string" \
              --accepted-at "$now" \
              --outcome "accepted" > /dev/null 2>&1 &!
          fi
        fi
      fi
      _ZAGE_LAST_SUGGESTION=""
      _ZAGE_LAST_SUGGESTION_AT=""
      _ZAGE_LAST_SUGGESTION_ID=""
      _ZAGE_LAST_SUGGESTION_PWD=""
    fi

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
    local end_time=$EPOCHSECONDS

    # Ensure we don't record empty commands or the recording command itself
    if [[ -z "$_zage_cmd_string"
      || "$_zage_cmd_string" == zage\ record*
      || "$_zage_cmd_string" == zage\ feedback*
    ]]; then
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
      _ZAGE_RECORD_PID=$!
    else
      zage record \
        --command "$_zage_cmd_string" \
        --working-directory "$_zage_cmd_pwd" \
        --exit-status "$exit_status" \
        --start-timestamp "$_zage_cmd_start_time" \
        --end-timestamp "$end_time" \
        --session-id "$$" > /dev/null 2>&1 &!  # Use &! to disown and suppress job messages
      _ZAGE_RECORD_PID=$!
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
add-zsh-hook precmd _zage_install_line_init_hook

# Optional: Initial message
# echo "Zage Zsh integration enabled."
