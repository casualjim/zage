#!/usr/bin/env bash

set -euo pipefail

proj_dir=$(git rev-parse --show-toplevel)
cd "$proj_dir" || exit 1

raw_dir="$proj_dir/data/pretrain/raw/uci_unix"
derived_dir="$proj_dir/data/pretrain/derived"
tar_path="$raw_dir/UNIX_user_data.tar.gz"
extract_dir="$raw_dir/extracted"

mkdir -p "$raw_dir" "$derived_dir" "$extract_dir"

if [ ! -f "$tar_path" ]; then
  echo "Downloading UCI UNIX User Data..."
  curl -L "https://kdd.ics.uci.edu/databases/UNIX_user_data/UNIX_user_data.tar.gz" -o "$tar_path"
fi

if [ -z "$(ls -A "$extract_dir" 2>/dev/null)" ]; then
  echo "Extracting UCI UNIX User Data..."
  tar -xzf "$tar_path" -C "$extract_dir"
fi

echo "Converting UCI UNIX User Data to bash history..."
python3 "$proj_dir/scripts/corpora/convert_uci_unix.py" "$extract_dir" "$derived_dir"
