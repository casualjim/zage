#!/usr/bin/env bash

set -euo pipefail

proj_dir=$(git rev-parse --show-toplevel)
cd "$proj_dir" || exit 1

raw_dir="$proj_dir/data/pretrain/raw/sea"
derived_dir="$proj_dir/data/pretrain/derived"
zip_path="$raw_dir/masquerade-data.zip"
extract_dir="$raw_dir/extracted"

mkdir -p "$raw_dir" "$derived_dir" "$extract_dir"

if [ ! -f "$zip_path" ]; then
  echo "Downloading SEA masquerade dataset..."
  curl -L "https://schonlau.net/masquerade/masquerade-data.zip" -o "$zip_path"
fi

if [ -z "$(ls -A "$extract_dir" 2>/dev/null)" ]; then
  echo "Extracting SEA masquerade dataset..."
  unzip -q -o "$zip_path" -d "$extract_dir"
fi

echo "Converting SEA masquerade data to bash history..."
python3 "$proj_dir/scripts/corpora/convert_sea.py" "$extract_dir" "$derived_dir"
