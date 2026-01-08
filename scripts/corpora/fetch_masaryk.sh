#!/usr/bin/env bash

set -euo pipefail

proj_dir=$(git rev-parse --show-toplevel)
cd "$proj_dir" || exit 1

raw_dir="$proj_dir/data/pretrain/raw/masaryk"
derived_dir="$proj_dir/data/pretrain/derived"
zip_path="$raw_dir/data.zip"
extract_dir="$raw_dir/extracted"

mkdir -p "$raw_dir" "$derived_dir" "$extract_dir"

if [ ! -f "$zip_path" ]; then
  echo "Downloading Masaryk dataset..."
  curl -L "https://zenodo.org/records/8136017/files/data.zip?download=1" -o "$zip_path"
fi

if [ ! -d "$extract_dir/CTF-Logs" ]; then
  echo "Extracting Masaryk dataset..."
  unzip -q -o "$zip_path" -d "$extract_dir"
fi

echo "Converting Masaryk logs to bash history..."
python3 "$proj_dir/scripts/corpora/convert_masaryk.py" "$extract_dir" "$derived_dir"
