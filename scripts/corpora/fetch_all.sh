#!/usr/bin/env bash

set -euo pipefail

proj_dir=$(git rev-parse --show-toplevel)
cd "$proj_dir" || exit 1

scripts_dir="$proj_dir/scripts/corpora"

if [ -x "$scripts_dir/fetch_masaryk.sh" ]; then
  "$scripts_dir/fetch_masaryk.sh"
fi

echo "Corpora fetch complete."
