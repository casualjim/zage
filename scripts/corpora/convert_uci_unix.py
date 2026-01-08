#!/usr/bin/env python3

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: convert_uci_unix.py <extracted_dir> <derived_dir>")
        return 1

    extracted = Path(sys.argv[1])
    derived = Path(sys.argv[2])
    derived.mkdir(parents=True, exist_ok=True)

    data_files = [p for p in extracted.rglob("*") if p.is_file()]
    if not data_files:
        print("No UCI UNIX user data files found.")
        return 0

    for data_file in data_files:
        if data_file.name.startswith("."):
            continue
        commands: list[str] = []
        with data_file.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                cmd = line.strip()
                if cmd:
                    commands.append(cmd)
        if not commands:
            continue
        out_path = derived / f"uci_{data_file.name}.bash_history"
        with out_path.open("w", encoding="utf-8") as handle:
            ts = 1
            for cmd in commands:
                handle.write(f"#{ts}\n")
                handle.write(f"{cmd}\n")
                ts += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
