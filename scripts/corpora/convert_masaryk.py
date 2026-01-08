#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
from datetime import datetime


def parse_timestamp(value: str | None) -> int | None:
    if not value:
        return None
    try:
        # Try ISO 8601 with timezone
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def load_commands(path: Path) -> list[tuple[int, str]]:
    commands: list[tuple[int, str]] = []
    counter = 1
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cmd = obj.get("cmd") or obj.get("command")
            if not cmd or not isinstance(cmd, str):
                continue
            ts = parse_timestamp(obj.get("timestamp_str") or obj.get("timestamp"))
            if ts is None:
                ts = counter
            commands.append((ts, cmd))
            counter += 1
    return commands


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: convert_masaryk.py <extracted_dir> <derived_dir>")
        return 1

    extracted = Path(sys.argv[1])
    derived = Path(sys.argv[2])
    derived.mkdir(parents=True, exist_ok=True)

    log_files = list(extracted.rglob("*useractions.json"))
    if not log_files:
        print("No useractions.json files found.")
        return 0

    for log in log_files:
        commands = load_commands(log)
        if not commands:
            continue
        out_name = log.stem.replace(".useractions", "")
        out_path = derived / f"masaryk_{out_name}.bash_history"
        commands.sort(key=lambda item: item[0])
        with out_path.open("w", encoding="utf-8") as handle:
            for ts, cmd in commands:
                handle.write(f"#{ts}\n")
                handle.write(f"{cmd}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
