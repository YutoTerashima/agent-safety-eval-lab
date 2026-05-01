from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import write_experiment_outputs
from .replay import replay_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate and replay agent safety traces.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run-demo", help="Generate built-in mock evaluation reports.")
    replay = sub.add_parser("replay", help="Replay JSON traces from a directory.")
    replay.add_argument("trace_dir", type=Path)
    replay.add_argument("--out", type=Path, default=Path("reports/replay_results.json"))
    args = parser.parse_args()

    if args.command == "run-demo":
        for path in write_experiment_outputs():
            print(path)
    elif args.command == "replay":
        results = replay_directory(args.trace_dir)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(args.out)


if __name__ == "__main__":
    main()
