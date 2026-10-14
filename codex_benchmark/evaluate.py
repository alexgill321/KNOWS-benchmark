"""Grade a saved episode without loading evaluators in the agent process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .knows import evaluate_task, find_task


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path, help="Saved episode JSON")
    parser.add_argument("--output", type=Path, help="Default: evaluation.json beside episode")
    args = parser.parse_args(argv)
    episode = json.loads(args.episode.read_text(encoding="utf-8"))
    task = find_task(episode.get("benchmark_root", Path.cwd()), episode["task_id"])
    output = args.output or args.episode.parent / "evaluation.json"
    report = evaluate_task(task, episode["workspace_id"], episode["browsing_history"], output,
                           run_targets_path=episode.get("run_targets_path"),
                           client_doc_id=episode.get("client_doc_id"),
                           judge_model=episode.get("judge_model"),
                           final_text=episode.get("final_text"))
    print(json.dumps({"status": report["status"], "reward": report["reward"], "output": str(output)}))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
