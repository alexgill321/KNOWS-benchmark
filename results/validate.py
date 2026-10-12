"""Validate published result dictionaries without private archives or API access."""
from __future__ import annotations

from collections import Counter
from html.parser import HTMLParser
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
METRICS = ("success_rate", "asc", "average_checkpoint_fraction", "step_fraction")


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, label):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
            f"{label}: {actual!r} != {expected!r}")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_means(records, field):
    return {key: sum(r[field][key] for r in records) / len(records) for key in METRICS}


def validate_codex(model, config, expected):
    data = read(RESULTS / "codex" / f"{model}.json")
    require(data["schema_version"] == 1 and data["model"] == model, "Model identity differs")
    require(data["configuration_file"] == "configuration.json", "Missing configuration link")
    require(data["snapshot_sha256"] == config["snapshot_sha256"], "Snapshot differs")
    require(data["selected_summary_sha256"] == config["selected_summary_sha256"], "Summary differs")
    require(set(data["results"]) == expected, f"{model}: task coverage differs")
    for task, record in data["results"].items():
        require(record["status"] == "eligible", f"{task}: unexpected snapshot status")
        require(record["evaluation"]["status"] == "completed", f"{task}: missing completed report")
        require(record["evaluation"]["judge_model"] == config["judge_model"], f"{task}: judge differs")
        require(record["run"]["effort"] == config["efforts"][model], f"{task}: effort differs")
        require(record["run"]["max_actions"] == config["max_actions"], f"{task}: action limit differs")
        require(record["run"]["timeout_seconds"] == config["episode_timeout_seconds"], f"{task}: timeout differs")
        final = record["result"]["final_score"]
        checkpoints = record["result"]["checkpoints"]
        require(bool(checkpoints), f"{task}: no checkpoints")
        equal(final["result"], sum(c["result"] for c in checkpoints), f"{task}: earned points")
        equal(final["total"], sum(c["total"] for c in checkpoints), f"{task}: possible points")
        equal(record["reward"], final["result"] / final["total"], f"{task}: reward")
        steps = []
        for cp in checkpoints:
            require(cp["total"] > 0 and 0 <= cp["result"] <= cp["total"], f"{task}: invalid checkpoint score")
            summary = cp["step_summary"]
            require(summary["total_steps"] == len(summary["steps"]), f"{task}: step count differs")
            require(all(type(s["success"]) is bool for s in summary["steps"]), f"{task}: invalid step success flag")
            correct = sum(s["success"] for s in summary["steps"])
            require(summary["successful_steps"] == correct, f"{task}: successful step count differs")
            require(summary["failed_steps"] == len(summary["steps"]) - correct, f"{task}: failed step count differs")
            steps.extend(summary["steps"])
        require(bool(steps), f"{task}: no steps")
        full = sum(c["result"] == c["total"] for c in checkpoints)
        derived = {
            "success_rate": int(full == len(checkpoints)),
            "asc": full / len(checkpoints),
            "average_checkpoint_fraction": sum(c["result"] / c["total"] for c in checkpoints) / len(checkpoints),
            "step_fraction": sum(s["success"] for s in steps) / len(steps),
        }
        override = record["annotations"].get("metric_override")
        if override:
            require(override["task_id"] == task and override["model"] == model, f"{task}: override identity differs")
            require(override["metric"] == "success_rate", f"{task}: unsupported override")
            equal(override["reported_value"], derived["success_rate"], f"{task}: pre-override value")
        for metric in METRICS:
            equal(record["metrics_from_recorded_scores"][metric], derived[metric], f"{task}: {metric}")
            reported = override["value"] if override and override["metric"] == metric else derived[metric]
            equal(record["reported_metrics"][metric], reported, f"{task}: reported {metric}")
        for key in ("source_report_sha256", "source_episode_sha256"):
            require(re.fullmatch(r"[a-f0-9]{64}", record["provenance"][key]) is not None, f"{task}: invalid {key}")
    records = list(data["results"].values())
    summary = data["summary"]
    require(summary["expected"] == summary["eligible"] == len(records) == 110, "Task denominator differs")
    equal(summary["mean_reward"], sum(r["reward"] for r in records) / len(records), "Mean reward")
    for field in ("metrics_from_recorded_scores", "reported_metrics"):
        for key, value in metric_means(records, field).items():
            equal(summary[field][key], value, f"{model}: overall {field}.{key}")
    for artifact, expected_count in config["task_counts_by_artifact"].items():
        group = [r for r in records if r["artifact"] == artifact]
        aggregate = summary["reported_metrics_by_artifact"][artifact]
        require(aggregate["expected"] == aggregate["scored"] == len(group) == expected_count, "Artifact denominator differs")
        for key, value in metric_means(group, "reported_metrics").items():
            equal(aggregate[key], value, f"{model}: {artifact}.{key}")
    require(summary["selected_origins"] == dict(Counter(r["annotations"]["origin"] for r in records)), "Origin counts differ")
    require(summary["provisional_source_checks"] == sum(r["annotations"]["provisional_source_checks"] for r in records), "Provisional count differs")
    require(summary["partial_runs"] == sum(r["annotations"]["partial_run"] for r in records), "Partial count differs")
    require(summary["reports_with_execution_errors"] == sum(r["evaluation"]["has_execution_errors"] for r in records), "Execution-error count differs")
    require(summary["metric_overrides"] == sum("metric_override" in r["annotations"] for r in records), "Override count differs")
    print(f"{model}: 110 tasks; checkpoint totals, step counts, metrics, and annotations verified")


class TableRows(HTMLParser):
    def __init__(self):
        super().__init__()
        self.row = self.cell = None
        self.rows = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row = []
        if tag == "td" and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag == "td" and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        if tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def validate_comet():
    data = read(RESULTS / "comet/results.json")
    parser = TableRows()
    parser.feed((ROOT / data["summary_source"]["path"]).read_text())
    row = next(r for r in parser.rows if r and r[0] == "Perplexity Comet")
    require(len(row) == 9, "Comet source table differs")
    for artifact, offset in (("Docs", 1), ("Sheets", 3), ("Slides", 5), ("Overall", 7)):
        metrics = data["summary"]["by_artifact"][artifact]
        equal(metrics["success_rate_percent"], float(row[offset]), f"Comet {artifact} SR")
        equal(metrics["average_checkpoint_fraction_percent"], float(row[offset + 1]), f"Comet {artifact} ACF")
    require(data["example_count"] == len(data["results"]) == 3, "Comet example count differs")
    for task, record in data["results"].items():
        path = ROOT / record["source"]
        require(sha(path) == record["source_sha256"], f"{task}: Comet source changed")
        source = read(path)
        require(task == f"knows.{source['task']}.{source['instance'].split('_')[-1]}", "Comet task identity differs")
        agent = source["sides"]["agent"]
        require(record["result"] == {k: agent[k] for k in record["result"]}, f"{task}: Comet result changed")
        require(all((ROOT / page).is_file() for page in record["artifact_pages"]), f"{task}: missing artifact image")
    print("Comet: published aggregate scores and all three examples match existing website data")


def main():
    tasks = ROOT / "src/browsergym/knows/eval/tasks"
    expected = {f"knows.{p.parent.parent.name}.{p.parent.name.split('_')[-1]}"
                for p in tasks.glob("*/instance_*/task.md")}
    require(len(expected) == 110, "Repository task catalog differs")
    config = read(RESULTS / "codex/configuration.json")
    for model in config["models"]:
        validate_codex(model, config, expected)
    validate_comet()


if __name__ == "__main__":
    main()
