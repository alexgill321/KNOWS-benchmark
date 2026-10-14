from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from codex_benchmark.evaluation_corrections import evaluation_for_scoring
from codex_benchmark.image_credit_review import (
    ARCHIVE_NAME, KIND, MANIFEST_NAME, PROBE_CAVEAT, REASON, TARGETS,
    load_image_credit_review,
)
from codex_benchmark.runner import summarize


class ImageCreditReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.run = self.root / "smoke"
        self.archive = self.root / "recovery" / ARCHIVE_NAME
        self.manifest_path = self.root / MANIFEST_NAME
        self.fixtures = []
        self.write(self.run / "manifest.json", {"models": ["gpt-6-astra", "gpt-6-luna"], "tasks_per_model": 110})
        for (model, task), archived_name in TARGETS.items():
            directory = self.run / model / task
            archived = self.archive / archived_name
            agent = {"model": model, "effort": "max", "thread_id": "thread-" + model, "turn_id": "turn",
                     "status": "action_limit", "tool_calls": 150, "final_text": "done", "token_usage": {}}
            episode = {**agent, "task_id": task, "workspace_id": "workspace-" + model,
                       "judge_model": "gemini-3.8-flash", "phase": "finished"}
            attempt = str(uuid4())
            report = {"task_id": task, "workspace_id": episode["workspace_id"], "status": "completed",
                      "evaluation_attempt_id": attempt, "step_journal": {"attempt_id": attempt},
                      "judge_model": episode["judge_model"], "judge_errors": [], "evaluator_invoked": True,
                      "score_source": "official_evaluator", "reward": .5}
            for name, obj in (("episode.json", episode), ("agent-result.json", agent), ("evaluation.json", report)):
                self.write(directory / name, obj)
                self.write(archived / name, obj)
            self.write(directory / "evaluation-attempts" / (attempt + ".json"), report)
            self.write_lines(directory / "evaluation-steps.jsonl", [
                {"event": event, "attempt_id": attempt, "task_id": task, "workspace_id": episode["workspace_id"]}
                for event in ("evaluation_started", "evaluation_finished")])
            url = "https://images.example.test/exact-" + model + ".png?view=original"
            url_sha = hashlib.sha256(url.encode()).hexdigest()
            actions = archived / "browser/actions.jsonl"
            self.write_lines(actions, [{"action": 49, "tool": "browser_observe", "success": True,
                                       "timestamp": "2026-10-05T13:00:05+00:00"}])
            navs = archived / "browser/navigations.jsonl"
            self.write_lines(navs, [{"page_id": "page_4", "url": url, "timestamp": "2026-10-05T13:00:00+00:00"}])
            screenshot = archived / "browser/screenshots/00049_page_4.png"
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            screenshot.write_bytes(b"saved pixels; visual interpretation belongs to the reviewed proof")
            log = archived / "evaluate.log"
            log.write_text("  URL validation: Failed to download image from " + url + "\n")
            response = self.archive / (model + "-probe-response.bin")
            response.write_bytes(b"403 Forbidden")
            probe_path = self.archive / (model + "-probe.json")
            probe = {"kind": "single_unauthenticated_exact_submitted_image_probe", "task_id": task,
                     "url_sha256": url_sha, "status_code": 403, "cookies_or_credentials_used": False,
                     "browser_session_created": False, "model_or_judge_called": False,
                     "limitation": PROBE_CAVEAT, "body_sha256": self.sha(response), "body_bytes": response.stat().st_size}
            self.write(probe_path, probe)
            originals = {name: self.sha(directory / name) for name in ("episode.json", "evaluation.json", "agent-result.json")}
            entry = {"model": model, "task_id": task, "workspace_id": episode["workspace_id"],
                     "reason": REASON, "model_replayed": False, "original_sha256": originals,
                     "evaluation_attempt_id": attempt}
            proof_path = self.archive / (model + "-proof.json")
            independent_path = self.archive / (model + "-independent.json")
            self.write(independent_path, {"kind": "independent_new_report_review", "image_credit_review_cases": [{
                "model": model, "task_id": task, "attempt_id": attempt, "report_sha256": originals["evaluation.json"],
                "exact_url_sha256": url_sha, "recommendation": "hold_pending_retrieval_review_preserve_raw_grade",
                "screenshot": {"sha256": self.sha(screenshot), "visually_reviewed": True}}]})
            proof = {**entry, "kind": KIND, "reviewed": True, "exact_browser_url_loaded": True,
                     "evaluator_fetch_failed": True, "exact_url_sha256": url_sha,
                     "independent_report_review": self.descriptor(independent_path),
                     "evidence": {"browser_actions": self.descriptor(actions), "browser_navigations": self.descriptor(navs),
                                  "browser_screenshot": self.descriptor(screenshot), "evaluator_log": self.descriptor(log)},
                     "current_unauthenticated_probe": {**self.descriptor(probe_path), "response_path": str(response),
                         "response_sha256": self.sha(response), "distinct_from_original_evaluator_request": True,
                         "original_http_cause_proven": False}}
            self.write(proof_path, proof)
            entry.update(proof_path=str(proof_path), proof_sha256=self.sha(proof_path))
            self.fixtures.append({"directory": directory, "archived": archived, "path": directory / "episode.json",
                                  "report": report, "entry": entry, "proof": proof, "proof_path": proof_path,
                                  "probe_path": probe_path, "probe": probe})
        self.manifest = {"version": 1, "kind": KIND, "run_dir": str(self.run),
                         "targets": [fixture["entry"] for fixture in self.fixtures]}
        self.write(self.manifest_path, self.manifest)

    def write(self, path, obj):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj) + "\n")

    def write_lines(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def descriptor(self, path):
        return {"path": str(path), "sha256": self.sha(path)}

    def refresh_proof(self, fixture, proof):
        self.write(fixture["proof_path"], proof)
        fixture["entry"]["proof_sha256"] = self.sha(fixture["proof_path"])
        self.write(self.manifest_path, self.manifest)

    def assert_held(self, fixture=None):
        fixture = fixture or self.fixtures[0]
        result = evaluation_for_scoring(fixture["path"], fixture["report"])
        self.assertIsNotNone(result["error"])
        self.assertEqual(result["evaluation"], fixture["report"])
        self.assertIsNone(result["provenance"])

    def test_valid_review_holds_both_scores_without_exclusion_or_record_writes(self):
        before = {p: p.read_bytes() for f in self.fixtures for p in f["directory"].rglob("*") if p.is_file()}
        for fixture in self.fixtures:
            review = load_image_credit_review(fixture["path"])
            self.assertEqual(review["status"], "under_review")
            self.assertFalse(review["original_http_cause_proven"])
            self.assert_held(fixture)
        summary = summarize(self.run)
        for model in ("gpt-6-astra", "gpt-6-luna"):
            self.assertEqual(summary["models"][model]["scored"], 0)
            self.assertEqual(summary["models"][model]["infrastructure_errors"], 0)
            self.assertIsNone(summary["models"][model]["mean_reward"])
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_absent_authority_no_hold_and_malformed_authority_never_holds_other_tasks(self):
        self.manifest_path.unlink()
        for fixture in self.fixtures:
            self.assertIsNone(load_image_credit_review(fixture["path"]))
            self.assertIsNone(evaluation_for_scoring(fixture["path"], fixture["report"])["error"])
        self.manifest_path.write_text("malformed")
        for fixture in self.fixtures:
            self.assert_held(fixture)
        for model, task in (("gpt-6-luna", self.fixtures[0]["entry"]["task_id"]),
                            ("gpt-6-astra", self.fixtures[1]["entry"]["task_id"]),
                            ("gpt-6-astra", "knows.slides_17_removeimagesaddplaceholders.5")):
            path = self.run / model / task / "episode.json"
            self.write(path, {"model": model, "task_id": task})
            self.assertIsNone(load_image_credit_review(path))
            self.assertIsNone(evaluation_for_scoring(path, {"status": "completed", "reward": .8})["error"])

    def test_archive_candidate_validation_does_not_publish_or_hold_official_scores(self):
        self.manifest_path.unlink()
        candidate = self.archive / "candidate-authority.json"
        self.write(candidate, self.manifest)
        for fixture in self.fixtures:
            self.assertEqual(load_image_credit_review(fixture["path"], candidate)["status"], "under_review")
            self.assertIsNone(evaluation_for_scoring(fixture["path"], fixture["report"])["error"])
            for bad in (self.root / "outside.json", Path("relative.json")):
                with self.assertRaises(ValueError): load_image_credit_review(fixture["path"], bad)
        self.assertFalse(self.manifest_path.exists())

    def test_authority_exact_scope_and_reason_are_required(self):
        variants = [{**self.manifest, "version": True}, {**self.manifest, "kind": "other"},
                    {**self.manifest, "run_dir": str(self.root / "retry")},
                    {**self.manifest, "targets": self.manifest["targets"][:1]},
                    {**self.manifest, "targets": [self.manifest["targets"][0]] * 2}]
        for field, value in (("model", "other"), ("reason", "automatic_exclusion"), ("model_replayed", True)):
            item = deepcopy(self.manifest); item["targets"][0][field] = value; variants.append(item)
        for item in variants:
            with self.subTest(item=item):
                self.write(self.manifest_path, item)
                for fixture in self.fixtures:
                    with self.assertRaises((ValueError, TypeError)):
                        load_image_credit_review(fixture["path"])

    def test_tampering_any_bound_original_or_browser_probe_evidence_fails_closed(self):
        fixture = self.fixtures[0]
        paths = [*fixture["directory"].rglob("*.json"), fixture["directory"] / "evaluation-steps.jsonl",
                 fixture["proof_path"], fixture["probe_path"],
                 *[fixture["archived"] / name for name in ("episode.json", "evaluation.json", "agent-result.json")],
                 *[Path(item["path"]) for item in fixture["proof"]["evidence"].values()],
                 Path(fixture["proof"]["current_unauthenticated_probe"]["response_path"])]
        for path in paths:
            with self.subTest(path=path.name):
                old = path.read_bytes()
                path.write_bytes(old.splitlines()[0] + b"\n" if path.name == "evaluation-steps.jsonl" else old + b" ")
                self.assert_held(fixture)
                with self.assertRaises((ValueError, OSError)):
                    load_image_credit_review(fixture["path"])
                path.write_bytes(old)

    def test_wrong_proof_identity_and_missing_review_flags_rejected_even_with_new_hash(self):
        fixture = self.fixtures[0]
        for field, value in (("reviewed", False), ("kind", "other"), ("task_id", "other"),
                             ("workspace_id", "other"), ("evaluation_attempt_id", str(uuid4())),
                             ("original_sha256", {}), ("model_replayed", True),
                             ("exact_browser_url_loaded", False), ("evaluator_fetch_failed", False)):
            with self.subTest(field=field):
                self.refresh_proof(fixture, {**fixture["proof"], field: value})
                with self.assertRaises(ValueError): load_image_credit_review(fixture["path"])

    def test_probe_requires_actual_exact_url_403_no_credentials_and_fresh_cause_caveat(self):
        fixture = self.fixtures[0]
        for field, value in (("status_code", 200), ("cookies_or_credentials_used", True),
                             ("url_sha256", "0" * 64), ("task_id", "other"),
                             ("browser_session_created", True), ("model_or_judge_called", True),
                             ("limitation", "Original evaluator returned 403"), ("body_bytes", 999),
                             ("body_sha256", "0" * 64)):
            with self.subTest(field=field):
                self.write(fixture["probe_path"], {**fixture["probe"], field: value})
                proof = deepcopy(fixture["proof"])
                proof["current_unauthenticated_probe"].update(self.descriptor(fixture["probe_path"]))
                self.refresh_proof(fixture, proof)
                with self.assertRaises(ValueError): load_image_credit_review(fixture["path"])
        self.write(fixture["probe_path"], fixture["probe"])
        for field, value in (("distinct_from_original_evaluator_request", False), ("original_http_cause_proven", True)):
            proof = deepcopy(fixture["proof"]); proof["current_unauthenticated_probe"][field] = value
            self.refresh_proof(fixture, proof)
            with self.assertRaises(ValueError): load_image_credit_review(fixture["path"])

    def test_browser_proof_requires_matching_screenshot_action_page_url_and_failure_line(self):
        fixture = self.fixtures[0]
        evidence = fixture["proof"]["evidence"]
        changes = [("browser_actions", [{"action": 49, "success": False}]),
                   ("browser_actions", [{"action": 48, "success": True}]),
                   ("browser_navigations", [{"page_id": "page_5", "url": "other", "timestamp": "2026-10-05T13:00:00+00:00"}]),
                   ("browser_navigations", [{"page_id": "page_4", "url": "other", "timestamp": "2026-10-05T13:00:00+00:00"}])]
        for field, rows in changes:
            path = Path(evidence[field]["path"]); old = path.read_bytes()
            self.write_lines(path, rows)
            proof = deepcopy(fixture["proof"]); proof["evidence"][field] = self.descriptor(path)
            self.refresh_proof(fixture, proof)
            with self.assertRaises(ValueError): load_image_credit_review(fixture["path"])
            path.write_bytes(old)
        path = Path(evidence["evaluator_log"]["path"])
        for text in ("Failed to download image from another URL", "URL validation: Failed to download image from https://other.test/a.png"):
            path.write_text(text)
            proof = deepcopy(fixture["proof"]); proof["evidence"]["evaluator_log"] = self.descriptor(path)
            self.refresh_proof(fixture, proof)
            with self.assertRaises(ValueError): load_image_credit_review(fixture["path"])

    def test_archive_escape_and_cross_episode_screenshot_rejected(self):
        fixture = self.fixtures[0]
        for path in (self.fixtures[1]["proof_path"], self.root / "outside.json"):
            if not path.exists(): self.write(path, fixture["proof"])
            entry = fixture["entry"]
            entry.update(proof_path=str(path), proof_sha256=self.sha(path))
            self.write(self.manifest_path, self.manifest)
            with self.assertRaises(ValueError): load_image_credit_review(fixture["path"])
        fixture["entry"].update(proof_path=str(fixture["proof_path"]), proof_sha256=self.sha(fixture["proof_path"]))
        proof = deepcopy(fixture["proof"])
        proof["evidence"]["browser_screenshot"] = self.fixtures[1]["proof"]["evidence"]["browser_screenshot"]
        self.refresh_proof(fixture, proof)
        with self.assertRaises(ValueError): load_image_credit_review(fixture["path"])

    def test_updated_hashes_cannot_hide_wrong_terminal_identity_or_evaluator_attempt(self):
        fixture = self.fixtures[0]
        for name, field, value in (("episode.json", "workspace_id", "other"),
                                   ("agent-result.json", "turn_id", "different"),
                                   ("evaluation.json", "evaluation_attempt_id", str(uuid4())),
                                   ("evaluation.json", "evaluator_invoked", False)):
            path = fixture["directory"] / name; old = path.read_bytes()
            obj = json.loads(old); obj[field] = value; self.write(path, obj)
            old_digest = fixture["entry"]["original_sha256"][name]
            fixture["entry"]["original_sha256"][name] = self.sha(path)
            self.write(self.manifest_path, self.manifest)
            with self.assertRaises(ValueError): load_image_credit_review(fixture["path"])
            path.write_bytes(old); fixture["entry"]["original_sha256"][name] = old_digest

    def test_independent_review_hash_and_exact_report_and_image_scope_required(self):
        fixture = self.fixtures[0]
        path = Path(fixture["proof"]["independent_report_review"]["path"])
        original = path.read_bytes()
        path.write_bytes(original + b" ")
        with self.assertRaises(ValueError): load_image_credit_review(fixture["path"])
        for field, value in (("attempt_id", str(uuid4())), ("report_sha256", "0" * 64),
                             ("exact_url_sha256", "0" * 64), ("recommendation", "award_credit"),
                             ("model", "other"), ("screenshot", {"visually_reviewed": False})):
            independent = json.loads(original); independent["image_credit_review_cases"][0][field] = value
            self.write(path, independent)
            proof = deepcopy(fixture["proof"]); proof["independent_report_review"] = self.descriptor(path)
            self.refresh_proof(fixture, proof)
            with self.assertRaises(ValueError): load_image_credit_review(fixture["path"])

    def test_existing_monitor_classifies_valid_hold_as_under_review_not_excluded(self):
        script = Path(__file__).resolve().parents[1] / "codex-results/monitor-checks/20261004T111240Z/audit_campaigns.py"
        spec = importlib.util.spec_from_file_location("image_credit_monitor_test", script)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        result = module.audit(self.run, {}, 0, {})
        self.assertEqual(result["under_review_count"], 2)
        self.assertEqual(result["eligible"], 0)
        self.assertTrue(all(row["reward"] is None for row in result["grades"]))
        self.assertEqual(len(result["evaluation_correction_errors"]), 2)


if __name__ == "__main__":
    unittest.main()
