import copy
import asyncio
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from codex_benchmark.evaluator_dependency import (
    MARKER, apply_dependency_review, is_reviewed_evaluator_dependency_diagnostic,
)


class EvaluatorDependencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "luna" / "knows.sheets_55_Movie_Recommendation.1"
        self.directory.mkdir(parents=True)
        self.path = self.directory / "episode.json"
        self.result = {"model": "luna", "effort": "max", "thread_id": "thread1", "turn_id": "turn1",
                       "status": "completed", "tool_calls": 53, "final_text": "Done.",
                       "token_usage": {"total": 20}, "error": None}
        self.episode = {**self.result, "task_id": self.directory.name, "workspace_id": "sheet1",
                        "workspace_url": "https://docs.google.com/spreadsheets/d/sheet1/edit",
                        "prompt_sha256": "prompt1", "seed": 0, "repetition": 1,
                        "browsing_history": ["https://example.test"], "phase": "evaluating"}
        self.report = {"status": "completed", "task_id": self.directory.name, "workspace_id": "sheet1",
                       "reward": 0.2, "judge_errors": [], "evaluation_attempt_id": "attempt1",
                       "step_journal": {"attempt_id": "attempt1"},
                       "detailed_report": {"checkpoints": [{"name": "Accuracy", "steps": [
                           {"name": "Genre Verification", "success": False, "category": "execution_error",
                            "details": "0/1 movies match. Could not retrieve IMDb data: Titanic"}]}]}}
        self.log = ("[IMDb] Suggest API matches for 'Titanic' (1997) -> ['tt0120338']\n"
                    "[IMDb] No valid JSON-LD Movie data for 'Titanic' (tt0120338)\n")
        self.save_fixture()

    def write(self, name, value):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def ending(self, status="completed"):
        return json.dumps({"status": status, "reward": 0.2,
                           "output": str(self.directory / "evaluation.json")}) + "\n"

    def save_fixture(self):
        self.write("episode.json", self.episode)
        self.write("agent-result.json", self.result)
        self.save_report()
        (self.directory / "evaluate.log").write_text(self.log + self.ending())

    def save_report(self):
        self.write("evaluation.json", self.report)
        attempt = self.report["evaluation_attempt_id"]
        self.write(f"evaluation-attempts/{attempt}.json", self.report)
        events = [{"event": event, "attempt_id": attempt, "task_id": self.report["task_id"],
                   "workspace_id": self.report["workspace_id"]}
                  for event in ("evaluation_started", "evaluation_finished")]
        (self.directory / "evaluation-steps.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    def apply(self, offset=0):
        return apply_dependency_review(self.episode, self.path, offset)

    def persisted_review(self):
        self.assertTrue(self.apply())
        self.episode.update(phase="finished", phase_started_at="later", finished_at="later")
        self.write("episode.json", self.episode)
        self.assertTrue(is_reviewed_evaluator_dependency_diagnostic(self.path))

    def test_archives_and_excludes_without_modifying_model_episode_or_grade(self):
        before = {name: (self.directory / name).read_bytes() for name in
                  ("episode.json", "agent-result.json", "evaluation.json", "evaluate.log")}
        self.assertTrue(self.apply())
        for name, payload in before.items():
            self.assertEqual((self.directory / name).read_bytes(), payload, name)
        self.assertEqual(self.episode["status"], "completed")
        self.assertEqual(self.episode["benchmark_validity"]["status"], "excluded_infrastructure")
        proof_path = self.directory / self.episode[MARKER]["proof_path"]
        proof = json.loads(proof_path.read_text())
        self.assertEqual(proof["failures"][0]["corroborated_titles"],
                         [{"title": "Titanic", "candidate_ids": ["tt0120338"]}])
        for name, payload in before.items():
            self.assertEqual((proof_path.parent / name).read_bytes(), payload)
        self.assertEqual(proof_path.stat().st_mode & 0o777, 0o600)
        self.assertFalse(self.apply())  # Also idempotent before the caller saves.
        self.write("episode.json", self.episode)
        self.assertTrue(is_reviewed_evaluator_dependency_diagnostic(self.path))
        self.assertFalse(self.apply())

    def test_archive_reused_if_caller_has_not_saved_new_metadata(self):
        unchanged = copy.deepcopy(self.episode)
        self.assertTrue(self.apply())
        first_marker = copy.deepcopy(self.episode[MARKER])
        self.assertTrue(apply_dependency_review(unchanged, self.path))
        self.assertEqual(unchanged[MARKER], first_marker)
        self.assertEqual(len(list((self.directory / "evaluator-dependency-reviews").iterdir())), 1)

    def test_colorhexa_exact_failure_only_and_action_limit_preserved(self):
        new_directory = self.directory.with_name("knows.sheets_45_Personal_WeddingPlanner_weddingcolorpallette.5")
        self.directory.rename(new_directory)
        self.directory, self.path = new_directory, new_directory / "episode.json"
        self.episode.update(task_id=new_directory.name, status="action_limit", tool_calls=150)
        self.result.update(status="action_limit", tool_calls=150)
        self.report["task_id"] = new_directory.name
        step = self.report["detailed_report"]["checkpoints"][0]["steps"][0]
        step.update(name="Hex Matches colorhexa", details="colorhexa unreachable - could not verify any hex value.")
        self.log = "Sheet content fetched successfully.\n"
        self.save_fixture()
        self.persisted_review()
        self.assertEqual(self.episode["status"], "action_limit")
        self.assertEqual(json.loads((self.directory / "evaluation.json").read_text())["reward"], 0.2)

    def test_unrelated_errors_and_uncorroborated_imdb_do_not_exclude(self):
        original = self.log
        cases = ["", original.replace("tt0120338']", "tt1111111']"),
                 original.replace("for 'Titanic' (tt", "for 'Other' (tt"),
                 original + "[IMDb] Got data for 'Titanic' (tt0120338) (genre=['Drama'])\n"]
        for log in cases:
            with self.subTest(log=log):
                (self.directory / "evaluate.log").write_text(log + self.ending())
                self.assertFalse(self.apply())
        (self.directory / "evaluate.log").write_text(original + self.ending())
        for detail in ("No table found", "Missing ingredient column", "No valid Craigslist URLs found in spreadsheet",
                       "No images found to verify.", "Could not retrieve awards data: Titanic"):
            self.report["detailed_report"]["checkpoints"][0]["steps"][0]["details"] = detail
            self.save_report()
            self.assertFalse(self.apply(), detail)
        self.assertNotIn(MARKER, self.episode)
        self.assertFalse((self.directory / "evaluator-dependency-reviews").exists())

    def test_old_attempt_log_cannot_corroborate_current_failure(self):
        prior = self.log + self.ending()
        (self.directory / "evaluate.log").write_text(prior + "Current attempt: no candidate matched.\n" + self.ending())
        self.assertFalse(self.apply(len(prior.encode())))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.apply(0)
        current = self.log + self.ending()
        (self.directory / "evaluate.log").write_text(prior + current)
        self.assertTrue(self.apply(len(prior.encode())))
        self.write("episode.json", self.episode)
        self.assertTrue(is_reviewed_evaluator_dependency_diagnostic(self.path))
        proof = json.loads((self.directory / self.episode[MARKER]["proof_path"]).read_text())
        self.assertEqual(proof["log_start"], len(prior.encode()))

    def test_mismatched_model_identity_report_or_attempt_fails_closed(self):
        for file, field, value in (("agent-result.json", "turn_id", "other"),
                                   ("agent-result.json", "status", "inProgress"),
                                   ("evaluation.json", "workspace_id", "other"),
                                   ("evaluation.json", "evaluation_attempt_id", "missing"),
                                   ("evaluation.json", "judge_errors", ["API unavailable"])):
            with self.subTest(file=file, field=field):
                original = (self.directory / file).read_bytes()
                changed = json.loads(original)
                changed[field] = value
                self.write(file, changed)
                with self.assertRaises(ValueError):
                    self.apply()
                (self.directory / file).write_bytes(original)
                self.assertNotIn(MARKER, self.episode)
        (self.directory / "evaluation-steps.jsonl").write_text("")
        with self.assertRaises(ValueError):
            self.apply()
        self.assertFalse((self.directory / "evaluator-dependency-reviews").exists())

    def test_does_not_replace_existing_infrastructure_exclusion(self):
        self.episode["benchmark_validity"] = {"status": "excluded_infrastructure", "reason": "Browser sign-out"}
        unchanged = copy.deepcopy(self.episode)
        self.assertFalse(self.apply())
        self.assertEqual(self.episode, unchanged)
        self.assertFalse((self.directory / "evaluator-dependency-reviews").exists())

    def test_unrelated_or_successful_report_does_not_require_log(self):
        (self.directory / "evaluate.log").unlink()
        for task in ("knows.docs_1_formal_letter.1", "knows.sheets_2_personal_recipe_foodcomposition.1"):
            self.assertFalse(apply_dependency_review({"task_id": task}, self.path))
        self.report["status"] = "evaluation_error"
        self.save_report()
        self.assertFalse(self.apply())
        self.report["status"] = "completed"
        self.report["detailed_report"]["checkpoints"][0]["steps"][0]["details"] = "No data in table"
        self.save_report()
        self.assertFalse(self.apply())

    def test_grade_saved_episode_applies_review_then_finalizes_without_replaying(self):
        from codex_benchmark.runner import grade_saved_episode
        report_before = (self.directory / "evaluation.json").read_bytes()
        result_before = (self.directory / "agent-result.json").read_bytes()
        log_before = (self.directory / "evaluate.log").read_bytes()

        async def evaluate_only(*args, **kwargs):
            with (self.directory / "evaluate.log").open("a") as handle:
                handle.write(self.log + self.ending())
            return 0

        with patch("codex_benchmark.runner.subprocess_logged", side_effect=evaluate_only) as mocked:
            asyncio.run(grade_saved_episode({"evaluation_timeout_seconds": 900}, self.episode, self.path))
            self.assertEqual(mocked.call_count, 1)
            self.assertIn("codex_benchmark.evaluate", mocked.call_args.args[0])
        self.assertEqual(self.episode["phase"], "finished")
        self.assertEqual(self.episode["status"], "completed")
        self.assertTrue(is_reviewed_evaluator_dependency_diagnostic(self.path))
        self.assertEqual((self.directory / "evaluation.json").read_bytes(), report_before)
        self.assertEqual((self.directory / "agent-result.json").read_bytes(), result_before)
        proof = json.loads((self.directory / self.episode[MARKER]["proof_path"]).read_text())
        self.assertEqual(proof["log_start"], len(log_before))

    def prepare_colorhexa_retry(self, *, completed=True, complete_journal=True):
        new_directory = self.directory.with_name("knows.sheets_45_Personal_WeddingPlanner_weddingcolorpallette.5")
        self.directory.rename(new_directory)
        self.directory, self.path = new_directory, new_directory / "episode.json"
        self.episode.update(task_id=new_directory.name, evaluation_error="Prior evaluator failed",
                            evaluation_deferred=True, phase="evaluation_error")
        self.report.update(task_id=new_directory.name, evaluation_attempt_id="prior_failed_attempt",
                           step_journal={"attempt_id": "prior_failed_attempt"},
                           status="evaluation_error", reward=None)
        step = self.report["detailed_report"]["checkpoints"][0]["steps"][0]
        step.update(name="Hex Matches colorhexa", details="colorhexa unreachable - could not verify any hex value.")
        self.log = "Sheet content fetched successfully.\n"
        self.save_fixture()
        (self.directory / "evaluate.log").write_text("Prior evaluator failed.\n" + self.ending("evaluation_error"))
        current = copy.deepcopy(self.report)
        current.update(status="completed" if completed else "evaluation_error",
                       reward=0.2 if completed else None, evaluation_attempt_id="new_attempt",
                       step_journal={"attempt_id": "new_attempt", "completed_steps": 1})
        before = {name: (self.directory / name).read_bytes() for name in
                  ("agent-result.json", "evaluation-attempts/prior_failed_attempt.json",
                   "evaluation-steps.jsonl", "evaluate.log")}

        async def evaluate_only(*args, **kwargs):
            self.write("evaluation.json", current)
            self.write("evaluation-attempts/new_attempt.json", current)
            identity = {"attempt_id": "new_attempt", "task_id": self.episode["task_id"],
                        "workspace_id": self.episode["workspace_id"]}
            events = [{**identity, "event": "evaluation_started"},
                      {**identity, "event": "step_completed", "checkpoint": "Accuracy", "step": step}]
            if complete_journal:
                events.append({**identity, "event": "evaluation_finished"})
            with (self.directory / "evaluation-steps.jsonl").open("a") as handle:
                handle.write("".join(json.dumps(event) + "\n" for event in events))
            with (self.directory / "evaluate.log").open("a") as handle:
                handle.write(self.log + self.ending(current["status"]))
            return 0 if completed else 1

        return evaluate_only, current, before

    def test_successful_retry_persists_fresh_bookkeeping_before_real_colorhexa_review(self):
        from codex_benchmark.runner import grade_saved_episode, summarize
        from codex_benchmark.parallel_queue import classify_existing
        evaluate_only, report, before = self.prepare_colorhexa_retry()
        with patch("codex_benchmark.runner.subprocess_logged", side_effect=evaluate_only) as mocked:
            asyncio.run(grade_saved_episode({"evaluation_timeout_seconds": 900}, self.episode, self.path))
            self.assertEqual(mocked.call_count, 1)
            self.assertIn("codex_benchmark.evaluate", mocked.call_args.args[0])
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved, self.episode)
        self.assertNotIn("evaluation_error", saved)
        self.assertNotIn("evaluation_deferred", saved)
        self.assertEqual(saved["phase"], "finished")
        for field, value in self.result.items():
            self.assertEqual(saved[field], value, field)
        self.assertTrue(is_reviewed_evaluator_dependency_diagnostic(self.path))
        proof_path = self.directory / saved[MARKER]["proof_path"]
        proof = json.loads(proof_path.read_text())
        self.assertEqual(proof["evaluation_attempt_id"], "new_attempt")
        self.assertEqual(proof["log_start"], len(before["evaluate.log"]))
        self.assertEqual(proof["failures"][0]["dependency"], "colorhexa")
        self.assertNotIn("evaluation_error", json.loads((proof_path.parent / "episode.json").read_text()))
        for name in ("agent-result.json", "evaluation-attempts/prior_failed_attempt.json"):
            self.assertEqual((self.directory / name).read_bytes(), before[name])
        for name in ("evaluate.log", "evaluation-steps.jsonl"):
            self.assertTrue((self.directory / name).read_bytes().startswith(before[name]))
        self.assertEqual(json.loads((self.directory / "evaluation.json").read_text()), report)
        self.assertEqual(classify_existing(self.directory), "done")
        (self.root / "manifest.json").write_text(json.dumps({"models": ["luna"], "tasks_per_model": 110}))
        summary = summarize(self.root)
        self.assertEqual(summary["models"]["luna"]["scored"], 0)
        self.assertIsNone(summary["episodes"][0]["reward"])
        self.assertEqual(summary["episodes"][0]["diagnostic_reward"], 0.2)

    def test_dependency_finalization_failure_preserves_model_and_report_but_withholds_score(self):
        from codex_benchmark.runner import grade_saved_episode, summarize
        from codex_benchmark.parallel_queue import classify_existing
        evaluate_only, report, before = self.prepare_colorhexa_retry(complete_journal=False)
        with patch("codex_benchmark.runner.subprocess_logged", side_effect=evaluate_only) as mocked:
            asyncio.run(grade_saved_episode({"evaluation_timeout_seconds": 900}, self.episode, self.path))
            self.assertEqual(mocked.call_count, 1)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved, self.episode)
        self.assertIn("Evaluator finalization failed: ValueError:", saved["evaluation_error"])
        self.assertIn("journal is incomplete", saved["evaluation_error"])
        self.assertEqual(saved["phase"], "evaluation_error")
        self.assertNotIn(MARKER, saved)
        self.assertNotIn("benchmark_validity", saved)
        for field, value in self.result.items():
            self.assertEqual(saved[field], value, field)
        for name in ("agent-result.json", "evaluation-attempts/prior_failed_attempt.json"):
            self.assertEqual((self.directory / name).read_bytes(), before[name])
        self.assertEqual(json.loads((self.directory / "evaluation.json").read_text()), report)
        self.assertEqual((self.directory / "evaluation.json").read_bytes(),
                         (self.directory / "evaluation-attempts/new_attempt.json").read_bytes())
        self.assertFalse(is_reviewed_evaluator_dependency_diagnostic(self.path))
        self.assertNotEqual(classify_existing(self.directory), "done")
        (self.root / "manifest.json").write_text(json.dumps({"models": ["luna"], "tasks_per_model": 110}))
        summary = summarize(self.root)
        self.assertEqual(summary["models"]["luna"]["scored"], 0)
        self.assertIsNone(summary["episodes"][0]["reward"])

    def test_retry_that_still_fails_cannot_become_dependency_diagnostic(self):
        from codex_benchmark.runner import grade_saved_episode
        evaluate_only, report, before = self.prepare_colorhexa_retry(completed=False)
        with patch("codex_benchmark.runner.subprocess_logged", side_effect=evaluate_only) as mocked:
            asyncio.run(grade_saved_episode({"evaluation_timeout_seconds": 900}, self.episode, self.path))
            self.assertEqual(mocked.call_count, 1)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["phase"], "evaluation_error")
        self.assertIn("Evaluator exited with error", saved["evaluation_error"])
        self.assertNotIn(MARKER, saved)
        self.assertNotIn("benchmark_validity", saved)
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(json.loads((self.directory / "evaluation.json").read_text()), report)
        self.assertEqual((self.directory / "agent-result.json").read_bytes(), before["agent-result.json"])
        self.assertFalse((self.directory / "evaluator-dependency-reviews").exists())

    def test_successful_retry_keeps_existing_unrelated_exclusion(self):
        from codex_benchmark.runner import grade_saved_episode
        evaluate_only, _, _ = self.prepare_colorhexa_retry()
        validity = {"status": "excluded_infrastructure", "reason": "Browser sign-out"}
        self.episode["benchmark_validity"] = copy.deepcopy(validity)
        self.write("episode.json", self.episode)
        with patch("codex_benchmark.runner.subprocess_logged", side_effect=evaluate_only):
            asyncio.run(grade_saved_episode({"evaluation_timeout_seconds": 900}, self.episode, self.path))
        self.assertEqual(json.loads(self.path.read_text())["benchmark_validity"], validity)
        self.assertNotIn(MARKER, self.episode)
        for field, value in self.result.items():
            self.assertEqual(self.episode[field], value, field)

    def test_tampered_live_archived_proof_and_identity_are_rejected(self):
        self.persisted_review()
        proof_path = self.directory / self.episode[MARKER]["proof_path"]
        for path in (proof_path, proof_path.parent / "episode.json", proof_path.parent / "evaluation.json",
                     self.directory / "evaluation.json", self.directory / "agent-result.json",
                     self.directory / "evaluate.log", self.directory / "evaluation-attempts/attempt1.json"):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                self.assertFalse(is_reviewed_evaluator_dependency_diagnostic(self.path))
                path.write_bytes(original)
                self.assertTrue(is_reviewed_evaluator_dependency_diagnostic(self.path))
        for field, value in (("model", "astra"), ("tool_calls", 99), ("final_text", "different"),
                             ("workspace_id", "other"), ("status", "infrastructure_error")):
            self.write("episode.json", {**self.episode, field: value})
            self.assertFalse(is_reviewed_evaluator_dependency_diagnostic(self.path), field)
        self.write("episode.json", self.episode)

    def test_queue_and_summary_accept_only_verified_diagnostic(self):
        from codex_benchmark.parallel_queue import classify_existing
        from codex_benchmark.runner import summarize
        self.persisted_review()
        self.assertEqual(classify_existing(self.directory), "done")
        (self.root / "manifest.json").write_text(json.dumps({"models": ["luna"], "tasks_per_model": 110}))
        summary = summarize(self.root)
        self.assertEqual(summary["models"]["luna"]["evaluated"], 1)
        self.assertEqual(summary["models"]["luna"]["scored"], 0)
        self.assertIsNone(summary["episodes"][0]["reward"])
        self.assertEqual(summary["episodes"][0]["diagnostic_reward"], 0.2)
        proof = self.directory / self.episode[MARKER]["proof_path"]
        proof.write_bytes(proof.read_bytes() + b" ")
        self.assertEqual(classify_existing(self.directory), "blocked")


if __name__ == "__main__":
    unittest.main()
