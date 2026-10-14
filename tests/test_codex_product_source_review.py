from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from codex_benchmark import product_source_review as m


def encoded(value):
    return (json.dumps(value, indent=2) + "\n").encode()


class ProductSourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "smoke"
        self.directory = self.run / m.TARGET[0] / m.TARGET[1]
        self.archive = self.root / "recovery" / m.ARCHIVE_NAME
        self.original = self.archive / "original-episode"
        self.manifest = self.root / m.MANIFEST_NAME
        self.directory.mkdir(parents=True)
        self.original.mkdir(parents=True)
        self.episode_path = self.directory / "episode.json"
        self.agent = {"model": m.TARGET[0], "effort": "max", "thread_id": "thread-1", "turn_id": "turn-1",
                      "status": "completed", "tool_calls": 150, "final_text": "Done", "token_usage": {}, "error": None}
        self.episode = {**self.agent, "task_id": m.TARGET[1], "workspace_id": "workspace-1",
                        "judge_model": "judge", "phase": "finished", "verified": True}
        self.events = []
        self.report = {"task_id": m.TARGET[1], "workspace_id": "workspace-1", "judge_model": "judge",
                       "status": "completed", "reward": 56 / 57, "score_source": "official_evaluator",
                       "evaluator_invoked": True, "evaluation_attempt_id": m.ATTEMPT,
                       "detailed_report": {"checkpoints": [], "final_score": {"total": 57, "result": 56}},
                       "result": {"checkpoints": [], "final_score": {"total": 57, "result": 56}},
                       "step_journal": {"attempt_id": m.ATTEMPT, "completed_steps": 57, "checkpoint_count": 7, "checkpoints": []}}
        self.event("evaluation_started")
        for index, (name, count) in enumerate(m.CONTRACT, 1):
            self.event("checkpoint_registered", checkpoint_index=index,
                       checkpoint={"name": name, "total": count, "result": 0, "steps": []})
            steps = [{"name": f"Criterion {i}", "step_id": i, "success": True, "details": "Matched",
                      "score": 1, "max_score": 1, "execution_time": 0.1, "category": "deterministic"}
                     for i in range(1, count + 1)]
            if index == 4:
                steps[-1].update(name="Device 3 - Content From Sources", success=False,
                                 details="No source content found", score=0, category="execution_error")
            points = sum(s["score"] for s in steps)
            for step in steps:
                self.event("step_completed", checkpoint_index=index, checkpoint_name=name, step=deepcopy(step))
            cp = {"name": name, "total": count, "result": points, "steps": deepcopy(steps), "execution_time": 0.2}
            self.event("checkpoint_snapshot", checkpoint_index=index, checkpoint=cp)
            self.report["step_journal"]["checkpoints"].append({"checkpoint_index": index, **deepcopy(cp)})
            self.report["detailed_report"]["checkpoints"].append({"name": name, "score": f"{points}/{count}", "steps": deepcopy(steps)})
            self.report["result"]["checkpoints"].append({"name": name, "total": count, "result": points,
                "step_summary": {"steps": deepcopy(steps), "total_steps": count,
                                 "successful_steps": points, "failed_steps": count - points}})
        self.event("evaluation_finished", completed_steps=57, checkpoint_count=7)
        self.actions = [{"action": 26, "tool": "browser_navigate", "success": True,
                         "arguments": {"url": m.BROWSER_URL}, "timestamp": "2026-10-05T20:05:36+00:00"},
                        {"action": 103, "tool": "browser_type", "success": True,
                         "arguments": {"text": "Source:\n" + m.SUBMITTED_URL}, "timestamp": "2026-10-05T20:13:46+00:00"}]
        self.navigations = [{"page_id": "page_4", "url": m.NAVIGATION_URL, "timestamp": "2026-10-05T20:05:23+00:00"}]
        self.entry = {"model": m.TARGET[0], "task_id": m.TARGET[1], "workspace_id": "workspace-1",
                      "reason": m.REASON, "model_replayed": False, "evaluation_attempt_id": m.ATTEMPT}
        self.proof = {**self.entry, "kind": m.KIND, "reviewed": True, "exact_browser_source_loaded": True,
                      "evaluator_fetch_failed": True, "http_status_or_challenge_cause_proven": False, "evidence": {}}
        self.proof_path = self.archive / "proof.json"
        self.write_originals()
        self.write_evidence("browser_actions", "browser/actions.jsonl", self.jsonlines(self.actions))
        self.write_evidence("browser_navigations", "browser/navigations.jsonl", self.jsonlines(self.navigations))
        self.write_evidence("browser_screenshot", "browser/screenshots/00026_page_4.png", b"saved screenshot bytes")
        self.write_evidence("evaluator_log", "evaluate.log", ("Downloading web content for Device 3...\n" + m.ERROR_LINE +
                            "\nDevice 3: No source content found.\n").encode())
        self.write_evidence("evaluation_journal", "evaluation-steps.jsonl", self.jsonlines(self.events))
        self.write_evidence("evaluation_attempt", f"evaluation-attempts/{m.ATTEMPT}.json", encoded(self.report))
        self.write_reviews()
        self.seal()

    def event(self, name, **fields):
        self.events.append({"event": name, "attempt_id": m.ATTEMPT, "sequence": len(self.events) + 1,
                            "task_id": m.TARGET[1], "workspace_id": "workspace-1", **fields})

    @staticmethod
    def jsonlines(records):
        return b"".join(json.dumps(r).encode() + b"\n" for r in records)

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write_originals(self):
        hashes = {}
        for name, value in (("episode.json", self.episode), ("agent-result.json", self.agent), ("evaluation.json", self.report)):
            data = encoded(value)
            self.write(self.directory / name, data)
            self.write(self.original / name, data)
            hashes[name] = m._sha(data)
        self.entry["original_sha256"] = hashes
        self.proof["original_sha256"] = hashes

    def write_evidence(self, key, relative, data):
        self.write(self.original / relative, data)
        self.write(self.directory / relative, data)
        self.proof["evidence"][key] = {"path": str(self.original / relative), "sha256": m._sha(data)}

    def write_reviews(self):
        for role in ("root", "independent"):
            review = {"kind": f"{role}_product_source_retrieval_review", "model": m.TARGET[0], "task_id": m.TARGET[1],
                      "evaluation_attempt_id": m.ATTEMPT, "report_sha256": self.entry["original_sha256"]["evaluation.json"],
                      "original_sha256": self.entry["original_sha256"], "decision": "hold_pending_source_retrieval_review",
                      "scored_reward": None, "raw_reward_preserved": self.report["reward"], "model_terminal": "completed",
                      "tool_calls": 150, "source_url_submitted": m.SUBMITTED_URL, "source_url_evaluator": m.EVALUATOR_URL,
                      "source_url_browser": m.BROWSER_URL, "browser_action": 26, "submitted_action": 103,
                      "evaluator_error": m.ERROR_LINE, "full_keyed_journal_steps": 57,
                      "infrastructure_exclusion": False, "all_keyed_steps_match": True,
                      "http_status_or_challenge_cause_proven": False, "new_source_probe": False,
                      "browser_session_created": False, "model_replayed": False, "evaluator_called": False,
                      "unrelated_criteria_unchanged": True,
                      "affected_step": {"checkpoint_index": 4, "checkpoint_name": "Device Slides",
                                        **self.report["detailed_report"]["checkpoints"][3]["steps"][23]},
                      "screenshot": {**self.proof["evidence"]["browser_screenshot"], "visually_reviewed": True}}
            path = self.archive / f"{role}-review.json"
            self.write(path, encoded(review))
            self.proof[f"{role}_review"] = {"path": str(path), "sha256": m._sha(path.read_bytes())}

    def seal(self):
        self.write(self.proof_path, encoded(self.proof))
        self.entry.update(proof_path=str(self.proof_path), proof_sha256=m._sha(self.proof_path.read_bytes()))
        self.authority = {"version": 1, "kind": m.KIND, "run_dir": str(self.run), "targets": [self.entry]}
        self.write(self.manifest, encoded(self.authority))

    def load(self):
        return m.load_product_source_review(self.episode_path)

    def test_valid_hold_is_read_only_and_not_exclusion(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        review = self.load()
        self.assertEqual(review["status"], "under_review")
        self.assertIsNone(review["scored_reward"])
        self.assertFalse(review["infrastructure_exclusion"])
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_shared_scoring_holds_valid_and_invalid_authority_and_releases_absence(self):
        from codex_benchmark.evaluation_corrections import evaluation_for_scoring
        before = (self.directory / "evaluation.json").read_bytes()
        for malformed in (False, True):
            if malformed:
                self.manifest.write_text("not-json")
            with self.subTest(malformed=malformed):
                result = evaluation_for_scoring(self.episode_path, self.report)
                self.assertTrue(result["error"])
                self.assertEqual(result["evaluation"], self.report)
        self.manifest.unlink()
        result = evaluation_for_scoring(self.episode_path, self.report)
        self.assertIsNone(result["error"])
        self.assertEqual(result["evaluation"], self.report)
        self.assertEqual((self.directory / "evaluation.json").read_bytes(), before)

    def test_absent_optional_authority_and_unaffected_other_targets(self):
        self.manifest.unlink()
        self.assertIsNone(self.load())
        self.manifest.write_text("not-json")
        for model, task in (("gpt-6-luna", m.TARGET[1]), (m.TARGET[0], m.TARGET[1][:-1] + "1")):
            self.assertIsNone(m.load_product_source_review(self.run / model / task / "episode.json"))
        with self.assertRaises(ValueError): self.load()

    def test_candidate_validation_never_publishes(self):
        candidate = self.archive / "candidate.json"
        self.manifest.rename(candidate)
        self.assertEqual(m.load_product_source_review(self.episode_path, candidate)["status"], "under_review")
        self.assertFalse(self.manifest.exists())
        with self.assertRaises(ValueError): m.load_product_source_review(self.episode_path, self.root / "elsewhere.json")

    def test_wrong_scope_attempt_and_boolean_authority_rejected(self):
        for field, bad in (("model", "gpt-6-luna"), ("task_id", m.TARGET[1][:-1] + "1"),
                           ("evaluation_attempt_id", "00000000-0000-0000-0000-000000000000"),
                           ("reason", "other"), ("model_replayed", 0)):
            with self.subTest(field=field):
                saved = self.entry[field]; self.entry[field] = bad; self.seal()
                with self.assertRaises(ValueError): self.load()
                self.entry[field] = saved
        self.seal(); self.authority["version"] = True; self.manifest.write_bytes(encoded(self.authority))
        with self.assertRaises(ValueError): self.load()

    def test_live_or_archived_original_and_evidence_tampering_rejected(self):
        paths = [self.directory / name for name in m.ORIGINAL_NAMES] + [self.original / name for name in m.ORIGINAL_NAMES]
        paths += [Path(item["path"]) for item in self.proof["evidence"].values()]
        paths += [self.directory / "evaluation-steps.jsonl", self.directory / f"evaluation-attempts/{m.ATTEMPT}.json"]
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes(); path.write_bytes(original + b" ")
                with self.assertRaises(ValueError): self.load()
                path.write_bytes(original)

    def test_resigned_agent_identity_mismatch_rejected(self):
        self.agent["turn_id"] = "another-turn"
        self.write_originals(); self.write_reviews(); self.seal()
        with self.assertRaises(ValueError): self.load()

    def test_successful_actions_exact_urls_and_submission_required(self):
        original = deepcopy(self.actions)
        mutations = [(0, "success", False), (0, "tool", "browser_type"),
                     (0, "arguments", {"url": m.BROWSER_URL + "?different"}),
                     (1, "arguments", {"text": "https://" + m.SUBMITTED_URL}),
                     (1, "arguments", {"text": m.SUBMITTED_URL + "wrong"}), (1, "success", False)]
        for index, key, bad in mutations:
            with self.subTest(index=index, key=key, bad=bad):
                self.actions = deepcopy(original); self.actions[index][key] = bad
                self.write_evidence("browser_actions", "browser/actions.jsonl", self.jsonlines(self.actions)); self.seal()
                with self.assertRaises(ValueError): self.load()
        self.actions = original + [deepcopy(original[0])]
        self.write_evidence("browser_actions", "browser/actions.jsonl", self.jsonlines(self.actions)); self.seal()
        with self.assertRaises(ValueError): self.load()

    def test_navigation_must_be_exact_saved_product_page_before_screenshot(self):
        for field, bad in (("url", m.NAVIGATION_URL + "?different"), ("url", m.BROWSER_URL),
                           ("page_id", "page_5"), ("timestamp", "2026-10-05T20:15:00+00:00")):
            with self.subTest(field=field, bad=bad):
                record = deepcopy(self.navigations); record[0][field] = bad
                self.write_evidence("browser_navigations", "browser/navigations.jsonl", self.jsonlines(record)); self.seal()
                with self.assertRaises(ValueError): self.load()

    def test_exact_error_and_device_three_log_context_required(self):
        path = self.original / "evaluate.log"; original = path.read_bytes()
        for bad in (original.replace(b"Page.content", b"HTTP 404"), original.replace(b"Device 3", b"Device 2"),
                    original.replace(m.EVALUATOR_URL.encode(), m.BROWSER_URL.encode()), original + m.ERROR_LINE.encode()):
            with self.subTest(bad=bad):
                self.write_evidence("evaluator_log", "evaluate.log", bad); self.seal()
                with self.assertRaises(ValueError): self.load()

    def test_signed_journal_missing_duplicate_wrong_key_and_step_rejected(self):
        variants = []
        events = deepcopy(self.events); events.pop(2); variants.append(events)
        events = deepcopy(self.events); events[2]["step"]["step_id"] = 2; variants.append(events)
        events = deepcopy(self.events); events[2]["checkpoint_name"] = "other"; variants.append(events)
        events = deepcopy(self.events); events[2]["step"]["details"] = "changed"; variants.append(events)
        events = deepcopy(self.events); events[-1]["completed_steps"] = 56; variants.append(events)
        events = deepcopy(self.events); events[-1]["event"] = "evaluation_error"; variants.append(events)
        events = deepcopy(self.events); events[2]["workspace_id"] = "wrong"; variants.append(events)
        for events in variants:
            with self.subTest(events=events[-1]):
                self.write_evidence("evaluation_journal", "evaluation-steps.jsonl", self.jsonlines(events)); self.seal()
                with self.assertRaises(ValueError): self.load()

    def test_both_independent_and_root_reviews_must_match_saved_identity(self):
        for role in ("root", "independent"):
            path = Path(self.proof[f"{role}_review"]["path"]); original = json.loads(path.read_text())
            for field, bad in (("evaluation_attempt_id", "other"), ("report_sha256", "0" * 64),
                               ("decision", "exclude"), ("scored_reward", 0.5), ("new_source_probe", True),
                               ("model_replayed", 0), ("source_url_browser", m.EVALUATOR_URL)):
                with self.subTest(role=role, field=field):
                    review = deepcopy(original); review[field] = bad; path.write_bytes(encoded(review))
                    self.proof[f"{role}_review"]["sha256"] = m._sha(path.read_bytes()); self.seal()
                    with self.assertRaises(ValueError): self.load()
            review = deepcopy(original); review["screenshot"]["visually_reviewed"] = False
            path.write_bytes(encoded(review)); self.proof[f"{role}_review"]["sha256"] = m._sha(path.read_bytes()); self.seal()
            with self.assertRaises(ValueError): self.load()
            path.write_bytes(encoded(original)); self.proof[f"{role}_review"]["sha256"] = m._sha(path.read_bytes())

    def test_changed_source_criterion_or_failed_evaluation_rejected(self):
        self.report["detailed_report"]["checkpoints"][3]["steps"][23]["details"] = "Wrong source URL"
        self.write_originals(); self.write_reviews()
        self.write_evidence("evaluation_attempt", f"evaluation-attempts/{m.ATTEMPT}.json", encoded(self.report)); self.seal()
        with self.assertRaises(ValueError): self.load()
        self.report["status"] = "evaluation_error"
        self.write_originals(); self.write_reviews(); self.seal()
        with self.assertRaises(ValueError): self.load()

    def test_short_or_malformed_checkpoint_arrays_fail_closed_before_indexing(self):
        for key in ("result", "detailed_report"):
            original = deepcopy(self.report[key]["checkpoints"])
            for bad in ([], original[:-1], None, [None] * 7):
                with self.subTest(key=key, bad=bad):
                    self.report[key]["checkpoints"] = bad
                    with self.assertRaises(ValueError):
                        m._report_steps(self.report, self.jsonlines(self.events))
            self.report[key]["checkpoints"] = original


if __name__ == "__main__":
    unittest.main()
