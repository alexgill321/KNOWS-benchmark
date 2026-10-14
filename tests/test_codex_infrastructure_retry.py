"""Offline contract and preservation tests for the authorized Luna retry scope."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from codex_benchmark.browser import TOOLS
from codex_benchmark.config import MODELS
from codex_benchmark.infrastructure_retry import (
    AUTHORIZED_TASK_IDS, RETRY_KIND, RETRY_MODEL, prepare_retry_manifest,
    select_retry_tasks, validate_retry_contract,
)
from codex_benchmark.runner import SMOKE_TASK


class InfrastructureRetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.parent = self.root / "main"
        self.retry = self.root / "retry"
        self.parent.mkdir()
        ids = list(AUTHORIZED_TASK_IDS) + [SMOKE_TASK] + [f"knows.fixture.{i}" for i in range(100)]
        self.tasks = [SimpleNamespace(task_id=task) for task in ids]
        self.manifest = {"models": list(MODELS), "efforts": {model: "max" for model in MODELS},
                         "judge_model": "gemini-3.8-flash", "max_actions": 150,
                         "episode_timeout_seconds": 1800, "seed": 0,
                         "benchmark_commit": "saved-commit", "browser_tools_sha256": hashlib.sha256(
                             json.dumps(TOOLS, sort_keys=True).encode()).hexdigest(),
                         "repetitions": 1, "task_ids": ids, "tasks_per_model": 110,
                         "total_planned": 330}
        self.write(self.parent / "manifest.json", self.manifest)
        self.config = {key: deepcopy(self.manifest[key]) for key in
                       ("efforts", "judge_model", "max_actions", "episode_timeout_seconds", "seed")}
        self.config.update(models=[RETRY_MODEL], benchmark_root=str(self.root))
        self.authorization = {"kind": "user_authorized_infrastructure_retry", "model": RETRY_MODEL,
                              "task_ids": list(AUTHORIZED_TASK_IDS), "requested_at": "2026-10-04T10:20:00Z",
                              "statement": "Rerun these nine infrastructure-excluded tasks after repairs."}
        for model in MODELS:
            self.write(self.parent / model / SMOKE_TASK / "evaluation.json",
                       {"status": "completed", "has_execution_errors": False})
        for task in AUTHORIZED_TASK_IDS:
            self.original(task)
        self.git = patch("codex_benchmark.parallel.subprocess.check_output", return_value="saved-commit\n").start()
        self.dependency = patch("codex_benchmark.evaluator_dependency.is_reviewed_evaluator_dependency_diagnostic",
                                return_value=True).start()
        self.authentication = patch("codex_benchmark.auth_recovery.is_reviewed_auth_diagnostic", return_value=True).start()
        self.addCleanup(patch.stopall)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def original(self, task):
        directory = self.parent / RETRY_MODEL / task
        episode = {"task_id": task, "model": RETRY_MODEL, "effort": "max", "workspace_id": task,
                   "status": "completed", "thread_id": "thread", "turn_id": task,
                   "tool_calls": 12, "final_text": "Done", "benchmark_validity": {
                       "status": "excluded_infrastructure", "kind": "evaluator_dependency"}}
        if task == "knows.docs_31_education_lesson_plan.4":
            episode["benchmark_validity"].pop("kind")
            archived = directory / "recovery" / "reviewed" / "episode.json"
            self.write(archived, episode)
            episode["recovery"] = {"reviewed": True, "model_replayed": False,
                                   "source": "app-server.jsonl:run_result", "original_episode": str(archived)}
            self.write(directory / "app-server.jsonl", {"direction": "run_result", "message": {
                key: episode[key] for key in ("model", "effort", "thread_id", "turn_id", "status", "tool_calls", "final_text")}})
        elif task in {"knows.docs_5_influential_papers.5", "knows.sheets_10_paper_sorting.3"}:
            episode["benchmark_validity"].pop("kind")
            episode["status"] = "infrastructure_error"
            episode["recovery"] = {"kind": "auth_interrupted", "reviewed": True}
        self.write(directory / "episode.json", episode)
        self.write(directory / "evaluation.json", {"task_id": task, "workspace_id": task,
                   "status": "completed", "reward": 0.2})

    def prepare(self, **kwargs):
        return prepare_retry_manifest(self.config, self.retry, self.parent, self.tasks,
                                      task_ids=kwargs.get("task_ids", list(AUTHORIZED_TASK_IDS)),
                                      authorization=kwargs.get("authorization", self.authorization))

    def snapshot(self):
        return {str(p.relative_to(self.parent)): p.read_bytes() for p in self.parent.rglob("*") if p.is_file()}

    def test_exact_scope_preserves_originals_and_creates_only_manifest(self):
        before = self.snapshot()
        manifest = self.prepare()
        self.assertEqual(manifest["kind"], RETRY_KIND)
        self.assertEqual(manifest["total_planned"], 9)
        self.assertEqual(len(manifest["originals"]), 9)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual([p.name for p in self.retry.iterdir()], ["manifest.json"])
        self.assertEqual(validate_retry_contract(self.config, self.retry, self.tasks), manifest)
        selected = select_retry_tasks(self.config, self.retry, self.tasks)
        self.assertEqual([task.task_id for task in selected], sorted(AUTHORIZED_TASK_IDS))
        self.assertNotIn(SMOKE_TASK, [task.task_id for task in selected])

    def test_extra_empty_duplicate_or_nonstring_targets_are_rejected_before_write(self):
        for ids in (list(AUTHORIZED_TASK_IDS) + [SMOKE_TASK], [],
                    [AUTHORIZED_TASK_IDS[0]] * 9, [None] * 9, "all"):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self.prepare(task_ids=ids)
            self.assertFalse(self.retry.exists())

    def test_authorization_must_name_exact_tasks_model_and_timezone(self):
        cases = [{}, {**self.authorization, "model": "gpt-6-astra"},
                 {**self.authorization, "task_ids": list(AUTHORIZED_TASK_IDS)[:-1]},
                 {**self.authorization, "statement": ""},
                 {**self.authorization, "requested_at": "2026-10-04T10:20:00"}]
        for auth in cases:
            with self.subTest(auth=auth), self.assertRaises(ValueError):
                self.prepare(authorization=auth)
        self.assertFalse(self.retry.exists())

    def test_nonexcluded_or_wrong_identity_original_cannot_be_retried(self):
        path = self.parent / RETRY_MODEL / AUTHORIZED_TASK_IDS[-1] / "episode.json"
        initial = json.loads(path.read_text())
        for change in ({"benchmark_validity": {}}, {"model": "gpt-6-astra"}, {"task_id": SMOKE_TASK}):
            self.write(path, {**initial, **change})
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "matching excluded"):
                self.prepare()
        self.assertFalse(self.retry.exists())

    def test_current_original_and_parent_hash_tampering_fail_validation(self):
        self.prepare()
        for path in (self.parent / RETRY_MODEL / AUTHORIZED_TASK_IDS[-1] / "episode.json",
                     self.parent / RETRY_MODEL / AUTHORIZED_TASK_IDS[-1] / "evaluation.json",
                     self.parent / "manifest.json"):
            before = path.read_bytes()
            path.write_bytes(before + b" \n")
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "changed"):
                validate_retry_contract(self.config, self.retry, self.tasks)
            path.write_bytes(before)

    def test_configuration_drift_is_rejected(self):
        self.prepare()
        for change in ({"models": ["gpt-6-astra"]}, {"models": list(MODELS)},
                       {"efforts": {RETRY_MODEL: "high"}}, {"judge_model": "different"},
                       {"max_actions": 151}, {"episode_timeout_seconds": 1801}, {"seed": 3}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_retry_contract({**self.config, **change}, self.retry, self.tasks)
        with patch("codex_benchmark.parallel.subprocess.check_output", return_value="different-commit"):
            with self.assertRaisesRegex(RuntimeError, "contract"):
                validate_retry_contract(self.config, self.retry, self.tasks)
        with patch("codex_benchmark.browser.TOOLS", []):
            with self.assertRaisesRegex(RuntimeError, "contract"):
                validate_retry_contract(self.config, self.retry, self.tasks)

    def test_runtime_limits_and_browser_settings_are_pinned_at_preparation(self):
        self.config.update(evaluation_timeout_seconds=900, headless=True, channel="chrome",
                           finalization_timeout_seconds=10, browser_timeout_ms=15000,
                           viewport={"width": 1440, "height": 900})
        manifest = self.prepare()
        self.assertEqual(manifest["runtime_config"]["evaluation_timeout_seconds"], 900)
        self.assertEqual(manifest["runtime_config"]["viewport"], self.config["viewport"])
        for change in ({"evaluation_timeout_seconds": 901}, {"headless": False},
                       {"headless": 1}, {"channel": "chromium"},
                       {"finalization_timeout_seconds": 30}, {"browser_timeout_ms": 16000},
                       {"browser_timeout_seconds": 16}, {"timeout_ms": 16000},
                       {"viewport": {"width": 1920, "height": 1080}}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "runtime configuration"):
                validate_retry_contract({**self.config, **change}, self.retry, self.tasks)
        altered = deepcopy(manifest)
        altered["runtime_config"]["evaluation_timeout_seconds"] = 901
        self.write(self.retry / "manifest.json", altered)
        with self.assertRaisesRegex(ValueError, "runtime configuration"):
            validate_retry_contract(self.config, self.retry, self.tasks)

    def test_omitted_runtime_fields_match_normalized_runtime_defaults(self):
        manifest = self.prepare()
        defaults = {"evaluation_timeout_seconds": 900, "headless": True, "channel": "chrome",
                    "finalization_timeout_seconds": 10}
        self.assertEqual(validate_retry_contract({**self.config, **defaults}, self.retry, self.tasks), manifest)

    def test_supported_review_proofs_are_required_not_just_excluded_label(self):
        for checker in (self.dependency, self.authentication):
            checker.return_value = False
            with self.assertRaisesRegex(ValueError, "proof is invalid"):
                self.prepare()
            checker.return_value = True
        path = self.parent / RETRY_MODEL / AUTHORIZED_TASK_IDS[0] / "episode.json"
        episode = json.loads(path.read_text()); episode["recovery"]["reviewed"] = False
        self.write(path, episode)
        with self.assertRaisesRegex(ValueError, "not been reviewed"):
            self.prepare()

    def test_lesson_plan_archived_recovery_and_terminal_hashes_are_pinned(self):
        self.prepare()
        path = self.parent / RETRY_MODEL / AUTHORIZED_TASK_IDS[0] / "recovery/reviewed/episode.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            validate_retry_contract(self.config, self.retry, self.tasks)

    def test_retry_cannot_reuse_original_tree_or_overwrite_existing_records(self):
        for path in (self.parent, self.parent / "retry", self.parent.parent):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "non-nested"):
                prepare_retry_manifest(self.config, path, self.parent, self.tasks,
                                       task_ids=list(AUTHORIZED_TASK_IDS), authorization=self.authorization)
        self.prepare()
        before = (self.retry / "manifest.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "empty new"):
            self.prepare()
        self.assertEqual((self.retry / "manifest.json").read_bytes(), before)

    def test_parent_smoke_gate_is_used_without_copying_a_smoke_episode(self):
        path = self.parent / RETRY_MODEL / SMOKE_TASK / "evaluation.json"
        self.write(path, {"status": "completed", "has_execution_errors": True})
        with self.assertRaisesRegex(RuntimeError, "smoke"):
            self.prepare()
        self.assertFalse(self.retry.exists())

    def test_tampered_manifest_scope_or_lineage_is_rejected(self):
        manifest = self.prepare()
        path = self.retry / "manifest.json"
        for change in ({"task_ids": [SMOKE_TASK] * 9}, {"models": ["gpt-6-astra"]},
                       {"originals": []}, {"total_planned": 10}, {"schema_version": 2}):
            self.write(path, {**manifest, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_retry_contract(self.config, self.retry, self.tasks)

    def test_existing_retry_terminal_is_not_replayed_by_original_worker(self):
        # The new manifest supplies scope only; existing durable queue semantics
        # remain in charge of avoiding a second turn after a terminal result.
        from codex_benchmark.parallel_queue import initialize_queue, claim_next
        self.prepare()
        task = sorted(AUTHORIZED_TASK_IDS)[0]
        self.write(self.retry / RETRY_MODEL / task / "episode.json", {"status": "completed"})
        self.write(self.retry / RETRY_MODEL / task / "evaluation.json", {"status": "completed", "reward": .5})
        schedule = [(item.task_id, RETRY_MODEL) for item in select_retry_tasks(self.config, self.retry, self.tasks)]
        initialize_queue(self.retry / "parallel.sqlite", schedule, [RETRY_MODEL], self.retry)
        claim = claim_next(self.retry / "parallel.sqlite", "worker-1", 123)
        self.assertNotEqual(claim["task_id"], task)

    def test_staged_subset_keeps_full_authorization_and_all_original_proofs(self):
        selected = list(AUTHORIZED_TASK_IDS[:3])
        manifest = self.prepare(task_ids=selected)
        self.assertEqual(manifest["task_ids"], sorted(selected))
        self.assertEqual(manifest["authorized_task_ids"], sorted(AUTHORIZED_TASK_IDS))
        self.assertEqual(len(manifest["originals"]), 9)
        self.assertEqual(manifest["total_planned"], 3)
        self.assertEqual([task.task_id for task in select_retry_tasks(self.config, self.retry, self.tasks)],
                         sorted(selected))
        # Even a deferred original stays pinned to the same authorized plan.
        deferred = self.parent / RETRY_MODEL / AUTHORIZED_TASK_IDS[-1] / "evaluation.json"
        deferred.write_text(deferred.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            validate_retry_contract(self.config, self.retry, self.tasks)

    def test_disjoint_batches_share_registry_and_overlap_cannot_replay(self):
        first = self.prepare(task_ids=list(AUTHORIZED_TASK_IDS[:3]))
        self.retry = self.root / "second-batch"
        second = self.prepare(task_ids=list(AUTHORIZED_TASK_IDS[3:]))
        self.assertEqual(first["reservation_plan_id"], second["reservation_plan_id"])
        self.assertEqual(first["reservation_registry"], second["reservation_registry"])
        self.retry = self.root / "overlapping-batch"
        with self.assertRaisesRegex(ValueError, "overlapping replay"):
            self.prepare(task_ids=[AUTHORIZED_TASK_IDS[0]])
        self.assertFalse((self.retry / "manifest.json").exists())
        with sqlite3.connect(first["reservation_registry"]) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM reservations").fetchone()[0], 9)

    def test_copying_manifest_or_tampering_reservations_cannot_create_runnable_batch(self):
        manifest = self.prepare(task_ids=list(AUTHORIZED_TASK_IDS[:3]))
        copied = self.root / "copied"
        copied.mkdir()
        shutil.copy2(self.retry / "manifest.json", copied / "manifest.json")
        with self.assertRaisesRegex(ValueError, "reservations"):
            validate_retry_contract(self.config, copied, self.tasks)
        with sqlite3.connect(manifest["reservation_registry"]) as connection:
            connection.execute("UPDATE reservations SET run_dir=? WHERE task_id=?",
                               (str(copied), AUTHORIZED_TASK_IDS[0]))
        with self.assertRaisesRegex(ValueError, "reservations"):
            validate_retry_contract(self.config, self.retry, self.tasks)

    def test_failed_manifest_install_rolls_back_task_reservations(self):
        with patch("codex_benchmark.infrastructure_retry.os.link", side_effect=OSError("install failed")):
            with self.assertRaisesRegex(OSError, "install failed"):
                self.prepare(task_ids=list(AUTHORIZED_TASK_IDS[:3]))
        self.assertFalse((self.retry / "manifest.json").exists())
        self.retry = self.root / "successful-batch"
        manifest = self.prepare(task_ids=list(AUTHORIZED_TASK_IDS[:3]))
        self.assertEqual(validate_retry_contract(self.config, self.retry, self.tasks), manifest)

    def test_missing_registry_fails_closed_without_creating_it(self):
        manifest = self.prepare(task_ids=list(AUTHORIZED_TASK_IDS[:3]))
        registry = Path(manifest["reservation_registry"])
        registry.unlink()
        with self.assertRaisesRegex(ValueError, "registry is unavailable"):
            validate_retry_contract(self.config, self.retry, self.tasks)
        self.assertFalse(registry.exists())


if __name__ == "__main__":
    unittest.main()
