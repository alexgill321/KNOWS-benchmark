from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from codex_benchmark.educational_source_corrections import (
    ARCHIVE_NAME, CORRECTIONS_NAME, PENDING_NAME, REASON, TARGETS, load_educational_source_correction,
    _collector_source,
)
from codex_benchmark.evaluation_corrections import evaluation_for_scoring
from codex_benchmark.runner import summarize


class EducationalSourceCorrectionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve(); self.run = self.root / "smoke"
        self.archive = self.root / "recovery" / ARCHIVE_NAME
        self.pending_path = self.root / PENDING_NAME; self.manifest_path = self.root / CORRECTIONS_NAME
        self.write(self.run / "manifest.json", {"models": ["gpt-6-luna", "gpt-6-astra"], "tasks_per_model": 110})
        self.fixtures = []
        sources = []
        for relative in ["utils.py", *(f"instance_{n}/evaluator.py" for n in range(1, 6))]:
            path = self.archive / "fixed-source" / relative
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text("# Frozen " + relative)
            sources.append({"path": str(path), "sha256": self.sha(path), "relative_path": relative})
        for model, task in sorted(TARGETS):
            directory = self.run / model / task; corrected = self.archive / model / task
            agent = {"model": model, "effort": "max", "status": "completed", "thread_id": model + task,
                     "turn_id": "turn1", "tool_calls": 150, "final_text": "done", "token_usage": {}}
            episode = {**agent, "task_id": task, "workspace_id": task + model, "judge_model": "gemini-3.8-flash"}
            original = {"status": "completed", "reward": .4, "task_id": task, "workspace_id": episode["workspace_id"],
                        "judge_model": episode["judge_model"], "judge_errors": [], "evaluator_invoked": True,
                        "score_source": "official_evaluator", "evaluation_attempt_id": str(uuid4())}
            for name, value in (("episode.json", episode), ("agent-result.json", agent), ("evaluation.json", original)):
                self.write(directory / name, value)
            hashes = {name: self.sha(directory / name) for name in ("episode.json", "evaluation.json", "agent-result.json")}
            raw = corrected / "raw-presentation.json"; self.write(raw, {"slides": [], "source": "saved capture"})
            proof_path = corrected / "source-proof.json"
            proof = {"kind": REASON, "verified": True, "unchanged_rubric": True, "model_replayed": False,
                     "model": model, "task_id": task, "workspace_id": episode["workspace_id"],
                     "original_sha256": hashes, "source_snapshots": sources,
                     "raw_snapshots": [{"path": str(raw), "sha256": self.sha(raw)}]}
            self.write(proof_path, proof)
            pending = {"model": model, "task_id": task, "workspace_id": episode["workspace_id"], "reason": REASON,
                       "model_replayed": False, "original_sha256": hashes, "evaluation_attempt_id": original["evaluation_attempt_id"],
                       "original_reward": original["reward"], "status": "pending_saved_output_regrade",
                       "source_proof_path": str(proof_path), "source_proof_sha256": self.sha(proof_path)}
            attempt = str(uuid4()); report_path = corrected / "evaluation.json"; journal = corrected / "evaluation-steps.jsonl"
            rows = [{"event": event, "attempt_id": attempt, "task_id": task, "workspace_id": episode["workspace_id"]}
                    for event in ["evaluation_started", *(["step_completed"] * 24), "evaluation_finished"]]
            journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
            source_sha = next(s["sha256"] for s in sources if s["relative_path"] == f"instance_{task[-1]}/evaluator.py")
            report = {**original, "reward": .5, "evaluation_attempt_id": attempt, "has_execution_errors": True,
                      "evaluator_source": {"original_sha256": source_sha, "effective_sha256": source_sha},
                      "step_journal": {"attempt_id": attempt, "path": str(journal), "completed_steps": 24},
                      "detailed_report": {"checkpoints": [{"name": "rubric", "steps": [{"name": f"s{i}", "step_id": i} for i in range(24)]}]}}
            self.write(report_path, report); self.write(corrected / "evaluation-attempts" / (attempt + ".json"), report)
            entry = {**pending, "evaluation_attempt_id": attempt, "report_path": str(report_path), "report_sha256": self.sha(report_path),
                     "journal_sha256": self.sha(journal)}
            review = {"reviewed": True, "scoring_valid": True, "model_replayed": False, "execution_errors_reviewed": True,
                      "no_auth_or_judge_failure": True, "no_unresolved_evaluator_error": True,
                      "model": model, "task_id": task, "workspace_id": episode["workspace_id"],
                      "evaluation_attempt_id": attempt, "report_sha256": entry["report_sha256"],
                      "journal_sha256": entry["journal_sha256"], "original_report_sha256": hashes["evaluation.json"],
                      "original_episode_sha256": hashes["episode.json"], "scored_reward": .5}
            for label in ("report_review", "independent_review"):
                path = corrected / (label + ".json"); self.write(path, {**review, "reviewer": label})
                entry[label + "_path"] = str(path); entry[label + "_sha256"] = self.sha(path)
            self.fixtures.append({"directory": directory, "path": directory / "episode.json", "original": original,
                                  "pending": pending, "entry": entry, "proof": proof, "proof_path": proof_path,
                                  "report": report, "report_path": report_path, "journal": journal})
        self.pending = {"version": 1, "kind": "pending_educational_source_evaluation_corrections", "run_dir": str(self.run),
                        "model_replayed": False, "targets": [f["pending"] for f in self.fixtures]}
        self.write(self.pending_path, self.pending)
        self.manifest = {"version": 1, "kind": "reviewed_educational_source_evaluation_corrections", "run_dir": str(self.run),
                         "model_replayed": False, "pending_manifest_sha256": self.sha(self.pending_path),
                         "corrections": [f["entry"] for f in self.fixtures]}
        self.write(self.manifest_path, self.manifest)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value) + "\n")

    def sha(self, path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def assert_held(self, fixture):
        result = evaluation_for_scoring(fixture["path"], fixture["original"])
        self.assertIsNotNone(result["error"]); self.assertEqual(result["evaluation"], fixture["original"])

    def test_verified_three_overlays_preserve_all_originals_and_allow_reviewed_output_errors(self):
        before = {p: p.read_bytes() for f in self.fixtures for p in f["directory"].iterdir()}
        for fixture in self.fixtures:
            result = load_educational_source_correction(fixture["path"])
            self.assertEqual(result["evaluation"]["reward"], .5)
            self.assertTrue(result["evaluation"]["has_execution_errors"])
        summary = summarize(self.run)
        self.assertEqual(sum(m["scored"] for m in summary["models"].values()), 3)
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_absent_authority_no_hold_invalid_scope_never_holds_unrelated_tasks(self):
        self.manifest_path.unlink(); self.pending_path.unlink()
        self.assertIsNone(load_educational_source_correction(self.fixtures[0]["path"]))
        self.pending_path.write_text("malformed")
        for fixture in self.fixtures: self.assert_held(fixture)
        for model, task in (("gpt-6-luna", "knows.slides_26_basic_educational_slide_deck.4"),
                            ("gpt-6-astra", "knows.slides_26_basic_educational_slide_deck.3")):
            path = self.run / model / task / "episode.json"; self.write(path, {"model": model, "task_id": task})
            self.assertIsNone(evaluation_for_scoring(path, {"status": "completed", "reward": .7})["error"])

    def test_pending_and_partial_review_holds_only_unreleased_target_scores(self):
        self.manifest_path.unlink()
        for fixture in self.fixtures: self.assert_held(fixture)
        self.manifest["corrections"] = [self.fixtures[0]["entry"]]; self.write(self.manifest_path, self.manifest)
        self.assertIsNone(evaluation_for_scoring(self.fixtures[0]["path"], self.fixtures[0]["original"])["error"])
        for fixture in self.fixtures[1:]: self.assert_held(fixture)

    def test_wrong_pending_identity_duplicate_or_missing_target_fails_closed(self):
        variants = [{**self.pending, "targets": self.pending["targets"][:2]},
                    {**self.pending, "targets": [self.pending["targets"][0]] * 3}]
        for field, value in (("model", "other"), ("reason", "best_score"), ("original_sha256", {}), ("original_reward", .9)):
            obj = deepcopy(self.pending); obj["targets"][0][field] = value; variants.append(obj)
        for obj in variants:
            self.write(self.pending_path, obj)
            for fixture in self.fixtures: self.assert_held(fixture)

    def test_any_original_raw_frozen_source_or_review_tamper_is_rejected(self):
        fixture = self.fixtures[0]
        paths = [*fixture["directory"].iterdir(), fixture["proof_path"], fixture["report_path"], fixture["journal"],
                 *(Path(p["path"]) for p in fixture["proof"]["source_snapshots"]),
                 *(Path(p["path"]) for p in fixture["proof"]["raw_snapshots"]),
                 Path(fixture["entry"]["report_review_path"]), Path(fixture["entry"]["independent_review_path"])]
        for path in paths:
            with self.subTest(path=path.name):
                old = path.read_bytes(); path.write_bytes(old + b" ")
                self.assert_held(fixture); path.write_bytes(old)

    def test_proof_must_bind_exact_originals_all_frozen_wrappers_and_raw_capture(self):
        fixture = self.fixtures[0]
        for field, value in (("raw_snapshots", []), ("source_snapshots", fixture["proof"]["source_snapshots"][:1]),
                             ("model_replayed", True), ("original_sha256", {}), ("unchanged_rubric", False)):
            proof = {**fixture["proof"], field: value}; self.write(fixture["proof_path"], proof)
            fixture["pending"]["source_proof_sha256"] = self.sha(fixture["proof_path"])
            self.write(self.pending_path, self.pending)
            with self.assertRaises(ValueError): load_educational_source_correction(fixture["path"])

    def test_root_and_independent_reviews_need_exact_scoring_approval(self):
        fixture = self.fixtures[0]
        for label in ("report_review", "independent_review"):
            path = Path(fixture["entry"][label + "_path"]); original = json.loads(path.read_text())
            for field, value in (("scoring_valid", False), ("execution_errors_reviewed", False),
                                 ("no_unresolved_evaluator_error", False), ("model_replayed", True),
                                 ("report_sha256", "0" * 64), ("evaluation_attempt_id", str(uuid4())), ("scored_reward", .8)):
                self.write(path, {**original, field: value}); fixture["entry"][label + "_sha256"] = self.sha(path)
                self.write(self.manifest_path, self.manifest)
                self.assert_held(fixture)
            self.write(path, original); fixture["entry"][label + "_sha256"] = self.sha(path)
        fixture["entry"]["independent_review_path"] = fixture["entry"]["report_review_path"]
        fixture["entry"]["independent_review_sha256"] = fixture["entry"]["report_review_sha256"]
        self.write(self.manifest_path, self.manifest); self.assert_held(fixture)

    def test_correction_cannot_replace_pending_lineage_or_escape_archive(self):
        fixture = self.fixtures[0]
        for field, value in (("original_sha256", {}), ("workspace_id", "other"), ("source_proof_sha256", "0" * 64)):
            manifest = deepcopy(self.manifest); manifest["corrections"][0][field] = value
            self.write(self.manifest_path, manifest); self.assert_held(fixture)
        self.write(self.manifest_path, self.manifest)
        with self.assertRaises(ValueError): load_educational_source_correction(fixture["path"], self.root / "outside.json")

    def test_incomplete_report_or_missing_independent_attempt_rejected(self):
        fixture = self.fixtures[0]
        report = deepcopy(fixture["report"]); report["detailed_report"]["checkpoints"][0]["steps"].pop()
        self.write(fixture["report_path"], report)
        self.write(fixture["report_path"].parent / "evaluation-attempts" / (report["evaluation_attempt_id"] + ".json"), report)
        fixture["entry"]["report_sha256"] = self.sha(fixture["report_path"])
        self.write(self.manifest_path, self.manifest); self.assert_held(fixture)

    def collector_fixture(self):
        fixture = next(f for f in self.fixtures if f['entry']['model'] == 'gpt-6-astra')
        entry = deepcopy(fixture['entry']); report = deepcopy(fixture['report'])
        sources = []
        for item in fixture['proof']['source_snapshots']:
            path = self.archive / 'collector-fixed-source' / item['relative_path']
            path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# New collector ' + item['relative_path'])
            sources.append({**item, 'path': str(path), 'sha256': self.sha(path)})
        review_path = self.archive / 'collector-independent-review.json'
        self.write(review_path, dict(approved=True, collector_repair_approved=True, source_snapshots=sources))
        proof = {key: entry[key] for key in ('model', 'task_id', 'workspace_id', 'original_sha256')}
        proof.update(kind=REASON, repair_kind='educational_image_match_collection', verified=True, model_replayed=False,
                     unchanged_rubric=True, parent_source_proof_sha256=entry['source_proof_sha256'],
                     outer_evaluation_timeout_seconds=900, max_workers=5, collection_timeout_seconds=240,
                     retains_completed_results=True, fails_closed=True, source_snapshots=sources,
                     independent_review_path=str(review_path), independent_review_sha256=self.sha(review_path))
        proof_path = self.archive / 'collector-proof.json'; self.write(proof_path, proof)
        entry.update(collector_recovery_proof_path=str(proof_path), collector_recovery_proof_sha256=self.sha(proof_path))
        sha = next(s['sha256'] for s in sources if s['relative_path'] == f"instance_{entry['task_id'][-1]}/evaluator.py")
        report['evaluator_source'] = dict(original_sha256=sha, effective_sha256=sha)
        return entry, report, proof, proof_path

    def test_collector_proof_accepts_new_sources_without_editing_parent_authority(self):
        before = self.pending_path.read_bytes()
        entry, report, _, _ = self.collector_fixture()
        _collector_source(entry, report, self.archive)
        self.assertEqual(before, self.pending_path.read_bytes())
        for fixture in self.fixtures:
            _collector_source(fixture['entry'], fixture['report'], self.archive)

    def test_collector_overlay_validates_new_source_and_keeps_both_provenance_links(self):
        pending_before = self.pending_path.read_bytes()
        entry, report, _, _ = self.collector_fixture()
        fixture = next(f for f in self.fixtures if f['entry']['task_id'] == entry['task_id'] and f['entry']['model'] == entry['model'])
        self.write(fixture['report_path'], report)
        self.write(fixture['report_path'].parent / 'evaluation-attempts' / (report['evaluation_attempt_id'] + '.json'), report)
        entry['report_sha256'] = self.sha(fixture['report_path'])
        for label in ('report_review', 'independent_review'):
            path = Path(entry[label + '_path']); review = json.loads(path.read_text())
            review['report_sha256'] = entry['report_sha256']; self.write(path, review)
            entry[label + '_sha256'] = self.sha(path)
        fixture['entry'].update(entry); self.write(self.manifest_path, self.manifest)
        correction = load_educational_source_correction(fixture['path'])
        self.assertEqual(correction['evaluation'], report)
        self.assertEqual(correction['provenance']['source_proof_sha256'], entry['source_proof_sha256'])
        self.assertEqual(correction['provenance']['collector_recovery_proof_sha256'], entry['collector_recovery_proof_sha256'])
        self.assertEqual(self.pending_path.read_bytes(), pending_before)

    def test_unreviewed_collector_or_wrong_report_source_fails_closed(self):
        entry, report, _, _ = self.collector_fixture()
        for key in ('collector_recovery_proof_path', 'collector_recovery_proof_sha256'):
            broken = deepcopy(entry); broken.pop(key)
            with self.assertRaises((ValueError, KeyError)): _collector_source(broken, report, self.archive)
        broken = deepcopy(report); broken['evaluator_source']['effective_sha256'] = '0' * 64
        with self.assertRaises(ValueError): _collector_source(entry, broken, self.archive)
        plain = {k:v for k,v in entry.items() if not k.startswith('collector_')}
        with self.assertRaises(ValueError): _collector_source(plain, report, self.archive)

    def test_collector_proof_rejects_changed_scope_budget_and_missing_sources(self):
        entry, report, proof, path = self.collector_fixture()
        for field, value in (('model', 'gpt-6-luna'), ('workspace_id', 'wrong'), ('original_sha256', {}),
                             ('parent_source_proof_sha256', '0' * 64), ('outer_evaluation_timeout_seconds', 901),
                             ('max_workers', 6), ('collection_timeout_seconds', 241), ('retains_completed_results', False),
                             ('fails_closed', False), ('model_replayed', True), ('source_snapshots', [])):
            self.write(path, {**proof, field:value}); entry['collector_recovery_proof_sha256'] = self.sha(path)
            with self.subTest(field=field), self.assertRaises(ValueError): _collector_source(entry, report, self.archive)

    def test_collector_frozen_source_and_independent_review_tamper_rejected(self):
        entry, report, proof, _ = self.collector_fixture()
        for path in [Path(proof['independent_review_path']), *(Path(s['path']) for s in proof['source_snapshots'])]:
            original = path.read_bytes(); path.write_bytes(original + b' ')
            with self.assertRaises(ValueError): _collector_source(entry, report, self.archive)
            path.write_bytes(original)

    def test_collector_independent_review_must_approve_exact_sources(self):
        entry, report, proof, path = self.collector_fixture()
        review_path = Path(proof['independent_review_path'])
        for field,value in (('approved', False), ('collector_repair_approved', False), ('source_snapshots', [])):
            self.write(review_path, dict(approved=True, collector_repair_approved=True, source_snapshots=proof['source_snapshots']) | {field:value})
            proof['independent_review_sha256'] = self.sha(review_path); self.write(path, proof)
            entry['collector_recovery_proof_sha256'] = self.sha(path)
            with self.assertRaises(ValueError): _collector_source(entry, report, self.archive)


if __name__ == "__main__": unittest.main()
