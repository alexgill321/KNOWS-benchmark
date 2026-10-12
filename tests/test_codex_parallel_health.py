import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timezone

from codex_benchmark.parallel_health import inspect_episode


NOW = 1_800_000_000


def iso(age=0):
    return datetime.fromtimestamp(NOW - age, timezone.utc).isoformat()


class ParallelHealthTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.directory = self.root / "gpt-6-luna" / "knows.docs_1_formal_letter.1"
        self.directory.mkdir(parents=True)
        self.clock = patch("codex_benchmark.parallel_health.time.time", return_value=NOW)
        self.clock.start()
        self.episode = {"task_id": self.directory.name, "model": "gpt-6-luna", "status": "running",
                        "started_at": iso(400), "timeout_seconds": 1800}
        self.save("episode.json", self.episode)

    def tearDown(self):
        self.clock.stop()
        self.temporary.cleanup()

    def save(self, name, value, age=0):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        os.utime(path, (NOW - age, NOW - age))
        return path

    def journal(self, name, entries):
        path = self.directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
        return path

    def test_recent_browser_progress_and_screenshot_are_read_only(self):
        self.journal("browser/actions.jsonl", [{"action": 9, "timestamp": iso(20), "tool": "browser_click", "success": True}])
        self.journal("browser/navigations.jsonl", [{"page_id": "page_1", "url": "https://docs.google.com/document/d/example/edit?secret=hidden", "timestamp": iso(30)}])
        screenshot = self.directory / "browser/screenshots/00009_page_1.png"
        screenshot.parent.mkdir(parents=True); screenshot.write_bytes(b"pixels")
        os.utime(screenshot, (NOW - 25, NOW - 25))
        before = {str(path): (path.stat().st_mtime_ns, path.read_bytes()) for path in self.directory.rglob("*") if path.is_file()}
        health = inspect_episode(self.directory)
        after = {str(path): (path.stat().st_mtime_ns, path.read_bytes()) for path in self.directory.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(health["progress_age_seconds"], 20)
        self.assertEqual(health["latest_screenshot"], str(screenshot.resolve()))
        self.assertEqual(health["auth_status"], "no_login_redirect")
        self.assertEqual(health["browser_actions"], 9)
        self.assertNotIn("hidden", json.dumps(health))
        self.assertFalse(health["blocked"])

    def test_login_links_in_page_text_do_not_indicate_auth_failure(self):
        self.journal("app-server.jsonl", [{"time": NOW - 1, "direction": "send", "message": {"id": 1,
            "result": {"contentItems": [{"type": "inputText", "text": "Log in: https://accounts.google.com/signin?secret=hidden"}]}}}])
        self.journal("browser/actions.jsonl", [{"timestamp": iso(1), "message": "https://accounts.google.com/signin", "arguments": {"url": "https://accounts.google.com/signin"}, "success": False}])
        health = inspect_episode(self.directory)
        self.assertEqual(health["auth_status"], "unknown")
        self.assertFalse(health["blocked"])

    def test_actual_login_navigation_is_blocked_and_query_is_removed(self):
        self.journal("browser/navigations.jsonl", [{"page_id": "page_1", "url": "https://user:password@accounts.google.com/v3/signin/challenge?email=secret#token", "timestamp": iso(2)}])
        health = inspect_episode(self.directory)
        self.assertEqual(health["reason"], "google_login_required")
        self.assertTrue(health["blocked"])
        self.assertEqual(health["auth_location"], {"host": "accounts.google.com", "path": "/v3/signin/challenge"})
        self.assertNotIn("secret", json.dumps(health))
        self.assertFalse(health["stalled"])

    def test_sticky_auth_health_survives_later_unrelated_navigation(self):
        self.save("auth-health.json", {"status": "authentication_required", "host": "accounts.google.com",
            "path": "/v3/signin/challenge/[redacted]", "timestamp": iso(100)})
        self.journal("browser/navigations.jsonl", [{"page_id": "page_2", "url": "https://docs.google.com/document/d/example", "timestamp": iso(1)}])
        health = inspect_episode(self.directory)
        self.assertTrue(health["blocked"])
        self.assertEqual(health["auth_status"], "login_required")

    def test_google_cookie_bridge_is_not_a_login_failure(self):
        self.journal("browser/navigations.jsonl", [{"page_id": "page_1", "url": "https://accounts.google.com/CheckCookie?continue=private", "timestamp": iso(2)}])
        self.assertFalse(inspect_episode(self.directory)["blocked"])

    def test_oauth_permission_review_navigation_is_not_lost_google_authentication(self):
        self.journal("browser/navigations.jsonl", [{"page_id": "page_4",
            "url": "https://accounts.google.com/signin/oauth/legacy/consent?scope=private#token",
            "timestamp": iso(2)}])
        health = inspect_episode(self.directory)
        self.assertFalse(health["blocked"])
        self.assertNotEqual(health["auth_status"], "login_required")
        self.assertNotIn("private", json.dumps(health))
        self.assertNotIn("token", json.dumps(health))

    def test_consent_does_not_clear_existing_sticky_auth_evidence(self):
        self.save("browser/auth-health.json", {"auth_status": "login_required", "checked_at": iso(10),
            "auth_location": {"host": "accounts.google.com", "path": "/v3/signin/challenge"}})
        self.journal("browser/navigations.jsonl", [{"page_id": "page_4",
            "url": "https://accounts.google.com/signin/oauth/legacy/consent", "timestamp": iso(2)}])
        self.assertTrue(inspect_episode(self.directory)["blocked"])

    def test_browser_and_monitor_agree_on_exact_consent_exception_and_real_login_routes(self):
        from codex_benchmark.browser import _google_auth_location as browser_location
        from codex_benchmark.parallel_health import _google_auth_location as monitor_location
        for path, blocked in (("/signin/oauth/legacy/consent", False),
                              ("/signin/oauth/legacy/consent/", False),
                              ("/signin/oauth/legacy/consent/challenge", True),
                              ("/signin/oauth/legacy/consentish", True),
                              ("/v3/signin/identifier", True), ("/AccountChooser", True),
                              ("/signin/oauth/challenge", True), ("/ServiceLogin", True)):
            with self.subTest(path=path):
                url = "https://accounts.google.com" + path + "?private=hidden#secret"
                self.assertEqual(browser_location(url), monitor_location(url))
                self.assertEqual(monitor_location(url) is not None, blocked)

    def test_auth_health_compatibility_location_redacts_private_path(self):
        self.save("browser/auth-health.json", {"auth_status": "login_required", "checked_at": iso(4),
            "auth_location": {"host": "accounts.google.com", "path": "/v3/signin/challenge/private-email@example.com?token=hidden"}})
        health = inspect_episode(self.directory)
        self.assertEqual(health["auth_location"], {"host": "accounts.google.com", "path": "/v3/signin/challenge/[redacted]"})
        self.assertNotIn("private-email", json.dumps(health))
        self.assertNotIn("hidden", json.dumps(health))

    def test_terminal_protocol_result_detects_finalization_stall(self):
        self.journal("app-server.jsonl", [{"time": NOW - 301, "direction": "run_result", "message": {"status": "completed"}}])
        health = inspect_episode(self.directory)
        self.assertEqual(health["phase"], "finalizing")
        self.assertTrue(health["stalled"])
        self.assertEqual(health["reason"], "terminal_finalization_stalled")

    def test_running_deadline_needs_absence_of_recent_progress(self):
        self.episode.update(started_at=iso(2200)); self.save("episode.json", self.episode)
        self.assertEqual(inspect_episode(self.directory)["reason"], "model_deadline_exceeded_without_progress")
        self.journal("app-server.jsonl", [{"time": NOW - 10, "direction": "receive", "message": {"method": "item/reasoning/textDelta", "params": {"delta": "working"}}}])
        self.assertFalse(inspect_episode(self.directory)["stalled"])

    def test_evaluator_deadline_and_checkpoint_progress(self):
        self.episode.update(status="completed", phase="evaluating", phase_started_at=iso(1300), evaluation_timeout_seconds=900)
        self.save("episode.json", self.episode)
        self.journal("evaluation-steps.jsonl", [{"timestamp": iso(1290), "attempt_id": "a", "event": "evaluation_started"},
            {"timestamp": iso(5), "attempt_id": "a", "event": "step_completed", "checkpoint_index": 3}])
        health = inspect_episode(self.directory)
        self.assertEqual(health["reason"], "evaluator_deadline_exceeded")
        self.assertEqual(health["evaluation_steps"], 1)
        self.assertEqual(health["evaluation_checkpoints"], 3)
        self.assertEqual(health["progress_age_seconds"], 5)

    def test_old_evaluation_error_does_not_block_active_retry(self):
        self.episode.update(status="completed", phase="evaluating", phase_started_at=iso(30), evaluation_error="old failure with secret URL")
        self.save("episode.json", self.episode)
        self.save("evaluation.json", {"status": "evaluation_error", "error": "secret"}, age=100)
        self.journal("evaluation-steps.jsonl", [{"timestamp": iso(29), "attempt_id": "b", "event": "evaluation_started"}])
        self.assertFalse(inspect_episode(self.directory)["blocked"])
        self.save("evaluation.json", {"status": "evaluation_error", "error": "secret"}, age=1)
        health = inspect_episode(self.directory)
        self.assertEqual(health["reason"], "evaluation_error")
        self.assertNotIn("secret", json.dumps(health))

    def test_completed_and_explicitly_deferred_episodes_are_not_stalled(self):
        self.episode.update(status="completed", finished_at=iso(1000))
        self.save("episode.json", self.episode)
        self.save("evaluation.json", {"status": "completed", "reward": 1}, age=990)
        self.assertEqual(inspect_episode(self.directory)["phase"], "finished")
        self.assertFalse(inspect_episode(self.directory)["stalled"])
        (self.directory / "evaluation.json").unlink()
        self.episode.update(evaluation_deferred=True, phase="ready_for_evaluation")
        self.save("episode.json", self.episode)
        self.assertFalse(inspect_episode(self.directory)["stalled"])

    def test_partial_journals_and_credential_symlinks_are_not_read(self):
        credential = self.root / "browser-state.json"; credential.write_text('{"cookies": "private-cookie"}')
        (self.directory / "auth-health.json").symlink_to(credential)
        path = self.journal("browser/actions.jsonl", [{"timestamp": iso(5), "action": 2}])
        with path.open("a") as handle: handle.write('{"timestamp":')
        health = inspect_episode(self.directory)
        self.assertEqual(health["browser_actions"], 2)
        self.assertEqual(health["auth_status"], "unknown")
        self.assertNotIn("private-cookie", json.dumps(health))


if __name__ == "__main__":
    unittest.main()
