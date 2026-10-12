import unittest
import json
from pathlib import Path
import tempfile
import sys
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import MagicMock, Mock, patch
from types import ModuleType, SimpleNamespace

from googleapiclient.errors import HttpError
from codex_benchmark.provision import capture_provision_failure, provision, retry_new_file_visibility


class ProvisionTests(unittest.TestCase):
    def error(self, status):
        return HttpError(SimpleNamespace(status=status, reason="error"), b'{"error":{"message":"file not visible"}}')

    def test_new_file_visibility_retries_but_does_not_accept_permission_errors(self):
        request = Mock(side_effect=[self.error(404), self.error(404), {"id": "file"}])
        sleep = Mock()
        self.assertEqual(retry_new_file_visibility(request, sleep=sleep), {"id": "file"})
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 2])
        request = Mock(side_effect=self.error(403))
        with self.assertRaises(HttpError):
            retry_new_file_visibility(request, sleep=sleep)
        self.assertEqual(request.call_count, 1)


class ProvisionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "workspace.json"

    def test_auth_evidence_is_sanitized_before_bounded_screenshot(self):
        def screenshot(**kwargs):
            self.assertEqual(kwargs["timeout"], 5000)
            self.assertEqual(kwargs["type"], "png")
            self.assertTrue((self.directory / "provision-failure.json").exists())
            self.assertTrue((self.directory / "auth-health.json").exists())
            Path(kwargs["path"]).write_bytes(b"fixture")

        page = SimpleNamespace(url="https://accounts.google.com/v3/signin/challenge/private%40example.com?token=private-token#private-fragment",
                               screenshot=Mock(side_effect=screenshot))
        evidence = capture_provision_failure(page, self.output, TimeoutError("private-error-details"))
        self.assertEqual(evidence["location"], {"host": "accounts.google.com", "path": "/v3/signin/challenge/[redacted]"})
        self.assertEqual(evidence["auth_status"], "login_required")
        self.assertEqual(evidence["screenshot_status"], "saved")
        for file in self.directory.glob("*.json"):
            self.assertNotIn("private", file.read_text())
        self.assertFalse(self.output.exists())

    def test_failed_screenshot_preserves_metadata_and_partial_workspace(self):
        page = SimpleNamespace(url="https://docs.google.com/spreadsheets/d/workspace_123/edit?token=private-token",
                               screenshot=Mock(side_effect=TimeoutError("private-page-data")))
        evidence = capture_provision_failure(page, self.output, TimeoutError("create timed out"))
        self.assertEqual(evidence["screenshot_status"], "failed")
        self.assertEqual(evidence["screenshot_error_type"], "TimeoutError")
        self.assertEqual(evidence["auth_status"], "unknown")
        saved = json.loads((self.directory / "partial-workspace.json").read_text())
        self.assertEqual(saved, {"workspace_id": "workspace_123", "workspace_url":
            "https://docs.google.com/spreadsheets/d/workspace_123/edit", "verified": False})
        self.assertFalse((self.directory / "auth-health.json").exists())
        self.assertNotIn("private", (self.directory / "provision-failure.json").read_text())

    def test_google_login_link_in_non_google_url_is_not_auth_evidence(self):
        page = SimpleNamespace(url="https://example.com/private%40example.com?next=https://accounts.google.com/ServiceLogin",
                               screenshot=Mock())
        evidence = capture_provision_failure(page, self.output, TimeoutError())
        self.assertEqual(evidence["location"], {"host": "example.com", "path": "/[redacted]"})
        self.assertEqual(evidence["auth_status"], "unknown")
        self.assertFalse((self.directory / "auth-health.json").exists())

    def test_workspace_url_in_query_does_not_create_partial_workspace(self):
        page = SimpleNamespace(url="https://example.com/?next=https://docs.google.com/document/d/not_created/edit",
                               screenshot=Mock())
        capture_provision_failure(page, self.output, TimeoutError())
        self.assertFalse((self.directory / "partial-workspace.json").exists())


class ProvisionFailureIntegrationTests(unittest.TestCase):
    """Exercise the real orchestration with browser and Google clients fully mocked."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.output = self.directory / "workspace.json"
        service_account = self.directory / "service-account.json"
        service_account.write_text(json.dumps({"client_email": "evaluator@example.test"}))
        self.config = {
            "benchmark_root": str(self.directory), "token_path": "unused-token-path",
            "service_account_path": str(service_account), "storage_state": "unused-browser-state",
            "headless": True, "channel": "chrome", "google_owner_email": "owner@example.test",
        }
        self.workspace = {"workspace_id": "workspace_123", "workspace_url":
            "https://docs.google.com/presentation/d/workspace_123/edit"}
        self.page = SimpleNamespace(url=self.workspace["workspace_url"], screenshot=Mock())
        self.context = Mock()
        self.context.new_page.return_value = self.page
        self.browser = Mock()
        self.browser.new_context.return_value = self.context
        playwright = MagicMock()
        playwright.__enter__.return_value.chromium.launch.return_value = self.browser
        self.owner_drive, self.evaluator_drive = Mock(), Mock()
        self.owner_drive.about.return_value.get.return_value.execute.return_value = {
            "user": {"emailAddress": "owner@example.test"}}
        self.evaluator_drive.files.return_value.get.return_value.execute.return_value = {
            "id": "workspace_123", "capabilities": {"canEdit": True}}
        setup = ModuleType("browsergym.knows.doc_setup")
        self.create_workspace = Mock(return_value=tuple(self.workspace.values()))
        setup.create_task_workspace = self.create_workspace
        self.enterContext(patch.dict(sys.modules, {"browsergym.knows.doc_setup": setup}))
        self.enterContext(patch.object(sys, "path", list(sys.path)))
        self.enterContext(patch("playwright.sync_api.sync_playwright", return_value=playwright))
        self.enterContext(patch("google.oauth2.credentials.Credentials.from_authorized_user_file",
                                return_value=SimpleNamespace(valid=True)))
        self.enterContext(patch("google.oauth2.service_account.Credentials.from_service_account_file"))
        self.enterContext(patch("googleapiclient.discovery.build",
                                side_effect=[self.owner_drive, self.evaluator_drive]))
        self.sleep = Mock()
        self.enterContext(patch("codex_benchmark.provision.retry_new_file_visibility",
            side_effect=lambda request: retry_new_file_visibility(request, sleep=self.sleep)))

    def run_provision(self):
        return provision(self.config, "knows.slides_20_Illustrated_Book_Report.1", "gpt-6-luna", self.output)

    def assert_evidence(self, phase, error_type):
        evidence = json.loads((self.directory / "provision-failure.json").read_text())
        self.assertEqual(evidence["phase"], phase)
        self.assertEqual(evidence["error_type"], error_type)
        self.browser.close.assert_called_once_with()
        self.context.storage_state.assert_not_called()
        if self.output.exists():
            self.assertEqual(json.loads(self.output.read_text()), self.workspace)
        return evidence

    def test_permission_404_preserves_identity_and_captures_before_close(self):
        error = HttpError(SimpleNamespace(status=404, reason="error"),
                          b'{"error":{"message":"private-api-details"}}')
        request = self.owner_drive.permissions.return_value.create.return_value.execute
        request.side_effect = error
        self.page.url += "?token=private-token#private-fragment"

        def screenshot(**kwargs):
            self.browser.close.assert_not_called()
            self.assertEqual(json.loads(self.output.read_text()), self.workspace)
            Path(kwargs["path"]).write_bytes(b"fixture")

        self.page.screenshot.side_effect = screenshot
        with self.assertRaises(HttpError) as caught:
            self.run_provision()
        self.assertIs(caught.exception, error)
        self.assertEqual(request.call_count, 5)
        self.assertEqual([call.args[0] for call in self.sleep.call_args_list], [1, 2, 4, 8])
        self.assertEqual(self.assert_evidence("grant_evaluator_access", "HttpError")["screenshot_status"], "saved")
        self.assertEqual(json.loads((self.directory / "partial-workspace.json").read_text()),
                         {**self.workspace, "verified": False})
        for name in ("provision-failure.json", "partial-workspace.json", "workspace.json"):
            self.assertNotIn("private", (self.directory / name).read_text())

    def test_evaluator_access_failure_is_captured(self):
        self.evaluator_drive.files.return_value.get.return_value.execute.return_value = {
            "id": "workspace_123", "capabilities": {"canEdit": False}}
        with self.assertRaisesRegex(RuntimeError, "cannot edit"):
            self.run_provision()
        self.assert_evidence("verify_evaluator_access", "RuntimeError")

    def test_browser_auth_failure_preserves_workspace_and_records_auth_health(self):
        self.page.url = "https://accounts.google.com/v3/signin/challenge/private?token=private-token"
        with self.assertRaisesRegex(RuntimeError, "authentication expired"):
            self.run_provision()
        evidence = self.assert_evidence("verify_browser_authentication", "RuntimeError")
        self.assertEqual(evidence["auth_status"], "login_required")
        auth = (self.directory / "auth-health.json").read_text()
        self.assertEqual(json.loads(auth)["status"], "authentication_required")
        self.assertNotIn("private", auth)

    def test_create_failure_still_captures_without_inventing_workspace_identity(self):
        error = TimeoutError("creation failed")
        self.create_workspace.side_effect = error
        with self.assertRaises(TimeoutError) as caught:
            self.run_provision()
        self.assertIs(caught.exception, error)
        self.assert_evidence("create_workspace", "TimeoutError")
        self.assertFalse(self.output.exists())
        self.owner_drive.permissions.assert_not_called()

    def test_capture_failure_cannot_replace_original_failure(self):
        error = HttpError(SimpleNamespace(status=404, reason="error"), b'{"error":{"message":"original"}}')
        self.owner_drive.permissions.return_value.create.return_value.execute.side_effect = error
        stderr = StringIO()
        with patch("codex_benchmark.provision.capture_provision_failure",
                   side_effect=OSError("private-diagnostic-details")), redirect_stderr(stderr):
            with self.assertRaises(HttpError) as caught:
                self.run_provision()
        self.assertIs(caught.exception, error)
        self.assertEqual(json.loads(self.output.read_text()), self.workspace)
        self.assertEqual(stderr.getvalue().strip(), "Provisioning evidence capture failed: OSError")
        self.browser.close.assert_called_once_with()
        self.context.storage_state.assert_not_called()

    def test_success_remains_verified_without_failure_artifacts(self):
        self.assertEqual(self.run_provision(), {**self.workspace, "verified": True})
        self.assertEqual(json.loads(self.output.read_text()), {**self.workspace, "verified": True})
        self.context.storage_state.assert_called_once_with(path=self.config["storage_state"])
        self.page.screenshot.assert_not_called()
        self.assertFalse((self.directory / "provision-failure.json").exists())
        self.browser.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
