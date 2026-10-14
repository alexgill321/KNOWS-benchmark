import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError


SOURCE = Path(__file__).resolve().parents[1] / "src/browsergym/knows/doc_setup.py"
SPEC = importlib.util.spec_from_file_location("_codex_doc_setup", SOURCE)
doc_setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(doc_setup)


class SlidesWelcomeTests(unittest.TestCase):
    def setUp(self):
        self.welcome = Mock()
        self.welcome.is_visible.return_value = True
        self.title = Mock()
        self.title.is_visible.return_value = True
        self.close = self.welcome.get_by_role.return_value
        self.dialogs = Mock()
        self.dialogs.filter.return_value.first = self.welcome
        self.ready = self.welcome.or_.return_value.first
        self.page = SimpleNamespace(
            url="https://docs.google.com/presentation/d/original_file/edit",
            get_by_role=Mock(return_value=self.dialogs),
            locator=Mock(return_value=SimpleNamespace(first=self.title)),
        )

    def dismiss(self, kind="slides", expected_id="original_file"):
        return doc_setup._dismiss_slides_welcome(self.page, kind=kind, expected_id=expected_id)

    def test_known_welcome_closes_without_adding_content(self):
        self.assertTrue(self.dismiss())
        self.page.get_by_role.assert_called_once_with("dialog")
        self.dialogs.filter.assert_called_once_with(has_text="Let’s start creating.")
        self.welcome.get_by_role.assert_called_once_with("button", name="Close", exact=True)
        self.close.click.assert_called_once_with(timeout=5000)
        self.welcome.wait_for.assert_called_once_with(state="hidden", timeout=10000)
        self.title.wait_for.assert_called_once_with(state="visible", timeout=5000)
        self.assertEqual(self.page.url, "https://docs.google.com/presentation/d/original_file/edit")

    def test_normal_editor_and_unrelated_dialogs_are_immediate_noops(self):
        # Filtering an unrelated visible dialog yields no matching welcome.
        self.welcome.is_visible.return_value = False
        self.assertFalse(self.dismiss())
        self.ready.wait_for.assert_not_called()
        self.welcome.get_by_role.assert_not_called()
        self.welcome.wait_for.assert_not_called()

    def test_delayed_welcome_is_handled_before_rename(self):
        self.welcome.is_visible.return_value = False
        self.title.is_visible.return_value = False

        def becomes_ready(**kwargs):
            self.welcome.is_visible.return_value = True

        self.ready.wait_for.side_effect = becomes_ready
        self.assertTrue(self.dismiss())
        self.welcome.or_.assert_called_once_with(self.title)
        self.ready.wait_for.assert_called_once_with(state="visible", timeout=5000)
        self.close.click.assert_called_once_with(timeout=5000)

    def test_no_recognized_ui_defers_to_original_rename_readiness(self):
        self.welcome.is_visible.return_value = False
        self.title.is_visible.return_value = False
        self.ready.wait_for.side_effect = PlaywrightTimeoutError("not ready")
        self.assertFalse(self.dismiss())
        self.close.click.assert_not_called()

    def test_wrong_workspace_kind_does_not_inspect_or_act(self):
        for kind in ("docs", "sheets"):
            with self.subTest(kind=kind):
                self.assertFalse(self.dismiss(kind=kind))
        self.page.get_by_role.assert_not_called()
        self.page.locator.assert_not_called()

    def test_confirmed_welcome_click_and_dismissal_failures_propagate(self):
        for method in (self.close.click, self.welcome.wait_for):
            with self.subTest(method=method):
                error = PlaywrightTimeoutError("welcome failed")
                method.side_effect = error
                with self.assertRaises(PlaywrightTimeoutError) as caught:
                    self.dismiss()
                self.assertIs(caught.exception, error)
                method.side_effect = None

    def test_identity_mismatch_prevents_any_click(self):
        with self.assertRaisesRegex(RuntimeError, "identity changed before"):
            self.dismiss(expected_id="different_file")
        self.close.click.assert_not_called()

    def test_identity_change_after_close_is_rejected(self):
        def changes_identity(**kwargs):
            self.page.url = "https://docs.google.com/presentation/d/different_file/edit"

        self.close.click.side_effect = changes_identity
        with self.assertRaisesRegex(RuntimeError, "identity changed during"):
            self.dismiss()
        self.title.wait_for.assert_not_called()

    def test_workspace_creation_dismisses_before_rename_and_keeps_same_id(self):
        events = []
        self.page.goto = Mock()
        self.page.wait_for_url = Mock()
        self.page.wait_for_load_state = Mock(side_effect=lambda *args, **kwargs: events.append("loaded"))
        self.close.click.side_effect = lambda **kwargs: events.append("closed welcome")
        with patch.object(doc_setup, "_rename_via_ui", side_effect=lambda *args: events.append("renamed") or True):
            result = doc_setup.create_task_workspace(
                self.page, "agent", "knows.slides_20_Illustrated_Book_Report.1", 1,
                kind="slides", share_with_evaluator=False, max_attempts=1,
            )
        self.assertEqual(events, ["loaded", "closed welcome", "renamed"])
        self.assertEqual(result, ("original_file", "https://docs.google.com/presentation/d/original_file/edit"))
        self.page.goto.assert_called_once_with(
            "https://docs.google.com/presentation/create", timeout=60000, wait_until="commit")

    def test_workspace_creation_does_not_rename_or_share_after_dismissal_failure(self):
        self.page.goto = Mock()
        self.page.wait_for_url = Mock()
        self.page.wait_for_load_state = Mock()
        self.welcome.wait_for.side_effect = PlaywrightTimeoutError("dialog still visible")
        with patch.object(doc_setup, "_rename_via_ui") as rename, \
                patch.object(doc_setup, "share_doc_with_service_account") as share:
            with self.assertRaises(PlaywrightTimeoutError):
                doc_setup.create_task_workspace(
                    self.page, "agent", "knows.slides_20_Illustrated_Book_Report.1", 1,
                    kind="slides", max_attempts=1,
                )
        rename.assert_not_called()
        share.assert_not_called()


if __name__ == "__main__":
    unittest.main()
