"""Browser boundary and accounting tests; no browser or network is required."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from codex_benchmark.browser import ActionBudgetExceeded, BrowserAuthenticationError, BrowserSession, validate_url


class URLBoundaryTests(unittest.TestCase):
    def test_preserves_full_public_urls(self):
        for url in [
            "https://docs.google.com/document/d/abc/edit?tab=t.0#heading=h.abc",
            "https://example.com:443/path?q=one%20two&ref=full",
            "https://abc.de/",
            "http://8.8.8.8/example",
            "https://[2606:4700:4700::1111]/",
        ]:
            with self.subTest(url=url):
                self.assertEqual(validate_url(url), url)

    def test_blocks_nonweb_local_and_ambiguous_urls(self):
        for url in [
            "file:///etc/passwd", "javascript:alert(1)", "data:text/html,test",
            "about:blank", "https://", "/relative", "https://localhost/",
            "https://LOCALHOST./", "http://api.localhost/", "http://router.local/",
            "http://127.0.0.1/", "http://10.1.2.3/", "http://169.254.169.254/",
            "http://[::1]/", "http://[::ffff:127.0.0.1]/", "http://[fe80::1%25en0]/",
            "http://127.1/", "http://2130706433/", "http://0177.0.0.1/",
            "http://0x7f000001/", "https://%6cocalhost/", "https://intranet/",
            "https://example.com:99999/", "https://example.com:notaport/",
            "https://public.example\\@localhost/", "https://user:pass@example.com/",
            "https://example.com/\nlocalhost", " https://example.com/",
        ]:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_url(url)


class FakePage:
    def __init__(self, url="about:blank"):
        self.url = url
        self.main_frame = SimpleNamespace(url=url)
        self.closed = False
        self.callbacks = {}
        self.goto = AsyncMock()
        self.bring_to_front = AsyncMock()

    def is_closed(self):
        return self.closed

    def on(self, event, callback):
        self.callbacks[event] = callback

    def navigate(self, url, *, child=False):
        frame = SimpleNamespace(url=url) if child else self.main_frame
        frame.url = url
        if not child:
            self.url = url
        self.callbacks["framenavigated"](frame)


class BrowserAccountingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.session = BrowserSession(output_dir=self.tempdir.name, max_actions=2)
        self.session._observe = AsyncMock(return_value=[{"type": "inputText", "text": "observed"}])

    async def test_exact_budget_including_failed_calls(self):
        execute = AsyncMock(return_value="done")
        self.session._execute = execute
        self.assertFalse((await self.session.call("not_a_tool", {}))["success"])
        self.assertTrue((await self.session.call("browser_observe", {}))["success"])
        with self.assertRaises(ActionBudgetExceeded):
            await self.session.call("browser_observe", {})
        self.assertEqual(self.session.action_count, 2)
        self.assertEqual(execute.await_count, 1)
        self.assertEqual(self.session._observe.await_count, 2)

    async def test_concurrent_calls_cannot_overrun_budget(self):
        self.session._execute = AsyncMock(return_value="done")
        results = await asyncio.gather(*[
            self.session.call("browser_observe", {}) for _ in range(5)
        ], return_exceptions=True)
        self.assertEqual(sum(isinstance(item, ActionBudgetExceeded) for item in results), 3)
        self.assertEqual(self.session._execute.await_count, 2)

    async def test_bad_argument_never_executes_but_gets_observation(self):
        self.session._execute = AsyncMock()
        result = await self.session.call("browser_click", {"x": True, "y": 2})
        self.assertFalse(result["success"])
        self.session._execute.assert_not_awaited()
        self.session._observe.assert_awaited_once()
        self.assertIn("finite number", result["contentItems"][0]["text"])

    async def test_action_exception_is_recorded_and_budget_exception_propagates(self):
        self.session._execute = AsyncMock(side_effect=RuntimeError("button vanished"))
        result = await self.session.call("browser_observe", {})
        self.assertFalse(result["success"])
        records = Path(self.tempdir.name, "actions.jsonl").read_text().splitlines()
        self.assertIn("button vanished", json.loads(records[0])["message"])
        self.session._execute = AsyncMock(side_effect=ActionBudgetExceeded("exhausted"))
        with self.assertRaises(ActionBudgetExceeded):
            await self.session.call("browser_observe", {})

    async def test_history_records_full_main_frame_urls_and_popup_start(self):
        page = FakePage()
        self.session._attach_page(page)
        page.navigate("https://example.com/page?query=a%20b#part")
        page.navigate("https://example.com/iframe?hidden=true", child=True)
        page.navigate("https://example.com/page?query=changed")
        popup = FakePage("https://example.org/popup?full=query")
        self.session._attach_page(popup)
        self.session._attach_page(popup)  # Listener registration is idempotent.
        popup.navigate("https://example.org/next#fragment")
        self.assertEqual(self.session.history, [
            "https://example.com/page?query=a%20b#part",
            "https://example.com/page?query=changed",
            "https://example.org/popup?full=query",
            "https://example.org/next#fragment",
        ])
        self.assertEqual(self.session.page_id, "page_2")
        popup.closed = True
        self.assertEqual(self.session.page_id, "page_1")
        next_popup = FakePage()
        self.session._attach_page(next_popup)
        self.assertEqual(self.session.page_id, "page_3")
        await self.session.save_history()
        self.assertEqual(json.loads(Path(self.tempdir.name, "history.json").read_text()), self.session.history)

    async def test_private_redirect_is_aborted(self):
        request = SimpleNamespace(url="http://127.0.0.1/admin", is_navigation_request=lambda: True)
        route = SimpleNamespace(request=request, abort=AsyncMock(), continue_=AsyncMock())
        await self.session._route(route)
        route.abort.assert_awaited_once_with("blockedbyclient")
        route.continue_.assert_not_awaited()

    async def test_invalid_new_tab_url_does_not_create_tab(self):
        self.session.context = SimpleNamespace(new_page=AsyncMock())
        result = await self.session.call("browser_new_tab", {"url": "file:///etc/passwd"})
        self.assertFalse(result["success"])
        self.session.context.new_page.assert_not_awaited()

    async def test_observation_contains_real_png_and_url(self):
        session = BrowserSession(output_dir=self.tempdir.name)
        page = FakePage("https://example.com/?full=query")
        page.screenshot = AsyncMock(return_value=b"pngbytes")
        locator = SimpleNamespace(aria_snapshot=AsyncMock(return_value="- heading: Example"))
        page.locator = lambda selector: locator
        session._attach_page(page)
        result = await session._observe()
        self.assertEqual(json.loads(result[0]["text"])["url"], page.url)
        self.assertEqual(result[1], {"type": "inputImage", "imageUrl": "data:image/png;base64,cG5nYnl0ZXM="})
        self.assertFalse(page.screenshot.call_args.kwargs.get("full_page", False))

    async def test_download_saved_with_opaque_id_and_safe_filename(self):
        async def save_as(path):
            Path(path).write_bytes(b"download bytes")

        download = SimpleNamespace(suggested_filename="../../outside.pdf", save_as=AsyncMock(side_effect=save_as))
        self.session._start_download(download)
        await asyncio.gather(*self.session._download_tasks)
        artifact = self.session._artifacts["download_00001"]
        self.assertEqual(artifact["status"], "ready")
        self.assertTrue(artifact["path"].resolve().is_relative_to(Path(self.tempdir.name).resolve()))
        result = await self.session.call("browser_files", {})
        metadata = json.loads(result["contentItems"][0]["text"])["files"][0]
        self.assertNotIn("path", metadata)
        self.assertEqual(metadata["bytes"], len(b"download bytes"))

    async def test_upload_rejects_arbitrary_host_paths(self):
        page = FakePage("https://example.com/upload")
        locator = SimpleNamespace(set_input_files=AsyncMock())
        page.locator = lambda selector: locator
        self.session._attach_page(page)
        result = await self.session.call("browser_upload", {"artifact_id": "/etc/passwd", "selector": "input[type=file]"})
        self.assertFalse(result["success"])
        locator.set_input_files.assert_not_awaited()

    async def test_ready_download_can_be_uploaded_through_pending_chooser(self):
        artifact = self.session._new_artifact("download", "paper.pdf")
        artifact["path"].write_bytes(b"paper")
        artifact["status"] = "ready"
        page = FakePage("https://example.com/upload")
        self.session._attach_page(page)
        chooser = SimpleNamespace(set_files=AsyncMock())
        page.callbacks["filechooser"](chooser)
        result = await self.session.call("browser_upload", {"artifact_id": artifact["artifact_id"]})
        self.assertTrue(result["success"])
        chooser.set_files.assert_awaited_once_with(str(artifact["path"].resolve()))
        self.assertNotIn("page_1", self.session._file_choosers)

    async def test_screenshot_region_becomes_uploadable_artifact(self):
        async def screenshot(**kwargs):
            Path(kwargs["path"]).write_bytes(b"cropped screenshot")
            return b"cropped screenshot"

        page = FakePage("https://example.com/figure")
        page.screenshot = AsyncMock(side_effect=screenshot)
        self.session._attach_page(page)
        region = {"x": 10, "y": 20, "width": 100, "height": 200}
        result = await self.session.call("browser_screenshot", region)
        self.assertTrue(result["success"])
        artifact = json.loads(result["contentItems"][0]["text"])
        self.assertEqual(artifact["artifact_id"], "screenshot_00001")
        self.assertEqual(artifact["status"], "ready")
        self.assertNotIn("path", artifact)
        self.assertEqual(page.screenshot.call_args.kwargs["clip"], region)

    async def test_partial_or_outside_screenshot_crop_never_captures(self):
        page = FakePage("https://example.com/figure")
        page.screenshot = AsyncMock()
        self.session._attach_page(page)
        partial = await self.session.call("browser_screenshot", {"width": 100})
        outside = await self.session.call("browser_screenshot", {"x": 1400, "y": 0, "width": 100, "height": 200})
        self.assertFalse(partial["success"])
        self.assertFalse(outside["success"])
        page.screenshot.assert_not_awaited()

    async def test_drag_releases_mouse_after_motion_error(self):
        page = FakePage("https://example.com/slide")
        page.mouse = SimpleNamespace(move=AsyncMock(side_effect=[None, RuntimeError("motion failed")]), down=AsyncMock(), up=AsyncMock())
        self.session._attach_page(page)
        result = await self.session.call("browser_drag", {"from_x": 20, "from_y": 30, "to_x": 200, "to_y": 300})
        self.assertFalse(result["success"])
        page.mouse.up.assert_awaited_once()

    async def test_failed_browser_close_still_stops_driver_and_preserves_results(self):
        self.session._browser = SimpleNamespace(close=AsyncMock(side_effect=TimeoutError("slow Chrome")))
        self.session._playwright = SimpleNamespace(stop=AsyncMock())
        await self.session._close()
        self.session._playwright.stop.assert_awaited_once()
        self.assertIn("slow Chrome", self.session.cleanup_errors[0])
        self.assertTrue(Path(self.tempdir.name, "cleanup-errors.json").is_file())


class BrowserAuthenticationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.session = BrowserSession(output_dir=self.root)
        self.page = FakePage("https://docs.google.com/document/d/example/edit")
        self.session._attach_page(self.page)
        self.session._observe = AsyncMock(return_value=[{"type": "inputText", "text": "observed"}])

    async def test_active_signin_blocks_action_before_execution(self):
        # Also catches current URL changes if a navigation event was missed.
        self.page.url = "https://accounts.google.com/v3/signin/identifier?Email=private@example.com"
        self.session._execute = AsyncMock()
        with self.assertRaises(BrowserAuthenticationError):
            await self.session.call("browser_type", {"text": "do not type"})
        self.session._execute.assert_not_awaited()
        self.session._observe.assert_not_awaited()
        self.assertEqual(self.session.action_count, 0)
        self.assertTrue((self.root / "auth-health.json").exists())

    async def test_navigation_auth_failure_escapes_after_evidence_is_saved(self):
        async def redirected(*args):
            self.page.navigate("https://accounts.google.com/ServiceLogin?continue=private-token")

        self.session._execute = AsyncMock(side_effect=redirected)
        with self.assertRaises(BrowserAuthenticationError):
            await self.session.call("browser_navigate", {"url": "https://docs.google.com/"})
        self.session._observe.assert_awaited_once()
        self.assertEqual(len((self.root / "actions.jsonl").read_text().splitlines()), 1)
        self.assertTrue((self.root / "history.json").exists())

    async def test_auth_record_and_exception_redact_query_fragment_and_unknown_path(self):
        self.page.navigate("https://accounts.google.com/v3/signin/challenge/private%40example.com?token=private-query#private-fragment")
        with self.assertRaises(BrowserAuthenticationError) as raised:
            self.session.assert_authenticated()
        saved = (self.root / "auth-health.json").read_text()
        record = json.loads(saved)
        self.assertEqual(record["host"], "accounts.google.com")
        self.assertEqual(record["path"], "/v3/signin/challenge/[redacted]")
        self.assertEqual(record["page_id"], "page_1")
        self.assertEqual(record["auth_status"], "login_required")
        self.assertEqual(record["auth_location"], {"host": record["host"], "path": record["path"]})
        self.assertEqual(record["checked_at"], record["timestamp"])
        self.assertIn("timestamp", record)
        self.assertNotIn("private", saved + str(raised.exception))
        self.assertNotIn("?", saved)

    async def test_actual_popup_challenge_remains_sticky_after_tab_closes(self):
        popup = FakePage("https://accounts.google.com/signin/v2/challenge/pwd")
        self.session._attach_page(popup)
        popup.closed = True
        self.assertEqual(self.session.page_id, "page_1")
        with self.assertRaises(BrowserAuthenticationError):
            self.session.assert_authenticated()

    async def test_exact_apps_script_oauth_consent_popup_does_not_invent_signout(self):
        popup = FakePage("https://accounts.google.com/signin/oauth/legacy/consent?scope=private#private")
        self.session._attach_page(popup)
        self.session._execute = AsyncMock(return_value="")
        self.assertTrue((await self.session.call("browser_observe", {}))["success"])
        self.session._execute.assert_awaited_once_with("browser_observe", {})
        self.assertIsNone(self.session.auth_issue)
        self.assertFalse((self.root / "auth-health.json").exists())
        # Only observation was requested: the guard neither clicks nor grants.
        self.assertEqual(self.session.action_count, 1)

    async def test_consent_followed_by_real_google_challenge_still_stops_actions(self):
        popup = FakePage("https://accounts.google.com/signin/oauth/legacy/consent?scope=private")
        self.session._attach_page(popup)
        popup.navigate("https://accounts.google.com/v3/signin/challenge/pwd?token=private")
        self.session._execute = AsyncMock()
        with self.assertRaises(BrowserAuthenticationError):
            await self.session.call("browser_click", {"x": 1, "y": 1})
        self.session._execute.assert_not_awaited()
        saved = (self.root / "auth-health.json").read_text()
        self.assertNotIn("private", saved)
        self.assertIn("challenge", saved)

    async def test_consent_exception_is_exact_and_never_clears_real_sticky_failure(self):
        from codex_benchmark.browser import _google_auth_location
        for path in ("/signin/oauth/legacy/consent/challenge", "/signin/oauth/legacy/consent/identifier",
                     "/signin/oauth/legacy/consent/AccountChooser", "/signin/oauth/legacy/consentish",
                     "/signin/oauth/challenge", "/v3/signin/identifier", "/AccountChooser"):
            with self.subTest(path=path):
                self.assertIsNotNone(_google_auth_location("https://accounts.google.com" + path))
        self.page.navigate("https://accounts.google.com/AccountChooser")
        self.page.navigate("https://accounts.google.com/signin/oauth/legacy/consent")
        with self.assertRaises(BrowserAuthenticationError):
            self.session.assert_authenticated()

    async def test_non_google_pages_login_links_iframes_and_cookie_bridge_are_benign(self):
        self.page.navigate("https://accounts.google.com/signin/identifier", child=True)
        self.session._observe = AsyncMock(return_value=[{"type": "inputText", "text":
            'Sign in: <a href="https://accounts.google.com/ServiceLogin">Login</a>'}])
        self.session._execute = AsyncMock(return_value="")
        for url in [
            "https://www.usgs.gov/login?next=https://accounts.google.com/ServiceLogin",
            "https://accounts.google.com.example.org/signin/challenge",
            "https://example.com/accounts.google.com/signin",
            "https://accounts.google.com/CheckCookie?continue=https://docs.google.com/",
        ]:
            self.page.navigate(url)
            self.assertTrue((await self.session.call("browser_observe", {}))["success"])
        self.assertIsNone(self.session.auth_issue)
        self.assertFalse((self.root / "auth-health.json").exists())

    async def test_auth_failure_prevents_overwriting_saved_login_state(self):
        destination = self.root / "login.json"
        prior = '{"cookies": [], "origins": [], "prior": true}'
        destination.write_text(prior)
        self.page.navigate("https://accounts.google.com/AccountChooser")
        self.session.context = SimpleNamespace(storage_state=AsyncMock(return_value={"cookies": [], "origins": []}))
        self.assertFalse(await self.session.save_storage_state(destination))
        self.session.context.storage_state.assert_not_awaited()
        self.assertEqual(destination.read_text(), prior)


class BrowserFinalizationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.destination = self.root / "login-state.json"
        self.previous = '{"cookies": [], "origins": [], "prior": true}\n'
        self.destination.write_text(self.previous)
        self.session = BrowserSession(output_dir=self.root / "browser", timeout_ms=10)
        self.state = {"cookies": [{"name": "test", "value": "private-cookie"}], "origins": []}

    async def test_successful_export_is_private_and_atomic(self):
        self.session.context = SimpleNamespace(storage_state=AsyncMock(return_value=self.state))
        self.assertTrue(await self.session.save_storage_state(self.destination))
        self.session.context.storage_state.assert_awaited_once_with()
        self.assertEqual(json.loads(self.destination.read_text()), self.state)
        self.assertEqual(self.destination.stat().st_mode & 0o777, 0o600)
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    async def test_export_errors_keep_prior_state_and_do_not_log_secrets(self):
        self.session.context = SimpleNamespace(storage_state=AsyncMock(side_effect=RuntimeError("private-cookie")))
        self.assertFalse(await self.session.save_storage_state(self.destination))
        self.assertEqual(self.destination.read_text(), self.previous)
        self.assertNotIn("private-cookie", (self.root / "browser/cleanup-errors.json").read_text())

    async def test_invalid_export_and_failed_replace_preserve_prior_state(self):
        self.session.context = SimpleNamespace(storage_state=AsyncMock(return_value={"cookies": None, "origins": []}))
        self.assertFalse(await self.session.save_storage_state(self.destination))
        self.assertEqual(self.destination.read_text(), self.previous)
        self.session.context.storage_state = AsyncMock(return_value=self.state)
        with patch("codex_benchmark.browser.os.replace", side_effect=OSError("disk error")):
            self.assertFalse(await self.session.save_storage_state(self.destination))
        self.assertEqual(self.destination.read_text(), self.previous)
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    async def test_export_deadline_does_not_wait_for_cancellation_or_write_late_result(self):
        release = asyncio.Event()

        async def stalled_export():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()
                return self.state

        self.session.context = SimpleNamespace(storage_state=AsyncMock(side_effect=stalled_export))
        try:
            result = await asyncio.wait_for(self.session.save_storage_state(self.destination, timeout_seconds=.01), .5)
            self.assertFalse(result)
            self.assertEqual(self.destination.read_text(), self.previous)
        finally:
            tasks = tuple(self.session._cleanup_tasks)
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        self.assertEqual(self.destination.read_text(), self.previous)

    async def test_external_cancellation_preserves_state_and_propagates(self):
        entered = asyncio.Event()

        async def stalled_export():
            entered.set()
            await asyncio.Event().wait()

        self.session.context = SimpleNamespace(storage_state=AsyncMock(side_effect=stalled_export))
        export = asyncio.create_task(self.session.save_storage_state(self.destination))
        await entered.wait()
        export.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await export
        await asyncio.gather(*tuple(self.session._cleanup_tasks), return_exceptions=True)
        self.assertEqual(self.destination.read_text(), self.previous)

    async def test_download_cancellation_cannot_block_browser_and_driver_cleanup(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def stalled_download():
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                await release.wait()

        download = asyncio.create_task(stalled_download())
        await entered.wait()
        self.session._download_tasks.add(download)
        self.session._browser = SimpleNamespace(close=AsyncMock())
        self.session._playwright = SimpleNamespace(stop=AsyncMock())
        try:
            await asyncio.wait_for(self.session._close(), .5)
            self.session._browser.close.assert_awaited_once()
            self.session._playwright.stop.assert_awaited_once()
            self.assertTrue(any("cancel browser tasks" in error for error in self.session.cleanup_errors))
        finally:
            release.set()
            await download


if __name__ == "__main__":
    unittest.main()
