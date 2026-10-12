"""A hung browser-state export must not hide or replay a completed model turn."""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from codex_benchmark.browser import BrowserSession
from codex_benchmark.runner import run_episode


class FinalizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_hung_state_export_preserves_result_before_teardown_and_still_grades(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "model" / "task"
            directory.mkdir(parents=True)
            episode_path = directory / "episode.json"
            workspace = {"verified": True, "workspace_id": "doc", "workspace_url": "https://docs.google.com/document/d/doc/edit"}
            (directory / "workspace.json").write_text(json.dumps(workspace))
            state = root / "state.json"
            state.write_text('{"cookies": [], "origins": []}')
            state_before = state.read_bytes()
            browser = BrowserSession(output_dir=directory / "browser", timeout_ms=20)
            browser.history = [workspace["workspace_url"], "https://example.org/source"]
            browser.action_count = 2
            browser._active_page_id = "page_1"
            browser._pages = {"page_1": SimpleNamespace(url=workspace["workspace_url"], is_closed=lambda: False, goto=AsyncMock())}

            async def hung_state():
                await asyncio.Event().wait()

            browser.context = SimpleNamespace(storage_state=hung_state)
            outcome = {"status": "completed", "model": "model", "effort": "max",
                       "final_text": "Saved result", "tool_calls": 2, "token_usage": {"total": 123}}
            test = self

            class FakeServer:
                async def __aenter__(self):
                    return self

                async def run_turn(self, **kwargs):
                    return outcome

                async def __aexit__(self, *args):
                    saved = json.loads(episode_path.read_text())
                    test.assertEqual(saved["final_text"], outcome["final_text"])
                    test.assertEqual(saved["browsing_history"], browser.history)
                    test.assertEqual(saved["browser_actions"], 2)
                    test.assertEqual(json.loads((directory / "agent-result.json").read_text()), outcome)

            class FakeBrowserContext:
                async def __aenter__(self):
                    return browser

                async def __aexit__(self, *args):
                    await browser._close()

            async def grade(*args, **kwargs):
                saved = json.loads(episode_path.read_text())
                test.assertEqual(saved["status"], "completed")
                test.assertFalse(saved["storage_state_saved"])
                (directory / "evaluation.json").write_text(json.dumps({"status": "completed", "reward": .5}))
                return 0

            config = {"efforts": {"model": "max"}, "benchmark_root": str(root), "max_actions": 150,
                      "episode_timeout_seconds": 60, "evaluation_timeout_seconds": 1,
                      "headless": True, "channel": "chrome", "storage_state": str(state),
                      "finalization_timeout_seconds": .01}
            with patch("codex_benchmark.browser.BrowserSession", return_value=FakeBrowserContext()), \
                 patch("codex_benchmark.app_server.AppServer", return_value=FakeServer()), \
                 patch("codex_benchmark.knows.initial_workspace_url", return_value=None), \
                 patch("codex_benchmark.runner.build_prompt", return_value="Task"), \
                 patch("codex_benchmark.runner.subprocess_logged", side_effect=grade) as grader:
                episode = await asyncio.wait_for(run_episode(config, SimpleNamespace(task_id="task", family="docs"), "model", root), 1)
            self.assertEqual(episode["status"], "completed")
            self.assertEqual(episode["phase"], "finished")
            self.assertEqual(episode["final_text"], outcome["final_text"])
            self.assertEqual(state.read_bytes(), state_before)
            self.assertEqual(grader.call_count, 1)
            self.assertTrue(episode["browser_cleanup_errors"])


if __name__ == "__main__":
    unittest.main()
