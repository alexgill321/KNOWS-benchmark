"""Worker isolation tests: local fixtures only, no browser or Google requests."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from codex_benchmark.worker_state import ensure_worker_storage_state, prepare_worker_config


class WorkerStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.source = self.root / "source"
        self.source.mkdir()
        self.state = {"cookies": [{"name": "test", "value": "private-cookie", "domain": ".example.com", "path": "/"}], "origins": []}
        self.token = {"token": "old-access", "refresh_token": "private-refresh", "client_id": "test-client",
                      "client_secret": "private-secret", "token_uri": "https://oauth2.googleapis.com/token",
                      "expiry": "2000-01-01T00:00:00Z"}
        (self.source / "browser.json").write_text(json.dumps(self.state))
        (self.source / "token.json").write_text(json.dumps(self.token))
        self.config = {"config_path": str(self.source / "benchmark.json"), "storage_state": "browser.json",
                       "token_path": "token.json", "env_files": ["judge.env", str(self.root / "shared.env")],
                       "benchmark_root": "..", "service_account_path": "service.json",
                       "credentials_path": "client.json", "output_dir": "../results",
                       "models": ["gpt-6-luna", "gpt-6.1-sol", "gpt-6-astra"],
                       "efforts": {"gpt-6-luna": "max"}, "headless": True, "max_actions": 150}
        (self.source / "benchmark.json").write_text(json.dumps(self.config))

    def test_four_workers_have_independent_private_browser_and_token_files(self):
        original = deepcopy(self.config)
        workers = [prepare_worker_config(self.config, self.root / f"worker-{n}") for n in range(4)]
        self.assertEqual(self.config, original)
        for config in workers:
            directory = Path(config["config_path"]).parent
            self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
            for key, expected in (("storage_state", self.state), ("token_path", self.token)):
                path = Path(config[key])
                self.assertEqual(json.loads(path.read_text()), expected)
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
                self.assertEqual(path.stat().st_nlink, 1)
            self.assertEqual(Path(config["config_path"]).stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(Path(config["config_path"]).read_text()), config)
            self.assertEqual(config["models"], original["models"])
            self.assertEqual(config["efforts"], original["efforts"])
            self.assertEqual(config["max_actions"], original["max_actions"])
            self.assertEqual(config["env_files"], [str(self.source / "judge.env"), str(self.root / "shared.env")])
            for key in ("benchmark_root", "service_account_path", "credentials_path", "output_dir"):
                self.assertTrue(Path(config[key]).is_absolute())
        self.assertEqual(len({Path(c["storage_state"]).stat().st_ino for c in workers}), 4)
        self.assertEqual(len({Path(c["token_path"]).stat().st_ino for c in workers}), 4)
        self.assertEqual(json.loads((self.source / "browser.json").read_text()), self.state)
        self.assertEqual(json.loads((self.source / "token.json").read_text()), self.token)

    def test_restart_preserves_refreshed_worker_credentials(self):
        first = prepare_worker_config(self.config, self.root / "worker")
        state = deepcopy(self.state); state["cookies"][0]["value"] = "worker-refresh"
        token = dict(self.token, token="new-access")
        Path(first["storage_state"]).write_text(json.dumps(state))
        Path(first["token_path"]).write_text(json.dumps(token))
        (self.source / "browser.json").unlink()
        (self.source / "token.json").unlink()
        second = prepare_worker_config(self.config, self.root / "worker")
        self.assertEqual(first, second)
        self.assertEqual(json.loads(Path(second["storage_state"]).read_text()), state)
        self.assertEqual(json.loads(Path(second["token_path"]).read_text()), token)

    def test_invalid_existing_state_is_not_silently_reset(self):
        config = prepare_worker_config(self.config, self.root / "worker")
        path = Path(config["storage_state"])
        path.write_text('bad private-cookie')
        with self.assertRaises(ValueError) as raised:
            prepare_worker_config(self.config, self.root / "worker")
        self.assertNotIn("private-cookie", str(raised.exception))
        self.assertEqual(path.read_text(), 'bad private-cookie')

    def test_invalid_owner_token_does_not_write_worker_config(self):
        (self.source / "token.json").write_text('{"refresh_token": "private-refresh"}')
        with self.assertRaisesRegex(ValueError, "Invalid owner token structure"):
            prepare_worker_config(self.config, self.root / "worker")
        self.assertFalse((self.root / "worker/config.json").exists())

    def test_aliases_do_not_mutate_source(self):
        source = self.source / "browser.json"
        original = source.read_bytes()
        with self.assertRaises(ValueError):
            ensure_worker_storage_state(source, source)
        alias = self.root / "alias.json"
        alias.symlink_to(source)
        with self.assertRaises(ValueError):
            ensure_worker_storage_state(source, alias)
        alias.unlink()
        alias.hardlink_to(source)
        with self.assertRaises(ValueError):
            ensure_worker_storage_state(source, alias)
        self.assertEqual(source.read_bytes(), original)

    def test_concurrent_bootstrap_preserves_single_complete_copy(self):
        destination = self.root / "worker/browser-state.json"
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: ensure_worker_storage_state(self.source / "browser.json", destination), range(8)))
        self.assertEqual(results, [destination] * 8)
        self.assertEqual(json.loads(destination.read_text()), self.state)
        self.assertEqual(destination.stat().st_nlink, 1)
        self.assertEqual(list(destination.parent.glob("*.tmp")), [])

if __name__ == "__main__":
    unittest.main()
