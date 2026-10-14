import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from codex_benchmark.knows import TASKS_RELATIVE, find_task
from codex_benchmark.targets import provision_targets


class PrivateTargetsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.owner = MagicMock()
        self.evaluator = MagicMock()
        self.created = []
        self.owner.about.return_value.get.return_value.execute.return_value = {
            "user": {"emailAddress": "owner@example.test"}}
        self.evaluator.files.return_value.get.return_value.execute.return_value = {
            "id": "file", "capabilities": {"canEdit": True}}
        self.owner.permissions.return_value.list.return_value.execute.return_value = {"permissions": []}

        def create(**kwargs):
            identifier = f"file{len(self.created) + 1}"
            self.created.append({"id": identifier, **kwargs})
            return SimpleNamespace(execute=lambda: {"id": identifier})
        self.owner.files.return_value.create.side_effect = create
        self.owner.files.return_value.copy.side_effect = create
        self.clients = patch("codex_benchmark.targets._build_clients", return_value=(
            self.owner, self.evaluator, "judge@project.iam.gserviceaccount.com"))
        self.clients.start()
        self.addCleanup(self.clients.stop)

    def task(self, family, instance=1):
        directory = self.root / TASKS_RELATIVE / family / f"instance_{instance}"
        directory.mkdir(parents=True)
        (directory / "evaluator.py").write_text("")
        (directory / "task.md").write_text("Use https://docs.google.com/presentation/d/source123/edit")
        return find_task(self.root, f"knows.{family}.{instance}")

    def test_sheets_schema_is_private_and_output_points_to_parent(self):
        task = self.task("sheets_10_paper_sorting")
        output = self.root / "episode/run_targets.json"
        result = provision_targets({}, task, output)
        self.assertEqual([item["body"]["name"] for item in self.created[1:]],
                         [task.family, "instance_1", "run_0001", "pdfs", "figures"])
        self.assertNotIn("parents", self.created[0]["body"])
        self.assertEqual(result[task.family]["instance_1"]["OUTPUT_FOLDER_URL"],
                         "https://drive.google.com/drive/folders/file4")
        permissions = self.owner.permissions.return_value.create.call_args_list
        self.assertEqual(len(permissions), 1)
        self.assertEqual(permissions[0].kwargs["body"], {
            "type": "user", "role": "writer", "emailAddress": "judge@project.iam.gserviceaccount.com"})
        self.assertFalse(permissions[0].kwargs["sendNotificationEmail"])
        metadata = json.loads(output.with_suffix(".metadata.json").read_text())
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(len(metadata["created_files"]), 6)
        with self.assertRaises(FileExistsError):
            provision_targets({}, task, output)

    def test_slides1_copies_source_and_verifies_evaluator_can_edit(self):
        task = self.task("slides_17_removeimagesaddplaceholders")
        result = provision_targets({}, task, self.root / "targets.json")
        tokens = result[task.family]["instance_1"]
        self.assertEqual(tokens["WORKING_COPY_URL"], "https://docs.google.com/presentation/d/file6/edit")
        self.assertEqual(self.owner.files.return_value.copy.call_args.kwargs["fileId"], "source123")
        accessed = [call.kwargs["fileId"] for call in self.evaluator.files.return_value.get.call_args_list]
        self.assertIn("file6", accessed)
        self.assertIn("file4", accessed)

    def test_slides2_leaves_empty_copy_folder_for_agent(self):
        task = self.task("slides_17_removeimagesaddplaceholders", 2)
        result = provision_targets({}, task, self.root / "targets.json")
        tokens = result[task.family]["instance_2"]
        self.assertNotIn("WORKING_COPY_URL", tokens)
        self.assertTrue(tokens["OUTPUT_FOLDER_URL"].endswith("file6"))
        self.assertEqual(self.created[-1]["body"]["name"], "copies")
        self.owner.files.return_value.copy.assert_not_called()

    def test_partial_failure_trashes_only_new_root_and_preserves_ids(self):
        task = self.task("sheets_10_paper_sorting")
        output = self.root / "targets.json"
        self.owner.permissions.return_value.create.return_value.execute.side_effect = RuntimeError("sharing unavailable")
        with self.assertRaisesRegex(RuntimeError, "sharing unavailable"):
            provision_targets({}, task, output)
        self.owner.files.return_value.update.assert_called_once_with(
            fileId="file1", body={"trashed": True}, fields="id,trashed")
        metadata = json.loads(output.with_suffix(".metadata.json").read_text())
        self.assertEqual(metadata["status"], "error")
        self.assertEqual(metadata["created_files"][0]["id"], "file1")
        self.assertEqual(metadata["cleanup"]["status"], "root_trashed")
        self.assertFalse(output.exists())

    def test_owner_auth_failure_stops_before_any_drive_mutation(self):
        task = self.task("sheets_10_paper_sorting")
        output = self.root / "targets.json"
        with patch("codex_benchmark.targets._build_clients", side_effect=RuntimeError("Owner OAuth expired; reauthorize")):
            with self.assertRaisesRegex(RuntimeError, "reauthorize"):
                provision_targets({}, task, output)
        self.assertEqual(self.created, [])
        self.owner.permissions.return_value.create.assert_not_called()
        metadata = json.loads(output.with_suffix(".metadata.json").read_text())
        self.assertEqual(metadata["cleanup"]["status"], "no_created_root")


if __name__ == "__main__":
    unittest.main()
