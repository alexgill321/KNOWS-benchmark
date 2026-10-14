"""Durable evaluator progress, installed only in the isolated grading worker."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from functools import wraps
import importlib
import json
import os
from pathlib import Path
from threading import RLock
from uuid import uuid4

from .config import write_json


def save_evaluation_report(path, report):
    """Archive every report before replacing the latest, including timeouts."""
    path = Path(path)
    attempts = path.parent / "evaluation-attempts"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous_id = previous.get("evaluation_attempt_id") or "legacy-" + str(uuid4())
        previous_path = attempts / (previous_id + ".json")
        if not previous_path.exists():
            write_json(previous_path, previous)
    report.setdefault("evaluation_attempt_id", str(uuid4()))
    write_json(attempts / (report["evaluation_attempt_id"] + ".json"), report)
    write_json(path, report)


class EvaluationJournal:
    """Append each completed rubric step before grading proceeds.

    The final snapshots retain checkpoint adjustments and directly supplied
    steps as well. Multiple attempts share one journal but have distinct IDs.
    Patching is local to the evaluator process and always undone on exit.
    """

    def __init__(self, path, task_id, workspace_id, *, scoring_module=None):
        self.path = Path(path).expanduser().resolve()
        self.task_id = task_id
        self.workspace_id = workspace_id
        self.attempt_id = str(uuid4())
        self._scoring = scoring_module
        self._checkpoints = []
        self._indices = {}
        self._lock = RLock()
        self._stream = None
        self._sequence = 0

    @property
    def completed_steps(self):
        return sum(len(checkpoint.steps) for checkpoint in self._checkpoints)

    @property
    def checkpoints(self):
        return [{"checkpoint_index": index, **asdict(checkpoint)}
                for index, checkpoint in enumerate(self._checkpoints, 1)]

    @property
    def report(self):
        with self._lock:
            return {"attempt_id": self.attempt_id, "path": str(self.path),
                    "completed_steps": self.completed_steps,
                    "checkpoint_count": len(self._checkpoints),
                    "checkpoints": self.checkpoints}

    def _write(self, event, **fields):
        # The lock also prevents interleaved JSON lines from evaluator threads.
        with self._lock:
            self._sequence += 1
            record = {"event": event, "attempt_id": self.attempt_id,
                      "sequence": self._sequence,
                      "timestamp": datetime.now(timezone.utc).isoformat(),
                      "task_id": self.task_id, "workspace_id": self.workspace_id,
                      **fields}
            self._stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())

    def _register(self, checkpoint):
        with self._lock:
            key = id(checkpoint)
            if key not in self._indices:
                # Retain objects so Python cannot reuse a checkpoint ID.
                self._checkpoints.append(checkpoint)
                self._indices[key] = len(self._checkpoints)
                self._write("checkpoint_registered", checkpoint_index=self._indices[key],
                            checkpoint=asdict(checkpoint))
            return self._indices[key]

    def __enter__(self):
        if self._stream is not None:
            raise RuntimeError("An evaluation journal cannot be entered twice")
        scoring = self._scoring or importlib.import_module("browsergym.knows.eval.eval_utils.scoring")
        self._checkpoint_class = scoring.Checkpoint
        self._original_init = scoring.Checkpoint.__post_init__
        self._original_add = scoring.Checkpoint.add_step
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8")
        try:
            self._write("evaluation_started")
        except BaseException:
            self._stream.close()
            raise

        @wraps(self._original_init)
        def registered(checkpoint, *args, **kwargs):
            value = self._original_init(checkpoint, *args, **kwargs)
            self._register(checkpoint)
            return value

        @wraps(self._original_add)
        def completed(checkpoint, *args, **kwargs):
            with self._lock:
                index = self._register(checkpoint)
                step = self._original_add(checkpoint, *args, **kwargs)
                self._write("step_completed", checkpoint_index=index,
                            checkpoint_name=checkpoint.name, step=asdict(step),
                            checkpoint_result=checkpoint.result,
                            checkpoint_total=checkpoint.total)
                return step

        scoring.Checkpoint.__post_init__ = registered
        scoring.Checkpoint.add_step = completed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            for snapshot in self.checkpoints:
                self._write("checkpoint_snapshot", checkpoint_index=snapshot["checkpoint_index"],
                            checkpoint={key: value for key, value in snapshot.items()
                                        if key != "checkpoint_index"})
            self._write("evaluation_error" if exc_type else "evaluation_finished",
                        completed_steps=self.completed_steps, checkpoint_count=len(self._checkpoints),
                        **({"error": f"{exc_type.__name__}: {exc_value}"} if exc_type else {}))
        finally:
            self._checkpoint_class.__post_init__ = self._original_init
            self._checkpoint_class.add_step = self._original_add
            self._stream.close()
        return False
