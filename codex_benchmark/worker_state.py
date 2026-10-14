"""Private worker credential copies; bootstrapping never contacts a browser."""
from __future__ import annotations

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import tempfile


def _read_valid_json(path: Path, kind: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise ValueError(f"Cannot read valid {kind} JSON") from None
    valid = isinstance(value, dict)
    if valid and kind == "browser state":
        valid = all(isinstance(value.get(key), list) for key in ("cookies", "origins"))
        if valid:
            valid = all(isinstance(cookie, dict) and all(isinstance(cookie.get(key), str)
                        for key in ("name", "value", "domain", "path")) for cookie in value["cookies"])
            valid = valid and all(isinstance(origin, dict) and isinstance(origin.get("origin"), str)
                and isinstance(origin.get("localStorage"), list)
                and all(isinstance(item, dict) and all(isinstance(item.get(key), str) for key in ("name", "value"))
                        for item in origin["localStorage"]) for origin in value["origins"])
    elif valid and kind == "owner token":
        # Access-token expiry is intentionally allowed: workers refresh their
        # own copy. Bootstrapping never requests new consent or validates online.
        valid = all(isinstance(value.get(key), str) and bool(value[key])
                    for key in ("refresh_token", "client_id", "client_secret", "token_uri"))
    if not valid:
        raise ValueError(f"Invalid {kind} structure")
    return value


def _private_directory(path: Path) -> Path:
    path = path.expanduser().absolute()
    if path.is_symlink():
        raise ValueError("Worker directory must not be a symbolic link")
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)
    return path.resolve()


def _write_temporary_json(path: Path, value: dict) -> Path:
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _ensure_private_copy(source: str | Path, destination: str | Path, kind: str) -> Path:
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().absolute()
    if destination.is_symlink():
        raise ValueError("Worker credential must not be a symbolic link")
    destination = destination.parent.resolve() / destination.name
    if source == destination or (source.exists() and destination.exists() and source.samefile(destination)):
        raise ValueError("Worker credential must be separate from its source")
    _private_directory(destination.parent)

    def existing() -> None:
        if destination.is_symlink() or destination.stat().st_nlink != 1:
            raise ValueError("Worker credential must be an independent regular file")
        _read_valid_json(destination, kind)
        destination.chmod(0o600)

    lock_descriptor = os.open(destination.with_name(f".{destination.name}.bootstrap.lock"),
                              os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        if destination.exists():
            existing()
            return destination
        value = _read_valid_json(source, kind)
        temporary = _write_temporary_json(destination, value)
        try:
            try:
                # An exclusive atomic install: a racing bootstrap cannot
                # overwrite state already refreshed by another process.
                os.link(temporary, destination)
            except FileExistsError:
                existing()
        finally:
            temporary.unlink(missing_ok=True)
    finally:
        os.close(lock_descriptor)
    return destination


def ensure_worker_storage_state(source: str | Path, destination: str | Path) -> Path:
    """Create one private state copy, preserving any valid existing worker copy."""
    return _ensure_private_copy(source, destination, "browser state")


def prepare_worker_config(config: dict, worker_dir: str | Path) -> dict:
    """Save a worker config with independent browser/OAuth state and absolute paths.

    Model settings, tool settings, and the source configuration are unchanged.
    Call after the serial runner drains, before launching worker subprocesses.
    Returned configuration contains credentials paths, never credential values.
    """
    source_directory = Path(config["config_path"]).expanduser().resolve().parent
    worker_directory = _private_directory(Path(worker_dir))
    prepared = deepcopy(config)
    for key in ("benchmark_root", "storage_state", "service_account_path", "token_path",
                "credentials_path", "output_dir", "python"):
        if prepared.get(key):
            path = Path(prepared[key]).expanduser()
            prepared[key] = str((path if path.is_absolute() else source_directory / path).resolve())
    if "env_files" in prepared:
        prepared["env_files"] = [str((path if path.is_absolute() else source_directory / path).resolve())
                                 for path in (Path(name).expanduser() for name in prepared["env_files"])]
    prepared["storage_state"] = str(ensure_worker_storage_state(
        prepared["storage_state"], worker_directory / "browser-state.json"))
    prepared["token_path"] = str(_ensure_private_copy(
        prepared["token_path"], worker_directory / "owner-token.json", "owner token"))
    destination = worker_directory / "config.json"
    if destination.is_symlink():
        raise ValueError("Worker config must not be a symbolic link")
    prepared["config_path"] = str(destination)
    temporary = _write_temporary_json(destination, prepared)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return prepared
