"""Locate gold evaluation data, downloading it on first use when installed from PyPI.

Seventy-three task instances keep gold assets (reference images, CSVs,
JSON) in an ``instance_N/data/`` directory next to their ``evaluator.py``.
Those directories total ~225 MB, far beyond PyPI's 100 MB per-file limit, so
the ``browsergym-knows`` wheel ships without them and fetches one archive at
runtime instead.

Two layouts are therefore supported:

* **Source checkout** (git clone, the public KNOWS-benchmark repo, the
  BrowserGym-Knows fork): ``data/`` sits beside each evaluator and nothing is
  downloaded.
* **Installed wheel**: no instance has a ``data/`` directory. The gold-data
  archive for this package version is downloaded once into a cache and each
  instance is materialised there as ``<cache>/<family>/instance_N/`` holding
  the extracted ``data/`` plus a copy of the instance's small non-code files
  (``task.md``, ``id.txt``, ...) so evaluator code that reads
  ``TASK_DIR/<file>`` keeps working unchanged.

Configuration (environment variables):

``KNOWS_DATA_DIR``
    Cache root. Default ``~/.cache/browsergym-knows/gold-data``. The archive
    for version *V* is extracted into ``<root>/<V>/``.
``KNOWS_DATA_URL``
    Archive URL. Default is the GitHub release asset for this package version,
    :data:`DEFAULT_URL_TEMPLATE`. Point it at a local HTTP server to test the
    download path offline.

Public entry points: :func:`ensure_gold_data`, :func:`resolve_instance_dir`,
:func:`resolve_task_dir`.
"""

import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import requests

_PACKAGE_DIR = Path(__file__).resolve().parents[2]  # browsergym/knows/
_TASKS_DIR = _PACKAGE_DIR / "eval" / "tasks"

DEFAULT_URL_TEMPLATE = (
    "https://github.com/alexgill321/KNOWS-benchmark/releases/download/"
    "v{version}/knows-gold-data-v{version}.zip"
)
DEFAULT_CACHE_ROOT = Path("~/.cache/browsergym-knows/gold-data").expanduser()
_COMPLETE_MARKER = ".complete"

# Files copied from the package instance dir into the cache instance dir so
# ``TASK_DIR``-relative reads of non-data files resolve there too. Code and
# development notes are deliberately left out.
_INSTANCE_FILES_TO_SYNC = {"task.md", "checkpoints.md", "id.txt", "gold_instances.csv", "room_type.txt"}


def package_version() -> str:
    """Return the ``browsergym.knows`` version without importing the package.

    Reads ``__version__`` from ``browsergym/knows/__init__.py`` directly, which
    avoids a circular import (``task.py`` imports this module at import time).

    Returns:
        str: Version string such as ``"1.1.0"``.

    Raises:
        RuntimeError: If no ``__version__`` line is found.
    """
    init_text = (_PACKAGE_DIR / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', init_text, re.MULTILINE)
    if not match:
        raise RuntimeError(f"No __version__ in {_PACKAGE_DIR / '__init__.py'}")
    return match.group(1)


def gold_data_url(version: Optional[str] = None) -> str:
    """Return the archive URL for *version*, honouring ``KNOWS_DATA_URL``."""
    override = os.environ.get("KNOWS_DATA_URL")
    if override:
        return override
    return DEFAULT_URL_TEMPLATE.format(version=version or package_version())


def cache_dir(version: Optional[str] = None) -> Path:
    """Return the directory the archive for *version* is extracted into."""
    root = Path(os.environ.get("KNOWS_DATA_DIR") or DEFAULT_CACHE_ROOT).expanduser()
    return root / (version or package_version())


def package_has_gold_data() -> bool:
    """True when any bundled instance still has its ``data/`` directory.

    This distinguishes a source checkout (data beside the evaluators) from an
    installed wheel (data stripped at build time). Instances that never had
    gold data must not trigger a download in a checkout, so the check is
    made across the whole tasks tree rather than per instance.
    """
    return any(p.is_dir() for p in _TASKS_DIR.glob("*/instance_*/data"))


def _download(url: str, dest: Path) -> None:
    """Stream *url* to *dest*, printing coarse progress.

    Args:
        url (str): Archive URL.
        dest (Path): File to write; parent directory must exist.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        next_report = 10
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total and done * 100 // total >= next_report:
                    print(f"[knows] Download {done * 100 // total}%", file=sys.stderr)
                    next_report += 10


def ensure_gold_data(version: Optional[str] = None, force: bool = False) -> Path:
    """Make sure the gold-data archive for *version* is extracted in the cache.

    Idempotent: a completed extraction is detected by a marker file and
    returned immediately. Extraction happens in a sibling temp directory that
    is renamed into place at the end, so an interrupted run never leaves a
    half-populated cache that looks complete.

    Args:
        version (str, optional): Package version; defaults to the installed one.
        force (bool): Re-download even if the cache is complete.

    Returns:
        Path: Cache directory containing ``<family>/instance_N/data/`` trees.

    Raises:
        requests.HTTPError: If the archive cannot be downloaded.
        zipfile.BadZipFile: If the download is not a valid zip.
    """
    version = version or package_version()
    target = cache_dir(version)
    if target.joinpath(_COMPLETE_MARKER).is_file() and not force:
        return target

    url = gold_data_url(version)
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[knows] Fetching gold data v{version} from {url}", file=sys.stderr)
    staging = Path(tempfile.mkdtemp(prefix=f"{version}.partial-", dir=target.parent))
    try:
        archive = staging / "gold-data.zip"
        _download(url, archive)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                # Refuse anything that would escape the staging dir.
                if member.startswith("/") or ".." in Path(member).parts:
                    raise zipfile.BadZipFile(f"Unsafe path in archive: {member}")
            zf.extractall(staging)
        archive.unlink()
        staging.joinpath(_COMPLETE_MARKER).write_text(url + "\n", encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"[knows] Gold data ready at {target}", file=sys.stderr)
    return target


def _sync_instance_files(package_instance_dir: Path, cache_instance_dir: Path) -> None:
    """Copy small non-code instance files into the cache when missing or stale."""
    for name in _INSTANCE_FILES_TO_SYNC:
        src = package_instance_dir / name
        if not src.is_file():
            continue
        dst = cache_instance_dir / name
        if dst.is_file() and dst.stat().st_size == src.stat().st_size and dst.stat().st_mtime >= src.stat().st_mtime:
            continue
        shutil.copy2(src, dst)


def resolve_instance_dir(package_instance_dir: Path) -> Path:
    """Return the directory an evaluator should treat as its ``TASK_DIR``.

    Args:
        package_instance_dir (Path): The ``instance_N`` directory inside the
            installed package (or checkout) that holds ``evaluator.py``.

    Returns:
        Path: *package_instance_dir* itself in a source checkout, or for an
        instance that has no gold data. Otherwise the cache instance directory
        holding the extracted ``data/`` plus synced ``task.md`` etc.
    """
    package_instance_dir = Path(package_instance_dir).resolve()
    if package_has_gold_data():
        return package_instance_dir

    family = package_instance_dir.parent.name
    instance = package_instance_dir.name
    cache_instance_dir = ensure_gold_data() / family / instance
    if not (cache_instance_dir / "data").is_dir():
        return package_instance_dir  # this instance ships no gold data
    _sync_instance_files(package_instance_dir, cache_instance_dir)
    return cache_instance_dir


def resolve_task_dir(evaluator_file: str) -> str:
    """Convenience for evaluator modules: ``TASK_DIR = resolve_task_dir(__file__)``.

    Returns:
        str: Resolved instance directory with a trailing separator, matching
        the form the legacy ``TASK_DIR`` constants used.
    """
    return str(resolve_instance_dir(Path(evaluator_file).parent)) + os.sep
