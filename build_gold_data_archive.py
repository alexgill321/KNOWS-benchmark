#!/usr/bin/env python3
"""Build the gold-data release asset that the PyPI wheel downloads at runtime.

The wheel excludes every ``instance_N/data/`` directory (see pyproject.toml);
this script zips exactly those trees, taking only git-tracked files so
runtime scratch output and ``.DS_Store`` never leak in. Upload the result to
the KNOWS-benchmark GitHub release whose tag matches the package version:

    python build_gold_data_archive.py            # -> dist/knows-gold-data-v<ver>.zip
    gh release upload v<ver> dist/knows-gold-data-v<ver>.zip -R alexgill321/KNOWS-benchmark

Archive layout mirrors the package: ``<family>/instance_N/data/...``.

Release artifacts are built from a clone of the public repo, never from the
private Agent-Benchmark checkout, so this refuses to run unless the git
remote is KNOWS-benchmark (override with --allow-any-remote for rehearsals).
"""

import argparse
import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
TASKS_DIR = REPO_ROOT / "src" / "browsergym" / "knows" / "eval" / "tasks"
sys.path.insert(0, str(REPO_ROOT / "src" / "browsergym" / "knows" / "eval" / "eval_utils"))
from data_paths import package_version  # noqa: E402


def tracked_data_files() -> list:
    """Return git-tracked files under any ``instance_N/data/`` directory."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", str(TASKS_DIR)],
        cwd=REPO_ROOT, check=True, capture_output=True,
    ).stdout
    files = []
    for rel in out.decode().split("\0"):
        if not rel:
            continue
        parts = Path(rel).parts
        try:
            i = parts.index("tasks")
        except ValueError:
            continue
        # tasks/<family>/instance_N/data/...
        if len(parts) > i + 3 and parts[i + 2].startswith("instance_") and parts[i + 3] == "data":
            files.append(REPO_ROOT / rel)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "dist")
    parser.add_argument("--version", default=None, help="override the package version")
    parser.add_argument("--allow-any-remote", action="store_true",
                        help="skip the public-repo remote check (rehearsals only)")
    args = parser.parse_args()

    remote = subprocess.run(["git", "remote", "get-url", "origin"], cwd=REPO_ROOT,
                            capture_output=True, text=True).stdout.strip()
    if "KNOWS-benchmark" not in remote and not args.allow_any_remote:
        print(f"Refusing to build from a non-public checkout (origin={remote!r}). "
              "Release artifacts are built from the public repo; see RELEASING.md.", file=sys.stderr)
        return 2

    version = args.version or package_version()
    files = tracked_data_files()
    if not files:
        print("No tracked data files found; is this a source checkout?", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / f"knows-gold-data-v{version}.zip"
    instances = set()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for n, path in enumerate(files, 1):
            arcname = path.relative_to(TASKS_DIR).as_posix()
            instances.add("/".join(arcname.split("/")[:2]))
            zf.write(path, arcname)
            if n % 100 == 0 or n == len(files):
                print(f"Archive {n * 100 // len(files)}%", file=sys.stderr)

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (archive.with_suffix(".zip.sha256")).write_text(f"{digest}  {archive.name}\n")
    print(f"{archive}  {archive.stat().st_size / 1048576:.1f} MB  "
          f"{len(files)} files  {len(instances)} instances  sha256={digest[:12]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
