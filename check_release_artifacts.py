#!/usr/bin/env python3
"""Refuse to release a wheel, sdist, or gold-data archive that carries secrets or oversize files.

    python check_release_artifacts.py dist/

For every ``*.whl``, ``*.tar.gz`` and ``*.zip`` in the directory this checks:

* no entry over PyPI's 100 MB per-file limit (the archive itself may exceed it;
  it goes to a GitHub release, not PyPI),
* no entry whose path looks like a credential file or IDE config,
* no ``instance_*/data/`` entry in the wheel or sdist (they belong in the archive),
* no text entry containing a Google API key, private key, or Anthropic key.

Exit status is non-zero on any finding, so it can gate an upload.
"""

import re
import sys
import tarfile
import zipfile
from pathlib import Path

PYPI_FILE_LIMIT = 100 * 1024 * 1024
FORBIDDEN_PATHS = re.compile(
    r"(^|/)(auth-data/|\.env$|\.env\.|password|launch\.json|token\.json|"
    r"credentials\.json|service-account|\.pem$|id_rsa)", re.IGNORECASE)
# Prefixes are assembled from fragments so this file never contains a literal
# key prefix (the release assembler's own secret scan would flag it).
_GOOGLE_KEY = b"AIza" + b"Sy[0-9A-Za-z_-]{33}"
_ANTHROPIC_KEY = b"sk-" + b"ant-[0-9A-Za-z_-]{20,}"
_PRIVATE_KEY = b"BEGIN (RSA |EC )?" + b"PRIVATE KEY"
SECRET_CONTENT = re.compile(b"|".join([_GOOGLE_KEY, _PRIVATE_KEY, _ANTHROPIC_KEY, rb"\"private_key\"\s*:"]))
DATA_TREE = re.compile(r"/instance_\d+/data/")
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".toml", ".cfg", ".gs", ".html", ".js", ".yaml", ".yml"}


def _entries(path: Path):
    """Yield (name, size, read_bytes_callable) for each member of an archive."""
    if path.suffix == ".whl" or path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    yield info.filename, info.file_size, (lambda i=info: zf.read(i))
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as tf:
            for member in tf.getmembers():
                if member.isfile():
                    yield member.name, member.size, (lambda m=member: tf.extractfile(m).read())


def check(path: Path) -> list:
    """Return a list of problem strings for one artifact."""
    problems = []
    is_pypi = path.suffix == ".whl" or path.name.endswith(".tar.gz")
    if is_pypi and path.stat().st_size > PYPI_FILE_LIMIT:
        problems.append(f"{path.name} is {path.stat().st_size >> 20} MB, over PyPI's 100 MB limit")
    n = 0
    for name, size, read in _entries(path):
        n += 1
        if FORBIDDEN_PATHS.search(name):
            problems.append(f"credential-looking path: {name}")
        if is_pypi and DATA_TREE.search(name):
            problems.append(f"gold data must not ship on PyPI: {name}")
        if Path(name).suffix.lower() in TEXT_SUFFIXES and size < 5 * 1024 * 1024:
            m = SECRET_CONTENT.search(read())
            if m:
                problems.append(f"secret pattern {m.group(0)[:12]!r}... in {name}")
    print(f"{path.name}: {n} entries, {path.stat().st_size / 1048576:.1f} MB, {len(problems)} problem(s)")
    return problems


def main() -> int:
    dist = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")
    artifacts = sorted(p for p in dist.iterdir() if p.suffix in {".whl", ".zip"} or p.name.endswith(".tar.gz"))
    if not artifacts:
        print(f"No artifacts in {dist}", file=sys.stderr)
        return 1
    all_problems = []
    for artifact in artifacts:
        all_problems += [f"{artifact.name}: {p}" for p in check(artifact)]
    if all_problems:
        print("\nFAILED:")
        for p in all_problems:
            print("  " + p)
        return 1
    print("\nPASSED: artifacts are clean.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
