"""Per-run write targets for tasks that ask the agent to upload files.

Most KNOWS tasks only read from Drive, so their prompts can name a hosted
view-only asset that ships with the benchmark. Two families are different:
``sheets_10_paper_sorting`` and ``slides_17_removeimagesaddplaceholders`` ask
the agent to *write* into Drive. A write destination cannot be shared between
users -- each run needs a folder its own Google account owns.

Those prompts therefore carry ``{{PLACEHOLDER}}`` tokens instead of URLs. This
module resolves a token to the location the current user provisioned, so no
account-specific URL is ever committed:

* the harness substitutes tokens into the goal text before the agent sees it
  (:func:`resolve_prompt`), and
* evaluators look up the same location when grading
  (:func:`target_folder_id`).

Provision the locations with ``provision_run_targets.py``; see ASSETS.md.

Resolution order for each token:

1. environment variable ``KNOWS_TARGET_<FAMILY>_<INSTANCE>_<TOKEN>``
2. the JSON config at ``$KNOWS_RUN_TARGETS`` (default ``run_targets.json`` in
   the repository root)

The config maps family -> instance -> token -> URL::

    {
      "sheets_10_paper_sorting": {
        "instance_1": {"OUTPUT_FOLDER_URL": "https://drive.google.com/drive/folders/..."}
      }
    }
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG_NAME = "run_targets.json"

PLACEHOLDER_RE = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
_DRIVE_ID_RE = re.compile(r"/(?:d|folders)/([A-Za-z0-9_-]{10,})")

#: Tokens each family/instance needs. ``"*"`` applies to any instance without a
#: more specific entry. Provisioning and validation both read this table, so
#: adding a token in one place is enough.
REQUIRED_TARGETS: Dict[str, Dict[str, tuple]] = {
    "sheets_10_paper_sorting": {
        # One folder that must contain `pdfs/` and `figures/` subfolders.
        "*": ("OUTPUT_FOLDER_URL",),
    },
    "slides_17_removeimagesaddplaceholders": {
        # Instance 1 hands the agent a ready-made copy to edit; the others ask
        # the agent to make the copy itself and drop it in a folder.
        "instance_1": ("WORKING_COPY_URL", "IMAGES_FOLDER_URL"),
        "*": ("OUTPUT_FOLDER_URL", "IMAGES_FOLDER_URL"),
    },
}


class MissingRunTarget(RuntimeError):
    """A prompt or evaluator needs a write target that is not provisioned."""


def required_tokens(family: str, instance: str) -> tuple:
    """Tokens *family*/*instance* needs, or ``()`` if it needs none."""
    by_instance = REQUIRED_TARGETS.get(family)
    if not by_instance:
        return ()
    return by_instance.get(instance, by_instance.get("*", ()))


def needs_provisioning(family: str) -> bool:
    """True if *family* writes into Drive and must be provisioned per run."""
    return family in REQUIRED_TARGETS


def config_path() -> Path:
    """Where the run-target config lives."""
    override = os.environ.get("KNOWS_RUN_TARGETS", "").strip()
    return Path(override).expanduser() if override else REPO_ROOT / DEFAULT_CONFIG_NAME


def load_targets() -> dict:
    """Read the config, or ``{}`` when it does not exist yet."""
    path = config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MissingRunTarget(f"Could not read run-target config {path}: {exc}") from exc


def _env_key(family: str, instance: str, token: str) -> str:
    return f"KNOWS_TARGET_{family}_{instance}_{token}".upper()


def get_target(family: str, instance: str, token: str) -> str:
    """Resolve one token to its URL.

    Raises:
        MissingRunTarget: if neither an environment override nor the config
            supplies the token, with the command that would fix it.
    """
    env_value = os.environ.get(_env_key(family, instance, token), "").strip()
    if env_value:
        return env_value

    value = (load_targets().get(family, {}).get(instance, {}) or {}).get(token)
    if not value:
        raise MissingRunTarget(
            f"No write target for {family}/{instance} token {{{{{token}}}}}.\n"
            f"This task uploads files to your Google Drive, so it needs a folder "
            f"you own. Provision one with:\n"
            f"    python src/browsergym/knows/eval/tasks/provision_run_targets.py "
            f"--family {family} --instance {instance.split('_')[-1]}\n"
            f"or set {_env_key(family, instance, token)}. "
            f"Config searched: {config_path()}"
        )
    return value


def save_targets(family: str, instance: str, values: Dict[str, str]) -> Path:
    """Merge *values* into the config for *family*/*instance* and write it.

    Returns:
        Path: the config file written.
    """
    path = config_path()
    config = load_targets()
    config.setdefault(family, {}).setdefault(instance, {}).update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def drive_id(url_or_id: str) -> str:
    """Extract the Drive/Docs id from a URL, or pass a bare id through.

    Raises:
        MissingRunTarget: if the value looks like a URL but no id can be read
            from it. Returning the URL instead would surface much later as an
            opaque Drive API error.
    """
    value = url_or_id.strip()
    match = _DRIVE_ID_RE.search(value)
    if match:
        return match.group(1)
    if value.lower().startswith(("http://", "https://")):
        raise MissingRunTarget(
            f"Could not read a Drive id from {value!r}. Expected a URL like "
            f"https://drive.google.com/drive/folders/<id> or "
            f"https://docs.google.com/presentation/d/<id>/edit."
        )
    return value


def target_folder_id(family: str, instance: str, token: str) -> str:
    """Drive id of a provisioned write target, for evaluators."""
    return drive_id(get_target(family, instance, token))


def resolve_prompt(text: str, family: str, instance: str) -> str:
    """Substitute every ``{{TOKEN}}`` in *text* with its provisioned URL.

    Text with no tokens is returned unchanged, so this is safe to call for
    every task.
    """
    if "{{" not in text:
        return text
    return PLACEHOLDER_RE.sub(
        lambda m: get_target(family, instance, m.group(1)), text
    )


def missing_tokens(family: str, instance: str) -> Iterable[str]:
    """Tokens *family*/*instance* still needs. Empty when ready to run."""
    missing = []
    for token in required_tokens(family, instance):
        try:
            get_target(family, instance, token)
        except MissingRunTarget:
            missing.append(token)
    return missing
