"""Helpers to pre-create a Google Doc for a knows benchmark task.

The task ``setup()`` calls :func:`create_task_doc` with the same Playwright
``page`` it received from BrowserGym. Because the surrounding browser context
already has the user's Google ``storage_state`` applied, the helper can drive
the standard Google Docs UI (create + rename) without any additional auth
plumbing. The captured doc id is then handed straight to the evaluator,
removing the need to parse it out of the agent's "DONE" message.

A small ``__main__`` is included so the helper can be exercised manually on a
fresh Playwright context for debugging:

.. code:: bash

    python -m browsergym.knows.doc_setup \\
        --storage-state storage_state.json \\
        --agent-name "GenericAgent-gpt-5.4-2026-03-05" \\
        --task-name "knows.docs_1_formal_letter" \\
        --instance-id 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional, Tuple

import playwright.sync_api

logger = logging.getLogger(__name__)


def _debug_progress(message: str) -> None:
    if os.environ.get("AGENTLAB_DEBUG_PROGRESS", "").strip():
        print(f"[doc-setup-progress] {message}", flush=True)


# Default location of the credentials shipped with the package.
# ``share_doc_with_service_account`` authenticates with the evaluator's
# service-account JSON. The OAuth paths remain for the manual reauth CLI helper.
_AUTH_DIR = Path(__file__).resolve().parent.parent.parent.parent / "auth-data"
_DEFAULT_TOKEN_PATH = _AUTH_DIR / "token.json"
_DEFAULT_CLIENT_SECRETS_PATH = _AUTH_DIR / "credentials.json"
_DEFAULT_SERVICE_ACCOUNT_PATH = _AUTH_DIR / "service-account.json"

# Maximum length for a Google Docs title — Drive technically allows a few
# hundred characters, but keeping it shorter avoids DOM-side ellipsis and
# filesystem issues if the name is reused elsewhere.
_MAX_DOC_NAME_LEN = 180

# Patterns reused across this module.
_DOC_ID_RE = re.compile(r"/document/d/([a-zA-Z0-9_-]+)")
_TASK_INSTANCE_SUFFIX_RE = re.compile(r"\.(\d+)$")

# Per-workspace-kind metadata: ``create_url`` is what we navigate to in
# order to spawn a fresh file, ``id_re`` is used to extract the file id
# from the resulting URL, and ``kind_label`` is the human-readable name
# used in log messages.
_WORKSPACE_KINDS = {
    "docs": {
        "create_url": "https://docs.google.com/document/create",
        "url_segment": "document",
        "id_re": re.compile(r"/document/d/([a-zA-Z0-9_-]+)"),
        "kind_label": "Google Doc",
    },
    "sheets": {
        "create_url": "https://docs.google.com/spreadsheets/create",
        "url_segment": "spreadsheets",
        "id_re": re.compile(r"/spreadsheets/d/([a-zA-Z0-9_-]+)"),
        "kind_label": "Google Sheet",
    },
    "slides": {
        "create_url": "https://docs.google.com/presentation/create",
        "url_segment": "presentation",
        "id_re": re.compile(r"/presentation/d/([a-zA-Z0-9_-]+)"),
        "kind_label": "Google Slides",
    },
}


def sanitize_doc_name(raw: str) -> str:
    """Replace characters that are awkward in doc names / file paths.

    Spaces, dots, and slashes become ``_``; control characters are dropped.
    The result is truncated to :data:`_MAX_DOC_NAME_LEN`.
    """
    if raw is None:
        return ""

    cleaned_chars = []
    for ch in str(raw):
        if ch.isspace() or ch in {".", "/", "\\", ":", "*", "?", "<", ">", "|", '"'}:
            cleaned_chars.append("_")
        elif ord(ch) < 32:
            continue
        else:
            cleaned_chars.append(ch)

    cleaned = "".join(cleaned_chars)
    # Collapse runs of underscores introduced by sanitization.
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned[:_MAX_DOC_NAME_LEN]


def _short_task_name(task_name: str) -> str:
    """Strip the ``knows.`` prefix and turn a trailing ``.N`` into ``_instance_N``.

    ``knows.docs_1_formal_letter.1`` -> ``docs_1_formal_letter_instance_1``.
    ``knows.docs_1_formal_letter``   -> ``docs_1_formal_letter``.
    """
    name = task_name
    if name.startswith("knows."):
        name = name[len("knows.") :]
    m = _TASK_INSTANCE_SUFFIX_RE.search(name)
    if m:
        name = name[: m.start()] + f"_instance_{m.group(1)}"
    return name


def build_doc_name(agent_name: str, task_name: str, instance_id: int) -> str:
    """Compose the human-readable doc title.

    Form: ``<sanitized_agent_name>_<short_task_name>``. If the short task name
    does not already encode the instance (because the caller passed a generic
    task id like ``knows.docs_1_formal_letter``), ``_instance_<id>`` is
    appended so each instance produces a distinct doc.
    """
    base = _short_task_name(task_name)
    if "instance_" not in base:
        base = f"{base}_instance_{instance_id}"
    composed = f"{sanitize_doc_name(agent_name)}_{base}"
    return sanitize_doc_name(composed)


def _extract_doc_id(url: str, kind: str = "docs") -> str:
    """Extract a Drive file id from *url*. ``kind`` selects which URL
    pattern to match (``docs`` / ``sheets`` / ``slides``)."""
    info = _WORKSPACE_KINDS.get(kind)
    if info is None:
        raise ValueError(
            f"Unknown workspace kind {kind!r}; "
            f"expected one of {sorted(_WORKSPACE_KINDS.keys())}."
        )
    m = info["id_re"].search(url or "")
    if not m:
        raise RuntimeError(
            f"Could not extract {info['kind_label']} id from URL: {url!r}"
        )
    return m.group(1)


def _read_service_account_email(
    service_account_path: Optional[Path] = None,
) -> Optional[str]:
    """Return the ``client_email`` from the evaluator's service-account JSON.

    Looks at, in order: ``service_account_path`` arg, the
    ``SERVICE_ACCOUNT_PATH`` env var, then the bundled
    ``browsergym/knows/auth-data/service-account.json``. Returns ``None`` if
    no usable file is found.
    """
    candidates = []
    if service_account_path is not None:
        candidates.append(Path(service_account_path))
    sa_env = os.environ.get("SERVICE_ACCOUNT_PATH")
    if sa_env:
        candidates.append(Path(sa_env))
    candidates.append(_DEFAULT_SERVICE_ACCOUNT_PATH)

    for p in candidates:
        if not p.is_file():
            continue
        try:
            with open(p) as f:
                data = json.load(f)
        except Exception as exc:  # noqa: BLE001 - best-effort read
            logger.debug("Could not parse service-account file %s: %s", p, exc)
            continue
        email = data.get("client_email")
        if email:
            return email
    return None


def _resolve_service_account_path(service_account_path: Optional[Path] = None) -> Path:
    """Return the service-account JSON path used for Drive API calls."""
    if service_account_path is not None:
        return Path(service_account_path)
    sa_env = os.environ.get("SERVICE_ACCOUNT_PATH")
    if sa_env:
        return Path(sa_env)
    return _DEFAULT_SERVICE_ACCOUNT_PATH


def _load_service_account_credentials(service_account_path: Optional[Path] = None):
    """Load Drive credentials from the evaluator service-account JSON."""
    from google.oauth2.service_account import Credentials

    sa_path = _resolve_service_account_path(service_account_path)
    if not sa_path.is_file():
        raise FileNotFoundError(f"Service-account file not found at {sa_path}")

    scopes = ["https://www.googleapis.com/auth/drive"]
    return Credentials.from_service_account_file(str(sa_path), scopes=scopes)


def _load_user_oauth_credentials(
    token_path: Optional[Path] = None,
    client_secrets_path: Optional[Path] = None,
    *,
    interactive: bool = False,
):
    """Load (and refresh if needed) the *user's* OAuth credentials.

    These are the same credentials that ``initialize_google_services`` falls
    back to when no service account is configured. We need them here because
    the agent creates docs in the user's personal Drive — the service account
    can't add itself as a collaborator, only the owner can.

    If the stored refresh token is invalid and ``interactive=True``, an
    ``InstalledAppFlow`` is launched to regenerate ``token.json`` (a browser
    window will open for the user to log in). When ``interactive=False`` (the
    default — used during automated runs), invalid tokens raise.

    Returns a ``google.oauth2.credentials.Credentials`` instance, or raises if
    no usable token / client-secrets file is available.
    """
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_p = Path(token_path) if token_path else None
    if token_p is None:
        env_token = os.environ.get("TOKEN_PATH")
        token_p = Path(env_token) if env_token else _DEFAULT_TOKEN_PATH

    secrets_p = Path(client_secrets_path) if client_secrets_path else None
    if secrets_p is None:
        env_secrets = os.environ.get("CLIENT_SECRETS_PATH")
        secrets_p = Path(env_secrets) if env_secrets else _DEFAULT_CLIENT_SECRETS_PATH

    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/documents",
    ]

    creds = None
    if token_p.is_file():
        creds = Credentials.from_authorized_user_file(str(token_p), scopes)

    def _persist_and_return(c):
        try:
            with open(token_p, "w") as f:
                f.write(c.to_json())
        except Exception as exc:  # noqa: BLE001 - best-effort persist
            logger.debug("Could not persist token to %s: %s", token_p, exc)
        return c

    if creds and creds.valid:
        return creds

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            if not interactive:
                raise RuntimeError(
                    f"OAuth refresh token at {token_p} is invalid ({exc}); "
                    "regenerate it by running `python -m "
                    "browsergym.knows.doc_setup --reauth-oauth` (a browser "
                    "window will open) and try again."
                ) from exc
            creds = None  # fall through to InstalledAppFlow below
        else:
            return _persist_and_return(creds)

    if not interactive:
        raise RuntimeError(
            f"OAuth token at {token_p} is missing or invalid; regenerate it "
            "by running `python -m browsergym.knows.doc_setup --reauth-oauth`."
        )

    if not secrets_p.is_file():
        raise FileNotFoundError(
            f"OAuth client-secrets file not found at {secrets_p}; cannot run "
            "the interactive OAuth flow."
        )

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(secrets_p), scopes)
    creds = flow.run_local_server(port=0)
    return _persist_and_return(creds)


def share_doc_with_service_account(
    doc_id: str,
    *,
    role: str = "writer",
    service_account_email: Optional[str] = None,
    token_path: Optional[Path] = None,
    client_secrets_path: Optional[Path] = None,
    service_account_path: Optional[Path] = None,
) -> bool:
    """Grant the evaluator's service account access to *doc_id*.

    This helper intentionally authenticates the Drive API request with the
    evaluator service account, matching the credentials used by the KNOWS
    evaluators (see ``google_services_utils.initialize_google_services``).
    That means it can only update sharing for files the service account can
    already access and has permission to reshare.

    Parameters
    ----------
    doc_id : str
        Google Docs / Drive file id (the long token in the doc URL).
    role : str, default ``"writer"``
        Drive permission role; ``"reader"`` or ``"writer"`` are typical. The
        evaluator only needs read access, but ``writer`` makes manual debug
        edits convenient.
    service_account_email : str, optional
        Override the service-account email. By default the email is read
        from the ``client_email`` field of the service-account JSON.
    token_path / client_secrets_path : Path, optional
        Deprecated compatibility arguments. Sharing no longer uses OAuth.
    service_account_path : Path, optional
        Override path to the service-account credentials.

    Returns
    -------
    bool
        ``True`` if a permission was created (or the request reported the
        permission already existed); ``False`` on best-effort failure (e.g.
        the service account cannot access the doc, cannot reshare it, or the
        doc no longer exists).
    """
    if not doc_id:
        logger.warning("share_doc_with_service_account: doc_id is empty")
        return False

    if token_path or client_secrets_path:
        logger.debug(
            "share_doc_with_service_account: ignoring deprecated OAuth path "
            "arguments because sharing uses service-account credentials."
        )

    sa_email = service_account_email or _read_service_account_email(service_account_path)
    if not sa_email:
        logger.warning(
            "share_doc_with_service_account: no service-account email found; "
            "skipping share for doc %s",
            doc_id,
        )
        return False

    try:
        creds = _load_service_account_credentials(service_account_path)
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.warning(
            "share_doc_with_service_account: could not load service-account credentials "
            "(%s); skipping share for doc %s",
            exc,
            doc_id,
        )
        return False

    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except Exception as exc:  # noqa: BLE001 - dependency missing
        logger.warning(
            "share_doc_with_service_account: googleapiclient not installed "
            "(%s); skipping share for doc %s",
            exc,
            doc_id,
        )
        return False

    drive = build("drive", "v3", credentials=creds, cache_discovery=False)
    try:
        drive.permissions().create(
            fileId=doc_id,
            body={"type": "user", "role": role, "emailAddress": sa_email},
            sendNotificationEmail=False,
            fields="id",
            supportsAllDrives=True,
        ).execute()
        logger.info(
            "Shared doc %s with service account %s (role=%s)",
            doc_id,
            sa_email,
            role,
        )
        return True
    except HttpError as err:
        # Drive raises 400 with reason ``invalidSharingRequest`` when the
        # permission already exists for that user — treat that as success.
        msg = str(err)
        if "already" in msg.lower() or err.resp.status == 400:
            logger.info(
                "Service account %s already has access to doc %s (or sharing "
                "request was a no-op): %s",
                sa_email,
                doc_id,
                err,
            )
            return True
        logger.warning(
            "Failed to share doc %s with service account %s: %s",
            doc_id,
            sa_email,
            err,
        )
        return False
    except Exception as exc:  # noqa: BLE001 - best-effort
        logger.warning(
            "Unexpected error while sharing doc %s with service account %s: %s",
            doc_id,
            sa_email,
            exc,
        )
        return False


def _rename_via_ui(page: playwright.sync_api.Page, new_name: str) -> bool:
    """Best-effort rename of the currently open Google Doc.

    Tries a small set of selectors known to work for the document title. Any
    failure is logged and ``False`` is returned; the caller can decide whether
    to treat that as fatal. The doc remains usable either way (it just keeps
    its default "Untitled document" title).
    """
    selectors = (
        'input[aria-label="Rename"]',
        'input[aria-label="Document title"]',
        'input.docs-title-input',
    )

    for selector in selectors:
        try:
            locator = page.locator(selector).first
            locator.wait_for(state="visible", timeout=5000)
            locator.click()
            # Select-all then type to fully replace any existing title.
            page.keyboard.press("Meta+A")
            page.keyboard.press("Control+A")
            locator.fill(new_name)
            page.keyboard.press("Enter")
            return True
        except Exception as exc:  # noqa: BLE001 — UI variability
            logger.debug("Rename via selector %s failed: %s", selector, exc)
            continue

    # Fallback: try the menu-driven rename via role-based lookup.
    try:
        title_box = page.get_by_role("textbox", name=re.compile("Rename|Document title", re.I))
        title_box.click()
        page.keyboard.press("Meta+A")
        page.keyboard.press("Control+A")
        title_box.fill(new_name)
        page.keyboard.press("Enter")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not rename Google Doc to %r: %s", new_name, exc)
        return False


def create_task_workspace(
    page: playwright.sync_api.Page,
    agent_name: str,
    task_name: str,
    instance_id: int,
    *,
    kind: str = "docs",
    timeout_ms: int = 60000,
    share_with_evaluator: bool = True,
    max_attempts: int = 3,
) -> Tuple[str, str]:
    """Create a fresh Google workspace file (Doc / Sheet / Slides), rename
    it, and return ``(doc_id, doc_url)``.

    Parameters
    ----------
    kind : str, default ``"docs"``
        Which Google app to spawn:

        - ``"docs"``   -> ``docs.google.com/document/create``
        - ``"sheets"`` -> ``docs.google.com/spreadsheets/create``
        - ``"slides"`` -> ``docs.google.com/presentation/create``

    timeout_ms : int, default ``60000``
        Per-attempt timeout for individual Playwright navigation calls. Note
        that Google Sheets / Slides editors are heavy and routinely take
        >30s to reach a full ``load`` state when several workers are
        spinning up Chromium in parallel, so we default to 60s and rely on
        ``wait_until="commit"`` for the URL-pattern check (which only needs
        the URL to change, not the full page to finish loading).
    max_attempts : int, default ``3``
        How many times to retry the create+url-match step on transient
        Playwright timeouts before giving up. Each retry uses a fresh
        ``goto`` to the create URL.

    The provided ``page`` is navigated to the new file so the agent can
    start the task immediately on that tab.

    When ``share_with_evaluator`` is true (the default), the new file is
    also shared with the KNOWS evaluator's service account so the grading
    step at episode end can read it via the Drive APIs. Sharing is
    best-effort: if the service account cannot access or reshare the file,
    the share attempt is logged and the file is still returned.
    """
    workspace = _WORKSPACE_KINDS.get(kind)
    if workspace is None:
        raise ValueError(
            f"Unknown workspace kind {kind!r}; "
            f"expected one of {sorted(_WORKSPACE_KINDS.keys())}."
        )

    doc_name = build_doc_name(agent_name, task_name, instance_id)
    logger.info("Creating %s %r", workspace["kind_label"], doc_name)

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            _debug_progress(
                f"goto create URL for {kind} instance={instance_id} "
                f"(attempt {attempt}/{max_attempts})"
            )
            # ``wait_until="commit"`` returns as soon as the network
            # response for the create URL is committed — we don't need
            # the page to be fully loaded to read the redirected URL.
            page.goto(
                workspace["create_url"],
                timeout=timeout_ms,
                wait_until="commit",
            )
            _debug_progress(f"waiting for created {kind} URL")
            # Likewise, only wait for the redirect to the
            # ``/<segment>/d/<id>/...`` URL to commit. Waiting for the
            # full editor to load here is what was timing out under
            # parallel load — the rename step below has its own
            # ``domcontentloaded`` wait.
            page.wait_for_url(
                workspace["id_re"],
                timeout=timeout_ms,
                wait_until="commit",
            )
            break
        except Exception as exc:  # noqa: BLE001 - retry on any nav failure
            last_exc = exc
            logger.warning(
                "Workspace create attempt %d/%d for %s instance=%s failed: %s",
                attempt,
                max_attempts,
                kind,
                instance_id,
                exc,
            )
            if attempt >= max_attempts:
                raise
            # Brief backoff before the next attempt; helps when the
            # failure was due to transient overload of the Google
            # frontend or local Chromium.
            try:
                page.wait_for_timeout(2000 * attempt)
            except Exception:  # noqa: BLE001
                pass
    else:
        # Defensive: ``for/else`` only runs if the loop completed without
        # ``break``, which can't happen here because we either ``break``
        # on success or ``raise`` on the final attempt. Re-raise just in
        # case to be explicit.
        if last_exc is not None:
            raise last_exc

    doc_id = _extract_doc_id(page.url, kind=kind)
    doc_url = f"https://docs.google.com/{workspace['url_segment']}/d/{doc_id}/edit"
    _debug_progress(f"created {kind} id={doc_id}")

    # Give the editor a moment to fully render before we try to rename.
    try:
        _debug_progress(f"waiting for {kind} domcontentloaded")
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:  # noqa: BLE001
        pass

    _debug_progress(f"renaming {kind} id={doc_id}")
    if not _rename_via_ui(page, doc_name):
        logger.warning(
            "%s %s was created but could not be renamed to %r; continuing.",
            workspace["kind_label"],
            doc_id,
            doc_name,
        )

    if share_with_evaluator:
        api_share_ok = False
        try:
            _debug_progress(f"sharing {kind} id={doc_id} with evaluator (Drive API)")
            api_share_ok = bool(share_doc_with_service_account(doc_id))
        except Exception as exc:  # noqa: BLE001 - best-effort
            logger.warning(
                "Could not share %s %s with the evaluator service account "
                "via Drive API: %s",
                workspace["kind_label"],
                doc_id,
                exc,
            )

        # Playwright UI fallback: when the Drive API path fails (e.g. the
        # service account cannot reshare the file, or googleapiclient is
        # not installed in this env), drive the editor's Share dialog on
        # the same ``page`` we already have open. The fallback is itself
        # idempotent (it bails out if the SA is already listed), so it's
        # safe to run on re-attempts.
        ui_disabled = os.environ.get(
            "KNOWS_DISABLE_UI_SHARE_FALLBACK", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not api_share_ok and not ui_disabled:
            sa_email = _read_service_account_email()
            if not sa_email:
                logger.warning(
                    "UI share fallback skipped for %s %s: no service-account email found.",
                    workspace["kind_label"],
                    doc_id,
                )
            else:
                try:
                    from .share_ui_fallback import share_workspace_via_ui

                    _debug_progress(
                        f"sharing {kind} id={doc_id} with evaluator (UI fallback)"
                    )
                    share_workspace_via_ui(
                        page,
                        doc_id=doc_id,
                        kind=kind,
                        sa_email=sa_email,
                    )
                except Exception as exc:  # noqa: BLE001 - best-effort
                    logger.warning(
                        "UI share fallback failed for %s %s: %s",
                        workspace["kind_label"],
                        doc_id,
                        exc,
                    )

    _debug_progress(f"workspace ready {kind} id={doc_id}")
    return doc_id, doc_url


def create_task_doc(
    page: playwright.sync_api.Page,
    agent_name: str,
    task_name: str,
    instance_id: int,
    *,
    timeout_ms: int = 60000,
    share_with_evaluator: bool = True,
) -> Tuple[str, str]:
    """Backwards-compatible wrapper around :func:`create_task_workspace`
    that always creates a Google Doc."""
    return create_task_workspace(
        page,
        agent_name=agent_name,
        task_name=task_name,
        instance_id=instance_id,
        kind="docs",
        timeout_ms=timeout_ms,
        share_with_evaluator=share_with_evaluator,
    )


def _cli_main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-create a Google Doc for a knows benchmark task.",
    )
    parser.add_argument(
        "--reauth-oauth",
        action="store_true",
        help=(
            "Run the InstalledAppFlow OAuth flow to (re)generate "
            "auth-data/token.json. Use this when share_doc_with_service_account "
            "fails with 'invalid_grant'. All other arguments are ignored."
        ),
    )
    parser.add_argument(
        "--share-doc",
        metavar="DOC_ID",
        help=(
            "Share the given Google Doc with the evaluator's service account "
            "and exit. Useful for re-grading old trials whose docs were "
            "created before auto-sharing was enabled."
        ),
    )
    parser.add_argument(
        "--storage-state",
        help="Path to a Playwright storage_state.json with Google auth.",
    )
    parser.add_argument("--agent-name")
    parser.add_argument("--task-name")
    parser.add_argument("--instance-id", type=int)
    parser.add_argument(
        "--kind",
        default="docs",
        choices=sorted(_WORKSPACE_KINDS.keys()),
        help="Which Google app to spawn (default: docs).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the browser headless (default: headed for easier debugging).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.reauth_oauth:
        creds = _load_user_oauth_credentials(interactive=True)
        print(
            f"OAuth token regenerated at {_DEFAULT_TOKEN_PATH} "
            f"(valid={creds.valid})."
        )
        return 0

    if args.share_doc:
        ok = share_doc_with_service_account(args.share_doc)
        if ok:
            print(f"Doc {args.share_doc} shared with the evaluator's service account.")
            return 0
        print(
            f"Could not share doc {args.share_doc}. Ensure the service "
            "account can access the file and has permission to reshare it."
        )
        return 1

    if not (args.storage_state and args.agent_name and args.task_name and args.instance_id):
        parser.error(
            "--storage-state, --agent-name, --task-name and --instance-id are "
            "required unless --reauth-oauth or --share-doc is passed."
        )

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=args.headless)
        context = browser.new_context(storage_state=args.storage_state)
        page = context.new_page()
        try:
            doc_id, doc_url = create_task_workspace(
                page,
                agent_name=args.agent_name,
                task_name=args.task_name,
                instance_id=args.instance_id,
                kind=args.kind,
            )
        finally:
            context.close()
            browser.close()

    print(doc_id)
    print(doc_url)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())
