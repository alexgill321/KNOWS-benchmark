"""Playwright UI fallback for sharing a Google workspace file with the SA.

The KNOWS evaluator authenticates as a service account
(``your-evaluator-sa@your-project.iam.gserviceaccount.com`` by
default). The primary share path is the Drive API call in
:func:`browsergym.knows.doc_setup.share_doc_with_service_account`, which is
fast and runs server-side. When that path fails (e.g. the service account
cannot reshare the file, or the underlying ``googleapiclient`` is missing),
this module drives the editor's "Share" dialog with Playwright as a
fallback so the evaluator can still read the doc at grading time.

The selectors used here work for all three Google workspace editors —
Docs, Sheets, and Slides — so the only kind-specific bit is the URL we
navigate to. The kind is always known by the caller (``create_task_workspace``
already takes a ``kind`` argument), so there's no auto-detection.

Idempotency is layered. The primary guarantee comes from the Drive-API
share path in :func:`browsergym.knows.doc_setup.share_doc_with_service_account`,
which is always called first and returns ``True`` (success) when the SA
already has access -- in that case ``create_task_workspace`` skips the
UI fallback entirely. As a best-effort second layer this module's
:func:`_already_shared_with` scans the dialog's participant list when
the dialog exposes it, but Google's modern Share dialog hides the
participant list by default, so detection there is not guaranteed.

Usage from Python::

    from browsergym.knows.share_ui_fallback import share_workspace_via_ui
    share_workspace_via_ui(page, doc_id="abc...", kind="sheets",
                           sa_email="doc-evaluator@...")

CLI (testing only)::

    python -m browsergym.knows.share_ui_fallback \\
        --doc-id 1abc... --kind sheets [--headed] [--storage-state ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# URL fragment used by Drive for each kind. Mirrors
# ``doc_setup._WORKSPACE_KINDS`` -- kept small here to avoid an import
# cycle (this module is imported from ``doc_setup`` itself).
_URL_SEGMENT = {
    "docs": "document",
    "sheets": "spreadsheets",
    "slides": "presentation",
}


# Repo root, computed early so diagnostic helpers (``_dump_debug_state``)
# can reference ``_debug_artifacts/`` without forward-reference issues.
# This module lives at ``browsergym/knows/src/browsergym/knows/share_ui_fallback.py``,
# so six ``parent`` hops climb back to the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent


def _file_url(doc_id: str, kind: str) -> str:
    segment = _URL_SEGMENT.get(kind)
    if segment is None:
        raise ValueError(
            f"Unknown workspace kind {kind!r}; expected one of {sorted(_URL_SEGMENT)}."
        )
    return f"https://docs.google.com/{segment}/d/{doc_id}/edit"


# URL fragments that indicate Google bounced us to a sign-in surface
# instead of the editor. When ``page.goto(_file_url(...))`` lands on any
# of these, the storage_state cookies are stale and we must re-mint
# before the Share dialog can be opened.
_SIGNIN_URL_FRAGMENTS = (
    "accounts.google.com/ServiceLogin",
    "accounts.google.com/AccountChooser",
    "accounts.google.com/v3/signin",
    "accounts.google.com/signin/",
)


def _is_signin_url(url: str) -> bool:
    if not url:
        return False
    return any(frag in url for frag in _SIGNIN_URL_FRAGMENTS)


def _refresh_storage_state(
    storage_state: Path, *, timeout_s: float = 240.0
) -> bool:
    """Best-effort re-mint of *storage_state* via ``scripts/google_auto_login.py``.

    Used to recover from stale Google session cookies: if the standalone
    UI fallback navigates to the editor and is redirected to a sign-in
    page, this helper spawns the auto-login subprocess (the same one the
    benchmark's per-PID mint pool uses) to overwrite the snapshot in
    place. Returns ``True`` if the file was refreshed and exists on disk
    afterwards, ``False`` on any failure (caller should surface a clear
    error to the operator).
    """
    script = _REPO_ROOT / "scripts" / "google_auto_login.py"
    if not script.is_file():
        logger.warning(
            "Cannot refresh storage state: %s not found.", script
        )
        return False

    cmd = [
        sys.executable,
        str(script),
        "--output",
        str(storage_state),
        "--headless",
    ]
    logger.info("Refreshing storage state via %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(_REPO_ROOT),
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning(
            "storage_state refresh timed out after %.1fs: %s", timeout_s, exc
        )
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("storage_state refresh failed to start: %s", exc)
        return False

    if result.returncode != 0:
        logger.warning(
            "storage_state refresh exited with %d. stderr=%s",
            result.returncode,
            (result.stderr or "").strip(),
        )
        return False

    if not storage_state.is_file():
        logger.warning(
            "storage_state refresh reported success but %s does not exist.",
            storage_state,
        )
        return False
    return True


# The Share dialog's content is rendered inside an iframe (Drive's
# "drivesharing" UI), not in the editor's top-level DOM. After clicking
# the Share button we have to drop into this iframe via ``page.frame_locator``
# to reach the email input, the "Notify people" checkbox, and the Send
# button. Multiple selectors are listed because the exact attribute set
# has shifted between editor builds and across kinds.
_SHARE_IFRAME_SELECTORS = (
    'iframe.share-client-content-iframe',
    'iframe[src*="/drivesharing/driveshare"]',
    'iframe[title="Content"]',
)


def _open_share_dialog(page, timeout_ms: int):
    """Click the editor's Share button and return a FrameLocator into
    the share-dialog iframe.

    Tries a series of selectors in decreasing specificity. Each candidate
    gets a short individual timeout (rather than the full ``timeout_ms``)
    so we cycle through alternatives quickly when the UI surface differs
    between Docs / Sheets / Slides or between editor versions.

    The click uses ``force=True`` so we don't get stuck retrying after
    the iframe loads on top of the original Share button (the visible
    pointer-events host changes mid-click). Once the dialog is open we
    wait for the share-iframe to render and return a ``FrameLocator``
    pointing at it; all subsequent dialog interactions go through that
    locator instead of ``page``.
    """
    candidates = (
        # Most specific first: the precise aria-label pattern used by the
        # editor's top-bar Share button. The visible text varies between
        # "Share", "Share. Private to only me.", or "Share. Anyone with the link.".
        'div[role="button"][aria-label^="Share. "]',
        'button[aria-label^="Share. "]',
        # Generic Share-button match (most current Docs layouts).
        'div[role="button"][aria-label*="Share"]',
        'button[aria-label*="Share"]',
        # data-tooltip surface (older editor builds).
        'div[role="button"][data-tooltip^="Share"]',
        # Plain visible-text match -- last resort, may match menu items too.
        'button:has-text("Share")',
        'div[role="button"]:has-text("Share")',
    )
    per_candidate_ms = min(3000, max(1000, timeout_ms // len(candidates)))
    last_err: Optional[Exception] = None
    for sel in candidates:
        try:
            btn = page.locator(sel).first
            btn.wait_for(state="visible", timeout=per_candidate_ms)
            # ``force=True`` skips Playwright's actionability checks --
            # the iframe loading on top of the button right after the
            # click would otherwise trigger an endless retry loop.
            btn.click(force=True, timeout=per_candidate_ms)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    else:
        raise RuntimeError(f"Could not click Share button (last error: {last_err})")

    # Now wait for the share dialog's iframe to appear.
    iframe_last_err: Optional[Exception] = None
    for sel in _SHARE_IFRAME_SELECTORS:
        try:
            page.wait_for_selector(sel, state="visible", timeout=timeout_ms)
            frame = page.frame_locator(sel).first
            return frame
        except Exception as exc:  # noqa: BLE001
            iframe_last_err = exc
            continue
    raise RuntimeError(
        f"Share dialog opened but its iframe did not appear "
        f"(last error: {iframe_last_err})"
    )


def _dump_debug_state(page, doc_id: str, kind: str, reason: str) -> None:
    """Best-effort diagnostic dump on share failure.

    Captures the current page URL, page title, and a full-page screenshot
    into ``_debug_artifacts/share_<kind>_<doc_id>_<reason>.{png,txt}`` at
    the repo root. Failures here are swallowed -- this is purely for
    triage when the share UI moves under us.
    """
    try:
        debug_dir = _REPO_ROOT / "_debug_artifacts"
        debug_dir.mkdir(parents=True, exist_ok=True)
        stem = f"share_{kind}_{doc_id[:12]}_{reason}"
        info_path = debug_dir / f"{stem}.txt"
        png_path = debug_dir / f"{stem}.png"
        try:
            url = page.url
        except Exception:  # noqa: BLE001
            url = "(unavailable)"
        try:
            title = page.title()
        except Exception:  # noqa: BLE001
            title = "(unavailable)"
        info_path.write_text(
            f"reason: {reason}\nkind: {kind}\ndoc_id: {doc_id}\nurl: {url}\ntitle: {title}\n",
            encoding="utf-8",
        )
        try:
            page.screenshot(path=str(png_path), full_page=True, timeout=5000)
        except Exception as exc:  # noqa: BLE001
            info_path.write_text(
                info_path.read_text() + f"screenshot_error: {exc}\n",
                encoding="utf-8",
            )
        logger.warning(
            "share_workspace_via_ui: dumped debug state to %s (and %s)",
            info_path,
            png_path,
        )
    except Exception as exc:  # noqa: BLE001 - diagnostic must never raise
        logger.debug("share_workspace_via_ui: debug dump failed: %s", exc)


def _close_share_dialog(page, frame=None) -> None:
    """Best-effort: close the Share dialog so the editor is interactable again.

    The Cancel / Done buttons live inside the share-iframe (when we have
    one); fall back to top-level selectors and finally an Escape key.
    """
    iframe_candidates = (
        'div[role="button"][aria-label*="Cancel"]',
        'button:has-text("Cancel")',
        'div[role="button"]:has-text("Cancel")',
        'button:has-text("Done")',
        'div[role="button"]:has-text("Done")',
    )
    if frame is not None:
        for sel in iframe_candidates:
            try:
                btn = frame.locator(sel).first
                if btn.count() == 0:
                    continue
                btn.click()
                return
            except Exception:  # noqa: BLE001
                continue
    page_candidates = (
        'div[role="dialog"] button[aria-label*="Close"]',
        'div[role="dialog"] div[role="button"][aria-label*="Close"]',
    )
    for sel in page_candidates:
        try:
            btn = page.locator(sel).first
            if btn.count() == 0:
                continue
            btn.click()
            return
        except Exception:  # noqa: BLE001
            continue
    # Last resort: press Escape.
    try:
        page.keyboard.press("Escape")
    except Exception:  # noqa: BLE001
        pass


def _already_shared_with(frame, sa_email: str) -> bool:
    """Return True if the open Share dialog already lists ``sa_email``.

    All selectors are scoped to the dialog's iframe (``frame``). Looks
    for the SA email in any of: the participant list's data attributes,
    aria-labels, or visible text.
    """
    selectors = (
        f'[data-email="{sa_email}"]',
        f'[aria-label*="{sa_email}"]',
        f'div[role="listitem"]:has-text("{sa_email}")',
    )
    for sel in selectors:
        try:
            if frame.locator(sel).count() > 0:
                return True
        except Exception:  # noqa: BLE001
            continue
    # Fall back to any visible-text match inside the iframe body.
    try:
        if frame.locator(f'body:has-text("{sa_email}")').count() > 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _add_service_account_email(frame, email: str, timeout_ms: int) -> None:
    """Type the SA email into the people-input and confirm the entry.

    All selectors are scoped to the share-dialog iframe. We try the most
    common aria-label variants in order; the trailing ``input``-tag
    fallback catches builds that lose the aria-label entirely.
    """
    selectors = (
        'input[aria-label*="Add people"]',
        'input[aria-label*="people, groups"]',
        'input[aria-label*="people"]',
        'input[aria-label*="email"]',
        'input[type="text"]',
        'textarea',
    )
    per_candidate_ms = min(3000, max(1000, timeout_ms // len(selectors)))
    last_err: Optional[Exception] = None
    for sel in selectors:
        try:
            inp = frame.locator(sel).first
            inp.wait_for(state="visible", timeout=per_candidate_ms)
            inp.click()
            inp.fill("")
            inp.type(email, delay=20)
            time.sleep(0.5)
            inp.press("Enter")
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(
        f"Could not enter service account email (last error: {last_err})"
    )


def _disable_notification(frame) -> None:
    """Best-effort: untick the "Notify people" checkbox if present.

    Scoped to the share-dialog iframe. The checkbox is only present in
    the "Add people" sub-flow, not in the link-sharing tab. Failures
    here are not fatal -- the worst case is the SA receives a useless
    notification email; the actual share still goes through.
    """
    try:
        cb = frame.get_by_role(
            "checkbox", name="Notify people", exact=False
        ).first
        if cb.count() > 0 and cb.is_checked():
            cb.click()
        return
    except Exception:  # noqa: BLE001 -- the role-based query may fail on
        # builds where the checkbox is rendered as a styled <div role="checkbox">.
        pass

    candidates = (
        'div[role="checkbox"][aria-label*="Notify"]',
        'input[type="checkbox"][aria-label*="Notify"]',
        'div[role="checkbox"]:has-text("Notify")',
    )
    for sel in candidates:
        try:
            box = frame.locator(sel).first
            if box.count() == 0:
                continue
            checked = box.get_attribute("aria-checked")
            if checked == "true":
                box.click()
            return
        except Exception:  # noqa: BLE001
            continue


def _click_send_or_share(frame, timeout_ms: int) -> None:
    """Click the final Send / Share confirmation button.

    Scoped to the share-dialog iframe. After the SA email has been
    entered, the primary action button is labeled "Send" (when "Notify
    people" is checked) or "Share" / "Done" otherwise.

    Strategy:

    1. ``get_by_role("button", name=..., exact=True)`` for the three
       known labels. Most reliable on Material-Design-rendered buttons
       where the visible text lives inside a nested ``<span jsname="V67aGc">``
       and ``has-text`` matches against descendants but ``.first`` may
       resolve to a hidden ancestor.
    2. Fall back to the historical attribute / text selectors. We
       deliberately drop the over-broad ``:has-text("Share")`` candidates
       that used to match the dialog's title (``Share "..."``) instead
       of the action button.
    """
    settle_ms = min(2000, max(800, timeout_ms // 10))
    time.sleep(settle_ms / 1000.0)

    role_names = ("Send", "Share", "Done")
    role_timeout_ms = min(4000, max(1500, timeout_ms // 5))
    for name in role_names:
        try:
            btn = frame.get_by_role("button", name=name, exact=True).first
            btn.wait_for(state="visible", timeout=role_timeout_ms)
            btn.click()
            time.sleep(2)
            return
        except Exception:  # noqa: BLE001 -- many DOM variants, try them all
            continue

    candidates = (
        'div[role="button"][aria-label="Send"]',
        'button[aria-label="Send"]',
        'div[role="button"][aria-label*="Send"]',
        'button[aria-label*="Send"]',
        'button:has-text("Send")',
        'div[role="button"]:has-text("Send")',
        'div[role="button"][aria-label*="Share"]',
        'button[aria-label*="Share"]',
    )
    per_candidate_ms = min(3000, max(1000, timeout_ms // len(candidates)))
    last_err: Optional[Exception] = None
    for sel in candidates:
        try:
            btn = frame.locator(sel).first
            btn.wait_for(state="visible", timeout=per_candidate_ms)
            btn.click()
            time.sleep(2)
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    raise RuntimeError(f"Could not click Send/Share (last error: {last_err})")


def share_workspace_via_ui(
    page,
    *,
    doc_id: str,
    kind: str,
    sa_email: str,
    timeout_ms: int = 20000,
    navigate: bool = False,
) -> bool:
    """Share *doc_id* with *sa_email* by driving the editor's Share dialog.

    Parameters
    ----------
    page :
        A Playwright ``Page`` already authenticated as the file's owner.
        When ``navigate=False`` (default), the page is assumed to already
        be on the editor for ``doc_id``; we only navigate when invoked
        standalone (see :func:`share_workspace_via_ui_standalone`).
    doc_id :
        Drive file id for the workspace file.
    kind :
        ``"docs"``, ``"sheets"``, or ``"slides"`` -- selects the URL pattern.
    sa_email :
        Service-account email to add as a writer.
    timeout_ms :
        Per-step Playwright timeout. Defaults to 20s.
    navigate :
        When True, navigate ``page`` to the file's edit URL before opening
        the Share dialog. Used by the standalone CLI; the in-benchmark
        callsite already has the file loaded so leaves this False.

    Returns
    -------
    bool
        True if the share completed (or was already in place); False on a
        best-effort failure.
    """
    if not doc_id:
        logger.warning("share_workspace_via_ui: empty doc_id")
        return False
    if not sa_email:
        logger.warning("share_workspace_via_ui: empty sa_email")
        return False

    frame = None
    try:
        if navigate:
            page.goto(_file_url(doc_id, kind), timeout=timeout_ms)
            try:
                page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(2)

        frame = _open_share_dialog(page, timeout_ms)

        # Idempotent fast-path: if the SA is already in the dialog, just close.
        if _already_shared_with(frame, sa_email):
            logger.info(
                "share_workspace_via_ui: %s already shared with %s; skipping",
                doc_id,
                sa_email,
            )
            _close_share_dialog(page, frame)
            return True

        _add_service_account_email(frame, sa_email, timeout_ms)
        _disable_notification(frame)
        _click_send_or_share(frame, timeout_ms)
        logger.info(
            "share_workspace_via_ui: shared %s (%s) with %s",
            doc_id,
            kind,
            sa_email,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - best-effort fallback
        logger.warning(
            "share_workspace_via_ui: UI share failed for %s (%s): %s",
            doc_id,
            kind,
            exc,
        )
        # Diagnostic: dump page URL + screenshot so we can see what state
        # the editor was in when the Share-dialog selectors failed. This
        # is invaluable when Google rotates the Share-button DOM.
        _dump_debug_state(page, doc_id, kind, reason="share_failed")
        # Try to leave the editor in a usable state for downstream code.
        try:
            _close_share_dialog(page, frame)
        except Exception:  # noqa: BLE001
            pass
        return False


def _load_service_account_email(sa_path: Path) -> str:
    if not sa_path.is_file():
        raise FileNotFoundError(f"Service account file not found: {sa_path}")
    with open(sa_path) as f:
        data = json.load(f)
    email = data.get("client_email")
    if not email:
        raise ValueError(f"No client_email in {sa_path}")
    return email


def share_workspace_via_ui_standalone(
    *,
    doc_id: str,
    kind: str,
    storage_state: Path,
    service_account_path: Path,
    headless: bool = True,
    timeout_ms: int = 20000,
) -> bool:
    """Standalone variant: launches its own Playwright context.

    Mirrors the historical behaviour of the root-level ``share_doc_with_sa.py``
    CLI: load ``storage_state``, open a fresh browser, navigate to the file,
    and drive the editor's Share dialog.

    Resilience to stale auth: if the post-navigation URL lands on a
    Google sign-in surface (``accounts.google.com/...``) the function
    closes the context, re-mints ``storage_state`` via
    :func:`_refresh_storage_state`, opens a fresh context with the new
    snapshot, and retries the navigation once. This is what previously
    failed with ``Could not click Share button (Locator.wait_for: ...)``
    when ``storage_state.json`` had aged past Google's session window
    (visible in the dumped ``_debug_artifacts/share_*_share_failed.txt``
    URL pointing at ``accounts.google.com/v3/signin/...``).
    """
    from playwright.sync_api import sync_playwright

    sa_email = _load_service_account_email(service_account_path)
    print(f"Service account: {sa_email}")
    print(f"Doc id         : {doc_id} ({kind})")

    target_url = _file_url(doc_id, kind)
    refreshed = False

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        try:
            for attempt in range(2):
                context = browser.new_context(storage_state=str(storage_state))
                page = context.new_page()
                try:
                    try:
                        page.goto(target_url, timeout=timeout_ms)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "share_workspace_via_ui_standalone: page.goto(%s) failed: %s",
                            target_url,
                            exc,
                        )
                    try:
                        page.wait_for_load_state(
                            "domcontentloaded", timeout=timeout_ms
                        )
                    except Exception:  # noqa: BLE001
                        pass
                    # Settle so Google's client-side redirect chain has
                    # time to bounce us to either the editor or the
                    # sign-in page before we sample ``page.url``.
                    time.sleep(2)

                    current_url = page.url
                    if _is_signin_url(current_url):
                        if attempt == 0 and not refreshed:
                            print(
                                f"[share] storage_state.json stale -- landed on sign-in page "
                                f"({current_url}); refreshing and retrying once."
                            )
                            try:
                                context.close()
                            except Exception:  # noqa: BLE001
                                pass
                            if not _refresh_storage_state(storage_state):
                                print(
                                    f"[share] Could not auto-refresh {storage_state}. "
                                    "Run `python scripts/google_auto_login.py "
                                    f"--output {storage_state}` (or `--headed` once "
                                    "for first-run device trust) and retry."
                                )
                                return False
                            refreshed = True
                            continue  # Retry navigation with the freshly minted state.
                        print(
                            f"[share] Still on Google sign-in after refreshing storage_state "
                            f"({current_url}); cannot drive Share dialog. "
                            "See _debug_artifacts/share_*_share_failed.{png,txt}."
                        )
                        _dump_debug_state(
                            page, doc_id, kind, reason="signin_after_refresh"
                        )
                        return False

                    # Page is on the editor (or close enough); hand off
                    # to the shared share-driving logic. ``navigate=False``
                    # because we already navigated above.
                    return share_workspace_via_ui(
                        page,
                        doc_id=doc_id,
                        kind=kind,
                        sa_email=sa_email,
                        timeout_ms=timeout_ms,
                        navigate=False,
                    )
                finally:
                    try:
                        context.close()
                    except Exception:  # noqa: BLE001
                        pass
            return False
        finally:
            try:
                browser.close()
            except Exception:  # noqa: BLE001
                pass


# Default paths used by the CLI -- match the layout the rest of the package
# already assumes (auth-data dir is two levels above this file's package).
# ``_REPO_ROOT`` is defined at the top of the module.
_DEFAULT_STORAGE_STATE = _REPO_ROOT / "storage_state.json"
_DEFAULT_SERVICE_ACCOUNT = (
    _REPO_ROOT / "browsergym" / "knows" / "auth-data" / "service-account.json"
)

# Recognized workspace kinds for the UI fallback dispatch. Mirrors the
# keys of ``_URL_SEGMENT`` above; kept as a frozenset so callers can
# membership-test without importing the segment table itself.
_KNOWN_KINDS = frozenset(_URL_SEGMENT.keys())


def _ui_fallback_disabled() -> bool:
    """Mirror the gate used by ``doc_setup.create_task_workspace``."""
    return os.environ.get("KNOWS_DISABLE_UI_SHARE_FALLBACK", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def share_doc_with_fallback(
    doc_id: str,
    *,
    kind: Optional[str],
    storage_state: Optional[Path] = None,
    service_account_path: Optional[Path] = None,
    headless: bool = True,
    timeout_ms: int = 20000,
) -> bool:
    """Two-tier share for *doc_id* used by helper scripts.

    1. Calls :func:`browsergym.knows.doc_setup.share_doc_with_service_account`
       (Drive API). Fast path; returns ``True`` when the SA was newly added
       *or* already had access (idempotent).
    2. On failure, falls back to :func:`share_workspace_via_ui_standalone`,
       which drives the editor's Share dialog with a freshly-minted (or
       just-refreshed) ``storage_state.json``.

    Parameters
    ----------
    doc_id :
        Drive file id (long token from the doc URL).
    kind :
        ``"docs"`` / ``"sheets"`` / ``"slides"``. Required for the UI
        fallback. Pass ``None`` to skip the fallback entirely (e.g. when
        the caller can't infer the kind reliably).
    storage_state, service_account_path :
        Override paths. Default to the package-level defaults next to this
        file (``_DEFAULT_STORAGE_STATE`` / ``_DEFAULT_SERVICE_ACCOUNT``).
    headless, timeout_ms :
        Forwarded to :func:`share_workspace_via_ui_standalone`.

    Returns
    -------
    bool
        ``True`` if either tier succeeded; ``False`` otherwise. The
        function never raises -- diagnostic context is printed via
        ``print`` so callers see it in their CLI output.

    Set ``KNOWS_DISABLE_UI_SHARE_FALLBACK=1`` to skip the UI fallback
    (useful when isolating Drive-API failures during debugging).
    """
    api_ok = False
    try:
        from .doc_setup import share_doc_with_service_account  # type: ignore

        api_ok = bool(share_doc_with_service_account(doc_id))
        if api_ok:
            print(f"[share] Document {doc_id} shared via Drive API.")
        else:
            print(
                f"[share] Drive API share returned False for {doc_id} -- "
                "trying Playwright UI fallback."
            )
    except Exception as exc:  # noqa: BLE001 - best-effort
        print(
            f"[share] Drive API share raised {exc!r}; "
            "trying Playwright UI fallback."
        )

    if api_ok:
        return True

    if kind is None:
        print(
            "[share] Warning: cannot run UI fallback (workspace kind unknown); "
            "grading may fail if the doc isn't already accessible."
        )
        return False

    if kind not in _KNOWN_KINDS:
        print(
            f"[share] Warning: unknown workspace kind {kind!r}; "
            f"expected one of {sorted(_KNOWN_KINDS)}. Skipping UI fallback."
        )
        return False

    if _ui_fallback_disabled():
        print(
            "[share] UI fallback disabled via KNOWS_DISABLE_UI_SHARE_FALLBACK; "
            "grading may fail if the doc isn't already accessible."
        )
        return False

    storage_state = storage_state or _DEFAULT_STORAGE_STATE
    service_account_path = service_account_path or _DEFAULT_SERVICE_ACCOUNT

    if not storage_state.is_file():
        # Cold-start: no snapshot at all. Mint one before driving the UI.
        print(
            f"[share] storage_state.json not found at {storage_state}; "
            "minting a fresh one via google_auto_login.py..."
        )
        if not _refresh_storage_state(storage_state):
            print(
                f"[share] UI fallback skipped: could not mint {storage_state}. "
                "Run `python scripts/google_auto_login.py --headed "
                f"--output {storage_state}` once for first-run device trust."
            )
            return False

    try:
        print(f"[share] Trying Playwright UI fallback (kind={kind})...")
        ui_ok = share_workspace_via_ui_standalone(
            doc_id=doc_id,
            kind=kind,
            storage_state=storage_state,
            service_account_path=service_account_path,
            headless=headless,
            timeout_ms=timeout_ms,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort
        print(f"[share] Warning: UI fallback raised {exc!r}; continuing anyway.")
        return False

    if ui_ok:
        print(f"[share] Document {doc_id} shared via Playwright UI fallback.")
        return True

    print(
        f"[share] Warning: UI fallback could not share {doc_id} -- "
        "grading may fail if the doc isn't already accessible. "
        "See _debug_artifacts/share_*_share_failed.{png,txt} for state."
    )
    return False


def kind_from_split_or_family(name: str) -> Optional[str]:
    """Extract the workspace kind from a split short-name or task-family folder.

    ``"docs_1"`` / ``"docs_1_formal_letter"`` -> ``"docs"``.
    ``"slides_51"`` / ``"slides_51_event_announcement_poster"`` -> ``"slides"``.
    Returns ``None`` when the prefix is not recognized.
    """
    if not name:
        return None
    head = name.split("_", 1)[0].lower()
    return head if head in _KNOWN_KINDS else None


def _cli_main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc-id", required=True, help="Drive file id to share.")
    parser.add_argument(
        "--kind",
        default="docs",
        choices=sorted(_URL_SEGMENT.keys()),
        help="Which Google app the file belongs to (default: docs).",
    )
    parser.add_argument(
        "--storage-state",
        type=Path,
        default=_DEFAULT_STORAGE_STATE,
        help=f"Path to Playwright storage_state.json (default: {_DEFAULT_STORAGE_STATE}).",
    )
    parser.add_argument(
        "--service-account",
        type=Path,
        default=_DEFAULT_SERVICE_ACCOUNT,
        help=f"Path to service-account.json (default: {_DEFAULT_SERVICE_ACCOUNT}).",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run with a visible browser (handy for debugging Share-UI changes).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    try:
        ok = share_workspace_via_ui_standalone(
            doc_id=args.doc_id,
            kind=args.kind,
            storage_state=args.storage_state,
            service_account_path=args.service_account,
            headless=not args.headed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Share failed: {exc}", file=sys.stderr)
        return 1
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())
