"""Playwright UI fallback for sharing a Google workspace file with the SA.

The KNOWS evaluator authenticates as a service account
(``doc-evaluator@your-gcp-project.iam.gserviceaccount.com`` by
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
    the "Add people" sub-flow, not in the link-sharing tab.
    """
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
    people" is checked) or "Share" / "Done" otherwise. We try the
    candidates in order with short individual timeouts so an unmatched
    layout fails quickly instead of spending ``timeout_ms`` on each one.
    """
    candidates = (
        'div[role="button"][aria-label*="Send"]',
        'button[aria-label*="Send"]',
        'button:has-text("Send")',
        'div[role="button"]:has-text("Send")',
        'div[role="button"][aria-label*="Share"]',
        'button[aria-label*="Share"]',
        'button:has-text("Share")',
        'div[role="button"]:has-text("Share")',
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
    and invoke :func:`share_workspace_via_ui` with ``navigate=True``.
    """
    from playwright.sync_api import sync_playwright

    sa_email = _load_service_account_email(service_account_path)
    print(f"Service account: {sa_email}")
    print(f"Doc id         : {doc_id} ({kind})")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=headless)
        context = browser.new_context(storage_state=str(storage_state))
        page = context.new_page()
        try:
            return share_workspace_via_ui(
                page,
                doc_id=doc_id,
                kind=kind,
                sa_email=sa_email,
                timeout_ms=timeout_ms,
                navigate=True,
            )
        finally:
            context.close()
            browser.close()


# Default paths used by the CLI -- match the layout the rest of the package
# already assumes (auth-data dir is two levels above this file's package).
# ``_REPO_ROOT`` is defined at the top of the module.
_DEFAULT_STORAGE_STATE = _REPO_ROOT / "storage_state.json"
_DEFAULT_SERVICE_ACCOUNT = (
    _REPO_ROOT / "browsergym" / "knows" / "auth-data" / "service-account.json"
)


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
