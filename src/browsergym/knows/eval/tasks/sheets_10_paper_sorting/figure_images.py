"""Read and validate Paper Sorting figures before sending them to the judge.

Drive thumbnails refer to an existing image file. Read that exact file through
the evaluator's Drive service, rather than downloading an HTML sign-in page
from the public thumbnail endpoint. Retrieval/decoding failures are evaluator
errors; they do not establish that a submitted figure is incorrect.
"""
from __future__ import annotations

import io
import os
import re
import tempfile
from urllib.parse import parse_qs, urlsplit, urlunsplit

from PIL import Image


class FigureImageError(RuntimeError):
    """A figure could not be obtained as a judge-readable image."""


def drive_thumbnail_id(url: str) -> str | None:
    """Recognize only the real Drive thumbnail endpoint, never lookalike hosts."""
    try:
        parsed = urlsplit(url)
        port = parsed.port
    except ValueError:
        raise FigureImageError("Invalid figure URL") from None
    if parsed.hostname != "drive.google.com" or parsed.path != "/thumbnail":
        return None
    if (parsed.scheme != "https" or parsed.username or parsed.password
            or port not in (None, 443)):
        raise FigureImageError("Invalid Google Drive thumbnail URL")
    ids = parse_qs(parsed.query, keep_blank_values=True).get("id", [])
    if len(ids) != 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", ids[0]):
        raise FigureImageError("Invalid Google Drive thumbnail file ID")
    return ids[0]


def validate_figure_image(path: str, *, source: str) -> None:
    """Check real bytes, not an extension or an HTTP content-type assertion."""
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except Exception:
        # Avoid URLs, query strings, identifiers, or remote response bodies in
        # the error. The caller decides whether this is a reference/candidate.
        raise FigureImageError(f"{source} is not a decodable image") from None


def _save_png(content: bytes) -> str:
    path = None
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            image.seek(0)
            image.load()
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as output:
                path = output.name
                if image.format == "PNG":
                    # Preserve an already valid PNG byte for byte.
                    output.write(content)
                else:
                    mode = "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB"
                    image.convert(mode).save(output, format="PNG")
        return path
    except Exception:
        if path is not None:
            os.unlink(path)
        raise FigureImageError("Retrieved figure is not a decodable image") from None


def download_figure_image(url: str, *, drive_service, http_get, arxiv_get, headers) -> str:
    """Download the submitted image, raising on unavailable or invalid bytes.

    Callers provide the already initialized Drive service and HTTP transports;
    this module never loads credentials, browser state, or a Google session.
    """
    file_id = drive_thumbnail_id(url)
    if file_id is not None:
        try:
            content = drive_service.files().get_media(
                fileId=file_id, supportsAllDrives=True).execute()
        except Exception:
            raise FigureImageError("Google Drive figure retrieval failed") from None
    else:
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https"):
            raise FigureImageError("Unsupported figure URL scheme")
        is_arxiv = parsed.hostname in ("arxiv.org", "export.arxiv.org")
        if parsed.hostname == "arxiv.org":
            url = urlunsplit(parsed._replace(netloc="export.arxiv.org"))
        try:
            response = (arxiv_get if is_arxiv else http_get)(url, headers=headers, timeout=30)
            response.raise_for_status()
            content = response.content
        except Exception:
            raise FigureImageError("Figure HTTP retrieval failed") from None
    return _save_png(content)
