"""A deliberately small, UI-only Playwright surface for Codex episodes.

Each session owns a fresh browser context. The agent gets pixels, accessibility
text, and normal mouse/keyboard/navigation actions, never a JavaScript evaluator
or an HTTP client. Login state is an input to the context, not a mutable profile.
"""

from __future__ import annotations

import asyncio
import base64
import ipaddress
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit


class ActionBudgetExceeded(RuntimeError):
    """Raised before executing a tool once the episode's budget is exhausted."""


class BrowserAuthenticationError(RuntimeError):
    """A top-level Google sign-in or challenge page needs supervisor attention."""


def _google_auth_location(url: str) -> dict[str, str] | None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or (parsed.hostname or "").lower().rstrip(".") != "accounts.google.com":
        return None
    segments = [segment for segment in unquote(parsed.path).split("/") if segment]
    if [segment.lower() for segment in segments] == ["signin", "oauth", "legacy", "consent"]:
        # Apps Script opens this permission-review route while already signed
        # in. It is not evidence of lost authentication and grants nothing here.
        return None
    markers = {"signin", "servicelogin", "accountchooser", "interactivelogin", "challenge", "webreauth"}
    if segments and not any(segment.lower() in markers for segment in segments):
        # Cookie/session bridge endpoints are not evidence of lost access.
        return None
    safe_segments = markers | {"v1", "v2", "v3", "identifier", "pwd", "password", "totp", "ipp", "dp",
                               "selection", "denied", "rejected", "disabled", "oauth"}
    path = "/" + "/".join(segment if segment.lower() in safe_segments else "[redacted]" for segment in segments)
    return {"host": "accounts.google.com", "path": path}


def validate_url(url: str) -> str:
    """Validate public web navigation, including browser-parsed IP spellings.

    This is a navigation boundary, not a replacement for an OS network sandbox.
    Browser subresources and public host DNS resolution remain browser-managed.
    """
    if not isinstance(url, str) or not url or len(url) > 16_384:
        raise ValueError("url must be a nonempty HTTP(S) URL")
    if any(ord(character) < 33 for character in url) or "\\" in url:
        raise ValueError("URL whitespace, control characters, and backslashes are forbidden")
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only absolute HTTP(S) URLs are allowed")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URLs containing credentials are forbidden")
    # Accessing .port also rejects malformed/non-numeric/out-of-range ports.
    _ = parsed.port
    hostname = parsed.hostname.lower().rstrip(".")
    if "%" in hostname or hostname == "localhost" or hostname.endswith(
        (".localhost", ".local", ".internal")
    ):
        raise ValueError("Local browser destinations are forbidden")
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        # WHATWG URL parsers accept legacy numeric hosts (127.1, 0177.0.0.1,
        # 2130706433, 0x7f000001) that Python's ipaddress intentionally rejects.
        if "." not in hostname or re.fullmatch(r"(?:[0-9]+|0x[0-9a-f]+)", hostname.rsplit(".", 1)[-1], re.I):
            raise ValueError("Local or ambiguous numeric browser destinations are forbidden")
    else:
        if not address.is_global:
            raise ValueError("Private, loopback, and non-public IP addresses are forbidden")
    return url


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def _tool(name: str, description: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "name": name, "description": description, "inputSchema": schema}


TOOLS = [
    _tool("browser_observe", "Observe the active browser tab with a screenshot and accessibility text.", _schema({})),
    _tool("browser_navigate", "Open a public HTTP(S) URL in the active tab.", _schema({"url": {"type": "string"}}, ["url"])),
    _tool("browser_click", "Click viewport pixel coordinates shown in the latest screenshot.", _schema({
        "x": {"type": "number", "minimum": 0}, "y": {"type": "number", "minimum": 0},
        "button": {"type": "string", "enum": ["left", "right", "middle"]},
        "click_count": {"type": "integer", "minimum": 1, "maximum": 3},
    }, ["x", "y"])),
    _tool("browser_type", "Insert literal text into the focused field or editor.", _schema({"text": {"type": "string", "maxLength": 100_000}}, ["text"])),
    _tool("browser_press", "Press a keyboard key or chord, e.g. Enter, Tab, ControlOrMeta+a, or ArrowDown.", _schema({"key": {"type": "string", "minLength": 1, "maxLength": 100}}, ["key"])),
    _tool("browser_scroll", "Scroll the active viewport by pixel deltas; positive delta_y scrolls down.", _schema({
        "delta_x": {"type": "number", "minimum": -10_000, "maximum": 10_000},
        "delta_y": {"type": "number", "minimum": -10_000, "maximum": 10_000},
    }, ["delta_y"])),
    _tool("browser_back", "Go back once in the active tab's navigation history.", _schema({})),
    _tool("browser_tabs", "List open browser tabs and their stable page IDs.", _schema({})),
    _tool("browser_new_tab", "Open a new tab at a public HTTP(S) URL and select it.", _schema({"url": {"type": "string"}}, ["url"])),
    _tool("browser_select_tab", "Select an open tab using its stable page ID.", _schema({"page_id": {"type": "string"}}, ["page_id"])),
    _tool("browser_wait", "Wait briefly for a page or editor to settle, then observe it.", _schema({"milliseconds": {"type": "integer", "minimum": 0, "maximum": 5000}}, ["milliseconds"])),
    _tool("browser_drag", "Drag from one viewport pixel coordinate to another, for selecting or moving slide content.", _schema({
        key: {"type": "number", "minimum": 0} for key in ["from_x", "from_y", "to_x", "to_y"]
    }, ["from_x", "from_y", "to_x", "to_y"])),
    _tool("browser_files", "List files downloaded in this episode and captured screenshot artifacts, with their upload IDs and readiness.", _schema({})),
    _tool("browser_upload", "Upload a file from browser_files to the pending file chooser, or an observed file input selected by a CSS/Playwright selector. Only this episode's artifact IDs are accepted.", _schema({
        "artifact_id": {"type": "string"}, "selector": {"type": "string", "minLength": 1, "maxLength": 500},
    }, ["artifact_id"])),
    _tool("browser_screenshot", "Save an uploadable PNG of the viewport or a visible rectangular region. Region requires all four x/y/width/height values. Returns its artifact ID.", _schema({
        "x": {"type": "number", "minimum": 0}, "y": {"type": "number", "minimum": 0},
        "width": {"type": "number", "minimum": 1}, "height": {"type": "number", "minimum": 1},
    })),
]


class BrowserSession:
    """One episode's isolated browser, observations, history, and action log."""

    tools = TOOLS

    def __init__(
        self,
        *,
        storage_state: str | Path | None = None,
        headless: bool = True,
        output_dir: str | Path,
        max_actions: int = 100,
        timeout_ms: int = 15_000,
        channel: str | None = "chrome",
        viewport: dict[str, int] | None = None,
    ) -> None:
        if isinstance(max_actions, bool) or not isinstance(max_actions, int) or max_actions < 1:
            raise ValueError("max_actions must be a positive integer")
        if timeout_ms < 1:
            raise ValueError("timeout_ms must be positive")
        self.storage_state = str(storage_state) if storage_state else None
        self.headless = headless
        self.output_dir = Path(output_dir)
        self.max_actions = max_actions
        self.timeout_ms = timeout_ms
        self.channel = channel
        self.viewport = viewport or {"width": 1440, "height": 900}
        self.action_count = 0
        self.history: list[str] = []
        self.navigation_events: list[dict[str, Any]] = []
        self.context: Any = None
        self._browser: Any = None
        self._playwright: Any = None
        self._pages: dict[str, Any] = {}
        self._active_page_id: str | None = None
        self._lock = asyncio.Lock()
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._download_tasks: set[asyncio.Task[Any]] = set()
        self._cleanup_tasks: set[asyncio.Task[Any]] = set()
        self._file_choosers: dict[str, Any] = {}
        self.cleanup_errors: list[str] = []
        self.auth_issue: dict[str, Any] | None = None

    async def __aenter__(self) -> BrowserSession:
        from playwright.async_api import async_playwright

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "screenshots").mkdir(exist_ok=True)
        self._playwright = await asyncio.wait_for(async_playwright().start(), timeout=self.timeout_ms / 1000)
        try:
            options = {"headless": self.headless, "timeout": self.timeout_ms}
            if self.channel:
                options["channel"] = self.channel
            self._browser = await asyncio.wait_for(
                self._playwright.chromium.launch(**options), timeout=(self.timeout_ms + 5000) / 1000
            )
            self.context = await asyncio.wait_for(self._browser.new_context(
                    storage_state=self.storage_state,
                    viewport=self.viewport,
                    accept_downloads=True,
                    service_workers="block",
                ), timeout=self.timeout_ms / 1000)
            self.context.set_default_timeout(self.timeout_ms)
            self.context.set_default_navigation_timeout(self.timeout_ms)
            await asyncio.wait_for(self.context.route("**/*", self._route), timeout=self.timeout_ms / 1000)
            self.context.on("page", self._attach_page)
            await asyncio.wait_for(
                self.context.tracing.start(screenshots=True, snapshots=True, sources=False), timeout=self.timeout_ms / 1000
            )
            page = await asyncio.wait_for(self.context.new_page(), timeout=self.timeout_ms / 1000)
            self._attach_page(page)
            return self
        except BaseException:
            await self._close()
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            await self.save_history()
            if self._download_tasks:
                try:
                    _, pending = await asyncio.wait(tuple(self._download_tasks), timeout=self.timeout_ms / 1000)
                    if pending:
                        raise TimeoutError("Downloads did not finish before cleanup")
                except Exception as error:
                    self._record_cleanup_error("finish downloads", error)
            if self.context:
                try:
                    await asyncio.wait_for(
                        self.context.tracing.stop(path=str(self.output_dir / "browser-trace.zip")), timeout=self.timeout_ms / 1000
                    )
                except Exception as error:
                    self._record_cleanup_error("save browser trace", error)
        finally:
            await self._close()

    def _record_cleanup_error(self, operation: str, error: Exception) -> None:
        self.cleanup_errors.append(f"{operation}: {type(error).__name__}: {error}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "cleanup-errors.json").write_text(json.dumps(self.cleanup_errors, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _consume_task_exception(task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            task.exception()

    async def save_storage_state(self, path: str | Path, timeout_seconds: float = 10.0) -> bool:
        """Export state privately and atomically; failed exports preserve the old file.

        A browser that ignores cancellation cannot hold up the caller. It never
        receives the destination path, so even a late result cannot replace it.
        External cancellation still propagates to the episode supervisor.
        """
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if self.auth_issue:
            self._record_cleanup_error("save storage state", RuntimeError("Google authentication requires attention; previous state preserved"))
            return False
        task = None
        temporary = None
        try:
            task = asyncio.create_task(self.context.storage_state())
            self._cleanup_tasks.add(task)
            task.add_done_callback(self._cleanup_tasks.discard)
            task.add_done_callback(self._consume_task_exception)
            done, _ = await asyncio.wait((task,), timeout=timeout_seconds)
            if not done:
                raise TimeoutError("Browser state export timed out")
            state = task.result()
            if self.auth_issue:
                raise BrowserAuthenticationError("Google authentication requires attention")
            if not isinstance(state, dict) or not all(isinstance(state.get(key), list) for key in ("cookies", "origins")):
                raise ValueError("Invalid browser storage state")
            # Serialize before opening a file so invalid data cannot touch the
            # prior state. mkstemp creates the temporary file with mode 0600.
            payload = json.dumps(state, ensure_ascii=False, allow_nan=False) + "\n"
            destination = Path(path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, filename = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
            temporary = Path(filename)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            temporary = None
            return True
        except asyncio.CancelledError:
            self._record_cleanup_error("save storage state", RuntimeError("Cancelled; previous state preserved"))
            raise
        except Exception as error:
            # Exception strings from browser transports can contain state or
            # page data. Record the error class only, never cookies or origins.
            self._record_cleanup_error("save storage state", RuntimeError(f"{type(error).__name__}; previous state preserved"))
            return False
        finally:
            if task is not None and not task.done():
                task.cancel()
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    async def _close(self) -> None:
        tasks = tuple(self._download_tasks | self._cleanup_tasks)
        for task in tasks:
            task.cancel()
            task.add_done_callback(self._consume_task_exception)
        try:
            if tasks:
                _, pending = await asyncio.wait(tasks, timeout=min(5.0, self.timeout_ms / 1000))
                if pending:
                    self._record_cleanup_error("cancel browser tasks", TimeoutError(f"{len(pending)} tasks did not stop"))
            if self._browser:
                try:
                    await asyncio.wait_for(self._browser.close(), timeout=self.timeout_ms / 1000)
                except Exception as error:
                    self._record_cleanup_error("close browser", error)
        finally:
            if self._playwright:
                try:
                    await asyncio.wait_for(self._playwright.stop(), timeout=self.timeout_ms / 1000)
                except Exception as error:
                    self._record_cleanup_error("stop Playwright", error)

    @property
    def page(self) -> Any:
        if self._active_page_id:
            page = self._pages[self._active_page_id]
            if not page.is_closed():
                return page
        for page_id, page in reversed(list(self._pages.items())):
            if not page.is_closed():
                self._active_page_id = page_id
                return page
        raise RuntimeError("No browser tabs are open")

    @property
    def page_id(self) -> str:
        _ = self.page
        assert self._active_page_id is not None
        return self._active_page_id

    async def _route(self, route: Any) -> None:
        request = route.request
        if request.is_navigation_request():
            try:
                validate_url(request.url)
            except ValueError:
                await route.abort("blockedbyclient")
                return
        await route.continue_()

    def _attach_page(self, page: Any) -> None:
        for page_id, existing in self._pages.items():
            if existing is page:
                self._active_page_id = page_id
                return
        page_id = f"page_{len(self._pages) + 1}"
        self._pages[page_id] = page
        self._active_page_id = page_id

        def on_navigation(frame: Any) -> None:
            if frame is page.main_frame:
                self._record_navigation(page_id, frame.url)

        page.on("framenavigated", on_navigation)
        page.on("filechooser", lambda chooser: self._file_choosers.__setitem__(page_id, chooser))
        page.on("download", self._start_download)
        # Popups can already have committed their first document when Playwright
        # emits the context's page event; preserve that initial destination too.
        if page.url and page.url != "about:blank":
            self._record_navigation(page_id, page.url)

    def _record_navigation(self, page_id: str, url: str) -> None:
        if not url or url == "about:blank":
            return
        self.history.append(url)
        event = {"page_id": page_id, "url": url, "timestamp": datetime.now(timezone.utc).isoformat()}
        self.navigation_events.append(event)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "navigations.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._capture_auth_issue(page_id, url)

    def _capture_auth_issue(self, page_id: str, url: str) -> None:
        location = _google_auth_location(url)
        if location is None or self.auth_issue is not None:
            return
        timestamp = datetime.now(timezone.utc).isoformat()
        self.auth_issue = {"status": "authentication_required", "auth_status": "login_required",
                           **location, "auth_location": location, "page_id": page_id,
                           "timestamp": timestamp, "checked_at": timestamp}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "auth-health.json").write_text(
            json.dumps(self.auth_issue, indent=2) + "\n", encoding="utf-8")

    def assert_authenticated(self) -> None:
        """Reject observed sign-in pages without inspecting links or page text.

        Top-level navigation evidence is sticky even if a popup later closes or
        the active tab changes. Missing tabs remain ordinary browser errors.
        """
        if self._pages:
            try:
                page = self.page
            except RuntimeError:
                page = None
            if page is not None:
                self._capture_auth_issue(self.page_id, page.url)
        if self.auth_issue:
            raise BrowserAuthenticationError(
                f"Google browser authentication requires attention ({self.auth_issue['host']}{self.auth_issue['path']})")

    async def save_history(self, path: str | Path | None = None) -> None:
        destination = Path(path) if path else self.output_dir / "history.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.history, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def _artifact_metadata(self, artifact: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in artifact.items() if key != "path"}

    def _save_artifacts(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "artifacts.json").write_text(
            json.dumps([self._artifact_metadata(item) for item in self._artifacts.values()], indent=2) + "\n",
            encoding="utf-8",
        )

    def _new_artifact(self, kind: str, filename: str) -> dict[str, Any]:
        artifact_id = f"{kind}_{len(self._artifacts) + 1:05d}"
        filename = Path(filename.replace("\\", "/")).name
        filename = re.sub(r"[\x00-\x1f\x7f]", "_", filename).encode("utf-8")[:180].decode("utf-8", errors="ignore")
        if filename in {"", ".", ".."}:
            filename = "download.bin"
        # Isolate each artifact in its own directory so uploading preserves the
        # website's filename and duplicate names never overwrite earlier files.
        directory = self.output_dir / ("downloads" if kind == "download" else "screenshots") / artifact_id
        directory.mkdir(parents=True, exist_ok=True)
        artifact = {
            "artifact_id": artifact_id, "kind": kind, "filename": filename,
            "path": directory / filename, "status": "pending",
        }
        self._artifacts[artifact_id] = artifact
        self._save_artifacts()
        return artifact

    def _start_download(self, download: Any) -> None:
        artifact = self._new_artifact("download", download.suggested_filename)
        task = asyncio.create_task(self._save_download(download, artifact))
        self._download_tasks.add(task)
        task.add_done_callback(self._download_tasks.discard)

    async def _save_download(self, download: Any, artifact: dict[str, Any]) -> None:
        try:
            await asyncio.wait_for(download.save_as(str(artifact["path"])), timeout=self.timeout_ms / 1000)
            artifact.update(status="ready", bytes=artifact["path"].stat().st_size)
        except asyncio.CancelledError:
            artifact.update(status="failed", error="Download canceled when the episode ended")
            raise
        except Exception as error:
            artifact.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            self._save_artifacts()

    def _validate_arguments(self, name: str, arguments: Any) -> dict[str, Any]:
        spec = next((item for item in self.tools if item["name"] == name), None)
        if spec is None:
            raise ValueError(f"Unknown browser tool: {name}")
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be an object")
        schema = spec["inputSchema"]
        unknown = set(arguments) - set(schema["properties"])
        missing = set(schema["required"]) - set(arguments)
        if unknown or missing:
            raise ValueError(f"Invalid arguments; missing={sorted(missing)}, unknown={sorted(unknown)}")
        for key, value in arguments.items():
            rule = schema["properties"][key]
            kind = rule["type"]
            if kind == "string":
                if not isinstance(value, str):
                    raise ValueError(f"{key} must be a string")
                if len(value) < rule.get("minLength", 0) or len(value) > rule.get("maxLength", 100_000):
                    raise ValueError(f"{key} has invalid length")
            elif kind in {"number", "integer"}:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{key} must be a finite number")
                if kind == "integer" and not isinstance(value, int):
                    raise ValueError(f"{key} must be an integer")
                if value < rule.get("minimum", -math.inf) or value > rule.get("maximum", math.inf):
                    raise ValueError(f"{key} is outside the allowed range")
            if "enum" in rule and value not in rule["enum"]:
                raise ValueError(f"{key} is not one of the allowed values")
        return arguments

    async def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:
            self.assert_authenticated()
            if self.action_count >= self.max_actions:
                raise ActionBudgetExceeded(f"Browser action budget exhausted ({self.max_actions})")
            self.action_count += 1
            success = True
            message = ""
            try:
                arguments = self._validate_arguments(name, arguments)
                message = await asyncio.wait_for(
                    self._execute(name, arguments), timeout=(self.timeout_ms + 5000) / 1000
                )
            except ActionBudgetExceeded:
                raise
            except Exception as error:
                success = False
                message = f"{type(error).__name__}: {error}"
            content = [{"type": "inputText", "text": message}] if message else []
            try:
                content.extend(await asyncio.wait_for(
                    self._observe(), timeout=(self.timeout_ms + 5000) / 1000
                ))
            except Exception as error:
                success = False
                content.append({"type": "inputText", "text": f"Observation failed: {type(error).__name__}: {error}"})
            record = {
                "action": self.action_count, "tool": name, "arguments": arguments,
                "success": success, "message": message,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "actions.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            await self.save_history()
            self.assert_authenticated()
            return {"success": success, "contentItems": content}

    async def _execute(self, name: str, args: dict[str, Any]) -> str:
        if name == "browser_navigate":
            await self.page.goto(validate_url(args["url"]), wait_until="domcontentloaded")
        elif name == "browser_new_tab":
            url = validate_url(args["url"])
            page = await self.context.new_page()
            self._attach_page(page)
            await page.goto(url, wait_until="domcontentloaded")
        elif name == "browser_click":
            if args["x"] >= self.viewport["width"] or args["y"] >= self.viewport["height"]:
                raise ValueError("Click coordinates must be inside the screenshot viewport")
            await self.page.mouse.click(args["x"], args["y"], button=args.get("button", "left"), click_count=args.get("click_count", 1))
        elif name == "browser_type":
            await self.page.keyboard.insert_text(args["text"])
        elif name == "browser_press":
            await self.page.keyboard.press(args["key"])
        elif name == "browser_scroll":
            await self.page.mouse.wheel(args.get("delta_x", 0), args["delta_y"])
        elif name == "browser_back":
            await self.page.go_back(wait_until="domcontentloaded")
        elif name == "browser_tabs":
            tabs = []
            for page_id, page in self._pages.items():
                if not page.is_closed():
                    tabs.append({"page_id": page_id, "url": page.url, "active": page_id == self.page_id})
            return json.dumps({"tabs": tabs}, ensure_ascii=False)
        elif name == "browser_select_tab":
            page_id = args["page_id"]
            if page_id not in self._pages or self._pages[page_id].is_closed():
                raise ValueError(f"No open tab with page ID {page_id}")
            self._active_page_id = page_id
            await self.page.bring_to_front()
        elif name == "browser_wait":
            await asyncio.sleep(args["milliseconds"] / 1000)
        elif name == "browser_drag":
            for prefix in ("from", "to"):
                if args[f"{prefix}_x"] >= self.viewport["width"] or args[f"{prefix}_y"] >= self.viewport["height"]:
                    raise ValueError("Drag coordinates must be inside the screenshot viewport")
            await self.page.mouse.move(args["from_x"], args["from_y"])
            await self.page.mouse.down()
            try:
                await self.page.mouse.move(args["to_x"], args["to_y"], steps=10)
            finally:
                await self.page.mouse.up()
        elif name == "browser_files":
            return json.dumps({"files": [self._artifact_metadata(item) for item in self._artifacts.values()]})
        elif name == "browser_upload":
            artifact = self._artifacts.get(args["artifact_id"])
            if not artifact or artifact["status"] != "ready":
                raise ValueError("Upload requires a ready artifact ID from browser_files")
            path = artifact["path"].resolve()
            if not path.is_relative_to(self.output_dir.resolve()) or not path.is_file():
                raise ValueError("Artifact is missing or outside this episode's directory")
            if "selector" in args:
                await self.page.locator(args["selector"]).set_input_files(str(path))
            else:
                chooser = self._file_choosers.get(self.page_id)
                if chooser is None:
                    raise ValueError("No file chooser is pending; click Upload or provide an observed file input selector")
                await chooser.set_files(str(path))
                self._file_choosers.pop(self.page_id, None)
            return json.dumps({"uploaded_artifact_id": artifact["artifact_id"]})
        elif name == "browser_screenshot":
            if args and set(args) != {"x", "y", "width", "height"}:
                raise ValueError("Screenshot region requires x, y, width, and height together")
            if args and (args["x"] + args["width"] > self.viewport["width"] or args["y"] + args["height"] > self.viewport["height"]):
                raise ValueError("Screenshot region must be inside the viewport")
            artifact = self._new_artifact("screenshot", "capture.png")
            try:
                await self.page.screenshot(
                    path=str(artifact["path"]), type="png", timeout=self.timeout_ms, clip=args or None
                )
                artifact.update(status="ready", bytes=artifact["path"].stat().st_size)
            except Exception:
                artifact.update(status="failed", error="Screenshot capture failed")
                raise
            finally:
                self._save_artifacts()
            return json.dumps(self._artifact_metadata(artifact))
        return ""

    async def _observe(self) -> list[dict[str, str]]:
        page = self.page
        page_id = self.page_id
        # An editor's animation or network activity must not consume an unbounded
        # wait. Pixel state is authoritative; the accessibility tree is auxiliary.
        screenshot_path = self.output_dir / "screenshots" / f"{self.action_count:05d}_{page_id}.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot = await page.screenshot(path=str(screenshot_path), type="png", timeout=self.timeout_ms)
        try:
            aria = await page.locator("body").aria_snapshot(timeout=min(3000, self.timeout_ms))
        except Exception as error:
            aria = f"[Accessibility snapshot unavailable: {type(error).__name__}]"
        metadata = {
            "page_id": page_id, "url": page.url,
            "viewport": self.viewport, "action": self.action_count,
            "actions_remaining": self.max_actions - self.action_count,
            "file_chooser_pending": page_id in self._file_choosers,
            "files": [self._artifact_metadata(item) for item in self._artifacts.values()],
            "accessibility_snapshot": aria[:12_000],
        }
        return [
            {"type": "inputText", "text": json.dumps(metadata, ensure_ascii=False)},
            {"type": "inputImage", "imageUrl": "data:image/png;base64," + base64.b64encode(screenshot).decode("ascii")},
        ]
