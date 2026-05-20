try:
    from browsergym.core.registration import register_task
    from browsergym.core.task import AbstractBrowserTask
except ImportError:
    register_task = None
    AbstractBrowserTask = object
from pathlib import Path
try:
    import playwright.sync_api
except ImportError:
    playwright = None
from typing import Tuple, List, Dict, Any, Optional
import inspect
import time
import os
import pathlib
import json
import importlib
import importlib.util
import sys
import subprocess
import types
import re
from abc import abstractmethod

# Recognized workspace kinds (i.e. which Google app the task targets). Used to
# pick the right doc-creation URL and the right id-extraction regex.
WORKSPACE_KIND_DOCS = "docs"
WORKSPACE_KIND_SHEETS = "sheets"
WORKSPACE_KIND_SLIDES = "slides"

# URL fragment used by Drive for each kind. Same key set as above.
_WORKSPACE_URL_SEGMENT = {
    WORKSPACE_KIND_DOCS: "document",
    WORKSPACE_KIND_SHEETS: "spreadsheets",
    WORKSPACE_KIND_SLIDES: "presentation",
}

# Regex that matches a Drive-hosted file id for any workspace kind. Captures
# the id in group(2); group(1) is the URL segment that disambiguates the kind.
_WORKSPACE_ID_RE = re.compile(
    r"/(document|spreadsheets|presentation)/d/([a-zA-Z0-9_-]+)"
)

# Root of the bundled `eval/` directory shipped inside this package.
_PACKAGE_DIR = pathlib.Path(__file__).resolve().parent
EVAL_TASKS_DIR = _PACKAGE_DIR / "eval" / "tasks"


def _install_eval_import_shim() -> None:
    """Make legacy ``src.browsergym.*`` eval imports resolve locally.

    Evaluator files under ``eval/tasks/.../evaluator.py`` and the
    ``eval/eval_utils/*`` modules were authored before the eval tree was
    moved into this Python package. They still use ``src.browsergym.eval...``
    and ``src.browsergym.knows.eval...``
    paths. Rather than rewrite every legacy import, we install a meta-path
    finder that transparently redirects those namespaces to the real,
    importable ``browsergym.knows.eval`` package.
    """
    if any(getattr(f, "_knows_eval_shim", False) for f in sys.meta_path):
        return

    # Ensure parent stub packages exist so attribute lookups inside the
    # interpreter don't fail before our finder runs.
    for stub_name in ("src", "src.browsergym"):
        if stub_name not in sys.modules:
            stub = types.ModuleType(stub_name)
            stub.__path__ = []  # mark as namespace package
            sys.modules[stub_name] = stub
            parent_name, _, child = stub_name.rpartition(".")
            if parent_name:
                setattr(sys.modules[parent_name], child, stub)

    _ALIASES = {
        "src.browsergym.knows": "browsergym.knows",
        "src.browsergym.knows.eval": "browsergym.knows.eval",
        "src.browsergym.eval": "browsergym.knows.eval",
    }

    class _AliasLoader:
        """Loader that returns an already-imported module instead of executing
        new code. We use this together with the finder below to expose
        ``browsergym.knows.*`` modules under legacy ``src.browsergym.*`` names.
        """

        def __init__(self, real_mod):
            self._real_mod = real_mod

        def create_module(self, spec):
            return self._real_mod

        def exec_module(self, module):
            return None

    class _KnowsEvalShim:
        """Meta-path finder that redirects legacy eval namespaces."""

        _knows_eval_shim = True

        def find_spec(self, fullname, path=None, target=None):
            real_name = None
            for old, new in _ALIASES.items():
                if fullname == old or fullname.startswith(old + "."):
                    real_name = new + fullname[len(old):]
                    break
            if real_name is None:
                return None
            try:
                real_mod = importlib.import_module(real_name)
            except Exception:
                return None
            # Also expose the alias as an attribute of its parent so plain
            # attribute lookups keep working (e.g. ``src.browsergym.eval``).
            parent_name, _, child = fullname.rpartition(".")
            if parent_name in sys.modules:
                setattr(sys.modules[parent_name], child, real_mod)
            return importlib.util.spec_from_loader(fullname, _AliasLoader(real_mod))

    sys.meta_path.insert(0, _KnowsEvalShim())


_install_eval_import_shim()
class KnowsBenchTask(AbstractBrowserTask):
    """
    BrowserGym task class that inherits from AbstractBrowserTask.
    This class is used to define a browser task for the BrowserGym framework.
    
    Provides functionality to track visited websites during task execution,
    measure performance metrics, and maintain browser session state.
    """

    # URL fragments that indicate Google has bounced the browser out to a
    # sign-in / account-chooser surface (typically because Google's passive
    # re-auth check fired mid-episode and the WebLiteSignIn flow rejected
    # the automated browser). Once we see these, the agent cannot recover
    # on its own and any further steps are wasted budget.
    _AUTH_LOST_URL_FRAGMENTS: Tuple[str, ...] = (
        "/v3/signin/rejected",
        "/v3/signin/accountchooser",
        "AccountChooser",
        "/signin/identifier",
        "/v3/signin/identifier",
        "/v3/signin/confirmidentifier",
        "ServiceLogin",
        "deniedsigninrejected",
    )

    # How many consecutive ``validate()`` calls must observe a sign-in URL
    # before we declare the session lost. Two avoids spuriously aborting on
    # a single transient redirect that Google sometimes resolves on its own
    # (e.g. an OAuth bounce that completes within one step).
    _AUTH_LOST_CONSECUTIVE_THRESHOLD: int = 2

    @classmethod
    def get_task_id(cls) -> str:
        """
        Generic class for several task ids, this way of obtaining the task id is not compatible for now.
        """
        raise NotImplementedError
    
    def __init__(self, seed: int = 0, task_name: str = "browser_task", 
                user_data_dir: Optional[str] = None,
                persistent_context: bool = False):
        super().__init__(seed)
        self._task_name = task_name
        
        # Track visited URLs
        self._visited_urls: List[Dict[str, Any]] = []
        self._start_time: float = 0
        self._end_time: Optional[float] = None
        
        # Browser session configuration
        self._user_data_dir = user_data_dir
        self._persistent_context = persistent_context
        
        # Task configuration
        self.viewport = {"width": 1280, "height": 720}
        self.timeout = 30000  # ms (30 seconds)
        
        # Prepare browser context options
        self.pw_chromium_kwargs = {}
        self.pw_context_kwargs = {}

        # Mid-episode auth-loss tracking. Updated by ``validate()`` every
        # step; ``_auth_lost`` flips to True after the URL has stayed on a
        # sign-in surface for ``_AUTH_LOST_CONSECUTIVE_THRESHOLD`` calls
        # in a row, at which point validate() returns done=True so the
        # episode terminates instead of looping through the chooser.
        self._consecutive_auth_lost_observations: int = 0
        self._auth_lost: bool = False
        self._auth_lost_url: Optional[str] = None
        
        # Configure persistent browser session if requested
        if self._persistent_context and self._user_data_dir:
            self.pw_context_kwargs = {
                "user_data_dir": self._user_data_dir,
                "persistent_context": True,
                "headless" : False
            }

    def _is_auth_lost_url(self, url: str) -> bool:
        """Return True iff *url* is on a Google sign-in / chooser surface.

        Used by :meth:`validate` to detect mid-episode session eviction.
        Substring match against :attr:`_AUTH_LOST_URL_FRAGMENTS` -- Google
        cycles between ``/signin/accountchooser`` and ``/signin/rejected``
        on a blocked re-auth, and we want to catch both.
        """
        if not url:
            return False
        return any(fragment in url for fragment in self._AUTH_LOST_URL_FRAGMENTS)
        
    def _track_url_visit(self, page: playwright.sync_api.Page):
        """
        Track URL visits by adding an event listener to the page.
        
        Args:
            page: the active playwright page.
        """
        # Add event listener for navigation
        def _on_navigated(frame):
            if frame != page.main_frame:
                return
            try:
                title = frame.title()
            except Exception:
                title = None
            self._visited_urls.append({
                "url": frame.url,
                "timestamp": time.time(),
                "title": title,
            })

        page.on("framenavigated", _on_navigated)
        
        # Set start time when tracking begins
        self._start_time = time.time()

    def setup(self, page: playwright.sync_api.Page) -> tuple[str, dict]:
        """
        Set up everything needed to execute the task.
        
        This includes:
        - Tracking URL visits
        - Checking authentication status for services like Google
        - Loading task-specific configuration
        - Navigating to the starting URL if needed

        Args:
            page: the active playwright page.

        Returns:
            goal: str, goal of the task.
            info: dict, custom information from the task.
        """
        # Start tracking URL visits
        self._track_url_visit(page)
        
        # Check if we're using a persistent context and need to verify login status
        auth_status = {}
        if self._persistent_context and self._user_data_dir:
            # Check Google login if needed for this task
            if hasattr(self, 'requires_google_auth') and self.requires_google_auth:
                try:
                    # Verify Google auth by quickly checking accounts page
                    page.goto("https://accounts.google.com/", timeout=self.timeout)
                    page.wait_for_load_state("networkidle")
                    
                    # If redirected to myaccount.google.com or no login form is present, we're logged in
                    is_logged_in = "myaccount.google.com" in page.url or not page.query_selector('form[data-identitybrowser]')
                    auth_status['google_authenticated'] = is_logged_in
                    
                    if not is_logged_in:
                        print("Warning: Google authentication required but not detected in browser session")
                except Exception as e:
                    print(f"Error checking Google authentication: {str(e)}")
                    auth_status['google_authenticated'] = False
        
        # Load task-specific description if available.
        # `self._task_name` is a relative path under the bundled `eval/tasks/`
        # directory, e.g. "docs_1_formal_letter/instance_2".
        task_desc_path = EVAL_TASKS_DIR / self._task_name / "task.md"

        task_description = f"Complete the {self._task_name} task"
        if task_desc_path.exists():
            with open(task_desc_path, "r") as f:
                task_description = f.read()

        # Append run-time guidance injected via env var (set by run.sh).
        # Lets us add sign-in fallbacks / account credentials / etc. without
        # editing each per-instance task.md.
        extra_instructions = os.environ.get("BROWSERGYM_EXTRA_GOAL_INSTRUCTIONS", "").strip()
        if extra_instructions:
            task_description = (
                task_description.rstrip() + "\n\n---\n" + extra_instructions
            )

        # Return the task description and setup information
        info = {
            "task_name": self._task_name,
            "auth_status": auth_status
        }
        
        # This is a base implementation
        # Subclasses should override this method with their specific setup logic
        # but should call super().setup(page) to ensure URL tracking is initialized
        return task_description, info

    def validate(
        self, page: playwright.sync_api.Page, chat_messages: list[str]
    ) -> Tuple[float, bool, str, dict]:
        """
        Validate the task was completed successfully

        Args:
            page: the active playwright page.
            chat_messages: the chat messages.

        Returns:
            reward: float, the reward obtained since last call to validate().
            done: boolean flag, indicates if the task has finished or not (be it success or fail).
            message: string, a new user message for the chat.
            info: dictionnary, custom information from the task.
        """
        # This is a base implementation
        # Subclasses should override this method with their specific validation logic
        reward, done, message = 0.0, False, ""

        try:
            current_url = page.url or ""
        except Exception:
            current_url = ""

        # Include browsing history in the info dictionary
        info = {
            "visited_urls": self._visited_urls,
            "total_pages_visited": len(self._visited_urls),
            "current_url": current_url,
        }

        # Mid-episode auth-loss detection. Once Google's passive re-auth
        # bounces the browser to a sign-in / chooser URL, the WebLiteSignIn
        # flow rejects the automated browser and the agent burns the rest
        # of its budget cycling through the chooser. End the episode early
        # so the run is flagged as an auth failure rather than scored as
        # a 0-reward task failure. We require the URL to persist across
        # ``_AUTH_LOST_CONSECUTIVE_THRESHOLD`` validate() calls so a single
        # transient redirect (e.g. an OAuth bounce) does not abort.
        if self._is_auth_lost_url(current_url):
            self._consecutive_auth_lost_observations += 1
            if (
                self._consecutive_auth_lost_observations
                >= self._AUTH_LOST_CONSECUTIVE_THRESHOLD
                and not self._auth_lost
            ):
                self._auth_lost = True
                self._auth_lost_url = current_url
                done = True
                message = (
                    "Episode terminated: Google sign-out detected "
                    f"(URL={current_url!r}). The persistent session was "
                    "invalidated mid-episode and the automated browser "
                    "cannot recover; subsequent steps would loop through "
                    "the account chooser."
                )
                info["auth_lost_mid_episode"] = True
                info["auth_lost_url"] = current_url
                info["auth_lost_observed_steps"] = (
                    self._consecutive_auth_lost_observations
                )
        else:
            self._consecutive_auth_lost_observations = 0

        return reward, done, message, info
    
    def teardown(self) -> None:
        """
        Tear down the task and clean up any resources created by the task.
        """
        # Record end time
        self._end_time = time.time()
        
    def get_browsing_history(self) -> List[Dict[str, Any]]:
        """
        Get the list of visited URLs during this task.
        
        Returns:
            List of dictionaries containing url, timestamp, and title information.
        """
        return self._visited_urls
    
    def get_task_duration(self) -> float:
        """
        Get the duration of the task execution in seconds.
        
        Returns:
            Task duration in seconds, or time elapsed so far if task is not complete.
        """
        end_time = self._end_time or time.time()
        return end_time - self._start_time if self._start_time > 0 else 0
    
    def cheat(self, page: playwright.sync_api.Page, chat_messages: list[str]) -> None:
        """
        Solve the task using a pre-defined solution (optional).
        
        Subclasses should override this method with their specific cheat logic.
        """
        raise NotImplementedError("Cheat functionality not implemented for this task")
    

class KnowsWorkspaceTask(KnowsBenchTask):
    """Generic KNOWS task that pre-creates a Google workspace file for the
    agent (a Doc / Sheet / Slides deck depending on :py:attr:`WORKSPACE_KIND`)
    and can grade it at episode end via the bundled evaluator.

    Subclasses configure the task family by setting class attributes:

    - :py:attr:`TASK_FAMILY_FOLDER` — directory name under
      ``browsergym/knows/eval/tasks/`` that holds ``instance_<n>/`` folders
      with ``task.md`` + ``evaluator.py``.
    - :py:attr:`TASK_ID_PREFIX` — gym task id prefix without the trailing
      instance number (e.g. ``"knows.docs_1_formal_letter"``).
    - :py:attr:`WORKSPACE_KIND` — ``"docs"``, ``"sheets"`` or ``"slides"``;
      controls which Google app is opened in :py:meth:`setup` and how doc
      ids are extracted from URLs.
    - :py:attr:`AVAILABLE_INSTANCES` — tuple of numeric instance ids that
      have been bundled with this package.
    """

    requires_google_auth = True

    # ---- Subclass configuration ----------------------------------------
    TASK_FAMILY_FOLDER: str = ""  # e.g. "docs_1_formal_letter"
    TASK_ID_PREFIX: str = ""  # e.g. "knows.docs_1_formal_letter"
    WORKSPACE_KIND: str = WORKSPACE_KIND_DOCS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)

    @classmethod
    def get_task_id(cls) -> str:
        if not cls.TASK_ID_PREFIX:
            raise NotImplementedError(
                f"{cls.__name__} must set TASK_ID_PREFIX (e.g. "
                "'knows.<task_family>')."
            )
        return cls.TASK_ID_PREFIX

    def __init__(
        self,
        seed: int = 0,
        instance_id: int = 1,
        task_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        user_data_dir: Optional[str] = None,
        persistent_context: bool = False,
        run_evaluator: bool = False,
        existing_doc_id: Optional[str] = None,
    ):
        if not self.TASK_FAMILY_FOLDER:
            raise NotImplementedError(
                f"{type(self).__name__} must set TASK_FAMILY_FOLDER (the "
                "directory name under eval/tasks/)."
            )
        if instance_id not in self.AVAILABLE_INSTANCES:
            raise ValueError(
                f"Unknown {self.TASK_FAMILY_FOLDER} instance_id={instance_id}; "
                f"expected one of {self.AVAILABLE_INSTANCES}."
            )
        if task_name is None:
            task_name = f"{self.TASK_FAMILY_FOLDER}/instance_{instance_id}"

        super().__init__(
            seed=seed,
            task_name=task_name,
            user_data_dir=user_data_dir,
            persistent_context=persistent_context,
        )
        self._instance_id = instance_id
        self._agent_name = agent_name
        self._existing_doc_id = existing_doc_id.strip() if existing_doc_id else None
        # Benchmark runs skip grading by default; callers can opt in when
        # they need evaluator scores instead of just task artifacts.
        self._run_evaluator = run_evaluator
        self._created_doc_id: Optional[str] = None
        self._created_doc_url: Optional[str] = None
        # Tracks whether evaluation has been handled for this episode so we
        # never grade twice (validate() may be called multiple times, and
        # finalize() may be invoked at episode end as a safety net).
        self._graded: bool = False
        # The score breakdown produced by the most recent grading run, so we
        # can re-attach it to the final step's task_info if grading happened
        # in finalize() rather than validate().
        self._last_score_breakdown: Dict[str, Any] = {}
        self._last_reward: float = 0.0

    # Human-readable label of the workspace kind, used in goal text.
    _WORKSPACE_LABEL = {
        WORKSPACE_KIND_DOCS: "Google Doc",
        WORKSPACE_KIND_SHEETS: "Google Sheet",
        WORKSPACE_KIND_SLIDES: "Google Slides presentation",
    }

    def setup(self, page: playwright.sync_api.Page) -> Tuple[str, dict]:
        goal, info = super().setup(page)

        if self._existing_doc_id:
            doc_id = self._existing_doc_id
            doc_url = self._workspace_url(doc_id)
            self._created_doc_id = doc_id
            self._created_doc_url = doc_url
            info["doc_id"] = doc_id
            info["created_doc_id"] = doc_id
            info["created_doc_url"] = doc_url
            info["existing_doc_id"] = doc_id
            info["existing_doc_url"] = doc_url

            workspace_label = self._WORKSPACE_LABEL.get(
                self.WORKSPACE_KIND, "Google workspace file"
            )
            goal = (
                goal.rstrip()
                + "\n\n---\n"
                + f"Continue this task in the existing {workspace_label}; "
                + "do NOT create a new file and do NOT rename it.\n"
                + f"File URL: {doc_url}\n"
                + "When you are finished, reply with 'DONE' and include "
                + "this URL in your message."
            )
        elif self._agent_name:
            try:
                from .doc_setup import create_task_workspace

                doc_id, doc_url = create_task_workspace(
                    page,
                    agent_name=self._agent_name,
                    task_name=self.get_task_id(),
                    instance_id=self._instance_id,
                    kind=self.WORKSPACE_KIND,
                )
                self._created_doc_id = doc_id
                self._created_doc_url = doc_url
                info["created_doc_id"] = doc_id
                info["created_doc_url"] = doc_url

                workspace_label = self._WORKSPACE_LABEL.get(
                    self.WORKSPACE_KIND, "Google workspace file"
                )
                goal = (
                    goal.rstrip()
                    + "\n\n---\n"
                    + f"A {workspace_label} has already been created for you. "
                    + "You MUST complete this task inside that file; "
                    + "do NOT create a new file and do NOT rename it.\n"
                    + f"File URL: {doc_url}\n"
                    + "When you are finished, reply with 'DONE' and include "
                    + "this URL in your message."
                )
            except Exception as exc:
                print(
                    f"Workspace pre-creation failed "
                    f"({self.TASK_FAMILY_FOLDER} instance={self._instance_id}): {exc}"
                )
                info["doc_creation_error"] = str(exc)

        return goal, info

    def _workspace_url(self, doc_id: str) -> str:
        if self.WORKSPACE_KIND == WORKSPACE_KIND_SHEETS:
            return f"https://docs.google.com/spreadsheets/d/{doc_id}/edit"
        if self.WORKSPACE_KIND == WORKSPACE_KIND_SLIDES:
            return f"https://docs.google.com/presentation/d/{doc_id}/edit"
        return f"https://docs.google.com/document/d/{doc_id}/edit"

    def validate(
        self, page: playwright.sync_api.Page, chat_messages: list
    ) -> Tuple[float, bool, str, dict]:
        last_text = ""
        last_role = ""
        if chat_messages:
            last = chat_messages[-1]
            last_text = last if isinstance(last, str) else last.get("message", "")
            last_role = last.get("role", "") if isinstance(last, dict) else ""

        # Episode is "done from the agent's side" when:
        #  - assistant explicitly signals completion with DONE, or
        #  - the agent reports the task infeasible (gives up), which BrowserGym
        #    surfaces as a chat message with role='infeasible'.
        agent_done = (last_role == "assistant" and "DONE" in last_text) or (
            last_role == "infeasible"
        )
        if agent_done:
            doc_id = self._created_doc_id or self._extract_doc_id(last_text, page.url)
            # If we already detected mid-episode auth loss, prefer that
            # reason over "agent_done" -- the agent's DONE on a sign-out
            # page is meaningless and we don't want to grade it as a
            # legitimate completion.
            done_reason = (
                "auth_lost_mid_episode" if self._auth_lost else "agent_done"
            )
            if not self._run_evaluator:
                info = self._evaluation_skipped_info(doc_id, reason=done_reason)
                if self._auth_lost:
                    info["auth_lost_mid_episode"] = True
                    if self._auth_lost_url:
                        info["auth_lost_url"] = self._auth_lost_url
                self._graded = True
                self._last_score_breakdown = info
                self._last_reward = 0.0
                return 0.0, True, "", info
            reward, info = self._grade_doc(doc_id, last_text, page)
            if self._auth_lost:
                info["auth_lost_mid_episode"] = True
                if self._auth_lost_url:
                    info["auth_lost_url"] = self._auth_lost_url
            return reward, True, "", info

        return super().validate(page, chat_messages)

    def finalize(
        self, page: Optional[playwright.sync_api.Page] = None
    ) -> Optional[Dict[str, Any]]:
        """Grade the document at episode end if grading hasn't happened yet.

        The runner (AgentLab loop) calls this AFTER the episode loop exits
        (DONE / infeasible / truncation / empty-action / error) so that
        per-checkpoint scores and an aggregated reward always end up in
        ``task_info.json`` and ``summary_info.json`` — even when the agent
        never produced a clean DONE message.

        Returns the score-breakdown dict (with ``eval.*`` flat keys plus
        nested ``checkpoints`` / ``evaluation`` payloads) when grading was
        actually performed in this call; ``None`` if grading was already
        done during ``validate()``.
        """
        if self._graded:
            return None

        page_url = ""
        if page is not None:
            try:
                page_url = page.url
            except Exception:
                page_url = ""
        doc_id = self._created_doc_id or (
            self._extract_doc_id("", page_url) if page_url else None
        )
        # Prefer the auth-loss reason over the generic "episode ended
        # without DONE" so the summary clearly distinguishes environmental
        # failures (Google bounced us out) from agent failures (ran out of
        # steps / gave up).
        finalize_reason = (
            "auth_lost_mid_episode" if self._auth_lost
            else "episode_end_without_done"
        )

        if not self._run_evaluator:
            info = self._evaluation_skipped_info(doc_id, reason=finalize_reason)
            info["finalize.reason"] = finalize_reason
            if self._auth_lost:
                info["auth_lost_mid_episode"] = True
                if self._auth_lost_url:
                    info["auth_lost_url"] = self._auth_lost_url
            self._graded = True
            self._last_score_breakdown = info
            self._last_reward = 0.0
            return info

        reward, info = self._grade_doc(doc_id, "", page)
        info["finalize.reason"] = finalize_reason
        info["cum_reward_override"] = reward
        if self._auth_lost:
            info["auth_lost_mid_episode"] = True
            if self._auth_lost_url:
                info["auth_lost_url"] = self._auth_lost_url
        return info

    def _evaluation_skipped_info(
        self, doc_id: Optional[str], reason: str
    ) -> Dict[str, Any]:
        return {
            "visited": self._visited_urls,
            "doc_id": doc_id,
            "instance_id": self._instance_id,
            "task_family": self.TASK_FAMILY_FOLDER,
            "workspace_kind": self.WORKSPACE_KIND,
            "created_doc_id": self._created_doc_id,
            "created_doc_url": self._created_doc_url,
            "evaluation_skipped": True,
            "evaluation_skip_reason": reason,
        }

    def _grade_doc(
        self,
        doc_id: Optional[str],
        last_text: str,
        page: Optional[playwright.sync_api.Page],
    ) -> Tuple[float, Dict[str, Any]]:
        """Run the evaluator on *doc_id* and return ``(reward, info)``.

        Always sets ``self._graded = True`` after this returns so subsequent
        calls (validate / finalize) become no-ops.
        """
        info: Dict[str, Any] = {
            "visited": self._visited_urls,
            "doc_id": doc_id,
            "instance_id": self._instance_id,
            "task_family": self.TASK_FAMILY_FOLDER,
            "workspace_kind": self.WORKSPACE_KIND,
            "created_doc_id": self._created_doc_id,
            "created_doc_url": self._created_doc_url,
        }
        reward = 0.0

        if doc_id:
            try:
                evaluator = self._load_evaluator()
                # Evaluators across families have slightly different
                # signatures (some take ``cached_models`` and / or
                # ``browsing_history``, others don't). Inspect the function
                # at call time and only forward the kwargs it accepts so we
                # don't have to maintain per-family wrappers.
                grade_fn = evaluator.grade_checkpoints
                accepted = self._accepted_kwargs(grade_fn)
                call_kwargs: Dict[str, Any] = {}
                if "workspace_doc_id" in accepted:
                    call_kwargs["workspace_doc_id"] = doc_id
                browsing_history_urls = self._visited_url_strings()
                if "browsing_history" in accepted:
                    call_kwargs["browsing_history"] = browsing_history_urls
                if "browsing_history_list" in accepted:
                    call_kwargs["browsing_history_list"] = browsing_history_urls
                # ``cached_models`` is opaque from our side; pass ``None``
                # so the evaluator falls back to its default lazy loader.
                if "cached_models" in accepted and "cached_models" not in call_kwargs:
                    call_kwargs["cached_models"] = None
                result = grade_fn(**call_kwargs)
                reward, score_breakdown = self._summarize_result(result, info)
                info.update(score_breakdown)
                print(
                    f"Evaluation complete — family={self.TASK_FAMILY_FOLDER}, "
                    f"instance={self._instance_id}, doc_id={doc_id}, "
                    f"score={info['eval.score_result']}/{info['eval.score_total']} "
                    f"(reward={reward:.3f})"
                )
            except Exception as exc:
                print(
                    f"Evaluator error ({self.TASK_FAMILY_FOLDER} "
                    f"instance={self._instance_id}): {exc}"
                )
                info["evaluation_error"] = str(exc)
                info["eval.score_fraction"] = 0.0
        else:
            print(
                "No Google workspace URL found in DONE message or page URL; "
                "skipping evaluation."
            )
            info["evaluation_error"] = "no_doc_id"
            info["eval.score_fraction"] = 0.0

        self._graded = True
        self._last_score_breakdown = info
        self._last_reward = reward
        return reward, info

    def _visited_url_strings(self) -> List[str]:
        """Return browsing history in the string format evaluator code expects."""
        return [
            v.get("url", "") if isinstance(v, dict) else str(v)
            for v in self._visited_urls
        ]

    @staticmethod
    def _accepted_kwargs(fn) -> set:
        """Return the set of keyword-argument names accepted by *fn*.

        Falls back to ``set()`` if introspection fails (e.g. C-implemented
        callables); callers should treat that as "accept nothing" and pass
        positional args instead.
        """
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            return set()
        accepted: set = set()
        for name, param in sig.parameters.items():
            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                # If the function accepts **kwargs, we can pass anything.
                if param.kind == inspect.Parameter.VAR_KEYWORD:
                    return {
                        "workspace_doc_id",
                        "cached_models",
                        "browsing_history",
                        "browsing_history_list",
                    }
                continue
            accepted.add(name)
        return accepted

    def _load_evaluator(self):
        """Dynamically load the evaluator module for this task instance.

        Instance directories under ``eval/tasks/<family>/`` are not Python
        packages (no ``__init__.py``), so we load the evaluator module
        directly from its file path. Two extra fix-ups happen here so the
        legacy evaluator code keeps working after the eval tree was moved
        inside the ``browsergym.knows`` package:

        1. ``TOKEN_PATH`` and ``CLIENT_SECRETS_PATH`` env vars are set to the
           bundled OAuth files (``browsergym/knows/auth-data/...``) so
           ``initialize_google_services`` can authenticate without the user
           having to ``cd`` into the package directory.
        2. After the module loads, its path constants (``TASK_DIR``,
           ``DATA_DIR``, ``DOC_IMAGES_DIR``, ``GOLD_IMAGES_DIR``, …) are
           rewritten to point inside this package's
           ``eval/tasks/<family>/instance_X/`` folder. The legacy code
           computes them as ``<cwd>/src/browsergym/eval/...``, which no
           longer exists in the new layout.
        """
        evaluator_path = (
            EVAL_TASKS_DIR
            / self.TASK_FAMILY_FOLDER
            / f"instance_{self._instance_id}"
            / "evaluator.py"
        )
        if not evaluator_path.exists():
            raise FileNotFoundError(f"Evaluator not found at {evaluator_path}")

        # 1. Point the Google auth helpers at the bundled credentials (if
        #    present) BEFORE exec_module runs — ``initialize_google_services``
        #    is invoked at evaluator-module import time, so env vars set
        #    after exec_module would be too late.
        #
        #    Priority inside the evaluator (see google_services_utils.py):
        #      1. SERVICE_ACCOUNT_PATH file → service-account auth (preferred)
        #      2. GCP_PROJECT_ID + DRIVE_SA_SECRET_ID → Secret Manager
        #      3. OAuth via TOKEN_PATH / CLIENT_SECRETS_PATH → installed-app flow
        package_root = _PACKAGE_DIR.parents[2]  # browsergym/knows/
        auth_dir = package_root / "auth-data"
        sa_file = auth_dir / "service-account.json"
        token_file = auth_dir / "token.json"
        creds_file = auth_dir / "credentials.json"
        if sa_file.exists() and "SERVICE_ACCOUNT_PATH" not in os.environ:
            os.environ["SERVICE_ACCOUNT_PATH"] = str(sa_file)
        if token_file.exists() and "TOKEN_PATH" not in os.environ:
            os.environ["TOKEN_PATH"] = str(token_file)
        if creds_file.exists() and "CLIENT_SECRETS_PATH" not in os.environ:
            os.environ["CLIENT_SECRETS_PATH"] = str(creds_file)

        # Some checkpoints fall back to a Gemini judge for image / structure
        # comparisons; load API keys from any local ``.env`` files so the
        # evaluator's ``load_model(...)`` call doesn't die with
        # "GOOGLE_AI_API_KEY environment variable is required".
        for env_path in (
            package_root / ".env",
            package_root.parents[1] / ".env",  # repo root .env if present
        ):
            if not env_path.is_file():
                continue
            try:
                with open(env_path) as _f:
                    for _raw in _f:
                        _line = _raw.strip()
                        if not _line or _line.startswith("#"):
                            continue
                        if _line.startswith("export "):
                            _line = _line[len("export ") :]
                        if "=" not in _line:
                            continue
                        _k, _, _v = _line.partition("=")
                        _k = _k.strip()
                        _v = _v.strip().strip('"').strip("'")
                        if _k and _k not in os.environ:
                            os.environ[_k] = _v
            except Exception:
                pass  # best-effort; live envs (CI, etc.) can set these explicitly

        module_name = (
            f"browsergym.knows._evaluators.{self.TASK_FAMILY_FOLDER}."
            f"instance_{self._instance_id}"
        )
        spec = importlib.util.spec_from_file_location(module_name, str(evaluator_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build module spec for {evaluator_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # 2. Redirect every path constant to the actual on-disk location.
        # Different task families use different subset of these constants;
        # ``hasattr`` guards make the redirect a no-op when an attribute is
        # absent (e.g. sheets evaluators that only use ``TASK_DIR``).
        instance_dir = evaluator_path.parent
        path_overrides = {
            "TASK_DIR": str(instance_dir) + os.sep,
            "DATA_DIR": str(instance_dir / "data") + os.sep,
            "DOC_IMAGES_DIR": str(instance_dir / "data" / "images") + os.sep,
            "DOC_IMAGES_CROPPED_DIR": str(instance_dir / "data" / "cropped_images")
            + os.sep,
            "PDF_IMAGES_DIR": str(instance_dir / "data" / "pdf_images") + os.sep,
            "GOLD_IMAGES_DIR": str(instance_dir / "data" / "gold_images") + os.sep,
            "GOLDS_DIR": str(instance_dir / "data" / "golds") + os.sep,
            "GOLD_DESCRIPTIONS_CSV": str(instance_dir / "data" / "gold_descriptions.csv"),
            "ORIGINAL_LOCATIONS_JSON": str(
                instance_dir / "data" / "original_image_locations.json"
            ),
            "ORIGINAL_TEXTBOX_LOCATIONS_JSON": str(
                instance_dir / "data" / "original_textbox_locations.json"
            ),
        }
        for attr, new_value in path_overrides.items():
            if hasattr(module, attr):
                setattr(module, attr, new_value)

        return module

    @staticmethod
    def _summarize_result(result, info: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Translate an evaluator ``Result`` into:
        1. ``reward``: a single 0-1 normalized float (sum of points / max points).
        2. ``score_breakdown``: a dict combining nested structured data
           (``evaluation`` detailed report, ``checkpoints`` list) and *flat
           scalar* keys prefixed with ``eval.``. The flat keys are what end
           up as columns in ``result_df_*.csv`` — one row per task run with
           an explicit score for every checkpoint.

        ``info`` is the validate() info dict; passed in only so the detailed
        report is also stashed there for inspection.
        """
        score_breakdown: Dict[str, Any] = {}

        score = result.final_score or {}
        if not isinstance(score, dict):
            score = {"total": 0, "result": float(score) if score else 0}

        total_max = int(score.get("total", 0) or 0)
        total_result = int(score.get("result", 0) or 0)
        reward = (total_result / total_max) if total_max > 0 else 0.0

        info["evaluation"] = result.get_detailed_report()

        info["checkpoints"] = [
            {
                "index": i + 1,
                "name": cp.name or f"checkpoint_{i + 1}",
                "result": cp.result,
                "total": cp.total,
                "fraction": (cp.result / cp.total) if cp.total else 0.0,
                "execution_time": cp.execution_time,
                "steps": [
                    {
                        "step_id": s.step_id,
                        "name": s.name,
                        "success": s.success,
                        "score": s.score,
                        "max_score": s.max_score,
                        "details": s.details,
                        "execution_time": s.execution_time,
                    }
                    for s in cp.steps
                ],
            }
            for i, cp in enumerate(result.checkpoints)
        ]

        score_breakdown["eval.score_result"] = total_result
        score_breakdown["eval.score_total"] = total_max
        score_breakdown["eval.score_fraction"] = reward
        score_breakdown["eval.n_checkpoints"] = len(result.checkpoints)
        if result.total_execution_time is not None:
            score_breakdown["eval.total_execution_time"] = float(
                result.total_execution_time
            )

        for i, cp in enumerate(result.checkpoints, start=1):
            score_breakdown[f"eval.cp{i}_name"] = cp.name or f"checkpoint_{i}"
            score_breakdown[f"eval.cp{i}_result"] = cp.result
            score_breakdown[f"eval.cp{i}_total"] = cp.total
            score_breakdown[f"eval.cp{i}_fraction"] = (
                (cp.result / cp.total) if cp.total else 0.0
            )
            if cp.execution_time is not None:
                score_breakdown[f"eval.cp{i}_execution_time"] = float(cp.execution_time)

        return reward, score_breakdown

    @classmethod
    def _extract_doc_id(cls, text: str, fallback_url: str = "") -> Optional[str]:
        """Extract a Google workspace file ID (Doc / Sheet / Slides) from a
        URL found in *text*, falling back to *fallback_url* (e.g. the current
        page URL) if nothing is found.

        When :py:attr:`WORKSPACE_KIND` is set, only matches whose URL segment
        agrees with that kind are returned — so a stray
        ``/document/d/<id>`` link in chat doesn't get accepted as the doc id
        for a Sheets task. If no constrained match is found, falls back to
        any kind so we still return something useful.
        """
        expected_segment = _WORKSPACE_URL_SEGMENT.get(cls.WORKSPACE_KIND)
        # First pass: only accept matches whose segment matches our kind.
        if expected_segment:
            for source in (text, fallback_url):
                if not source:
                    continue
                for match in _WORKSPACE_ID_RE.finditer(source):
                    if match.group(1) == expected_segment:
                        return match.group(2)
        # Second pass (fallback): accept any kind. Useful when the goal text
        # already includes the original tutorial url etc.
        for source in (text, fallback_url):
            if not source:
                continue
            match = _WORKSPACE_ID_RE.search(source)
            if match:
                return match.group(2)
        return None


class DocsFormalLetterTask(KnowsWorkspaceTask):
    """Formal-letter KNOWS task (docs_1_formal_letter).

    Concrete configuration: pre-creates a Google Doc, loads task text from
    ``eval/tasks/docs_1_formal_letter/instance_<instance_id>/task.md``, and
    grades via the bundled evaluator.
    """

    TASK_FAMILY_FOLDER = "docs_1_formal_letter"
    TASK_ID_PREFIX = "knows.docs_1_formal_letter"
    WORKSPACE_KIND = WORKSPACE_KIND_DOCS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SheetsPersonalRecipeTask(KnowsWorkspaceTask):
    """sheets_2_personal_recipe — pre-creates a Google Sheet and grades it
    against the food-composition evaluator."""

    TASK_FAMILY_FOLDER = "sheets_2_personal_recipe_foodcomposition"
    TASK_ID_PREFIX = "knows.sheets_2_personal_recipe"
    WORKSPACE_KIND = WORKSPACE_KIND_SHEETS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class DocsInfluentialPapersTask(KnowsWorkspaceTask):
    """docs_5_influential_papers — pre-creates a Google Doc and grades it
    against the influential-papers evaluator."""

    TASK_FAMILY_FOLDER = "docs_5_influential_papers"
    TASK_ID_PREFIX = "knows.docs_5_influential_papers"
    WORKSPACE_KIND = WORKSPACE_KIND_DOCS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SheetsStockTrackerTask(KnowsWorkspaceTask):
    """sheets_6_stock_tracker — backed by the bundled
    ``sheets_6_investmenttracker`` evaluator (the only sheets_6 evaluator
    that exists in the repo)."""

    TASK_FAMILY_FOLDER = "sheets_6_investmenttracker"
    TASK_ID_PREFIX = "knows.sheets_6_stock_tracker"
    WORKSPACE_KIND = WORKSPACE_KIND_SHEETS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SheetsSkiTourPlanTask(KnowsWorkspaceTask):
    """sheets_25_skitourplan — pre-creates a Google Sheet and grades it
    against the ski-tour-plan evaluator."""

    TASK_FAMILY_FOLDER = "sheets_25_skitourplan"
    TASK_ID_PREFIX = "knows.sheets_25_skitourplan"
    WORKSPACE_KIND = WORKSPACE_KIND_SHEETS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SlidesIllustratedBookReportTask(KnowsWorkspaceTask):
    """slides_20_illustrated_book_report — pre-creates a Google Slides deck
    and grades it against the illustrated-book-report evaluator."""

    TASK_FAMILY_FOLDER = "slides_20_Illustrated_Book_Report"
    TASK_ID_PREFIX = "knows.slides_20_illustrated_book_report"
    WORKSPACE_KIND = WORKSPACE_KIND_SLIDES
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SheetsApartmentFinderTask(KnowsWorkspaceTask):
    """sheets_38_apartment_finder — pre-creates a Google Sheet and grades it
    against the apartment-finder evaluator."""

    TASK_FAMILY_FOLDER = "sheets_38_apartment_finder"
    TASK_ID_PREFIX = "knows.sheets_38_apartment_finder"
    WORKSPACE_KIND = WORKSPACE_KIND_SHEETS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SlidesPersonalLookbookPaintColorsTask(KnowsWorkspaceTask):
    """slides_39_personal_lookbook_paintcolors — pre-creates a Google Slides
    deck and grades it against the personal-lookbook paint-colors evaluator."""

    TASK_FAMILY_FOLDER = "slides_39_Personal_Lookbook_PaintColors"
    TASK_ID_PREFIX = "knows.slides_39_personal_lookbook_paintcolors"
    WORKSPACE_KIND = WORKSPACE_KIND_SLIDES
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SlidesRemoveImagesAddPlaceholdersTask(KnowsWorkspaceTask):
    """slides_17_remove_images_add_placeholders — pre-creates a Google Slides
    deck and grades it against the remove-images-add-placeholders evaluator."""

    TASK_FAMILY_FOLDER = "slides_17_removeimagesaddplaceholders"
    TASK_ID_PREFIX = "knows.slides_17_remove_images_add_placeholders"
    WORKSPACE_KIND = WORKSPACE_KIND_SLIDES
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class DocsPersonalRecipeOcrTask(KnowsWorkspaceTask):
    """docs_11_personal_recipe_ocr — pre-creates a Google Doc and grades it
    against the personal-recipe OCR evaluator."""

    TASK_FAMILY_FOLDER = "docs_11_personal_recipe_ocr"
    TASK_ID_PREFIX = "knows.docs_11_personal_recipe_ocr"
    WORKSPACE_KIND = WORKSPACE_KIND_DOCS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class DocsEducationLessonPlanTask(KnowsWorkspaceTask):
    """docs_31_education_lesson_plan — pre-creates a Google Doc and grades it
    against the education lesson plan evaluator."""

    TASK_FAMILY_FOLDER = "docs_31_education_lesson_plan"
    TASK_ID_PREFIX = "knows.docs_31_education_lesson_plan"
    WORKSPACE_KIND = WORKSPACE_KIND_DOCS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class DocsReferenceListTask(KnowsWorkspaceTask):
    """docs_37_reference_list — pre-creates a Google Doc and grades it
    against the reference list evaluator."""

    TASK_FAMILY_FOLDER = "docs_37_reference_list"
    TASK_ID_PREFIX = "knows.docs_37_reference_list"
    WORKSPACE_KIND = WORKSPACE_KIND_DOCS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SheetsPaperSortingTask(KnowsWorkspaceTask):
    """sheets_10_paper_sorting — pre-creates a Google Sheet and grades it
    against the paper sorting evaluator."""

    TASK_FAMILY_FOLDER = "sheets_10_paper_sorting"
    TASK_ID_PREFIX = "knows.sheets_10_paper_sorting"
    WORKSPACE_KIND = WORKSPACE_KIND_SHEETS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)

    # Set to True to run full gold data pipeline (preprocess, figures, keywords)
    # during setup. Default False — only creates fresh Drive folders.
    RUN_GOLD_PIPELINE = False

    def setup(self, page: playwright.sync_api.Page) -> Tuple[str, dict]:
        self._run_setup_pipeline()
        return super().setup(page)

    def _run_setup_pipeline(self) -> None:
        """Run pre-benchmark pipeline for sheets_10.

        Always creates fresh Drive folders. Optionally runs gold data
        collection if RUN_GOLD_PIPELINE is True.
        """
        self._run_task_script("setup_run.py")
        if self.RUN_GOLD_PIPELINE:
            self._run_task_script("preprocess.py", ["--rematch-only"])
            self._run_task_script("extract_figures.py", ["--skip-existing"])
            self._run_task_script("detect_keyword.py", ["--skip-existing"])

    def _run_task_script(self, script_name: str, extra_args: list = None) -> None:
        """Run a task-level script as a subprocess.

        Args:
            script_name: Name of the script in the task family folder.
            extra_args: Additional CLI arguments to pass after --instance.
        """
        script_path = EVAL_TASKS_DIR / self.TASK_FAMILY_FOLDER / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Required script not found: {script_path}")

        package_root = _PACKAGE_DIR.parents[2]  # browsergym/knows/
        env = os.environ.copy()
        self._load_local_env(env, package_root)

        command = [
            sys.executable,
            str(script_path),
            "--instance",
            str(self._instance_id),
        ]
        if extra_args:
            command.extend(extra_args)

        print(f"Running sheets_10 {script_name}: " + " ".join(command))
        try:
            subprocess.run(
                command,
                cwd=str(package_root),
                env=env,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"sheets_10 {script_name} failed for "
                f"{self.TASK_FAMILY_FOLDER} instance {self._instance_id}."
            ) from exc

    @staticmethod
    def _load_local_env(env: Dict[str, str], package_root: Path) -> None:
        """Populate subprocess env from local .env files without overriding live env."""
        for env_path in (
            package_root / ".env",
            package_root.parents[1] / ".env",
        ):
            if not env_path.is_file():
                continue
            try:
                with open(env_path) as env_file:
                    for raw_line in env_file:
                        line = raw_line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("export "):
                            line = line[len("export ") :]
                        if "=" not in line:
                            continue
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in env:
                            env[key] = value
            except OSError:
                continue


class SheetsPersonalTravelPlannerTask(KnowsWorkspaceTask):
    """sheets_28_personal_travel_planner — pre-creates a Google Sheet and
    grades it against the personal travel planner evaluator."""

    TASK_FAMILY_FOLDER = "sheets_28_personal_travel_planner"
    TASK_ID_PREFIX = "knows.sheets_28_personal_travel_planner"
    WORKSPACE_KIND = WORKSPACE_KIND_SHEETS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1,)


class SheetsWeddingPlannerTask(KnowsWorkspaceTask):
    """sheets_45_Personal_WeddingPlanner_weddingcolorpallette — pre-creates a
    Google Sheet and grades it against the wedding planner evaluator."""

    TASK_FAMILY_FOLDER = "sheets_45_Personal_WeddingPlanner_weddingcolorpallette"
    TASK_ID_PREFIX = "knows.sheets_45_wedding_planner"
    WORKSPACE_KIND = WORKSPACE_KIND_SHEETS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SheetsMovieRecommendationTask(KnowsWorkspaceTask):
    """sheets_55_Movie_Recommendation — pre-creates a Google Sheet and grades
    it against the movie recommendation evaluator."""

    TASK_FAMILY_FOLDER = "sheets_55_Movie_Recommendation"
    TASK_ID_PREFIX = "knows.sheets_55_movie_recommendation"
    WORKSPACE_KIND = WORKSPACE_KIND_SHEETS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SheetsRunningAnalysisTask(KnowsWorkspaceTask):
    """sheets_7_running_analysis — pre-creates a Google Sheet and grades it
    against the running analysis evaluator."""

    TASK_FAMILY_FOLDER = "sheets_7_running_analysis"
    TASK_ID_PREFIX = "knows.sheets_7_running_analysis"
    WORKSPACE_KIND = WORKSPACE_KIND_SHEETS
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SlidesBasicEducationalSlideDeckTask(KnowsWorkspaceTask):
    """slides_26_basic_educational_slide_deck — pre-creates a Google Slides
    deck and grades it against the educational slide deck evaluator."""

    TASK_FAMILY_FOLDER = "slides_26_basic_educational_slide_deck"
    TASK_ID_PREFIX = "knows.slides_26_basic_educational_slide_deck"
    WORKSPACE_KIND = WORKSPACE_KIND_SLIDES
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SlidesBuyCarPresTask(KnowsWorkspaceTask):
    """slides_29_buy_car_pres — pre-creates a Google Slides deck and grades
    it against the buy car presentation evaluator."""

    TASK_FAMILY_FOLDER = "slides_29_buy_car_pres"
    TASK_ID_PREFIX = "knows.slides_29_buy_car_pres"
    WORKSPACE_KIND = WORKSPACE_KIND_SLIDES
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SlidesWikipediaPhotosTask(KnowsWorkspaceTask):
    """slides_30_Work_Wikipedia_Photos — pre-creates a Google Slides deck and
    grades it against the Wikipedia photos evaluator."""

    TASK_FAMILY_FOLDER = "slides_30_Work_Wikipedia_Photos"
    TASK_ID_PREFIX = "knows.slides_30_wikipedia_photos"
    WORKSPACE_KIND = WORKSPACE_KIND_SLIDES
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


class SlidesProductComparisonTask(KnowsWorkspaceTask):
    """slides_42_personal_none_product_comparison — pre-creates a Google Slides
    deck and grades it against the product comparison evaluator."""

    TASK_FAMILY_FOLDER = "slides_42_personal_none_product_comparison"
    TASK_ID_PREFIX = "knows.slides_42_product_comparison"
    WORKSPACE_KIND = WORKSPACE_KIND_SLIDES
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1,)


class SlidesEventAnnouncementPosterTask(KnowsWorkspaceTask):
    """slides_51_event_announcement_poster — pre-creates a Google Slides deck
    and grades it against the event announcement poster evaluator."""

    TASK_FAMILY_FOLDER = "slides_51_event_announcement_poster"
    TASK_ID_PREFIX = "knows.slides_51_event_announcement_poster"
    WORKSPACE_KIND = WORKSPACE_KIND_SLIDES
    AVAILABLE_INSTANCES: Tuple[int, ...] = (1, 2, 3, 4, 5)


