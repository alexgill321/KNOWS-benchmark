"""Parallel execution utilities for evaluator performance optimization."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from threading import Semaphore, Lock
from typing import List, Dict, Any, Callable, Tuple, Optional

# Global semaphores for rate limiting
# NOTE: Google API client has SSL/threading issues with high concurrency
# Keep these low to avoid SSL errors and memory corruption
GOOGLE_API_SEMAPHORE = Semaphore(2)  # Max 2 concurrent Google API calls (conservative)
VLM_API_SEMAPHORE = Semaphore(3)     # Max 3 concurrent VLM calls (avoid overload)

# Lock for thread-safe result collection
RESULTS_LOCK = Lock()


def parallel_download(
    download_tasks: List[Dict[str, Any]],
    max_workers: int = 3,
    use_rate_limit: bool = True,
    max_retries: int = 2,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Download multiple items in parallel with rate limiting and retry logic.

    Args:
        download_tasks: List of dicts with keys:
            - 'id': Unique identifier for this task
            - 'func': Download function to call
            - 'args': Tuple of positional arguments
            - 'kwargs': Dict of keyword arguments (optional)
        max_workers: Maximum concurrent downloads (kept low for stability)
        use_rate_limit: Whether to use the Google API semaphore
        max_retries: Number of retry attempts on failure
        timeout: Optional wall-clock cap (seconds). When set, downloads that
            don't finish within this budget are recorded as None and the
            remaining futures are cancelled. None (default) waits indefinitely.

    Returns:
        Dict mapping task 'id' to download result (or None if failed/timed out)
    """
    results = {}

    def download_with_limit(task):
        task_id = task['id']
        func = task['func']
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                if use_rate_limit:
                    with GOOGLE_API_SEMAPHORE:
                        result = func(*args, **kwargs)
                        return task_id, result
                else:
                    result = func(*args, **kwargs)
                    return task_id, result
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    # Exponential backoff: 0.5s, 1s, 2s...
                    time.sleep(0.5 * (2 ** attempt))
                else:
                    print(f"  Parallel download failed for {task_id} after {max_retries + 1} attempts: {e}")

        return task_id, None

    # Use conservative max_workers to avoid SSL/threading issues
    effective_workers = min(max_workers, 3) if use_rate_limit else max_workers

    with ThreadPoolExecutor(max_workers=effective_workers) as executor:
        futures = {executor.submit(download_with_limit, task): task['id'] for task in download_tasks}

        try:
            for future in as_completed(futures, timeout=timeout):
                try:
                    task_id, result = future.result()
                    with RESULTS_LOCK:
                        results[task_id] = result
                except Exception as e:
                    print(f"  Future failed: {e}")
        except FuturesTimeoutError:
            for fut, task_id in futures.items():
                if not fut.done():
                    print(f"  Download {task_id} timed out after {timeout}s")
                    with RESULTS_LOCK:
                        results.setdefault(task_id, None)
                    fut.cancel()

    return results


def parallel_execute(
    tasks: List[Dict[str, Any]],
    max_workers: int = 3,
    semaphore: Optional[Semaphore] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Execute multiple tasks in parallel with optional rate limiting.

    Args:
        tasks: List of dicts with keys:
            - 'id': Unique identifier for this task
            - 'func': Function to call
            - 'args': Tuple of positional arguments
            - 'kwargs': Dict of keyword arguments (optional)
        max_workers: Maximum concurrent executions
        semaphore: Optional semaphore for rate limiting
        timeout: Optional wall-clock cap (seconds). When set, tasks that don't
            finish within this budget are recorded as None and the remaining
            futures are cancelled. None (default) waits indefinitely.

    Returns:
        Dict mapping task 'id' to result (or None if failed/timed out)
    """
    results = {}

    def execute_with_limit(task):
        task_id = task['id']
        func = task['func']
        args = task.get('args', ())
        kwargs = task.get('kwargs', {})

        try:
            if semaphore:
                with semaphore:
                    return task_id, func(*args, **kwargs)
            else:
                return task_id, func(*args, **kwargs)
        except Exception as e:
            print(f"  Parallel execution failed for {task_id}: {e}")
            return task_id, None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(execute_with_limit, task): task['id'] for task in tasks}

        try:
            for future in as_completed(futures, timeout=timeout):
                try:
                    task_id, result = future.result()
                    results[task_id] = result
                except Exception as e:
                    print(f"  Future failed: {e}")
        except FuturesTimeoutError:
            # Wall-clock cap exceeded — record unresolved tasks as None and let the
            # ThreadPoolExecutor.__exit__ handle cleanup (workers may still finish in
            # the background but their results are discarded).
            for fut, task_id in futures.items():
                if not fut.done():
                    print(f"  Task {task_id} timed out after {timeout}s")
                    results[task_id] = None
                    fut.cancel()

    return results


def parallel_vlm_calls(
    vlm_tasks: List[Dict[str, Any]],
    model: Callable,
    max_workers: int = 3,
    timeout: Optional[float] = None,
) -> Dict[str, bool]:
    """
    Execute multiple VLM calls in parallel with rate limiting.

    Args:
        vlm_tasks: List of dicts with keys:
            - 'id': Unique identifier (e.g., filename)
            - 'messages': The messages to send to the model
        model: The loaded model callable
        max_workers: Maximum concurrent VLM calls
        timeout: Optional wall-clock cap (seconds). When set, VLM calls that
            don't finish within this budget are recorded as False and the
            remaining futures are cancelled. None (default) waits indefinitely.

    Returns:
        Dict mapping task 'id' to boolean result
    """
    results = {}

    def call_vlm(task):
        task_id = task['id']
        messages = task['messages']

        try:
            with VLM_API_SEMAPHORE:
                response = model(messages).strip().lower()
                return task_id, 'yes' in response
        except Exception as e:
            print(f"  VLM call failed for {task_id}: {e}")
            return task_id, False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(call_vlm, task): task['id'] for task in vlm_tasks}

        try:
            for future in as_completed(futures, timeout=timeout):
                try:
                    task_id, result = future.result()
                    results[task_id] = result
                except Exception as e:
                    print(f"  VLM future failed: {e}")
        except FuturesTimeoutError:
            for fut, task_id in futures.items():
                if not fut.done():
                    print(f"  VLM call {task_id} timed out after {timeout}s")
                    results.setdefault(task_id, False)
                    fut.cancel()

    return results


def parallel_image_match(
    match_tasks: List[Dict[str, Any]],
    max_workers: int = 5,
    timeout: Optional[float] = None,
) -> Dict[str, Tuple[bool, str]]:
    """
    Perform parallel image matching (exact + perceptual hash tiers only).

    Args:
        match_tasks: List of dicts with keys:
            - 'id': Unique identifier
            - 'candidate_path': Path to candidate image
            - 'gold_path': Path to gold/reference image
        max_workers: Maximum concurrent matches
        timeout: Optional wall-clock cap (seconds). When set, matches that
            don't finish within this budget are recorded as (False, None) and
            the remaining futures are cancelled. None (default) waits
            indefinitely.

    Returns:
        Dict mapping task 'id' to (matched: bool, method: str) tuple
    """
    from src.browsergym.knows.eval.eval_utils.image_utils import image_exact_match, perceptual_hash_match

    results = {}

    def match_image(task):
        task_id = task['id']
        candidate = task['candidate_path']
        gold = task['gold_path']

        # Tier 1: Exact match
        try:
            if image_exact_match(candidate, gold):
                return task_id, (True, 'exact')
        except Exception:
            pass

        # Tier 2: Perceptual hash
        try:
            if perceptual_hash_match(candidate, gold, threshold=15):
                return task_id, (True, 'perceptual_hash')
        except Exception:
            pass

        return task_id, (False, None)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(match_image, task): task['id'] for task in match_tasks}

        try:
            for future in as_completed(futures, timeout=timeout):
                try:
                    task_id, result = future.result()
                    results[task_id] = result
                except Exception as e:
                    print(f"  Image match future failed: {e}")
        except FuturesTimeoutError:
            for fut, task_id in futures.items():
                if not fut.done():
                    print(f"  Image match {task_id} timed out after {timeout}s")
                    results.setdefault(task_id, (False, None))
                    fut.cancel()

    return results


def fast_parallel_vlm_calls(
    vlm_tasks: List[Dict[str, Any]],
    model: Callable,
    max_workers: int = 10,
    timeout: Optional[float] = None,
) -> Dict[str, bool]:
    """
    Execute multiple VLM calls in parallel without global semaphore bottleneck.

    This is a faster version of parallel_vlm_calls that relies only on max_workers
    for concurrency control, without using VLM_API_SEMAPHORE. Use when you need
    higher throughput and the model/API can handle the load.

    Args:
        vlm_tasks: List of dicts with keys:
            - 'id': Unique identifier (e.g., filename)
            - 'messages': The messages to send to the model
        model: The loaded model callable
        max_workers: Maximum concurrent VLM calls (default 10)
        timeout: Optional wall-clock cap (seconds). When set, VLM calls that
            don't finish within this budget are recorded as False and the
            remaining futures are cancelled. None (default) waits indefinitely.

    Returns:
        Dict mapping task 'id' to boolean result (True if response contains 'yes')
    """
    results = {}

    def call_vlm(task):
        task_id = task['id']
        messages = task['messages']
        try:
            response = model(messages).strip().lower()
            return task_id, 'yes' in response
        except Exception as e:
            print(f"  VLM call failed for {task_id}: {e}")
            return task_id, False

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(call_vlm, task): task['id'] for task in vlm_tasks}
        try:
            for future in as_completed(futures, timeout=timeout):
                try:
                    task_id, result = future.result()
                    results[task_id] = result
                except Exception:
                    pass
        except FuturesTimeoutError:
            for fut, task_id in futures.items():
                if not fut.done():
                    print(f"  VLM call {task_id} timed out after {timeout}s")
                    results.setdefault(task_id, False)
                    fut.cancel()

    return results
