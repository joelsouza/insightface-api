"""Thread pool executor for blocking inference operations.

This module provides an async-compatible executor for running CPU-bound
inference operations without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

from src.exceptions import InferenceTimeoutError, ServiceOverloadedError

T = TypeVar("T")


class InferenceExecutor:
    """Manages a thread pool for CPU-bound inference operations.

    This executor allows async code to run blocking inference operations
    in a separate thread pool, preventing the event loop from being blocked.
    ONNX runtime releases the GIL during compute, so multiple threads can
    run inference concurrently.

    The number of requests in flight is bounded to `max_workers + max_queue`.
    Anything beyond that is rejected immediately, because `asyncio.wait_for`
    cannot cancel a job that is already running in a thread: without the bound,
    a burst just makes every queued request wait for the full timeout and then
    fail.

    Attributes:
        timeout: Default timeout for inference operations in seconds.
        max_workers: Number of threads in the pool.
        max_queue: Requests allowed to wait for a free thread.

    Example:
        executor = InferenceExecutor(max_workers=4, timeout=30.0, max_queue=16)
        result = await executor.run(model.get_faces, image)
    """

    def __init__(
        self,
        max_workers: int = 4,
        timeout: float = 30.0,
        max_queue: int = 16,
    ) -> None:
        """Initialize the inference executor.

        Args:
            max_workers: Maximum number of concurrent inference threads.
                         Defaults to 4 for typical CPU workloads.
            timeout: Default timeout for inference operations in seconds.
                     Defaults to 30.0 seconds.
            max_queue: Requests allowed to wait for a free thread before
                       new ones are rejected. Defaults to 16.
        """
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="inference",
        )
        self._timeout = timeout
        self._max_workers = max_workers
        self._max_queue = max_queue
        self._slots = asyncio.Semaphore(max_workers + max_queue)
        self._is_shutdown = False
        self.submitted = 0
        self.rejected = 0
        self.timed_out = 0
        self.completed = 0
        self._in_flight = 0

    @property
    def timeout(self) -> float:
        """Return the default timeout for inference operations."""
        return self._timeout

    @property
    def max_workers(self) -> int:
        """Return the number of worker threads."""
        return self._max_workers

    @property
    def max_queue(self) -> int:
        """Return the number of requests allowed to wait for a thread."""
        return self._max_queue

    @property
    def capacity(self) -> int:
        """Return the total number of requests allowed in flight."""
        return self._max_workers + self._max_queue

    @property
    def in_flight(self) -> int:
        """Return the number of accepted requests not yet finished."""
        return self._in_flight

    async def run(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout: Optional[float] = None,
        stats: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> T:
        """Run a blocking function in the thread pool with timeout.

        Args:
            func: The blocking function to execute.
            *args: Positional arguments to pass to the function.
            timeout: Override the default timeout (None uses default).
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function call.

        Raises:
            ServiceOverloadedError: If no slot is free.
            InferenceTimeoutError: If the operation exceeds the timeout.
        """
        if self._is_shutdown:
            raise RuntimeError("Executor has been shut down")

        if self._slots.locked():
            self.rejected += 1
            raise ServiceOverloadedError(
                f"Server is at capacity ({self.capacity} requests in flight), "
                f"retry shortly"
            )

        effective_timeout = timeout if timeout is not None else self._timeout
        loop = asyncio.get_running_loop()

        async with self._slots:
            self.submitted += 1
            self._in_flight += 1
            queued_at = time.perf_counter()

            def run_submitted() -> T:
                if stats is not None:
                    stats["queue_wait_ms"] = (
                        time.perf_counter() - queued_at
                    ) * 1000
                return func(*args, **kwargs)

            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor,
                        run_submitted,
                    ),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                self.timed_out += 1
                raise InferenceTimeoutError(
                    f"Inference operation timed out after {effective_timeout}s"
                )
            except Exception:
                self.completed += 1
                raise
            else:
                self.completed += 1
                return result
            finally:
                self._in_flight -= 1

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool.

        Args:
            wait: If True, wait for all pending tasks to complete.
        """
        self._is_shutdown = True
        self._executor.shutdown(wait=wait)
