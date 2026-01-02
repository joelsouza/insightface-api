"""Thread pool executor for blocking inference operations.

This module provides an async-compatible executor for running CPU-bound
inference operations without blocking the event loop.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, Optional, TypeVar

from src.exceptions import InferenceTimeoutError

T = TypeVar("T")


class InferenceExecutor:
    """Manages a thread pool for CPU-bound inference operations.

    This executor allows async code to run blocking inference operations
    in a separate thread pool, preventing the event loop from being blocked.
    ONNX runtime releases the GIL during compute, so multiple threads can
    run inference concurrently.

    Attributes:
        timeout: Default timeout for inference operations in seconds.
        max_workers: Number of threads in the pool.

    Example:
        executor = InferenceExecutor(max_workers=4, timeout=30.0)
        result = await executor.run(model.get_faces, image)
    """

    def __init__(self, max_workers: int = 4, timeout: float = 30.0) -> None:
        """Initialize the inference executor.

        Args:
            max_workers: Maximum number of concurrent inference threads.
                         Defaults to 4 for typical CPU workloads.
            timeout: Default timeout for inference operations in seconds.
                     Defaults to 30.0 seconds.
        """
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="inference",
        )
        self._timeout = timeout
        self._max_workers = max_workers
        self._is_shutdown = False

    @property
    def timeout(self) -> float:
        """Return the default timeout for inference operations."""
        return self._timeout

    @property
    def max_workers(self) -> int:
        """Return the number of worker threads."""
        return self._max_workers

    async def run(
        self,
        func: Callable[..., T],
        *args: Any,
        timeout: Optional[float] = None,
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
            InferenceTimeoutError: If the operation exceeds the timeout.
        """
        if self._is_shutdown:
            raise RuntimeError("Executor has been shut down")

        effective_timeout = timeout if timeout is not None else self._timeout
        loop = asyncio.get_running_loop()

        try:
            return await asyncio.wait_for(
                loop.run_in_executor(
                    self._executor,
                    partial(func, *args, **kwargs),
                ),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            raise InferenceTimeoutError(
                f"Inference operation timed out after {effective_timeout}s"
            )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool.

        Args:
            wait: If True, wait for all pending tasks to complete.
        """
        self._is_shutdown = True
        self._executor.shutdown(wait=wait)
