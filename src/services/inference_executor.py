"""Thread pool executor for blocking inference operations.

This module provides an async-compatible executor for running CPU-bound
inference operations without blocking the event loop.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class InferenceExecutor:
    """Manages a thread pool for CPU-bound inference operations.

    This executor allows async code to run blocking inference operations
    in a separate thread pool, preventing the event loop from being blocked.
    ONNX runtime releases the GIL during compute, so multiple threads can
    run inference concurrently.

    Example:
        executor = InferenceExecutor(max_workers=4)
        result = await executor.run(model.get_faces, image)
    """

    def __init__(self, max_workers: int = 4) -> None:
        """Initialize the inference executor.

        Args:
            max_workers: Maximum number of concurrent inference threads.
                         Defaults to 4 for typical CPU workloads.
        """
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="inference",
        )

    async def run(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Run a blocking function in the thread pool.

        Args:
            func: The blocking function to execute.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of the function call.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            partial(func, *args, **kwargs),
        )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the thread pool.

        Args:
            wait: If True, wait for all pending tasks to complete.
        """
        self._executor.shutdown(wait=wait)
