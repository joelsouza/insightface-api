"""
Tests for thread safety and concurrency behavior.

These tests verify that the ModelManager and InferenceExecutor
handle concurrent operations safely.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.config import Settings
from src.exceptions import InferenceTimeoutError, ServiceOverloadedError
from src.services import InferenceExecutor, ModelManager


class TestModelManagerThreadSafety:
    """Tests for ModelManager thread safety."""

    def test_concurrent_load_calls_only_load_once(self, settings: Settings):
        """Test that concurrent load() calls only initialize model once."""
        load_count = 0
        load_lock = threading.Lock()

        def mock_make_session(*args, **kwargs):
            nonlocal load_count
            with load_lock:
                load_count += 1
            time.sleep(0.05)  # Simulate session creation time
            return MagicMock()

        logger = MagicMock()
        manager = ModelManager(settings=settings, logger=logger)

        with patch(
            "src.services.model_manager._make_session",
            side_effect=mock_make_session,
        ):
            # Start multiple threads trying to load
            threads = []
            for _ in range(5):
                t = threading.Thread(target=manager.load)
                threads.append(t)
                t.start()

            # Wait for all threads
            for t in threads:
                t.join()

            # One session per model, created by a single thread
            assert load_count == 2
            assert manager.is_loaded is True

    def test_load_unload_thread_safety(self, settings: Settings):
        """Test that load/unload operations don't race."""
        logger = MagicMock()
        manager = ModelManager(settings=settings, logger=logger)

        with patch(
            "src.services.model_manager._make_session",
            return_value=MagicMock(),
        ):
            # Load first
            manager.load()
            assert manager.is_loaded is True

            # Run concurrent unload and load
            def do_unload():
                time.sleep(0.01)
                manager.unload()

            def do_load():
                time.sleep(0.02)
                manager.load()

            t1 = threading.Thread(target=do_unload)
            t2 = threading.Thread(target=do_load)

            t1.start()
            t2.start()
            t1.join()
            t2.join()

            # Should be in a consistent state (either loaded or not)
            # The lock ensures no partial state
            assert manager.is_loaded in [True, False]
            if manager.is_loaded:
                assert manager.model is not None
            else:
                assert manager.model is None


class TestInferenceExecutorTimeout:
    """Tests for InferenceExecutor timeout handling."""

    @pytest.mark.asyncio
    async def test_timeout_raises_inference_timeout_error(self):
        """Test that timeout raises InferenceTimeoutError."""
        executor = InferenceExecutor(max_workers=1, timeout=0.1)

        def slow_function():
            time.sleep(1.0)  # Much longer than timeout
            return "done"

        with pytest.raises(InferenceTimeoutError) as exc_info:
            await executor.run(slow_function)

        assert "timed out" in str(exc_info.value).lower()
        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_custom_timeout_override(self):
        """Test that per-call timeout can override default."""
        executor = InferenceExecutor(max_workers=1, timeout=10.0)

        def slow_function():
            time.sleep(1.0)
            return "done"

        # Use shorter timeout for this call
        with pytest.raises(InferenceTimeoutError):
            await executor.run(slow_function, timeout=0.1)

        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_successful_operation_within_timeout(self):
        """Test that operations completing within timeout succeed."""
        executor = InferenceExecutor(max_workers=1, timeout=5.0)

        def quick_function():
            return "success"

        result = await executor.run(quick_function)
        assert result == "success"
        assert executor.submitted == 1
        assert executor.completed == 1
        assert executor.rejected == 0
        assert executor.timed_out == 0
        assert executor.in_flight == 0
        executor.shutdown(wait=False)


class TestInferenceExecutorConcurrency:
    """Tests for InferenceExecutor concurrent execution."""

    @pytest.mark.asyncio
    async def test_concurrent_tasks_run_in_parallel(self):
        """Test that multiple tasks can run concurrently."""
        executor = InferenceExecutor(max_workers=4, timeout=10.0)

        execution_times = []
        lock = threading.Lock()

        def task(task_id: int):
            start = time.time()
            time.sleep(0.1)
            with lock:
                execution_times.append((task_id, start, time.time()))
            return task_id

        # Run 4 tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(
            executor.run(task, 1),
            executor.run(task, 2),
            executor.run(task, 3),
            executor.run(task, 4),
        )
        total_time = time.time() - start_time

        assert set(results) == {1, 2, 3, 4}
        # If running in parallel, total time should be ~0.1s, not ~0.4s
        assert total_time < 0.3  # Allow some overhead

        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_shutdown_prevents_new_tasks(self):
        """Test that shutdown prevents new task submission."""
        executor = InferenceExecutor(max_workers=1, timeout=5.0)
        executor.shutdown(wait=True)

        def simple_task():
            return "done"

        with pytest.raises(RuntimeError) as exc_info:
            await executor.run(simple_task)

        assert "shut down" in str(exc_info.value).lower()


class TestInferenceExecutorBackPressure:
    """Tests for bounded in-flight requests."""

    @pytest.mark.asyncio
    async def test_rejects_when_all_slots_are_taken(self):
        """Test that a full executor rejects instead of queueing forever."""
        executor = InferenceExecutor(max_workers=1, timeout=5.0, max_queue=1)
        release = threading.Event()

        def blocking_task():
            release.wait(timeout=5.0)
            return "done"

        # Fill both slots (1 running + 1 queued).
        running = [
            asyncio.create_task(executor.run(blocking_task)) for _ in range(2)
        ]
        await asyncio.sleep(0.05)

        with pytest.raises(ServiceOverloadedError) as exc_info:
            await executor.run(blocking_task)

        assert exc_info.value.status_code == 503
        assert exc_info.value.error_code == "OVERLOADED"
        assert executor.rejected == 1
        assert executor.submitted == 2
        assert executor.in_flight == 2

        release.set()
        assert await asyncio.gather(*running) == ["done", "done"]
        assert executor.completed == 2
        assert executor.in_flight == 0
        executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_capacity_reports_workers_plus_queue(self):
        """Test that capacity is the sum of pool size and queue size."""
        executor = InferenceExecutor(max_workers=4, timeout=5.0, max_queue=16)
        assert executor.capacity == 20
        assert executor.max_queue == 16
        executor.shutdown(wait=False)


class TestInferenceExecutorProperties:
    """Tests for InferenceExecutor property accessors."""

    def test_timeout_property(self):
        """Test timeout property returns configured value."""
        executor = InferenceExecutor(max_workers=2, timeout=15.0)
        assert executor.timeout == 15.0
        executor.shutdown(wait=False)

    def test_max_workers_property(self):
        """Test max_workers property returns configured value."""
        executor = InferenceExecutor(max_workers=8, timeout=30.0)
        assert executor.max_workers == 8
        executor.shutdown(wait=False)
