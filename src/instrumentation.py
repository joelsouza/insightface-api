"""
New Relic instrumentation utilities.

Thin wrappers around the New Relic agent API that gracefully degrade
when the agent is not installed or not active. Centralizes the
conditional import so business logic stays clean.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

try:
    import newrelic.agent

    HAS_NEWRELIC = True
except ImportError:
    newrelic = None  # type: ignore[assignment]
    HAS_NEWRELIC = False


@contextmanager
def trace(name: str, group: str = "Custom") -> Generator[None, None, None]:
    """Create a named FunctionTrace segment in the current transaction.

    Shows up as a distinct segment in the NR transaction breakdown,
    allowing drill-down into where wall time is spent.

    No-op when New Relic agent is not installed or no transaction is active.
    """
    if HAS_NEWRELIC:
        with newrelic.agent.FunctionTrace(name=name, group=group):
            yield
    else:
        yield


def add_attribute(key: str, value: Any) -> None:
    """Add a custom attribute to the current transaction.

    Attributes are queryable in NRQL via WHERE/FACET clauses.
    """
    if HAS_NEWRELIC:
        newrelic.agent.add_custom_attribute(key, value)


def record_metric(name: str, value: float) -> None:
    """Record a custom metric (time-series) for dashboards and alerting.

    Metrics are available under Custom/<name> in the NR Metrics explorer.
    """
    if HAS_NEWRELIC:
        newrelic.agent.record_custom_metric(f"Custom/{name}", value)


def notice_error() -> None:
    """Report the current exception to New Relic Error Analytics.

    Must be called inside an except block or error handler where
    sys.exc_info() returns the active exception.
    """
    if HAS_NEWRELIC:
        newrelic.agent.notice_error()


def background_task(name: str, group: str = "Python") -> Any:
    """Decorator: wrap a function as a New Relic background task.

    Creates a non-web transaction visible in the NR Transactions page.
    Useful for startup work (model loading) and periodic jobs.
    """
    if HAS_NEWRELIC:
        return newrelic.agent.background_task(name=name, group=group)

    def passthrough(func: Any) -> Any:
        return func

    return passthrough
