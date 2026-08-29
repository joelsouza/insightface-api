"""Tests for the optional New Relic instrumentation helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import src.instrumentation as instrumentation


def test_helpers_are_noops_without_newrelic() -> None:
    """Instrumentation must not change application behavior without the agent."""
    app = object()

    with patch.object(instrumentation, "HAS_NEWRELIC", False):
        assert instrumentation.asgi_wrap(app) is app
        instrumentation.add_attribute("key", "value")
        instrumentation.record_event("Event", {"key": "value"})
        instrumentation.set_transaction_name("name", "group")
        instrumentation.ignore_transaction()
        instrumentation.notice_error()

        with instrumentation.trace("segment"):
            pass


def test_asgi_wrap_uses_newrelic_framework_metadata() -> None:
    """The explicit wrapper must identify Quart and Uvicorn to New Relic."""
    app = object()
    wrapped = object()

    with patch.object(instrumentation, "HAS_NEWRELIC", True), patch.object(
        instrumentation.newrelic.agent,
        "ASGIApplicationWrapper",
        return_value=wrapped,
    ) as wrapper:
        assert instrumentation.asgi_wrap(app) is wrapped

    wrapper.assert_called_once_with(
        app,
        framework=("Quart", instrumentation.quart_version),
        dispatcher=("Uvicorn", instrumentation.uvicorn_version),
    )


def test_get_app_returns_the_explicitly_wrapped_app() -> None:
    """The Uvicorn factory result must be the New Relic-wrapped app."""
    import src.app_async as app_async

    created = object()
    wrapped = object()
    previous = app_async._app_instance
    try:
        app_async._app_instance = None
        with patch.object(app_async, "create_async_app", return_value=created), patch.object(
            app_async, "asgi_wrap", return_value=wrapped
        ):
            assert app_async.get_app() is wrapped
    finally:
        app_async._app_instance = previous


@pytest.mark.asyncio
async def test_represent_emits_one_event_with_pipeline_fields(
    async_app, mock_face, png_bytes
) -> None:
    """A successful represent request emits one complete custom event."""
    from werkzeug.datastructures import FileStorage
    import io

    async_app.config["model_manager"].model = MagicMock()
    async_app.config["model_manager"].model.get.return_value = [mock_face]

    with patch("src.app_async.record_event") as record_event:
        async with async_app.test_client() as client:
            response = await client.post(
                "/represent",
                files={
                    "image_file": FileStorage(
                        stream=io.BytesIO(png_bytes),
                        filename="test.png",
                        content_type="image/png",
                    )
                },
            )

    assert response.status_code == 200
    record_event.assert_called_once()
    name, params = record_event.call_args.args
    assert name == "FaceRepresent"
    assert params["status_code"] == 200
    assert params["input_mode"] == "file"
    assert params["image_bytes"] == len(png_bytes)
    assert params["faces_detected"] == 1
    for field in (
        "request_id",
        "image_width",
        "image_height",
        "downscaled",
        "queue_wait_ms",
        "decode_ms",
        "detect_ms",
        "align_ms",
        "embed_ms",
        "extract_ms",
        "total_ms",
        "pool_size",
        "max_queue",
        "in_flight",
    ):
        assert field in params


@pytest.mark.asyncio
async def test_overload_emits_error_event(async_app, png_bytes) -> None:
    """An operational 503 still emits an event with its stable error code."""
    import io
    from werkzeug.datastructures import FileStorage

    from src.exceptions import ServiceOverloadedError

    with patch.object(
        async_app.config["inference_executor"],
        "run",
        side_effect=ServiceOverloadedError("full"),
    ), patch("src.app_async.record_event") as record_event, patch(
        "src.app_async.notice_error"
    ) as notice_error:
        async with async_app.test_client() as client:
            response = await client.post(
                "/represent",
                files={
                    "image_file": FileStorage(
                        stream=io.BytesIO(png_bytes),
                        filename="test.png",
                        content_type="image/png",
                    )
                },
            )

    assert response.status_code == 503
    record_event.assert_called_once()
    _, params = record_event.call_args.args
    assert params["status_code"] == 503
    assert params["error_code"] == "OVERLOADED"
    notice_error.assert_called_once()
