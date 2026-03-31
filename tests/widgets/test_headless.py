"""Tests for headless widget rendering, focusing on the async event-loop path.

The async path (via _render_widget_async) uses Playwright's native async API
instead of the old ThreadPoolExecutor + sync Playwright approach, which
deadlocked in jupyter nbconvert --execute due to greenlet conflicts.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pymatviz.widgets._headless import (
    _SVG_EXTRACT_JS,
    _build_html,
    _has_running_event_loop,
    _render_widget_async,
    render_widget_headless,
)


HEADLESS_PATH = "pymatviz.widgets._headless"

DUMMY_WIDGET_DATA: dict[str, Any] = {"widget_type": "scatter_plot", "style": ""}
DUMMY_ESM = "export default { render() {} }"
DUMMY_CSS = ".widget {}"


# === _SVG_EXTRACT_JS constant ===


def test_svg_extract_js_is_callable_js() -> None:
    """The shared SVG extraction JS is a non-empty string containing key logic."""
    assert isinstance(_SVG_EXTRACT_JS, str)
    assert "querySelectorAll" in _SVG_EXTRACT_JS
    assert "XMLSerializer" in _SVG_EXTRACT_JS
    assert len(_SVG_EXTRACT_JS) > 100


# === _has_running_event_loop ===


def test_has_running_event_loop_outside_loop() -> None:
    """Returns False when no event loop is running."""
    assert _has_running_event_loop() is False


def test_has_running_event_loop_inside_loop() -> None:
    """Returns True when called from inside a running event loop."""
    result = None

    async def check() -> None:
        nonlocal result
        result = _has_running_event_loop()

    asyncio.run(check())
    assert result is True


# === Async path dispatch ===


def test_render_widget_headless_dispatches_async_in_event_loop() -> None:
    """render_widget_headless calls _render_widget_async when inside an event loop."""
    pytest.importorskip("nest_asyncio")
    import nest_asyncio

    fake_png = b"\x89PNG"

    async def fake_render(*_args: Any, **_kwargs: Any) -> bytes:
        return fake_png

    with (
        patch(f"{HEADLESS_PATH}._has_running_event_loop", return_value=True),
        patch(
            f"{HEADLESS_PATH}._render_widget_async", side_effect=fake_render
        ) as mock_async,
    ):
        loop = asyncio.new_event_loop()
        nest_asyncio.apply(loop)

        async def run_in_loop() -> bytes:
            return render_widget_headless(
                DUMMY_WIDGET_DATA, DUMMY_ESM, DUMMY_CSS, fmt="png"
            )

        result = loop.run_until_complete(run_in_loop())
        loop.close()

    assert result == fake_png
    mock_async.assert_called_once()
    call_args = mock_async.call_args[0]
    assert call_args[0] == {**DUMMY_WIDGET_DATA, "show_controls": False}
    assert call_args[3] == "png"


def test_render_widget_headless_uses_sync_path_without_event_loop() -> None:
    """render_widget_headless uses the sync Playwright path outside event loops."""
    fake_png = b"\x89PNG sync"

    mock_page = MagicMock()
    mock_page.goto = MagicMock()
    mock_page.wait_for_function = MagicMock()
    mock_page.evaluate.return_value = None
    mock_page.locator.return_value.screenshot.return_value = fake_png
    mock_page.close = MagicMock()

    mock_browser = MagicMock()
    mock_browser.new_page.return_value = mock_page

    with (
        patch(f"{HEADLESS_PATH}._has_running_event_loop", return_value=False),
        patch(f"{HEADLESS_PATH}._get_browser", return_value=mock_browser),
        patch(f"{HEADLESS_PATH}._render_widget_async") as mock_async,
    ):
        result = render_widget_headless(
            DUMMY_WIDGET_DATA, DUMMY_ESM, DUMMY_CSS, fmt="png"
        )

    mock_async.assert_not_called()
    assert result == fake_png
    mock_browser.new_page.assert_called_once()


# === _render_widget_async ===


def test_render_widget_async_calls_playwright_async_api() -> None:
    """_render_widget_async uses async Playwright API end-to-end."""
    fake_png = b"\x89PNG async"

    mock_locator = MagicMock()
    mock_locator.screenshot = AsyncMock(return_value=fake_png)

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.wait_for_function = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=None)
    mock_page.locator = MagicMock(return_value=mock_locator)
    mock_page.close = AsyncMock()

    mock_browser = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_browser.is_connected.return_value = True

    async def run() -> bytes:
        with patch(f"{HEADLESS_PATH}._get_async_browser", return_value=mock_browser):
            return await _render_widget_async(
                DUMMY_WIDGET_DATA, DUMMY_ESM, DUMMY_CSS, fmt="png"
            )

    result = asyncio.run(run())

    assert result == fake_png
    mock_browser.new_page.assert_awaited_once()
    mock_page.goto.assert_awaited_once()
    mock_page.wait_for_function.assert_awaited_once()
    mock_page.close.assert_awaited_once()


def test_render_widget_async_propagates_render_error() -> None:
    """_render_widget_async raises RuntimeError when JS reports a render error."""
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.wait_for_function = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value="ESM bundle has no render export")
    mock_page.close = AsyncMock()

    mock_browser = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_browser.is_connected.return_value = True

    async def run() -> None:
        with patch(f"{HEADLESS_PATH}._get_async_browser", return_value=mock_browser):
            await _render_widget_async(
                DUMMY_WIDGET_DATA, DUMMY_ESM, DUMMY_CSS, fmt="png"
            )

    with pytest.raises(RuntimeError, match="render failed"):
        asyncio.run(run())


# === _build_html ===


def test_build_html_contains_esm_and_widget_data() -> None:
    """Generated HTML embeds the ESM bundle and widget data."""
    html = _build_html(
        {"widget_type": "bar_plot", "style": "height: 300px"},
        "export default { render() {} }",
        ".widget {}",
    )
    assert "bar_plot" in html
    assert "300px" in html
    assert "__RENDER_DONE" in html
    assert "blob:" not in html  # blob URL is created at runtime, not in source
    assert "atob(" in html  # base64 decode of ESM


def test_build_html_applies_dimension_overrides() -> None:
    """Explicit width/height override widget style defaults."""
    html = _build_html(DUMMY_WIDGET_DATA, DUMMY_ESM, DUMMY_CSS, width=1200, height=900)
    assert "1200px" in html
    assert "900px" in html


# === No ThreadPoolExecutor remnants ===


def test_no_thread_pool_executor_in_module() -> None:
    """The old ThreadPoolExecutor-based async offload has been removed."""
    from pymatviz.widgets import _headless

    assert not hasattr(_headless, "_async_pool"), (
        "_async_pool should not exist — async path now uses Playwright async API"
    )
    assert not hasattr(_headless, "_async_pool_lock"), (
        "_async_pool_lock should not exist"
    )
