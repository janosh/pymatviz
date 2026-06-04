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
    _HAS_CANVAS_JS,
    _SVG_EXTRACT_JS,
    _build_html,
    _capture_page,
    _has_running_event_loop,
    _prepare_render,
    _render_widget_async,
    build_interactive_html,
    png_blank_fraction,
    render_widget_headless,
)
from pymatviz.widgets.chem_pot_diagram import ChemPotDiagramWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


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
    # Icon fallback was removed: only chart-sized SVGs (>= 20px) are returned,
    # so toolbar/legend icons can never be mistaken for the chart.
    assert "fallback" not in _SVG_EXTRACT_JS


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


# === Content-aware capture (_capture_page) ===
#
# Capture picks SVG-vs-raster and vector-vs-rasterized PDF from the rendered DOM
# (_HAS_CANVAS_JS / _SVG_EXTRACT_JS), not a static widget_type list. The async
# twin mirrors this logic and is smoke-tested via _render_widget_async above.


@pytest.mark.parametrize(
    ("fmt", "dpi", "expected_scale"),
    [("png", 144, 2.0), ("pdf", 72, 1.0), ("pdf", 216, 3.0), ("svg", 300, 1)],
)
def test_prepare_render_scale_factor(fmt: str, dpi: int, expected_scale: float) -> None:
    """Raster formats (png/jpeg/pdf) scale by DPI; SVG stays 1x (pure vector)."""
    import os

    tmp_path, _ms, scale = _prepare_render(
        {"widget_type": "bar_plot"}, "// esm", "/* css */", fmt, dpi, 30.0, None, None
    )
    try:
        assert scale == expected_scale
    finally:
        os.unlink(tmp_path)


def _mock_page(*, svg_result: str | None = None, has_canvas: bool = False) -> MagicMock:
    """Mock sync Playwright page whose evaluate() answers the SVG/canvas probes."""
    responses = {_SVG_EXTRACT_JS: svg_result, _HAS_CANVAS_JS: has_canvas}
    page = MagicMock()
    page.evaluate = MagicMock(side_effect=responses.get)
    page.locator.return_value.screenshot.return_value = b"PNGDATA"
    page.locator.return_value.bounding_box.return_value = {"width": 100, "height": 80}
    page.pdf.return_value = b"%PDF-vector"
    return page


def test_capture_page_svg_returns_chart() -> None:
    """SVG export returns the extracted chart SVG when one is present."""
    page = _mock_page(svg_result="<svg>x</svg>")
    assert _capture_page(page, "svg") == b"<svg>x</svg>"


@pytest.mark.parametrize(
    ("has_canvas", "match"), [(True, "WebGL"), (False, "No SVG element")]
)
def test_capture_page_svg_errors(has_canvas: bool, match: str) -> None:
    """SVG export of a canvas/HTML-only widget raises a content-tailored error."""
    with pytest.raises(RuntimeError, match=match):
        _capture_page(_mock_page(has_canvas=has_canvas), "svg")


def test_capture_page_pdf_canvas_rasterizes() -> None:
    """Canvas widgets are screenshot-wrapped to PDF (vector page.pdf can't see them)."""
    page = _mock_page(has_canvas=True)
    with patch(f"{HEADLESS_PATH}._png_to_pdf", return_value=b"%PDF-raster") as mock_pdf:
        assert _capture_page(page, "pdf", scale_factor=2.0) == b"%PDF-raster"
    mock_pdf.assert_called_once_with(b"PNGDATA", scale=2.0)
    page.pdf.assert_not_called()


def test_capture_page_pdf_no_canvas_uses_vector() -> None:
    """SVG/HTML widgets get Chromium's native vector PDF via page.pdf()."""
    page = _mock_page(has_canvas=False)
    assert _capture_page(page, "pdf") == b"%PDF-vector"
    page.pdf.assert_called_once()


# === to_html / build_interactive_html (browser-free) ===


def _scatter() -> ScatterPlotWidget:
    """Minimal scatter widget for HTML export tests."""
    return ScatterPlotWidget(
        series=[{"x": [0, 1], "y": [1, 2], "label": "s"}], style="height: 300px;"
    )


def test_build_interactive_html_requires_one_esm_source() -> None:
    """build_interactive_html demands exactly one of esm_url / esm_content."""
    for kwargs in ({}, {"esm_url": "x", "esm_content": "y"}):
        with pytest.raises(ValueError, match="exactly one of esm_url or esm_content"):
            build_interactive_html(
                {"widget_type": "scatter_plot"}, "/* css */", **kwargs
            )


def test_to_html_default_uses_cdn_url() -> None:
    """Default to_html references the matterviz-anywidget npm bundle + widget data."""
    html = _scatter().to_html()
    assert "https://cdn.jsdelivr.net/npm/matterviz-anywidget@" in html
    assert "matterviz.js" in html
    assert '"scatter_plot"' in html
    assert "atob(" not in html  # external, not inlined


def test_to_html_inline_embeds_bundle() -> None:
    """inline=True embeds the base64 bundle and references no CDN."""
    html = _scatter().to_html(inline=True)
    assert "atob(" in html
    assert "cdn.jsdelivr.net" not in html


def test_to_html_worker_shim_only_for_chem_pot() -> None:
    """to_html disables Worker for chem_pot (CDN/blob worker can't load) only."""
    chem_pot = ChemPotDiagramWidget(
        entries=[{"name": "Li", "energy": -1.9, "composition": {"Li": 1}}]
    )
    assert "globalThis.Worker = undefined" in chem_pot.to_html(inline=True)
    assert "globalThis.Worker = undefined" not in _scatter().to_html(inline=True)


def test_to_html_inline_and_esm_url_conflict() -> None:
    """Passing both inline and esm_url is rejected."""
    with pytest.raises(ValueError, match="not both"):
        _scatter().to_html(inline=True, esm_url="https://x/y.js")


def test_to_html_writes_file(tmp_path: Any) -> None:
    """to_html writes the returned HTML to disk when given a filename."""
    out_path = f"{tmp_path}/widget.html"
    html = _scatter().to_html(out_path, esm_url="https://example.com/mv.js")
    assert "https://example.com/mv.js" in html
    with open(out_path, encoding="utf-8") as file:
        assert file.read() == html


def test_png_blank_fraction_uniform_image() -> None:
    """png_blank_fraction returns 1.0 for a single-color image."""
    pytest.importorskip("PIL")
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="PNG")
    assert png_blank_fraction(buf.getvalue()) == 1.0
