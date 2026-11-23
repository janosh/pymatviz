"""Tests for headless widget rendering: browser thread, warm pages and capture."""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import functools
import importlib.util
import io
import json
import os
import threading
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

import pymatviz.widgets._headless as headless_mod
from pymatviz.widgets._headless import (
    _MOUNT_JS,
    _PROBE_JS,
    _SVG_EXPORT_JS,
    _TRANSPARENT_STYLE,
    _build_html,
    _trim_png,
    build_interactive_html,
    png_blank_fraction,
    render_diagnostics,
    render_widget_headless,
)
from pymatviz.widgets.chem_pot_diagram import ChemPotDiagramWidget
from pymatviz.widgets.matterviz import (
    MATTERVIZ_ANYWIDGET_VERSION,
    MatterVizWidget,
    _bundle_sources,
)
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


if TYPE_CHECKING:
    from collections.abc import Generator, Iterator

HEADLESS_PATH = "pymatviz.widgets._headless"

DUMMY_WIDGET_DATA: dict[str, Any] = {"widget_type": "scatter_plot", "style": ""}
DUMMY_ESM = "export default { render() {} }"
DUMMY_CSS = ".widget {}"
DUMMY_JOB = dict(
    widget_data=DUMMY_WIDGET_DATA, esm_content=DUMMY_ESM, css_content=DUMMY_CSS
)
_render = functools.partial(
    render_widget_headless, DUMMY_WIDGET_DATA, DUMMY_ESM, DUMMY_CSS
)


@pytest.fixture(autouse=True)
def _isolated_warm_pages() -> Iterator[None]:
    """Hide warm pages of real renders from mocked ones (and vice versa)."""
    with patch.object(headless_mod, "_warm_pages", []):
        yield


def _mock_page(
    *,
    svg_result: dict[str, str] | None = None,
    webgl: bool = False,
    render_error: str | None = None,
    on_mount: Any = None,
) -> MagicMock:
    """Mock async Playwright page answering the bundle-ready, mount, SVG and content
    probes; ``on_mount`` (an async callable) runs inside each mount.
    """

    async def evaluate(js: str, *_args: object) -> object:
        if page.is_closed():
            raise RuntimeError("Target page, context or browser has been closed")
        if js == _MOUNT_JS:
            if on_mount is not None:
                await on_mount()
            return render_error
        responses = {
            "() => window.__matterviz_ready": True,
            _SVG_EXPORT_JS: svg_result,
            _PROBE_JS: {
                "content_type": "canvas" if webgl else "svg",
                "webgl": webgl,
                "root_w": 100,
                "root_h": 80,
                "clipped": False,
            },
        }
        return responses.get(js)

    page = MagicMock()
    page.evaluate = AsyncMock(side_effect=evaluate)
    for method in ("goto", "wait_for_function", "emulate_media"):
        setattr(page, method, AsyncMock())
    page.is_closed.return_value = False

    async def close() -> None:
        page.is_closed.return_value = True

    page.close = AsyncMock(side_effect=close)
    page.pdf = AsyncMock(return_value=b"%PDF-vector")
    root = page.locator.return_value
    root.screenshot = AsyncMock(return_value=b"PNGDATA")
    root.bounding_box = AsyncMock(return_value={"width": 100, "height": 80})
    return page


@contextlib.contextmanager
def _serve_pages(*pages: MagicMock) -> Generator[MagicMock]:
    """Patch in a mock browser whose new_page() hands out ``pages`` in order."""
    browser = MagicMock()
    browser.new_page = AsyncMock(side_effect=list(pages))
    with patch(f"{HEADLESS_PATH}._get_browser", AsyncMock(return_value=browser)):
        yield browser


def _mount_count(page: MagicMock) -> int:
    """Number of widgets mounted on a mock page."""
    return sum(call.args[0] == _MOUNT_JS for call in page.evaluate.await_args_list)


def test_diagnostics_reports_browser_startup_failure() -> None:
    """Browser launch failures return a report instead of raising."""
    with patch(
        f"{HEADLESS_PATH}._get_browser",
        AsyncMock(side_effect=RuntimeError("missing Chromium")),
    ):
        report = render_diagnostics(DUMMY_WIDGET_DATA, DUMMY_ESM, DUMMY_CSS)
    assert report.ok is False
    assert report.error == "render failed: missing Chromium"


def test_sync_export_inside_running_loop() -> None:
    """A sync export from inside a running event loop (Jupyter) blocks on the browser
    thread instead of re-entering the caller's loop, so needs no nest_asyncio.
    """

    async def export_in_loop() -> tuple[bytes, threading.Thread]:
        """Export synchronously while this loop runs, as a notebook cell does."""
        return _render(), threading.current_thread()

    with _serve_pages(_mock_page()):
        png, caller_thread = asyncio.run(export_in_loop())
    assert png == b"PNGDATA"
    assert caller_thread is threading.main_thread()
    assert headless_mod._get_browser_thread().thread is not caller_thread


@pytest.mark.parametrize(
    ("fmt", "dpi", "webgl", "expected_scale", "expected_bytes", "transparent"),
    [
        ("png", 72, False, 1.0, b"\x89PNG dpi", False),
        ("png", 144, False, 2.0, b"\x89PNG dpi", False),
        ("png", 36, False, 1.0, b"\x89PNG dpi", False),  # sub-72 DPI clamps to 1x
        ("jpeg", 216, False, 3.0, b"\xff\xd8 dpi", False),
        ("svg", 144, False, 2.0, b"<svg>dpi</svg>", False),  # canvas layers at DPI
        ("pdf", 72, False, 1.0, b"%PDF-vector", False),  # SVG/HTML/2D canvas: vector
        ("pdf", 216, True, 3.0, b"%PDF-raster", False),  # print can't see WebGL
        ("png", 144, False, 2.0, b"trimmed", True),
        ("svg", 72, False, 1.0, b"<svg>trimmed</svg>", True),
    ],
)
def test_render_widget_headless_formats_and_dpi(
    fmt: str,
    dpi: int,
    webgl: bool,
    expected_scale: float,
    expected_bytes: bytes,
    transparent: bool,
) -> None:
    """DPI maps to the browser scale for every format, PDFs rasterize iff the widget
    shows WebGL, and transparent exports drop backgrounds and trim (PNG by pixels at
    the export scale, SVG by viewBox).
    """
    svg_result = {"svg": expected_bytes.decode()} if fmt == "svg" else None
    page = _mock_page(svg_result=svg_result, webgl=webgl)
    page.locator.return_value.screenshot.return_value = expected_bytes

    with (
        _serve_pages(page) as browser,
        patch(f"{HEADLESS_PATH}._png_to_pdf", return_value=b"%PDF-raster") as to_pdf,
        patch(f"{HEADLESS_PATH}._trim_png", return_value=b"trimmed") as trim_png,
    ):
        result = _render(fmt=fmt, dpi=dpi, transparent=transparent)

    assert result == expected_bytes
    mount_style = next(
        call.args[1][1]["style"]
        for call in page.evaluate.await_args_list
        if call.args[0] == _MOUNT_JS
    )
    # the transparent theme goes before the widget's own style, which may override it
    theme = f"{_TRANSPARENT_STYLE}; " if transparent else ""
    assert mount_style.startswith(f"{theme}width: 800px")
    if fmt == "png":
        page.locator.return_value.screenshot.assert_awaited_once_with(
            type="png", omit_background=transparent
        )
        if transparent:  # 4 CSS px of margin at the 2x export scale
            trim_png.assert_called_once_with(b"trimmed", 8)
    browser.new_page.assert_awaited_once_with(
        viewport={"width": 1024, "height": 768}, device_scale_factor=expected_scale
    )
    if fmt == "svg":  # MatterViz embeds canvas layers at the export DPI
        page.evaluate.assert_any_await(_SVG_EXPORT_JS, [dpi, 4 if transparent else 0])
    if webgl:
        to_pdf.assert_called_once_with(expected_bytes, scale=expected_scale)
    assert page.pdf.await_count == (fmt == "pdf" and not webgl)


# === Warm page reuse ===


def test_warm_page_reused_per_bundle_and_dpi() -> None:
    """Exports sharing bundle, CSS and DPI mount into one page (unmounting after
    each, also after a failed capture); a new DPI or bundle gets its own page.
    """
    pages = [_mock_page(svg_result={"error": "no vector form"}) for _ in range(3)]
    with _serve_pages(*pages) as browser:
        for _ in range(3):
            _render(dpi=72)
        with pytest.raises(RuntimeError, match="SVG export failed: no vector form"):
            _render(fmt="svg", dpi=72)
        render_diagnostics(DUMMY_WIDGET_DATA, DUMMY_ESM, DUMMY_CSS, dpi=72)
        _render(dpi=144)
        render_widget_headless(DUMMY_WIDGET_DATA, f"{DUMMY_ESM};", DUMMY_CSS, dpi=72)

    assert browser.new_page.await_count == 3
    assert [_mount_count(page) for page in pages] == [5, 1, 1]
    unmounts = [
        call
        for call in pages[0].evaluate.await_args_list
        if call.args[0] == "() => window.__matterviz_unmount()"
    ]
    assert len(unmounts) == 5
    for page in pages:
        page.close.assert_not_awaited()


def test_failed_render_retires_its_page() -> None:
    """A failing export closes and retires its page while a concurrent export
    renders on its own fresh page, which stays warm.
    """
    mounting = threading.Event()

    async def on_mount() -> None:
        """Let the queued export enqueue while this mount is in flight."""
        mounting.set()
        await asyncio.sleep(0.2)

    broken = _mock_page(render_error="boom", on_mount=on_mount)
    fresh = _mock_page()
    results: dict[str, object] = {}

    def export(name: str) -> None:
        """Export from a worker thread, recording the result or error."""
        try:
            results[name] = _render()
        except RuntimeError as exc:
            results[name] = str(exc)

    with _serve_pages(broken, fresh):
        failing = threading.Thread(target=export, args=("failing",))
        failing.start()
        assert mounting.wait(5)
        queued = threading.Thread(target=export, args=("queued",))
        queued.start()
        failing.join()
        queued.join()
    assert results == {"failing": "Widget render failed: boom", "queued": b"PNGDATA"}
    assert (_mount_count(broken), _mount_count(fresh)) == (1, 1)
    (warm,) = headless_mod._warm_pages
    assert warm.page is fresh
    broken.close.assert_awaited_once()
    fresh.close.assert_not_awaited()


def test_run_in_browser_thread_bounds_wait() -> None:
    """A stuck export times out (cancelling its coroutine), and a call from the
    browser thread itself errors instead of deadlocking it.
    """
    cancelled = threading.Event()

    async def stuck() -> None:
        """Never finish unless cancelled."""
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    with (
        patch(f"{HEADLESS_PATH}._call_budget", return_value=0.1),
        pytest.raises(TimeoutError, match=r"did not finish within 0\.1s"),
    ):
        headless_mod._run_in_browser_thread(stuck(), timeout=1)
    assert cancelled.wait(5)

    async def nested() -> None:
        """Export synchronously from a coroutine on the browser thread."""
        headless_mod._run_in_browser_thread(asyncio.sleep(0))

    with pytest.raises(RuntimeError, match="from the browser thread would deadlock"):
        headless_mod._run_in_browser_thread(nested())


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs os.fork")
def test_forked_child_starts_own_browser_thread() -> None:
    """A child forked after the parent started its browser thread gets a fresh one
    (the parent's didn't survive the fork) instead of blocking on a dead loop.
    """
    parent_thread = headless_mod._get_browser_thread().thread
    headless_mod._warm_pages.append(MagicMock())
    pid = os.fork()
    if pid == 0:  # child: exit code reports the outcome, never return into pytest
        patch(f"{HEADLESS_PATH}._call_budget", return_value=5.0).start()  # fail fast
        try:
            pages_cleared = headless_mod._warm_pages == []
            browser_thread = headless_mod._get_browser_thread().thread
            ok = pages_cleared and browser_thread is not parent_thread
            if ok:
                coro = asyncio.sleep(0, result=42)
                ok = headless_mod._run_in_browser_thread(coro, timeout=1) == 42
        except BaseException:  # noqa: BLE001
            ok = False
        os._exit(0 if ok else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert len(headless_mod._warm_pages) == 1  # the parent's cache is untouched


def test_idle_warm_pages_evicted_lru() -> None:
    """Beyond _MAX_WARM_PAGES idle pages, the least recently used one closes."""
    n_pages = headless_mod._MAX_WARM_PAGES + 1
    pages = [_mock_page() for _ in range(n_pages)]
    with _serve_pages(*pages):
        for dpi in range(72, 72 * (n_pages + 1), 72):
            _render(dpi=dpi)
        assert len(headless_mod._warm_pages) == headless_mod._MAX_WARM_PAGES
    pages[0].close.assert_awaited_once()
    for page in pages[1:]:
        page.close.assert_not_awaited()


def test_batch_exports_render_concurrently_on_capped_pages() -> None:
    """render_widgets_headless renders jobs on separate pages, at most
    _MAX_WARM_PAGES at once, reusing freed pages for the rest, in job order.
    """
    active, peak = 0, 0

    async def on_mount() -> None:
        """Record how many mounts overlap."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.05)
        active -= 1

    n_slots = headless_mod._MAX_WARM_PAGES
    pages = [_mock_page(on_mount=on_mount) for _ in range(n_slots)]
    jobs = [DUMMY_JOB] * (n_slots + 2)
    with _serve_pages(*pages) as browser:
        results = headless_mod.render_widgets_headless(jobs)
    assert results == [b"PNGDATA"] * len(jobs)
    assert (peak, browser.new_page.await_count) == (n_slots, n_slots)
    assert sum(map(_mount_count, pages)) == len(jobs)


def test_bundle_load_error_surfaces() -> None:
    """A bundle that fails to import reports its error and leaves no page behind."""
    page = _mock_page()
    page.evaluate = AsyncMock(return_value="Error: ESM bundle has no render export")
    with (
        _serve_pages(page),
        pytest.raises(RuntimeError, match=r"render failed: .*no render export"),
    ):
        _render()
    page.close.assert_awaited_once()
    assert headless_mod._warm_pages == []


# === Real Chromium with a synthetic bundle ===

# Renders an <svg>, then (per ``kind``) throws asynchronously, rejects a promise, or
# creates a WebGL canvas plus one without any context, kept on window for probing
SYNTHETIC_ESM = """\
export const export_svg = (el) => el.dataset.svg;
export default { render({ model, el }) {
  el.innerHTML = '<svg width="100" height="100"><rect width="50" height="50"/></svg>';
  el.dataset.svg = model.get("svg") ?? "";
  el.style.maxWidth = "100%";  // as the bundle caps widgets at their parent's width
  const kind = model.get("kind");
  if (kind === "throw") setTimeout(() => { throw new Error("async boom"); }, 0);
  if (kind === "reject") Promise.reject(new Error("rejected boom"));
  if (kind === "canvases") {
    const [gl_canvas, bare] = [0, 1].map(() => el.appendChild(
      document.createElement("canvas")));
    window.__gl = gl_canvas.getContext("webgl");
    window.__bare = bare;
  }
} };
"""
needs_playwright = pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None, reason="needs playwright"
)


@needs_playwright
@pytest.mark.parametrize(
    ("kind", "error"),
    [("throw", "Error: async boom"), ("reject", "Error: rejected boom")],
)
def test_uncaught_widget_error_fails_render(kind: str, error: str) -> None:
    """Errors a widget throws outside render (effects, timers, promises) fail the
    export and the render report instead of passing as a successful render.
    """
    data = {**DUMMY_WIDGET_DATA, "kind": kind}
    expected = f"Uncaught {error}"
    with pytest.raises(RuntimeError, match=f"Widget render failed: {expected}"):
        render_widget_headless(data, SYNTHETIC_ESM, DUMMY_CSS, timeout=10)
    report = render_diagnostics(data, SYNTHETIC_ESM, DUMMY_CSS, timeout=10)
    assert (report.ok, report.error) == (False, expected)
    assert headless_mod._warm_pages == []  # failed pages are retired


@needs_playwright
def test_webgl_pdf_rasterized_and_contexts_freed() -> None:
    """A widget's WebGL canvas makes its PDF raster, and unmount frees its WebGL
    contexts without creating contexts on canvases that had none.
    """
    data = {**DUMMY_WIDGET_DATA, "kind": "canvases"}
    with patch(f"{HEADLESS_PATH}._png_to_pdf", return_value=b"%PDF-raster"):
        pdf = render_widget_headless(
            data, SYNTHETIC_ESM, DUMMY_CSS, fmt="pdf", dpi=72, timeout=10
        )
    assert pdf == b"%PDF-raster"
    (warm,) = headless_mod._warm_pages
    probe = "() => [window.__gl?.isContextLost(), !!window.__bare.getContext('2d')]"
    try:
        assert headless_mod._run_in_browser_thread(warm.page.evaluate(probe)) == [
            True,
            True,
        ]
    finally:
        headless_mod._run_in_browser_thread(headless_mod._retire(warm))


@needs_playwright
@pytest.mark.parametrize(
    ("svg", "expected"),
    [
        # 10x5 rect at (20, 30) plus the 4px pad
        (
            '<rect x="20" y="30" width="10" height="5"/>',
            'viewBox="16 26 18 13" width="18" height="13"',
        ),
        ("", "trim failed: transparent export painted no pixels"),
        (None, "trim failed: exported SVG has no viewBox"),
    ],
)
def test_transparent_svg_trims_to_painted_pixels(
    svg: str | None, expected: str
) -> None:
    """Transparent SVG exports shrink the viewBox to the painted pixels plus a pad,
    and report unpaintable or unmeasurable SVGs as export errors.
    """
    view_box = "" if svg is None else ' viewBox="0 0 100 100"'
    data = {
        **DUMMY_WIDGET_DATA,
        "svg": f'<svg xmlns="http://www.w3.org/2000/svg"{view_box}>{svg or ""}</svg>',
    }
    export = functools.partial(
        render_widget_headless,
        data,
        SYNTHETIC_ESM,
        DUMMY_CSS,
        fmt="svg",
        dpi=72,
        timeout=10,
        transparent=True,
    )
    if expected.startswith("trim failed"):
        with pytest.raises(RuntimeError, match=f"SVG export failed: {expected}"):
            export()
    else:
        assert expected in export().decode()


@needs_playwright
def test_export_wider_than_page() -> None:
    """Widgets keep their export width past the 1024px page, though the bundle caps
    widgets at their parent's width.
    """
    png = render_widget_headless(
        DUMMY_WIDGET_DATA, SYNTHETIC_ESM, DUMMY_CSS, dpi=72, width=1300, timeout=10
    )
    assert Image.open(io.BytesIO(png)).size == (1300, 600)


# === _build_html ===


def test_build_html_contains_esm_and_mount() -> None:
    """The warm-page HTML embeds the CSS and base64 ESM, defines the mount hooks and
    hides loading spinners.
    """
    css = "body { color: red; }"
    html = _build_html("export default { render() {} }", css)
    assert f"<style>{css}</style>" in html
    assert "blob:" not in html  # blob URL is created at runtime, not in source
    assert "atob(" in html  # base64 decode of ESM
    hooks = ("mount", "unmount", "probe", "ready")
    for hook in (f"__matterviz_{name}" for name in hooks):
        assert f"window.{hook} =" in html
    # spinner guard so loading states are never captured
    assert '.spinner[role="status"]' in html


# === to_html / build_interactive_html (browser-free) ===


def _scatter() -> ScatterPlotWidget:
    """Minimal scatter widget for HTML export tests."""
    return ScatterPlotWidget(
        series=[{"x": [0, 1], "y": [1, 2], "label": "s"}], style="height: 300px;"
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"css_content": ""}, "esm_url or esm_content"),
        ({"css_content": "", "esm_url": "x", "esm_content": "y"}, "esm_url or esm"),
        ({"esm_url": "x"}, "css_url or css_content"),
        ({"esm_url": "x", "css_url": "y", "css_content": "z"}, "css_url or css"),
    ],
)
def test_build_interactive_html_requires_one_source_each(
    kwargs: dict[str, str], match: str
) -> None:
    """build_interactive_html demands exactly one ESM and one CSS source."""
    with pytest.raises(ValueError, match=f"exactly one of {match}"):
        build_interactive_html({"widget_type": "scatter_plot"}, **kwargs)


def test_widget_data_json_numpy_scalars_and_unknown_types() -> None:
    """Numpy scalars and arrays encode as Python values, buffers as base64 and
    ``<`` is escaped (``<!--<script>`` in a label kept exported pages from
    mounting); other objects raise, not stringify.
    """
    from pymatviz.widgets._headless import _widget_data_json

    data: dict[str, Any] = {"idx": np.int64(141), "val": np.float32(0.5)}
    data |= {"flag": np.bool_(1)}
    data |= {"arr": np.arange(2), "buf": memoryview(b"ab"), "tag": "</script>"}
    data |= {"label": "<!--<script>"}
    assert json.loads(_widget_data_json(data)) == {
        "idx": 141, "val": 0.5, "flag": True, "arr": [0, 1], "buf": "YWI=",
        "tag": "</script>", "label": "<!--<script>",
    }  # fmt: skip
    assert "<" not in _widget_data_json(data)
    with pytest.raises(TypeError, match="object is not JSON serializable"):
        _widget_data_json({"bad": object()})


def test_to_html_default_uses_cdn_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default to_html references the default-version bundle URL + widget data."""
    monkeypatch.delenv("MATTERVIZ_ANYWIDGET_DIR", raising=False)
    html = _scatter().to_html()
    esm_url, css_url = _bundle_sources(MATTERVIZ_ANYWIDGET_VERSION)
    assert esm_url in html  # split entry
    # the CSS is linked like the JS, not downloaded and inlined
    assert f'<link rel="stylesheet" href="{css_url}">' in html
    assert len(html) < 20_000
    assert '"scatter_plot"' in html
    assert "const esm_blob" not in html  # external, not inlined


def test_to_html_inline() -> None:
    """inline=True embeds the base64 bundle (no CDN), disables Worker only for
    chem_pot (a CDN/blob worker can't load) and conflicts with esm_url.
    """
    html = _scatter().to_html(inline=True)
    assert "const esm_blob" in html
    assert _bundle_sources(MATTERVIZ_ANYWIDGET_VERSION)[0] not in html
    assert ", el, false);" in html
    chem_pot = ChemPotDiagramWidget(
        entries=[{"name": "Li", "energy": -1.9, "composition": {"Li": 1}}]
    )
    assert ", el, true);" in chem_pot.to_html(inline=True)
    with pytest.raises(ValueError, match="not both"):
        _scatter().to_html(inline=True, esm_url="https://x/y.js")


def test_to_html_writes_file(tmp_path: Any) -> None:
    """to_html writes the returned HTML to disk when given a filename."""
    out_path = f"{tmp_path}/widget.html"
    html = _scatter().to_html(out_path, esm_url="https://example.com/mv.js")
    assert "https://example.com/mv.js" in html
    with open(out_path, encoding="utf-8") as file:
        assert file.read() == html


@pytest.mark.parametrize("fmt", ["jpeg", "pdf"])
def test_transparent_rejects_opaque_formats(fmt: str) -> None:
    """Transparent export raises for formats without alpha (JPEG) or trimming (PDF)."""
    with pytest.raises(ValueError, match=f"supports png and svg, got fmt='{fmt}'"):
        _render(fmt=fmt, transparent=True)


@pytest.mark.parametrize(
    ("mode", "painted_box", "pad", "expected_size"),
    [
        ("RGBA", (10, 20, 30, 25), 0, (20, 5)),
        ("RGBA", (10, 20, 30, 25), 4, (28, 13)),
        ("RGBA", (0, 0, 40, 40), 4, (40, 40)),  # pad clamps to the image edges
        ("RGB", (0, 0, 40, 40), 4, (40, 40)),  # Chromium drops alpha when opaque
        ("RGBA", None, 4, None),  # nothing painted
    ],
)
def test_trim_png(
    mode: str,
    painted_box: tuple[int, int, int, int] | None,
    pad: int,
    expected_size: tuple[int, int] | None,
) -> None:
    """_trim_png crops to opaque pixels plus a pad, and raises on blank captures."""
    img = Image.new(mode, (40, 40))
    if painted_box:
        img.paste((200, 0, 0, 255)[: len(mode)], painted_box)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    if expected_size is None:
        with pytest.raises(RuntimeError, match="no painted pixels"):
            _trim_png(buf.getvalue(), pad)
        return
    trimmed = Image.open(io.BytesIO(_trim_png(buf.getvalue(), pad)))
    assert trimmed.size == expected_size
    assert trimmed.getpixel((pad, pad)) == (200, 0, 0, 255)


def test_png_blank_fraction_uniform_image() -> None:
    """png_blank_fraction returns 1.0 for a single-color image."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 255, 255)).save(buf, format="PNG")
    assert png_blank_fraction(buf.getvalue()) == 1.0


def test_overlapping_exports_trim_idle_pages_on_release() -> None:
    """Exports that overlap beyond _MAX_WARM_PAGES each open a page; once they
    finish, idle pages are trimmed back to the limit.
    """
    n_exports = headless_mod._MAX_WARM_PAGES + 2
    barrier = threading.Barrier(n_exports)

    async def on_mount() -> None:
        """Hold each mount until every export has checked out its own page."""
        await asyncio.to_thread(barrier.wait, 5)

    pages = [_mock_page(on_mount=on_mount) for _ in range(n_exports)]

    with _serve_pages(*pages) as browser:
        threads = [threading.Thread(target=_render) for _ in range(n_exports)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    assert browser.new_page.await_count == n_exports
    assert len(headless_mod._warm_pages) == headless_mod._MAX_WARM_PAGES
    assert sum(page.close.await_count for page in pages) == 2


def test_batch_failure_cancels_sibling_renders() -> None:
    """A failing job cancels the batch's other renders before the error surfaces."""
    cancelled: list[bool] = []

    async def slow_mount() -> None:
        """Mount that only ends by cancellation."""
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise

    n_slow = headless_mod._MAX_WARM_PAGES - 1
    pages = [_mock_page(render_error="boom")]
    pages += [_mock_page(on_mount=slow_mount) for _ in range(n_slow)]
    with (
        _serve_pages(*pages),
        pytest.raises(RuntimeError, match="Widget render failed: boom"),
    ):
        headless_mod.render_widgets_headless([DUMMY_JOB] * (n_slow + 1))
    assert cancelled == [True] * n_slow


def test_batch_budget_covers_full_waves() -> None:
    """A batch may take one full single-export budget per wave of renders."""
    n_jobs = 5 * headless_mod._MAX_WARM_PAGES
    with patch(f"{HEADLESS_PATH}._run_in_browser_thread") as run:
        headless_mod.render_widgets_headless([DUMMY_JOB | {"timeout": 1}] * n_jobs)
    (coro, _), kwargs = run.call_args
    coro.close()
    assert kwargs["budget"] == 5 * headless_mod._call_budget(1)


def test_cancel_during_eviction_releases_page() -> None:
    """A render cancelled while evicting idle pages still releases and retires its
    checked-out page, instead of leaving it marked in use forever.
    """
    page = _mock_page()
    evict = AsyncMock(side_effect=[asyncio.CancelledError(), None])
    with (
        _serve_pages(page),
        patch(f"{HEADLESS_PATH}._evict_idle", evict),
        pytest.raises((asyncio.CancelledError, concurrent.futures.CancelledError)),
    ):
        _render()
    assert headless_mod._warm_pages == []
    page.close.assert_awaited_once()


# === to_img and mount options ===


_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50


def _make_widget() -> MatterVizWidget:
    """Create a bare scatter-plot MatterVizWidget for to_img tests."""
    return MatterVizWidget(widget_type="scatter_plot")


@pytest.fixture
def mock_render() -> Iterator[Mock]:
    """Patch the headless renderer to return fake PNG bytes."""
    with patch(
        f"{HEADLESS_PATH}.render_widget_headless", return_value=_FAKE_PNG
    ) as mock:
        yield mock


@pytest.mark.parametrize(
    ("filename", "explicit_fmt", "expected_capture_fmt"),
    [
        ("plot.png", None, "png"),
        ("plot.svg", None, "svg"),
        ("plot.pdf", None, "pdf"),
        ("plot.jpg", None, "jpeg"),
        ("plot.jpeg", None, "jpeg"),
        (None, "png", "png"),
        (None, "svg", "svg"),
        (None, "jpeg", "jpeg"),
        (None, "pdf", "pdf"),
        (None, None, "png"),
    ],
)
def test_to_img_format_inference(
    filename: str | None,
    explicit_fmt: str | None,
    expected_capture_fmt: str,
    tmp_path: Any,
    mock_render: Mock,
) -> None:
    """to_img infers capture format from filename or explicit fmt argument."""
    resolved = f"{tmp_path}/{filename}" if filename else None
    _make_widget().to_img(filename=resolved, fmt=explicit_fmt)  # ty: ignore[invalid-argument-type]
    assert mock_render.call_args.kwargs["fmt"] == expected_capture_fmt


@pytest.mark.parametrize(
    ("img_kwargs", "match"),
    [
        ({"filename": "plot.tiff"}, "Unsupported fmt='tiff'"),
        ({"fmt": "tiff"}, "Unsupported fmt='tiff'"),
        ({"filename": "plot"}, "Cannot infer format"),
        ({"dpi": 0}, "dpi must be positive"),
        ({"width": -1}, "width must be positive"),
        ({"quality": 101}, "quality must be between 1 and 100"),
    ],
)
def test_to_img_rejects_invalid_args(
    tmp_path: Any, img_kwargs: dict[str, Any], match: str
) -> None:
    """to_img raises ValueError for bad formats and render settings."""
    if filename := img_kwargs.get("filename"):
        img_kwargs = {**img_kwargs, "filename": f"{tmp_path}/{filename}"}
    with pytest.raises(ValueError, match=match):
        _make_widget().to_img(**img_kwargs)


@pytest.mark.parametrize("path_suffix", ["output.png", "sub/dir/output.png"])
@pytest.mark.usefixtures("mock_render")
def test_to_img_writes_file(tmp_path: Any, path_suffix: str) -> None:
    """to_img writes bytes to disk, creating parent directories as needed."""
    out_path = f"{tmp_path}/{path_suffix}"
    assert _make_widget().to_img(filename=out_path) == _FAKE_PNG
    with open(out_path, "rb") as fh:
        assert fh.read() == _FAKE_PNG


@pytest.mark.parametrize(
    "img_kwargs",
    [
        {"fmt": "pdf", "dpi": 200},
        {"fmt": "jpeg", "quality": 75},
        {"fmt": "png", "width": 1200, "height": None},
        {"fmt": "png", "width": None, "height": 900},
        {"fmt": "png", "width": 1200, "height": 900},
    ],
    ids=["pdf-dpi", "jpeg-quality", "width", "height", "width-height"],
)
def test_to_img_forwards_render_kwargs(
    img_kwargs: dict[str, Any], mock_render: Mock
) -> None:
    """to_img forwards fmt/dpi/quality/width/height to the renderer, returning its
    bytes unchanged.
    """
    assert _make_widget().to_img(**img_kwargs) == _FAKE_PNG
    for key, value in img_kwargs.items():
        assert mock_render.call_args.kwargs[key] == value


def test_to_img_falls_back_from_marimo_url(mock_render: Mock) -> None:
    """to_img uses class assets when instance _esm is a marimo virtual-file URL."""
    widget = _make_widget()
    widget._esm = "./@file/virtual-esm-url.js"
    widget.to_img(fmt="png")
    esm_passed = mock_render.call_args.kwargs["esm_content"]
    assert esm_passed != "./@file/virtual-esm-url.js"
    assert isinstance(esm_passed, str)


# === headless mount option unit tests ===


_W8 = "width: 800px"
_H6 = "height: 600px"


@pytest.mark.parametrize("transparent", [False, True])
@pytest.mark.parametrize(
    ("style", "width", "height", "expected"),
    [
        (None, None, None, f"{_W8}; {_H6}"),
        ("", None, None, f"{_W8}; {_H6}"),
        ("width: 500px; height: 300px;", None, None, "width: 500px; height: 300px;"),
        ("height: 400px;", None, None, f"height: 400px;; {_W8}"),
        ("width: 400px;", None, None, f"width: 400px;; {_H6}"),
        # whitespace before a declaration and uppercase property names still count
        ("  width: 400px; HEIGHT:300px", None, None, "  width: 400px; HEIGHT:300px"),
        ("min-width: 400px", None, None, f"min-width: 400px; {_W8}; {_H6}"),
        ("max-height: 1px", None, None, f"max-height: 1px; {_W8}; {_H6}"),
        # explicit width/height follow the user style (later declarations win)
        ("width: 500px;", 1200, None, f"width: 500px;; width: 1200px; {_H6}"),
        ("width: 500px;", None, 900, "width: 500px;; height: 900px"),
        ("width: 5px;", 1200, 900, "width: 5px;; width: 1200px; height: 900px"),
    ],
)
def test_container_style_dimensions(
    style: str | None,
    width: int | None,
    height: int | None,
    expected: str,
    transparent: bool,
) -> None:
    """_container_style puts the transparent theme first, then the user style, then
    explicit sizes and 800x600 defaults for dimensions the style lacks.
    """
    data: dict[str, Any] = {"widget_type": "bar_plot"}
    if style is not None:
        data["style"] = style
    css = headless_mod._container_style(data, width, height, transparent=transparent)
    assert css == (f"{_TRANSPARENT_STYLE}; " if transparent else "") + expected


def test_shutdown_browser_tolerates_none() -> None:
    """_shutdown_browser is a no-op when no browser thread was started."""
    with patch.object(headless_mod, "_browser_thread", None):
        headless_mod._shutdown_browser()


def test_get_browser_import_error() -> None:
    """_get_browser raises ImportError with install instructions."""
    state = headless_mod._get_browser_thread()
    blocked = {"playwright": None, "playwright.async_api": None}
    with (
        patch.object(state, "browser", None),
        patch.dict("sys.modules", blocked),
        pytest.raises(ImportError, match="playwright is required"),
    ):
        headless_mod._run_in_browser_thread(headless_mod._get_browser())


@pytest.mark.parametrize(
    ("widget_type", "expect_html_fallback", "expect_worker_shim"),
    [
        ("periodic_table", True, False),
        ("heatmap_matrix", True, False),
        ("scatter_plot", False, False),
        ("structure", False, False),
        ("chem_pot_diagram", False, True),
    ],
)
def test_build_html_render_gates(
    widget_type: str, expect_html_fallback: bool, expect_worker_shim: bool
) -> None:
    """_mount_options toggles HTML fallback and Worker shim per widget type, and
    renders hide the controls without mutating the widget data.
    """
    widget_data = {"widget_type": widget_type, "show_controls": True}
    inputs = headless_mod._RenderInputs(widget_data, "// esm", "/* css */")
    assert inputs.widget_data == {"widget_type": widget_type, "show_controls": False}
    assert widget_data["show_controls"] is True
    options = headless_mod._mount_options(inputs)
    assert options["allow_html"] is expect_html_fallback
    assert options["no_worker"] is expect_worker_shim


def test_render_report_never_raises_on_asset_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """render_report honors its never-raises contract if asset resolution fails."""
    widget = ScatterPlotWidget(series=[{"x": [0, 1], "y": [1, 2]}])

    def _boom() -> tuple[str, str]:
        raise FileNotFoundError("bundle unavailable offline")

    monkeypatch.setattr(widget, "_resolve_assets", _boom)
    report = widget.render_report(timeout=5, dpi=72)  # must not raise
    assert report.ok is False
    assert report.error is not None
    assert "bundle unavailable offline" in report.error
    # data-derived summary/warnings survive a render failure
    assert report.summary["widget_type"] == "scatter_plot"
