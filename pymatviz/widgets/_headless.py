"""Headless widget rendering via Playwright for image export.

Renders MatterViz widgets in headless Chromium with the ESM bundle and CSS the
notebook frontend uses, and captures them as PNG, JPEG, SVG or PDF.

All Playwright calls run on one daemon thread with its own event loop, so sync
callers (scripts, and Jupyter kernels whose loop is already running) just block on a
future: no loop is left running in the caller's thread and no ``nest_asyncio`` is
needed. Each (bundle, CSS, DPI) combination keeps a warm page with the bundle
loaded; exports mount into it and unmount afterwards.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import contextlib
import dataclasses
import functools
import html as html_mod
import io
import json
import math
import os
import re
import tempfile
import threading
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Coroutine, Mapping, Sequence

    from playwright.async_api import Browser, Page, Playwright

try:
    from playwright.async_api import TimeoutError as PlaywrightTimeout

    _TIMEOUT_ERRORS: tuple[type[Exception], ...] = (TimeoutError, PlaywrightTimeout)
except ImportError:  # optional: without playwright, _get_browser raises first
    _TIMEOUT_ERRORS = (TimeoutError,)

_CAPTURE_FORMATS = ("png", "jpeg", "svg", "pdf")
# Transparent exports read on any background: no backdrops, and a mid gray for the
# theme's text (3.9:1 contrast on white, 4.8:1 on GitHub's dark #0d1117)
_TRANSPARENT_GRAY = "#808080"
_TRANSPARENT_STYLE = (
    f"background: transparent; color: {_TRANSPARENT_GRAY}; "
    f"--text-color: {_TRANSPARENT_GRAY}; --text-color-muted: {_TRANSPARENT_GRAY}; "
    f"--plot-legend-item-color: {_TRANSPARENT_GRAY}; "
    f"--text-muted: {_TRANSPARENT_GRAY}; "
    "--page-bg: transparent; --surface-bg: transparent; --struct-bg: transparent; "
    "--plot-legend-bg-color: transparent"
)
_TRIM_PAD_PX = 4  # CSS px of margin kept around trimmed content
_MAX_WARM_PAGES = 4  # idle warm pages kept open at once
# Seconds a mount may run past the render timeout (resize nudge, settle delays), and
# an unmount or shutdown step may take before the page or browser counts as stuck
_STEP_GRACE = 10.0
_PLAYWRIGHT_DEFAULT_TIMEOUT = 30.0  # browser launch, screenshots, PDF


@dataclasses.dataclass
class _BrowserThread:
    """Event loop on a daemon thread that owns Playwright, the browser and its locks."""

    loop: asyncio.AbstractEventLoop
    thread: threading.Thread
    launch_lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    pages_lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    playwright: Playwright | None = None
    browser: Browser | None = None


@dataclasses.dataclass
class _WarmPage:
    """A page with the bundle loaded, reused by later exports with the same key."""

    key: tuple[str, str, float]
    page: Page
    users: int = 0  # the export using it (0 or 1): never evicted while in use


_browser_thread: _BrowserThread | None = None
_browser_thread_lock = threading.Lock()
# Most recently used last, several per key for concurrent exports. Only touched from
# the browser thread.
_warm_pages: list[_WarmPage] = []


def _get_browser_thread() -> _BrowserThread:
    """Return the browser thread, starting it on first use."""
    global _browser_thread  # noqa: PLW0603
    with _browser_thread_lock:
        if _browser_thread is None:
            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, name="pymatviz-headless", daemon=True
            )
            thread.start()
            _browser_thread = _BrowserThread(loop, thread)
        return _browser_thread


def _reset_after_fork() -> None:
    """Forget the parent's browser thread in a forked child: the thread didn't survive
    the fork, so its loop never runs and the child would block on it forever.
    """
    global _browser_thread, _browser_thread_lock
    _browser_thread, _browser_thread_lock = None, threading.Lock()
    _warm_pages.clear()  # the parent's pages (and their asyncio locks) stay its own


if hasattr(os, "register_at_fork"):  # absent on Windows
    os.register_at_fork(after_in_child=_reset_after_fork)


def _call_budget(timeout: float) -> float:
    """Upper bound in seconds on one export once its warm page is free: browser
    launch, page open (goto + bundle ready), mount, capture and unmount.
    """
    return 3 * timeout + 2 * _PLAYWRIGHT_DEFAULT_TIMEOUT + 2 * _STEP_GRACE


def _run_in_browser_thread[T](
    coro: Coroutine[Any, Any, T], timeout: float = 30.0, *, budget: float | None = None
) -> T:
    """Run ``coro`` on the browser thread and block until it finishes.

    Safe from any thread but the browser thread, including one whose event loop is
    running (Jupyter): that loop is merely blocked, never re-entered. Waits at most
    ``_call_budget(timeout)`` seconds; on timeout or interrupt the coroutine is
    cancelled, which unmounts and retires its page.
    """
    state = _get_browser_thread()
    if threading.current_thread() is state.thread:
        coro.close()
        raise RuntimeError(
            "Headless export called from the browser thread would deadlock it; "
            "await the coroutine instead"
        )
    future = asyncio.run_coroutine_threadsafe(coro, state.loop)
    budget = budget or _call_budget(timeout)
    try:
        return future.result(timeout=budget)
    except TimeoutError as exc:
        if future.done():
            raise  # the coroutine itself raised TimeoutError
        future.cancel()
        raise TimeoutError(
            f"Headless export did not finish within {budget}s (render {timeout=}s); "
            f"browser thread alive: {state.thread.is_alive()}"
        ) from exc
    except BaseException:
        future.cancel()
        raise


async def _get_browser() -> Browser:
    """Return the cached headless Chromium browser, launching it if needed."""
    state = _get_browser_thread()
    async with state.launch_lock:
        if state.browser is not None and state.browser.is_connected():
            return state.browser
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "playwright is required for headless widget export.\nInstall it "
                "with:  uv pip install playwright && playwright install chromium"
            ) from None

        if state.playwright is None:
            state.playwright = await async_playwright().start()
        state.browser = await state.playwright.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
        )
        return state.browser


@atexit.register
def _shutdown_browser() -> None:
    """Close the browser and stop its thread (atexit; safe to call twice)."""
    global _browser_thread
    with _browser_thread_lock:
        state, _browser_thread = _browser_thread, None
    if state is None:
        return

    async def close() -> None:
        """Close the browser (and with it the warm pages) and Playwright."""
        _warm_pages.clear()
        with contextlib.suppress(Exception):
            if state.browser is not None:
                await asyncio.wait_for(state.browser.close(), _STEP_GRACE)
            if state.playwright is not None:
                await asyncio.wait_for(state.playwright.stop(), _STEP_GRACE)

    future = asyncio.run_coroutine_threadsafe(close(), state.loop)
    with contextlib.suppress(Exception):
        future.result(timeout=3 * _STEP_GRACE)
    state.loop.call_soon_threadsafe(state.loop.stop)
    state.thread.join(timeout=_STEP_GRACE)
    if not state.thread.is_alive():
        state.loop.close()


@dataclasses.dataclass(frozen=True)
class _RenderInputs:
    """Inputs of one headless render (see render_widget_headless), controls hidden.

    Raises:
        ValueError: For transparent formats other than PNG and SVG.
    """

    widget_data: dict[str, Any]
    esm_content: str
    css_content: str
    fmt: str = "png"
    dpi: int = 150
    timeout: float = 30.0
    width: int | None = None
    height: int | None = None
    quality: int = 90
    transparent: bool = False

    def __post_init__(self) -> None:
        """Check the transparent format and hide the widget's controls."""
        if self.transparent and (fmt := self.fmt) not in ("png", "svg"):
            raise ValueError(f"transparent export supports png and svg, got {fmt=}")
        hidden_controls = {**self.widget_data, "show_controls": False}
        object.__setattr__(self, "widget_data", hidden_controls)

    @property
    def scale_factor(self) -> float:
        """Device scale factor for the DPI (72 DPI = 1x, never below 1x). Applies to
        all formats since SVG and PDF can embed raster canvas layers.
        """
        return max(1, self.dpi / 72)


async def _capture_page(page: Page, inputs: _RenderInputs) -> bytes:
    """Capture the mounted widget as ``inputs.fmt`` (png, jpeg, svg or pdf) bytes.

    PDFs of widgets showing WebGL are rasterized (Chromium's print pipeline can't
    see WebGL), others are vector with selectable text. That is decided from the
    rendered DOM since some widgets render SVG or canvas depending on their data.
    Transparent PNGs and SVGs are trimmed to their painted content.
    """
    fmt, scale_factor = inputs.fmt, inputs.scale_factor
    if fmt == "svg":
        result = await page.evaluate(
            _SVG_EXPORT_JS,
            [scale_factor * 72, _TRIM_PAD_PX if inputs.transparent else 0],
        )
        if "error" in result:
            raise RuntimeError(f"SVG export failed: {result['error']}")
        return result["svg"].encode("utf-8")
    root = page.locator("#widget-root")
    if fmt == "jpeg":
        return await root.screenshot(type="jpeg", quality=inputs.quality)
    if fmt != "pdf":
        png = await root.screenshot(type="png", omit_background=inputs.transparent)
        if inputs.transparent:
            return _trim_png(png, round(_TRIM_PAD_PX * scale_factor))
        return png
    if (await page.evaluate(_PROBE_JS))["webgl"]:
        return _png_to_pdf(await root.screenshot(type="png"), scale=scale_factor)
    if (bbox := await root.bounding_box()) is None:
        raise RuntimeError("Widget root has no bounding box for PDF export")
    await page.emulate_media(media="screen")
    return await page.pdf(
        width=f"{bbox['width']}px",
        height=f"{bbox['height']}px",
        print_background=True,
        margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
    )


async def _open_warm_page(
    esm_content: str, css_content: str, scale_factor: float, timeout_ms: int
) -> Page:
    """Open a page with the bundle imported and ``window.__matterviz_mount`` ready."""
    page = await (await _get_browser()).new_page(
        viewport={"width": 1024, "height": 768}, device_scale_factor=scale_factor
    )
    try:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", encoding="utf-8", delete_on_close=False
        ) as file:
            file.write(_build_html(esm_content, css_content))
            file.close()
            await page.goto(
                Path(file.name).as_uri(),
                wait_until="domcontentloaded",
                timeout=timeout_ms,
            )
            await page.wait_for_function(
                "() => window.__matterviz_ready !== undefined", timeout=timeout_ms
            )
        if (ready := await page.evaluate("() => window.__matterviz_ready")) is not True:
            raise RuntimeError(f"Widget render failed: {ready}")  # noqa: TRY301
    except BaseException:
        with contextlib.suppress(Exception):
            await page.close()
        raise
    return page


async def _retire(warm: _WarmPage) -> None:
    """Drop a warm page from the cache and close it."""
    if warm in _warm_pages:
        _warm_pages.remove(warm)
    with contextlib.suppress(Exception):
        await warm.page.close()


@contextlib.asynccontextmanager
async def _mounted_widget(
    inputs: _RenderInputs,
) -> AsyncGenerator[tuple[Page, str | None]]:
    """Yield ``(page, render_error)`` with the widget mounted on a warm page.

    Each export checks out an idle page with its key or opens one, so concurrent
    exports render side by side. Idle pages beyond ``_MAX_WARM_PAGES`` close, least
    recently used first. A page whose mount errored, timed out or failed to unmount
    is retired; errors raised by the caller's capture leave it in service.
    """
    key = (inputs.esm_content, inputs.css_content, inputs.scale_factor)
    async with _get_browser_thread().pages_lock:
        warm = next(
            (
                old
                for old in reversed(_warm_pages)
                if old.key == key and not old.users and not old.page.is_closed()
            ),
            None,
        )
        if warm is None:
            page = await _open_warm_page(*key, int(inputs.timeout * 1000))
            warm = _WarmPage(key, page)
        else:
            _warm_pages.remove(warm)  # re-appended as most recently used
        _warm_pages.append(warm)
        warm.users += 1  # no await until the try below releases it in any case
    healthy = False
    try:
        async with _get_browser_thread().pages_lock:
            await _evict_idle()
        mount_args = [
            _widget_data_json(_filling_root(inputs.widget_data)),
            _mount_options(inputs),
        ]
        render_error = await asyncio.wait_for(
            warm.page.evaluate(_MOUNT_JS, mount_args), inputs.timeout + _STEP_GRACE
        )
        try:
            yield warm.page, render_error
        finally:
            await asyncio.wait_for(
                warm.page.evaluate("() => window.__matterviz_unmount()"), _STEP_GRACE
            )
            healthy = render_error is None
    finally:
        # release and retire together, so no checkout picks up a page being retired
        async with _get_browser_thread().pages_lock:
            warm.users -= 1
            if not healthy:
                await _retire(warm)
            # overlapping exports may each have opened a page: trim them once idle
            await _evict_idle()


async def _evict_idle() -> None:
    """Close the least recently used idle pages beyond _MAX_WARM_PAGES (call with
    pages_lock held).
    """
    idle = [old for old in _warm_pages if old.users == 0]
    for old in idle[: max(0, len(_warm_pages) - _MAX_WARM_PAGES)]:
        await _retire(old)


async def _render_widget(inputs: _RenderInputs) -> bytes:
    """Mount the widget on a warm page and capture it (on the browser thread)."""
    try:
        async with _mounted_widget(inputs) as (page, render_error):
            if render_error:
                raise RuntimeError(f"Widget render failed: {render_error}")
            return await _capture_page(page, inputs)
    except _TIMEOUT_ERRORS as exc:
        raise TimeoutError(
            f"Widget did not finish rendering within {inputs.timeout}s"
        ) from exc


def _widget_data_json(widget_data: Mapping[str, Any]) -> str:
    """Widget data as JSON safe to embed in a ``<script>`` (``</`` escaped), with
    binary arrays base64-encoded for ``_REVIVE_BUFFERS_JS`` (see ``json_default``).

    Raises:
        TypeError: For any other non-JSON value, instead of stringifying it.
    """
    from pymatviz.widgets._binary import json_default

    return json.dumps(widget_data, default=json_default).replace("</", r"<\/")


@functools.lru_cache(maxsize=_MAX_WARM_PAGES)  # base64 of a multi-MB bundle is slow
def _esm_blob_loader_js(esm_content: str) -> str:
    """JS defining ``esm_url``, a ``blob:`` URL of the base64-inlined bundle."""
    esm_b64 = base64.b64encode(esm_content.encode("utf-8")).decode("ascii")
    return (
        f'const esm_bytes = Uint8Array.from(atob("{esm_b64}"), c => c.charCodeAt(0));\n'
        'const esm_blob = new Blob([esm_bytes], { type: "application/javascript" });\n'
        "const esm_url = URL.createObjectURL(esm_blob);"
    )


# Binary buffers arrive base64-encoded; decode each synchronously (no fetch of a data:
# URL, which a strict CSP blocks) to the DataView a live comm gives
_REVIVE_BUFFERS_JS = """\
// unset traits arrive as null, which destructuring defaults would let through
const { trajectory, volumetric_data, series } = widget_data;
for (const envelope of [trajectory, ...(volumetric_data ?? []), ...(series ?? [])]) {
  if (envelope?.format !== "matterviz-binary-v1") continue;
  for (const spec of Object.values(envelope.arrays)) {
    const raw = atob(spec.data);
    const bytes = new Uint8Array(raw.length);
    for (let idx = 0; idx < raw.length; idx++) bytes[idx] = raw.charCodeAt(idx);
    spec.data = new DataView(bytes.buffer);
  }
}
"""
# Imports the bundle from ``esm_url`` and defines async ``mount(widget_data, el,
# no_worker)``, which renders into ``el`` through an anywidget model mock (render
# only needs get()) and resolves to render's cleanup function. ``no_worker`` hides
# Worker from ChemPotDiagram: it resolves its worker URL against ``import.meta.url``,
# which fails for an embedded (blob/CDN) bundle and leaves a loading spinner, while
# without Worker it computes on the main thread.
_MOUNT_FN_JS = (
    """\
const mod = await import(esm_url);
const render_fn = mod.default?.render ?? mod.render;
if (!render_fn) throw new Error("ESM bundle has no render export");
const native_worker = globalThis.Worker;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const mount = async (widget_data, el, no_worker) => {
"""
    + _REVIVE_BUFFERS_JS
    + """\
globalThis.Worker = no_worker ? undefined : native_worker;
const model = {
  get: (key) => widget_data[key],
  set: () => {}, on: () => {}, off: () => {}, save_changes: () => {}, send: () => {},
};
// set first: after a throwing render, export reports "no rendered widget"
window.__matterviz_export_svg = (dpi) => mod.export_svg(el, dpi);
const cleanup = await render_fn({ model, el });
// Svelte's bind:clientWidth/Height can miss a size the element already has at mount:
// briefly widening the wrapper makes its ResizeObserver fire with the real size
const wrapper = el.firstElementChild;
if (wrapper) {
  const width = wrapper.style.width;
  wrapper.style.width = `${wrapper.clientWidth + 1}px`;
  await sleep(50);
  wrapper.style.width = width;
  await sleep(50);
}
return cleanup;
};
"""
)
# Capture hooks of the warm page, run after _MOUNT_FN_JS. __matterviz_mount resolves
# once the widget painted capturable content, to an error message or null on success
# (returned, not thrown, so Playwright needn't translate them).
_CAPTURE_HOOKS_JS = """\
// WebGL contexts the widget created, to probe and to free on unmount: calling
// getContext() on canvases instead would create contexts on bare ones
const gl_contexts = new Set();
const native_get_context = HTMLCanvasElement.prototype.getContext;
HTMLCanvasElement.prototype.getContext = function (kind, ...rest) {
  const ctx = native_get_context.call(this, kind, ...rest);
  if (ctx && String(kind).includes("webgl")) gl_contexts.add(ctx);
  return ctx;
};
// Uncaught errors/rejections of the mounted widget (e.g. a crashing $effect) fail
// the render. ResizeObserver loop notices are benign browser warnings.
let uncaught = [];
const on_uncaught = (err) => {
  const msg = String(err);
  if (!msg.startsWith("ResizeObserver loop")) uncaught.push(`Uncaught ${msg}`);
};
window.addEventListener("error", (event) => on_uncaught(event.error ?? event.message));
window.addEventListener("unhandledrejection", (event) => on_uncaught(event.reason));
let cleanup = null;
window.__matterviz_unmount = () => {
  const unmount = cleanup;
  cleanup = null;
  unmount?.();
  // free WebGL contexts now, not at GC: Chromium caps live contexts per page
  for (const ctx of gl_contexts) ctx.getExtension("WEBGL_lose_context")?.loseContext();
  gl_contexts.clear();
};
// Content kind, WebGL use and overflow of the root. Canvases under 50 px and SVGs
// under 20 px are decorations, as in MatterViz export_svg's MIN_FIGURE_*_PX.
window.__matterviz_probe = () => {
  const root = document.getElementById("widget-root");
  const min_size = (node, px) => {
    const { width, height } = node.getBoundingClientRect();
    return width >= px && height >= px;
  };
  const canvases = [...root.querySelectorAll("canvas")].filter((c) => min_size(c, 50));
  const has_svg = [...root.querySelectorAll("svg")].some((svg) => min_size(svg, 20));
  const has_text = root.textContent.trim().length > 0;
  const rect = root.getBoundingClientRect();
  const [root_w, root_h] = [Math.round(rect.width), Math.round(rect.height)];
  return {
    content_type:
      canvases.length ? "canvas" : has_svg ? "svg" : has_text ? "html" : "empty",
    webgl: [...gl_contexts].some((ctx) => canvases.includes(ctx.canvas)),
    root_w, root_h,
    clipped: (root_w > 0 && root.scrollWidth > root_w + 4)
      || (root_h > 0 && root.scrollHeight > root_h + 4),
  };
};
window.__matterviz_mount = async (data_json, options) => {
  const { style, allow_html, no_worker, max_wait } = options;
  window.__matterviz_unmount();
  uncaught = [];
  const uncaught_error = () => (uncaught.length ? uncaught.join("; ") : null);
  const el = document.createElement("div");
  el.id = "widget-root";
  el.style.cssText = style;
  document.getElementById("widget-root").replaceWith(el);
  const two_frames = () =>
    new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)));
  try {
    cleanup = await mount(JSON.parse(data_json), el, no_worker);
    // poll for painted content, skipping loading spinners (e.g. async compute) so a
    // transient "Computing..." state is never captured
    for (const start = Date.now(); Date.now() - start < max_wait; await sleep(100)) {
      await two_frames();
      if (uncaught.length) return uncaught_error();
      if (el.querySelector('.spinner[role="status"]')) continue;
      if (el.querySelector("canvas, svg")) {
        await sleep(500); // let Three.js scenes finish their first paint
        return uncaught_error();
      }
      // HTML-only widgets render neither: accept laid-out, non-empty content
      const child = el.firstElementChild;
      if (allow_html && child?.getBoundingClientRect().height > 20
          && el.textContent.trim()) {
        await sleep(200);
        return uncaught_error();
      }
    }
    return `No capturable content (canvas, SVG, or HTML) after ${max_wait}ms `
      + `(widget may still be loading)`;
  } catch (err) {
    return String(err);
  }
};
"""
_MOUNT_JS = "([data_json, options]) => window.__matterviz_mount(data_json, options)"
_PROBE_JS = "() => window.__matterviz_probe()"
# MatterViz's own vector export (the bundle's export_svg, behind the widget's Export
# pane), embedding canvas layers at the export DPI. It throws for figures without a
# vector form; its reason comes back as data so Playwright needn't translate it.
# With a trim pad, the viewBox shrinks to the painted pixels (the SVG drawn on a canvas
# at 1 px per unit; getBBox() would count clipped-away geometry) plus that pad.
_SVG_EXPORT_JS = """\
async ([dpi, pad]) => {
    let svg;
    try { svg = await window.__matterviz_export_svg(dpi); }
    catch (error) { return { error: String(error?.message ?? error) }; }
    if (!pad) return { svg };
    try { return { svg: await trim_svg() }; }
    catch (error) { return { error: `trim failed: ${error?.message ?? error}` }; }

    async function trim_svg() {  // the svg and pad above
        const doc = new DOMParser().parseFromString(svg, "image/svg+xml");
        const node = doc.documentElement;
        const view_box = node.getAttribute("viewBox");
        if (!view_box) throw new Error("exported SVG has no viewBox");
        const [vb_x, vb_y, vb_w, vb_h] = view_box.split(/[\\s,]+/).map(Number);
        const [width, height] = [Math.ceil(vb_w), Math.ceil(vb_h)];
        node.setAttribute("width", width);
        node.setAttribute("height", height);
        const source = new XMLSerializer().serializeToString(node);
        const url = URL.createObjectURL(new Blob([source], { type: "image/svg+xml" }));
        const img = new Image();
        img.src = url;
        try { await img.decode(); } finally { URL.revokeObjectURL(url); }
        const ctx = Object.assign(document.createElement("canvas"), { width, height })
            .getContext("2d", { willReadFrequently: true });
        ctx.drawImage(img, 0, 0, width, height);
        const alpha = ctx.getImageData(0, 0, width, height).data;
        let [x0, y0, x1, y1] = [width, height, -1, -1];
        for (let idx = 3; idx < alpha.length; idx += 4) {
            if (!alpha[idx]) continue;
            const px = ((idx - 3) / 4) % width, py = Math.floor((idx - 3) / 4 / width);
            x0 = Math.min(x0, px); x1 = Math.max(x1, px);
            y0 = Math.min(y0, py); y1 = Math.max(y1, py);
        }
        if (x1 < 0) throw new Error("transparent export painted no pixels");
        const [sx, sy] = [vb_w / width, vb_h / height];
        const box = [
            vb_x + x0 * sx - pad, vb_y + y0 * sy - pad,
            (x1 + 1 - x0) * sx + 2 * pad, (y1 + 1 - y0) * sy + 2 * pad,
        ].map((num) => +num.toFixed(2));
        node.setAttribute("viewBox", box.join(" "));
        node.setAttribute("width", box[2]);
        node.setAttribute("height", box[3]);
        return new XMLSerializer().serializeToString(node);
    }
}
"""


def _container_style(
    widget_data: Mapping[str, Any],
    width: int | None,
    height: int | None,
    *,
    transparent: bool = False,
) -> str:
    """Inline style of the widget root (later declarations win): the transparent
    theme if requested, the widget's style, explicit overrides, then 800x600 defaults
    for any missing dimension.
    """
    user_style = widget_data.get("style") or ""
    parts = [_TRANSPARENT_STYLE] if transparent else []
    parts += [user_style] if user_style else []
    for name, value, default in (("width", width, 800), ("height", height, 600)):
        if value is not None:
            parts.append(f"{name}: {value}px")
        elif not re.search(rf"(?:^|;)\s*{name}\s*:", user_style, re.IGNORECASE):
            parts.append(f"{name}: {default}px")
    return "; ".join(parts)


def _filling_root(widget_data: Mapping[str, Any]) -> dict[str, Any]:
    """Widget data whose ``style`` (which MatterViz applies to the widget's own
    wrapper) fills the export root: the root carries the export size, and a wrapper
    without one shrinks charts to their 300 px minimum height.
    """
    user_style = widget_data.get("style") or ""
    return {**widget_data, "style": f"{user_style}; width: 100%; height: 100%"}


def _mount_options(inputs: _RenderInputs) -> dict[str, Any]:
    """Options of ``window.__matterviz_mount``. Only HTML-only widgets (tile grids, no
    <svg>/<canvas> to wait for) accept HTML content as painted, so chart widgets
    aren't captured half laid out via stray labels or legends.
    """
    widget_type = inputs.widget_data.get("widget_type")
    return {
        "style": _container_style(
            inputs.widget_data,
            inputs.width,
            inputs.height,
            transparent=inputs.transparent,
        ),
        "allow_html": widget_type in ("periodic_table", "heatmap_matrix"),
        "no_worker": widget_type == "chem_pot_diagram",
        "max_wait": int(inputs.timeout * 1000),
    }


def _html_page(css_content: str, script: str, head: str = "", style: str = "") -> str:
    """HTML page with the CSS, an empty ``#widget-root`` and a module script."""
    return f"""\
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">{head}
<style>{css_content}</style>
</head><body style="margin: 0; padding: 0; font-family: system-ui, sans-serif;">
<div id="widget-root" style="{style}"></div>
<script type="module">
{script}
</script>
</body></html>"""


def _build_html(esm_content: str, css_content: str) -> str:
    """Warm-page HTML loading the bundle and defining the capture hooks.
    ``window.__matterviz_ready`` is true once loaded, else the load error.
    """
    esm_loader = _esm_blob_loader_js(esm_content)
    on_error = "} catch (err) {\n  window.__matterviz_ready = String(err);\n}"
    ready = f"{_CAPTURE_HOOKS_JS}window.__matterviz_ready = true;\n"
    script = f"{esm_loader}\ntry {{\n{_MOUNT_FN_JS}{ready}{on_error}"
    # the bundle caps a mounted widget at its parent's width (inline max-width: 100%),
    # here the 1024px page: exports keep their requested width
    head = "<style>#widget-root { max-width: none !important }</style>"
    return _html_page(css_content, script, head)


def build_interactive_html(
    widget_data: dict[str, Any],
    *,
    css_content: str | None = None,
    css_url: str | None = None,
    esm_url: str | None = None,
    esm_content: str | None = None,
    title: str = "MatterViz widget",
    description: str = "",
) -> str:
    """Build a standalone, interactive HTML page rendering one widget with controls.

    Exactly one ESM source must be given: ``esm_url`` (imported by URL, so the host
    must serve it with CORS and a JS MIME type) or ``esm_content`` (inlined as a
    base64 ``blob:`` URL: self-contained, opens offline). Likewise exactly one of
    ``css_url`` (linked) or ``css_content`` (inlined).
    Client-side interactions work; controls that round-trip through a Python kernel
    don't (the anywidget model is a no-op mock).
    """
    if (esm_url is None) == (esm_content is None):
        raise ValueError("provide exactly one of esm_url or esm_content")
    if (css_url is None) == (css_content is None):
        raise ValueError("provide exactly one of css_url or css_content")
    esm_loader = (
        f"const esm_url = {json.dumps(esm_url)};"
        if esm_content is None
        else _esm_blob_loader_js(esm_content)
    )
    no_worker = json.dumps(widget_data.get("widget_type") == "chem_pot_diagram")
    head = (
        '\n<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"\n<title>{html_mod.escape(title)}</title>"
        f'\n<meta name="description" content="{html_mod.escape(description)}">'
    )
    if css_url is not None:
        head += f'\n<link rel="stylesheet" href="{html_mod.escape(css_url)}">'
    get_root = 'const el = document.getElementById("widget-root");'
    mount_call = f"mount({_widget_data_json(widget_data)}, el, {no_worker});"
    script = f"{esm_loader}\n{_MOUNT_FN_JS}{get_root}\n{mount_call}"
    style = html_mod.escape(widget_data.get("style") or "width: 100%; height: 600px;")
    return _html_page(css_content or "", script, head, style)


def _trim_png(png_bytes: bytes, pad: int) -> bytes:
    """Crop a transparent PNG to its non-transparent pixels plus ``pad`` pixels."""
    from PIL import Image

    # Chromium drops the alpha channel of fully painted screenshots
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    if (bbox := img.getchannel("A").getbbox()) is None:
        raise RuntimeError("Transparent export captured no painted pixels")
    left, top, right, bottom = bbox
    box = (max(0, left - pad), max(0, top - pad))
    box += (min(img.width, right + pad), min(img.height, bottom + pad))
    out = io.BytesIO()
    img.crop(box).save(out, format="PNG", optimize=True)
    return out.getvalue()


def _png_to_pdf(png_bytes: bytes, scale: float = 1.0) -> bytes:
    """Wrap a PNG in a single-page PDF sized ``pixels / scale`` points, where
    ``scale`` is the screenshot's device scale factor. Requires Pillow.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for PDF export of WebGL (canvas) widgets.\n"
            "Install it with:  uv pip install Pillow"
        ) from None

    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    img = Image.alpha_composite(Image.new("RGBA", img.size, "white"), img)
    px_w, px_h = img.size
    page_w, page_h = f"{px_w / scale:.2f}", f"{px_h / scale:.2f}"
    pixels = zlib.compress(img.convert("RGB").tobytes(), 6)
    draw = zlib.compress(f"q {page_w} 0 0 {page_h} 0 0 cm /Im0 Do Q".encode())
    page = (
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w} {page_h}] /Contents "
        "5 0 R /Resources << /XObject << /Im0 4 0 R >> >> >>"
    )
    image_header = (
        f"<< /Type /XObject /Subtype /Image /Width {px_w} /Height {px_h} /ColorSpace "
        "/DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length "
        f"{len(pixels)} >>\nstream\n"
    )
    draw_header = f"<< /Length {len(draw)} /Filter /FlateDecode >>\nstream\n"

    image = image_header.encode() + pixels + b"\nendstream"
    contents = draw_header.encode() + draw + b"\nendstream"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        page.encode(),
        image,
        contents,
    ]
    pdf, offsets = bytearray(b"%PDF-1.4\n"), []
    for obj_num, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{obj_num} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref_pos, n_entries = len(pdf), len(objects) + 1
    entries = "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    pdf += (
        f"xref\n0 {n_entries}\n0000000000 65535 f \n{entries}trailer\n<< /Size "
        f"{n_entries} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(pdf)


def render_widget_headless(
    widget_data: dict[str, Any],
    esm_content: str,
    css_content: str,
    fmt: str = "png",
    dpi: int = 150,
    timeout: float = 30.0,
    quality: int = 90,
    width: int | None = None,
    height: int | None = None,
    *,
    transparent: bool = False,
) -> bytes:
    """Render a widget in headless Chromium and capture it.

    Mounts the widget through a mock anywidget model on a warm page with the bundle
    loaded, waits for it to paint and captures it. Blocks the calling thread, which
    may run an event loop (Jupyter): the browser lives on its own thread.

    Args:
        widget_data: Traitlet values (from ``widget.to_dict()``).
        esm_content: ESM bundle text (``matterviz.js``).
        css_content: CSS text (``matterviz.css``).
        fmt: ``"png"``, ``"jpeg"``, ``"svg"`` or ``"pdf"``.
        dpi: Resolution as browser ``device_scale_factor`` (72 DPI = 1x, 144 = 2x).
            Applies to PNG/JPEG and canvas pixels in SVG/PDF, not to vector content.
        timeout: Max seconds to wait for the widget to render.
        quality: JPEG quality (1-100), ignored for other formats.
        width: Override container width in pixels.
        height: Override container height in pixels.
        transparent: Transparent background, gray theme text, trimmed to the
            painted content (PNG and SVG only).

    Returns:
        PNG, JPEG, SVG or PDF bytes.

    Raises:
        RuntimeError: If the widget fails to render or has no vector form for SVG.
        TimeoutError: If the widget does not finish rendering in time.
    """
    inputs = _RenderInputs(
        widget_data, esm_content, css_content, fmt, dpi, timeout, width, height,
        quality, transparent,
    )  # fmt: skip
    return _run_in_browser_thread(_render_widget(inputs), timeout)


def render_widgets_headless(jobs: Sequence[Mapping[str, Any]]) -> list[bytes]:
    """render_widget_headless for several widgets at once, each job a dict of its
    arguments. Up to ``_MAX_WARM_PAGES`` render concurrently on separate pages.

    Returns:
        The image bytes in job order.
    """
    inputs = [_RenderInputs(**job) for job in jobs]
    slots = math.ceil(len(inputs) / _MAX_WARM_PAGES)

    async def render_all() -> list[bytes]:
        """Render every job, at most _MAX_WARM_PAGES at a time."""
        limit = asyncio.Semaphore(_MAX_WARM_PAGES)

        async def render(one: _RenderInputs) -> bytes:
            """Render one job once a slot frees up."""
            async with limit:
                return await _render_widget(one)

        tasks = [asyncio.ensure_future(render(one)) for one in inputs]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:  # stop sibling renders before reporting the failure
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    # each wave of up to _MAX_WARM_PAGES renders gets a full single-export budget
    timeout = max((one.timeout for one in inputs), default=30.0)
    budget = _call_budget(timeout) * max(slots, 1)
    return _run_in_browser_thread(render_all(), timeout, budget=budget)


# === Render diagnostics (render_report) ===


@dataclasses.dataclass(frozen=True)
class RenderReport:
    """Result of a headless render health check (see ``render_report``).

    Attributes:
        ok: True if the widget rendered without error or timeout.
        summary: ``describe_widget`` facts about the widget's data.
        content_type: ``"canvas"``, ``"svg"``, ``"html"``, ``"empty"`` or None.
        width/height: Rendered widget-root size in CSS pixels.
        blank_fraction: Most-common-color pixel fraction (1.0 == uniform), or None
            without Pillow.
        is_blank: True if ``blank_fraction`` exceeds 0.99.
        clipped: Heuristic: content overflows the widget root (may false-positive
            on intentional scroll areas).
        warnings: Cheap input warnings (empty/degenerate data).
        error: Render error or timeout message, or None.
    """

    ok: bool
    summary: dict[str, Any] = dataclasses.field(default_factory=dict)
    content_type: str | None = None
    width: int | None = None
    height: int | None = None
    blank_fraction: float | None = None
    is_blank: bool = False
    clipped: bool = False
    warnings: tuple[str, ...] = ()
    error: str | None = None


def png_blank_fraction(png_bytes: bytes) -> float | None:
    """Return the most-common-color pixel fraction (1.0 == blank), or None without
    Pillow.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    import numpy as np

    pixels = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB")).reshape(-1, 3)
    # one int per pixel: far faster to unique than np.unique(pixels, axis=0)
    packed = (pixels.astype(np.uint32) << np.array([16, 8, 0], np.uint32)).sum(axis=1)
    _, counts = np.unique(packed, return_counts=True)
    return float(counts.max() / len(packed))


async def _diagnose(inputs: _RenderInputs) -> RenderReport:
    """Render once and return visual + DOM diagnostics; never raises."""
    try:
        async with _mounted_widget(inputs) as (page, render_error):
            probe = await page.evaluate(_PROBE_JS)
            png_bytes = await page.locator("#widget-root").screenshot(type="png")
        blank_fraction = png_blank_fraction(png_bytes)
        return RenderReport(
            ok=render_error is None,
            error=render_error,
            content_type=probe["content_type"],
            width=probe["root_w"],
            height=probe["root_h"],
            blank_fraction=blank_fraction,
            is_blank=blank_fraction is not None and blank_fraction > 0.99,
            clipped=probe["clipped"],
        )
    except Exception as exc:  # noqa: BLE001
        return RenderReport(ok=False, error=f"render failed: {exc}")


def render_diagnostics(
    widget_data: dict[str, Any],
    esm_content: str,
    css_content: str,
    *,
    timeout: float = 30.0,
    dpi: int = 72,
) -> RenderReport:
    """Render a widget headlessly once and return visual + DOM diagnostics.

    Render failures return ``ok=False`` with ``error`` set (``summary``/``warnings``
    are left for the caller), but the browser-thread call itself can raise
    ``TimeoutError`` past its budget or ``RuntimeError`` from the browser thread.
    """
    inputs = _RenderInputs(
        widget_data, esm_content, css_content, dpi=dpi, timeout=timeout
    )
    return _run_in_browser_thread(_diagnose(inputs), timeout)
