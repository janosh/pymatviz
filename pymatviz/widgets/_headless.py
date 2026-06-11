"""Headless widget rendering via Playwright for image export.

Renders MatterViz widgets in a headless Chromium browser using the same
ESM bundle and CSS that the notebook frontend uses, then captures the
output as PNG, JPEG, SVG, or PDF. Works without any notebook/IDE frontend.
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
import os
import re
import tempfile
import threading
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Mapping

    from playwright.async_api import Browser as AsyncBrowser
    from playwright.async_api import Page as AsyncPage
    from playwright.sync_api import Browser, Page

# Module-level browser cache -- reused across to_img() calls to amortize
# the ~2s Chromium startup cost.
_pw: Any = None
_browser: Browser | None = None
_atexit_registered = False
_browser_lock = threading.Lock()

# Async browser cache for event-loop contexts (e.g. jupyter nbconvert --execute)
_async_pw: Any = None
_async_browser: AsyncBrowser | None = None
_async_atexit_registered = False
_async_browser_lock: asyncio.Lock | None = None


def _get_browser() -> Browser:
    """Return a cached headless Chromium browser, launching one if needed.

    Thread-safe: concurrent callers block on a lock during launch.
    """
    global _pw, _browser, _atexit_registered  # noqa: PLW0603
    if _browser is not None and _browser.is_connected():
        return _browser

    with _browser_lock:
        # Double-check after acquiring the lock
        if _browser is not None and _browser.is_connected():
            return _browser

        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "playwright is required for headless widget export.\n"
                "Install it with:  uv pip install playwright"
                " && playwright install chromium"
            ) from None

        pw = sync_playwright().start()
        try:
            launch_args = ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"]
            browser = pw.chromium.launch(headless=True, args=launch_args)
        except Exception:
            try:
                pw.stop()
            except (OSError, RuntimeError):
                pass
            raise
        _pw = pw
        _browser = browser
        if not _atexit_registered:
            atexit.register(_shutdown_browser)
            _atexit_registered = True
        return _browser


def _shutdown_browser() -> None:
    """Clean up the browser and Playwright on interpreter exit."""
    global _pw, _browser  # noqa: PLW0603
    try:
        if _browser is not None:
            _browser.close()
    except (OSError, RuntimeError):
        pass
    _browser = None
    try:
        if _pw is not None:
            _pw.stop()
    except (OSError, RuntimeError):
        pass
    _pw = None


async def _get_async_browser() -> AsyncBrowser:
    """Return a cached async headless Chromium browser, launching one if needed.

    Uses an asyncio.Lock so only one coroutine performs browser startup.
    """
    global _async_pw, _async_browser, _async_atexit_registered, _async_browser_lock  # noqa: PLW0603
    if _async_browser is not None and _async_browser.is_connected():
        return _async_browser

    if _async_browser_lock is None:
        _async_browser_lock = asyncio.Lock()

    async with _async_browser_lock:
        if _async_browser is not None and _async_browser.is_connected():
            return _async_browser

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "playwright is required for headless widget export.\n"
                "Install it with:  uv pip install playwright"
                " && playwright install chromium"
            ) from None

        pw = await async_playwright().start()
        try:
            launch_args = [
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
            ]
            browser = await pw.chromium.launch(headless=True, args=launch_args)
        except Exception:
            try:
                await pw.stop()
            except (OSError, RuntimeError):
                pass
            raise
        _async_pw = pw
        _async_browser = browser
        if not _async_atexit_registered:
            atexit.register(_shutdown_async_browser)
            _async_atexit_registered = True
        return _async_browser


def _shutdown_async_browser() -> None:
    """Clean up the async browser references on interpreter exit.

    The Chromium subprocess is killed automatically when the Python process
    exits, so we only need to clear references — no need to poke at Playwright
    private APIs for a synchronous close.
    """
    global _async_pw, _async_browser  # noqa: PLW0603
    _async_browser = None
    _async_pw = None


@dataclasses.dataclass(frozen=True)
class _RenderInputs:
    """Inputs threaded through the headless render pipeline (one bundle, not 8 args)."""

    widget_data: dict[str, Any]
    esm_content: str
    css_content: str
    fmt: str = "png"
    dpi: int = 150
    timeout: float = 30.0
    width: int | None = None
    height: int | None = None


def _prepare_render(inputs: _RenderInputs) -> tuple[str, int, float]:
    """Build HTML, compute scale factor, write temp file.

    Returns (tmp_path, timeout_ms, scale_factor).
    """
    html = _build_html(
        inputs.widget_data,
        inputs.esm_content,
        inputs.css_content,
        inputs.timeout,
        width=inputs.width,
        height=inputs.height,
    )
    timeout_ms = int(inputs.timeout * 1000)
    # PDF capture rasterizes WebGL canvases (Chromium's vector page.pdf() can't
    # see them), so scale every raster-capable format by DPI. The canvas-vs-SVG
    # choice is made at capture time from the DOM. SVG stays 1x (pure vector).
    needs_raster = inputs.fmt in ("png", "jpeg", "pdf")
    scale_factor = max(1, inputs.dpi / 72) if needs_raster else 1
    return _write_temp_html(html), timeout_ms, scale_factor


_CAPTURE_FORMATS = ("png", "jpeg", "svg", "pdf")


def _validate_capture_format(fmt: str) -> None:
    """Raise ValueError if fmt is not a supported capture format."""
    if fmt not in _CAPTURE_FORMATS:
        raise ValueError(
            f"Unsupported capture format {fmt!r}, expected one of {_CAPTURE_FORMATS}"
        )


def _vector_pdf_kwargs(bbox: Mapping[str, object] | None) -> dict[str, Any]:
    """Page.pdf() options sizing a borderless PDF to the widget bounding box."""
    if bbox is None:
        raise RuntimeError("Widget root has no bounding box for PDF export")
    return {
        "width": f"{bbox['width']}px",
        "height": f"{bbox['height']}px",
        "print_background": True,
        "margin": {"top": "0", "bottom": "0", "left": "0", "right": "0"},
    }


async def _async_capture_page(
    page: AsyncPage,
    fmt: str,
    quality: int = 90,
    scale_factor: float = 1.0,
) -> bytes:
    """Async version of _capture_page."""
    _validate_capture_format(fmt)

    if fmt == "svg":
        svg_string = await page.evaluate(_SVG_EXTRACT_JS)
        if svg_string is None:
            has_canvas = await page.evaluate(_HAS_CANVAS_JS)
            raise RuntimeError(_no_svg_message(has_canvas=has_canvas))
        return svg_string.encode("utf-8")

    if fmt == "pdf":
        # WebGL canvas content is invisible to Chromium's print PDF pipeline,
        # so rasterize canvas widgets; SVG/HTML widgets get a native vector PDF.
        if await page.evaluate(_HAS_CANVAS_JS):
            return _png_to_pdf(
                await page.locator("#widget-root").screenshot(type="png"),
                scale=scale_factor,
            )
        await page.emulate_media(media="screen")
        bbox = await page.locator("#widget-root").bounding_box()
        return await page.pdf(**_vector_pdf_kwargs(bbox))

    root = page.locator("#widget-root")
    if fmt == "jpeg":
        return await root.screenshot(type="jpeg", quality=quality)
    return await root.screenshot(type="png")


def _raise_if_timeout(exc: BaseException, timeout: float) -> None:
    """Re-raise Playwright/builtin timeouts as TimeoutError with a clear message."""
    timeout_types: tuple[type[BaseException], ...] = (TimeoutError,)
    try:  # sync_api and async_api re-export the same TimeoutError class
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        timeout_types = (TimeoutError, PlaywrightTimeout)
    except ImportError:  # playwright absent (mocked browser in tests)
        pass

    if isinstance(exc, timeout_types):
        raise TimeoutError(
            f"Widget did not finish rendering within {timeout}s"
        ) from exc


@contextlib.contextmanager
def _rendered_page(
    browser: Browser, inputs: _RenderInputs
) -> Iterator[tuple[Page, str | None, float]]:
    """Yield ``(page, render_error, scale_factor)`` for a fully loaded widget page.

    Builds the capture HTML, opens it in a new page, and waits for the JS side
    to signal render completion. Closes the page and deletes the temp HTML
    file on exit.
    """
    tmp_path, timeout_ms, scale_factor = _prepare_render(inputs)
    page: Page | None = None
    try:
        page = browser.new_page(
            viewport={"width": 1024, "height": 768}, device_scale_factor=scale_factor
        )
        page.goto(
            Path(tmp_path).as_uri(), wait_until="domcontentloaded", timeout=timeout_ms
        )
        page.wait_for_function(
            "() => window.__RENDER_DONE === true", timeout=timeout_ms
        )
        yield page, page.evaluate("() => window.__RENDER_ERROR"), scale_factor
    finally:
        if page is not None:
            try:
                page.close()
            except (OSError, RuntimeError):
                pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@contextlib.asynccontextmanager
async def _rendered_page_async(
    browser: AsyncBrowser, inputs: _RenderInputs
) -> AsyncIterator[tuple[AsyncPage, str | None, float]]:
    """Async twin of ``_rendered_page`` for event-loop (Jupyter) contexts."""
    tmp_path, timeout_ms, scale_factor = _prepare_render(inputs)
    page: AsyncPage | None = None
    try:
        page = await browser.new_page(
            viewport={"width": 1024, "height": 768}, device_scale_factor=scale_factor
        )
        await page.goto(
            Path(tmp_path).as_uri(), wait_until="domcontentloaded", timeout=timeout_ms
        )
        await page.wait_for_function(
            "() => window.__RENDER_DONE === true", timeout=timeout_ms
        )
        yield page, await page.evaluate("() => window.__RENDER_ERROR"), scale_factor
    finally:
        if page is not None:
            try:
                await page.close()
            except (OSError, RuntimeError):
                pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def _render_widget_async(inputs: _RenderInputs, quality: int = 90) -> bytes:
    """Async implementation of render_widget_headless for event-loop contexts."""
    browser = await _get_async_browser()
    try:
        async with _rendered_page_async(browser, inputs) as (page, render_error, scale):
            if render_error:
                raise RuntimeError(f"Widget render failed: {render_error}")  # noqa: TRY301
            return await _async_capture_page(page, inputs.fmt, quality, scale)
    except Exception as exc:
        _raise_if_timeout(exc, inputs.timeout)
        raise


@functools.lru_cache(maxsize=1)
def _get_esm_b64(esm_content: str) -> str:
    """Return base64-encoded ESM bundle, cached to avoid re-encoding ~4.5 MB."""
    return base64.b64encode(esm_content.encode("utf-8")).decode("ascii")


def _worker_shim(widget_type: str | None) -> str:
    """JS that disables Web Workers when the embedded ESM can't load its worker.

    ChemPotDiagram offloads its compute to a Web Worker whose URL is resolved
    against ``import.meta.url``. When we embed the bundle (blob URL in headless,
    CDN URL in ``to_html``) that resolution either throws or 404s, leaving the
    widget stuck on a loading spinner. Disabling Worker triggers the component's
    documented synchronous main-thread fallback. Scoped to chem_pot_diagram so
    fflate's async gzip/zip workers keep working for other widgets (e.g.
    ``.gz`` ``data_url`` loading).
    """
    if widget_type == "chem_pot_diagram":
        return "globalThis.Worker = undefined;\n"
    return ""


def _widget_data_json(widget_data: dict[str, Any]) -> str:
    """Serialize widget data for embedding in a ``<script>`` tag.

    Escapes ``</`` so the JSON can't prematurely close the script tag.
    """
    return json.dumps(widget_data, default=str).replace("</", r"<\/")


def _esm_blob_loader_js(esm_content: str) -> str:
    """JS that decodes the base64 ESM bundle into a ``blob:`` URL named ``esm_url``.

    blob URLs work with dynamic ``import()``; ``data:`` module URLs are blocked by
    browsers, hence the base64 + Blob dance.
    """
    esm_b64 = _get_esm_b64(esm_content)
    return (
        f'const esm_bytes = Uint8Array.from(atob("{esm_b64}"), c => c.charCodeAt(0));\n'
        'const esm_blob = new Blob([esm_bytes], { type: "application/javascript" });\n'
        "const esm_url = URL.createObjectURL(esm_blob);"
    )


# Anywidget model mock (only get() is needed to render) + mount. Assumes
# ``esm_url`` and ``widget_data`` are already defined; defines ``el`` so callers
# can reuse the mounted root. Plain single-brace JS (injected via substitution).
_MOUNT_WIDGET_JS = """\
const model = {
  get: (key) => widget_data[key],
  set: () => {}, on: () => {}, off: () => {}, save_changes: () => {}, send: () => {},
};
const mod = await import(esm_url);
const render_fn = mod.default?.render ?? mod.render;
if (!render_fn) throw new Error("ESM bundle has no render export");
const el = document.getElementById("widget-root");
render_fn({ model, el });"""


def _build_html(
    widget_data: dict[str, Any],
    esm_content: str,
    css_content: str,
    timeout: float = 30.0,
    *,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Build a self-contained HTML page that renders one widget.

    The page embeds the ESM bundle as a blob URL, injects the widget data
    as a global, creates a mock anywidget model, and calls render().
    After the Svelte component mounts and a short settle delay,
    ``window.__RENDER_DONE`` is set so the Python side knows when to capture.

    Args:
        widget_data: Traitlet values from ``widget.to_dict()``.
        esm_content: Full text of ``matterviz.js``.
        css_content: Full text of ``matterviz.css``.
        timeout: Max seconds for the JS-side render polling loop.
        width: Override container width in pixels (takes precedence
            over widget style).
        height: Override container height in pixels (takes precedence
            over widget style).

    Returns:
        Complete HTML document as a string.
    """
    data_json = _widget_data_json(widget_data)
    esm_loader = _esm_blob_loader_js(esm_content)

    # Build container style: user style first, then explicit overrides
    # (later declarations win in inline style), then defaults for any
    # missing dimension.
    user_style = widget_data.get("style") or ""
    has_width = bool(re.search(r"(?:^|;\s*)width\s*:", user_style))
    has_height = bool(re.search(r"(?:^|;\s*)height\s*:", user_style))
    parts = []
    if user_style:
        parts.append(user_style)
    if width is not None:
        parts.append(f"width: {width}px")
    elif not has_width:
        parts.append("width: 800px")
    if height is not None:
        parts.append(f"height: {height}px")
    elif not has_height:
        parts.append("height: 600px")
    widget_style = html_mod.escape("; ".join(parts), quote=True)

    # Chart widgets render an <svg> or <canvas> we can wait for. HTML-only
    # widgets (PeriodicTable/HeatmapMatrix render a grid of tiles) have neither,
    # so only those enable the HTML-content fallback. Keeping it off for chart
    # widgets prevents a half-laid-out plot or not-yet-painted WebGL canvas from
    # being captured prematurely via stray HTML (labels, legends).
    widget_type = widget_data.get("widget_type")
    allow_html = "true" if widget_type in _HTML_WIDGET_TYPES else "false"
    worker_shim = _worker_shim(widget_type)

    return f"""\
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>{css_content}</style>
</head><body style="margin: 0; padding: 0;">
<div id="widget-root" style="{widget_style}"></div>
<script type="module">
{worker_shim}{esm_loader}

const widget_data = {data_json};

try {{
{_MOUNT_WIDGET_JS}

  // Svelte 5's bind:clientWidth/Height uses ResizeObserver internally.
  // When elements already have their final size at mount time, the
  // observer may fire once but the initial $state([0,0]) might not
  // update if the microtask runs before the observer callback. Force
  // a layout change by briefly resizing the wrapper, which guarantees
  // the ResizeObserver fires with the correct non-zero dimensions.
  const wrapper = el.firstElementChild;
  if (wrapper) {{
    const orig_w = wrapper.style.width;
    wrapper.style.width = (wrapper.clientWidth + 1) + "px";
    await new Promise(r => setTimeout(r, 50));
    wrapper.style.width = orig_w;
    await new Promise(r => setTimeout(r, 50));
  }}
  // Double RAF to let Svelte process the reactive update
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

  // Poll for visible content (canvas, SVG, or rendered HTML) up to the timeout.
  const start = Date.now();
  const max_wait = {int(timeout * 1000)};
  const allow_html_fallback = {allow_html};
  let found_content = false;
  while (Date.now() - start < max_wait) {{
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    // Skip while a loading spinner is visible (e.g. async compute) so we never
    // capture a transient "Computing..." state instead of the finished plot.
    if (!el.querySelector('.spinner[role="status"]')) {{
      if (el.querySelector("canvas") || el.querySelector("svg")) {{
        // Extra settle time for Three.js scenes to complete first paint
        await new Promise(r => setTimeout(r, 500));
        found_content = true;
        break;
      }}
      // Fallback for HTML-only widgets (e.g. PeriodicTableWidget) that render
      // neither canvas nor svg: accept once the widget root has laid-out,
      // sized, non-empty content.
      if (allow_html_fallback) {{
        const child = el.firstElementChild;
        if (child && child.getBoundingClientRect().height > 20
            && el.textContent.trim().length > 0) {{
          await new Promise(r => setTimeout(r, 200));
          found_content = true;
          break;
        }}
      }}
    }}
    await new Promise(r => setTimeout(r, 100));
  }}

  if (!found_content) {{
    window.__RENDER_ERROR =
      `No capturable content (canvas, SVG, or HTML) after ${{max_wait}}ms `
      + `(widget may still be loading)`;
  }}
  window.__RENDER_DONE = true;
}} catch (err) {{
  window.__RENDER_ERROR = String(err);
  window.__RENDER_DONE = true;
}}
</script>
</body></html>"""


def build_interactive_html(
    widget_data: dict[str, Any],
    css_content: str,
    *,
    esm_url: str | None = None,
    esm_content: str | None = None,
    title: str = "MatterViz widget",
    description: str = "",
) -> str:
    """Build a standalone, interactive HTML page rendering one widget.

    Unlike ``_build_html`` (which is tuned for headless capture), this keeps the
    widget's controls and omits the render-done/capture machinery so the result
    is a shareable, interactive page.

    Exactly one ESM source must be given: ``esm_url`` (referenced via
    ``import(url)`` -- the host must serve it with CORS and a JS MIME type) or
    ``esm_content`` (inlined as a base64 ``blob:`` URL -- self-contained, opens
    offline). CSS is always inlined.

    Note: the no-op anywidget model mock means client-side interactions (3D
    rotate/zoom, hover, control-pane local state) work, but controls that
    round-trip through a Python kernel do not (there is no kernel).
    """
    if (esm_url is None) == (esm_content is None):
        raise ValueError("provide exactly one of esm_url or esm_content")

    data_json = _widget_data_json(widget_data)
    worker_shim = _worker_shim(widget_data.get("widget_type"))
    if esm_content is not None:
        esm_loader = _esm_blob_loader_js(esm_content)
    else:
        esm_loader = f"const esm_url = {json.dumps(esm_url)};"

    style = html_mod.escape(
        widget_data.get("style") or "width: 100%; height: 600px;", quote=True
    )
    title_esc = html_mod.escape(title, quote=True)
    desc_esc = html_mod.escape(description, quote=True)

    return f"""\
<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title_esc}</title>
<meta name="description" content="{desc_esc}">
<style>{css_content}</style>
</head><body style="margin: 0; padding: 0;">
<div id="widget-root" style="{style}"></div>
<script type="module">
{worker_shim}{esm_loader}

const widget_data = {data_json};
{_MOUNT_WIDGET_JS}

// Nudge the ResizeObserver so bind:clientWidth/Height pick up the final size.
const wrapper = el.firstElementChild;
if (wrapper) {{
  const orig_w = wrapper.style.width;
  wrapper.style.width = (wrapper.clientWidth + 1) + "px";
  requestAnimationFrame(() => {{ wrapper.style.width = orig_w; }});
}}
</script>
</body></html>"""


# DOM size thresholds shared by the capture and diagnostics probes: canvases
# below _MIN_CANVAS_PX are decorative (not the chart), SVGs below _MIN_SVG_PX
# are toolbar/legend icons.
_MIN_CANVAS_PX = 50
_MIN_SVG_PX = 20

# Extract the largest chart-sized SVG from the widget for vector export.
# Sub-threshold elements are toolbar/legend icons, never the chart, so they
# are ignored and the function returns null when only icons exist.
# Returning null (rather than an icon) lets the caller raise a clear error.
_SVG_EXTRACT_JS = """\
() => {
    const svgs = Array.from(document.querySelectorAll('#widget-root svg'));
    if (!svgs.length) return null;
    let best = null, best_area = 0;
    for (const svg of svgs) {
        const rect = svg.getBoundingClientRect();
        const area = rect.width * rect.height;
        if (rect.width >= %(svg_px)d && rect.height >= %(svg_px)d
            && area > best_area) {
            best = svg; best_area = area;
        }
    }
    if (!best) return null;
    const cloned = best.cloneNode(true);
    if (!cloned.hasAttribute('xmlns'))
        cloned.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
    const style = cloned.getAttribute('style') || '';
    if (!/font-family/.test(style))
        cloned.setAttribute('style', style + ';font-family:sans-serif;');
    cloned.setAttribute('font-family', 'sans-serif');
    return '<?xml version="1.0" encoding="UTF-8"?>\\n'
        + new XMLSerializer().serializeToString(cloned);
}
""" % {"svg_px": _MIN_SVG_PX}  # noqa: UP031  -- %-format keeps JS braces literal

# Detect whether the widget painted a real WebGL <canvas> (vs decorative ones).
# Capture decisions (SVG-vs-raster, vector-vs-rasterized PDF) are made from the
# actual DOM rather than a static widget_type list, because some widgets render
# SVG or canvas depending on their data (e.g. ConvexHull/ChemPotDiagram render
# 2D SVG for binary systems but 3D WebGL for ternary+).
_HAS_CANVAS_JS = """\
() => {
    const root = document.getElementById('widget-root');
    if (!root) return false;
    for (const canvas of root.querySelectorAll('canvas')) {
        const rect = canvas.getBoundingClientRect();
        if (rect.width >= %(canvas_px)d && rect.height >= %(canvas_px)d) return true;
    }
    return false;
}
""" % {"canvas_px": _MIN_CANVAS_PX}  # noqa: UP031  -- %-format keeps JS braces literal

# Widget types that render plain HTML (no <canvas>/<svg> chart) and so need the
# HTML-content fallback in _build_html to detect render completion. All other
# widgets render an <svg> or <canvas> we wait for instead.
_HTML_WIDGET_TYPES = frozenset({"periodic_table", "heatmap_matrix"})


def _no_svg_message(*, has_canvas: bool) -> str:
    """Error message for failed SVG export, tailored to the widget content."""
    if has_canvas:
        return "SVG export not supported for WebGL (canvas) widgets"
    return (
        "No SVG element found in widget. This widget type may not "
        "support SVG export. Use fmt='png' instead."
    )


def _png_to_pdf(png_bytes: bytes, scale: float = 1.0) -> bytes:
    """Wrap a PNG screenshot in a single-page PDF.

    Requires Pillow for RGBA→RGB compositing. Page dimensions are
    pixel dimensions ÷ ``scale`` so high-DPI screenshots produce
    correctly sized pages.

    Args:
        png_bytes: Raw PNG image bytes.
        scale: Device scale factor used when capturing the screenshot.
    """
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for PDF export of WebGL (canvas) widgets.\n"
            "Install it with:  uv pip install Pillow"
        ) from None

    img = Image.open(io.BytesIO(png_bytes))
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    px_w, px_h = img.size
    page_w, page_h = px_w / scale, px_h / scale
    img_data = zlib.compress(img.tobytes(), 6)

    # Build minimal single-page PDF with one FlateDecode image
    draw_cmd = zlib.compress(
        f"q {page_w:.2f} 0 0 {page_h:.2f} 0 0 cm /Im0 Do Q".encode()
    )
    objects: list[bytes] = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        (
            f"3 0 obj\n<< /Type /Page /Parent 2 0 R"
            f" /MediaBox [0 0 {page_w:.2f} {page_h:.2f}]"
            f" /Contents 5 0 R"
            f" /Resources << /XObject << /Im0 4 0 R >> >> >>\n"
            f"endobj\n"
        ).encode(),
        (
            f"4 0 obj\n<< /Type /XObject /Subtype /Image"
            f" /Width {px_w} /Height {px_h} /ColorSpace /DeviceRGB"
            f" /BitsPerComponent 8 /Filter /FlateDecode"
            f" /Length {len(img_data)} >>\nstream\n"
        ).encode()
        + img_data
        + b"\nendstream\nendobj\n",
        (
            f"5 0 obj\n<< /Length {len(draw_cmd)} /Filter /FlateDecode >>\nstream\n"
        ).encode()
        + draw_cmd
        + b"\nendstream\nendobj\n",
    ]

    header = b"%PDF-1.4\n"
    body = bytearray()
    offsets: list[int] = []
    for obj in objects:
        offsets.append(len(header) + len(body))
        body.extend(obj)
    xref_pos = len(header) + len(body)
    xref_entries = "".join(f"{off:010d} 00000 n \n" for off in offsets)
    xref = (
        f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n{xref_entries}"
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    )
    return bytes(header) + bytes(body) + xref.encode()


def _capture_page(
    page: Page,
    fmt: str,
    quality: int = 90,
    scale_factor: float = 1.0,
) -> bytes:
    """Capture the rendered widget from a Playwright page.

    Canvas-vs-SVG is detected from the rendered DOM (not the widget type), so
    widgets that switch between SVG and WebGL based on their data export
    correctly in both modes.

    Args:
        page: Playwright page with the widget already rendered.
        fmt: ``"png"``, ``"jpeg"``, ``"svg"``, or ``"pdf"``.
        quality: JPEG compression quality (1-100). Ignored for other formats.
        scale_factor: Device scale factor for DPI-correct PDF page sizing.

    Returns:
        Raw image/document bytes.
    """
    _validate_capture_format(fmt)

    if fmt == "svg":
        svg_string = page.evaluate(_SVG_EXTRACT_JS)
        if svg_string is None:
            has_canvas = page.evaluate(_HAS_CANVAS_JS)
            raise RuntimeError(_no_svg_message(has_canvas=has_canvas))
        return svg_string.encode("utf-8")

    if fmt == "pdf":
        if page.evaluate(_HAS_CANVAS_JS):
            # WebGL canvas content is invisible to Chromium's print PDF
            # pipeline. Take a high-res screenshot and wrap it in a PDF.
            return _png_to_pdf(
                page.locator("#widget-root").screenshot(type="png"),
                scale=scale_factor,
            )

        # SVG/HTML widgets: native vector PDF with selectable text
        page.emulate_media(media="screen")
        bbox = page.locator("#widget-root").bounding_box()
        return page.pdf(**_vector_pdf_kwargs(bbox))

    # PNG/JPEG -- screenshot the widget container
    root = page.locator("#widget-root")
    if fmt == "jpeg":
        return root.screenshot(type="jpeg", quality=quality)
    return root.screenshot(type="png")


def _write_temp_html(html: str) -> str:
    """Write HTML to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".html")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


def _has_running_event_loop() -> bool:
    """Return True if there is a running asyncio event loop in this thread."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _apply_nest_asyncio(loop: asyncio.AbstractEventLoop) -> bool:
    """Patch the running loop for re-entrant ``run_until_complete``.

    Returns False when nest_asyncio is not installed (callers decide whether
    that is a hard error or a soft diagnostic).
    """
    try:
        import nest_asyncio  # ty: ignore[unresolved-import]
    except ImportError:
        return False
    nest_asyncio.apply(loop)
    return True


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
) -> bytes:
    """Render a widget headlessly and capture it as an image.

    Launches (or reuses) a headless Chromium browser, loads a minimal HTML
    page that runs the ESM bundle with a mock anywidget model, waits for
    the render to complete, and captures the output.

    When called from inside a running asyncio event loop (e.g. jupyter
    nbconvert --execute), uses Playwright's async API via nest_asyncio
    to avoid the greenlet deadlock that occurs with sync Playwright in
    a ThreadPoolExecutor.

    Args:
        widget_data: Dict of traitlet values (from ``widget.to_dict()``).
        esm_content: Full ESM bundle text (``matterviz.js``).
        css_content: Full CSS text (``matterviz.css``).
        fmt: Output format — ``"png"``, ``"jpeg"``, ``"svg"``, or ``"pdf"``.
        dpi: Resolution for raster capture. Maps to ``device_scale_factor``
            in the headless browser (72 DPI = 1x, 144 = 2x, 216 = 3x).
            Defaults to 150 (~2x). Applies to PNG, JPEG, and canvas/WebGL PDFs;
            ignored for SVG and pure-vector (SVG-content) PDFs.
        timeout: Max seconds to wait for the widget to render.
        quality: JPEG compression quality (1-100). Ignored for other formats.
        width: Override container width in pixels.
        height: Override container height in pixels.

    Returns:
        Raw image/document bytes (PNG, JPEG, SVG, or PDF).

    Raises:
        RuntimeError: If the widget fails to render or the requested
            format is unsupported for the widget type.
        TimeoutError: If the widget does not finish rendering in time.
    """
    widget_data = {**widget_data, "show_controls": False}
    inputs = _RenderInputs(
        widget_data, esm_content, css_content, fmt, dpi, timeout, width, height
    )

    if _has_running_event_loop():
        loop = asyncio.get_running_loop()
        if not _apply_nest_asyncio(loop):
            raise RuntimeError(
                "nest_asyncio is required for headless widget export inside "
                "Jupyter/IPython event loops.\n"
                "Install it with:  uv pip install nest-asyncio"
            )
        return loop.run_until_complete(_render_widget_async(inputs, quality))

    browser = _get_browser()
    try:
        with _rendered_page(browser, inputs) as (page, render_error, scale):
            if render_error:
                raise RuntimeError(f"Widget render failed: {render_error}")  # noqa: TRY301
            return _capture_page(page, inputs.fmt, quality, scale)
    except Exception as exc:
        _raise_if_timeout(exc, inputs.timeout)
        raise


# === Render diagnostics (render_report) ===

# DOM probe: classify content and measure root vs content/scroll size to detect
# overflow (clipping). Shares size thresholds with _HAS_CANVAS_JS/_SVG_EXTRACT_JS
# so content_type classification always agrees with capture decisions.
_CONTENT_METRICS_JS = """\
() => {
    const root = document.getElementById('widget-root');
    if (!root) return null;
    const has_canvas = [...root.querySelectorAll('canvas')].some(c => {
        const r = c.getBoundingClientRect();
        return r.width >= %(canvas_px)d && r.height >= %(canvas_px)d;
    });
    const has_svg = [...root.querySelectorAll('svg')].some(s => {
        const r = s.getBoundingClientRect();
        return r.width >= %(svg_px)d && r.height >= %(svg_px)d;
    });
    const text_len = (root.textContent || '').trim().length;
    const rect = root.getBoundingClientRect();
    return {
        content_type: has_canvas ? 'canvas' : has_svg ? 'svg'
            : text_len ? 'html' : 'empty',
        root_w: Math.round(rect.width), root_h: Math.round(rect.height),
        scroll_w: root.scrollWidth, scroll_h: root.scrollHeight,
        text_len,
    };
}
""" % {"canvas_px": _MIN_CANVAS_PX, "svg_px": _MIN_SVG_PX}  # noqa: UP031


@dataclasses.dataclass(frozen=True)
class RenderReport:
    """Structured result of a headless render health check (see render_report).

    Attributes:
        ok: True if the widget rendered without error/timeout.
        summary: ``describe_widget`` facts about the widget's data.
        content_type: ``"canvas"``, ``"svg"``, ``"html"``, ``"empty"``, or None.
        width/height: Rendered widget-root size in CSS pixels.
        blank_fraction: Most-common-color fraction (1.0 == uniform), or None if
            Pillow is unavailable.
        is_blank: True if ``blank_fraction`` exceeds a conservative threshold.
        clipped: Heuristic -- content overflows the widget root (may
            false-positive on intentional scroll areas).
        warnings: Cheap input warnings (empty/degenerate data).
        error: Render error/timeout message, or None.
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
    """Return the most-common-color pixel fraction (1.0 == blank), or None.

    Returns None when Pillow is not installed.
    """
    try:
        from PIL import Image
    except ImportError:
        return None

    import numpy as np

    pixels = np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB")).reshape(-1, 3)
    # pack RGB into one int per pixel: orders of magnitude faster to unique
    # than np.unique(pixels, axis=0), which sorts rows via a void-dtype view
    packed = (
        (pixels[:, 0].astype(np.uint32) << 16)
        | (pixels[:, 1].astype(np.uint32) << 8)
        | pixels[:, 2]
    )
    _, counts = np.unique(packed, return_counts=True)
    return float(counts.max() / len(packed))


def _assemble_diagnostics(
    png_bytes: bytes | None, metrics: dict[str, Any] | None, render_error: str | None
) -> RenderReport:
    """Combine an optional screenshot and DOM metrics into a RenderReport.

    ``png_bytes``/``metrics`` are None when the render errored before capture;
    the report then carries just ``ok=False`` and ``error``. ``summary`` and
    ``warnings`` are left empty for the caller to fill in.
    """
    frac = png_blank_fraction(png_bytes) if png_bytes is not None else None
    info = metrics or {}
    root_w, root_h = info.get("root_w"), info.get("root_h")
    overflow_x = bool(root_w) and info.get("scroll_w", 0) > root_w + 4
    overflow_y = bool(root_h) and info.get("scroll_h", 0) > root_h + 4
    return RenderReport(
        ok=render_error is None,
        error=render_error,
        content_type=info.get("content_type"),
        width=root_w,
        height=root_h,
        blank_fraction=frac,
        is_blank=frac is not None and frac > 0.99,
        clipped=overflow_x or overflow_y,
    )


def _diagnose_sync(inputs: _RenderInputs) -> RenderReport:
    """Sync headless render that returns visual + DOM diagnostics in one pass."""
    try:
        with _rendered_page(_get_browser(), inputs) as (page, render_error, _scale):
            metrics = page.evaluate(_CONTENT_METRICS_JS)
            png_bytes = page.locator("#widget-root").screenshot(type="png")
            return _assemble_diagnostics(png_bytes, metrics, render_error)
    except Exception as exc:  # noqa: BLE001
        return _assemble_diagnostics(None, None, f"render failed: {exc}")


async def _diagnose_async(inputs: _RenderInputs) -> RenderReport:
    """Async twin of _diagnose_sync for event-loop (Jupyter) contexts."""
    browser = await _get_async_browser()
    try:
        async with _rendered_page_async(browser, inputs) as (
            page,
            render_error,
            _scale,
        ):
            metrics = await page.evaluate(_CONTENT_METRICS_JS)
            png_bytes = await page.locator("#widget-root").screenshot(type="png")
            return _assemble_diagnostics(png_bytes, metrics, render_error)
    except Exception as exc:  # noqa: BLE001
        return _assemble_diagnostics(None, None, f"render failed: {exc}")


def render_diagnostics(
    widget_data: dict[str, Any],
    esm_content: str,
    css_content: str,
    *,
    timeout: float = 30.0,
    dpi: int = 72,
) -> RenderReport:
    """Render a widget headlessly once and return visual + DOM diagnostics.

    Dispatches to the async path inside a running event loop (Jupyter), mirroring
    ``render_widget_headless``. Never raises -- failures return a report with
    ``ok=False`` and ``error`` set (``summary``/``warnings`` left empty).
    """
    widget_data = {**widget_data, "show_controls": False}
    inputs = _RenderInputs(
        widget_data, esm_content, css_content, dpi=dpi, timeout=timeout
    )

    if _has_running_event_loop():
        loop = asyncio.get_running_loop()
        if not _apply_nest_asyncio(loop):
            return _assemble_diagnostics(
                None, None, "nest_asyncio is required inside a running event loop"
            )
        return loop.run_until_complete(_diagnose_async(inputs))

    return _diagnose_sync(inputs)
