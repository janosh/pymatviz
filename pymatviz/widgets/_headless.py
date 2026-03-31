"""Headless widget rendering via Playwright for image export.

Renders MatterViz widgets in a headless Chromium browser using the same
ESM bundle and CSS that the notebook frontend uses, then captures the
output as PNG, JPEG, SVG, or PDF. Works without any notebook/IDE frontend.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
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


def _prepare_render(
    widget_data: dict[str, Any],
    esm_content: str,
    css_content: str,
    fmt: str,
    dpi: int,
    timeout: float,
    width: int | None,
    height: int | None,
) -> tuple[str, int, float]:
    """Build HTML, compute scale factor, write temp file.

    Returns (tmp_path, timeout_ms, scale_factor).
    """
    html = _build_html(
        widget_data, esm_content, css_content, timeout, width=width, height=height
    )
    timeout_ms = int(timeout * 1000)
    needs_raster = fmt in ("png", "jpeg") or (
        fmt == "pdf" and widget_data.get("widget_type") in _CANVAS_WIDGET_TYPES
    )
    scale_factor = max(1, dpi / 72) if needs_raster else 1
    return _write_temp_html(html), timeout_ms, scale_factor


async def _async_capture_page(
    page: AsyncPage,
    fmt: str,
    widget_type: str | None,
    quality: int = 90,
    scale_factor: float = 1.0,
) -> bytes:
    """Async version of _capture_page."""
    if fmt not in ("png", "jpeg", "svg", "pdf"):
        raise ValueError(
            f"Unsupported capture format {fmt!r}, "
            "expected 'png', 'jpeg', 'svg', or 'pdf'"
        )

    if fmt == "svg":
        if widget_type in _CANVAS_WIDGET_TYPES:
            raise RuntimeError("SVG export not supported for WebGL (canvas) widgets")
        svg_string = await page.evaluate(_SVG_EXTRACT_JS)
        if svg_string is None:
            raise RuntimeError(
                "No SVG element found in widget. This widget type may not "
                "support SVG export. Use fmt='png' instead."
            )
        return svg_string.encode("utf-8")

    if fmt == "pdf":
        if widget_type in _CANVAS_WIDGET_TYPES:
            return _png_to_pdf(
                await page.locator("#widget-root").screenshot(type="png"),
                scale=scale_factor,
            )
        await page.emulate_media(media="screen")
        bbox = await page.locator("#widget-root").bounding_box()
        if bbox is None:
            raise RuntimeError("Widget root has no bounding box for PDF export")
        return await page.pdf(
            width=f"{bbox['width']}px",
            height=f"{bbox['height']}px",
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        )

    root = page.locator("#widget-root")
    if fmt == "jpeg":
        return await root.screenshot(type="jpeg", quality=quality)
    return await root.screenshot(type="png")


async def _render_widget_async(
    widget_data: dict[str, Any],
    esm_content: str,
    css_content: str,
    fmt: str = "png",
    dpi: int = 150,
    timeout: float = 30.0,  # noqa: ASYNC109
    quality: int = 90,
    width: int | None = None,
    height: int | None = None,
) -> bytes:
    """Async implementation of render_widget_headless for event-loop contexts."""
    tmp_path, timeout_ms, scale_factor = _prepare_render(
        widget_data, esm_content, css_content, fmt, dpi, timeout, width, height
    )
    browser = await _get_async_browser()
    page: AsyncPage | None = None
    try:
        page = await browser.new_page(
            viewport={"width": 1024, "height": 768},
            device_scale_factor=scale_factor,
        )
        await page.goto(
            Path(tmp_path).as_uri(), wait_until="domcontentloaded", timeout=timeout_ms
        )
        await page.wait_for_function(
            "() => window.__RENDER_DONE === true", timeout=timeout_ms
        )
        render_error = await page.evaluate("() => window.__RENDER_ERROR")
        if render_error:
            raise RuntimeError(f"Widget render failed: {render_error}")  # noqa: TRY301

        return await _async_capture_page(
            page, fmt, widget_data.get("widget_type"), quality, scale_factor
        )
    except Exception as exc:
        from playwright.async_api import TimeoutError as PlaywrightTimeout

        if isinstance(exc, (PlaywrightTimeout, TimeoutError)):
            raise TimeoutError(
                f"Widget did not finish rendering within {timeout}s"
            ) from exc
        raise
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


@functools.lru_cache(maxsize=1)
def _get_esm_b64(esm_content: str) -> str:
    """Return base64-encoded ESM bundle, cached to avoid re-encoding ~11 MB."""
    return base64.b64encode(esm_content.encode("utf-8")).decode("ascii")


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
        esm_content: Full text of ``matterviz.mjs``.
        css_content: Full text of ``matterviz.css``.
        timeout: Max seconds for the JS-side render polling loop.
        width: Override container width in pixels (takes precedence
            over widget style).
        height: Override container height in pixels (takes precedence
            over widget style).

    Returns:
        Complete HTML document as a string.
    """
    # Escape </ sequences to prevent premature </script> closure
    data_json = json.dumps(widget_data, default=str).replace("</", r"<\/")

    # Base64-encode the ESM bundle so we can load it as a blob URL.
    # data: URLs with type=module are blocked by browsers for security,
    # but blob: URLs work fine. The encoding is cached across calls.
    esm_b64 = _get_esm_b64(esm_content)

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

    return f"""\
<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>{css_content}</style>
</head><body style="margin: 0; padding: 0;">
<div id="widget-root" style="{widget_style}"></div>
<script type="module">
// Decode the base64-encoded ESM bundle and load it as a blob URL
const esm_bytes = Uint8Array.from(atob("{esm_b64}"), c => c.charCodeAt(0));
const esm_blob = new Blob([esm_bytes], {{ type: "application/javascript" }});
const esm_url = URL.createObjectURL(esm_blob);

const widget_data = {data_json};

// Minimal anywidget model mock -- only get() is needed for rendering
const model = {{
  get: (key) => widget_data[key],
  set: () => {{}},
  on: () => {{}},
  off: () => {{}},
  save_changes: () => {{}},
  send: () => {{}},
}};

try {{
  const mod = await import(esm_url);
  const render_fn = mod.default?.render ?? mod.render;
  if (!render_fn) throw new Error("ESM bundle has no render export");

  const el = document.getElementById("widget-root");
  render_fn({{ model, el }});

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

  // Poll for visible content (canvas or SVG) up to the configured timeout.
  const start = Date.now();
  const max_wait = {int(timeout * 1000)};
  let found_content = false;
  while (Date.now() - start < max_wait) {{
    await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
    if (el.querySelector("canvas") || el.querySelector("svg")) {{
      // Extra settle time for Three.js scenes to complete first paint
      await new Promise(r => setTimeout(r, 500));
      found_content = true;
      break;
    }}
    await new Promise(r => setTimeout(r, 100));
  }}

  if (!found_content) {{
    window.__RENDER_ERROR = `No canvas or SVG element found after ${{max_wait}}ms`;
  }}
  window.__RENDER_DONE = true;
}} catch (err) {{
  window.__RENDER_ERROR = String(err);
  window.__RENDER_DONE = true;
}}
</script>
</body></html>"""


# Widget types that render via WebGL <canvas> (not SVG).
# Keep in sync with widget classes in pymatviz/widgets/ that use Three.js.
_SVG_EXTRACT_JS = """\
() => {
    const svgs = Array.from(document.querySelectorAll('#widget-root svg'));
    if (!svgs.length) return null;
    let best = null, best_area = 0, fallback = null, fallback_area = 0;
    for (const svg of svgs) {
        const rect = svg.getBoundingClientRect();
        const area = rect.width * rect.height;
        if (rect.width >= 20 && rect.height >= 20) {
            if (area > best_area) { best = svg; best_area = area; }
        } else if (area > fallback_area) { fallback = svg; fallback_area = area; }
    }
    best = best || fallback;
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
"""

_CANVAS_WIDGET_TYPES = frozenset(
    {
        "structure",
        "trajectory",
        "fermi_surface",
        "brillouin_zone",
        "scatter_plot_3d",
    }
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
    widget_type: str | None,
    quality: int = 90,
    scale_factor: float = 1.0,
) -> bytes:
    """Capture the rendered widget from a Playwright page.

    Args:
        page: Playwright page with the widget already rendered.
        fmt: ``"png"``, ``"jpeg"``, ``"svg"``, or ``"pdf"``.
        widget_type: The widget_type string for canvas-vs-SVG detection.
        quality: JPEG compression quality (1-100). Ignored for other formats.
        scale_factor: Device scale factor for DPI-correct PDF page sizing.

    Returns:
        Raw image/document bytes.
    """
    if fmt not in ("png", "jpeg", "svg", "pdf"):
        raise ValueError(
            f"Unsupported capture format {fmt!r}, "
            "expected 'png', 'jpeg', 'svg', or 'pdf'"
        )

    if fmt == "svg":
        if widget_type in _CANVAS_WIDGET_TYPES:
            raise RuntimeError("SVG export not supported for WebGL (canvas) widgets")

        svg_string = page.evaluate(_SVG_EXTRACT_JS)
        if svg_string is None:
            raise RuntimeError(
                "No SVG element found in widget. This widget type may not "
                "support SVG export. Use fmt='png' instead."
            )
        return svg_string.encode("utf-8")

    if fmt == "pdf":
        if widget_type in _CANVAS_WIDGET_TYPES:
            # WebGL canvas content is invisible to Chromium's print PDF
            # pipeline. Take a high-res screenshot and wrap it in a PDF.
            return _png_to_pdf(
                page.locator("#widget-root").screenshot(type="png"),
                scale=scale_factor,
            )

        # SVG widgets: native vector PDF with selectable text
        page.emulate_media(media="screen")
        bbox = page.locator("#widget-root").bounding_box()
        if bbox is None:
            raise RuntimeError("Widget root has no bounding box for PDF export")
        return page.pdf(
            width=f"{bbox['width']}px",
            height=f"{bbox['height']}px",
            print_background=True,
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
        )

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
        esm_content: Full ESM bundle text (``matterviz.mjs``).
        css_content: Full CSS text (``matterviz.css``).
        fmt: Output format — ``"png"``, ``"jpeg"``, ``"svg"``, or ``"pdf"``.
        dpi: Resolution for raster capture. Maps to ``device_scale_factor``
            in the headless browser (72 DPI = 1x, 144 = 2x, 216 = 3x).
            Defaults to 150 (~2x). Ignored for SVG and PDF.
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

    if _has_running_event_loop():
        loop = asyncio.get_running_loop()
        try:
            import nest_asyncio

            nest_asyncio.apply(loop)
        except ImportError:
            raise RuntimeError(
                "nest_asyncio is required for headless widget export inside "
                "Jupyter/IPython event loops.\n"
                "Install it with:  uv pip install nest-asyncio"
            ) from None
        return loop.run_until_complete(
            _render_widget_async(
                widget_data,
                esm_content,
                css_content,
                fmt,
                dpi,
                timeout,
                quality,
                width,
                height,
            )
        )

    tmp_path, timeout_ms, scale_factor = _prepare_render(
        widget_data, esm_content, css_content, fmt, dpi, timeout, width, height
    )
    browser = _get_browser()
    page: Page | None = None
    try:
        page = browser.new_page(
            viewport={"width": 1024, "height": 768},
            device_scale_factor=scale_factor,
        )
        page.goto(
            Path(tmp_path).as_uri(), wait_until="domcontentloaded", timeout=timeout_ms
        )
        page.wait_for_function(
            "() => window.__RENDER_DONE === true", timeout=timeout_ms
        )

        render_error = page.evaluate("() => window.__RENDER_ERROR")
        if render_error:
            raise RuntimeError(f"Widget render failed: {render_error}")  # noqa: TRY301

        return _capture_page(
            page, fmt, widget_data.get("widget_type"), quality, scale_factor
        )
    except Exception as exc:
        from playwright.sync_api import TimeoutError as PlaywrightTimeout

        if isinstance(exc, (PlaywrightTimeout, TimeoutError)):
            raise TimeoutError(
                f"Widget did not finish rendering within {timeout}s"
            ) from exc
        raise
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
