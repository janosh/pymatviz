"""Headless widget rendering via Playwright for image export.

Renders MatterViz widgets in a headless Chromium browser using the same
ESM bundle and CSS that the notebook frontend uses, then captures the
output as PNG, SVG, or PDF. Works without any notebook/IDE frontend.
"""

from __future__ import annotations

import atexit
import base64
import json
import os
import tempfile
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from playwright.sync_api import Browser, Page

# Module-level browser cache -- reused across to_img() calls to amortize
# the ~2s Chromium startup cost.
_browser: Browser | None = None


def _get_browser() -> Browser:
    """Return a cached headless Chromium browser, launching one if needed."""
    global _browser  # noqa: PLW0603
    if _browser is not None and _browser.is_connected():
        return _browser

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise ImportError(
            "playwright is required for headless widget export.\n"
            "Install it with:  uv pip install playwright && playwright install chromium"
        ) from None

    pw = sync_playwright().start()
    _browser = pw.chromium.launch(headless=True)
    atexit.register(_shutdown_browser, pw)
    return _browser


def _shutdown_browser(pw: Any) -> None:
    """Clean up the browser and Playwright on interpreter exit."""
    global _browser  # noqa: PLW0603
    if _browser is not None:
        try:
            _browser.close()
        except Exception:  # noqa: BLE001, S110
            pass
        _browser = None
    try:
        pw.stop()
    except Exception:  # noqa: BLE001, S110
        pass


def _build_html(
    widget_data: dict[str, Any],
    esm_content: str,
    css_content: str,
    timeout: float = 30.0,
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

    Returns:
        Complete HTML document as a string.
    """
    # Escape </ sequences to prevent premature </script> closure
    data_json = json.dumps(widget_data, default=str).replace("</", r"<\/")

    # Base64-encode the ESM bundle so we can load it as a blob URL.
    # data: URLs with type=module are blocked by browsers for security,
    # but blob: URLs work fine.
    esm_b64 = base64.b64encode(esm_content.encode("utf-8")).decode("ascii")

    # Ensure the container has explicit dimensions so SVG/Canvas components
    # get a non-zero bounding box and actually render.
    user_style = widget_data.get("style") or ""
    parts = []
    if "width" not in user_style:
        parts.append("width: 800px")
    if "height" not in user_style:
        parts.append("height: 600px")
    if user_style:
        parts.append(user_style)
    widget_style = "; ".join(parts)

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
    const has_content = el.querySelector("canvas") || el.querySelector("svg");
    if (has_content) {{
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
_CANVAS_WIDGET_TYPES = frozenset(
    {
        "structure",
        "trajectory",
        "fermi_surface",
        "brillouin_zone",
        "scatter_plot_3d",
    }
)


def _capture_page(
    page: Page,
    fmt: str,
    widget_type: str | None,
) -> bytes:
    """Capture the rendered widget from a Playwright page.

    Args:
        page: Playwright page with the widget already rendered.
        fmt: ``"png"``, ``"jpeg"``, or ``"svg"``.
        widget_type: The widget_type string for canvas-vs-SVG detection.

    Returns:
        Raw image bytes.
    """
    _valid_capture_fmts = ("png", "jpeg", "svg")
    if fmt not in _valid_capture_fmts:
        raise ValueError(
            f"Unsupported capture format {fmt!r}, expected one of {_valid_capture_fmts}"
        )
    if fmt == "svg":
        if widget_type in _CANVAS_WIDGET_TYPES:
            raise RuntimeError("SVG export not supported for WebGL (canvas) widgets")

        # Find the main chart SVG -- the largest one by area, ignoring
        # small icon SVGs (fullscreen toggles, etc.). Requires at
        # least 100x100 px to distinguish real charts from icons.
        svg_string = page.evaluate("""\
            () => {
                const svgs = Array.from(
                    document.querySelectorAll('#widget-root svg')
                );
                if (!svgs.length) return null;

                // Pick the SVG with the largest bounding box, filtering
                // out tiny icon SVGs (< 100px in either dimension)
                let best = null;
                let best_area = 0;
                for (const svg of svgs) {
                    const rect = svg.getBoundingClientRect();
                    if (rect.width < 100 || rect.height < 100) continue;
                    const area = rect.width * rect.height;
                    if (area > best_area) {
                        best = svg;
                        best_area = area;
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
                const body = new XMLSerializer().serializeToString(cloned);
                return '<?xml version="1.0" encoding="UTF-8"?>\\n' + body;
            }
        """)
        if svg_string is None:
            raise RuntimeError(
                "No SVG element found in widget. This widget type may not "
                "support SVG export. Use fmt='png' instead."
            )
        return svg_string.encode("utf-8")

    # PNG/JPEG -- screenshot the widget container
    root = page.locator("#widget-root")
    if fmt == "jpeg":
        return root.screenshot(type="jpeg", quality=90)
    return root.screenshot(type="png")


def _write_temp_html(html: str) -> str:
    """Write HTML to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".html")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(html)
    return path


def render_widget_headless(
    widget_data: dict[str, Any],
    esm_content: str,
    css_content: str,
    fmt: str = "png",
    dpi: int = 150,
    timeout: float = 30.0,
) -> bytes:
    """Render a widget headlessly and capture it as an image.

    Launches (or reuses) a headless Chromium browser, loads a minimal HTML
    page that runs the ESM bundle with a mock anywidget model, waits for
    the render to complete, and captures the output.

    Args:
        widget_data: Dict of traitlet values (from ``widget.to_dict()``).
        esm_content: Full ESM bundle text (``matterviz.mjs``).
        css_content: Full CSS text (``matterviz.css``).
        fmt: Output format -- ``"png"``, ``"jpeg"``, or ``"svg"``.
        dpi: Resolution for PNG capture. Maps to ``device_scale_factor``
            in the headless browser (72 DPI = 1x, 144 = 2x, 216 = 3x).
            Defaults to 150 (~2x). Ignored for SVG.
        timeout: Max seconds to wait for the widget to render.

    Returns:
        Raw image bytes (PNG, JPEG, or SVG).

    Raises:
        RuntimeError: If the widget fails to render or the requested
            format is unsupported for the widget type.
        TimeoutError: If the widget does not finish rendering in time.
    """
    html = _build_html(widget_data, esm_content, css_content, timeout=timeout)
    browser = _get_browser()
    timeout_ms = int(timeout * 1000)

    # Map DPI to Chromium's device_scale_factor (72 DPI = 1x baseline)
    scale_factor = max(1, dpi / 72) if fmt in ("png", "jpeg") else 1

    # Write to a temp file and load via file:// URL.
    # page.set_content() uses about:blank which breaks Svelte's
    # ResizeObserver-based bind:clientWidth/Height -- components
    # never detect their container dimensions and skip rendering.
    tmp_path = _write_temp_html(html)
    page = browser.new_page(
        viewport={"width": 1024, "height": 768},
        device_scale_factor=scale_factor,
    )
    try:
        page.goto(
            f"file://{tmp_path}", wait_until="domcontentloaded", timeout=timeout_ms
        )
        page.wait_for_function(
            "() => window.__RENDER_DONE === true", timeout=timeout_ms
        )

        render_error = page.evaluate("() => window.__RENDER_ERROR")
        if render_error:
            raise RuntimeError(f"Widget render failed: {render_error}")  # noqa: TRY301

        return _capture_page(page, fmt, widget_data.get("widget_type"))
    except (RuntimeError, TimeoutError):
        raise
    except Exception as exc:
        if "Timeout" in type(exc).__name__:
            raise TimeoutError(
                f"Widget did not finish rendering within {timeout}s"
            ) from exc
        raise
    finally:
        page.close()
        os.unlink(tmp_path)
