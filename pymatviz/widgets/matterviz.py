"""Lazy loading of MatterViz widget assets from GitHub releases."""

from __future__ import annotations

import json
import os
import re
import shutil
import urllib.request
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urljoin

import traitlets as tl
from anywidget import AnyWidget


if TYPE_CHECKING:
    from typing import Literal

    from pymatviz.widgets._headless import RenderReport


# npm version of the prebuilt ``matterviz-anywidget`` bundle (built from
# matterviz/extensions/anywidget) that pymatviz renders. Only bump this to a
# version already published to npm: the default to_html/to_img path fetches
# ``matterviz-anywidget@<this>`` from jsDelivr at runtime, so an unpublished pin
# breaks widget rendering for everyone.
MATTERVIZ_ANYWIDGET_VERSION = "0.3.7"
_ANYWIDGET_CDN = "https://cdn.jsdelivr.net/npm/matterviz-anywidget"


def _default_cdn_url() -> str:
    """Return the jsDelivr URL for the pinned ``matterviz-anywidget`` ESM bundle.

    jsDelivr serves npm packages with CORS and a JavaScript MIME type (both
    required for cross-origin ``import()``), so this URL works in ``to_html``.
    """
    return f"{_ANYWIDGET_CDN}@{MATTERVIZ_ANYWIDGET_VERSION}/build/matterviz.js"


def _in_marimo_runtime() -> bool:
    """Return whether execution is inside a live marimo runtime context."""
    try:
        from marimo._runtime.context import ContextNotInitializedError, get_context
    except ImportError:
        return False
    try:
        get_context()
    except (ContextNotInitializedError, RuntimeError):
        return False
    else:
        return True


def _marimo_esm_url(esm_text: str) -> str | None:
    """Resolve ESM text to a marimo virtual-file URL, or None if unsupported."""
    try:
        from marimo._output.data.data import js
        from marimo._runtime.context import get_context
    except ImportError:
        return None

    # Virtual files required — without them js() creates ~4.5 MB data URLs
    try:
        if not get_context().virtual_files_supported:
            return None
    except (RuntimeError, AttributeError):
        return None

    try:
        relative_url = js(esm_text).url
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None

    if not relative_url or not relative_url.startswith("./@file/"):
        return None

    # Resolve ./@file/ to absolute URL via request context, fall back to relative
    try:
        request = get_context().request
    except (RuntimeError, AttributeError, TypeError):
        return relative_url
    if request is None:
        return relative_url
    base = request.base_url
    if not isinstance(base, dict):
        return relative_url
    scheme, netloc = base.get("scheme"), base.get("netloc")
    if not isinstance(scheme, str) or not isinstance(netloc, str):
        return relative_url
    return urljoin(f"{scheme}://{netloc}{base.get('path', '/')}", relative_url)


def _read_asset_source(src: str) -> str:
    """Read asset content from an HTTP(S) URL, ``file://`` URI, or local path."""
    if src.startswith(("http://", "https://")):
        with urllib.request.urlopen(src) as response:  # noqa: S310
            return response.read().decode("utf-8")
    src = src.removeprefix("file://")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"Asset file not found: {src}")
    with open(src, encoding="utf-8") as file:
        return file.read()


def configure_assets(
    *,
    version: str | None = None,
    esm_src: str | None = None,
    css_src: str | None = None,
) -> None:
    """Configure matterviz widget assets globally for all subsequent widgets.

    Call with no arguments to reset to default (the pinned bundle version).

    Args:
        version: ``matterviz-anywidget`` npm version to fetch (e.g. ``"0.3.7"``).
            Mutually exclusive with ``esm_src``/``css_src``.
        esm_src: URL or local file path for the ESM JavaScript bundle.
        css_src: URL or local file path for the CSS stylesheet.
            Derived from ``esm_src`` by replacing ``.js`` → ``.css`` if omitted.
    """
    if version and (esm_src or css_src):
        raise ValueError(
            "configure_assets() accepts either 'version' or 'esm_src'/'css_src', "
            "not both."
        )
    if css_src and not esm_src:
        raise ValueError(
            "configure_assets() requires 'esm_src' when 'css_src' is provided."
        )

    cls = MatterVizWidget
    cls._asset_cache.clear()

    if not version and not esm_src:
        for attr in ("_esm", "_css"):
            if attr in cls.__dict__:
                delattr(cls, attr)
        return

    if version:
        esm_content = fetch_widget_asset("matterviz.js", version)
        css_content = fetch_widget_asset("matterviz.css", version)
    else:
        if esm_src is None:
            raise ValueError(
                "configure_assets() requires 'esm_src' when no version is provided."
            )
        if css_src is None:
            css_src = re.sub(r"\.m?js$", ".css", esm_src)
        esm_content = _read_asset_source(esm_src)
        css_content = _read_asset_source(css_src)

    cls._asset_cache["default"] = (esm_content, css_content)
    cls._esm = esm_content
    cls._css = css_content


def fetch_widget_asset(filename: str, version_override: str | None = None) -> str:
    """Fetch a widget asset (``matterviz.js``/``matterviz.css``) as source text.

    The bundle lives in the ``matterviz-anywidget`` npm package (built from
    ``matterviz/extensions/anywidget``); this repo no longer builds it. Resolution
    order:

    1. ``MATTERVIZ_ANYWIDGET_DIR`` env var -- a local ``build/`` dir, for
       developing against an unpublished bundle.
    2. on-disk cache (``~/.cache/pymatviz/matterviz-anywidget/<version>/``).
    3. download from jsDelivr (the ``matterviz-anywidget`` npm package), cached.

    Args:
        filename: ``"matterviz.js"`` or ``"matterviz.css"``.
        version_override: ``matterviz-anywidget`` npm version to fetch instead of
            the pinned ``MATTERVIZ_ANYWIDGET_VERSION``.
    """
    version = version_override or MATTERVIZ_ANYWIDGET_VERSION

    def read_file(path: str) -> str:
        """Read a local/cached asset file."""
        with open(path, encoding="utf-8") as file:
            return file.read()

    dev_dir = os.environ.get("MATTERVIZ_ANYWIDGET_DIR")
    if dev_dir and os.path.isfile(dev_path := f"{dev_dir}/{filename}"):
        return read_file(dev_path)

    cache_root = os.path.expanduser("~/.cache/pymatviz/matterviz-anywidget")
    cache_dir = f"{cache_root}/{version}"
    cache_path = f"{cache_dir}/{filename}"
    if os.path.isfile(cache_path):
        return read_file(cache_path)

    if not re.match(r"^\d+\.\d+\.\d+([-+].+)?$", version):  # semver, optional pre/build
        raise ValueError(f"Invalid matterviz-anywidget version: {version=}")

    os.makedirs(cache_dir, exist_ok=True)
    cdn_url = f"{_ANYWIDGET_CDN}@{version}/build/{filename}"
    # Download to a temp file and atomically rename so an interrupted download
    # never leaves a partial file that later cache-hits and serves a broken bundle.
    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
    try:
        urllib.request.urlretrieve(cdn_url, tmp_path)  # noqa: S310
        os.replace(tmp_path, cache_path)
    except Exception as exc:
        if os.path.isfile(tmp_path):
            os.unlink(tmp_path)
        raise FileNotFoundError(
            f"Could not load {filename} for matterviz-anywidget@{version} from "
            f"{cdn_url}. Ensure that version is published to npm, set "
            "MATTERVIZ_ANYWIDGET_DIR to a local build dir, or check connectivity."
        ) from exc
    return read_file(cache_path)


def clear_widget_cache(version_override: str | None = None) -> None:
    """Clear the downloaded widget-asset cache.

    Args:
        version_override: Only clear this ``matterviz-anywidget`` version's cache.
    """
    cache_dir = os.path.expanduser("~/.cache/pymatviz")
    if version_override:
        version_cache_dir = f"{cache_dir}/matterviz-anywidget/{version_override}"
        if os.path.isdir(version_cache_dir):
            shutil.rmtree(version_cache_dir)
    # Clear entire cache
    elif os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)


class MatterVizWidget(AnyWidget):
    """Base widget class that lazily loads and caches MatterViz widget assets.

    Note:
        The marimo VS Code extension does not support virtual files, so
        widgets with large JS bundles (>5 MB) cannot render there.  Use
        ``marimo edit`` in a browser instead.
    """

    _EXCLUDED_TRAITS: ClassVar[frozenset[str]] = frozenset(
        {"layout", "tabbable", "tooltip"}
    )

    _esm: str
    _css: str
    _asset_cache: ClassVar[dict[str, tuple[str, str] | str | None]] = {}
    widget_type = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    style = tl.Unicode(allow_none=True).tag(sync=True)
    show_controls = tl.Bool(default_value=True).tag(sync=True)

    @classmethod
    def _set_class_assets(cls) -> None:
        """Load and cache default widget assets on the class, not instances."""
        if "default" not in cls._asset_cache:
            cls._asset_cache["default"] = (
                fetch_widget_asset("matterviz.js"),
                fetch_widget_asset("matterviz.css"),
            )
        cls._esm, cls._css = cls._asset_cache["default"]

    def __init__(self, version_override: str | None = None, **kwargs: Any) -> None:
        """Initialize with lazy-loaded widget assets.

        Args:
            version_override: Fetch assets for this version tag instead of the
                installed version. Prefer ``configure_assets()`` for global overrides.
            **kwargs: Passed through to AnyWidget.
        """
        if version_override is not None:
            self._esm = fetch_widget_asset("matterviz.js", version_override)
            self._css = fetch_widget_asset("matterviz.css", version_override)
        elif _in_marimo_runtime():
            self._init_marimo_assets()
        else:
            # Set assets on the class so Jupyter/VS Code share them across
            # instances via ipywidgets comms without per-widget serialization.
            type(self)._set_class_assets()

        super().__init__(**kwargs)

    def _init_marimo_assets(self) -> None:
        """Configure assets for marimo: ESM via virtual-file URL, CSS inline.

        In browser mode, ESM is served via marimo's virtual-file system to
        avoid embedding ~3.4 MB of JS per widget.  In VS Code extension mode,
        virtual files are unavailable — use ``marimo edit`` in a browser.
        """
        cls = type(self)
        cls._set_class_assets()

        # Cache the resolved URL so js() is only called once per session
        if "marimo_esm" not in cls._asset_cache:
            cls._asset_cache["marimo_esm"] = _marimo_esm_url(cls._esm)

        esm_url = cls._asset_cache["marimo_esm"]
        if isinstance(esm_url, str):
            self._esm = esm_url

    def show(self) -> None:
        """Display this widget in notebook environments.

        Safe to call as the last expression in a cell — returns None to
        prevent Jupyter from auto-displaying a second copy.

        When the ``PYMATVIZ_STATIC`` env var is set, renders a static PNG
        via headless Chromium instead of the interactive widget protocol.
        Useful for CI notebook execution where outputs must be viewable
        without a live widget frontend (e.g. GitHub .ipynb rendering).
        """
        from IPython.display import display as ipython_display

        if os.environ.get("PYMATVIZ_STATIC"):
            from IPython.display import Image

            ipython_display(Image(data=self.to_img(fmt="png")))
            return

        ipython_display(self)

    def to_dict(self) -> dict[str, Any]:
        """Return all public synced traitlet values as a plain dict."""
        return {
            name: getattr(self, name)
            for name in self.traits(sync=True)
            if not name.startswith("_") and name not in self._EXCLUDED_TRAITS
        }

    def to_img(
        self,
        filename: str | None = None,
        fmt: Literal["png", "jpeg", "svg", "pdf"] | None = None,
        dpi: int = 150,
        timeout: float = 30.0,
        width: int | None = None,
        height: int | None = None,
        quality: int = 90,
    ) -> bytes:
        """Capture the widget as an image via headless Chromium rendering.

        Launches a headless browser (reused across calls), loads the ESM
        bundle with the widget's data, waits for the component to render,
        and captures the output. Works in plain Python scripts, CI, and
        agent workflows -- no notebook frontend required.

        Note: SVG export is only available for SVG-based (2D) widgets.
        Canvas-based widgets (Structure, Trajectory, etc.) always export
        as PNG/JPEG. PDF export uses Chromium's native PDF generator,
        producing vector output with selectable text for SVG widgets.

        Requires ``playwright`` (install with
        ``uv pip install playwright && playwright install chromium``).

        Args:
            filename: Optional file path to write the image to. Format is
                inferred from the extension if ``fmt`` is not provided.
                Both ``.jpg`` and ``.jpeg`` extensions map to JPEG.
            fmt: Image format -- ``"png"``, ``"jpeg"``, ``"svg"``, or
                ``"pdf"``. Defaults to ``"png"`` if not specified and
                cannot be inferred from ``filename``.
            dpi: Resolution for raster output. Maps to the headless browser's
                device scale factor (72 = 1x, 144 = 2x, etc.). Defaults to 150
                (~2x). Applies to PNG, JPEG, and PDFs that rasterize canvas/WebGL
                content; ignored for SVG and for pure-vector (SVG-content) PDFs.
            timeout: Seconds to wait for headless rendering before
                raising ``TimeoutError``. Defaults to 30.
            width: Override container width in pixels for this export.
                Takes precedence over the widget's ``style``. If omitted,
                uses the widget's style or defaults to 800px.
            height: Override container height in pixels for this export.
                Takes precedence over the widget's ``style``. If omitted,
                uses the widget's style or defaults to 600px.
            quality: JPEG compression quality (1-100). Defaults to 90.
                Ignored for PNG, SVG, and PDF.

        Returns:
            Raw image/document bytes (PNG, JPEG, SVG, or PDF).

        Raises:
            ValueError: If ``fmt`` is not a supported format.
            TimeoutError: If the widget does not finish rendering in time.
            RuntimeError: If the widget fails to render or SVG export is
                requested for a canvas-based widget.
            ImportError: If ``playwright`` is not installed.
        """
        from pymatviz.widgets._headless import render_widget_headless

        if dpi <= 0:
            raise ValueError(f"dpi must be positive, got {dpi}")
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        if not 1 <= quality <= 100:
            raise ValueError(f"quality must be between 1 and 100, got {quality}")
        if width is not None and width <= 0:
            raise ValueError(f"width must be positive, got {width}")
        if height is not None and height <= 0:
            raise ValueError(f"height must be positive, got {height}")

        valid_fmts = ("png", "jpeg", "svg", "pdf")
        if fmt is not None and fmt not in valid_fmts:
            raise ValueError(
                f"Unsupported format {fmt!r}, expected one of {valid_fmts}"
            )
        resolved_fmt = fmt
        if resolved_fmt is None and filename:
            ext = os.path.splitext(filename)[1].lower().lstrip(".")
            if not ext:
                raise ValueError(
                    f"Cannot infer format from {filename=!r} "
                    "(no extension). Pass fmt= explicitly."
                )
            resolved_fmt = {"jpg": "jpeg"}.get(ext, ext)
            if resolved_fmt not in valid_fmts:
                raise ValueError(
                    f"Unsupported file extension '.{ext}'; "
                    f"supported: {sorted(valid_fmts)}"
                )
        if resolved_fmt is None:
            resolved_fmt = "png"

        esm_content, css_content = self._resolve_assets()

        img_bytes = render_widget_headless(
            widget_data=self.to_dict(),
            esm_content=esm_content,
            css_content=css_content,
            fmt=resolved_fmt,
            dpi=dpi,
            timeout=timeout,
            quality=quality,
            width=width,
            height=height,
        )

        if filename:
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            with open(filename, "wb") as fh:
                fh.write(img_bytes)

        return img_bytes

    def _resolve_assets(self) -> tuple[str, str]:
        """Return ``(esm_content, css_content)`` as real source strings.

        We need actual ESM/CSS source for headless rendering and HTML export,
        not a marimo virtual-file URL. ``version_override`` sets both on the
        instance with real content; marimo only overrides ``_esm`` with a URL.
        Use instance attrs only when both are present in ``__dict__``.
        """
        inst_esm = self.__dict__.get("_esm")
        inst_css = self.__dict__.get("_css")
        if isinstance(inst_esm, str) and isinstance(inst_css, str):
            return inst_esm, inst_css

        cls = type(self)
        cls._set_class_assets()
        default_assets = cls._asset_cache["default"]
        if not isinstance(default_assets, tuple):
            raise TypeError("Default widget assets were not initialized correctly.")
        return default_assets

    def describe(self) -> dict[str, Any]:
        """Return structured, machine-parseable facts about this widget's data.

        Pure Python (no browser): derives counts, ranges, formula, chemical
        system, etc. from the synced traitlets. Always includes ``widget_type``.
        Useful for agents to reason about output and as image alt-text.

        Examples:
            >>> PeriodicTableWidget(heatmap_values={"Fe": 42}).describe()["n_elements"]
            1
        """
        from pymatviz.widgets._describe import describe_widget

        return describe_widget(self.to_dict())

    def to_html(
        self,
        filename: str | None = None,
        *,
        inline: bool = False,
        esm_url: str | None = None,
    ) -> str:
        """Export this widget as a standalone, interactive HTML page.

        The page renders the live widget (pan/zoom/rotate/hover work) with no
        notebook required. By default it references the MatterViz bundle from a
        CDN, keeping the file small; pass ``inline=True`` for a self-contained
        file that opens offline.

        Args:
            filename: Optional path to write the HTML to (parent dirs created).
            inline: If True, embed the ~4.5 MB ESM bundle in the file (opens via
                ``file://`` offline). If False (default), reference an external
                URL (small file, requires internet + a CORS-enabled JS host;
                will not load via ``file://``).
            esm_url: Explicit URL to load the bundle from (a host serving it with
                CORS and a JavaScript MIME type). Mutually exclusive with
                ``inline``. When omitted and ``inline`` is False, defaults to the
                jsDelivr URL for the pinned ``matterviz-anywidget`` version.

        Returns:
            The complete HTML document as a string.

        Raises:
            ValueError: If both ``inline=True`` and ``esm_url`` are given.

        Note:
            Controls that round-trip through a Python kernel do not function in
            the exported page (there is no kernel); client-side interactions do.
        """
        from pymatviz.widgets._describe import short_summary
        from pymatviz.widgets._headless import build_interactive_html

        if inline and esm_url is not None:
            raise ValueError("pass either inline=True or esm_url=..., not both")

        esm_content, css_content = self._resolve_assets()
        if inline:
            esm_url = None
        else:
            esm_content = None
            esm_url = esm_url or _default_cdn_url()

        report = self.describe()
        html = build_interactive_html(
            self.to_dict(),
            css_content,
            esm_url=esm_url,
            esm_content=esm_content,
            title=short_summary(report),
            description=json.dumps(report),
        )

        if filename:
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            with open(filename, "w", encoding="utf-8") as fh:
                fh.write(html)

        return html

    def render_report(self, *, timeout: float = 30.0, dpi: int = 72) -> RenderReport:
        """Headlessly render this widget and report on its visual health.

        Combines cheap input checks (empty/degenerate data) with a single
        headless render that measures whether the output is blank, clipped, and
        whether it is canvas/SVG/HTML. Intended for agents to self-verify a
        render without "looking" at it. Never raises -- failures populate
        ``RenderReport.error`` with ``ok=False``.

        Args:
            timeout: Seconds to wait for the headless render.
            dpi: Capture resolution (affects the blank-fraction screenshot).

        Returns:
            A ``RenderReport`` (see its fields). Requires ``playwright``;
            ``blank_fraction`` additionally requires ``Pillow`` (else None).
        """
        from pymatviz.widgets._describe import check_inputs, describe_widget
        from pymatviz.widgets._headless import RenderReport, render_diagnostics

        data = self.to_dict()
        esm_content, css_content = self._resolve_assets()
        diag = render_diagnostics(
            data, esm_content, css_content, timeout=timeout, dpi=dpi
        )
        # diag's keys (ok, error, content_type, width, height, blank_fraction,
        # is_blank, clipped) are exactly RenderReport's non-summary/warnings fields.
        return RenderReport(
            summary=describe_widget(data), warnings=tuple(check_inputs(data)), **diag
        )
