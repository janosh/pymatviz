"""MatterVizWidget base class with image/HTML/JSON export, and loading of the
MatterViz bundle from npm, the disk cache or a local build.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import re
import shutil
import urllib.request
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urljoin

import traitlets as tl
from anywidget import AnyWidget

from pymatviz.widgets._traits import CONTROLS_MODES, optional_trait


if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal

    from pymatviz.widgets._headless import RenderReport


# Bundle matching the Python traits (test_js_prop_parity.py). "main" tracks
# matterviz main's bundle on its docs site; releases must pin a published npm
# version (fetched from jsDelivr) so installed wheels keep a fixed frontend.
MATTERVIZ_ANYWIDGET_VERSION = "0.10.0"
_ANYWIDGET_CDN = "https://cdn.jsdelivr.net/npm/matterviz-anywidget"
# matterviz's gh-pages workflow deploys main's bundle here on every push to main
_ANYWIDGET_MAIN_URL = "https://matterviz.janosh.dev/anywidget"
# expanded at call time (not import time) so tests can patch os.path.expanduser
_CACHE_ROOT = "~/.cache/pymatviz"
# first release with the split/ build, widget-model-keys.json, export_svg and the
# _synced_keys check, so older pins can't serve this package's widgets
_MIN_ANYWIDGET_VERSION = (0, 10, 0)
# a matterviz-anywidget version, or a custom (esm_src, css_src) from configure_assets
type _Bundle = str | tuple[str, str]


def _check_version(version: str) -> None:
    """Raise ValueError for a malformed or unsupported ``matterviz-anywidget`` version
    (``"main"`` or semver >= ``_MIN_ANYWIDGET_VERSION``, optional pre/build suffix).
    """
    if version == "main":
        return
    if not (match := re.match(r"^(\d+)\.(\d+)\.(\d+)([-+].+)?$", version)):
        raise ValueError(f"Invalid matterviz-anywidget version: {version=}")
    if tuple(map(int, match.groups()[:3])) < _MIN_ANYWIDGET_VERSION:
        min_version = ".".join(map(str, _MIN_ANYWIDGET_VERSION))
        raise ValueError(
            f"matterviz-anywidget@{version} is too old for this pymatviz, which "
            f"needs >= {min_version} or 'main'"
        )


def _cdn_url(version: str, filename: str = "matterviz.js") -> str:
    """Return the URL for a ``matterviz-anywidget`` bundle file.

    ``version="main"`` maps to the bundle built from matterviz main and served by
    its docs site; any other version to the jsDelivr URL of that npm release.
    Both hosts serve with CORS and a JavaScript MIME type (required for
    cross-origin ``import()``), so these URLs work in ``to_html``.
    """
    if version == "main":
        return f"{_ANYWIDGET_MAIN_URL}/{filename}"
    return f"{_ANYWIDGET_CDN}@{version}/build/{filename}"


def _bundle_source(version: str, filename: str = "matterviz.js") -> str:
    """Select the actual bundle source for both loading and standalone export.

    Raises:
        FileNotFoundError: If ``MATTERVIZ_ANYWIDGET_DIR`` is set but lacks the file
            (a typo would otherwise silently test the remote bundle).
    """
    if not (dev_dir := os.environ.get("MATTERVIZ_ANYWIDGET_DIR")):
        return _cdn_url(version, filename)
    if not os.path.isfile(path := f"{dev_dir}/{filename}"):
        raise FileNotFoundError(f"MATTERVIZ_ANYWIDGET_DIR has no {filename}: {path}")
    return path


def _bundle_sources(bundle: _Bundle) -> tuple[str, str]:
    """``(esm_src, css_src)`` the frontend loads: a custom pair as is, else a
    version's ``_bundle_source`` files, with remote JS swapped for the code-split
    entry (``build/split/``), whose chunks load per component (~1.2 MB for a 2D plot
    vs ~5.8 MB). Text reads (``_load_bundle``) use the single self-contained file.

    Raises:
        ValueError: For a version ``_check_version`` rejects.
    """
    if isinstance(bundle, tuple):
        return bundle
    _check_version(bundle)
    esm_src = _bundle_source(bundle)
    if _is_url(esm_src):
        esm_src = _cdn_url(bundle, "split/matterviz.js")
    return esm_src, _bundle_source(bundle, "matterviz.css")


def _load_bundle(bundle: _Bundle, ext: Literal["js", "css"]) -> str:
    """Text of a bundle's single-file JS or CSS, cached until configure_assets."""
    cache = MatterVizWidget._asset_cache
    if (bundle, ext) not in cache:
        cache[bundle, ext] = (
            _read_asset_source(bundle[ext == "css"])
            if isinstance(bundle, tuple)
            else fetch_widget_asset(f"matterviz.{ext}", bundle)
        )
    return cache[bundle, ext]


def _is_url(src: str) -> bool:
    """Whether an asset source is an HTTP(S) URL rather than a local path."""
    return src.startswith(("http://", "https://"))


def _cache_dir(version: str) -> str:
    """On-disk cache dir for one downloaded ``matterviz-anywidget`` version."""
    return f"{os.path.expanduser(_CACHE_ROOT)}/matterviz-anywidget/{version}"


def _in_marimo_runtime() -> bool:
    """Return whether execution is inside a live marimo runtime context."""
    try:
        from marimo._runtime.context import (  # ty: ignore[import-not-found]
            ContextNotInitializedError,
            get_context,
        )
    except ImportError:
        return False
    try:
        get_context()
    except (ContextNotInitializedError, RuntimeError):
        return False
    else:
        return True


def _marimo_esm_url(esm_text: str) -> str | None:
    """Serve ESM text as a marimo virtual file and return its absolute URL, or None
    if marimo is absent or lacks virtual files (``js()`` would then inline the
    ~6 MB bundle as a data URL).
    """
    try:
        from marimo._output.data.data import js  # ty: ignore[import-not-found]
        from marimo._runtime.context import get_context  # ty: ignore[import-not-found]
    except ImportError:
        return None
    context = get_context()
    if not context.virtual_files_supported:
        return None
    relative_url = js(esm_text).url
    if context.request is None:
        return relative_url
    base = context.request.base_url
    path = base.get("path", "/")
    return urljoin(f"{base['scheme']}://{base['netloc']}{path}", relative_url)


def _read_asset_source(src: str) -> str:
    """Read asset content from an HTTP(S) URL, ``file://`` URI, or local path."""
    if _is_url(src):
        # timeout so a stalled connection fails fast instead of hanging
        with urllib.request.urlopen(src, timeout=30) as response:  # noqa: S310
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

    Call with no arguments to reset to default (``MATTERVIZ_ANYWIDGET_VERSION``).

    Args:
        version: ``matterviz-anywidget`` npm version to fetch (e.g. ``"0.10.0"``),
            or ``"main"`` for the bundle built from matterviz main.
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
    if not version and esm_src and css_src is None:
        if not re.search(r"\.m?js$", esm_src):
            raise ValueError(
                "configure_assets() can derive 'css_src' only when 'esm_src' "
                "ends with .js or .mjs."
            )
        css_src = re.sub(r"\.m?js$", ".css", esm_src)

    if version:
        _check_version(version)
    # nothing is fetched here: widgets sync remote sources as URLs and load the
    # bundle text on first use (headless render, inline export)
    cls = MatterVizWidget
    cls._asset_cache.clear()
    cls._marimo_urls.clear()
    cls._configured = version or ((esm_src, css_src) if esm_src and css_src else None)


def fetch_widget_asset(filename: str, version_override: str | None = None) -> str:
    """Fetch a widget asset (``matterviz.js``/``matterviz.css``, or the bundle's
    ``widget-model-keys.json``/``widget-model-schema.json`` trait contract) as
    source text.

    The bundle lives in the ``matterviz-anywidget`` npm package (built from
    ``matterviz/extensions/anywidget``). Resolution order:

    1. ``MATTERVIZ_ANYWIDGET_DIR`` env var -- a local ``build/`` dir, for
       developing against an unpublished bundle.
    2. on-disk cache (``~/.cache/pymatviz/matterviz-anywidget/<version>/``).
    3. download from ``_cdn_url(version)`` (jsDelivr for npm versions), cached.

    ``version="main"`` skips the disk cache (steps 2 and 3's write): main's bundle
    changes with every matterviz deploy, so it is re-downloaded once per process
    (``MatterVizWidget`` caches assets in memory).

    Args:
        filename: A file of the bundle's ``build/`` dir, e.g. ``"matterviz.js"``.
        version_override: ``matterviz-anywidget`` npm version (or ``"main"``) to
            fetch instead of the default ``MATTERVIZ_ANYWIDGET_VERSION``.
    """
    version = version_override or MATTERVIZ_ANYWIDGET_VERSION
    _check_version(version)
    source = _bundle_source(version, filename)
    if not _is_url(source):
        return _read_asset_source(source)

    is_main = version == "main"
    cache_dir = _cache_dir(version)
    cache_path = f"{cache_dir}/{filename}"
    if not is_main and os.path.isfile(cache_path):
        return _read_asset_source(cache_path)

    try:
        content = _read_asset_source(source)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load {filename} for matterviz-anywidget@{version} from "
            f"{source}. Ensure that version is published to npm, set "
            "MATTERVIZ_ANYWIDGET_DIR to a local build dir, or check connectivity."
        ) from exc
    if is_main:  # never persist main: it changes with every matterviz deploy
        return content

    # Write to a temp file and atomically rename so an interrupted write never
    # leaves a partial file that later cache-hits and serves a broken bundle.
    os.makedirs(cache_dir, exist_ok=True)
    tmp_path = f"{cache_path}.{os.getpid()}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(content)
        os.replace(tmp_path, cache_path)
    except Exception:
        if os.path.isfile(tmp_path):
            os.unlink(tmp_path)
        raise
    return content


def clear_widget_cache(version_override: str | None = None) -> None:
    """Clear the downloaded widget-asset cache.

    Args:
        version_override: Only clear this ``matterviz-anywidget`` version's cache.
    """
    if version_override:
        version_cache_dir = _cache_dir(version_override)
        if os.path.isdir(version_cache_dir):
            shutil.rmtree(version_cache_dir)
    elif os.path.isdir(cache_root := os.path.expanduser(_CACHE_ROOT)):
        shutil.rmtree(cache_root)


def static_png_bundle(widget: MatterVizWidget, **img_kwargs: Any) -> dict[str, str]:
    """MIME bundle of the widget as a base64 PNG (img_kwargs go to to_img, e.g.
    width, dpi) plus a plain-text fallback.
    """
    png = base64.b64encode(widget.to_img(fmt="png", **img_kwargs)).decode("ascii")
    return {"image/png": png, "text/plain": f"<{type(widget).__name__}>"}


class MatterVizWidget(AnyWidget):
    """Base widget class that lazily loads and caches MatterViz widget assets.

    Note:
        The marimo VS Code extension does not support virtual files, so
        widgets with large JS bundles (>5 MB) cannot render there.  Use
        ``marimo edit`` in a browser instead.
    """

    _esm: str
    _css: str
    _bundle: _Bundle
    # what the frontend loads, see _bundle_sources
    _sources: tuple[str, str]
    # bundle text for headless rendering and inline HTML export, loaded on first use
    # and kept, so a widget outlives configure_assets() with the bundle it synced
    _asset_sources: tuple[str, str] | None
    _asset_cache: ClassVar[dict[tuple[_Bundle, str], str]] = {}
    _marimo_urls: ClassVar[dict[_Bundle, str | None]] = {}
    _configured: ClassVar[_Bundle | None] = None  # set by configure_assets
    widget_type = optional_trait(tl.Unicode)
    style = tl.Unicode(allow_none=True).tag(sync=True)
    # None keeps each component's default (hover-revealed, pinned for trajectories);
    # True/False pin controls on/off, a mode picks "always"/"hover"/"never", and a
    # {"mode", "hidden", "style"} dict also hides named controls
    show_controls = optional_trait(
        tl.Union,
        trait_types=[
            tl.Bool(),
            tl.CaselessStrEnum(CONTROLS_MODES),
            tl.Dict(
                per_key_traits={
                    "mode": tl.CaselessStrEnum(CONTROLS_MODES),
                    "hidden": tl.List(tl.Unicode()),
                    "style": tl.Unicode(),
                }
            ),
        ],
    )
    # This widget's public synced traits, which the anywidget model can't enumerate.
    # The bundle's render() checks them against what its widget spec reads, so a
    # bundle older than this package names the traits it would silently drop.
    _synced_keys = tl.List(tl.Unicode(), read_only=True).tag(sync=True)

    @tl.default("_synced_keys")
    def _default_synced_keys(self) -> list[str]:
        """Public synced trait names of this widget class."""
        return type(self).public_trait_names()

    @classmethod
    def public_trait_names(cls) -> list[str]:
        """Sorted names of the synced traits MatterViz widget classes declare, i.e.
        the props the frontend reads. Leaves out private traits and those inherited
        from anywidget/ipywidgets (``layout``, ``_model_name``, ...) unless a
        MatterViz class redeclares them (e.g. ``TrajectoryWidget.layout``).
        """
        matterviz_classes = cls.__mro__[: cls.__mro__.index(MatterVizWidget) + 1]
        return sorted(
            name
            for name in cls.class_traits(sync=True)
            if not name.startswith("_")
            and any(
                isinstance(vars(klass).get(name), tl.TraitType)
                for klass in matterviz_classes
            )
        )

    def __init__(self, version_override: str | None = None, **kwargs: Any) -> None:
        """Initialize with lazy-loaded widget assets.

        Args:
            version_override: Fetch assets for this ``matterviz-anywidget``
                version (or ``"main"``) instead of ``MATTERVIZ_ANYWIDGET_VERSION``.
                Prefer ``configure_assets()`` for global overrides.
            **kwargs: Widget traits, passed through to AnyWidget.

        Raises:
            TypeError: For kwargs that aren't traits of this widget (typos or
                unsupported options would otherwise be silently dropped).
        """
        if unknown := kwargs.keys() - self.trait_names() - {"model_id"}:
            raise TypeError(f"Unknown {type(self).__name__} options: {sorted(unknown)}")
        cls = MatterVizWidget
        self._bundle = (
            version_override or cls._configured or MATTERVIZ_ANYWIDGET_VERSION
        )
        self._sources = _bundle_sources(self._bundle)
        self._asset_sources = None
        # Remote sources sync as URLs, which anywidget imports (JS) or links (CSS):
        # construction fetches nothing (works offline) and saved notebooks carry a
        # short URL per widget, not a ~6 MB bundle copy. Local files (e.g. a
        # MATTERVIZ_ANYWIDGET_DIR build) sync as text, the browser can't load a path.
        self._esm, self._css = (
            self._sources
            if all(map(_is_url, self._sources))
            else self._resolve_assets()
        )
        # marimo serves local ESM text as a virtual file (once per bundle) instead of
        # embedding it per widget; its VS Code extension lacks virtual files
        if not _is_url(self._esm) and _in_marimo_runtime():
            if self._bundle not in cls._marimo_urls:
                cls._marimo_urls[self._bundle] = _marimo_esm_url(self._esm)
            self._esm = cls._marimo_urls[self._bundle] or self._esm

        super().__init__(**kwargs)

    def _repr_mimebundle_(
        self, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Notebook display: the live widget, or with the ``PYMATVIZ_STATIC`` env var
        set a PNG rendered in headless Chromium, for executed notebooks viewed
        without a widget frontend (GitHub, nbconvert HTML on the docs site).
        """
        if os.environ.get("PYMATVIZ_STATIC"):
            return static_png_bundle(self), {}
        return super()._repr_mimebundle_(**kwargs)

    def show(self) -> None:
        """Display this widget in notebook environments (a static PNG when
        ``PYMATVIZ_STATIC`` is set). Returns None, so as a cell's last expression
        Jupyter doesn't display a second copy.
        """
        from IPython.display import display as ipython_display

        ipython_display(self)

    def to_dict(self) -> dict[str, Any]:
        """Return all public synced traitlet values as a plain dict."""
        return {name: getattr(self, name) for name in self.public_trait_names()}

    def to_json(self, filename: str | None = None) -> str:
        """This widget's public synced traits as JSON (binary array buffers
        base64-encoded), which ``from_json`` turns back into an equal widget.

        Args:
            filename: Optional path to also write the JSON to.

        Returns:
            The JSON text.
        """
        from pymatviz.widgets._binary import json_default

        text = json.dumps(self.to_dict(), default=json_default)
        if filename:
            with open(filename, mode="w", encoding="utf-8") as file:
                file.write(text)
        return text

    def _frontend_state(self) -> dict[str, Any]:
        """``to_dict()`` plus ``_synced_keys``: the model a headless or exported
        render hands the bundle, which checks the keys like a live one.
        """
        return {**self.to_dict(), "_synced_keys": self._synced_keys}

    def to_img(
        self,
        filename: str | None = None,
        fmt: Literal["png", "jpeg", "svg", "pdf"] | None = None,
        dpi: int = 150,
        timeout: float = 30.0,
        width: int | None = None,
        height: int | None = None,
        quality: int = 90,
        *,
        transparent: bool = False,
    ) -> bytes:
        """Capture the widget as an image via headless Chromium rendering.

        Launches a headless browser (reused across calls), loads the ESM
        bundle with the widget's data, waits for the component to render,
        and captures the output. Works in plain Python scripts, CI, and
        agent workflows -- no notebook frontend required.

        Note: SVG export is only available for SVG-based (2D) widgets.
        Canvas layers inside the SVG chart (e.g. scatter markers with
        ``marker_renderer="canvas"``, the default above 1000 points) are
        embedded as PNG ``<image>`` elements; set ``marker_renderer="svg"``
        for fully vector markers. Canvas-based widgets (Structure, Trajectory,
        etc.) always export as PNG/JPEG. PDF export uses Chromium's native PDF
        generator, producing vector output with selectable text (2D canvas
        layers embedded as images); only WebGL widgets get a rasterized PDF.

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
                (~2x). Applies to PNG/JPEG and to canvas pixels in SVG/PDF
                exports; pure-vector content is unaffected.
            timeout: Seconds allowed per render step (mount, paint, capture).
                The whole call raises ``TimeoutError`` after a budget of a few
                times this (plus browser launch). Defaults to 30.
            width: Override container width in pixels for this export.
                Takes precedence over the widget's ``style``. If omitted,
                uses the widget's style or defaults to 800px.
            height: Override container height in pixels for this export.
                Takes precedence over the widget's ``style``. If omitted,
                uses the widget's style or defaults to 600px.
            quality: JPEG compression quality (1-100). Defaults to 90.
                Ignored for PNG, SVG, and PDF.
            transparent: Export PNG or SVG without backgrounds, with theme text in a
                mid gray and trimmed to the painted content, so the image reads on
                light and dark pages alike (e.g. READMEs). Defaults to False.

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

        if fmt is None and filename:
            ext = os.path.splitext(filename)[1].lower().lstrip(".")
            if not ext:
                raise ValueError(
                    f"Cannot infer format from {filename=!r} "
                    "(no extension). Pass fmt= explicitly."
                )
            fmt = {"jpg": "jpeg"}.get(ext, ext)  # ty: ignore[invalid-assignment]
        img_bytes = render_widget_headless(
            **self._render_job(
                fmt or "png", dpi, timeout, quality, width, height, transparent
            )
        )

        if filename:
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            with open(filename, "wb") as file:
                file.write(img_bytes)

        return img_bytes

    def _render_job(
        self,
        fmt: str,
        dpi: int,
        timeout: float,
        quality: int,
        width: int | None,
        height: int | None,
        transparent: bool,  # noqa: FBT001
    ) -> dict[str, Any]:
        """Checked render_widget_headless arguments for exporting this widget.

        Raises:
            ValueError: For a non-positive size, dpi or timeout, quality outside
                [1, 100] or an unsupported format.
        """
        from pymatviz.widgets._headless import _CAPTURE_FORMATS

        dims = dict(dpi=dpi, timeout=timeout, width=width, height=height)
        for name, val in dims.items():
            if val is not None and val <= 0:
                raise ValueError(f"{name} must be positive, got {val}")
        if not 1 <= quality <= 100:
            raise ValueError(f"quality must be between 1 and 100, got {quality}")
        if fmt not in _CAPTURE_FORMATS:
            raise ValueError(f"Unsupported {fmt=}, expected one of {_CAPTURE_FORMATS}")
        esm_content, css_content = self._resolve_assets()
        return dict(
            widget_data=self._frontend_state(), esm_content=esm_content,
            css_content=css_content, fmt=fmt, transparent=transparent, **dims,
            quality=quality,
        )  # fmt: skip

    def _resolve_assets(self) -> tuple[str, str]:
        """Return ``(esm_content, css_content)`` of the bundle selected at
        construction, loading it on first use.
        """
        if not self._asset_sources:
            esm_content = _load_bundle(self._bundle, "js")
            css_content = _load_bundle(self._bundle, "css")
            self._asset_sources = (esm_content, css_content)
        return self._asset_sources

    @staticmethod
    def _normalize_series(
        value: Any,
        component_name: str,
        keys: tuple[str, ...] = ("x", "y"),
        *,
        binary: bool = True,
    ) -> list[dict[str, Any]] | None:
        """normalize_plot_series for a ``series`` validator, sending large numeric
        series as binary envelopes unless ``binary=False``. Raises TraitError, which
        rolls back batched updates in hold_trait_notifications().
        """
        from pymatviz.widgets._normalize import normalize_plot_series

        try:
            return normalize_plot_series(
                value, component_name=component_name, keys=keys, binary=binary
            )
        except (TypeError, ValueError) as exc:
            raise tl.TraitError(str(exc)) from exc

    def describe(self) -> dict[str, Any]:
        """Return structured, machine-parseable facts about this widget's data.

        Pure Python (no browser): derives counts, ranges, formula, chemical
        system, etc. from the synced traitlets. Always includes ``widget_type``.
        Useful for agents to reason about output and as image alt-text.

        Examples:
            >>> import pymatviz as pmv
            >>> pmv.periodic_table(heatmap_values={"Fe": 42}).describe()["n_elements"]
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
            inline: If True, embed the ~6 MB ESM bundle in the file (opens via
                ``file://`` offline). If False (default), reference an external
                URL (small file, requires internet + a CORS-enabled JS host;
                will not load via ``file://``).
            esm_url: Explicit URL to load the bundle from (a host serving it with
                CORS and a JavaScript MIME type). Mutually exclusive with
                ``inline``. When omitted and ``inline`` is False, references the
                bundle matching the widget's active assets (the
                ``MATTERVIZ_ANYWIDGET_VERSION`` URL by default, or the version/URL
                from configure_assets / version_override); local custom assets are
                inlined instead.

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

        # reference the bundle matching the active CSS (avoid a JS/CSS version
        # mismatch); a local source has no public URL, so it is inlined instead
        if not inline and esm_url is None and _is_url(self._sources[0]):
            esm_url = self._sources[0]
        # a referenced bundle links its CSS too when that is remote; anything
        # local is inlined. build_interactive_html takes one ESM and one CSS source
        esm_content = css_content = css_url = None
        if esm_url is None:
            esm_content, css_content = self._resolve_assets()
        elif _is_url(self._sources[1]):
            css_url = self._sources[1]
        else:
            css_content = _load_bundle(self._bundle, "css")

        report = self.describe()
        html = build_interactive_html(
            self._frontend_state(),
            css_content=css_content,
            css_url=css_url,
            esm_url=esm_url,
            esm_content=esm_content,
            title=short_summary(report),
            description=json.dumps(report),
        )

        if filename:
            os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
            with open(filename, "w", encoding="utf-8") as file:
                file.write(html)

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
        # asset resolution (cache miss + offline fetch) can fail before the
        # never-raising render
        try:
            esm_content, css_content = self._resolve_assets()
            report = render_diagnostics(
                self._frontend_state(),
                esm_content,
                css_content,
                timeout=timeout,
                dpi=dpi,
            )
        except Exception as exc:  # noqa: BLE001 -- documented never-raises contract
            report = RenderReport(ok=False, error=f"render_report failed: {exc}")
        return dataclasses.replace(
            report, summary=describe_widget(data), warnings=tuple(check_inputs(data))
        )


def _revive_buffers(value: Any) -> Any:
    """Decode the base64 array buffers of binary envelopes anywhere in value."""
    from pymatviz.widgets._binary import BINARY_FORMAT

    if isinstance(value, list):
        return [_revive_buffers(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("format") == BINARY_FORMAT:
        arrays = {
            name: spec | {"data": memoryview(base64.b64decode(spec["data"]))}
            for name, spec in value["arrays"].items()
        }
        return value | {"arrays": arrays}
    return {key: _revive_buffers(val) for key, val in value.items()}


def from_json(source: str) -> MatterVizWidget:
    """Widget saved with ``MatterVizWidget.to_json``.

    Args:
        source: A path to a JSON file, or the JSON text itself.

    Returns:
        A widget of the saved type with the saved traits.

    Raises:
        ValueError: If the JSON names no known widget_type.
    """
    import pymatviz.widgets

    if os.path.isfile(source):
        with open(source, encoding="utf-8") as file:
            source = file.read()
    traits = _revive_buffers(json.loads(source))
    widget_type = traits.pop("widget_type", None)
    classes = {
        cls.class_traits()["widget_type"].default_value: cls
        for cls in vars(pymatviz.widgets).values()
        if isinstance(cls, type) and issubclass(cls, MatterVizWidget)
    }
    classes.pop(None, None)  # MatterVizWidget itself
    if (widget_cls := classes.get(widget_type)) is None:
        raise ValueError(f"Unknown {widget_type=}, expected one of {sorted(classes)}")
    return widget_cls(**traits)


def export_widgets(
    widgets: Mapping[str, MatterVizWidget],
    out_dir: str,
    *,
    fmt: Literal["png", "jpeg", "svg", "pdf"] = "png",
    dpi: int = 150,
    timeout: float = 30.0,
    width: int | None = None,
    height: int | None = None,
    quality: int = 90,
    transparent: bool = False,
) -> dict[str, str]:
    """Export widgets as ``{out_dir}/{name}.{fmt}`` images in headless Chromium,
    rendering up to 4 at once (``to_img`` renders one at a time). Options are those
    of ``MatterVizWidget.to_img``.

    Args:
        widgets: Map of file names (without extension) to widgets.
        out_dir: Directory to write the images to (created if missing).
        fmt: Image format of all files.
        dpi: Resolution (PNG/JPEG scale, SVG canvas layers).
        timeout: Seconds each widget may take to render.
        width: Container width in px for every widget.
        height: Container height in px for every widget.
        quality: JPEG quality (1-100).
        transparent: Transparent, trimmed PNG/SVG exports.

    Returns:
        Map of the widget names to the written file paths.
    """
    from pymatviz.widgets._headless import render_widgets_headless

    jobs = [
        widget._render_job(fmt, dpi, timeout, quality, width, height, transparent)
        for widget in widgets.values()
    ]
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for name, img_bytes in zip(widgets, render_widgets_headless(jobs), strict=True):
        paths[name] = f"{out_dir}/{name}.{'jpg' if fmt == 'jpeg' else fmt}"
        with open(paths[name], "wb") as file:
            file.write(img_bytes)
    return paths
