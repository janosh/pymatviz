"""Lazy loading of MatterViz widget assets from GitHub releases."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import urllib.request
from typing import TYPE_CHECKING, Any, ClassVar
from urllib.parse import urljoin

import traitlets as tl
from anywidget import AnyWidget


if TYPE_CHECKING:
    from typing import Literal


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

    # Virtual files required — without them js() creates ~14 MB data URLs
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
        base = get_context().request.base_url
    except (RuntimeError, AttributeError, TypeError):
        return relative_url
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

    Call with no arguments to reset to default (auto-detect from installed version).

    Args:
        version: GitHub release version tag (e.g. ``"v0.18.0"``).
            Mutually exclusive with ``esm_src``/``css_src``.
        esm_src: URL or local file path for the ESM JavaScript bundle.
        css_src: URL or local file path for the CSS stylesheet.
            Derived from ``esm_src`` by replacing ``.mjs`` → ``.css`` if omitted.
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
        esm_content = fetch_widget_asset("matterviz.mjs", version)
        css_content = fetch_widget_asset("matterviz.css", version)
    else:
        if css_src is None:
            css_src = re.sub(r"\.m?js$", ".css", esm_src)  # type: ignore[arg-type]
        esm_content = _read_asset_source(esm_src)  # type: ignore[arg-type]
        css_content = _read_asset_source(css_src)

    cls._asset_cache["default"] = (esm_content, css_content)
    cls._esm = esm_content
    cls._css = css_content


def fetch_widget_asset(filename: str, version_override: str | None = None) -> str:
    """Fetch a widget asset file, checking local build → cache → GitHub releases."""
    from pymatviz import __version__

    asset_version = version_override or f"v{__version__}"
    local_path = f"{os.path.dirname(__file__)}/web/build/{filename}"
    cache_dir = f"{os.path.expanduser('~/.cache/pymatviz/build')}/{asset_version}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/{filename}"

    if os.path.isfile(local_path):
        with open(local_path, encoding="utf-8") as file:
            return file.read()

    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as file:
            return file.read()

    if not re.match(r"^v\d+\.\d+\.\d+$", asset_version):
        raise ValueError(f"Invalid version format: {asset_version=}")

    github_url = f"https://github.com/janosh/pymatviz/releases/download/{asset_version}/{filename}"
    try:
        urllib.request.urlretrieve(github_url, cache_path)  # noqa: S310
        with open(cache_path, encoding="utf-8") as file:
            return file.read()
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load {filename} from GitHub releases for version "
            f"{asset_version}. Please check your internet connection."
        ) from exc


def clear_widget_cache(version_override: str | None = None) -> None:
    """Clear the widget asset cache.

    Args:
        version_override: Optional version to clear cache for specific version only
    """
    cache_dir = os.path.expanduser("~/.cache/pymatviz")
    if version_override:
        # Clear cache for specific version
        version_cache_dir = f"{cache_dir}/build/{version_override}"
        if os.path.isdir(version_cache_dir):
            shutil.rmtree(version_cache_dir)
    # Clear entire cache
    elif os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir)


def build_widget_assets() -> None:
    """Build widget assets locally for development."""
    web_dir = f"{os.path.dirname(__file__)}/web"
    if not os.path.isdir(f"{web_dir}/node_modules"):
        subprocess.run(["npm", "install"], cwd=web_dir, check=True)  # noqa: S607
    subprocess.run(["npm", "run", "build"], cwd=web_dir, check=True)  # noqa: S607


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
                fetch_widget_asset("matterviz.mjs"),
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
            self._esm = fetch_widget_asset("matterviz.mjs", version_override)
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
        avoid embedding ~10 MB of JS per widget.  In VS Code extension mode,
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
        """
        from IPython.display import display as ipython_display

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
            dpi: Resolution for PNG/JPEG output. Maps to the headless
                browser's device scale factor (72 = 1x, 144 = 2x, etc.).
                Defaults to 150 (~2x). Ignored for SVG and PDF.
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

        # We need actual ESM source code for headless rendering, not a
        # marimo virtual-file URL. version_override sets both _esm and _css
        # on the instance with real content; marimo only overrides _esm with
        # a URL. Use instance attrs only when both are present in __dict__.
        inst_esm = self.__dict__.get("_esm")
        inst_css = self.__dict__.get("_css")
        if isinstance(inst_esm, str) and isinstance(inst_css, str):
            esm_content, css_content = inst_esm, inst_css
        else:
            cls = type(self)
            cls._set_class_assets()
            esm_content, css_content = cls._asset_cache["default"]  # type: ignore[misc]

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
