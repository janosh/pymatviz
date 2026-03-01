"""Lazy loading of MatterViz widget assets from GitHub releases."""

from __future__ import annotations

import os
import re
import subprocess
import urllib.request
from typing import Any, ClassVar
from urllib.parse import urljoin

import traitlets as tl
from anywidget import AnyWidget


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
    if version and esm_src:
        raise ValueError(
            "configure_assets() accepts either 'version' or 'esm_src'/'css_src', "
            "not both."
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
    import shutil

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
    widgets_dir = os.path.dirname(__file__)
    cmd = ["deno", "task", "build"]
    subprocess.run(cmd, cwd=widgets_dir, check=True)  # noqa: S603


class MatterVizWidget(AnyWidget):
    """Base widget class that lazily loads and caches MatterViz widget assets.

    Note:
        The marimo VS Code extension does not support virtual files, so
        widgets with large JS bundles (>5 MB) cannot render there.  Use
        ``marimo edit`` in a browser instead.
    """

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
            if not name.startswith("_")
        }
