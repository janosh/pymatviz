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
    """Return an absolute http(s) URL for ESM assets via marimo's virtual file system.

    anywidget only treats _esm values starting with http(s):// as URLs;
    all other strings are parsed as inline JavaScript source.  Marimo's
    virtual-file API produces relative ./@file/… paths, so we resolve
    them to absolute URLs using the runtime request context.
    """
    try:
        from marimo._output.data.data import js
        from marimo._runtime.context import get_context
    except ImportError:
        return None

    try:
        relative_url = js(esm_text).url
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None

    if not relative_url or not relative_url.startswith("./@file/"):
        return None

    try:
        request = get_context().request
    except (RuntimeError, AttributeError):
        return None

    if request is None:
        return None

    base = request.base_url
    if not isinstance(base, dict):
        return None

    scheme = base.get("scheme")
    netloc = base.get("netloc")
    path = base.get("path", "/")
    if not isinstance(scheme, str) or not isinstance(netloc, str):
        return None

    return urljoin(f"{scheme}://{netloc}{path}", relative_url)


def fetch_widget_asset(filename: str, version_override: str | None = None) -> str:
    """Get widget assets with GitHub releases fallback.

    Args:
        filename (str): Name of the asset file to fetch
        version_override (str): Override current version from package metadata

    Returns:
        str: The contents of the asset file
    """
    from pymatviz import __version__

    # fallback to installed version
    asset_version = version_override or f"v{__version__}"
    repo_url = "https://github.com/janosh/pymatviz"

    # Paths
    local_path = f"{os.path.dirname(__file__)}/web/build/{filename}"
    cache_dir = f"{os.path.expanduser('~/.cache/pymatviz/build')}/{asset_version}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/{filename}"

    # Check local development files first
    if os.path.isfile(local_path):
        with open(local_path, encoding="utf-8") as file:
            return file.read()

    # Check cache
    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as file:
            return file.read()

    if not re.match(r"^v\d+\.\d+\.\d+$", asset_version):
        raise ValueError(f"Invalid version format: {asset_version=}")

    # Download from GitHub releases
    github_url = f"{repo_url}/releases/download/{asset_version}/{filename}"
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
    """Base widget class that lazily loads and caches MatterViz widget assets."""

    _esm: str
    _css: str
    _asset_cache: ClassVar[dict[str, tuple[str, str]]] = {}
    state_fields: tuple[str, ...] = ("widget_type", "style", "show_controls")
    widget_type = tl.Unicode(allow_none=True, default_value=None).tag(sync=True)
    style = tl.Unicode(allow_none=True).tag(sync=True)
    show_controls = tl.Bool(default_value=True).tag(sync=True)

    @classmethod
    def _set_class_assets(cls) -> None:
        """Load and cache default widget assets on the class, not instances."""
        cache_key = "default"
        if cache_key not in cls._asset_cache:
            cls._asset_cache[cache_key] = (
                fetch_widget_asset("matterviz.mjs"),
                fetch_widget_asset("matterviz.css"),
            )
        cls._esm, cls._css = cls._asset_cache[cache_key]

    def __init__(self, version_override: str | None = None, **kwargs: Any) -> None:
        """Initialize the widget with lazy loading of widget assets.

        Args:
            version_override (str | None): Override which asset versions to fetch.
                Defaults to currently installed package version and should only be
                used with good reason since different JS assets may be incompatible.
            **kwargs: Additional arguments passed to AnyWidget
        """
        if version_override is not None:
            self._esm = fetch_widget_asset("matterviz.mjs", version_override)
            self._css = fetch_widget_asset("matterviz.css", version_override)
        elif _in_marimo_runtime():
            self._init_marimo_assets()
        else:
            # Set assets on the class so Jupyter/VS Code share them across
            # instances via ipywidgets comms without per-widget serialization.
            self.__class__._set_class_assets()

        super().__init__(**kwargs)

    def _init_marimo_assets(self) -> None:
        """Configure assets for marimo: ESM via absolute URL, CSS inline.

        Marimo serializes each cell output independently and has an
        output_max_bytes limit (~10 MB default).  Serving ESM via marimo's
        virtual-file system avoids embedding ~10 MB of JS per widget.
        CSS stays inline (~166 KB) to sidestep stylesheet URL loading
        issues in marimo's anywidget integration.
        """
        self.__class__._set_class_assets()
        esm_text, css_text = self.__class__._esm, self.__class__._css

        esm_url = _marimo_esm_url(esm_text)
        if esm_url:
            self._esm = esm_url
            self._css = css_text
        # Fall through: class-level inline assets already set by _set_class_assets

    def display(self) -> MatterVizWidget:
        """Display this widget in notebook environments and return itself."""
        from IPython.display import display as ipython_display

        ipython_display(self)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Return public synced widget state as a plain dictionary."""
        return {field: getattr(self, field) for field in self.state_fields}
