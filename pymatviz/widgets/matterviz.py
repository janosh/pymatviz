"""Lazy loading of MatterViz widget assets from GitHub releases."""

from __future__ import annotations

import os
import re
import subprocess
import urllib.request
from typing import Any

from anywidget import AnyWidget


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

    def __init__(self, version_override: str | None = None, **kwargs: Any) -> None:
        """Initialize the widget with lazy loading of widget assets.

        Args:
            version_override (str | None): Override which asset versions to fetch.
                Defaults to currently installed package version and should only be
                used with good reason since different JS assets may be incompatible.
            **kwargs: Additional arguments passed to AnyWidget
        """
        self._esm = fetch_widget_asset("matterviz.mjs", version_override)
        self._css = fetch_widget_asset("matterviz.css", version_override)

        super().__init__(**kwargs)
