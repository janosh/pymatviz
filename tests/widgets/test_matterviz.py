"""Tests for widget asset loading functionality."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from pymatviz import PKG_NAME
from pymatviz.widgets import matterviz


if TYPE_CHECKING:
    from pathlib import Path

DOTTED_PATH = f"{PKG_NAME}.widgets.matterviz"


@pytest.mark.parametrize("cache_exists", [True, False])
def test_clear_widget_cache(cache_exists: bool, tmp_path: Path) -> None:
    """Test clearing widget cache."""
    cache_dir = tmp_path / ".cache" / PKG_NAME
    if cache_exists:
        cache_dir.mkdir(parents=True)
        (cache_dir / "test.txt").write_text("test")

    with patch(f"{DOTTED_PATH}.os.path.expanduser", return_value=str(tmp_path)):
        matterviz.clear_widget_cache()

    assert not cache_dir.exists()


@pytest.mark.parametrize("version_override", [None, "1.2.3", "2.0.0"])
def test_clear_widget_cache_version_specific(
    version_override: str | None, tmp_path: Path
) -> None:
    """Test clearing widget cache for specific versions."""
    # Create the correct path structure that the function expects
    build_dir = tmp_path / "build"
    build_dir.mkdir()

    # Create files for different versions
    (build_dir / "1.2.3").mkdir()
    (build_dir / "1.2.3" / "test.txt").write_text("test")
    (build_dir / "2.0.0").mkdir()
    (build_dir / "2.0.0" / "test.txt").write_text("test")

    with patch(f"{DOTTED_PATH}.os.path.expanduser", return_value=str(tmp_path)):
        matterviz.clear_widget_cache(version_override)

    if version_override:
        # Only the specific version should be deleted
        assert not (build_dir / version_override).exists()
        # Other versions should remain
        other_version = "2.0.0" if version_override == "1.2.3" else "1.2.3"
        assert (build_dir / other_version).exists()
    else:
        # Entire cache should be deleted
        assert not (tmp_path / ".cache" / PKG_NAME).exists()


def test_fetch_widget_asset_local_file(tmp_path: Path) -> None:
    """Test fetching widget asset from local file."""
    # Create the local file in the correct path structure
    web_build_dir = tmp_path / "web" / "build"
    web_build_dir.mkdir(parents=True)
    local_file = web_build_dir / "test.mjs"
    local_file.write_text("local content")

    with patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)):
        result = matterviz.fetch_widget_asset("test.mjs")

    assert result == "local content"


def test_fetch_widget_asset_cached_file(tmp_path: Path) -> None:
    """Test fetching widget asset from cache."""
    # Create cache file in the correct path structure that the function constructs
    cache_file = tmp_path / "v1.0.0" / "test.mjs"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("cached content")

    with (
        patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)),
        patch(
            f"{DOTTED_PATH}.os.path.expanduser",
            return_value=str(tmp_path),
        ),
        patch(f"{DOTTED_PATH}.os.path.isfile") as mock_isfile,
    ):
        # Return False for local file, True for cache file
        def isfile_side_effect(path: str) -> bool:
            # Return True only for the cache file path
            return path == str(cache_file)

        mock_isfile.side_effect = isfile_side_effect
        with (
            patch(f"{PKG_NAME}.__version__", "1.0.0"),
            patch(f"{DOTTED_PATH}.urllib.request.urlretrieve") as mock_urlretrieve,
        ):
            result = matterviz.fetch_widget_asset("test.mjs")

    assert result == "cached content"
    # Verify urlretrieve was not called since we found the cached file
    mock_urlretrieve.assert_not_called()


def test_fetch_widget_asset_version_specific_caching(tmp_path: Path) -> None:
    """Test that version override creates version-specific cache."""
    with (
        patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)),
        patch(
            f"{DOTTED_PATH}.os.path.expanduser",
            return_value=str(tmp_path),
        ),
        patch(f"{DOTTED_PATH}.os.path.isfile", return_value=False),
        patch(f"{DOTTED_PATH}.urllib.request.urlretrieve") as mock_urlretrieve,
    ):

        def mock_urlretrieve_side_effect(_url: str, path: str) -> None:
            # Create the file that would be downloaded
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, mode="w") as file:
                file.write("downloaded content")

        mock_urlretrieve.side_effect = mock_urlretrieve_side_effect
        with patch(f"{PKG_NAME}.__version__", "1.0.0"):
            result = matterviz.fetch_widget_asset("test.mjs", version_override="v2.0.0")

    assert result == "downloaded content"
    # Verify the file was created in the version-specific cache
    cache_file = tmp_path / "v2.0.0" / "test.mjs"
    assert cache_file.exists()
    assert cache_file.read_text() == "downloaded content"


def test_fetch_widget_asset_downloads_correct_version(tmp_path: Path) -> None:
    """Test that the correct version is used for downloading."""
    with (
        patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)),
        patch(f"{DOTTED_PATH}.os.path.expanduser", return_value=str(tmp_path)),
        patch(f"{DOTTED_PATH}.os.path.isfile", return_value=False),
        patch(f"{DOTTED_PATH}.urllib.request.urlretrieve") as mock_urlretrieve,
    ):

        def mock_urlretrieve_side_effect(_url: str, path: str) -> None:
            # Create the file that would be downloaded

            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, mode="w") as file:
                file.write("downloaded content")

        mock_urlretrieve.side_effect = mock_urlretrieve_side_effect
        with patch(f"{PKG_NAME}.__version__", "1.0.0"):
            matterviz.fetch_widget_asset("matterviz.mjs", version_override="v0.17.0")

    # Verify the correct URL was called
    mock_urlretrieve.assert_called_once()
    call_args = mock_urlretrieve.call_args[0]
    expected_url = (
        "https://github.com/janosh/pymatviz/releases/download/v0.17.0/matterviz.mjs"
    )
    assert call_args[0] == expected_url


def test_fetch_widget_asset_creates_version_specific_cache(tmp_path: Path) -> None:
    """Test that version override creates the correct cache directory structure."""
    with (
        patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)),
        patch(
            f"{DOTTED_PATH}.os.path.expanduser",
            return_value=f"{tmp_path}/.cache/{PKG_NAME}/build",
        ),
        patch(f"{DOTTED_PATH}.os.path.isfile", return_value=False),
        patch(f"{DOTTED_PATH}.urllib.request.urlretrieve") as mock_urlretrieve,
    ):

        def mock_urlretrieve_side_effect(_url: str, path: str) -> None:
            # Create the file that would be downloaded
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, mode="w") as file:
                file.write("downloaded content")

        mock_urlretrieve.side_effect = mock_urlretrieve_side_effect
        with patch(f"{PKG_NAME}.__version__", "1.0.0"):
            matterviz.fetch_widget_asset("test.mjs", version_override="v3.0.0")

        # Verify the version-specific cache directory was created
        # The function constructs: {expanduser('~/.cache/pymatviz/build')}/{version}
        # With our mock: {tmp_path}/.cache/pymatviz/build/{version}
        cache_dir = tmp_path / ".cache" / PKG_NAME / "build" / "v3.0.0"
        assert cache_dir.exists()
        # Verify the file was created in the cache
        cache_file = cache_dir / "test.mjs"
        assert cache_file.exists()
        assert cache_file.read_text() == "downloaded content"


def test_fetch_widget_asset_download_error(tmp_path: Path) -> None:
    """Test handling of download errors."""
    err_msg = re.escape(
        "Could not load test.mjs from GitHub releases for version v1.0.0"
    )
    with (
        patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)),
        patch(f"{DOTTED_PATH}.os.path.expanduser", return_value=str(tmp_path)),
        patch(f"{DOTTED_PATH}.os.path.isfile", return_value=False),
        patch(
            f"{DOTTED_PATH}.urllib.request.urlretrieve",
            side_effect=Exception("Network error"),
        ),
        patch(f"{PKG_NAME}.__version__", "1.0.0"),
        pytest.raises(FileNotFoundError, match=err_msg),
    ):
        matterviz.fetch_widget_asset("test.mjs")


def test_build_widget_assets(tmp_path: Path) -> None:
    """Test building widget assets."""
    with (
        patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)),
        patch(f"{DOTTED_PATH}.subprocess.run") as mock_run,
    ):
        matterviz.build_widget_assets()

    mock_run.assert_called_once_with(
        ["deno", "task", "build"], cwd=str(tmp_path), check=True
    )


def test_lazy_matterviz_widget(tmp_path: Path) -> None:
    """Test lazy loading of MatterViz widget."""
    with patch(f"{DOTTED_PATH}.fetch_widget_asset") as mock_fetch:
        mock_fetch.return_value = "widget content"
        with patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)):
            widget = matterviz.MatterVizWidget()

    assert mock_fetch.call_count == 2  # Called for both .mjs and .css files
    assert widget._esm == "widget content"
    assert widget._css == "widget content"


def test_lazy_matterviz_widget_version_override(tmp_path: Path) -> None:
    """Test lazy loading of MatterViz widget with version override."""
    with patch(f"{DOTTED_PATH}.fetch_widget_asset") as mock_fetch:
        mock_fetch.return_value = "widget content"
        with patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)):
            widget = matterviz.MatterVizWidget(version_override="v2.0.0")

    # Verify fetch_widget_asset was called with version_override
    calls = mock_fetch.call_args_list
    assert len(calls) == 2
    for call in calls:
        # Check that version_override is passed as a positional argument
        assert len(call.args) == 2
        assert call.args[0] in ["matterviz.mjs", "matterviz.css"]
        assert call.args[1] == "v2.0.0"

    assert widget._esm == "widget content"
    assert widget._css == "widget content"


@pytest.mark.parametrize(
    ("init_kwargs", "updates", "expected_state"),
    [
        (
            {
                "widget_type": "composition",
                "style": "width: 400px",
                "show_controls": False,
            },
            {},
            {
                "widget_type": "composition",
                "style": "width: 400px",
                "show_controls": False,
            },
        ),
        (
            {},
            {"widget_type": "xrd", "style": "height: 300px", "show_controls": False},
            {"widget_type": "xrd", "style": "height: 300px", "show_controls": False},
        ),
    ],
)
def test_matterviz_widget_to_dict(
    tmp_path: Path,
    init_kwargs: dict[str, Any],
    updates: dict[str, Any],
    expected_state: dict[str, Any],
) -> None:
    """Test to_dict exports public synced state and reflects updates."""
    with (
        patch(f"{DOTTED_PATH}.fetch_widget_asset", return_value="widget content"),
        patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)),
    ):
        widget = matterviz.MatterVizWidget(**init_kwargs)

    for key, value in updates.items():
        setattr(widget, key, value)

    state = widget.to_dict()
    assert set(state) == {"widget_type", "style", "show_controls"}
    assert "_esm" not in state
    assert "_css" not in state
    for key, value in expected_state.items():
        assert state[key] == value
