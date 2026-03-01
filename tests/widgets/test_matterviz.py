"""Tests for widget asset loading functionality."""

from __future__ import annotations

import builtins
import os
import re
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from pymatviz import PKG_NAME
from pymatviz.widgets import matterviz
from pymatviz.widgets.matterviz import _in_marimo_runtime, _marimo_esm_url


if TYPE_CHECKING:
    import types
    from collections.abc import Generator
    from pathlib import Path

DOTTED_PATH = f"{PKG_NAME}.widgets.matterviz"

_real_import = builtins.__import__


def _block_marimo(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
    """Import hook that raises ImportError for any marimo submodule."""
    if name.startswith("marimo"):
        raise ImportError(f"No module named '{name}'")
    return _real_import(name, *args, **kwargs)


def _mock_marimo_context(
    *,
    get_context_side_effect: Any = None,
    get_context_return: Any = None,
) -> MagicMock:
    """Build a mock ``marimo._runtime.context`` module."""
    mock_mod = MagicMock()
    mock_mod.ContextNotInitializedError = type(
        "ContextNotInitializedError", (Exception,), {}
    )
    if get_context_side_effect is not None:
        mock_mod.get_context.side_effect = get_context_side_effect
    elif get_context_return is not None:
        mock_mod.get_context.return_value = get_context_return
    return mock_mod


def _mock_urlretrieve_side_effect(_url: str, path: str) -> None:
    """Create a file on disk to simulate ``urllib.request.urlretrieve``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="w") as file:
        file.write("downloaded content")


@pytest.fixture
def _clean_asset_cache() -> Generator[None]:
    """Save and restore MatterVizWidget class-level asset state around a test."""
    saved_esm = getattr(matterviz.MatterVizWidget, "_esm", None)
    saved_css = getattr(matterviz.MatterVizWidget, "_css", None)
    saved_cache = matterviz.MatterVizWidget._asset_cache.copy()
    matterviz.MatterVizWidget._asset_cache.clear()
    yield
    matterviz.MatterVizWidget._asset_cache = saved_cache
    if saved_esm is not None:
        matterviz.MatterVizWidget._esm = saved_esm
    if saved_css is not None:
        matterviz.MatterVizWidget._css = saved_css


# === clear_widget_cache ===


@pytest.mark.parametrize("cache_exists", [True, False])
def test_clear_widget_cache(cache_exists: bool, tmp_path: Path) -> None:
    """Clearing cache removes the directory whether or not it exists."""
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
    """Version-specific clearing removes only the targeted version directory."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    for ver in ("1.2.3", "2.0.0"):
        (build_dir / ver).mkdir()
        (build_dir / ver / "test.txt").write_text("test")

    with patch(f"{DOTTED_PATH}.os.path.expanduser", return_value=str(tmp_path)):
        matterviz.clear_widget_cache(version_override)

    if version_override:
        assert not (build_dir / version_override).exists()
        other_version = "2.0.0" if version_override == "1.2.3" else "1.2.3"
        assert (build_dir / other_version).exists()
    else:
        assert not (tmp_path / ".cache" / PKG_NAME).exists()


# === fetch_widget_asset ===


def test_fetch_widget_asset_local_file(tmp_path: Path) -> None:
    """Local dev build files take priority over cache and download."""
    web_build_dir = tmp_path / "web" / "build"
    web_build_dir.mkdir(parents=True)
    (web_build_dir / "test.mjs").write_text("local content")

    with patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)):
        assert matterviz.fetch_widget_asset("test.mjs") == "local content"


def test_fetch_widget_asset_cached_file(tmp_path: Path) -> None:
    """Cached files are returned without triggering a download."""
    cache_file = tmp_path / "v1.0.0" / "test.mjs"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("cached content")

    with (
        patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)),
        patch(f"{DOTTED_PATH}.os.path.expanduser", return_value=str(tmp_path)),
        patch(
            f"{DOTTED_PATH}.os.path.isfile",
            side_effect=lambda path: path == str(cache_file),
        ),
        patch(f"{PKG_NAME}.__version__", "1.0.0"),
        patch(f"{DOTTED_PATH}.urllib.request.urlretrieve") as mock_dl,
    ):
        assert matterviz.fetch_widget_asset("test.mjs") == "cached content"

    mock_dl.assert_not_called()


def test_fetch_widget_asset_downloads_and_caches(tmp_path: Path) -> None:
    """Downloads from GitHub releases and writes to version-specific cache."""
    with (
        patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)),
        patch(f"{DOTTED_PATH}.os.path.expanduser", return_value=str(tmp_path)),
        patch(f"{DOTTED_PATH}.os.path.isfile", return_value=False),
        patch(
            f"{DOTTED_PATH}.urllib.request.urlretrieve",
            side_effect=_mock_urlretrieve_side_effect,
        ) as mock_dl,
        patch(f"{PKG_NAME}.__version__", "1.0.0"),
    ):
        result = matterviz.fetch_widget_asset(
            "matterviz.mjs", version_override="v0.17.0"
        )

    assert result == "downloaded content"
    expected_url = (
        "https://github.com/janosh/pymatviz/releases/download/v0.17.0/matterviz.mjs"
    )
    assert mock_dl.call_args[0][0] == expected_url
    assert (tmp_path / "v0.17.0" / "matterviz.mjs").read_text() == "downloaded content"


def test_fetch_widget_asset_download_error(tmp_path: Path) -> None:
    """Network failures raise FileNotFoundError with version info."""
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


# === build_widget_assets ===


def test_build_widget_assets(tmp_path: Path) -> None:
    """Delegates to deno task build in the widgets directory."""
    with (
        patch(f"{DOTTED_PATH}.os.path.dirname", return_value=str(tmp_path)),
        patch(f"{DOTTED_PATH}.subprocess.run") as mock_run,
    ):
        matterviz.build_widget_assets()

    mock_run.assert_called_once_with(
        ["deno", "task", "build"], cwd=str(tmp_path), check=True
    )


# === MatterVizWidget.__init__ ===


@pytest.mark.usefixtures("_clean_asset_cache")
def test_lazy_matterviz_widget() -> None:
    """Default init fetches both .mjs and .css and sets class-level assets."""
    with patch(f"{DOTTED_PATH}.fetch_widget_asset", return_value="widget content"):
        widget = matterviz.MatterVizWidget()

    assert widget._esm == "widget content"
    assert widget._css == "widget content"


@pytest.mark.usefixtures("_clean_asset_cache")
def test_lazy_matterviz_widget_version_override() -> None:
    """Version override passes the version to fetch_widget_asset."""
    with patch(f"{DOTTED_PATH}.fetch_widget_asset", return_value="v2 content") as mock:
        widget = matterviz.MatterVizWidget(version_override="v2.0.0")

    assert {call.args for call in mock.call_args_list} == {
        ("matterviz.mjs", "v2.0.0"),
        ("matterviz.css", "v2.0.0"),
    }
    assert widget._esm == "v2 content"
    assert widget._css == "v2 content"


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
    init_kwargs: dict[str, Any],
    updates: dict[str, Any],
    expected_state: dict[str, Any],
) -> None:
    """to_dict exports only public synced state fields."""
    with patch(f"{DOTTED_PATH}.fetch_widget_asset", return_value="widget content"):
        widget = matterviz.MatterVizWidget(**init_kwargs)

    for key, value in updates.items():
        setattr(widget, key, value)

    state = widget.to_dict()
    assert set(state) == {"widget_type", "style", "show_controls"}
    assert "_esm" not in state
    assert "_css" not in state
    for key, value in expected_state.items():
        assert state[key] == value


# === _in_marimo_runtime ===


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        ("no_marimo", False),
        ("context_not_initialized", False),
        ("runtime_error", False),
        ("context_exists", True),
    ],
)
def test_in_marimo_runtime(setup: str, expected: bool) -> None:
    """Detects marimo runtime presence across import/context error scenarios."""
    if setup == "no_marimo":
        with patch("builtins.__import__", side_effect=_block_marimo):
            assert _in_marimo_runtime() is expected
        return

    if setup == "context_not_initialized":
        ctx_mod = _mock_marimo_context(get_context_side_effect=None)
        ctx_mod.get_context.side_effect = ctx_mod.ContextNotInitializedError
    elif setup == "runtime_error":
        ctx_mod = _mock_marimo_context(
            get_context_side_effect=RuntimeError("no context")
        )
    else:
        ctx_mod = _mock_marimo_context(get_context_return=MagicMock())

    with patch.dict("sys.modules", {"marimo._runtime.context": ctx_mod}):
        assert _in_marimo_runtime() is expected


# === _marimo_esm_url ===


_VALID_BASE_URL = {"scheme": "http", "netloc": "localhost:8080", "path": "/"}


def _mock_marimo_modules(
    *, vfile_url: str = "./@file/12345-abc.js", request: Any = "valid"
) -> dict[str, MagicMock]:
    """Build mock marimo modules for _marimo_esm_url tests."""
    mock_vfile = MagicMock()
    mock_vfile.url = vfile_url
    mock_data = MagicMock()
    mock_data.js = MagicMock(return_value=mock_vfile)

    if request == "valid":
        mock_req = MagicMock()
        mock_req.base_url = _VALID_BASE_URL
        mock_ctx = MagicMock()
        mock_ctx.request = mock_req
    else:
        mock_ctx = MagicMock()
        mock_ctx.request = request

    mock_ctx_mod = MagicMock()
    mock_ctx_mod.get_context.return_value = mock_ctx

    return {
        "marimo._output.data.data": mock_data,
        "marimo._runtime.context": mock_ctx_mod,
    }


def test_marimo_esm_url_no_marimo() -> None:
    """Returns None when marimo is not installed."""
    with patch("builtins.__import__", side_effect=_block_marimo):
        assert _marimo_esm_url("console.log('hi')") is None


def test_marimo_esm_url_full_resolution() -> None:
    """Resolves ./@file/ URL to absolute http:// using request base_url."""
    mods = _mock_marimo_modules()
    with patch.dict("sys.modules", mods):
        result = _marimo_esm_url("const x = 1;")

    assert result == "http://localhost:8080/@file/12345-abc.js"
    mods["marimo._output.data.data"].js.assert_called_once_with("const x = 1;")


@pytest.mark.parametrize(
    ("vfile_url", "request_val"),
    [
        ("data:text/javascript;base64,abc", "valid"),
        ("./@file/12345-abc.js", None),
        ("./@file/12345-abc.js", MagicMock(base_url="not-a-dict")),
        ("./@file/12345-abc.js", MagicMock(base_url={"scheme": 123, "netloc": "h"})),
        (
            "./@file/12345-abc.js",
            MagicMock(base_url={"scheme": "http", "netloc": None}),
        ),
    ],
    ids=[
        "data_url_not_virtual_file",
        "no_request",
        "base_url_not_dict",
        "scheme_not_str",
        "netloc_none",
    ],
)
def test_marimo_esm_url_returns_none(vfile_url: str, request_val: Any) -> None:
    """Returns None for non-./@file/ URLs or malformed request contexts."""
    mods = _mock_marimo_modules(vfile_url=vfile_url, request=request_val)
    with patch.dict("sys.modules", mods):
        assert _marimo_esm_url("code") is None


# === _init_marimo_assets ===


@pytest.mark.usefixtures("_clean_asset_cache")
def test_init_marimo_assets_uses_esm_url() -> None:
    """In marimo, _esm is set to absolute URL on instance, CSS stays inline."""
    with (
        patch(f"{DOTTED_PATH}.fetch_widget_asset") as mock_fetch,
        patch(f"{DOTTED_PATH}._in_marimo_runtime", return_value=True),
        patch(
            f"{DOTTED_PATH}._marimo_esm_url",
            return_value="http://localhost:2718/@file/123-abc.js",
        ),
    ):
        mock_fetch.side_effect = lambda name, *_a, **_kw: (
            "esm_content" if "mjs" in name else "css_content"
        )
        widget = matterviz.MatterVizWidget()

    assert widget._esm == "http://localhost:2718/@file/123-abc.js"
    assert widget._css == "css_content"
    assert "_esm" in widget.__dict__, "ESM should be on instance, not class"


@pytest.mark.usefixtures("_clean_asset_cache")
def test_init_marimo_assets_falls_back_on_url_failure() -> None:
    """Falls back to class-level inline assets when URL resolution fails."""
    with (
        patch(f"{DOTTED_PATH}.fetch_widget_asset", return_value="fallback_content"),
        patch(f"{DOTTED_PATH}._in_marimo_runtime", return_value=True),
        patch(f"{DOTTED_PATH}._marimo_esm_url", return_value=None),
    ):
        widget = matterviz.MatterVizWidget()

    assert widget._esm == "fallback_content"
    assert widget._css == "fallback_content"
    assert "_esm" not in widget.__dict__, "Should use class-level, not instance"
