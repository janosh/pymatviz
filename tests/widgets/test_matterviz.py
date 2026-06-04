"""Tests for widget asset loading functionality."""

from __future__ import annotations

import builtins
import os
import urllib.request
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

from pymatviz import PKG_NAME
from pymatviz.widgets import matterviz
from pymatviz.widgets.matterviz import (
    _in_marimo_runtime,
    _marimo_esm_url,
    _read_asset_source,
    configure_assets,
)


if TYPE_CHECKING:
    import types
    from collections.abc import Generator
    from pathlib import Path

DOTTED_PATH = f"{PKG_NAME}.widgets.matterviz"

_real_import = builtins.__import__


def _block_marimo(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
    """Import hook that raises ImportError for any marimo submodule."""
    if name.startswith("marimo"):
        raise ImportError(name)
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


def _fake_urlretrieve(url: str, path: str) -> None:  # noqa: ARG001
    """Write a file on disk to simulate ``urllib.request.urlretrieve``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode="w") as file:
        file.write("downloaded content")


@pytest.fixture
def _clean_asset_cache() -> Generator[None]:
    """Save and restore MatterVizWidget class-level asset state around a test."""
    cls = matterviz.MatterVizWidget
    saved = {
        attr: (hasattr(cls, attr), getattr(cls, attr, None))
        for attr in ("_esm", "_css")
    }
    saved_cache = cls._asset_cache.copy()
    cls._asset_cache.clear()
    yield
    cls._asset_cache = saved_cache
    for attr, (had, val) in saved.items():
        if had:
            assert val is not None
            setattr(cls, attr, val)
        elif hasattr(cls, attr):
            delattr(cls, attr)


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


@pytest.mark.parametrize("version_override", ["1.2.3", "2.0.0"])
def test_clear_widget_cache_version_specific(
    version_override: str, tmp_path: Path
) -> None:
    """Version-specific clearing removes only the targeted version directory."""
    base_dir = tmp_path / "matterviz-anywidget"
    base_dir.mkdir()
    for ver in ("1.2.3", "2.0.0"):
        (base_dir / ver).mkdir()
        (base_dir / ver / "test.txt").write_text("test")

    with patch(f"{DOTTED_PATH}.os.path.expanduser", return_value=str(tmp_path)):
        matterviz.clear_widget_cache(version_override)

    assert not (base_dir / version_override).exists()
    other_version = "2.0.0" if version_override == "1.2.3" else "1.2.3"
    assert (base_dir / other_version).exists()


# === default CDN URL + pinned version ===


def test_default_cdn_url() -> None:
    """Default CDN URL points at the pinned matterviz-anywidget npm version."""
    assert matterviz._default_cdn_url() == (
        "https://cdn.jsdelivr.net/npm/matterviz-anywidget@"
        f"{matterviz.MATTERVIZ_ANYWIDGET_VERSION}/build/matterviz.js"
    )


def test_pinned_anywidget_version_resolves() -> None:
    """The pinned matterviz-anywidget version is fetchable from jsDelivr.

    Network-gated: skips (never fails) when offline or not yet published, so it
    stays green pre-publish but flags a broken pin once the bundle is live.
    """
    url = matterviz._default_cdn_url()
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
            status = response.status
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"matterviz-anywidget bundle not reachable at {url}: {exc}")
    assert status == 200


@pytest.mark.parametrize("bad_version", ["not-a-version", "1.2", "v1.2.3", "1.2.x"])
def test_fetch_widget_asset_rejects_bad_version(
    bad_version: str,
    cache_at_tmp: Path,  # noqa: ARG001  -- isolates env/cache so the regex is reached
) -> None:
    """fetch_widget_asset rejects malformed (non-semver) versions."""
    with pytest.raises(ValueError, match="Invalid matterviz-anywidget version"):
        matterviz.fetch_widget_asset("matterviz.js", version_override=bad_version)


# === fetch_widget_asset ===


def test_fetch_widget_asset_dev_dir(tmp_path: Path) -> None:
    """MATTERVIZ_ANYWIDGET_DIR takes priority over cache and download."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "matterviz.js").write_text("dev content")

    with patch.dict(os.environ, {"MATTERVIZ_ANYWIDGET_DIR": str(build_dir)}):
        assert matterviz.fetch_widget_asset("matterviz.js") == "dev content"


@pytest.fixture
def cache_at_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate fetch_widget_asset: no dev-dir override, cache rooted at tmp_path."""
    monkeypatch.delenv("MATTERVIZ_ANYWIDGET_DIR", raising=False)
    monkeypatch.setattr(matterviz.os.path, "expanduser", lambda _path: str(tmp_path))
    return tmp_path


def test_fetch_widget_asset_cached_file(cache_at_tmp: Path) -> None:
    """Cached files are returned without triggering a download."""
    cache_file = cache_at_tmp / matterviz.MATTERVIZ_ANYWIDGET_VERSION / "matterviz.js"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("cached content")

    with patch(f"{DOTTED_PATH}.urllib.request.urlretrieve") as mock_dl:
        assert matterviz.fetch_widget_asset("matterviz.js") == "cached content"

    mock_dl.assert_not_called()


def test_fetch_widget_asset_downloads_and_caches(cache_at_tmp: Path) -> None:
    """Missing assets are downloaded from the jsDelivr npm CDN and cached."""
    expected_url = (
        "https://cdn.jsdelivr.net/npm/matterviz-anywidget@0.3.9/build/matterviz.js"
    )
    with patch(
        f"{DOTTED_PATH}.urllib.request.urlretrieve", side_effect=_fake_urlretrieve
    ) as mock_dl:
        result = matterviz.fetch_widget_asset("matterviz.js", version_override="0.3.9")

    assert result == "downloaded content"
    assert mock_dl.call_args.args[0] == expected_url
    assert (cache_at_tmp / "0.3.9" / "matterviz.js").read_text() == "downloaded content"


def test_fetch_widget_asset_download_error(cache_at_tmp: Path) -> None:  # noqa: ARG001
    """Network failures raise FileNotFoundError naming the package and version."""
    with (
        patch(
            f"{DOTTED_PATH}.urllib.request.urlretrieve",
            side_effect=Exception("Network error"),
        ),
        pytest.raises(FileNotFoundError, match="for matterviz-anywidget@"),
    ):
        matterviz.fetch_widget_asset("matterviz.js")


# === _read_asset_source ===


@pytest.mark.parametrize(
    ("prefix", "content"),
    [("", "export default {}"), ("file://", ".widget { color: red; }")],
    ids=["local_path", "file_uri"],
)
def test_read_asset_source_file(prefix: str, content: str, tmp_path: Path) -> None:
    """Reads content from local file paths and file:// URIs."""
    asset_file = tmp_path / "test.js"
    asset_file.write_text(content)
    assert _read_asset_source(f"{prefix}{asset_file}") == content


def test_read_asset_source_missing_file() -> None:
    """Raises FileNotFoundError for non-existent paths."""
    with pytest.raises(FileNotFoundError, match="Asset file not found"):
        _read_asset_source("/nonexistent/matterviz.js")


def test_read_asset_source_http_url() -> None:
    """Fetches content from an HTTP(S) URL."""
    with patch(f"{DOTTED_PATH}.urllib.request.urlopen") as mock_urlopen:
        mock_response = MagicMock()
        mock_response.read.return_value = b"export default {}"
        mock_response.__enter__ = lambda self: self
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = _read_asset_source("https://cdn.example.com/matterviz.js")

    assert result == "export default {}"
    mock_urlopen.assert_called_once_with("https://cdn.example.com/matterviz.js")


# === configure_assets ===


@pytest.mark.usefixtures("_clean_asset_cache")
def test_configure_assets_with_version() -> None:
    """configure_assets(version=...) fetches from jsDelivr npm CDN."""
    with patch(f"{DOTTED_PATH}.fetch_widget_asset") as mock_fetch:
        mock_fetch.side_effect = lambda name, ver: f"{name}@{ver}"
        configure_assets(version="v0.19.0")

    cls = matterviz.MatterVizWidget
    assert cls._esm == "matterviz.js@v0.19.0"
    assert cls._css == "matterviz.css@v0.19.0"
    assert "default" in cls._asset_cache


@pytest.mark.usefixtures("_clean_asset_cache")
def test_configure_assets_with_urls(tmp_path: Path) -> None:
    """configure_assets(esm_src=..., css_src=...) reads from provided sources."""
    esm_file = tmp_path / "custom.js"
    css_file = tmp_path / "custom.css"
    esm_file.write_text("custom esm")
    css_file.write_text("custom css")

    configure_assets(esm_src=str(esm_file), css_src=str(css_file))

    cls = matterviz.MatterVizWidget
    assert cls._esm == "custom esm"
    assert cls._css == "custom css"


@pytest.mark.usefixtures("_clean_asset_cache")
def test_configure_assets_css_auto_derived(tmp_path: Path) -> None:
    """CSS path is auto-derived from ESM path when css_src is omitted."""
    esm_file = tmp_path / "matterviz.js"
    css_file = tmp_path / "matterviz.css"
    esm_file.write_text("esm content")
    css_file.write_text("css content")

    configure_assets(esm_src=str(esm_file))

    cls = matterviz.MatterVizWidget
    assert cls._esm == "esm content"
    assert cls._css == "css content"


@pytest.mark.usefixtures("_clean_asset_cache")
def test_configure_assets_reset() -> None:
    """configure_assets() with no args resets to auto-detect."""
    with patch(f"{DOTTED_PATH}.fetch_widget_asset", return_value="preset"):
        configure_assets(version="v1.0.0")

    assert matterviz.MatterVizWidget._asset_cache.get("default") == (
        "preset",
        "preset",
    )

    configure_assets()

    assert matterviz.MatterVizWidget._asset_cache == {}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"version": "v1.0.0", "esm_src": "/f.js"}, "not both"),
        ({"version": "v1.0.0", "css_src": "/f.css"}, "not both"),
        ({"css_src": "/f.css"}, "requires 'esm_src'"),
    ],
)
def test_configure_assets_rejects_invalid_combos(
    kwargs: dict[str, str], match: str
) -> None:
    """configure_assets rejects invalid argument combinations."""
    with pytest.raises(ValueError, match=match):
        configure_assets(**kwargs)


@pytest.mark.usefixtures("_clean_asset_cache")
def test_configure_assets_applies_to_subsequent_widgets(tmp_path: Path) -> None:
    """Widgets created after configure_assets use the configured assets."""
    esm_file = tmp_path / "matterviz.js"
    css_file = tmp_path / "matterviz.css"
    esm_file.write_text("configured esm")
    css_file.write_text("configured css")

    configure_assets(esm_src=str(esm_file))

    widget = matterviz.MatterVizWidget(widget_type="test")
    assert widget._esm == "configured esm"
    assert widget._css == "configured css"


# === MatterVizWidget.__init__ ===


@pytest.mark.usefixtures("_clean_asset_cache")
def test_lazy_matterviz_widget() -> None:
    """Default init fetches both .js and .css and sets class-level assets."""
    with patch(f"{DOTTED_PATH}.fetch_widget_asset", return_value="widget content"):
        widget = matterviz.MatterVizWidget()

    assert widget._esm == "widget content"
    assert widget._css == "widget content"


@pytest.mark.usefixtures("_clean_asset_cache")
def test_lazy_matterviz_widget_version_override() -> None:
    """Version override passes the version to fetch_widget_asset."""
    with patch(f"{DOTTED_PATH}.fetch_widget_asset", return_value="v2 content") as mock:
        widget = matterviz.MatterVizWidget(version_override="2.0.0")

    assert {mock_call.args for mock_call in mock.call_args_list} == {
        ("matterviz.js", "2.0.0"),
        ("matterviz.css", "2.0.0"),
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
    assert {"widget_type", "style", "show_controls"} <= set(state)
    assert not any(key.startswith("_") for key in state)
    non_synced_internals = {"comm", "keys", "log"}
    assert not non_synced_internals & set(state), (
        f"Non-synced traitlets leaked into to_dict: {non_synced_internals & set(state)}"
    )
    for key, value in expected_state.items():
        assert state[key] == value


# === MatterVizWidget.show ===


@pytest.mark.parametrize("static_val", ["", None])
def test_show_interactive(static_val: str | None) -> None:
    """show() displays the widget interactively when PYMATVIZ_STATIC is unset/empty."""
    with (
        patch(f"{DOTTED_PATH}.fetch_widget_asset", return_value="content"),
        patch(f"{DOTTED_PATH}._in_marimo_runtime", return_value=False),
        patch.dict(os.environ, {}, clear=False),
        patch("IPython.display.display") as mock_display,
    ):
        os.environ.pop("PYMATVIZ_STATIC", None)
        if static_val is not None:
            os.environ["PYMATVIZ_STATIC"] = static_val
        widget = matterviz.MatterVizWidget()
        widget.show()

    mock_display.assert_called_once_with(widget)


@pytest.mark.parametrize("static_val", ["1", "0"], ids=["val_1", "val_0"])
def test_show_static_png(static_val: str) -> None:
    """show() renders static PNG when PYMATVIZ_STATIC is any non-empty string.

    Note: "0" is truthy in Python strings, so PYMATVIZ_STATIC=0 still
    enables static mode. Only unsetting or setting to "" disables it.
    """
    fake_png = b"\x89PNG fake image bytes"
    mock_image_cls = MagicMock()
    mock_image_instance = MagicMock()
    mock_image_cls.return_value = mock_image_instance

    with (
        patch(f"{DOTTED_PATH}.fetch_widget_asset", return_value="content"),
        patch(f"{DOTTED_PATH}._in_marimo_runtime", return_value=False),
        patch.dict(os.environ, {"PYMATVIZ_STATIC": static_val}, clear=False),
        patch("IPython.display.display") as mock_display,
        patch("IPython.display.Image", mock_image_cls),
    ):
        widget = matterviz.MatterVizWidget()
        with patch.object(widget, "to_img", return_value=fake_png) as mock_to_img:
            widget.show()

    mock_to_img.assert_called_once_with(fmt="png")
    mock_image_cls.assert_called_once_with(data=fake_png)
    mock_display.assert_called_once_with(mock_image_instance)


# === _in_marimo_runtime ===


def test_in_marimo_runtime_no_marimo() -> None:
    """Returns False when marimo is not installed."""
    with patch("builtins.__import__", side_effect=_block_marimo):
        assert _in_marimo_runtime() is False


@pytest.mark.parametrize(
    ("ctx_kwargs", "expected"),
    [
        ({"get_context_return": MagicMock()}, True),
        ({"get_context_side_effect": RuntimeError("no context")}, False),
    ],
    ids=["context_exists", "runtime_error"],
)
def test_in_marimo_runtime_with_context(
    ctx_kwargs: dict[str, Any], expected: bool
) -> None:
    """Detects marimo runtime presence across context error scenarios."""
    ctx_mod = _mock_marimo_context(**ctx_kwargs)
    with patch.dict("sys.modules", {"marimo._runtime.context": ctx_mod}):
        assert _in_marimo_runtime() is expected


def test_in_marimo_runtime_context_not_initialized() -> None:
    """Returns False when marimo context is not initialized."""
    ctx_mod = _mock_marimo_context()
    ctx_mod.get_context.side_effect = ctx_mod.ContextNotInitializedError
    with patch.dict("sys.modules", {"marimo._runtime.context": ctx_mod}):
        assert _in_marimo_runtime() is False


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


def test_marimo_esm_url_returns_none_without_virtual_files() -> None:
    """Returns None when virtual files are not supported (VS Code extension)."""
    mock_data = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.virtual_files_supported = False
    mock_ctx_mod = MagicMock()
    mock_ctx_mod.get_context.return_value = mock_ctx

    with patch.dict(
        "sys.modules",
        {
            "marimo._output.data.data": mock_data,
            "marimo._runtime.context": mock_ctx_mod,
        },
    ):
        assert _marimo_esm_url("code") is None

    mock_data.js.assert_not_called()


def test_marimo_esm_url_returns_none_for_non_virtual_file() -> None:
    """Returns None when js() returns a non-./@file/ URL (e.g. data URL)."""
    mods = _mock_marimo_modules(vfile_url="data:text/javascript;base64,abc")
    with patch.dict("sys.modules", mods):
        assert _marimo_esm_url("code") is None


@pytest.mark.parametrize(
    "request_val",
    [None, MagicMock(base_url="not-a-dict"), MagicMock(base_url={"scheme": 123})],
    ids=["no_request", "base_url_not_dict", "scheme_not_str"],
)
def test_marimo_esm_url_falls_back_to_relative(request_val: Any) -> None:
    """Falls back to relative ./@file/ URL when request context is unavailable."""
    mods = _mock_marimo_modules(request=request_val)
    with patch.dict("sys.modules", mods):
        assert _marimo_esm_url("code") == "./@file/12345-abc.js"


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
