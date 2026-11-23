"""Tests for widget asset loading functionality."""

from __future__ import annotations

import base64
import builtins
import functools
import io
import os
import urllib.request
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest

import pymatviz as pmv
from pymatviz.widgets import matterviz
from pymatviz.widgets.matterviz import (
    MatterVizWidget,
    _in_marimo_runtime,
    _marimo_esm_url,
    _read_asset_source,
    configure_assets,
)


if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Generator
    from pathlib import Path

DOTTED_PATH = "pymatviz.widgets.matterviz"

_real_import = builtins.__import__


def _block_marimo(name: str, *args: Any, **kwargs: Any) -> types.ModuleType:
    """Import hook that raises ImportError for any marimo submodule."""
    if name.startswith("marimo"):
        raise ImportError(name)
    return _real_import(name, *args, **kwargs)


def _fake_urlopen(url: str, *args: Any, **kwargs: Any) -> io.BytesIO:  # noqa: ARG001
    """Return a readable context-manager response simulating ``urlopen``."""
    return io.BytesIO(b"downloaded content")


@pytest.fixture
def _clean_asset_cache(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Save and restore MatterVizWidget class-level asset state around a test, which
    starts from the default remote bundle (no MATTERVIZ_ANYWIDGET_DIR).
    """
    monkeypatch.delenv("MATTERVIZ_ANYWIDGET_DIR", raising=False)
    saved_cache = MatterVizWidget._asset_cache.copy()
    saved_urls = MatterVizWidget._marimo_urls.copy()
    saved_configured = MatterVizWidget._configured
    MatterVizWidget._asset_cache.clear()
    yield
    MatterVizWidget._asset_cache = saved_cache
    MatterVizWidget._marimo_urls = saved_urls
    MatterVizWidget._configured = saved_configured


# === clear_widget_cache ===


@pytest.mark.parametrize("cache_exists", [True, False])
def test_clear_widget_cache(cache_exists: bool, tmp_path: Path) -> None:
    """Clearing cache removes the directory whether or not it exists."""
    cache_dir = tmp_path / ".cache" / "pymatviz"
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


# === pinned version ===


@pytest.mark.parametrize(
    ("version", "expected_url"),
    [
        ("main", "https://matterviz.janosh.dev/anywidget/matterviz.css"),
        (
            "0.8.0",
            "https://cdn.jsdelivr.net/npm/matterviz-anywidget@0.8.0/build/matterviz.css",
        ),
    ],
)
def test_cdn_url(version: str, expected_url: str) -> None:
    """Version "main" maps to the matterviz docs site, npm versions to jsDelivr."""
    assert matterviz._cdn_url(version, "matterviz.css") == expected_url


@pytest.mark.parametrize("version", ["main", "0.10.0", "1.2.3-rc.1"])
def test_bundle_sources_split_remote_js(
    version: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote JS is the code-split entry (imported by URL, so its chunks resolve),
    remote CSS and local builds (synced as text) the single files.
    """
    monkeypatch.delenv("MATTERVIZ_ANYWIDGET_DIR", raising=False)
    assert matterviz._bundle_sources(version) == (
        matterviz._cdn_url(version, "split/matterviz.js"),
        matterviz._cdn_url(version, "matterviz.css"),
    )
    for ext in ("js", "css"):
        (tmp_path / f"matterviz.{ext}").write_text(f"/* {ext} */")
    monkeypatch.setenv("MATTERVIZ_ANYWIDGET_DIR", str(tmp_path))
    assert matterviz._bundle_sources(version) == (
        f"{tmp_path}/matterviz.js",
        f"{tmp_path}/matterviz.css",
    )
    # a dev dir without the bundle (typo, unbuilt) fails instead of using the CDN
    monkeypatch.setenv("MATTERVIZ_ANYWIDGET_DIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="MATTERVIZ_ANYWIDGET_DIR has no"):
        matterviz._bundle_sources(version)


@pytest.mark.parametrize(
    ("version", "match"),
    [("0.9.0", "too old"), ("0.3.9-rc.1", "too old")]
    + [(ver, "Invalid") for ver in ("v1.2.3", "not-a-version", "1.2", "1.2.x")],
)
@pytest.mark.usefixtures("cache_at_tmp")  # so fetch_widget_asset reaches the check
def test_rejects_unsupported_versions(version: str, match: str) -> None:
    """Pins predating the split build (0.10.0) or malformed ones raise in
    _bundle_sources, configure_assets, version_override and fetch_widget_asset alike.
    """
    for call in (
        matterviz._bundle_sources,
        lambda ver: configure_assets(version=ver),
        lambda ver: matterviz.MatterVizWidget(version_override=ver),
        lambda ver: matterviz.fetch_widget_asset("matterviz.js", ver),
    ):
        with pytest.raises(ValueError, match=match):
            call(version)


def test_pinned_anywidget_version_resolves() -> None:
    """The default matterviz-anywidget version's bundle URL is fetchable.

    Network-gated: skips (never fails) when offline or not yet published, so it
    stays green pre-publish but flags a broken pin once the bundle is live.
    """
    url = matterviz._cdn_url(matterviz.MATTERVIZ_ANYWIDGET_VERSION)
    try:
        with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310
            status = response.status
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"matterviz-anywidget bundle not reachable at {url}: {exc}")
    assert status == 200


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
    """Cached npm versions skip the download; main ignores the cache, re-downloads
    and never writes it (its bundle changes with every matterviz deploy).
    """
    for version in ("0.10.1", "main"):
        cache_file = cache_at_tmp / "matterviz-anywidget" / version / "matterviz.js"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text("cached content")

    with patch(f"{DOTTED_PATH}.urllib.request.urlopen") as mock_dl:
        content = matterviz.fetch_widget_asset(
            "matterviz.js", version_override="0.10.1"
        )
        assert content == "cached content"
    mock_dl.assert_not_called()

    main_dir = cache_at_tmp / "matterviz-anywidget" / "main"
    with patch(
        f"{DOTTED_PATH}.urllib.request.urlopen", side_effect=_fake_urlopen
    ) as mock_dl:
        result = matterviz.fetch_widget_asset("matterviz.js", version_override="main")
    assert result == "downloaded content"
    assert mock_dl.call_args.args[0] == matterviz._cdn_url("main")
    # stale main file neither served nor overwritten
    assert [path.name for path in main_dir.iterdir()] == ["matterviz.js"]
    assert (main_dir / "matterviz.js").read_text() == "cached content"


def test_fetch_widget_asset_downloads_and_caches(cache_at_tmp: Path) -> None:
    """Missing assets are downloaded from the jsDelivr npm CDN and cached."""
    expected_url = (
        "https://cdn.jsdelivr.net/npm/matterviz-anywidget@0.10.1/build/matterviz.js"
    )
    with patch(
        f"{DOTTED_PATH}.urllib.request.urlopen", side_effect=_fake_urlopen
    ) as mock_dl:
        result = matterviz.fetch_widget_asset("matterviz.js", version_override="0.10.1")

    assert result == "downloaded content"
    assert mock_dl.call_args.args[0] == expected_url
    cached = cache_at_tmp / "matterviz-anywidget" / "0.10.1" / "matterviz.js"
    assert cached.read_text() == "downloaded content"


def test_fetch_widget_asset_download_error(cache_at_tmp: Path) -> None:  # noqa: ARG001
    """Network failures raise FileNotFoundError naming the package and version."""
    with (
        patch(
            f"{DOTTED_PATH}.urllib.request.urlopen",
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
    """Fetches content from an HTTP(S) URL with a timeout."""
    url = "https://cdn.example.com/matterviz.js"
    with patch(
        f"{DOTTED_PATH}.urllib.request.urlopen", side_effect=_fake_urlopen
    ) as dl:
        assert _read_asset_source(url) == "downloaded content"
    dl.assert_called_once_with(url, timeout=30)


# === configure_assets ===


@pytest.mark.usefixtures("_clean_asset_cache")
def test_configure_assets_with_version() -> None:
    """configure_assets(version=...) fetches nothing: widgets sync the version's
    jsDelivr URLs and load its text on first use.
    """
    with patch(f"{DOTTED_PATH}.fetch_widget_asset") as mock_fetch:
        mock_fetch.side_effect = lambda name, ver: f"{name}@{ver}"
        configure_assets(version="0.19.0")
        widget = matterviz.MatterVizWidget()
        mock_fetch.assert_not_called()
        assets = widget._resolve_assets()

    cdn = f"{matterviz._ANYWIDGET_CDN}@0.19.0/build"
    # the frontend imports the CDN bundle, headless renders get its text
    assert (widget._esm, widget._css) == (
        f"{cdn}/split/matterviz.js",
        f"{cdn}/matterviz.css",
    )
    assert assets == ("matterviz.js@0.19.0", "matterviz.css@0.19.0")


@pytest.mark.usefixtures("_clean_asset_cache")
def test_to_html_references_configured_version() -> None:
    """to_html(inline=False) points JS and CSS at the configured version's CDN
    URLs and downloads neither.

    Guards against pairing the pinned-default JS with custom-version CSS.
    """
    with patch(f"{DOTTED_PATH}.fetch_widget_asset") as mock_fetch:
        mock_fetch.side_effect = lambda name, ver: f".mv {{}} /* {name}@{ver} */"
        configure_assets(version="0.10.6")
        html = matterviz.MatterVizWidget().to_html(inline=False)

    mock_fetch.assert_not_called()
    assert "matterviz-anywidget@0.10.6/build/split/matterviz.js" in html  # configured
    assert f"@{matterviz.MATTERVIZ_ANYWIDGET_VERSION}/" not in html  # not pinned
    assert (
        'href="https://cdn.jsdelivr.net/npm/matterviz-anywidget@0.10.6/build/matterviz.css"'
        in html
    )


@pytest.mark.parametrize("explicit_css", [True, False], ids=["css_src", "derived"])
@pytest.mark.usefixtures("_clean_asset_cache")
def test_configure_assets_from_sources(tmp_path: Path, explicit_css: bool) -> None:
    """configure_assets(esm_src=..., css_src=...) reads from provided sources (CSS
    path derived from the ESM path when omitted) and applies to the class and to
    widgets created afterwards.
    """
    esm_file = tmp_path / "matterviz.js"
    css_file = tmp_path / ("custom.css" if explicit_css else "matterviz.css")
    esm_file.write_text("configured esm")
    css_file.write_text("configured css")

    css_kwargs = {"css_src": str(css_file)} if explicit_css else {}
    configure_assets(esm_src=str(esm_file), **css_kwargs)

    widget = matterviz.MatterVizWidget(widget_type="test")
    # local files sync as text: the browser can't load a local path
    assert (widget._esm, widget._css) == ("configured esm", "configured css")
    assert widget._resolve_assets() == ("configured esm", "configured css")


@pytest.mark.usefixtures("_clean_asset_cache")
def test_local_bundle_survives_reconfiguration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing widgets export and sync with the local bundle they loaded."""
    from pymatviz.widgets import TrajectoryWidget

    for ext in ("js", "css"):
        (tmp_path / f"matterviz.{ext}").write_text(f"/* local {ext} */")
    monkeypatch.setenv("MATTERVIZ_ANYWIDGET_DIR", str(tmp_path))
    widget = TrajectoryWidget()
    configure_assets()
    monkeypatch.delenv("MATTERVIZ_ANYWIDGET_DIR")
    widget.current_step_idx = 1
    html = widget.to_html()
    assert "const esm_url = URL.createObjectURL" in html
    assert "/* local css */" in html
    assert widget.get_state("_esm") == {"_esm": "/* local js */"}


@pytest.mark.usefixtures("_clean_asset_cache")
def test_default_assets_sync_as_urls() -> None:
    """Remote bundles sync as URLs: construction fetches nothing (works offline),
    saved widget state holds a short URL per widget instead of the bundle text, and
    the text loads on the first headless render or inline export.
    """
    import ipywidgets

    import pymatviz as pmv

    offline = FileNotFoundError("offline")
    with patch(f"{DOTTED_PATH}.fetch_widget_asset", side_effect=offline) as fetch:
        widgets = [pmv.chem_sys_hierarchy(["Fe-O"], kind="treemap") for _ in range(3)]
        state = ipywidgets.Widget.get_manager_state(widgets=widgets)["state"]
    fetch.assert_not_called()
    assert widgets[0].data[0]["children"][0]["label"] == "Fe-O"
    sources = matterviz._bundle_sources(matterviz.MATTERVIZ_ANYWIDGET_VERSION)
    for widget in widgets:
        model_state = state[widget.model_id]["state"]
        assert (model_state["_esm"], model_state["_css"]) == sources

    def fake_fetch(name: str, _version: str | None = None) -> str:
        """Return stub bundle sources."""
        return "export default { render() {} }" if name.endswith(".js") else "/*c*/"

    with patch(f"{DOTTED_PATH}.fetch_widget_asset", side_effect=fake_fetch):
        html = widgets[0].to_html(inline=True)
    assert "/*c*/" in html
    assert widgets[0]._resolve_assets() == (fake_fetch("x.js"), "/*c*/")
    assert (widgets[0]._esm, widgets[0]._css) == sources  # synced URLs unchanged


@pytest.mark.usefixtures("_clean_asset_cache")
def test_configure_assets_reset() -> None:
    """configure_assets() with no args resets to the default bundle."""
    with patch(f"{DOTTED_PATH}.fetch_widget_asset", return_value="preset"):
        configure_assets(version="1.0.0")
        assert matterviz.MatterVizWidget()._resolve_assets() == ("preset", "preset")

    configure_assets()

    assert (MatterVizWidget._asset_cache, MatterVizWidget._configured) == ({}, None)
    pinned_esm = matterviz._bundle_sources(matterviz.MATTERVIZ_ANYWIDGET_VERSION)[0]
    assert MatterVizWidget()._esm == pinned_esm


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"version": "1.0.0", "esm_src": "/f.js"}, "not both"),
        ({"version": "1.0.0", "css_src": "/f.css"}, "not both"),
        ({"css_src": "/f.css"}, "requires 'esm_src'"),
        ({"esm_src": "/bundle"}, "derive 'css_src'"),
    ],
)
def test_configure_assets_rejects_invalid_combos(
    kwargs: dict[str, str], match: str
) -> None:
    """configure_assets rejects invalid argument combinations."""
    with pytest.raises(ValueError, match=match):
        configure_assets(**kwargs)


# === MatterVizWidget.__init__ ===


@pytest.mark.usefixtures("_clean_asset_cache")
def test_lazy_matterviz_widget_version_override() -> None:
    """Version override syncs that version's URLs and loads its text on first use,
    once per process for all widgets of that version.
    """
    with patch(f"{DOTTED_PATH}.fetch_widget_asset", return_value="v2 content") as mock:
        widget = matterviz.MatterVizWidget(version_override="2.0.0")
        mock.assert_not_called()
        for _ in range(2):
            assert widget._resolve_assets() == ("v2 content", "v2 content")
            widget = matterviz.MatterVizWidget(version_override="2.0.0")

    assert [mock_call.args for mock_call in mock.call_args_list] == [
        ("matterviz.js", "2.0.0"),
        ("matterviz.css", "2.0.0"),
    ]
    assert (widget._esm, widget._css) == matterviz._bundle_sources("2.0.0")


@pytest.mark.parametrize("set_after_init", [False, True])
@pytest.mark.usefixtures("_clean_asset_cache")
def test_matterviz_widget_to_dict(set_after_init: bool) -> None:
    """to_dict exports exactly the public synced traits, set at or after init."""
    values: dict[str, Any] = {
        "widget_type": "xrd",
        "style": "height: 300px",
        "show_controls": False,
    }
    widget = matterviz.MatterVizWidget(**({} if set_after_init else values))
    for key, value in values.items() if set_after_init else ():
        setattr(widget, key, value)
    assert widget.to_dict() == values


# === MatterVizWidget.show ===


@pytest.mark.parametrize("static_val", ["", None])
@pytest.mark.usefixtures("_clean_asset_cache")
def test_show_interactive(static_val: str | None) -> None:
    """show() displays the widget interactively when PYMATVIZ_STATIC is unset/empty."""
    with (
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
@pytest.mark.usefixtures("_clean_asset_cache")
def test_static_png_repr(static_val: str) -> None:
    """With PYMATVIZ_STATIC set to any non-empty string ("0" too), notebooks display
    widgets (bare expressions and show() alike) as a PNG instead of live state.
    """
    fake_png = b"\x89PNG fake image bytes"
    with patch.dict(os.environ, {"PYMATVIZ_STATIC": static_val}, clear=False):
        widget = matterviz.MatterVizWidget()
        with patch.object(widget, "to_img", return_value=fake_png) as mock_to_img:
            bundle = widget._repr_mimebundle_()

    mock_to_img.assert_called_once_with(fmt="png")
    assert bundle == (
        {
            "image/png": base64.b64encode(fake_png).decode("ascii"),
            "text/plain": "<MatterVizWidget>",
        },
        {},
    )


# === _in_marimo_runtime / _marimo_esm_url ===


@pytest.mark.parametrize(
    ("marimo_helper", "expected"),
    [
        (_in_marimo_runtime, False),
        (functools.partial(_marimo_esm_url, "console.log('hi')"), None),
    ],
)
def test_marimo_helpers_without_marimo(
    marimo_helper: Callable[[], Any], expected: Any
) -> None:
    """No marimo installed: no runtime detected and no virtual-file ESM URL."""
    with patch("builtins.__import__", side_effect=_block_marimo):
        assert marimo_helper() is expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [(None, True), ("RuntimeError", False), ("ContextNotInitializedError", False)],
)
def test_in_marimo_runtime_with_context(error: str | None, expected: bool) -> None:
    """Detects a live marimo runtime, False when getting its context fails."""
    ctx_mod = MagicMock()
    ctx_mod.ContextNotInitializedError = type(
        "ContextNotInitializedError", (Exception,), {}
    )
    errors = {"RuntimeError": RuntimeError("no context")}
    errors["ContextNotInitializedError"] = ctx_mod.ContextNotInitializedError()
    ctx_mod.get_context.side_effect = errors.get(error)
    with patch.dict("sys.modules", {"marimo._runtime.context": ctx_mod}):
        assert _in_marimo_runtime() is expected


@pytest.mark.parametrize(
    ("has_request", "virtual_files", "expected"),
    [
        # ./@file/ URL resolved against the request's base_url
        (True, True, "http://localhost:8080/@file/12345-abc.js"),
        (False, True, "./@file/12345-abc.js"),  # no request context: relative URL
        (True, False, None),  # no virtual files (VS Code extension)
    ],
)
def test_marimo_esm_url(
    has_request: bool, virtual_files: bool, expected: str | None
) -> None:
    """Serves the ESM as a marimo virtual file and returns its (absolute) URL."""
    data_mod, ctx_mod = MagicMock(), MagicMock()
    data_mod.js.return_value.url = "./@file/12345-abc.js"
    context = ctx_mod.get_context.return_value
    context.virtual_files_supported = virtual_files
    request = MagicMock()
    request.base_url = {"scheme": "http", "netloc": "localhost:8080", "path": "/"}
    context.request = request if has_request else None
    mods = {"marimo._output.data.data": data_mod, "marimo._runtime.context": ctx_mod}
    with patch.dict("sys.modules", mods):
        assert _marimo_esm_url("const x = 1;") == expected
    assert data_mod.js.call_count == virtual_files


# === _init_marimo_assets ===


@pytest.mark.parametrize(
    ("local", "virtual_url", "expected_esm"),
    [
        (True, "http://localhost:2718/@file/123-abc.js", "virtual"),
        (True, None, "text"),  # no virtual files: inline text
        (False, "http://localhost:2718/@file/123-abc.js", "remote"),
    ],
)
@pytest.mark.usefixtures("_clean_asset_cache")
def test_init_marimo_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local: bool,
    virtual_url: str | None,
    expected_esm: str,
) -> None:
    """In marimo, a local bundle's ESM is served as a virtual file (inline when that
    fails) with CSS inline; a remote bundle keeps its URLs.
    """
    if local:
        for ext in ("js", "css"):
            (tmp_path / f"matterviz.{ext}").write_text(f"/* {ext} */")
        monkeypatch.setenv("MATTERVIZ_ANYWIDGET_DIR", str(tmp_path))
    with (
        patch(f"{DOTTED_PATH}._in_marimo_runtime", return_value=True),
        patch(f"{DOTTED_PATH}._marimo_esm_url", return_value=virtual_url) as to_url,
    ):
        widget = matterviz.MatterVizWidget()

    remote_js, remote_css = matterviz._bundle_sources(
        matterviz.MATTERVIZ_ANYWIDGET_VERSION
    )
    expected = {"virtual": virtual_url, "text": "/* js */", "remote": remote_js}
    assert widget._esm == expected[expected_esm]
    assert widget._css == ("/* css */" if local else remote_css)
    assert to_url.call_count == local


def _widget_type_bytes(jobs: list[dict[str, Any]]) -> list[bytes]:
    """Fake render_widgets_headless: each job's widget type as its image bytes."""
    return [job["widget_data"]["widget_type"].encode() for job in jobs]


@pytest.mark.parametrize(
    ("fmt", "ext"), [("png", "png"), ("jpeg", "jpg"), ("svg", "svg")]
)
def test_export_widgets(fmt: str, ext: str, tmp_path: Any) -> None:
    """export_widgets renders all widgets in one batch and writes {name}.{ext} files,
    passing each widget's state and the shared options.
    """
    widgets = {"a": pmv.bar_plot([{"x": [1], "y": [2]}]), "b": pmv.structure()}
    out_dir = f"{tmp_path}/out"
    with (
        patch.object(MatterVizWidget, "_resolve_assets", return_value=("esm", "css")),
        patch(
            "pymatviz.widgets._headless.render_widgets_headless",
            side_effect=_widget_type_bytes,
        ) as render,
    ):
        paths = pmv.export_widgets(widgets, out_dir, fmt=fmt, dpi=300)  # ty: ignore[invalid-argument-type]
        # names that would escape out_dir or need a missing subdirectory fail before
        # rendering (writing used to fail after rendering all, or land elsewhere)
        for bad_name in ("../escaped", "sub/name", "", "..", "."):
            with pytest.raises(ValueError, match=r"must be plain file names, got \["):
                pmv.export_widgets(
                    {"ok": widgets["a"], bad_name: widgets["b"]}, out_dir
                )
        # names differing only in case overwrote each other on case-insensitive
        # filesystems (macOS, Windows)
        with pytest.raises(ValueError, match=r"more than case, got \[\['A', 'a'\]\]"):
            pmv.export_widgets(
                {"A": widgets["a"], "b": widgets["b"], "a": widgets["b"]}, out_dir
            )
        # macOS also folds full case mappings (ß as SS) and Unicode normalization
        # forms (precomposed é as e + combining acute), which used to pass the check
        for name_pair in (("straße", "STRASSE"), ("café", "café")):
            with pytest.raises(ValueError, match="must differ in more than case"):
                pmv.export_widgets(dict.fromkeys(name_pair, widgets["a"]), out_dir)
    assert render.call_count == 1
    assert not os.path.isfile(f"{tmp_path}/escaped.png")
    assert paths == {name: f"{out_dir}/{name}.{ext}" for name in widgets}
    for path, widget_type in zip(
        paths.values(), ("bar_plot", "structure"), strict=True
    ):
        with open(path, "rb") as file:
            assert file.read() == widget_type.encode()
    ((jobs,), _) = render.call_args
    options = {(job["fmt"], job["dpi"], job["esm_content"]) for job in jobs}
    assert options == {(fmt, 300, "esm")}
    with pytest.raises(ValueError, match="dpi must be positive"):
        pmv.export_widgets(widgets, out_dir, dpi=0)
