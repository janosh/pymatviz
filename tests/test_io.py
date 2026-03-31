from __future__ import annotations

import os
import tempfile
import urllib.request
from typing import TYPE_CHECKING
from unittest.mock import patch

import plotly.graph_objects as go
import pytest

import pymatviz as pmv


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    import pandas as pd


@pytest.mark.parametrize("ext", ["html", "svelte"])
@pytest.mark.parametrize(
    "plotly_config", [None, {"showTips": True}, {"scrollZoom": True}]
)
def test_save_fig(
    ext: str,
    tmp_path: Path,
    plotly_config: dict[str, Any] | None,
) -> None:
    """save_fig writes HTML/Svelte with correct config and structure."""
    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4])

    path = f"{tmp_path}/fig.{ext}"
    pmv.save_fig(fig, path, plotly_config=plotly_config, env_disable=[])

    assert os.path.isfile(path), f"{path=}, {ext=}, {plotly_config=}"

    with open(path) as file:
        html = file.read()
    if plotly_config and plotly_config.get("showTips"):
        assert '"showTips": true' in html
    else:
        assert '"showTips": false' in html
    assert '"modeBarButtonsToRemove": ' in html
    assert '"displaylogo": false' in html
    if plotly_config and plotly_config.get("scrollZoom"):
        assert '"scrollZoom": true' in html

    if ext == "svelte":
        assert html.startswith("<div {...$$props}>")
    else:
        assert html.startswith("<div>")


@patch.dict(os.environ, {"CI": "1"})
def test_save_fig_env_disable(tmp_path: Path) -> None:
    """save_fig skips writing when env_disable variable is set."""
    fig = go.Figure()
    path = f"{tmp_path}/fig.html"
    pmv.save_fig(fig, path, env_disable=["CI"])
    assert not os.path.isfile(path)


def test_plotly_pdf_no_mathjax_loading(tmp_path: Path) -> None:
    # https://github.com/plotly/plotly.py/issues/3469
    PyPDF2 = pytest.importorskip("PyPDF2")  # noqa: N806

    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4])
    path = f"{tmp_path}/test.pdf"
    pmv.save_fig(fig, path)

    # check PDF doesn't contain "Loading [MathJax]/extensions/MathMenu.js"
    with open(path, mode="rb") as file:
        pdf = PyPDF2.PdfReader(file)
        assert len(pdf.pages) == 1
        text = pdf.pages[0].extract_text()
        assert "Loading [MathJax]/extensions/MathMenu.js" not in text


@pytest.mark.parametrize(
    ("pre_table", "styles", "inline_props", "styler_css"),
    [
        (None, None, "", False),
        (None, "", None, False),
        ("", "body { margin: 0; padding: 1em; }", "<table class='table'", True),
        (
            "<script>import { something } from 'some-module'\n</script><table",
            "body { margin: 0; padding: 1em; }",
            "style='width: 100%'",
            {"tb, th, td": "border: 1px solid black;"},
        ),
    ],
)
def test_df_to_html(
    tmp_path: Path,
    pre_table: str | None,
    styles: str | None,
    inline_props: str,
    styler_css: bool | dict[str, str],
    df_mixed: pd.DataFrame,
) -> None:
    file_path = tmp_path / "test_df.svelte"

    html1 = pmv.io.df_to_html(
        df_mixed.style,
        pre_table=pre_table,
        styles=styles,
        inline_props=inline_props,
        styler_css=styler_css,
    )
    assert not file_path.is_file()
    html2 = pmv.io.df_to_html(
        df_mixed.style,
        file_path=file_path,
        pre_table=pre_table,
        styles=styles,
        inline_props=inline_props,
        styler_css=styler_css,
    )
    assert html1 == html2

    assert file_path.is_file()
    html_text = file_path.read_text()
    assert html2 == html_text

    if pre_table is not None:
        assert pre_table.split("\n")[0] in html_text, html_text
    if styles is not None:
        assert f"{styles}\n</style>" in html_text
    if inline_props:
        assert inline_props in html_text

    # check file contains original dataframe value
    assert str(df_mixed.iloc[0, 0].round(6)) in html_text


def test_save_fig_type_error(tmp_path: Path) -> None:
    """save_fig raises TypeError for non-Figure input."""
    with pytest.raises(TypeError, match=r"Unsupported figure type.*expected plotly"):
        pmv.save_fig("not a figure", f"{tmp_path}/dummy.html", env_disable=[])  # type: ignore[arg-type]


def test_save_fig_prec_rounds_floats(tmp_path: Path) -> None:
    """save_fig with prec rounds float coordinates."""
    fig = go.Figure()
    fig.add_scatter(x=[1.123456789, 2.987654321], y=[3.111111111, 4.999999999])

    path = f"{tmp_path}/fig_prec.html"
    pmv.save_fig(fig, path, prec=3, env_disable=[])

    with open(path) as file:
        html = file.read()
    # rounded values should appear, originals should not
    assert "1.12" in html
    assert "1.123456789" not in html


def test_save_fig_style_param(tmp_path: Path) -> None:
    """save_fig injects style attribute into HTML div."""
    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4])

    path = f"{tmp_path}/fig_style.html"
    pmv.save_fig(fig, path, style="color: red;", env_disable=[])

    with open(path) as file:
        html = file.read()
    assert "style='color: red;'" in html


@patch("pymatviz.io.sleep")
def test_save_fig_pdf_template_and_hidden_traces(
    mock_sleep: Any, tmp_path: Path
) -> None:
    """save_fig applies template and hides legendonly traces for PDF."""
    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4], visible="legendonly", name="hidden")
    fig.add_scatter(x=[1, 2], y=[5, 6], name="visible")
    orig_template = fig.layout.template

    path = f"{tmp_path}/fig.pdf"

    with patch.object(fig, "write_image") as mock_write:
        pmv.save_fig(fig, path, env_disable=[])
        # PDF writes image twice (MathJax workaround)
        assert mock_write.call_count == 2
        mock_sleep.assert_called_once()

    # template should be restored after save
    assert fig.layout.template == orig_template
    # hidden traces should be restored to legendonly
    assert fig.data[0].visible == "legendonly"


def test_save_and_compress_svg_type_error() -> None:
    """save_and_compress_svg raises TypeError for non-Figure."""
    with pytest.raises(TypeError, match="fig must be a plotly Figure"):
        pmv.io.save_and_compress_svg("not a figure", "test")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "filename", ["my-fig", f"{tempfile.gettempdir()}/abs/path.svg"]
)
def test_save_and_compress_svg(filename: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """save_and_compress_svg resolves paths and runs svgo when available."""
    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4])

    monkeypatch.setattr("pymatviz.io.which", lambda _cmd: None)

    with patch("pymatviz.save_fig") as mock_save:
        pmv.io.save_and_compress_svg(fig, filename)
        called_path = mock_save.call_args[0][1]
        if not filename.endswith(".svg") and not os.path.isabs(filename):
            assert called_path.endswith(f"{filename}.svg")
            assert "assets/svg" in called_path
        else:
            assert called_path == filename

    monkeypatch.setattr("pymatviz.io.which", lambda _cmd: "/usr/bin/svgo")

    with (
        patch("pymatviz.save_fig"),
        patch("subprocess.run") as mock_run,
    ):
        pmv.io.save_and_compress_svg(fig, filename)
        mock_run.assert_called_once()
        assert "svgo" in mock_run.call_args[0][0][0]


def test_df_to_html_table_deprecation(df_mixed: pd.DataFrame) -> None:
    """df_to_html_table emits DeprecationWarning and delegates to df_to_html."""
    with pytest.warns(DeprecationWarning, match="df_to_html_table is deprecated"):
        result = pmv.io.df_to_html_table(df_mixed.style, inline_props="")
    assert "<table" in result


def test_df_to_html_post_process(df_mixed: pd.DataFrame) -> None:
    """df_to_html applies post_process callback to output HTML."""
    marker = "<!-- POST PROCESSED -->"

    def add_marker(html: str) -> str:
        return html + marker

    result = pmv.io.df_to_html(df_mixed.style, post_process=add_marker, inline_props="")
    assert result.endswith(marker)


def test_df_to_html_inline_props_error(df_mixed: pd.DataFrame) -> None:
    """df_to_html raises ValueError if inline_props set but no table tag found."""
    styler = df_mixed.style
    styler.set_uuid("")

    with pytest.raises(ValueError, match=r"no '<table \.\.\.' tag found"):
        # pre_table replaces <table, so inline_props has nothing to attach to
        pmv.io.df_to_html(
            styler,
            pre_table="<div",
            inline_props="class='foo'",
            styler_css=False,
        )


def test_tqdm_download(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    test_url = "https://example.com/testfile"
    test_file_path = tmp_path / "testfile"

    total_size = 1024 * 1024  # 1 MB
    block_size = 1024  # 1 KB per block

    def mock_urlretrieve(
        *args: Any,  # noqa: ARG001
        reporthook: Callable[[int, int, int | None], bool] | None = None,
    ) -> None:
        # simulate a file download (in chunks)
        n_blocks = total_size // block_size

        for block_idx in range(1, n_blocks + 1):
            if reporthook:
                reporthook(block_idx, block_size, total_size)

    # apply mock urlretrieve
    monkeypatch.setattr(urllib.request, "urlretrieve", mock_urlretrieve)

    with pmv.io.TqdmDownload(desc=test_url) as pbar:
        urllib.request.urlretrieve(test_url, test_file_path, reporthook=pbar.update_to)  # noqa: S310

    assert pbar.n == total_size
    assert pbar.total == total_size
    assert pbar.desc == test_url
    assert pbar.unit == "B"
    assert pbar.unit_scale is True
    assert pbar.unit_divisor == 1024
