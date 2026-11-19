from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import plotly.graph_objects as go
import pytest

import pymatviz as pmv


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any


@pytest.mark.parametrize("ext", ["html", "svelte"])
@pytest.mark.parametrize(
    "plotly_config", [None, {"showTips": True}, {"scrollZoom": True}]
)
@patch.dict(os.environ, {"CI": "1"})
def test_save_fig(
    ext: str,
    tmp_path: Path,
    plotly_config: dict[str, Any] | None,
) -> None:
    """save_fig writes HTML/Svelte with correct config and structure."""
    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4])

    path = f"{tmp_path}/fig.{ext}"
    pmv.save_fig(fig, path, plotly_config=plotly_config)

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
        # injected {...$$props} precedes plotly's own div attributes
        assert html.startswith("<div {...$$props} style=")
    else:
        assert html.startswith("<div")


def test_ci_import_preserves_plotly_show() -> None:
    """Importing pymatviz in CI leaves Plotly display working for other libraries."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from plotly.graph_objects import Figure\n"
                "show = Figure.show\nimport pymatviz\nassert Figure.show is show"
            ),
        ],
        env=os.environ | {"CI": "1"},
        check=True,
    )


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
    ("save_func", "err_msg"),
    [
        (pmv.save_fig, r"Unsupported figure type.*expected plotly"),
        (pmv.io.save_and_compress_svg, "fig must be a plotly Figure"),
    ],
)
def test_save_non_figure_raises(
    tmp_path: Path, save_func: Callable[..., None], err_msg: str
) -> None:
    """Figure savers raise TypeError for non-Figure input."""
    with pytest.raises(TypeError, match=err_msg):
        save_func("not a figure", f"{tmp_path}/dummy.html")


@pytest.mark.parametrize("dtype", [float, np.float32])
def test_save_fig_prec_rounds_floats(tmp_path: Path, dtype: type) -> None:
    """Precision export handles mixed traces without mutating their coordinates."""
    fig = go.Figure()
    fig.add_scatter(
        x=np.array([1.123456789, 2.987654321], dtype=dtype),
        y=[3.111111111, 4.999999999],
    )
    fig.add_pie(values=[1, 2])
    fig.add_bar(x=["group"], y=[2**55 + 1])
    original_x = fig.data[0].x.copy()

    path = f"{tmp_path}/fig_prec.html"
    with patch.object(go.Figure, "write_html", autospec=True) as write_html:
        pmv.save_fig(fig, path, prec=3)
    exported = write_html.call_args.args[0]
    np.testing.assert_array_equal(exported.data[0].x, [1.123, 2.988])
    np.testing.assert_array_equal(exported.data[0].y, [3.111, 5.0])
    assert exported.data[2].x == ("group",)
    assert exported.data[2].y == (2**55 + 1,)
    np.testing.assert_array_equal(fig.data[0].x, original_x)
    assert fig.data[1]["values"] == (1, 2)


@pytest.mark.parametrize("full_html", [False, True])
def test_save_fig_style_param(tmp_path: Path, full_html: bool) -> None:
    """save_fig merges style into the outer plot div, also behind <html><head>."""
    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4])

    path = f"{tmp_path}/fig_style.html"
    pmv.save_fig(fig, path, style="color: red;", full_html=full_html)

    with open(path) as file:
        html = file.read()
    assert ("<html>" in html) is full_html
    div_match = re.search(r"<div\b[^>]*", html)
    assert div_match is not None
    div_tag = div_match[0]
    # one merged attribute (browsers drop duplicates), user style last so it wins
    assert div_tag.count("style=") == 1
    assert div_tag.endswith('; color: red;"')


def test_save_fig_pdf_template_and_hidden_traces(tmp_path: Path) -> None:
    """save_fig applies template and hides legendonly traces for PDF."""
    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4], visible="legendonly", name="hidden")
    fig.add_scatter(x=[1, 2], y=[5, 6], name="visible")
    orig_template = fig.layout.template

    path = f"{tmp_path}/fig.pdf"

    with patch.object(fig, "write_image") as mock_write:
        pmv.save_fig(fig, path)
        mock_write.assert_called_once_with(path)

    # template should be restored after save
    assert fig.layout.template == orig_template
    # hidden traces should be restored to legendonly
    assert fig.data[0].visible == "legendonly"


def test_save_fig_restores_pdf_mutations_on_write_error(tmp_path: Path) -> None:
    """Ensure save_fig restores layout/data mutations when write_image errors."""
    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4], visible="legendonly")
    orig_template = fig.layout.template

    with (
        patch.object(fig, "write_image", side_effect=RuntimeError("write failed")),
        pytest.raises(RuntimeError, match="write failed"),
    ):
        pmv.save_fig(fig, f"{tmp_path}/fig.pdf")

    assert fig.layout.template == orig_template
    assert fig.data[0].visible == "legendonly"


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
