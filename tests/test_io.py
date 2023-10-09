from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
import pytest
from matplotlib import pyplot as plt

from pymatviz.io import df_to_pdf, save_fig


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("fig", [go.Figure(), plt.figure()])
@pytest.mark.parametrize("ext", ["html", "svelte", "png", "svg", "pdf"])
@pytest.mark.parametrize(
    "plotly_config", [None, {"showTips": True}, {"scrollZoom": True}]
)
@pytest.mark.parametrize("env_disable", [[], ["CI"]])
@patch.dict(os.environ, {"CI": "1"})
def test_save_fig(
    fig: go.Figure | plt.Figure | plt.Axes,
    ext: str,
    tmp_path: Path,
    plotly_config: dict[str, Any] | None,
    env_disable: list[str],
) -> None:
    if isinstance(fig, plt.Figure) and ext in ("svelte", "html"):
        pytest.skip("saving to Svelte file not supported for matplotlib figures")

    path = f"{tmp_path}/fig.{ext}"
    save_fig(fig, path, plotly_config=plotly_config, env_disable=env_disable)

    if any(var in os.environ for var in env_disable):
        # if CI env var is set, we should not save the figure
        assert not os.path.exists(path)
        return

    assert os.path.isfile(path)

    if ext in ("svelte", "html"):
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


def test_plotly_pdf_no_mathjax_loading(tmp_path: Path) -> None:
    # https://github.com/plotly/plotly.py/issues/3469
    PyPDF2 = pytest.importorskip("PyPDF2")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4]))
    path = f"{tmp_path}/test.pdf"
    save_fig(fig, path)

    # check PDF doesn't contain "Loading [MathJax]/extensions/MathMenu.js"
    with open(path, "rb") as f:
        pdf = PyPDF2.PdfReader(f)
        assert len(pdf.pages) == 1
        text = pdf.pages[0].extract_text()
        assert "Loading [MathJax]/extensions/MathMenu.js" not in text


# skip on windows, failing with OSError: cannot load library 'gobject-2.0-0': error 0x7e
# https://stackoverflow.com/a/69816601
@pytest.mark.skipif(sys.platform == "win32", reason="fails on Windows")
@pytest.mark.parametrize(
    "crop, size, style",
    [
        # test with cropping, default size, and no extra style
        # TODO test crop=True in CI, kept failing with FileNotFoundError: No such file
        # or directory: 'gs'. Didn't manage to install Ghostscript in test.yml.
        (False, "landscape", ""),
        # test without cropping, portrait size, and additional styles
        (False, "portrait", "body { margin: 0; padding: 1em; }"),
    ],
)
def test_df_to_pdf(crop: bool, size: str, style: str, tmp_path: Path) -> None:
    # Create a test DataFrame and Styler object
    df: pd.DataFrame = pd._testing.makeDataFrame()  # random data
    file_path = tmp_path / "test.pdf"

    # Execute the function
    df_to_pdf(df.style, file_path, crop=crop, size=size, style=style)

    # Check if the file is created
    assert file_path.is_file()

    with open(file_path, "rb") as pdf_file:
        contents = pdf_file.read()

    # TODO: Add more specific checks here, like file content validation
    assert contents[:4] == b"%PDF"

    # Test file overwrite behavior
    file_size_before = file_path.stat().st_size
    df_to_pdf(df.style, file_path, crop=crop, size=size, style=style)
    file_size_after = file_path.stat().st_size

    # file size should be the same since content is unchanged
    assert file_size_before - 10 <= file_size_after <= file_size_before + 10
