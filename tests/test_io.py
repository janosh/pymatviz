from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path
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


@pytest.mark.parametrize("ext", ["html", "svelte", "png", "svg", "pdf"])
@pytest.mark.parametrize(
    "plotly_config", [None, {"showTips": True}, {"scrollZoom": True}]
)
@pytest.mark.parametrize("env_disable", [[], ["CI"]])
@patch.dict(os.environ, {"CI": "1"})
def test_save_fig(
    ext: str,
    tmp_path: Path,
    plotly_config: dict[str, Any] | None,
    env_disable: list[str],
) -> None:
    if ext in ("png", "svg", "pdf"):
        pytest.skip(
            "Kaleido seems broken in CI as of 2025-05-18, skipping Plotly image "
            "export test."
        )
    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4])

    path = f"{tmp_path}/fig.{ext}"
    pmv.save_fig(fig, path, plotly_config=plotly_config, env_disable=env_disable)

    if any(var in os.environ for var in env_disable):
        # if CI env var is set, we should not save the figure
        assert not os.path.isfile(path)
        return

    assert os.path.isfile(path), f"{path=}, {ext=}, {plotly_config=}, {env_disable=}"

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
    PyPDF2 = pytest.importorskip("PyPDF2")  # noqa: N806

    fig = go.Figure()
    fig.add_scatter(x=[1, 2], y=[3, 4])
    path = f"{tmp_path}/test.pdf"
    pmv.save_fig(fig, path)

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
    ("crop", "size", "style", "styler_css"),
    [
        # test with cropping, default size, and no extra style
        # TODO test crop=True in CI, kept failing with FileNotFoundError: No such file
        # or directory: 'gs'. Didn't manage to install Ghostscript in test.yml.
        (False, "landscape", "", False),
        # test without cropping, portrait size, and additional styles
        (False, "portrait", "body { margin: 0; padding: 1em; }", True),
        (False, None, "", True),
    ],
)
def test_df_to_pdf(
    crop: bool,
    size: str,
    style: str,
    styler_css: bool,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    df_float: pd.DataFrame,
) -> None:
    try:
        import weasyprint
    except ImportError:
        weasyprint = None
    try:
        import pdfCropMargins
    except ImportError:
        pdfCropMargins = None  # noqa: N806

    file_path = tmp_path / "test_df_to.pdf"

    kwds = dict(
        styler=df_float.style,
        file_path=file_path,
        crop=crop,
        size=size,
        style=style,
        styler_css=styler_css,
    )

    # check we're raising helpful error messages on missing deps
    if weasyprint is None:
        with pytest.raises(ImportError, match="weasyprint not installed\n"):
            pmv.io.df_to_pdf(**kwds)
        return

    if pdfCropMargins is None:
        with pytest.raises(ImportError, match="cropPdfMargins not installed\n"):
            pmv.io.df_to_pdf(**kwds)
        return

    pmv.io.df_to_pdf(**kwds)

    # Check if the file is created
    assert file_path.is_file()
    # ensure the function doesn't print to stdout or stderr
    stdout, stderr = capsys.readouterr()
    assert stderr == ""
    assert stdout == ""

    with open(file_path, "rb") as pdf_file:
        contents = pdf_file.read()

    # TODO: maybe add more specific checks here, like file content validation
    assert contents[:4] == b"%PDF"

    # Test file overwrite behavior
    file_size_before = file_path.stat().st_size  # ~7000 bytes
    pmv.io.df_to_pdf(**kwds)
    file_size_after = file_path.stat().st_size  # ~7000 bytes

    # file size should be the same since content is unchanged
    assert abs(file_size_before - file_size_after) < 2000
    # file size difference strangely increased from <10 to 7354-6156=1198 on 2024-05-04


def test_normalize_and_crop_pdf(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    # patch which('gs') to return None
    monkeypatch.setattr("pymatviz.io.which", lambda _: None)

    pmv.io.normalize_and_crop_pdf("tests/test_io.py", on_gs_not_found="ignore")
    stdout, stderr = capsys.readouterr()
    assert stdout == "" == stderr

    with pytest.warns(
        UserWarning,
        match="Ghostscript not found, skipping PDF normalization and cropping",
    ):
        pmv.io.normalize_and_crop_pdf("tests/test_io.py", on_gs_not_found="warn")
    stdout, stderr = capsys.readouterr()
    assert stderr == ""

    with pytest.raises(RuntimeError, match="Ghostscript not found in PATH"):
        pmv.io.normalize_and_crop_pdf("tests/test_io.py", on_gs_not_found="error")

    # patch which('gs') to return a path


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


def test_tqdm_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
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

    stdout, stderr = capsys.readouterr()
    assert stdout == ""
    assert f"{test_url}: 0.00B [00:00, ?B/s]" in stderr


@pytest.mark.parametrize(
    ("compress", "font_size", "use_styler", "width", "height"),
    [
        # Default font size, no compression, DataFrame
        (False, 14, False, 458, 875),
        # Larger font, with compression, Styler
        (True, 18, True, 559, 1121),
        # Smaller font, no compression, Styler
        (False, 10, True, 357, 629),
    ],
)
def test_df_to_svg(
    compress: bool,
    font_size: int,
    use_styler: bool,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    df_float: pd.DataFrame,
    width: int,
    height: int,
) -> None:
    pytest.skip(
        "df_to_svg unsupported since dropping matplotlib in https://github.com/janosh/pymatviz/pull/305"
    )

    from shutil import which
    from unittest.mock import patch
    from xml.etree import ElementTree as ET

    file_path = tmp_path / "test_df_to.svg"

    obj = df_float.style if use_styler else df_float

    with patch("pymatviz.io.which") as mock_which:
        mock_which.return_value = "/path/to/svgo" if compress else None

        with patch("subprocess.run") as mock_run:
            pmv.io.df_to_svg(obj, file_path, font_size=font_size, compress=compress)

            if compress:
                mock_run.assert_called_once()
            else:
                mock_run.assert_not_called()

    # Check if the file is created
    assert file_path.is_file()

    # Ensure the function doesn't print to stdout or stderr
    stdout, stderr = capsys.readouterr()
    assert stderr == ""
    assert stdout == "" or (not compress and "svgo not found in PATH" in stdout)

    # Parse the SVG and perform some basic checks
    tree = ET.parse(file_path)  # noqa: S314
    root = tree.getroot()

    # Check that it's a valid SVG
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    # assert "xmlns" in root.attrib
    assert abs(float(root.attrib["width"].rstrip("pt")) - width) < 3
    assert abs(float(root.attrib["height"].rstrip("pt")) - height) < 3

    # Check for some content from the DataFrame
    svg_content = ET.tostring(root, encoding="unicode")
    assert svg_content.startswith("<ns0:svg xmlns")

    # Test file overwrite behavior
    file_size_before = file_path.stat().st_size
    pmv.io.df_to_svg(obj, file_path, font_size=font_size, compress=compress)
    file_size_after = file_path.stat().st_size

    # check at least 10% file size reduction from compress=True
    if compress and which("svgo"):
        assert file_size_after < file_size_before * 0.9, (
            f"{file_size_before=} {file_size_after=}"
        )
    else:
        assert file_size_before == pytest.approx(file_size_after)
