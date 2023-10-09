from __future__ import annotations

import os
import subprocess
from os.path import dirname
from shutil import which
from time import sleep
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import plotly.graph_objects as go


if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from pandas.io.formats.style import Styler

ROOT = dirname(dirname(__file__))


def save_fig(
    fig: go.Figure | plt.Figure | plt.Axes,
    path: str,
    plotly_config: dict[str, Any] | None = None,
    env_disable: Sequence[str] = ("CI",),
    pdf_sleep: float = 0.6,
    style: str = "",
    **kwargs: Any,
) -> None:
    """Write a plotly or matplotlib figure to disk (as HTML/PDF/SVG/...).

    If the file is has .svelte extension, insert `{...$$props}` into the figure's
    top-level div so it can be later styled and customized from Svelte code.

    Args:
        fig (go.Figure | plt.Figure | plt.Axes): Plotly or matplotlib Figure or
            matplotlib Axes object.
        path (str): Path to image file that will be created.
        plotly_config (dict, optional): Configuration options for fig.write_html().
            Defaults to dict(showTips=False, responsive=True, modeBarButtonsToRemove=
            ["lasso2d", "select2d", "autoScale2d", "toImage"]).
            See https://plotly.com/python/configuration-options.
        env_disable (list[str], optional): Do nothing if any of these environment
            variables are set. Defaults to ("CI",).
        pdf_sleep (float, optional): Minimum time in seconds to wait before writing a
            plotly figure to PDF file. Workaround for this plotly issue
            https://github.com/plotly/plotly.py/issues/3469. Defaults to 0.6. Has no
            effect on matplotlib figures.
        style (str, optional): CSS style string to be inserted into the HTML file.
            Defaults to "". Only used if path ends with .svelte or .html.

        **kwargs: Keyword arguments passed to fig.write_html().
    """
    if any(var in os.environ for var in env_disable):
        return
    # handle matplotlib figures
    if isinstance(fig, (plt.Figure, plt.Axes)):
        if hasattr(fig, "figure"):
            fig = fig.figure  # unwrap Axes
        fig.savefig(path, **kwargs)
        return
    if not isinstance(fig, go.Figure):
        raise TypeError(
            f"Unsupported figure type {type(fig)}, expected plotly or matplotlib Figure"
        )
    is_pdf = path.lower().endswith((".pdf", ".pdfa"))
    if path.lower().endswith((".svelte", ".html")):
        config = dict(
            showTips=False,
            modeBarButtonsToRemove=[
                "lasso2d",
                "select2d",
                "autoScale2d",
                "toImage",
                "toggleSpikelines",
                "hoverClosestCartesian",
                "hoverCompareCartesian",
            ],
            responsive=True,
            displaylogo=False,
        )
        config.update(plotly_config or {})
        defaults = dict(include_plotlyjs=False, full_html=False, config=config)
        defaults.update(kwargs)
        fig.write_html(path, **defaults)
        if path.lower().endswith(".svelte"):
            # insert {...$$props} into top-level div to be able to post-process and
            # style plotly figures from within Svelte files
            with open(path) as file:
                text = file.read().replace("<div>", "<div {...$$props}>", 1)
            with open(path, "w") as file:
                # add trailing newline for pre-commit end-of-file commit hook
                file.write(text + "\n")
        if style:
            with open(path, "r+") as file:
                # replace first '<div ' with '<div {style=} '
                file.write(file.read().replace("<div ", f"<div {style=} ", 1))
    else:
        if is_pdf:
            orig_template = fig.layout.template
            fig.layout.template = "plotly_white"
        # hide click-to-show traces in PDF
        hidden_traces = []
        for trace in fig.data:
            if trace.visible == "legendonly":
                trace.visible = False
                hidden_traces.append(trace)
        fig.write_image(path, **kwargs)
        if is_pdf:
            # write PDFs twice to get rid of "Loading [MathJax]/extensions/MathMenu.js"
            # see https://github.com/plotly/plotly.py/issues/3469#issuecomment-994907721
            sleep(pdf_sleep)
            fig.write_image(path, **kwargs)

            fig.layout.template = orig_template
        for trace in hidden_traces:
            trace.visible = "legendonly"


def save_and_compress_svg(
    fig: go.Figure | plt.Figure | plt.Axes, filename: str
) -> None:
    """Save Plotly figure as SVG and HTML to assets/ folder. Compresses SVG
    file with svgo CLI if available in PATH.

    Args:
        fig (Figure): Plotly or matplotlib Figure/Axes instance.
        filename (str): Name of SVG file (w/o extension).

    Raises:
        ValueError: If fig is None and plt.gcf() is empty.
    """
    assert not filename.endswith(".svg"), f"{filename = } should not include .svg"
    filepath = f"{ROOT}/assets/{filename}.svg"
    if isinstance(fig, plt.Axes):
        fig = fig.figure

    if isinstance(fig, plt.Figure) and not fig.axes:
        raise ValueError("Passed fig contains no axes. Nothing to plot!")
    save_fig(fig, filepath)
    plt.close()

    if (svgo := which("svgo")) is not None:
        subprocess.run([svgo, "--multipass", filepath], check=True)


def df_to_pdf(
    styler: Styler,
    file_path: str | Path,
    crop: bool = True,
    size: str = "landscape",
    style: str = "",
    **kwargs: Any,
) -> None:
    """Export a pandas Styler to PDF with WeasyPrint.

    Args:
        styler (Styler): Styler object to export.
        file_path (str): Path to save the PDF to. Requires WeasyPrint.
        crop (bool): Whether to crop the PDF margins. Requires pdfCropMargins.
            Defaults to True.
        size (str): Page size. Defaults to "landscape". See
            https://developer.mozilla.org/@page for options.
        style (str): CSS style string to be inserted into the HTML file.
            Defaults to "".
        **kwargs: Keyword arguments passed to Styler.to_html().
    """
    try:
        from weasyprint import HTML
    except ImportError as exc:
        msg = "weasyprint not installed\nrun pip install weasyprint"
        raise ImportError(msg) from exc

    html_str = styler.to_html(**kwargs)

    # CSS to adjust layout and margins
    html_str = f"""
    <style>
        @page {{ size: {size}; }}
        {style}
    </style>
    {html_str}
    """

    html = HTML(string=html_str)

    html.write_pdf(file_path)

    if crop:
        normalize_and_crop_pdf(file_path)


def normalize_and_crop_pdf(file_path: str | Path) -> None:
    """Normalize a PDF using Ghostscript and then crop it.
    Without gs normalization, pdfCropMargins sometimes corrupts the PDF.

    Args:
        file_path (str | Path): Path to the PDF file.
    """
    try:
        normalized_file_path = f"{file_path}_normalized.pdf"
        from pdfCropMargins import crop

        # Normalize the PDF with Ghostscript
        subprocess.run(
            [
                *"gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.4".split(),
                *"-dPDFSETTINGS=/default -dNOPAUSE -dQUIET -dBATCH".split(),
                f"-sOutputFile={normalized_file_path}",
                str(file_path),
            ],
            check=True,
        )

        # Crop the normalized PDF
        cropped_file_path, exit_code, stdout, stderr = crop(
            ["--percentRetain", "0", normalized_file_path]
        )

        if stderr:
            print(f"pdfCropMargins {stderr=}")
            # something went wrong, remove the cropped PDF
            os.remove(cropped_file_path)
        else:
            # replace the original PDF with the cropped one
            os.replace(cropped_file_path, str(file_path))

        os.remove(normalized_file_path)

    except ImportError as exc:
        msg = "pdfCropMargins not installed\nrun pip install pdfCropMargins"
        raise ImportError(msg) from exc
    except Exception as exc:
        raise RuntimeError("Error cropping PDF margins") from exc
