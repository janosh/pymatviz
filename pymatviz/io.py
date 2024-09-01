"""I/O utilities for saving figures and dataframes to various image formats."""

from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path
from shutil import which
from time import sleep
from typing import TYPE_CHECKING, Any, Final, Literal

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import lines as mlines
from matplotlib import patches as mpatches
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.utils import ROOT


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    import pandas as pd
    from pandas.io.formats.style import Styler


class TqdmDownload(tqdm):
    """Progress bar for urllib.request.urlretrieve file download.

    Adapted from official TqdmUpTo example.
    See https://github.com/tqdm/tqdm/blob/4c956c20b83be4312460fc0c4812eeb3fef5e7df/README.rst#hooks-and-callbacks
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Sets default values appropriate for file downloads for unit, unit_scale,
        unit_divisor, miniters, desc.
        """
        for key, val in dict(
            unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading"
        ).items():
            kwargs.setdefault(key, val)
        super().__init__(*args, **kwargs)

    def update_to(
        self, n_blocks: int = 1, block_size: int = 1, total_size: int | None = None
    ) -> bool | None:
        """Update hook for urlretrieve.

        Args:
            n_blocks (int, optional): Number of blocks transferred so far. Default = 1.
            block_size (int, optional): Size of each block (in tqdm units). Default = 1.
            total_size (int, optional): Total size (in tqdm units). If None, remains
                unchanged. Defaults to None.

        Returns:
            bool | None: True if tqdm.display() was triggered.
        """
        if total_size is not None:
            self.total = total_size
        # update sets self.n = n_blocks * block_size
        return self.update(n_blocks * block_size - self.n)


def save_fig(
    fig: go.Figure | plt.Figure | plt.Axes,
    path: str,
    plotly_config: dict[str, Any] | None = None,
    env_disable: Sequence[str] = ("CI",),
    pdf_sleep: float = 0.6,
    style: str = "",
    prec: int | None = None,  # Added round keyword argument
    template: str | None = None,
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
        prec (int, optional): Number of significant digits to keep for any float
            in the exported file. Defaults to None (no rounding). Sensible values are
            usually 4, 5, 6.
        template (str, optional): Temporary plotly to apply to the figure before
            saving. Will be reset to the original after. Defaults to "pymatviz_white" if
            path ends with .pdf or .pdfa, else None. Set to None to disable. Only used
            if fig is a plotly figure.
        **kwargs: Keyword arguments passed to fig.write_html().
    """
    is_pdf = path.lower().endswith((".pdf", ".pdfa"))
    if template is None and is_pdf:
        template = "pymatviz_white"

    if prec is not None:
        # create a copy of figure and round all floats in fig.data to round significant
        # figures
        fig = copy.deepcopy(fig)
        for trace in fig.data:
            # trace is a go.Scatter or go.Bar or go.Heatmap or ...
            if trace.x is not None:
                trace.x = [
                    round(x, prec) if isinstance(x, float) else x for x in trace.x
                ]
            if trace.y is not None:
                trace.y = [
                    round(y, prec) if isinstance(y, float) else y for y in trace.y
                ]
    if any(var in os.environ for var in env_disable):
        return
    # handle matplotlib figures
    if isinstance(fig, plt.Figure | plt.Axes):
        if hasattr(fig, "figure"):
            fig = fig.figure  # unwrap Axes
        fig.savefig(path, **kwargs, transparent=True)
        return
    if not isinstance(fig, go.Figure):
        raise TypeError(
            f"Unsupported figure type {type(fig)}, expected plotly or matplotlib Figure"
            " or plt.Axes"
        )
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
            with open(path, encoding="utf-8") as file:
                text = file.read().replace("<div>", "<div {...$$props}>", 1)
            with open(path, mode="w", encoding="utf-8") as file:
                # add trailing newline for pre-commit end-of-file commit hook
                file.write(text + "\n")
        if style:
            with open(path, mode="r+", encoding="utf-8") as file:
                # replace first '<div ' with '<div {style=} '
                file.write(file.read().replace("<div ", f"<div {style=} ", 1))
    else:
        orig_template = fig.layout.template
        if is_pdf and template:
            fig.layout.template = template
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
    fig: go.Figure | plt.Figure | plt.Axes,
    filename: str,
) -> None:
    """Save Plotly figure as SVG and HTML to assets/ folder.
    Compresses SVG file with svgo CLI if available in PATH.

    Args:
        fig (Figure): Plotly or matplotlib Figure/Axes instance.
        filename (str): Name of SVG file (w/o extension).

    Raises:
        ValueError: If fig is None and plt.gcf() is empty.
    """
    if isinstance(fig, plt.Axes):
        fig = fig.figure

    if isinstance(fig, plt.Figure) and not fig.axes:
        raise ValueError("Passed fig contains no axes. Nothing to plot!")

    if filename.endswith(".svg"):
        filename = filename.rstrip(".svg")

    filepath = f"{ROOT}/assets/{filename}.svg"
    pmv.save_fig(fig, filepath)
    plt.close()

    # Compress SVG if svgo is available
    if (svgo := which("svgo")) is not None:
        subprocess.run([svgo, "--multipass", "--final-newline", filepath], check=True)  # noqa: S603


DEFAULT_DF_STYLES: Final = {
    "": "font-family: sans-serif; border-collapse: collapse;",
    "td, th": "border: none; padding: 4px 6px; white-space: nowrap;",
    "th.col_heading": "border: 1px solid; border-width: 1px 0; text-align: left;",
    "th.row_heading": "font-weight: normal; padding: 3pt;",
}


def df_to_pdf(
    styler: Styler,
    file_path: str | Path,
    *,
    crop: bool = True,
    size: str | None = None,
    style: str = "",
    styler_css: bool | dict[str, str] = True,
    **kwargs: Any,
) -> None:
    """Export a pandas Styler to PDF with WeasyPrint.

    Args:
        styler (Styler): Styler object to export.
        file_path (str): Path to save the PDF to. Requires WeasyPrint.
        crop (bool): Whether to crop the PDF margins. Requires pdfCropMargins.
            Defaults to True. Be careful to set size correctly (not much too large as
            is the default) if you set crop=False.
        size (str): Page size. Defaults to "4cm * n_cols x 2cm * n_rows"
            (width x height). See https://developer.mozilla.org/@page for 'landscape'
            and other special values.
        style (str): CSS style string to be inserted into the HTML file.
            Defaults to "".
        styler_css (bool | dict[str, str]): Whether to apply some sensible default CSS
            to the pandas Styler. Defaults to True. If dict, keys are selectors and
            values CSS strings. Example: dict("td, th": "border: none; padding: 4px;")
        **kwargs: Keyword arguments passed to Styler.to_html().
    """
    try:
        from weasyprint import HTML
    except ImportError as exc:
        msg = "weasyprint not installed\nrun pip install weasyprint"
        raise ImportError(msg) from exc

    if styler_css:
        styler_css = DEFAULT_DF_STYLES if styler_css is True else styler_css
        styler.set_table_styles(
            [dict(selector=sel, props=val) for sel, val in styler_css.items()],
            overwrite=False,
        )

    styler.set_uuid("")
    html_str = styler.to_html(**kwargs)

    if size is None:
        n_rows, n_cols = styler.data.shape
        size = f"{n_cols * 4}cm {n_rows * 2}cm"

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


def normalize_and_crop_pdf(
    file_path: str | Path, on_gs_not_found: Literal["ignore", "warn", "error"] = "warn"
) -> None:
    """Normalize a PDF using Ghostscript and then crop it.
    Without gs normalization, pdfCropMargins sometimes corrupts the PDF.

    Args:
        file_path (str | Path): Path to the PDF file.
        on_gs_not_found ("ignore" | "warn" | "error", optional): What to do if
            Ghostscript is not found in PATH. Defaults to "warn".
    """
    if which("gs") is None:
        if on_gs_not_found == "ignore":
            return
        if on_gs_not_found == "warn":
            print("Ghostscript not found, skipping PDF normalization and cropping")  # noqa: T201
            return
        raise RuntimeError("Ghostscript not found in PATH")
    try:
        normalized_file_path = f"{file_path}_normalized.pdf"
        from pdfCropMargins import crop

        # Normalize the PDF with Ghostscript
        subprocess.run(  # noqa: S603
            [
                *"gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.7".split(),
                *"-dPDFSETTINGS=/default -dNOPAUSE -dQUIET -dBATCH".split(),
                f"-sOutputFile={normalized_file_path}",
                str(file_path),
            ],
            check=True,
        )

        # Crop the normalized PDF
        cropped_file_path, _exit_code, _stdout, stderr = crop(
            ["--percentRetain", "0", normalized_file_path]
        )

        if stderr:
            print(f"pdfCropMargins {stderr=}")  # noqa: T201
            # something went wrong, remove the cropped PDF
            os.remove(cropped_file_path)
        else:
            # replace the original PDF with the cropped one
            os.replace(cropped_file_path, str(file_path))

        os.remove(normalized_file_path)

    except ImportError as exc:
        msg = "pdfCropMargins not installed\nrun pip install pdfCropMargins"
        raise ImportError(msg) from exc


ALLOW_TABLE_SCROLL = "table { overflow: scroll; max-width: 100%; display: block; }"
# https://stackoverflow.com/a/38994837
HIDE_SCROLL_BAR = """
table {
    scrollbar-width: none;  /* Firefox */
}
table::-webkit-scrollbar {
    display: none;  /* Safari and Chrome */
}"""


def df_to_html_table(
    styler: Styler,
    *,
    file_path: str | Path | None = None,
    inline_props: str | None = "",
    script: str | None = "",
    styles: str | None = ALLOW_TABLE_SCROLL + HIDE_SCROLL_BAR,
    styler_css: bool | dict[str, str] = True,
    sortable: bool = True,
    post_process: Callable[[str], str] | None = None,
    **kwargs: Any,
) -> str:
    """Convert a pandas Styler to a svelte table.

    Args:
        styler (Styler): Styler object to export.
        file_path (str): Path to the file to write the svelte table to.
        inline_props (str): Inline props to pass to the table element. Example:
            "class='table' style='width: 100%'". Defaults to "".
        script (str): JavaScript string to insert above the table. Will replace the
            opening HTML opening table tag to allow passing props to it. The default
            script uses ...props to enable Svelte props forwarding to the table element.
            See source code to inspect default script.
        styles (str): CSS rules to insert at the bottom of the style tag. Defaults to
            TABLE_SCROLL_CSS.
        styler_css (bool | dict[str, str]): Whether to apply some sensible default CSS
            to the pandas Styler. Defaults to True. If dict, keys are CSS selectors and
            values CSS strings. Example:
            dict("td, th": "border: none; padding: 4px 6px;")
        sortable (bool): Whether to enable sorting the table by clicking on column
            headers. Defaults to True. Requires npm install svelte-zoo.
        post_process (Callable[[str], str]): Function to post-process the HTML string
            before writing it to file. Defaults to None.
        **kwargs: Keyword arguments passed to Styler.to_html().

    Returns:
        str: pandas Styler as HTML.
    """
    sortable_script = """<script lang="ts">
      import { sortable } from 'svelte-zoo/actions'
    </script>

    <table use:sortable {...$$props}
    """

    styler.set_uuid("")
    if styler_css:
        styler_css = styler_css if isinstance(styler_css, dict) else DEFAULT_DF_STYLES
        styler.set_table_styles(
            [dict(selector=sel, props=val) for sel, val in styler_css.items()]
        )
    html = styler.to_html(**kwargs)
    if script:
        html = html.replace("<table", script)
    if sortable:
        html = html.replace("<table", sortable_script)
    if inline_props:
        if "<table " not in html:
            raise ValueError(
                f"Got {inline_props=} but no '<table ...' tag found in HTML string to "
                "attach to"
            )
        html = html.replace("<table", f"<table {inline_props}")
    if styles is not None:
        # insert styles at end of closing </style> tag so they override default styles
        html = html.replace("</style>", f"{styles}\n</style>")
    if callable(post_process):
        html = post_process(html)
    if file_path:
        with open(file_path, mode="w", encoding="utf-8") as file:
            file.write(html)

    return html


def df_to_svg(
    obj: pd.DataFrame | Styler,
    file_path: str | Path,
    *,
    font_size: int = 14,
    compress: bool = True,
    **kwargs: Any,
) -> Figure:
    """Export a pandas DataFrame or Styler to an SVG file and optionally compress it.

    TODO The SVG output has annoying margins that proved hard to remove. The goal is for
    this function to auto-crop the SVG viewBox to the content in the future.

    Args:
        obj (DataFrame | Styler): DataFrame or Styler object to save as SVG.
        file_path (str | Path): Where to save the SVG file.
        font_size (int): Font size in points. Defaults to 14.
        compress (bool): Whether to compress the SVG file using svgo. Defaults to True.
            svgo must be available in PATH.
        **kwargs: Passed to matplotlib.figure.Figure.savefig().

    Returns:
        Figure: Matplotlib Figure conversion of the DataFrame or Styler.

    Raises:
        subprocess.CalledProcessError: If SVG compression fails.
    """
    import bs4
    import cssutils
    from pandas.io.formats.style import Styler

    # TODO find a way to not have to hardcode these values
    fig_width, fig_height, dpi = 20, 4, 72  # Using dpi=72 as a standard value

    def parse_html(html: str) -> tuple[list[list[list[str | bool | int]]], int]:
        html = html.replace("<br>", "\n")
        soup = bs4.BeautifulSoup(html, features="lxml")
        style = soup.find("style")
        sheet = cssutils.parseString(style.text) if style else []

        def get_style_prop(element: bs4.element.Tag, prop_name: str) -> str | None:
            style = element.get("style", "").lower()
            if prop_name in style:
                return style.split(f"{prop_name}:")[1].split(";")[0].strip()
            if "id" in element.attrs:
                for rule in sheet:
                    if f"#{element['id']}" in rule.selectorText:
                        for prop in rule.style:
                            if prop.name == prop_name:
                                return prop.value
            return None

        rows = []
        for row in soup.find_all("tr"):
            cells = []
            for cell in row.find_all(["td", "th"]):
                text = cell.get_text()
                bold = cell.name == "th"
                align = (
                    get_style_prop(cell, "text-align")
                    or get_style_prop(row, "text-align")
                    or "left"
                )
                bg_color = get_style_prop(cell, "background-color") or "#ffffff"
                color = get_style_prop(cell, "color") or "#000000"
                col_span = int(cell.get("colspan", 1))
                row_span = int(cell.get("rowspan", 1))
                cells.append([text, bold, align, bg_color, color, row_span, col_span])
            rows.append(cells)

        num_header_rows = (
            len(soup.find("thead").find_all("tr")) if soup.find("thead") else 0
        )
        return rows, num_header_rows

    def calculate_dimensions(
        text_fig: Figure,
        renderer: RendererAgg,
        rows: list[list[list[str | bool | int]]],
    ) -> tuple[list[float], list[float]]:
        def get_text_width(text: str, weight: str | None) -> float:
            t = text_fig.text(0, 0, text, size=font_size, weight=weight)
            return t.get_window_extent(renderer=renderer).width

        all_text_widths = [
            [
                max(
                    get_text_width(text, "bold" if vals[1] else None)
                    for text in vals[0].split("\n")
                )
                for vals in row
                if isinstance(vals[0], str)
            ]
            for row in rows
        ]
        all_text_widths = np.array(all_text_widths) + 15  # Add padding

        max_col_widths = np.max(all_text_widths, axis=0)
        total_width = fig_width * dpi

        if sum(max_col_widths) >= total_width:
            max_col_widths *= total_width / sum(max_col_widths)

        col_widths = [width / total_width for width in max_col_widths]
        row_heights = [
            (max(val[0].count("\n") + 1 for val in row if isinstance(val[0], str)) + 1)
            * font_size
            / dpi
            for row in rows
        ]

        return col_widths, row_heights

    def print_table(
        fig: Figure,
        rows: list[list[list[str | bool | int]]],
        num_header_rows: int,
        col_widths: list[float],
        row_heights: list[float],
    ) -> Figure:
        row_colors = ("#f5f5f5", "#ffffff")
        padding = font_size / (fig_width * dpi) * 0.5
        total_width = sum(col_widths)
        fig_height = fig.get_figheight()
        row_locs = [height / fig_height for height in row_heights]

        x0 = (1 - total_width) / 2
        y_i = 1

        for idx, (yd, row) in enumerate(zip(row_locs, rows, strict=True)):
            x_i = x0
            y_i -= yd
            # table zebra stripes
            if idx >= num_header_rows and idx % 2 == 0:
                rect = mpatches.Rectangle(
                    (x0, y_i),
                    width=total_width,
                    height=yd,
                    fill=True,
                    color=row_colors[0],
                    transform=fig.transFigure,
                )
                fig.add_artist(rect)

            for xd, val in zip(col_widths, row, strict=True):
                text, weight, ha, bg_color, fg_color = val[:5]

                if bg_color != row_colors[1]:
                    rect_bg = mpatches.Rectangle(
                        (x_i, y_i),
                        width=xd,
                        height=yd,
                        fill=True,
                        color=bg_color,
                        transform=fig.transFigure,
                    )
                    fig.add_artist(rect_bg)

                if ha == "right":
                    x_pos = x_i + xd - padding
                elif ha == "center":
                    x_pos = x_i + xd / 2
                else:  # left align
                    x_pos = x_i + padding

                fig.text(
                    x_pos,
                    y_i + yd / 2,
                    text,
                    size=font_size,
                    ha=ha,
                    va="center",
                    weight="bold" if weight else None,
                    color=fg_color,
                )
                x_i += xd

            if idx == num_header_rows - 1:
                line = mlines.Line2D(
                    [x0, x0 + total_width],
                    [y_i, y_i],
                    color="black",
                    transform=fig.transFigure,
                )
                fig.add_artist(line)

        return fig

    html = obj.to_html() if isinstance(obj, Styler) else obj.to_html(notebook=True)
    text_fig = Figure()
    renderer = RendererAgg(fig_width, fig_height, dpi)

    rows, num_header_rows = parse_html(html)
    col_widths, row_heights = calculate_dimensions(text_fig, renderer, rows)
    fig = Figure(figsize=(fig_width, sum(row_heights)))

    fig = print_table(fig, rows, num_header_rows, col_widths, row_heights)
    fig.savefig(file_path, format="svg", transparent=True, **kwargs)

    # Compress SVG if requested and svgo is available
    if compress:
        if (svgo := which("svgo")) is not None:
            subprocess.run(  # noqa: S603
                [svgo, "--multipass", "--final-newline", str(file_path)], check=True
            )
        else:
            print("svgo not found in PATH. SVG compression skipped.")  # noqa: T201

    return fig
