"""I/O utilities for saving figures and dataframes to various image formats."""

from __future__ import annotations

import copy
import os
import subprocess
import warnings
from shutil import which
from time import sleep
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from tqdm import tqdm

import pymatviz as pmv
from pymatviz.utils import ROOT


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from typing import Any, Final

    from pandas.io.formats.style import Styler


class TqdmDownload(tqdm):
    """Progress bar for urllib.request.urlretrieve file download.

    Adapted from official TqdmUpTo example.
    See https://github.com/tqdm/tqdm/blob/4c956c2/README.rst#hooks-and-callbacks
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
    fig: go.Figure,
    path: str,
    *,
    plotly_config: dict[str, Any] | None = None,
    env_disable: Sequence[str] = ("CI",),
    pdf_sleep: float = 0.6,
    style: str = "",
    prec: int | None = None,  # Added round keyword argument
    template: str | None = None,
    **kwargs: Any,
) -> None:
    """Write a plotly figure to disk (as HTML/PDF/SVG/...).

    If the file is has .svelte extension, insert `{...$$props}` into the figure's
    top-level div so it can be later styled and customized from Svelte code.

    Args:
        fig (go.Figure): Plotly Figure object.
        path (str): Path to image file that will be created.
        plotly_config (dict, optional): Configuration options for fig.write_html().
            Defaults to dict(showTips=False, responsive=True, modeBarButtonsToRemove=
            ["lasso2d", "select2d", "autoScale2d", "toImage"]).
            See https://plotly.com/python/configuration-options.
        env_disable (list[str], optional): Do nothing if any of these environment
            variables are set. Defaults to ("CI",).
        pdf_sleep (float, optional): Minimum time in seconds to wait before writing a
            plotly figure to PDF file. Workaround for this plotly issue
            https://github.com/plotly/plotly.py/issues/3469. Defaults to 0.6.
        style (str, optional): CSS style string to be inserted into the HTML file.
            Defaults to "". Only used if path ends with .svelte or .html.
        prec (int, optional): Number of significant digits to keep for any float
            in the exported file. Defaults to None (no rounding). Sensible values are
            usually 4, 5, 6.
        template (str, optional): Temporary plotly to apply to the figure before
            saving. Will be reset to the original after. Defaults to "pymatviz_white" if
            path ends with .pdf or .pdfa, else None. Set to None to disable.
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

    if not isinstance(fig, go.Figure):
        raise TypeError(f"Unsupported figure type {type(fig)}, expected plotly Figure")
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
        fig_defaults = dict(include_plotlyjs=False, full_html=False, config=config)
        fig.write_html(path, **fig_defaults | kwargs)
        if path.lower().endswith(".svelte"):
            # insert {...$$props} into top-level div to be able to post-process and
            # style plotly figures from within Svelte files
            with open(path, encoding="utf-8") as file:
                text = file.read().replace("<div>", "<div {...$$props}>", 1)
            with open(path, mode="w", encoding="utf-8") as file:
                # add trailing newline for pre-commit end-of-file commit hook
                file.write(text + "\n")
        if style:
            with open(path, encoding="utf-8") as file:
                text = file.read()
            with open(path, mode="w", encoding="utf-8") as file:
                file.write(text.replace("<div ", f"<div {style=} ", 1))
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
    fig: go.Figure,
    filename: str,
) -> None:
    """Save Plotly figure as SVG and HTML to assets/ folder.
    Compresses SVG file with svgo CLI if available in PATH.

    If filename does not include .svg extension and is not absolute, will be treated as
    relative to assets/ folder. This function is mostly meant for pymatviz internal use.

    Args:
        fig (go.Figure): Plotly Figure instance.
        filename (str): Name of SVG file (w/o extension).

    Raises:
        ValueError: If fig is None.
    """
    if not isinstance(fig, go.Figure):
        raise TypeError("fig must be a plotly Figure instance")

    if not filename.endswith(".svg") and not os.path.isabs(filename):
        filepath = f"{ROOT}/assets/svg/{filename}.svg"
    else:
        filepath = filename

    pmv.save_fig(fig, filepath)

    # Compress SVG if svgo is available
    if (svgo := which("svgo")) is not None:
        subprocess.run([svgo, "--multipass", "--final-newline", filepath], check=True)  # noqa: S603


DEFAULT_DF_STYLES: Final = {
    "": "font-family: sans-serif; border-collapse: collapse;",
    "td, th": "border: none; padding: 4px 6px; white-space: nowrap;",
    "th.col_heading": "border: 1px solid; border-width: 1px 0; text-align: left;",
    "th.row_heading": "font-weight: normal; padding: 3pt;",
}


ALLOW_TABLE_SCROLL = "table { overflow: scroll; max-width: 100%; display: block; }"
# https://stackoverflow.com/a/38994837
HIDE_SCROLL_BAR = """
table {
    scrollbar-width: none;  /* Firefox */
}
table::-webkit-scrollbar {
    display: none;  /* Safari and Chrome */
}"""


def df_to_html_table(*args: Any, **kwargs: Any) -> str:  # noqa: D103
    msg = "df_to_html_table is deprecated. Use df_to_html instead."
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return df_to_html(*args, **kwargs)


def df_to_html(
    styler: Styler,
    *,
    file_path: str | Path | None = None,
    inline_props: str | None = "{{...$$props}} ",
    pre_table: str | None = "",
    styles: str | None = ALLOW_TABLE_SCROLL + HIDE_SCROLL_BAR,
    styler_css: bool | dict[str, str] = True,
    post_process: Callable[[str], str] | None = None,
    **kwargs: Any,
) -> str:
    """Convert a pandas Styler to a svelte table.

    Args:
        styler (Styler): Styler object to export.
        file_path (str): Path to the file to write the svelte table to.
        inline_props (str): Inline props to pass to the table element. Example:
            "class='table' style='width: 100%'". Defaults to "".
        pre_table (str): HTML string to insert above the table. Defaults to "". Will
            replace the opening table tag to allow passing props to it.
        styles (str): CSS rules to insert at the bottom of the style tag. Defaults to
            TABLE_SCROLL_CSS.
        styler_css (bool | dict[str, str]): Whether to apply some sensible default CSS
            to the pandas Styler. Defaults to True. If dict, keys are CSS selectors and
            values CSS strings. Example:
            dict("td, th": "border: none; padding: 4px 6px;")
        post_process (Callable[[str], str]): Function to post-process the HTML string
            before writing it to file. Defaults to None.
        **kwargs: Keyword arguments passed to Styler.to_html().

    Returns:
        str: pandas Styler as HTML.
    """
    styler.set_uuid("")
    if styler_css:
        styler_css = styler_css if isinstance(styler_css, dict) else DEFAULT_DF_STYLES
        styler.set_table_styles(
            [dict(selector=sel, props=val) for sel, val in styler_css.items()]
        )
    html = styler.to_html(**kwargs)
    if html is None:
        raise ValueError("Styler.to_html() returned None, don't pass buf kwarg")
    if pre_table:
        html = html.replace("<table", pre_table)

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
