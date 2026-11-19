"""Plotly figure export utilities."""

from __future__ import annotations

import copy
import html as html_mod
import os
import re
import subprocess
from numbers import Integral, Real
from shutil import which
from typing import TYPE_CHECKING

import plotly.graph_objects as go

import pymatviz as pmv
from pymatviz.utils import ROOT


if TYPE_CHECKING:
    from typing import Any


def save_fig(
    fig: go.Figure,
    path: str,
    *,
    plotly_config: dict[str, Any] | None = None,
    style: str = "",
    prec: int | None = None,
    template: str | None = None,
    **kwargs: Any,
) -> None:
    """Write a plotly figure to disk (as HTML/PDF/SVG/...).

    If the file has .svelte extension, insert `{...$$props}` into the figure's
    top-level div so it can be later styled and customized from Svelte code.

    Args:
        fig (go.Figure): Plotly Figure object.
        path (str): Path to image file that will be created.
        plotly_config (dict, optional): Configuration options for fig.write_html().
            Defaults to dict(showTips=False, responsive=True, modeBarButtonsToRemove=
            ["lasso2d", "select2d", "autoScale2d", "toImage"]).
            See https://plotly.com/python/configuration-options.
        style (str, optional): CSS style string to be inserted into the HTML file.
            Defaults to "". Only used if path ends with .svelte or .html.
        prec (int, optional): Number of decimal places to round trace x/y floats
            to in the exported file. Defaults to None (no rounding). Sensible
            values are usually 4, 5, 6.
        template (str, optional): Temporary plotly to apply to the figure before
            saving. Will be reset to the original after. Defaults to "pymatviz_white" if
            path ends with .pdf or .pdfa, else None. Set to None to disable.
        **kwargs: Keyword arguments passed to fig.write_html().
    """
    is_pdf = path.lower().endswith((".pdf", ".pdfa"))
    if template is None and is_pdf:
        template = "pymatviz_white"

    if not isinstance(fig, go.Figure):
        raise TypeError(f"Unsupported figure type {type(fig)}, expected plotly Figure")

    if prec is not None:
        # Round coordinates on a copy so exporting leaves the caller's figure intact.
        source_traces = fig.data
        fig = copy.deepcopy(fig)
        for trace, source in zip(fig.data, source_traces, strict=True):
            for axis in ("x", "y"):
                if (coords := getattr(source, axis, None)) is not None:
                    trace[axis] = [
                        round(float(value), prec)
                        if isinstance(value, Real) and not isinstance(value, Integral)
                        else value
                        for value in coords
                    ]

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
        is_svelte = path.lower().endswith(".svelte")
        if is_svelte or style:
            with open(path, encoding="utf-8") as file:
                text = file.read()
            if is_svelte:
                # Forward Svelte props to the outer div, preserving Plotly's attributes.
                text = re.sub(r"<div\b", "<div {...$$props}", text, count=1) + "\n"
            if style:  # merge into plotly's outer div style (browsers drop duplicate
                # attrs), user style last to win. First <div>, not first tag:
                # full_html=True starts with <html>.
                match = re.search(r'<div\b[^>]*?style="([^"]*)"', text)
                if match is None:
                    raise ValueError(
                        f"No styled <div> in plotly HTML written to {path}"
                    )
                merged = f"{match[1].rstrip('; ')}; {html_mod.escape(style)}"
                text = f"{text[: match.start(1)]}{merged}{text[match.end(1) :]}"
            with open(path, mode="w", encoding="utf-8") as file:
                file.write(text)
    else:
        orig_template = fig.layout.template
        hidden_traces = []
        try:
            if is_pdf and template:
                fig.layout.template = template
            # Hide legendonly (click-to-show) traces for static exports.
            for trace in fig.data:
                if trace.visible == "legendonly":
                    trace.visible = False
                    hidden_traces.append(trace)
            fig.write_image(path, **kwargs)
        finally:
            if is_pdf:
                fig.layout.template = orig_template
            for trace in hidden_traces:
                trace.visible = "legendonly"


def save_and_compress_svg(fig: go.Figure, filename: str) -> None:
    """Save a Plotly figure as SVG to assets/svg/.
    Compresses SVG file with svgo CLI if available in PATH.

    If filename does not include .svg extension and is not absolute, will be treated as
    relative to assets/ folder. This function is mostly meant for pymatviz internal use.

    Args:
        fig (go.Figure): Plotly Figure instance.
        filename (str): Name of SVG file (w/o extension).

    Raises:
        TypeError: If fig is not a Plotly figure.
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
