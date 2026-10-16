"""Define custom pymatviz templates (default styles) for plotly and matplotlib."""

from __future__ import annotations

from typing import TYPE_CHECKING

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


if TYPE_CHECKING:
    from typing import Literal


PKG_NAME = "pymatviz"


def _register_template(
    theme: str, line_color: str, grid_color: str
) -> go.layout.Template:
    """Register pymatviz_{theme} (plotly_{theme} with framed axes and slim margins)
    with plotly.io and return it.
    """
    axis = dict(
        mirror=True,
        showline=True,
        ticks="outside",
        zeroline=True,
        linewidth=1,
        showgrid=True,
        linecolor=line_color,
        gridcolor=grid_color,
    )
    template = go.layout.Template(pio.templates[f"plotly_{theme}"]).update(
        layout=dict(
            xaxis=axis,
            yaxis=axis,
            font=dict(color=line_color),
            margin=dict(l=10, r=10, t=10, b=10),
        )
    )
    pio.templates[f"{PKG_NAME}_{theme}"] = template
    return template


# activate with pio.templates.default = "pymatviz_white" (or "pymatviz_dark") for
# graph objects, px.defaults.template = ... for plotly express, or set_plotly_template
pmv_white_template = _register_template("white", "black", "lightgray")
pmv_dark_template = _register_template("dark", "white", "darkgray")


def set_plotly_template(
    template: Literal["pymatviz_white", "pymatviz_dark"] | str | go.layout.Template,  # noqa: PYI051
) -> None:
    """Set the default plotly express and graph objects template.

    Args:
        template ("pymatviz_white" | "pymatviz_dark" | go.layout.Template):
            Usually "pymatviz_white" or "pymatviz_dark" but any plotly.io.template
            name or the object itself is valid.

    Raises:
        ValueError: If the template is not recognized.
    """
    try:
        pio.templates.default = template
        px.defaults.template = template
    except ValueError as exc:
        valid_templates = list(pio.templates)
        raise ValueError(
            f"Unrecognized {template=}. Must be one of {valid_templates}."
        ) from exc
