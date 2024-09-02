"""Define custom pymatviz templates (default styles) for plotly and matplotlib."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


PKG_NAME = "pymatviz"
try:
    __version__ = version(PKG_NAME)
except PackageNotFoundError:
    pass  # package not installed


plt.rc("font", size=16)
plt.rc("savefig", bbox="tight", dpi=200)
plt.rc("figure", dpi=200, titlesize=18)
plt.rcParams["figure.constrained_layout.use"] = True

# activate the pymatviz white/dark templates by setting
# `pio.templates.default = "pymatviz_white|pymatviz_dark"`
# when using plotly or graph objects or
# `px.defaults.template = "pymatviz_white|pymatviz_dark"`
# when using plotly express or use set_plotly_template below
axis_template = dict(
    mirror=True,
    showline=True,
    ticks="outside",
    zeroline=True,
    linewidth=1,
    showgrid=True,
)
white_axis_template = axis_template | dict(linecolor="black", gridcolor="lightgray")
# inherit from plotly_white template
pmv_white_template = go.layout.Template(pio.templates["plotly_white"])
common_layout = dict(margin=dict(l=10, r=10, t=10, b=10))

pio.templates[f"{PKG_NAME}_white"] = pmv_white_template.update(
    layout=dict(
        xaxis=white_axis_template,
        yaxis=white_axis_template,
        font=dict(color="black"),
        **common_layout,
    )
)
dark_axis_template = axis_template | dict(linecolor="white", gridcolor="darkgray")
# inherit from plotly_dark template
pmv_dark_template = go.layout.Template(pio.templates["plotly_dark"])

pio.templates[f"{PKG_NAME}_dark"] = pmv_dark_template.update(
    layout=dict(
        xaxis=dark_axis_template,
        yaxis=dark_axis_template,
        font=dict(color="white"),
        **common_layout,
    )
)


def set_plotly_template(template: str | go.layout.Template) -> None:
    """Set the default plotly express and graph objects template.

    Args:
        template: Usually "pymatviz_white" or "pymatviz_dark" but any plotly.io.template
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
