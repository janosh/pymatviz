from __future__ import annotations

import re

import plotly.express as px
import plotly.graph_objects as go
import pytest
from plotly import io as pio

import pymatviz as pmv


def test_set_template() -> None:
    orig_template = pio.templates.default
    try:
        assert pio.templates.default == "plotly"
        assert px.defaults.template is None

        pmv.set_plotly_template("pymatviz_dark")
        assert pio.templates.default == "pymatviz_dark"
        assert px.defaults.template == "pymatviz_dark"

        pmv.set_plotly_template("pymatviz_white")
        assert pio.templates.default == "pymatviz_white"
        assert px.defaults.template == "pymatviz_white"

        pmv.set_plotly_template(pmv.pmv_dark_template)
        assert pio.templates.default == pmv.pmv_dark_template
        assert px.defaults.template == pmv.pmv_dark_template

        pmv.set_plotly_template(pmv.pmv_white_template)
        assert pio.templates.default == pmv.pmv_white_template
        assert px.defaults.template == pmv.pmv_white_template

        valid_templates = list(pio.templates)
        template = "bad_template"
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unrecognized {template=}. Must be one of {valid_templates}."
            ),
        ):
            pmv.set_plotly_template(template)
    finally:
        pio.templates.default = orig_template


@pytest.mark.parametrize(
    ("template_name", "template", "line_color", "grid_color", "font_color"),
    [
        ("pymatviz_dark", pmv.pmv_dark_template, "white", "darkgray", "white"),
        ("pymatviz_white", pmv.pmv_white_template, "black", "lightgray", "black"),
    ],
)
def test_template(
    template_name: str,
    template: go.layout.Template,
    line_color: str,
    grid_color: str,
    font_color: str,
) -> None:
    layout = template.layout
    assert layout.xaxis.linecolor == layout.yaxis.linecolor == line_color
    assert layout.xaxis.gridcolor == layout.yaxis.gridcolor == grid_color
    assert layout.font.color == font_color
    assert layout.margin.to_plotly_json() == dict(l=10, r=10, t=10, b=10)
    assert layout.xaxis.ticks == layout.yaxis.ticks == "outside"

    assert template == pio.templates[template_name]
