from __future__ import annotations

import re

import plotly.express as px
import pytest
from plotly import io as pio

import pymatviz as pmv


def test_set_template() -> None:
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
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"Unrecognized template='bad_template'. Must be one of {valid_templates}."
        ),
    ):
        pmv.set_plotly_template("bad_template")


def test_dark_template() -> None:
    layout = pmv.pmv_dark_template.layout
    assert layout.xaxis["linecolor"] == layout.yaxis["linecolor"] == "white"
    assert layout.xaxis["gridcolor"] == layout.yaxis["gridcolor"] == "darkgray"
    assert layout.font["color"] == "white"


def test_white_template() -> None:
    layout = pmv.pmv_white_template.layout
    assert layout.xaxis["linecolor"] == layout.yaxis["linecolor"] == "black"
    assert layout.xaxis["gridcolor"] == layout.yaxis["gridcolor"] == "lightgray"
    assert layout.font["color"] == "black"
