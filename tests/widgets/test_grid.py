"""Tests for the widget grid layout."""

from __future__ import annotations

import ipywidgets as ipw
import numpy as np
import pytest

from pymatviz.widgets.bar_plot import BarPlotWidget
from pymatviz.widgets.grid import widget_grid


@pytest.mark.parametrize(
    ("n_widgets", "n_cols", "expected_cols"),
    [(2, 3, 2), (5, 2, 2), (5, np.int64(4), 4)],  # numpy ints are ints too
)
def test_widget_grid(n_widgets: int, n_cols: int, expected_cols: int) -> None:
    """Cells pair an escaped HTML title with their widget; columns cap at the count."""
    widgets = {f"<s>{idx}": ipw.Label(str(idx)) for idx in range(n_widgets)}
    grid = widget_grid(widgets, n_cols)
    expected_template = f"repeat({expected_cols}, minmax(0, 1fr))"
    assert grid.layout.grid_template_columns == expected_template

    titles = [cell.children[0].value for cell in grid.children]
    assert titles == [f"<b>&lt;s&gt;{idx}</b>" for idx in range(n_widgets)]
    assert [cell.children[1] for cell in grid.children] == list(widgets.values())

    with pytest.raises(ValueError, match="n_cols=0 must be positive"):
        widget_grid(widgets, 0)
    with pytest.raises(ValueError, match="n_cols=-1 must be positive"):
        widget_grid(widgets, -1)
    # non-int n_cols made invalid CSS like repeat(2.5, ...)
    for bad_n_cols in (2.5, True, "2"):
        with pytest.raises(TypeError, match=f"n_cols={bad_n_cols!r} must be an int"):
            widget_grid(widgets, bad_n_cols)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="needs at least one widget"):
        widget_grid({})


def test_widget_grid_static_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under PYMATVIZ_STATIC a grid displays as HTML of its cells' PNGs, titled;
    non-MatterViz cells raise a TypeError naming them.
    """
    from pymatviz.widgets import matterviz

    grid = widget_grid({label: BarPlotWidget(style=label) for label in "ab"}, n_cols=2)
    live_bundle = grid._repr_mimebundle_()
    assert live_bundle is not None
    assert "application/vnd.jupyter.widget-view+json" in live_bundle
    monkeypatch.setenv("PYMATVIZ_STATIC", "1")
    monkeypatch.setattr(
        matterviz,
        "static_png_bundle",
        lambda widget, **_img_kwargs: {"image/png": widget.style * 3},
    )
    bundle = grid._repr_mimebundle_()
    assert bundle is not None
    assert bundle["text/plain"] == "<WidgetGrid>"
    html_grid = bundle["text/html"]
    assert "grid-template-columns:repeat(2, minmax(0, 1fr))" in html_grid
    for label in "ab":
        assert f"<figcaption><b>{label}</b></figcaption>" in html_grid
        assert f"data:image/png;base64,{label * 3}'" in html_grid
    # used to crash with AttributeError: 'Label' object has no attribute 'to_img'
    mixed_grid = widget_grid({"plot": BarPlotWidget(), "note": ipw.Label("hi")})
    with pytest.raises(TypeError, match=r"got \{'<b>note</b>': 'Label'\}"):
        mixed_grid._repr_mimebundle_()
