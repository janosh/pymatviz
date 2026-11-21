"""Tests for the widget grid layout."""

from __future__ import annotations

import ipywidgets as ipw
import pytest

from pymatviz.widgets.grid import widget_grid


@pytest.mark.parametrize(
    ("n_widgets", "n_cols", "expected_cols"), [(2, 3, 2), (5, 2, 2)]
)
def test_widget_grid(n_widgets: int, n_cols: int, expected_cols: int) -> None:
    """Cells pair an escaped HTML title with their widget; columns cap at the count."""
    widgets = {f"<s>{idx}": ipw.Label(str(idx)) for idx in range(n_widgets)}
    grid = widget_grid(widgets, n_cols)
    assert (
        grid.layout.grid_template_columns == f"repeat({expected_cols}, minmax(0, 1fr))"
    )
    assert [cell.children[0].value for cell in grid.children] == [
        f"<b>&lt;s&gt;{idx}</b>" for idx in range(n_widgets)
    ]
    assert [cell.children[1] for cell in grid.children] == list(widgets.values())

    with pytest.raises(ValueError, match="n_cols=0 must be positive"):
        widget_grid(widgets, 0)


def test_widget_grid_static_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """Under PYMATVIZ_STATIC a grid displays as HTML of its cells' PNGs, titled."""
    from pymatviz.widgets import matterviz

    grid = widget_grid({"a": ipw.Label("a"), "b": ipw.Label("b")}, n_cols=2)
    live_bundle = grid._repr_mimebundle_()
    assert live_bundle is not None
    assert "application/vnd.jupyter.widget-view+json" in live_bundle
    monkeypatch.setenv("PYMATVIZ_STATIC", "1")
    monkeypatch.setattr(
        matterviz,
        "static_png_bundle",
        lambda widget, **_img_kwargs: {"image/png": widget.value * 3},
    )
    bundle = grid._repr_mimebundle_()
    assert bundle is not None
    assert bundle["text/plain"] == "<WidgetGrid>"
    html_grid = bundle["text/html"]
    assert "grid-template-columns:repeat(2, minmax(0, 1fr))" in html_grid
    for label in "ab":
        assert f"<figcaption><b>{label}</b></figcaption>" in html_grid
        assert f"data:image/png;base64,{label * 3}'" in html_grid
