from typing import Any

import ipywidgets as ipw
import pytest
from pymatgen.core import (
    Composition,
    DummySpecies,
    Element,
    Lattice,
    PeriodicSite,
    Species,
    Structure,
)

from pymatviz.structure import helpers
from pymatviz.structure.helpers import (
    NO_SYM_MSG,
    get_site_elements,
    get_site_species,
    get_site_symbol,
    standardize_struct,
    widget_grid,
)


@pytest.mark.parametrize(
    ("species_input", "expected_type", "expected_elements", "expected_symbol"),
    [
        ("Si", Element, {"Si"}, "Si"),
        (Element("O"), Element, {"O"}, "O"),
        (Species("Fe", 2), Species, {"Fe"}, "Fe"),
        (Composition({"Fe": 0.75, "Co": 0.25}), Composition, {"Fe", "Co"}, "Fe"),
        (
            Composition({"Fe": 0.2, "Co": 0.3, "Ni": 0.5}),
            Composition,
            {"Fe", "Co", "Ni"},
            "Ni",
        ),
        (DummySpecies("X"), DummySpecies, {"X"}, "X"),
        (
            Composition({"Li": 0.75, DummySpecies("X"): 0.25}),
            Composition,
            {"Li", "X"},
            "Li",
        ),
    ],
)
def test_get_site_species_elements_symbol(
    species_input: Any,
    expected_type: type,
    expected_elements: set[str],
    expected_symbol: str,
) -> None:
    """Species, element set and majority symbol for ordered and disordered sites."""
    site = PeriodicSite(species_input, [0, 0, 0], Lattice.cubic(3.0))
    assert isinstance(get_site_species(site), expected_type)
    assert get_site_elements(site) == expected_elements
    assert get_site_symbol(site) == expected_symbol


def test_standardize_struct(monkeypatch: pytest.MonkeyPatch) -> None:
    """Standardization keeps site properties and warns if symmetry is undetermined."""
    struct = Structure(Lattice.cubic(4), ["Fe", "O"], [[-0.1, 0, 0], [0.5, 0.5, 0.5]])
    struct.add_site_property("force", [[1, 0, 0], [0, 1, 0]])

    std_struct = standardize_struct(struct)
    assert std_struct is not struct
    assert all(site.frac_coords.min() >= 0 for site in std_struct)
    # site properties used to be dropped, silently removing force/magmom vectors
    assert sorted(map(tuple, std_struct.site_properties["force"])) == [
        (0, 1, 0),
        (1, 0, 0),
    ]

    class FailingSpacegroupAnalyzer:
        """Stand-in for SpacegroupAnalyzer when spglib can't find symmetry."""

        def __init__(self, _struct: Structure) -> None:
            raise ValueError("Unable to determine symmetry")

    monkeypatch.setattr(helpers, "SpacegroupAnalyzer", FailingSpacegroupAnalyzer)
    with pytest.warns(
        UserWarning, match=f"{NO_SYM_MSG} for Fe1 O1: Unable to determine symmetry"
    ):
        assert standardize_struct(struct) is struct


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
