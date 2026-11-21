from __future__ import annotations

from typing import Any

import ipywidgets as ipw
import pandas as pd
import pytest
from pymatgen.core import IStructure, Lattice, Structure

import pymatviz as pmv
from pymatviz.structure import disordered_demo_structures
from pymatviz.widgets.structure import StructureWidget
from tests.conftest import SI_ATOMS, SI_STRUCTS


def test_structure_3d_single(fe3co4_disordered: IStructure) -> None:
    """A single structure gives one widget with every kwarg set as a trait."""
    widget = pmv.structure_3d(
        fe3co4_disordered, show_bonds=True, color_scheme="jmol", style="height: 9em"
    )
    assert isinstance(widget, StructureWidget)
    assert (widget.show_bonds, widget.color_scheme, widget.style) == (
        True,
        "Jmol",
        "height: 9em",
    )
    assert widget.structure == fe3co4_disordered.as_dict()


@pytest.mark.parametrize("cell_type", ["original", "conventional", "primitive"])
def test_structure_3d_cell_type(cell_type: str) -> None:
    """cell_type reaches MatterViz, which standardizes; the input cell is synced."""
    struct = Structure(Lattice.cubic(4), ["Fe", "O"], [[-0.1, 0, 0], [0.5, 0.5, 0.5]])
    struct.add_site_property("force", [[1, 0, 0], [0, 1, 0]])
    widget = pmv.structure_3d(struct, cell_type=cell_type)
    assert isinstance(widget, StructureWidget)
    assert widget.cell_type == cell_type
    assert widget.structure == struct.as_dict()


@pytest.mark.parametrize(
    ("structs", "n_cols", "expected_titles", "expected_cols"),
    [
        (dict(disordered_demo_structures), 3, list(disordered_demo_structures), 3),
        (SI_STRUCTS[:2], 1, None, 1),
        (pd.Series(SI_ATOMS[:2], index=["x", "y"]), 3, ["x", "y"], 2),
    ],
)
def test_structure_3d_grid(
    structs: Any, n_cols: int, expected_titles: list[str] | None, expected_cols: int
) -> None:
    """Several structures make a grid of titled cells sharing all widget kwargs."""
    grid = pmv.structure_3d(structs, n_cols=n_cols, show_image_atoms=False)
    assert isinstance(grid, ipw.GridBox)
    assert len(grid.children) == len(structs)
    assert (
        grid.layout.grid_template_columns == f"repeat({expected_cols}, minmax(0, 1fr))"
    )
    if expected_titles:
        titles = [cell.children[0].value for cell in grid.children]
        assert titles == [f"<b>{title}</b>" for title in expected_titles]
    widgets = [cell.children[1] for cell in grid.children]
    assert all(isinstance(widget, StructureWidget) for widget in widgets)
    assert not any(widget.show_image_atoms for widget in widgets)


@pytest.mark.parametrize(
    ("structs", "kwargs", "err_type", "match"),
    [
        ([], {}, ValueError, "Cannot plot empty set of structures"),
        ("invalid", {}, TypeError, "Input must be a pymatgen Structure"),
        (SI_STRUCTS[:2], {"n_cols": -1}, ValueError, "n_cols=-1 must be positive"),
    ],
)
def test_structure_3d_raises(
    structs: Any, kwargs: dict[str, Any], err_type: type[Exception], match: str
) -> None:
    """Reject empty and invalid inputs and non-positive column counts."""
    with pytest.raises(err_type, match=match):
        pmv.structure_3d(structs, **kwargs)
