from __future__ import annotations

from typing import Any

import ipywidgets as ipw
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from pymatviz.brillouin import brillouin_zone_3d, seekpath_primitive
from pymatviz.widgets.brillouin_zone import BrillouinZoneWidget
from tests.conftest import SI_ATOMS, SI_STRUCTS


pytest.importorskip("seekpath")

FCC_CU = Structure(
    Lattice.cubic(3.61),
    ["Cu"] * 4,
    [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
)


@pytest.mark.parametrize("pmg_struct", [FCC_CU, *SI_STRUCTS[:2]])
def test_seekpath_primitive(pmg_struct: Structure) -> None:
    """The primitive cell keeps the composition and space group of the input."""
    primitive = seekpath_primitive(pmg_struct)
    assert (
        primitive.composition.reduced_formula == pmg_struct.composition.reduced_formula
    )
    spg_num = SpacegroupAnalyzer(pmg_struct).get_space_group_number()
    assert SpacegroupAnalyzer(primitive).get_space_group_number() == spg_num
    # conventional fcc has 4 atoms, its primitive cell 1 with a quarter of the volume
    if pmg_struct is FCC_CU:
        assert len(primitive) == 1
        np.testing.assert_allclose(primitive.volume, FCC_CU.volume / 4, rtol=1e-12)


def test_seekpath_primitive_disordered_sites() -> None:
    """Sites with different partial occupancies but one majority element stay
    inequivalent, so no spurious body-centering halves the cell.
    """
    species = [{"Fe": 0.75, "Co": 0.25}, {"Fe": 0.75, "Ni": 0.25}]
    struct = Structure(Lattice.cubic(3), species, [[0, 0, 0], [0.5, 0.5, 0.5]])
    primitive = seekpath_primitive(struct)
    assert sorted(str(site.species) for site in primitive) == sorted(
        str(site.species) for site in struct
    )
    np.testing.assert_allclose(primitive.volume, struct.volume, rtol=1e-12)


@pytest.mark.parametrize(
    ("oxi_states", "expected_species"),
    [({"Cu": 2}, ["Cu2+"]), ([1, 2, 1, 2], ["Cu"])],  # uniform vs mixed valence
)
def test_seekpath_primitive_oxidation_states(
    oxi_states: dict[str, float] | list[float], expected_species: list[str]
) -> None:
    """Oxidation states don't split symmetry types: decorated fcc Cu reduces to the
    same 1-atom primitive cell as undecorated, keeping uniform oxidation states.
    """
    struct = FCC_CU.copy()
    if isinstance(oxi_states, dict):
        struct.add_oxidation_state_by_element(oxi_states)
    else:
        struct.add_oxidation_state_by_site(oxi_states)
    primitive = seekpath_primitive(struct)
    np.testing.assert_allclose(
        primitive.lattice.matrix,
        seekpath_primitive(FCC_CU).lattice.matrix,
        rtol=0,
        atol=1e-12,
    )
    assert [str(site.species) for site in primitive] == [
        f"{spec}1" for spec in expected_species
    ]


def test_brillouin_zone_3d_single() -> None:
    """A single structure returns one widget showing its primitive cell's zone."""
    widget = brillouin_zone_3d(FCC_CU, show_vectors=True, surface_opacity=0.4)
    assert isinstance(widget, BrillouinZoneWidget)
    assert (widget.show_vectors, widget.surface_opacity) == (True, 0.4)
    # the high-symmetry path the plotly plot drew comes from MatterViz by default
    assert (widget.default_k_path, widget.k_path) == (True, None)
    np.testing.assert_allclose(
        widget.structure["lattice"]["matrix"],
        seekpath_primitive(FCC_CU).lattice.matrix,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    ("structs", "n_cols", "expected_titles", "expected_cols"),
    [
        ({"s0": SI_STRUCTS[0], "a1": SI_ATOMS[1]}, 3, ["s0", "a1"], 2),
        ([SI_STRUCTS[0], SI_STRUCTS[1], FCC_CU], 2, None, 2),
    ],
)
def test_brillouin_zone_3d_grid(
    structs: Any, n_cols: int, expected_titles: list[str] | None, expected_cols: int
) -> None:
    """Several structures make a grid of titled cells, columns capped by count."""
    grid = brillouin_zone_3d(structs, n_cols=n_cols)
    assert isinstance(grid, ipw.GridBox)
    assert len(grid.children) == len(structs)
    assert (
        grid.layout.grid_template_columns == f"repeat({expected_cols}, minmax(0, 1fr))"
    )
    titles = [cell.children[0].value for cell in grid.children]
    if expected_titles:
        assert titles == [f"<b>{title}</b>" for title in expected_titles]
    assert all(
        isinstance(cell.children[1], BrillouinZoneWidget) for cell in grid.children
    )


@pytest.mark.parametrize(
    ("structs", "kwargs", "err_type", "match"),
    [
        ([], {}, ValueError, "Cannot plot empty set of structures"),
        ("invalid", {}, TypeError, "Input must be a pymatgen Structure"),
        (SI_STRUCTS[:2], {"n_cols": 0}, ValueError, "n_cols=0 must be positive"),
    ],
)
def test_brillouin_zone_3d_raises(
    structs: Any, kwargs: dict[str, Any], err_type: type[Exception], match: str
) -> None:
    """Reject empty and invalid inputs and non-positive column counts."""
    with pytest.raises(err_type, match=match):
        brillouin_zone_3d(structs, **kwargs)
