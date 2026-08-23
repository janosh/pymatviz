"""Tests for StructureWidget rendering and notebook integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import traitlets as tl

from pymatviz import StructureWidget
from tests.widgets.conftest import (
    assert_widget_build_files,
    assert_widget_notebook_integration,
    assert_widget_property_sync,
)


if TYPE_CHECKING:
    from pymatgen.core import Structure


# minimal flat matterviz VolumetricData payload (2x1x1 grid)
_VOLUME: dict[str, Any] = {
    "values": [0.1, 0.2],
    "dims": [2, 1, 1],
    "lattice": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    "origin": [0.0, 0.0, 0.0],
    "periodic": True,
}


@pytest.mark.parametrize(
    "structure_fixture",
    ["structures", "fe3co4_disordered", "fe3co4_disordered_with_props"],
)
def test_widget_structure_inputs(
    structure_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Widget must handle different structure input types and formats."""
    fixture_value = request.getfixturevalue(structure_fixture)

    # Handle both single structures and tuples of structures
    structures = fixture_value if isinstance(fixture_value, tuple) else (fixture_value,)

    for structure in structures:
        # Test with Structure object
        widget = StructureWidget(structure=structure)
        assert widget.widget_type == "structure"
        assert widget.structure == structure.as_dict()

        # Test with structure dict
        structure_dict = structure.as_dict()
        widget = StructureWidget(structure=structure_dict)
        assert widget.structure == structure_dict

        # Test structure assignment and update
        widget = StructureWidget()
        widget.structure = structure_dict
        assert widget.structure == structure_dict


@pytest.mark.parametrize("invalid_structure", ["invalid_structure", 123, ["invalid"]])
def test_widget_invalid_structure_handling(invalid_structure: Any) -> None:
    """Widget rejects non-Structure, non-dict inputs with TypeError."""
    with pytest.raises(TypeError):
        StructureWidget(structure=invalid_structure)


@pytest.mark.parametrize(
    ("property_name", "test_values"),
    [
        ("atom_radius", [1.0, 1.5, 2.0]),
        ("show_bonds", [True, False]),
        ("color_scheme", ["Jmol", "CPK", "VESTA"]),
        ("style", [None, "width: 400px; height: 600px", "width: 600px; height: 800px"]),
        ("show_controls", [True, False]),
        ("enable_info_pane", [True, False]),
        ("volumetric_data", [[], [_VOLUME]]),
    ],
)
def test_widget_property_sync_structure(
    structures: tuple[Structure, Structure],
    property_name: str,
    test_values: list[Any],
) -> None:
    """Widget properties must sync with frontend and handle various values."""
    widget = StructureWidget(structure=structures[0])

    for test_value in test_values:
        assert_widget_property_sync(widget, property_name, test_value)


def test_widget_build_files_and_display_structure() -> None:
    """Widget must load build files and display properly."""
    widget = StructureWidget()
    assert_widget_build_files(widget)


def test_widget_notebook_integration_structure(
    structures: tuple[Structure, Structure],
) -> None:
    """Widget must integrate properly with notebook environments."""
    widget = StructureWidget(structure=structures[0])
    assert_widget_notebook_integration(widget)


def test_widget_state_restoration(structures: tuple[Structure, Structure]) -> None:
    """Widget constructed from another widget's state preserves all values."""
    state = {
        "structure": structures[0].as_dict(),
        "atom_radius": 1.5,
        "show_bonds": True,
        "color_scheme": "Jmol",
        "style": "width: 800px; height: 600px",
        "show_controls": False,
    }
    original = StructureWidget(**state)
    restored = StructureWidget(**{k: getattr(original, k) for k in state})

    for key, value in state.items():
        assert getattr(restored, key) == value


def test_widget_handles_large_structures(
    structures: tuple[Structure, Structure],
) -> None:
    """Widget preserves all sites in a 2000-atom supercell."""
    widget = StructureWidget(structure=structures[0] * 10)
    assert len(widget.structure["sites"]) == 2000


def test_widget_no_structure_default() -> None:
    """Widget constructed with no args defaults structure to None."""
    widget = StructureWidget()
    assert widget.structure is None


def test_widget_preserves_disordered_site_properties(
    fe3co4_disordered_with_props: Structure,
) -> None:
    """Widget preserves magmom/force site properties from disordered structures."""
    widget = StructureWidget(structure=fe3co4_disordered_with_props)
    uniq_prop_keys = {
        key for site in widget.structure["sites"] for key in site.get("properties", {})
    }
    assert uniq_prop_keys == {"magmom", "force"}


def test_widget_preserves_multi_vector_site_properties(
    fe3co4_disordered: Structure,
) -> None:
    """Widget serialization preserves multiple vector site property keys."""
    struct = fe3co4_disordered.copy(
        site_properties={
            "force_DFT": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "force_MLFF": [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]],
        }
    )
    widget = StructureWidget(structure=struct)
    uniq_prop_keys = {
        key for site in widget.structure["sites"] for key in site.get("properties", {})
    }
    assert uniq_prop_keys == {"force_DFT", "force_MLFF"}


def test_widget_volumetric_data_trait() -> None:
    """volumetric_data defaults to [], keeps flat payloads as-is (dropping keys the
    renderer recomputes), flattens legacy nested grid[ix][iy][iz] dicts to z-fastest
    values + dims and supports mutation.
    """
    assert StructureWidget().volumetric_data == []
    vol = {**_VOLUME, "label": "chg"}
    widget = StructureWidget(volumetric_data=[vol])
    assert widget.to_dict()["volumetric_data"] == [vol]
    widget.volumetric_data = [vol, {**_VOLUME, "periodic": False, "label": "elf"}]
    assert widget.volumetric_data == [vol, {**vol, "periodic": False, "label": "elf"}]

    grid = np.random.default_rng(seed=0).random((2, 3, 4))
    meta = {k: v for k, v in vol.items() if k not in ("values", "dims")}
    legacy = {**meta, "grid": grid.tolist(), "grid_dims": [2, 3, 4], "data_order": "z"}
    (payload,) = StructureWidget(volumetric_data=[legacy]).volumetric_data
    # z fastest: index (ix * ny + iy) * nz + iz == numpy C-order ravel
    assert payload == {**meta, "values": grid.ravel().tolist(), "dims": [2, 3, 4]}
    assert payload["values"][(1 * 3 + 2) * 4 + 3] == grid[1, 2, 3]


def test_widget_volumetric_data_from_pymatgen() -> None:
    """Pymatgen VolumetricData expands to one flat payload per data key; Chgcar values
    are divided by the cell volume like matterviz's CHGCAR parser.
    """
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.vasp.outputs import Chgcar, VolumetricData

    struct = Structure(Lattice.cubic(2.0), ["Na"], [[0, 0, 0]])
    rng = np.random.default_rng(seed=0)
    data = {"total": rng.random((2, 2, 3)), "diff": rng.random((2, 2, 3))}
    total, diff = StructureWidget(
        volumetric_data=[Chgcar(struct, data)]
    ).volumetric_data
    assert diff["label"] == "diff"
    assert total == {
        "values": pytest.approx((data["total"] / 8.0).ravel().tolist(), rel=1e-15),
        "dims": [2, 2, 3],
        "lattice": struct.lattice.matrix.tolist(),
        "origin": [0.0, 0.0, 0.0],
        "periodic": True,
        "label": "total",
    }
    # generic VolumetricData (ELFCAR, LOCPOT, ...) is passed through unscaled
    volume = VolumetricData(struct, {"total": data["total"]})
    (raw,) = StructureWidget(volumetric_data=[volume]).volumetric_data
    assert raw["values"] == data["total"].ravel().tolist()


@pytest.mark.parametrize("bad_input", ["not_a_list", ["not_a_dict"], [42]])
def test_widget_volumetric_data_rejects_invalid(bad_input: Any) -> None:
    """volumetric_data rejects non-lists and elements that are neither dicts nor
    pymatgen VolumetricData (shape errors are left to the renderer).
    """
    with pytest.raises((tl.TraitError, TypeError), match="volumetric_data"):
        StructureWidget(volumetric_data=bad_input)


def test_widget_vector_configs_trait() -> None:
    """Test vector_configs trait round-trips, updates, and defaults."""
    configs = {
        "force_DFT": {"visible": True, "color": "#e74c3c", "scale": None},
        "force_MLFF": {"visible": True, "color": "#3498db", "scale": 1.5},
    }
    widget = StructureWidget(vector_configs=configs)
    assert widget.vector_configs == configs

    updated = {**configs, "force_MLFF": {**configs["force_MLFF"], "visible": False}}
    widget.vector_configs = updated
    assert widget.vector_configs["force_MLFF"]["visible"] is False

    assert StructureWidget().vector_configs is None
    assert StructureWidget(vector_configs={}).vector_configs == {}
