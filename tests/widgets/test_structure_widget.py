"""Tests for StructureWidget rendering and notebook integration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from pymatviz import StructureWidget
from tests.widgets.conftest import assert_widget_property_sync


if TYPE_CHECKING:
    from pymatgen.core import Structure


# minimal flat matterviz VolumetricData payload (2x1x1 grid)
_VOLUME: dict[str, Any] = {
    "id": "chg",
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
        assert widget.structure is None
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
        ("color_scheme", ["Jmol", "Vesta", "Dark Mode"]),
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


def test_widget_handles_large_structures(
    structures: tuple[Structure, Structure],
) -> None:
    """Widget preserves all sites in a 2000-atom supercell."""
    widget = StructureWidget(structure=structures[0] * 10)
    assert len(widget.structure["sites"]) == 2000


_MULTI_VECTOR_PROPS = {
    "force_DFT": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    "force_MLFF": [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]],
}


@pytest.mark.parametrize(
    ("structure_fixture", "site_properties", "expected_keys"),
    [
        ("fe3co4_disordered_with_props", None, {"magmom", "force"}),
        ("fe3co4_disordered", _MULTI_VECTOR_PROPS, {"force_DFT", "force_MLFF"}),
    ],
    ids=["magmom-force", "multi-vector"],
)
def test_widget_preserves_site_properties(
    structure_fixture: str,
    site_properties: dict[str, list[list[float]]] | None,
    expected_keys: set[str],
    request: pytest.FixtureRequest,
) -> None:
    """Disordered-structure site properties (incl. multiple vector keys) survive
    serialization.
    """
    struct = request.getfixturevalue(structure_fixture)
    if site_properties:
        struct = struct.copy(site_properties=site_properties)
    widget = StructureWidget(structure=struct)
    uniq_prop_keys = {
        key for site in widget.structure["sites"] for key in site.get("properties", {})
    }
    assert uniq_prop_keys == expected_keys


def test_widget_volumetric_data_trait() -> None:
    """volumetric_data defaults to a fresh [], keeps flat payloads as-is (dropping keys
    the renderer recomputes), flattens legacy nested grid[ix][iy][iz] dicts to z-fastest
    values + dims, uses a dict's label as missing id, prefixes ids/labels of mapping
    inputs and supports mutation.
    """
    default_widget = StructureWidget()
    assert default_widget.volumetric_data == []
    assert default_widget.volumetric_data is not StructureWidget().volumetric_data
    vol = {**_VOLUME, "label": "chg"}
    widget = StructureWidget(volumetric_data=[vol])
    assert widget.to_dict()["volumetric_data"] == [vol]
    elf = {**_VOLUME, "id": "elf", "periodic": False, "label": "elf"}
    widget.volumetric_data = [vol, elf]
    assert widget.volumetric_data == [vol, elf]

    grid = np.random.default_rng(seed=0).random((2, 3, 4))
    meta = {k: v for k, v in vol.items() if k not in ("values", "dims")}
    legacy = {**meta, "grid": grid.tolist(), "grid_dims": [2, 3, 4], "data_order": "z"}
    (payload,) = StructureWidget(volumetric_data=[legacy]).volumetric_data
    # z fastest: index (ix * ny + iy) * nz + iz == numpy C-order ravel
    assert payload == {**meta, "values": grid.ravel().tolist(), "dims": [2, 3, 4]}
    assert payload["values"][(1 * 3 + 2) * 4 + 3] == grid[1, 2, 3]

    # id defaults to label (like matterviz's parsers name volumes by block label)
    unnamed = {key: val for key, val in _VOLUME.items() if key != "id"}
    (labeled,) = StructureWidget(
        volumetric_data=[{**unnamed, "label": "charge density"}]
    ).volumetric_data
    assert (labeled["id"], labeled["label"]) == ("charge density", "charge density")
    # an explicit id wins over the label
    widget.volumetric_data = ({**_VOLUME, "label": "rho"},)
    assert [(pl["id"], pl["label"]) for pl in widget.volumetric_data] == [
        ("chg", "rho")
    ]
    # {prefix: volume} mappings namespace ids (and labels if present)
    widget.volumetric_data = {"a": _VOLUME, "b": {**unnamed, "label": "chg"}}
    assert [(pl["id"], pl.get("label")) for pl in widget.volumetric_data] == [
        ("a/chg", None),
        ("b/chg", "b/chg"),
    ]
    assert isinstance(widget.to_dict()["volumetric_data"], list)


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
    assert (diff["id"], diff["label"]) == ("diff", "diff")
    assert total == {
        "id": "total",
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

    # colliding data keys ("total" twice) load side by side via a prefix mapping
    payloads = StructureWidget(
        volumetric_data={"relaxed": Chgcar(struct, data), "initial": volume}
    ).volumetric_data
    assert [(pl["id"], pl["label"]) for pl in payloads] == [
        ("relaxed/total", "relaxed/total"),
        ("relaxed/diff", "relaxed/diff"),
        ("initial/total", "initial/total"),
    ]
    assert payloads[0]["values"] == total["values"]
    with pytest.raises(ValueError, match=r"duplicates \['total'\].*prefix: volume"):
        StructureWidget(volumetric_data=[Chgcar(struct, data), volume])


@pytest.mark.parametrize(
    ("bad_input", "err_cls", "match"),
    [
        ("not_a_list", TypeError, "list of volumes or a {prefix: volume} mapping"),
        (42, TypeError, "got int"),
        (["not_a_dict"], TypeError, "volumetric_data entries"),
        ([42], TypeError, "volumetric_data entries"),
        ({"a": 42}, TypeError, "volumetric_data entries"),
        ({"": _VOLUME}, TypeError, "nonempty strings"),
        ({1: _VOLUME}, TypeError, "nonempty strings"),
        ([{k: v for k, v in _VOLUME.items() if k != "id"}], ValueError, "'id' or"),
        ([{**_VOLUME, "id": " "}], ValueError, "'id' or 'label'"),
        ([{**_VOLUME, "id": 1, "label": "chg"}], ValueError, "'id' or 'label'"),
        ([_VOLUME, {**_VOLUME, "label": "copy"}], ValueError, r"duplicates \['chg'\]"),
        ([_VOLUME, {"label": "chg", "values": []}], ValueError, r"duplicates"),
    ],
)
def test_widget_volumetric_data_rejects_invalid(
    bad_input: Any, err_cls: type[Exception], match: str
) -> None:
    """volumetric_data rejects non-list/non-mapping values, entries that are neither
    dicts nor pymatgen VolumetricData, non-string mapping prefixes, and missing or
    duplicate volume ids (layers pick volumes by id). Shape errors are left to the
    renderer.
    """
    with pytest.raises(err_cls, match=match):
        StructureWidget(volumetric_data=bad_input)
