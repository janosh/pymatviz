"""Tests for StructureWidget rendering and notebook integration."""

from __future__ import annotations

from typing import Any

import matscikit as mk
import numpy as np
import pytest
import traitlets as tl
from matscikit.interop import structure_to_pymatgen_dict

from pymatviz.widgets import StructureWidget
from tests.widgets.conftest import assert_widget_property_sync, decode_envelope


_SITE_PROPS = {  # pymatgen-format site data (magmoms, other site properties)
    "magmoms": [[0, 0, 1], [0, 0, -1]],
    "site_properties": {"force": [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]},
}


def _with_site_data(struct: mk.Structure, site_data: dict[str, Any]) -> mk.Structure:
    """Copy of struct carrying pymatgen-format site data in info["pymatgen"]."""
    struct = struct.copy()
    struct.info = {"pymatgen": site_data}
    return struct


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
    ("structure_fixture", "site_data"),
    [
        ("structures", None),
        ("fe3co4_disordered", None),
        ("fe3co4_disordered", _SITE_PROPS),
    ],
)
def test_widget_structure_inputs(
    structure_fixture: str,
    site_data: dict[str, Any] | None,
    request: pytest.FixtureRequest,
) -> None:
    """Widget must handle different structure input types and formats."""
    fixture_value = request.getfixturevalue(structure_fixture)

    # Handle both single structures and tuples of structures
    structures = fixture_value if isinstance(fixture_value, tuple) else (fixture_value,)

    for structure in structures:
        if site_data:
            structure = _with_site_data(structure, site_data)  # noqa: PLW2901
        widget = StructureWidget(structure=structure)
        assert widget.widget_type == "structure"
        structure_dict = structure_to_pymatgen_dict(structure)
        assert widget.structure == structure_dict

        widget = StructureWidget(structure=structure_dict)
        assert widget.structure == structure_dict

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
        ("show_bonds", ["always", "never", "crystals"]),
        ("color_scheme", ["Jmol", "Vesta", "Dark Mode"]),
        ("style", [None, "width: 400px; height: 600px", "width: 600px; height: 800px"]),
        ("show_controls", [True, False]),
        ("enable_info_pane", [True, False]),
        ("volumetric_data", [[]]),
    ],
)
def test_widget_property_sync_structure(
    structures: tuple[mk.Structure, mk.Structure],
    property_name: str,
    test_values: list[Any],
) -> None:
    """Widget properties must sync with frontend and handle various values."""
    widget = StructureWidget(structure=structures[0])

    for test_value in test_values:
        assert_widget_property_sync(widget, property_name, test_value)


def test_widget_handles_large_structures(
    structures: tuple[mk.Structure, mk.Structure],
) -> None:
    """Widget preserves all sites in a 2000-atom supercell."""
    widget = StructureWidget(structure=structures[0].make_supercell(10))
    assert len(widget.structure["sites"]) == 2000


_MULTI_VECTOR_PROPS = {
    "site_properties": {
        "force_DFT": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "force_MLFF": [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]],
    }
}


@pytest.mark.parametrize(
    ("site_data", "expected_keys"),
    [
        (_SITE_PROPS, {"magmom", "force"}),
        (_MULTI_VECTOR_PROPS, {"force_DFT", "force_MLFF"}),
    ],
    ids=["magmom-force", "multi-vector"],
)
def test_widget_preserves_site_properties(
    site_data: dict[str, Any],
    expected_keys: set[str],
    fe3co4_disordered: mk.Structure,
) -> None:
    """Disordered-structure site properties (incl. multiple vector keys) survive
    serialization.
    """
    struct = _with_site_data(fe3co4_disordered, site_data)
    widget = StructureWidget(structure=struct)
    uniq_prop_keys = {
        key for site in widget.structure["sites"] for key in site.get("properties", {})
    }
    assert uniq_prop_keys == expected_keys


def _meta(payload: dict[str, Any]) -> dict[str, Any]:
    """Envelope JSON fields (everything but the binary arrays and tags)."""
    return {k: v for k, v in payload.items() if k not in ("format", "kind", "arrays")}


def test_widget_volumetric_data_trait() -> None:
    """volumetric_data defaults to a fresh [], sends dict volumes as float32 binary
    envelopes (flat values + dims or legacy nested grid[ix][iy][iz], z fastest), passes
    ready envelopes through, uses a dict's label as missing id, prefixes ids/labels of
    mapping inputs and supports mutation.
    """
    default_widget = StructureWidget()
    assert default_widget.volumetric_data == []
    assert default_widget.volumetric_data is not StructureWidget().volumetric_data
    vol = {**_VOLUME, "label": "chg"}
    widget = StructureWidget(volumetric_data=[vol])
    (payload,) = widget.to_dict()["volumetric_data"]
    meta = {k: v for k, v in vol.items() if k not in ("values", "dims")}
    assert (payload["format"], payload["kind"], _meta(payload)) == (
        "matterviz-binary-v1",
        "volume",
        meta,
    )
    assert payload["arrays"]["values"]["dtype"] == "float32"
    np.testing.assert_array_equal(
        decode_envelope(payload)["values"],
        np.array([[[0.1]], [[0.2]]], dtype=np.float32),
    )
    elf = {**_VOLUME, "id": "elf", "periodic": False, "label": "elf"}
    widget.volumetric_data = [vol, elf]
    assert [_meta(pl)["id"] for pl in widget.volumetric_data] == ["chg", "elf"]
    # envelopes round-trip unchanged through the validator
    widget.volumetric_data = widget.volumetric_data
    assert widget.volumetric_data[0] == payload

    grid = np.random.default_rng(seed=0).random((2, 3, 4))
    legacy = {**meta, "grid": grid.tolist(), "grid_dims": [2, 3, 4], "data_order": "z"}
    (legacy_payload,) = StructureWidget(volumetric_data=[legacy]).volumetric_data
    (flat_payload,) = StructureWidget(
        volumetric_data=[
            {**meta, "values": grid.ravel().tolist(), "dims": [2.0, 3.0, 4.0]}
        ]
    ).volumetric_data
    (empty,) = StructureWidget(
        volumetric_data=[{**meta, "values": [], "dims": [0, 2, 2]}]
    ).volumetric_data
    assert decode_envelope(empty)["values"].shape == (0, 2, 2)
    assert legacy_payload == flat_payload
    values = decode_envelope(legacy_payload)["values"]
    assert values.shape == (2, 3, 4)
    np.testing.assert_allclose(values, grid, rtol=6e-8, atol=0)  # float32 ulp / 2

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
        (
            [
                _VOLUME,
                {**{k: v for k, v in _VOLUME.items() if k != "id"}, "label": "chg"},
            ],
            ValueError,
            "duplicates",
        ),
        ([{**_VOLUME, "dims": [3, 1, 1]}], ValueError, r"flat values \(2\) need dims"),
        ([{**_VOLUME, "dims": [0.5, 4, 1]}], ValueError, "need dims"),
        ([{**_VOLUME, "values": [0.1, np.nan]}], ValueError, "non-finite value at"),
        ([{**_VOLUME, "values": [0.1, 1e39]}], ValueError, "beyond float32 range"),
    ],
)
def test_widget_volumetric_data_rejects_invalid(
    bad_input: Any, err_cls: type[Exception], match: str
) -> None:
    """volumetric_data rejects non-list/non-mapping values, entries that are neither
    dicts nor pymatgen VolumetricData, non-string mapping prefixes, and missing or
    duplicate volume ids (layers pick volumes by id), values not matching dims,
    non-finite or float32-overflowing grids and malformed geometry fields.
    """
    with pytest.raises(err_cls, match=match):
        StructureWidget(volumetric_data=bad_input)


@pytest.mark.parametrize(
    ("layer", "volumes", "match"),
    [
        ({"volume_id": "chg"}, [_VOLUME], None),
        ({"volume_id": "chg", "color_volume_id": "chg"}, [_VOLUME], None),
        ({"volume_id": "nope"}, [_VOLUME], r"volume_id='nope' matches no"),
        (
            {"volume_id": "chg", "color_volume_id": "x"},
            [_VOLUME],
            "color_volume_id='x'",
        ),
        ({"isovalue": 0.1}, [], "needs a nonempty string volume_id"),
        # data_url volumes are parsed in the browser: ids can't be checked
        ({"volume_id": "charge density"}, [], None),
    ],
)
def test_widget_isosurface_layer_volume_ids(
    layer: dict[str, Any], volumes: list[dict[str, Any]], match: str | None
) -> None:
    """Isosurface layers must name a volume_id that matches a loaded volume."""
    settings = {"layers": [layer]}
    for kwargs in (  # both trait orders: cross-validation runs after __init__
        dict(isosurface_settings=settings, volumetric_data=volumes),
        dict(volumetric_data=volumes, isosurface_settings=settings),
    ):
        if match is None:
            assert StructureWidget(**kwargs).isosurface_settings == settings
        else:
            with pytest.raises(tl.TraitError, match=match):
                StructureWidget(**kwargs)


def test_widget_isosurface_volume_swap() -> None:
    """Volumes and layers swap atomically in hold_trait_notifications; a lone volume
    swap that orphans a layer raises and leaves the widget unchanged.
    """
    widget = StructureWidget(
        volumetric_data=[_VOLUME],
        isosurface_settings={"layers": [{"volume_id": "chg"}]},
    )
    with pytest.raises(tl.TraitError, match="volume_id='chg' matches no"):
        widget.volumetric_data = [{**_VOLUME, "id": "elf"}]
    assert [vol["id"] for vol in widget.volumetric_data] == ["chg"]

    with widget.hold_trait_notifications():
        widget.volumetric_data = [{**_VOLUME, "id": "elf"}]
        widget.isosurface_settings = {"layers": [{"volume_id": "elf"}]}
    assert [vol["id"] for vol in widget.volumetric_data] == ["elf"]


@pytest.mark.parametrize("cell_type", ["original", "conventional", "primitive"])
def test_widget_cell_type(cell_type: str) -> None:
    """cell_type reaches MatterViz, which standardizes; the input cell is synced."""
    struct = mk.Structure(
        np.eye(3) * 4,
        ["Fe", "O"],
        [[-0.1, 0, 0], [0.5, 0.5, 0.5]],
        fractional=True,
        info={"pymatgen": {"site_properties": {"force": [[1, 0, 0], [0, 1, 0]]}}},
    )
    widget = StructureWidget(structure=struct, cell_type=cell_type)
    assert widget.cell_type == cell_type
    assert widget.structure == structure_to_pymatgen_dict(struct)
