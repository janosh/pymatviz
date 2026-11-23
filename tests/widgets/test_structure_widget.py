"""Tests for StructureWidget rendering and notebook integration."""

from __future__ import annotations

from typing import Any

import kemio
import numpy as np
import pytest
import traitlets as tl
from kemio.interop import structure_to_pymatgen_dict

from pymatviz.widgets import StructureWidget, TrajectoryWidget
from tests.widgets.conftest import assert_widget_property_sync, decode_envelope


_SITE_PROPS = {  # typed magmoms plus a custom site property
    "magmoms": np.array([[0, 0, 1], [0, 0, -1]]),
    "force": np.array([[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]]),
}
_MULTI_VECTOR_PROPS = {
    "force_DFT": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    "force_MLFF": np.array([[0.9, 0.1, 0.0], [0.1, 0.9, 0.0]]),
}

# minimal flat matterviz VolumetricData payload (2x1x1 grid)
_VOLUME: dict[str, Any] = {
    "id": "chg",
    "values": [0.1, 0.2],
    "dims": [2, 1, 1],
    "lattice": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
    "origin": [0.0, 0.0, 0.0],
    "periodic": True,
}
_UNNAMED_VOLUME = {key: val for key, val in _VOLUME.items() if key != "id"}


@pytest.mark.parametrize(
    ("structure_fixture", "site_data", "prop_keys"),
    [
        ("structures", None, set()),
        ("fe3co4_disordered", None, set()),
        ("fe3co4_disordered", _SITE_PROPS, {"magmom", "force"}),
        ("fe3co4_disordered", _MULTI_VECTOR_PROPS, {"force_DFT", "force_MLFF"}),
    ],
)
def test_widget_structure_inputs(
    structure_fixture: str,
    site_data: dict[str, Any] | None,
    prop_keys: set[str],
    request: pytest.FixtureRequest,
) -> None:
    """Structures and their dicts sync via init or assignment, site properties
    (incl. several vector keys on a disordered structure) included.
    """
    fixture_value = request.getfixturevalue(structure_fixture)
    structures = fixture_value if isinstance(fixture_value, tuple) else (fixture_value,)

    for structure in structures:
        if site_data:
            structure = structure.replace(site_data=site_data)  # noqa: PLW2901
        widget = StructureWidget(structure=structure)
        assert widget.widget_type == "structure"
        structure_dict = structure_to_pymatgen_dict(structure)
        assert widget.structure == structure_dict
        sites = widget.structure["sites"]
        site_prop_keys = {key for site in sites for key in site.get("properties", {})}
        assert site_prop_keys == prop_keys

        widget = StructureWidget(structure=structure_dict)
        assert widget.structure == structure_dict

        widget = StructureWidget()
        assert widget.structure is None
        widget.structure = structure_dict
        assert widget.structure == structure_dict


@pytest.mark.parametrize("invalid_structure", ["invalid_structure", 123, ["invalid"]])
def test_widget_invalid_structure_handling(invalid_structure: Any) -> None:
    """Widget rejects non-Structure, non-dict inputs with TraitError."""
    type_name = type(invalid_structure).__name__
    with pytest.raises(
        tl.TraitError, match=f"structure must be a dict, .* got {type_name}$"
    ):
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
    structures: tuple[kemio.Structure, kemio.Structure],
    property_name: str,
    test_values: list[Any],
) -> None:
    """Widget properties must sync with frontend and handle various values."""
    widget = StructureWidget(structure=structures[0])

    for test_value in test_values:
        assert_widget_property_sync(widget, property_name, test_value)


def test_widget_handles_large_structures(
    structures: tuple[kemio.Structure, kemio.Structure],
) -> None:
    """Widget preserves all sites in a 2000-atom supercell."""
    widget = StructureWidget(structure=structures[0].make_supercell(10))
    assert len(widget.structure["sites"]) == 2000


def _meta(payload: dict[str, Any]) -> dict[str, Any]:
    """Envelope JSON fields (everything but the binary arrays and tags)."""
    return {k: v for k, v in payload.items() if k not in ("format", "kind", "arrays")}


@pytest.mark.parametrize("trait_name", ["selected_sites", "highlighted_sites"])
def test_site_indices_in_range_of_sites(
    fe3co4_disordered: kemio.Structure, trait_name: str
) -> None:
    """Python site indices must address the structure's sites; frontend ones index
    the rendered sites (image atoms appended) and pass; a new structure drops
    indices beyond its sites.
    """
    n_sites = len(fe3co4_disordered)
    indices: dict[str, Any] = {trait_name: [0, 1]}
    widget = StructureWidget(structure=fe3co4_disordered, **indices)
    for bad in ([0, n_sites], [-1]):
        bad_idx = bad[-1]
        with pytest.raises(
            tl.TraitError, match=rf"{trait_name}\[\d\]={bad_idx} .*0 to {n_sites - 1}"
        ):
            setattr(widget, trait_name, bad)
    assert getattr(widget, trait_name) == [0, 1]
    image_idx = n_sites + 3  # an image atom clicked in the viewer
    widget.set_state({trait_name: [1, image_idx]})
    assert getattr(widget, trait_name) == [1, image_idx]

    widget.structure = {"sites": [*widget.structure["sites"][:1]]}
    assert getattr(widget, trait_name) == []  # [1, image_idx] beyond the one site
    # standardized cells and structures Python doesn't hold have unknown site counts
    widget.cell_type = "conventional"
    setattr(widget, trait_name, [5])
    indices = {trait_name: [5]}
    no_structure = StructureWidget(structure_string="...", **indices)
    assert getattr(no_structure, trait_name) == [5]


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
    (labeled,) = StructureWidget(
        volumetric_data=[{**_UNNAMED_VOLUME, "label": "charge density"}]
    ).volumetric_data
    assert (labeled["id"], labeled["label"]) == ("charge density", "charge density")
    # an explicit id wins over the label
    widget.volumetric_data = ({**_VOLUME, "label": "rho"},)
    ids_labels = [(pl["id"], pl["label"]) for pl in widget.volumetric_data]
    assert ids_labels == [("chg", "rho")]
    # {prefix: volume} mappings namespace ids (and labels if present)
    widget.set_trait(
        "volumetric_data", {"a": _VOLUME, "b": {**_UNNAMED_VOLUME, "label": "chg"}}
    )
    assert [(pl["id"], pl.get("label")) for pl in widget.volumetric_data] == [
        ("a/chg", None),
        ("b/chg", "b/chg"),
    ]
    assert isinstance(widget.to_dict()["volumetric_data"], list)


@pytest.mark.parametrize(
    ("bad_input", "match"),
    [
        ("not_a_list", "list of volumes or a {prefix: volume} mapping"),
        (42, "got int"),
        (["not_a_dict"], "volumetric_data entries"),
        ([42], "volumetric_data entries"),
        ({"a": 42}, "volumetric_data entries"),
        ({"": _VOLUME}, "nonempty strings"),
        ({1: _VOLUME}, "nonempty strings"),
        ([_UNNAMED_VOLUME], "'id' or"),
        ([{**_VOLUME, "id": " "}], "'id' or 'label'"),
        ([{**_VOLUME, "id": 1, "label": "chg"}], "'id' or 'label'"),
        ([_VOLUME, {**_VOLUME, "label": "copy"}], r"duplicates \['chg'\]"),
        ([_VOLUME, {**_UNNAMED_VOLUME, "label": "chg"}], "duplicates"),
        ([{**_VOLUME, "dims": [3, 1, 1]}], r"flat values \(2\) need dims"),
        ([{**_VOLUME, "dims": [0.5, 4, 1]}], "need dims"),
        ([{**_VOLUME, "values": [0.1, np.nan]}], "non-finite value at"),
        ([{**_VOLUME, "values": [0.1, 1e39]}], "beyond float32 range"),
    ],
)
def test_widget_volumetric_data_rejects_invalid(bad_input: Any, match: str) -> None:
    """volumetric_data rejects non-list/non-mapping values, entries that are neither
    dicts nor pymatgen VolumetricData, non-string mapping prefixes, and missing or
    duplicate volume ids (layers pick volumes by id), values not matching dims,
    non-finite or float32-overflowing grids and malformed geometry fields.
    """
    with pytest.raises(tl.TraitError, match=match):
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
        # raw inputs whose ids only exist after normalizing: label-only dicts and
        # {prefix: volume} mappings (layers are checked once volumes are normalized)
        ({"volume_id": "rho"}, [{**_UNNAMED_VOLUME, "label": "rho"}], None),
        ({"volume_id": "run/chg"}, {"run": _VOLUME}, None),
        ({"volume_id": "chg"}, {"run": _VOLUME}, r"volume_id='chg' matches no"),
    ],
)
def test_widget_isosurface_layer_volume_ids(
    layer: dict[str, Any],
    volumes: list[dict[str, Any]] | dict[str, dict[str, Any]],
    match: str | None,
) -> None:
    """Isosurface layers must name a volume_id that matches a loaded volume."""
    settings = {"layers": [layer]}
    # both trait orders: cross-validation runs after __init__
    trait_orders: list[dict[str, Any]] = [
        dict(isosurface_settings=settings, volumetric_data=volumes),
        dict(volumetric_data=volumes, isosurface_settings=settings),
    ]
    for kwargs in trait_orders:
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
    struct = kemio.Structure(
        np.eye(3) * 4,
        ["Fe", "O"],
        [[-0.1, 0, 0], [0.5, 0.5, 0.5]],
        fractional=True,
        site_data={"force": np.array([[1, 0, 0], [0, 1, 0]])},
    )
    widget = StructureWidget(structure=struct, cell_type=cell_type)
    assert widget.cell_type == cell_type
    assert widget.structure == structure_to_pymatgen_dict(struct)


def test_structure_widget_isosurface_settings() -> None:
    """StructureWidget passes layers-only isosurface_settings through to_dict."""
    layer = dict(isovalue=0.05, color="#3b82f6", opacity=0.6, visible=True)
    layer |= dict(show_negative=False, negative_color="#ef4444", volume_id="total")
    iso_settings = {"layers": [layer], "wireframe": False, "halo": 0}
    state = StructureWidget(
        isosurface_settings=iso_settings, active_volume_id="total"
    ).to_dict()
    assert state["isosurface_settings"] == iso_settings
    assert state["active_volume_id"] == "total"
    assert StructureWidget().active_volume_id is None
    assert state["widget_type"] == "structure"


@pytest.mark.parametrize("widget_cls", [StructureWidget, TrajectoryWidget])
def test_structure_viz_vector_props(
    widget_cls: type[StructureWidget | TrajectoryWidget],
) -> None:
    """Structure widgets sync vector_configs and all vector props; configs default to
    None (frontend auto-detects from site properties) and update after construction.
    """
    configs = {
        "force_DFT": {"visible": True, "color": "#e74c3c", "scale": None},
        "force_MLFF": {"visible": True, "color": "#3498db", "scale": 1.5},
    }
    vector_props: dict[str, Any] = dict(
        vector_configs=configs,
        vector_scale=3.0,
        vector_color="#00ff00",
        vector_normalize=True,
        vector_uniform_thickness=True,
        vector_origin_gap=0.25,
    )
    widget = widget_cls(**vector_props)
    assert {key: widget.to_dict()[key] for key in vector_props} == vector_props

    widget.vector_configs = {**configs, "force_MLFF": {"visible": False}}
    assert widget.to_dict()["vector_configs"]["force_MLFF"] == {"visible": False}
    assert widget_cls().vector_configs is None
    assert widget_cls(vector_configs={}).vector_configs == {}
