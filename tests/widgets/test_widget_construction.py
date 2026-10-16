"""Tests for widget construction, traitlet defaults, to_dict, and validation."""

from __future__ import annotations

import importlib.util
import json
from contextlib import nullcontext
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import traitlets as tl
from pymatgen.core import Lattice, Structure

from pymatviz.widgets._traits import BONDING_STRATEGIES, ELEMENT_COLOR_SCHEMES
from pymatviz.widgets.band_structure import BandStructureWidget
from pymatviz.widgets.bands_and_dos import BandsAndDosWidget
from pymatviz.widgets.bar_plot import BarPlotWidget
from pymatviz.widgets.brillouin_zone import BrillouinZoneWidget
from pymatviz.widgets.chem_pot_diagram import ChemPotDiagramWidget
from pymatviz.widgets.composition import CompositionWidget
from pymatviz.widgets.convex_hull import ConvexHullWidget
from pymatviz.widgets.dos import DosWidget
from pymatviz.widgets.fermi_surface import FermiSurfaceWidget
from pymatviz.widgets.heatmap_matrix import HeatmapMatrixWidget
from pymatviz.widgets.histogram import HistogramWidget
from pymatviz.widgets.matterviz import MatterVizWidget
from pymatviz.widgets.periodic_table import PeriodicTableWidget
from pymatviz.widgets.phase_diagram import PhaseDiagramWidget
from pymatviz.widgets.rdf_plot import RdfPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from pymatviz.widgets.scatter_plot_3d import ScatterPlot3DWidget
from pymatviz.widgets.spacegroup_bar import SpacegroupBarPlotWidget
from pymatviz.widgets.structure import StructureWidget
from pymatviz.widgets.trajectory import TrajectoryWidget
from pymatviz.widgets.treemap import TreemapWidget
from pymatviz.widgets.xrd import XrdWidget


# === Construction and widget_type ===


@pytest.mark.parametrize(
    ("widget_cls", "kwargs", "expected_type", "state_key"),
    [
        (
            ConvexHullWidget,
            {"entries": [{"composition": {"Li": 1}, "energy": -1.5}]},
            "convex_hull",
            "entries",
        ),
        (
            BandStructureWidget,
            {"band_structs": {"bands": []}},
            "band_structure",
            "band_structs",
        ),
        (DosWidget, {"doses": {"energies": [0]}}, "dos", "doses"),
        (
            BandsAndDosWidget,
            {"band_structs": {"bands": []}, "doses": {"energies": []}},
            "bands_and_dos",
            "doses",
        ),
        (
            FermiSurfaceWidget,
            {"fermi_data": {"isosurfaces": []}},
            "fermi_surface",
            "fermi_data",
        ),
        (
            FermiSurfaceWidget,
            {"band_data": {"energies": []}},
            "fermi_surface",
            "band_data",
        ),
        (
            BrillouinZoneWidget,
            {"structure": {"lattice": {}, "sites": []}},
            "brillouin_zone",
            "structure",
        ),
        (
            PhaseDiagramWidget,
            {"data": {"components": ["A", "B"]}},
            "phase_diagram",
            "data",
        ),
        (
            XrdWidget,
            {"patterns": {"x": [10], "y": [100], "kind": "profile"}},
            "xrd",
            "patterns",
        ),
        (
            ScatterPlotWidget,
            {"series": [{"x": [0, 1], "y": [1, 2], "label": "curve"}]},
            "scatter_plot",
            "series",
        ),
        (
            BarPlotWidget,
            {"series": [{"x": [0, 1], "y": [2, 3], "label": "bars"}]},
            "bar_plot",
            "series",
        ),
        (
            HistogramWidget,
            {"series": [{"values": [2, 2.5], "label": "hist"}]},
            "histogram",
            "series",
        ),
        (
            PeriodicTableWidget,
            {"heatmap_values": {"Fe": 42, "O": 100}},
            "periodic_table",
            "heatmap_values",
        ),
        (
            RdfPlotWidget,
            {"structures": {"lattice": {}, "sites": []}},
            "rdf_plot",
            "structures",
        ),
        (
            ScatterPlot3DWidget,
            {"series": [{"x": [1], "y": [2], "z": [3], "label": "pt"}]},
            "scatter_plot_3d",
            "series",
        ),
        (
            HeatmapMatrixWidget,
            {"x_items": ["A", "B"], "y_items": ["C", "D"], "values": [[1, 2], [3, 4]]},
            "heatmap_matrix",
            "values",
        ),
        (
            PeriodicTableWidget,
            {"heatmap_values": [1.0, 4.0, 6.9]},
            "periodic_table",
            "heatmap_values",
        ),
        (
            HeatmapMatrixWidget,
            {"values": {"Fe": {"Fe": 1.0, "O": 0.5}, "O": {"Fe": 0.5, "O": 1.0}}},
            "heatmap_matrix",
            "values",
        ),
        (
            PeriodicTableWidget,
            {"color_scale": {"type": "log", "value_range": [1, None]}},
            "periodic_table",
            "color_scale",
        ),
        (PeriodicTableWidget, {"color_bar": None}, "periodic_table", "color_bar"),
        (
            HeatmapMatrixWidget,
            {"x_items": ["A"], "y_items": ["B"], "color_scale": {"type": "log"}},
            "heatmap_matrix",
            "color_scale",
        ),
        (
            HeatmapMatrixWidget,
            {"color_bar": {"title": "count", "orientation": "vertical"}},
            "heatmap_matrix",
            "color_bar",
        ),
        (SpacegroupBarPlotWidget, {"data": [225, 166, 62]}, "spacegroup_bar", "data"),
        (
            ChemPotDiagramWidget,
            {"entries": [{"name": "Li2O", "energy": -14.3}]},
            "chem_pot_diagram",
            "entries",
        ),
        (
            TreemapWidget,
            {"data": {"label": "root", "children": [{"label": "leaf", "value": 3}]}},
            "treemap",
            "data",
        ),
    ],
)
def test_widget_construction_and_type(
    widget_cls: type,
    kwargs: dict[str, Any],
    expected_type: str,
    state_key: str,
) -> None:
    """Widget constructors preserve expected normalized state."""
    widget = widget_cls(**kwargs)
    assert widget.widget_type == expected_type
    assert getattr(widget, state_key) == kwargs[state_key]


@pytest.mark.parametrize(
    "widget_cls", [CompositionWidget, StructureWidget, TrajectoryWidget]
)
def test_widget_notebook_integration(widget_cls: type[MatterVizWidget]) -> None:
    """Widgets expose a Jupyter widget-view MIME bundle backed by the real ESM/CSS."""
    widget = widget_cls()
    # ty mis-resolves _repr_mimebundle_ as unbound; it is bound at runtime
    mime = widget._repr_mimebundle_()  # ty: ignore[missing-argument]
    assert mime is not None
    # accept dict or (data, metadata)
    mime_data, metadata = mime if isinstance(mime, tuple) else (mime, {})
    assert isinstance(metadata, dict)
    view = mime_data["application/vnd.jupyter.widget-view+json"]
    assert isinstance(view["model_id"], str)
    assert view["model_id"]
    assert isinstance(view["version_major"], int)
    assert widget_cls.__name__ in str(mime_data["text/plain"])

    for attr_name in ("_model_name", "_view_name", "_model_module", "_view_module"):
        assert hasattr(widget, attr_name)
    assert len(widget._esm) > 1000
    assert len(widget._css) > 100
    assert "export default" in widget._esm or " as default" in widget._esm
    assert "render" in widget._esm


# === to_dict auto-discovery ===


@pytest.mark.parametrize(
    ("widget_cls", "kwargs"),
    [
        (ScatterPlotWidget, {"series": [{"x": [0, 1], "y": [1, 2], "label": "s"}]}),
        (
            HistogramWidget,
            {"series": [{"values": [1, 2]}], "bins": 50, "mode": "overlay"},
        ),
        (BarPlotWidget, {"series": [{"x": [0], "y": [1]}], "mode": "grouped"}),
        (DosWidget, {"doses": {"energies": [0, 1]}, "sigma": 0.05}),
        (ConvexHullWidget, {"entries": [{"composition": {"Li": 1}, "energy": -1.5}]}),
        (PeriodicTableWidget, {"heatmap_values": {"Fe": 42}}),
        (ScatterPlot3DWidget, {"series": [{"x": [1], "y": [2], "z": [3]}]}),
        (HeatmapMatrixWidget, {"x_items": ["A"], "y_items": ["B"]}),
        (SpacegroupBarPlotWidget, {"data": [225]}),
        (ChemPotDiagramWidget, {"entries": [{"name": "Li", "energy": -1.9}]}),
        (TreemapWidget, {"data": {"label": "root", "value": 1}}),
    ],
)
def test_to_dict_includes_subclass_fields(
    widget_cls: type,
    kwargs: dict[str, Any],
) -> None:
    """to_dict returns all public synced traitlets."""
    widget = widget_cls(**kwargs)
    state = widget.to_dict()
    expected = {
        name
        for name in widget.traits(sync=True)
        if not name.startswith("_") and name not in widget._EXCLUDED_TRAITS
    }
    assert {"widget_type", "style", "show_controls"} <= set(state)
    assert set(state) == expected


def test_to_dict_reflects_constructor_values() -> None:
    """to_dict values match what was passed to the constructor."""
    widget = HistogramWidget(
        series=[{"values": [2, 3], "label": "h"}],
        bins=25,
        mode="overlay",
        selected_series_idx=1,
        x_axis={"label": "Value"},
    )
    state = widget.to_dict()
    assert state["bins"] == 25
    assert state["mode"] == "overlay"
    assert state["selected_series_idx"] == 1
    assert state["series"] == [{"values": [2.0, 3.0], "label": "h"}]
    assert state["x_axis"] == {"label": "Value"}


def test_to_dict_reflects_runtime_mutations() -> None:
    """to_dict picks up traitlet mutations after construction."""
    widget = ScatterPlotWidget(series=[{"x": [0], "y": [1]}])
    assert widget.to_dict()["x_axis"] is None

    widget.x_axis = {"label": "updated"}
    assert widget.to_dict()["x_axis"] == {"label": "updated"}

    widget.style = "height: 999px;"
    assert widget.to_dict()["style"] == "height: 999px;"


# === Traitlet defaults ===


@pytest.mark.parametrize(
    ("widget_cls", "attr", "expected"),
    [
        (BarPlotWidget, "series", None),
        (BarPlotWidget, "orientation", "vertical"),
        (BarPlotWidget, "mode", "overlay"),
        (HistogramWidget, "bins", 100),
        (HistogramWidget, "mode", "single"),
        (HistogramWidget, "show_legend", True),
        (PeriodicTableWidget, "color_scale", "interpolateViridis"),
        (PeriodicTableWidget, "color_bar", {}),  # shown by default
        (PeriodicTableWidget, "missing", None),
        (HeatmapMatrixWidget, "color_bar", None),  # hidden by default
        (HeatmapMatrixWidget, "missing", None),
        (HeatmapMatrixWidget, "values", None),  # None, not an empty list
        (HeatmapMatrixWidget, "x_items", []),
        (ChemPotDiagramWidget, "entries", []),
        (ScatterPlot3DWidget, "camera_projection", "perspective"),
        (HeatmapMatrixWidget, "color_scale", "interpolateViridis"),
        (HeatmapMatrixWidget, "tile_size", "50px"),
        (HeatmapMatrixWidget, "gap", "0px"),
        (SpacegroupBarPlotWidget, "data", []),
        (SpacegroupBarPlotWidget, "show_counts", True),
        (SpacegroupBarPlotWidget, "show_legend", False),
        (SpacegroupBarPlotWidget, "orientation", "vertical"),
        (RdfPlotWidget, "cutoff", 15),
        (RdfPlotWidget, "n_bins", 75),
        (RdfPlotWidget, "mode", "element_pairs"),
    ],
)
def test_widget_traitlet_defaults(widget_cls: type, attr: str, expected: Any) -> None:
    """Widgets construct without data and expose correct default traitlet values."""
    assert widget_cls().to_dict()[attr] == expected


def test_treemap_widget_all_config_traits_default_to_none() -> None:
    """Every treemap config trait defaults to None so frontend defaults apply.

    Guards against container traits regressing to []/{} defaults (traitlets
    List/Dict need explicit default_value=None) which would override the
    component's own defaults.
    """
    # controls_open is two-way view state mirroring the frontend's false default
    base_traits = {"widget_type", "style", "show_controls", "controls_open"}
    state = TreemapWidget().to_dict()
    non_none = {key for key, val in state.items() if val is not None} - base_traits
    assert not non_none, f"expected None defaults, got non-None for {sorted(non_none)}"


# === Input normalization at widget level ===


def test_histogram_widget_delegates_series_validation() -> None:
    """HistogramWidget requires sample arrays in `values`."""
    widget = HistogramWidget(series=[{"values": np.array([2, 1]), "label": "a"}])
    assert widget.to_dict()["series"] == [{"values": [2.0, 1.0], "label": "a"}]
    with pytest.raises(ValueError, match="must include key 'values'"):
        HistogramWidget(series=[{"x": [0, 1], "y": [2, 3]}])


@pytest.mark.parametrize(
    "samples", [[2, 1], (2, 1), np.array([2, 1]), pd.Series([2, 1])]
)
@pytest.mark.parametrize("hold_notifications", [False, True])
def test_histogram_series_updates_normalize(
    samples: Any, hold_notifications: bool
) -> None:
    """Updated samples match constructor normalization and produce JSON-safe state."""
    series: Any = ({"values": samples, "label": "updated", "opacity": np.float32(0.5)},)
    widget = HistogramWidget(series=[{"values": [0]}])
    with widget.hold_trait_notifications() if hold_notifications else nullcontext():
        widget.series = series

    expected = [{"values": [2.0, 1.0], "label": "updated", "opacity": 0.5}]
    assert widget.series == HistogramWidget(series=series).series == expected
    assert (
        json.loads(json.dumps(widget.to_dict(), allow_nan=False))["series"] == expected
    )
    assert (
        json.loads(json.dumps(widget.get_state(), allow_nan=False))["series"]
        == expected
    )


@pytest.mark.parametrize(
    ("series", "match"),
    [
        ([{"x": [0, 1], "y": [2, 3]}], "must include key 'values'"),
        ([{"values": [float("nan")]}], "non-finite"),
        ([{"values": [float("inf")]}], "non-finite"),
        ([{"values": ["bad"]}], "must be numeric"),
        ([{"values": 1}], "must be list-like"),
        ([1], "entries must be dicts"),
    ],
)
@pytest.mark.parametrize("hold_notifications", [False, True])
def test_histogram_series_updates_reject_invalid(
    series: Any, match: str, hold_notifications: bool
) -> None:
    """Invalid updates leave valid state intact and never notify the frontend."""
    widget = HistogramWidget(series=[{"values": [1, 2]}])
    old_state = widget.to_dict()
    observer = Mock()
    widget.observe(observer, names="series")
    with (
        pytest.raises(tl.TraitError, match=match),
        widget.hold_trait_notifications() if hold_notifications else nullcontext(),
    ):
        widget.series = series
    assert widget.to_dict() == old_state
    observer.assert_not_called()


@pytest.mark.parametrize("series", [None, [], [{"values": []}]])
def test_histogram_series_updates_clear(series: Any) -> None:
    """Clearing or providing an empty sample array stays valid after construction."""
    widget = HistogramWidget(series=[{"values": [1, 2]}])
    widget.series = series
    assert widget.series == HistogramWidget(series=series).series == series


# === New widget specific behaviors ===


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


@pytest.mark.parametrize(
    ("widget_cls", "kwargs", "attr", "expected"),
    [
        (
            HeatmapMatrixWidget,
            {
                "x_items": ["A", "B"],
                "y_items": ["A", "B"],
                "values": np.array([[1.0, 0.5], [0.5, 1.0]]),
            },
            "values",
            [[1.0, 0.5], [0.5, 1.0]],
        ),
        (
            PeriodicTableWidget,
            {"heatmap_values": np.array([1.0, 4.0, 6.9])},
            "heatmap_values",
            [1.0, 4.0, 6.9],
        ),
        (
            SpacegroupBarPlotWidget,
            {"data": np.array([225, 166, 62])},
            "data",
            [225, 166, 62],
        ),
        (
            BrillouinZoneWidget,
            {"bz_data": {"vertices": np.eye(3)}},
            "bz_data",
            {"vertices": np.eye(3).tolist()},
        ),
        (
            FermiSurfaceWidget,
            {"fermi_data": {"vertices": np.eye(3)}},
            "fermi_data",
            {"vertices": np.eye(3).tolist()},
        ),
        (
            FermiSurfaceWidget,
            {"band_data": {"energies": np.zeros((1, 1, 2, 2, 2))}},
            "band_data",
            {"energies": np.zeros((1, 1, 2, 2, 2)).tolist()},
        ),
    ],
)
def test_widget_normalizes_numpy_values(
    widget_cls: type,
    kwargs: dict[str, Any],
    attr: str,
    expected: list[Any] | dict[str, Any],
) -> None:
    """Widgets normalize numpy arrays to plain Python lists."""
    widget = widget_cls(**kwargs)
    result = getattr(widget, attr)
    assert result == expected
    assert type(result) is type(expected)


def test_spacegroup_bar_data_inputs() -> None:
    """SpacegroupBarPlotWidget normalizes supported data inputs."""
    pd = pytest.importorskip("pandas")
    widget = SpacegroupBarPlotWidget(data=pd.Series([225, 166, 62]))
    assert widget.data == [225, 166, 62]
    widget = SpacegroupBarPlotWidget(data={225: np.int64(2), 166: 0, "P2_1/c": 1})
    assert widget.data == [225, 225, "P2_1/c"]
    for bad_data, error_cls, match in [
        ({225: -1}, ValueError, "non-negative"),
        ({225: 1.5}, TypeError, "integer"),
    ]:
        with pytest.raises(error_cls, match=match):
            SpacegroupBarPlotWidget(data=bad_data)


def test_treemap_widget_traits_and_validation() -> None:
    """TreemapWidget normalizes data, validates enums, and syncs zoom state."""
    data = {"label": "root", "children": [{"label": "a", "value": np.float64(2.5)}]}
    config = {
        "value_mode": "total",
        "label_fit": "shrink",
        "label_min_font_size": 7,
        "label_max_font_size": 14,
        "parent_label_font_size": 16,
        "max_children": 3,
        "export_formats": ["svg", "png"],
        "zoom_root_id": "root/a",
    }
    widget = TreemapWidget(data=data, **config)
    leaf_value = widget.data["children"][0]["value"]
    assert leaf_value == 2.5
    assert type(leaf_value) is float  # numpy scalar normalized
    state = widget.to_dict()
    for key, expected in config.items():
        assert state[key] == expected

    # zoom_root_id is two-way synced: assignment after construction must work
    widget.zoom_root_id = None
    assert widget.zoom_root_id is None

    for bad_key in ("value_mode", "label_text", "label_fit"):
        with pytest.raises(tl.TraitError):
            TreemapWidget(data=data, **{bad_key: "not-an-option"})
    with pytest.raises(tl.TraitError):
        TreemapWidget(data=data, max_children=-1)
    with pytest.raises(tl.TraitError):  # not a matterviz ExportFormat
        TreemapWidget(data=data, export_formats=["pdf"])
    assert TreemapWidget(data=data, export_formats=[]).export_formats == []

    # forest input: list of root nodes passes through unchanged
    roots = [{"label": "x", "value": 1}, {"label": "y", "value": 2}]
    assert TreemapWidget(data=roots).to_dict()["data"] == roots


_COLOR_SCHEME_CASES = [
    *[(name, name) for name in ELEMENT_COLOR_SCHEMES],
    ("dark mode", "Dark Mode"),  # caseless, stored canonical
    ("CPK", tl.TraitError),  # never a matterviz scheme
    ("", tl.TraitError),
]
_BONDING_STRATEGY_CASES = [
    *[(name, name) for name in BONDING_STRATEGIES],
    (None, None),
    ("solid_angle", tl.TraitError),  # removed from matterviz
]


@pytest.mark.parametrize(
    ("widget_cls", "trait_name", "value", "expected"),
    [
        *[
            (cls, "color_scheme", *case)
            for cls in (StructureWidget, TrajectoryWidget, CompositionWidget)
            for case in _COLOR_SCHEME_CASES
        ],
        *[
            (cls, "bonding_strategy", *case)
            for cls in (StructureWidget, TrajectoryWidget)
            for case in _BONDING_STRATEGY_CASES
        ],
    ],
)
def test_matterviz_name_enums(
    widget_cls: type[MatterVizWidget],
    trait_name: str,
    value: str | None,
    expected: str | type[Exception] | None,
) -> None:
    """Scheme/strategy traits accept exactly matterviz's names, canonicalized."""
    widget = widget_cls()
    if isinstance(expected, type):
        with pytest.raises(expected):
            setattr(widget, trait_name, value)
    else:
        setattr(widget, trait_name, value)
        assert widget.to_dict()[trait_name] == expected


@pytest.mark.parametrize("widget_cls", [DosWidget, BandsAndDosWidget])
@pytest.mark.parametrize(
    ("sigma", "expected"), [(None, None), (0.0, 0.0), (0.2, 0.2), (-0.1, tl.TraitError)]
)
def test_dos_sigma_in_data_unit(
    widget_cls: type[MatterVizWidget],
    sigma: float | None,
    expected: float | type[Exception] | None,
) -> None:
    """DOS sigma syncs unconverted (THz/eV data unit) and rejects negatives."""
    if isinstance(expected, type):
        with pytest.raises(expected):
            widget_cls(sigma=sigma)
    else:
        assert widget_cls(sigma=sigma).to_dict()["sigma"] == expected


def test_heatmap_matrix_scalar_string_axis_items() -> None:
    """HeatmapMatrixWidget treats scalar string axis items as one label."""
    widget = HeatmapMatrixWidget(x_items="Fe", y_items="O")
    assert widget.x_items == [{"key": "Fe", "label": "Fe"}]
    assert widget.y_items == [{"key": "O", "label": "O"}]


@pytest.mark.parametrize("widget_cls", [PeriodicTableWidget, HeatmapMatrixWidget])
def test_color_scale_config_normalized_and_validated(widget_cls: type) -> None:
    """color_scale accepts a ColorScaleConfig dict with numpy bounds, not an int."""
    color_scale = {
        "scheme": "interpolateBlues",
        "type": "log",
        "value_range": [np.float64(0.5), None],
        "palette_range": np.array([0.25, 1.0]),
    }
    synced = widget_cls(color_scale=color_scale).to_dict()["color_scale"]
    assert synced == {
        **color_scale,
        "value_range": [0.5, None],
        "palette_range": [0.25, 1],
    }
    assert type(synced["value_range"][0]) is float
    with pytest.raises(tl.TraitError):
        widget_cls(color_scale=1)


def test_rdf_plot_accepts_pymatgen_structure() -> None:
    """RdfPlotWidget normalizes pymatgen Structure to dict via _to_dict."""
    struct = Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])
    widget = RdfPlotWidget(structures=struct)
    assert isinstance(widget.structures, dict)
    assert "lattice" in widget.structures


def test_rdf_plot_rejects_both_structures_and_patterns() -> None:
    """RdfPlotWidget rejects simultaneous structures and patterns."""
    with pytest.raises(ValueError, match="not both"):
        RdfPlotWidget(
            structures={"lattice": {}, "sites": []},
            patterns=[{"r": [1], "g_r": [1]}],
        )


@pytest.mark.parametrize(
    ("series", "exc_type", "match"),
    [
        ([{"x": [1], "y": [2], "label": "no-z"}], ValueError, "missing required key"),
        (["not-a-dict"], TypeError, "must be a dict"),
        ([{1: [0], "y": [1], "z": [2]}], TypeError, "non-string keys"),
    ],
)
def test_scatter_plot_3d_rejects_invalid_series(
    series: list[Any], exc_type: type[Exception], match: str
) -> None:
    """ScatterPlot3DWidget rejects malformed series entries."""
    with pytest.raises(exc_type, match=match):
        ScatterPlot3DWidget(series=series)


# === FermiSurfaceWidget input validation ===


@pytest.mark.parametrize(
    ("fermi_data", "band_data"),
    [
        (None, None),
        ({"isosurfaces": []}, {"energies": []}),
    ],
)
def test_fermi_surface_widget_invalid_input_combo(
    fermi_data: dict[str, Any] | None, band_data: dict[str, Any] | None
) -> None:
    """FermiSurfaceWidget rejects ambiguous or missing input sources."""
    with pytest.raises(ValueError, match="exactly one of fermi_data or band_data"):
        FermiSurfaceWidget(fermi_data=fermi_data, band_data=band_data)


@pytest.mark.parametrize(
    "widget_cls",
    [ScatterPlotWidget, BarPlotWidget, HistogramWidget],
)
def test_show_not_shadowed_by_display_traitlet(widget_cls: type) -> None:
    """show() remains callable on widgets that define a 'display' traitlet."""
    widget = widget_cls(display={"x_grid": True})
    with patch("IPython.display.display") as mock_display:
        assert widget.show() is None
    mock_display.assert_called_once_with(widget)


# === to_img ===

_HEADLESS = "pymatviz.widgets._headless"
_FAKE_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50


def _make_widget() -> MatterVizWidget:
    """Create a MatterVizWidget with mocked assets for testing."""
    with patch("pymatviz.widgets.matterviz.fetch_widget_asset", return_value="x"):
        return MatterVizWidget(widget_type="scatter_plot")


@pytest.mark.parametrize(
    ("filename", "explicit_fmt", "expected_capture_fmt"),
    [
        ("plot.png", None, "png"),
        ("plot.svg", None, "svg"),
        ("plot.pdf", None, "pdf"),
        ("plot.jpg", None, "jpeg"),
        ("plot.jpeg", None, "jpeg"),
        (None, "png", "png"),
        (None, "svg", "svg"),
        (None, "jpeg", "jpeg"),
        (None, "pdf", "pdf"),
        (None, None, "png"),
    ],
)
def test_to_img_format_inference(
    filename: str | None,
    explicit_fmt: str | None,
    expected_capture_fmt: str,
    tmp_path: Any,
) -> None:
    """to_img infers capture format from filename or explicit fmt argument."""
    widget = _make_widget()
    resolved = f"{tmp_path}/{filename}" if filename else None

    with patch(
        f"{_HEADLESS}.render_widget_headless", return_value=_FAKE_PNG
    ) as mock_render:
        widget.to_img(filename=resolved, fmt=explicit_fmt)  # ty: ignore[invalid-argument-type]

    assert mock_render.call_args.kwargs["fmt"] == expected_capture_fmt


@pytest.mark.parametrize(
    ("img_kwargs", "match"),
    [
        ({"filename": "plot.tiff"}, "Unsupported format 'tiff'"),
        ({"fmt": "tiff"}, "Unsupported format 'tiff'"),
        ({"filename": "plot"}, "Cannot infer format"),
        ({"dpi": 0}, "dpi must be positive"),
        ({"width": -1}, "width must be positive"),
        ({"quality": 101}, "quality must be between 1 and 100"),
    ],
)
def test_to_img_rejects_invalid_args(
    tmp_path: Any, img_kwargs: dict[str, Any], match: str
) -> None:
    """to_img raises ValueError for bad formats and render settings."""
    if filename := img_kwargs.get("filename"):
        img_kwargs = {**img_kwargs, "filename": f"{tmp_path}/{filename}"}
    with pytest.raises(ValueError, match=match):
        _make_widget().to_img(**img_kwargs)


@pytest.mark.parametrize("path_suffix", ["output.png", "sub/dir/output.png"])
def test_to_img_writes_file(tmp_path: Any, path_suffix: str) -> None:
    """to_img writes bytes to disk, creating parent directories as needed."""
    widget = _make_widget()

    with patch(f"{_HEADLESS}.render_widget_headless", return_value=_FAKE_PNG):
        out_path = f"{tmp_path}/{path_suffix}"
        result = widget.to_img(filename=out_path)

    assert result == _FAKE_PNG
    with open(out_path, "rb") as fh:
        assert fh.read() == _FAKE_PNG


@pytest.mark.parametrize(
    "img_kwargs",
    [
        {"fmt": "pdf", "dpi": 200},
        {"fmt": "jpeg", "quality": 75},
        {"fmt": "png", "width": 1200, "height": None},
        {"fmt": "png", "width": None, "height": 900},
        {"fmt": "png", "width": 1200, "height": 900},
    ],
    ids=["pdf-dpi", "jpeg-quality", "width", "height", "width-height"],
)
def test_to_img_forwards_render_kwargs(img_kwargs: dict[str, Any]) -> None:
    """to_img forwards fmt/dpi/quality/width/height to the headless renderer and
    returns its bytes unchanged.
    """
    widget = _make_widget()
    with patch(
        f"{_HEADLESS}.render_widget_headless", return_value=b"%PDF-fake"
    ) as mock_render:
        assert widget.to_img(**img_kwargs) == b"%PDF-fake"
    for key, value in img_kwargs.items():
        assert mock_render.call_args.kwargs[key] == value


def test_to_img_falls_back_from_marimo_url() -> None:
    """to_img uses class assets when instance _esm is a marimo virtual-file URL."""
    widget = _make_widget()
    widget._esm = "./@file/virtual-esm-url.js"

    with patch(
        f"{_HEADLESS}.render_widget_headless", return_value=_FAKE_PNG
    ) as mock_render:
        widget.to_img(fmt="png")

    esm_passed = mock_render.call_args.kwargs["esm_content"]
    assert esm_passed != "./@file/virtual-esm-url.js"
    assert isinstance(esm_passed, str)


# === _build_html unit tests ===


_W8 = "width: 800px"
_H6 = "height: 600px"


@pytest.mark.parametrize(
    ("style", "expect_in", "expect_not_in"),
    [
        (None, [_W8, _H6], []),
        ("", [_W8, _H6], []),
        (
            "width: 500px; height: 300px;",
            ["width: 500px", "height: 300px"],
            ["800px", "600px"],
        ),
        ("height: 400px;", [_W8, "height: 400px"], ["600px"]),
        ("width: 400px;", ["width: 400px", _H6], ["800px"]),
    ],
)
def test_build_html_dimensions(
    style: str | None,
    expect_in: list[str],
    expect_not_in: list[str],
) -> None:
    """_build_html applies dimension defaults only for missing width/height."""
    from pymatviz.widgets._headless import _build_html

    data: dict[str, Any] = {"widget_type": "bar_plot"}
    if style is not None:
        data["style"] = style
    html = _build_html(data, "// esm", "/* css */")
    assert 'id="widget-root"' in html
    for substr in expect_in:
        assert substr in html, f"expected {substr!r} in HTML"
    for substr in expect_not_in:
        assert substr not in html, f"unexpected {substr!r} in HTML"


@pytest.mark.parametrize(
    ("width", "height", "expect_in"),
    [
        (1200, None, ["width: 1200px", _H6]),
        (None, 900, ["width: 500px", "height: 900px"]),
        (1200, 900, ["width: 1200px", "height: 900px"]),
    ],
)
def test_build_html_width_height_override(
    width: int | None,
    height: int | None,
    expect_in: list[str],
) -> None:
    """_build_html explicit width/height override user style and defaults."""
    from pymatviz.widgets._headless import _build_html

    data: dict[str, Any] = {"widget_type": "bar_plot", "style": "width: 500px;"}
    html = _build_html(data, "// esm", "/* css */", width=width, height=height)
    for substr in expect_in:
        assert substr in html, f"expected {substr!r} in HTML"


def test_shutdown_browser_tolerates_none() -> None:
    """_shutdown_browser handles _browser/_pw/_pw_cm all None without error."""
    import pymatviz.widgets._headless as headless_mod
    from pymatviz.widgets._headless import _shutdown_browser

    with (
        patch.object(headless_mod, "_browser", None),
        patch.object(headless_mod, "_pw", None),
        patch.object(headless_mod, "_pw_cm", None),
    ):
        _shutdown_browser()


def test_get_browser_import_error() -> None:
    """_get_browser raises ImportError with install instructions."""
    import pymatviz.widgets._headless as headless_mod

    original = headless_mod._browser
    headless_mod._browser = None
    blocked = {"playwright": None, "playwright.sync_api": None}
    try:
        with (
            patch.dict("sys.modules", blocked),
            pytest.raises(ImportError, match="playwright is required"),
        ):
            headless_mod._get_browser()
    finally:
        headless_mod._browser = original


@pytest.mark.parametrize(
    ("widget_type", "expect_html_fallback", "expect_worker_shim"),
    [
        ("periodic_table", True, False),
        ("heatmap_matrix", True, False),
        ("scatter_plot", False, False),
        ("structure", False, False),
        ("chem_pot_diagram", False, True),
    ],
)
def test_build_html_render_gates(
    widget_type: str, expect_html_fallback: bool, expect_worker_shim: bool
) -> None:
    """_build_html toggles HTML fallback and Worker shim per widget type."""
    from pymatviz.widgets._headless import _build_html

    html = _build_html({"widget_type": widget_type}, "// esm", "/* css */")
    assert ("const allow_html_fallback = true" in html) is expect_html_fallback
    assert ("globalThis.Worker = undefined" in html) is expect_worker_shim
    # Spinner guard is always present so loading states are never captured
    assert '.spinner[role="status"]' in html


# === Headless integration tests (require playwright + chromium) ===

_has_playwright = importlib.util.find_spec("playwright") is not None
_has_pillow = importlib.util.find_spec("PIL") is not None
_skip_no_playwright = pytest.mark.skipif(
    not _has_playwright, reason="playwright not installed"
)
_skip_no_pillow = pytest.mark.skipif(not _has_pillow, reason="Pillow not installed")


@pytest.fixture
def bar_plot_widget() -> BarPlotWidget:
    """A minimal BarPlotWidget for integration tests."""
    return BarPlotWidget(
        series=[{"x": [0, 1], "y": [3.0, 5.0], "label": "test"}],
        style="height: 400px;",
    )


@_skip_no_playwright
@_skip_no_pillow
def test_headless_bar_plot_png(bar_plot_widget: BarPlotWidget) -> None:
    """BarPlotWidget.to_img(fmt='png') produces a valid, non-trivial PNG."""
    import io

    from PIL import Image

    png_bytes = bar_plot_widget.to_img(fmt="png", dpi=72)
    assert png_bytes[:4] == b"\x89PNG"
    img = Image.open(io.BytesIO(png_bytes))
    assert img.width >= 400
    assert img.height >= 200


@_skip_no_playwright
def test_headless_bar_plot_svg(bar_plot_widget: BarPlotWidget) -> None:
    """BarPlotWidget.to_img(fmt='svg') returns chart SVG, not a toolbar icon."""
    svg_text = bar_plot_widget.to_img(fmt="svg").decode("utf-8")
    assert svg_text.startswith("<?xml")
    assert len(svg_text) > 2000, f"SVG too small ({len(svg_text)} chars)"
    assert "<g" in svg_text, "No <g> groups -- likely an icon"


@_skip_no_playwright
@_skip_no_pillow
@pytest.mark.parametrize("marker_renderer", ["svg", "canvas"])
def test_headless_scatter_vector_export(marker_renderer: str) -> None:
    """SVG keeps SVG markers as elements, canvas markers as <image>; PDF is vector."""
    import base64
    import io
    import re

    from PIL import Image

    widget = ScatterPlotWidget(
        series=[{"x": [0, 1, 2], "y": [0, 1, 4], "label": "points"}],
        marker_renderer=marker_renderer,
        styles={"point": {"color": "#ff0000", "size": 8}},
        style="width: 500px; height: 350px;",
    )
    assert b"/Font" in widget.to_img(fmt="pdf"), "PDF rasterized, expected vector"
    svg_text = widget.to_img(fmt="svg", dpi=144).decode("utf-8")
    assert "<foreignObject" not in svg_text
    if marker_renderer == "svg":
        assert svg_text.count('data-roving-key="') == 3
        return
    data_uris = re.findall(
        r'<image [^>]*href="data:image/png;base64,([^"]+)"', svg_text
    )
    assert len(data_uris) == 1, "canvas markers not inlined as <image>"
    img = Image.open(io.BytesIO(base64.b64decode(data_uris[0]))).convert("RGB")
    assert img.size == (1000, 700), "embedded canvas should follow dpi"
    pixels = np.asarray(img)
    red_pixels = (pixels[..., 0] > 200) & (pixels[..., 1:] < 50).all(axis=-1)
    assert red_pixels.sum() > 50, "embedded canvas is missing the red markers"


@_skip_no_playwright
@pytest.mark.parametrize(
    ("fmt", "magic"), [("jpeg", b"\xff\xd8"), ("pdf", b"%PDF-")]
)  # JPEG SOI marker, PDF header
def test_headless_bar_plot_binary_formats(
    bar_plot_widget: BarPlotWidget, fmt: str, magic: bytes
) -> None:
    """BarPlotWidget.to_img produces a valid, non-trivial JPEG/PDF."""
    img_bytes = bar_plot_widget.to_img(fmt=fmt, dpi=72)  # ty: ignore[invalid-argument-type]
    assert img_bytes.startswith(magic)
    assert len(img_bytes) > 1000


@_skip_no_playwright
@pytest.mark.parametrize(
    ("filename", "fmt", "magic"),
    [("bar.jpg", None, b"\xff\xd8"), ("bar.png", "png", b"\x89PNG")],
    ids=["jpeg-from-jpg-extension", "explicit-png"],
)
def test_headless_bar_plot_writes_file(
    bar_plot_widget: BarPlotWidget,
    tmp_path: Any,
    filename: str,
    fmt: str | None,
    magic: bytes,
) -> None:
    """to_img writes the headless capture to disk, inferring JPEG from .jpg."""
    out_path = f"{tmp_path}/{filename}"
    result = bar_plot_widget.to_img(filename=out_path, fmt=fmt, dpi=72)  # ty: ignore[invalid-argument-type]
    assert result.startswith(magic)
    with open(out_path, "rb") as fh:
        assert fh.read() == result


@_skip_no_playwright
def test_headless_structure_png() -> None:
    """StructureWidget (canvas/WebGL) produces a valid PNG."""
    struct = Structure(Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    widget = StructureWidget(structure=struct, style="height: 400px;")
    png_bytes = widget.to_img(fmt="png", dpi=72)
    assert png_bytes[:4] == b"\x89PNG"
    assert len(png_bytes) > 5000


@_skip_no_playwright
@pytest.mark.parametrize("with_plot", [False, True])
def test_headless_webgl_structure_exports(with_plot: bool) -> None:
    """WebGL structures rasterize PDFs and raise for SVG, also beside an SVG plot."""
    if with_plot:
        frames = [
            {
                "structure": Structure(
                    Lattice.cubic(3 + 0.1 * idx), ["Si"], [[0, 0, 0]]
                ).as_dict(),
                "step": idx,
                "metadata": {"energy": -float(idx)},
            }
            for idx in range(4)
        ]
        widget = TrajectoryWidget(
            trajectory={"frames": frames},
            display_mode="structure+plot",
            style="height: 400px;",
        )
    else:
        struct = Structure(Lattice.cubic(3.0), ["Si"], [[0, 0, 0]])
        widget = StructureWidget(structure=struct, style="height: 300px;")
    assert b"/Font" not in widget.to_img(fmt="pdf", dpi=72), "expected raster PDF"
    with pytest.raises(RuntimeError, match="SVG export not supported"):
        widget.to_img(fmt="svg")


@_skip_no_playwright
@_skip_no_pillow
def test_headless_dpi_affects_png_dimensions() -> None:
    """Higher DPI produces larger pixel dimensions for the same widget."""
    import io

    from PIL import Image

    from pymatviz.widgets.scatter_plot import ScatterPlotWidget

    widget = ScatterPlotWidget(
        series=[{"x": [0, 1], "y": [0, 1], "label": "pt"}],
        style="width: 200px; height: 200px;",
    )
    img_lo = Image.open(io.BytesIO(widget.to_img(fmt="png", dpi=72)))
    img_hi = Image.open(io.BytesIO(widget.to_img(fmt="png", dpi=216)))
    assert img_hi.width > img_lo.width * 2
    assert img_hi.height > img_lo.height * 2


_CHEM_POT_ENTRIES = [
    {"name": "Li", "energy": -1.9, "composition": {"Li": 1}},
    {"name": "Fe", "energy": -8.3, "composition": {"Fe": 1}},
    {"name": "O2", "energy": -4.9, "composition": {"O": 1}},
    {"name": "Li2O", "energy": -14.3, "composition": {"Li": 2, "O": 1}},
    {"name": "Fe2O3", "energy": -25.0, "composition": {"Fe": 2, "O": 3}},
]


@_skip_no_playwright
@_skip_no_pillow
@pytest.mark.parametrize(
    ("widget_cls", "kwargs", "max_blank"),
    [
        # HTML-grid widgets (no canvas/svg) -- captured via the HTML fallback
        (PeriodicTableWidget, {"heatmap_values": {"Fe": 42, "O": 100, "Li": 15}}, 0.95),
        (
            HeatmapMatrixWidget,
            {
                "x_items": ["A", "B"],
                "y_items": ["A", "B"],
                "values": [[1.0, 0.5], [0.5, 1.0]],
            },
            0.95,
        ),
        # ChemPotDiagram computes in a Worker; its sync fallback must finish so we
        # capture the plot, not a stuck "Computing..." spinner (~0.99 blank).
        (ChemPotDiagramWidget, {"entries": _CHEM_POT_ENTRIES}, 0.9),
    ],
    ids=["periodic_table", "heatmap_matrix", "chem_pot_diagram"],
)
def test_headless_widget_renders_nonblank_png(
    widget_cls: type, kwargs: dict[str, Any], max_blank: float
) -> None:
    """Widgets without a simple SVG chart still export a non-blank PNG."""
    from pymatviz.widgets._headless import png_blank_fraction

    widget = widget_cls(style="height: 400px;", **kwargs)
    png_bytes = widget.to_img(fmt="png", dpi=72)
    assert png_bytes[:4] == b"\x89PNG"
    blank_frac = png_blank_fraction(png_bytes)
    assert blank_frac is not None
    assert blank_frac < max_blank


@_skip_no_playwright
def test_headless_convex_hull_ternary_canvas_export() -> None:
    """Ternary ConvexHull renders WebGL: PNG works, SVG raises, PDF is valid."""
    widget = ConvexHullWidget(
        entries=[
            {"composition": {"Li": 1}, "energy": -1.9},
            {"composition": {"Fe": 1}, "energy": -4.2},
            {"composition": {"O": 1}, "energy": -3.0},
            {"composition": {"Li": 2, "O": 1}, "energy": -15.8},
        ],
        style="height: 500px;",
    )
    assert widget.to_img(fmt="png", dpi=72)[:4] == b"\x89PNG"
    assert widget.to_img(fmt="pdf", dpi=72)[:5] == b"%PDF-"
    with pytest.raises(RuntimeError, match="WebGL"):
        widget.to_img(fmt="svg")


@_skip_no_playwright
def test_headless_convex_hull_binary_svg() -> None:
    """Binary ConvexHull renders SVG: vector export returns the chart, not an icon."""
    widget = ConvexHullWidget(
        entries=[
            {"composition": {"Li": 1}, "energy": -1.9},
            {"composition": {"O": 1}, "energy": -3.0},
            {"composition": {"Li": 2, "O": 1}, "energy": -15.8},
        ],
        style="height: 400px;",
    )
    svg_text = widget.to_img(fmt="svg").decode("utf-8")
    assert svg_text.startswith("<?xml")
    # A real chart SVG is far larger than the ~400-byte toolbar icons
    assert len(svg_text) > 2000, f"SVG too small ({len(svg_text)} chars)"


@_skip_no_playwright
def test_headless_repeated_exports_in_one_process() -> None:
    """Repeated to_img() calls succeed (1st sync render flips later ones to async)."""
    from pymatviz.widgets.scatter_plot import ScatterPlotWidget

    bar = BarPlotWidget(series=[{"x": [0, 1], "y": [1.0, 2.0]}], style="height: 300px;")
    scatter = ScatterPlotWidget(
        series=[{"x": [0, 1], "y": [1.0, 2.0]}], style="height: 300px;"
    )
    for widget in (bar, scatter, bar):
        assert widget.to_img(fmt="png", dpi=72)[:4] == b"\x89PNG"


@_skip_no_playwright
@pytest.mark.parametrize(
    ("widget", "expect_warning", "expect_type"),
    [
        (
            ScatterPlotWidget(
                series=[{"x": [0, 1, 2], "y": [0, 1, 4], "label": "s"}],
                style="height: 300px;",
            ),
            False,
            "svg",
        ),
        (ScatterPlotWidget(), True, "svg"),  # empty: renders axes but warns on data
        (
            StructureWidget(
                structure=Structure(Lattice.cubic(3.0), ["Si"], [[0, 0, 0]]),
                style="height: 300px;",
            ),
            False,
            "canvas",
        ),
    ],
    ids=["scatter", "empty_scatter", "structure"],
)
def test_render_report(widget: Any, expect_warning: bool, expect_type: str) -> None:
    """render_report classifies content, flags empty data, and never raises."""
    report = widget.render_report(timeout=25, dpi=72)
    assert report.ok is True
    assert report.is_blank is False
    assert report.content_type == expect_type
    assert bool(report.warnings) is expect_warning
    assert report.summary["widget_type"] == widget.widget_type


def test_render_report_never_raises_on_asset_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """render_report honors its never-raises contract if asset resolution fails."""
    widget = ScatterPlotWidget(series=[{"x": [0, 1], "y": [1, 2]}])

    def _boom() -> tuple[str, str]:
        raise FileNotFoundError("bundle unavailable offline")

    monkeypatch.setattr(widget, "_resolve_assets", _boom)
    report = widget.render_report(timeout=5, dpi=72)  # must not raise
    assert report.ok is False
    assert report.error is not None
    assert "bundle unavailable offline" in report.error
    # data-derived summary/warnings survive a render failure
    assert report.summary["widget_type"] == "scatter_plot"


@_skip_no_playwright
@_skip_no_pillow
def test_to_html_inline_renders_offline(tmp_path: Any) -> None:
    """to_html(inline=True) produces a self-contained page that actually renders.

    build_interactive_html has its own render path (controls on, no capture
    machinery) that to_img/render_report don't exercise, so load the file and
    confirm it paints a non-blank chart.
    """
    from pymatviz.widgets._headless import _get_browser, png_blank_fraction

    widget = ScatterPlotWidget(
        series=[{"x": [0, 1, 2], "y": [0, 1, 4], "label": "s"}],
        style="width: 500px; height: 350px;",
    )
    out_path = tmp_path / "widget.html"
    widget.to_html(str(out_path), inline=True)

    page = _get_browser().new_page(viewport={"width": 600, "height": 420})
    try:
        page.goto(out_path.as_uri(), wait_until="domcontentloaded")
        page.wait_for_selector("#widget-root svg", timeout=20000)
        page.wait_for_timeout(500)
        png_bytes = page.locator("#widget-root").screenshot(type="png")
    finally:
        page.close()

    assert png_bytes[:4] == b"\x89PNG"
    blank_frac = png_blank_fraction(png_bytes)
    assert blank_frac is not None
    assert blank_frac < 0.99  # real chart, not a blank frame
