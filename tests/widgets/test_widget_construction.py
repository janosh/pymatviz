"""Cross-widget tests: construction, widget_type, traitlet defaults, shared trait
validation, unknown kwargs and JSON round trips.
"""

from __future__ import annotations

import json
from typing import Any

import kemio
import numpy as np
import pandas as pd
import pytest
import traitlets as tl

import pymatviz as pmv
from pymatviz.widgets import matterviz
from pymatviz.widgets._schema_traits import StructureTraits
from pymatviz.widgets._traits import ELEMENT_COLOR_SCHEMES
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
from pymatviz.widgets.matterviz import (
    MATTERVIZ_ANYWIDGET_VERSION,
    MatterVizWidget,
    from_json,
)
from pymatviz.widgets.periodic_table import PeriodicTableWidget
from pymatviz.widgets.phase_diagram import PhaseDiagramWidget
from pymatviz.widgets.rdf_plot import RdfPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from pymatviz.widgets.scatter_plot_3d import ScatterPlot3DWidget
from pymatviz.widgets.spacegroup_bar import SpacegroupBarPlotWidget
from pymatviz.widgets.structure import StructureWidget
from pymatviz.widgets.sunburst import SunburstWidget
from pymatviz.widgets.trajectory import TrajectoryWidget
from pymatviz.widgets.treemap import TreemapWidget
from pymatviz.widgets.xrd import XrdWidget
from tests.widgets.test_browser_render import WIDGET_CASES


@pytest.mark.parametrize(
    ("widget_cls", "expected_type", "kwargs"),
    [
        (
            ConvexHullWidget,
            "convex_hull",
            {"entries": [{"composition": {"Li": 1}, "energy": -1.5}]},
        ),
        (BandStructureWidget, "band_structure", {"band_structs": {"bands": []}}),
        (DosWidget, "dos", {"doses": {"energies": [0]}}),
        (
            BandsAndDosWidget,
            "bands_and_dos",
            {"band_structs": {"bands": []}, "doses": {"energies": []}},
        ),
        (FermiSurfaceWidget, "fermi_surface", {"fermi_data": {"isosurfaces": []}}),
        (FermiSurfaceWidget, "fermi_surface", {"band_data": {"energies": []}}),
        (
            BrillouinZoneWidget,
            "brillouin_zone",
            {"structure": {"lattice": {}, "sites": []}},
        ),
        (PhaseDiagramWidget, "phase_diagram", {"data": {"components": ["A", "B"]}}),
        (XrdWidget, "xrd", {"patterns": {"x": [10], "y": [100], "kind": "profile"}}),
        (
            ScatterPlotWidget,
            "scatter_plot",
            {"series": [{"x": [0, 1], "y": [1, 2], "label": "curve"}]},
        ),
        (
            BarPlotWidget,
            "bar_plot",
            {"series": [{"x": [0, 1], "y": [2, 3], "label": "bars"}]},
        ),
        (
            HistogramWidget,
            "histogram",
            {"series": [{"values": [2, 2.5], "label": "hist"}]},
        ),
        (
            PeriodicTableWidget,
            "periodic_table",
            {"heatmap_values": {"Fe": 42, "O": 100}},
        ),
        (PeriodicTableWidget, "periodic_table", {"heatmap_values": [1.0, 4.0, 6.9]}),
        (
            PeriodicTableWidget,
            "periodic_table",
            {"color_scale": {"type": "log", "value_range": [1, None]}},
        ),
        (PeriodicTableWidget, "periodic_table", {"color_bar": None}),
        (RdfPlotWidget, "rdf_plot", {"structures": {"lattice": {}, "sites": []}}),
        (
            ScatterPlot3DWidget,
            "scatter_plot_3d",
            {"series": [{"x": [1], "y": [2], "z": [3], "label": "pt"}]},
        ),
        (
            HeatmapMatrixWidget,
            "heatmap_matrix",
            {"x_items": ["A", "B"], "y_items": ["C", "D"], "values": [[1, 2], [3, 4]]},
        ),
        (
            HeatmapMatrixWidget,
            "heatmap_matrix",
            {"values": {"Fe": {"Fe": 1.0, "O": 0.5}, "O": {"Fe": 0.5, "O": 1.0}}},
        ),
        (
            HeatmapMatrixWidget,
            "heatmap_matrix",
            {"x_items": ["A"], "y_items": ["B"], "color_scale": {"type": "log"}},
        ),
        (
            HeatmapMatrixWidget,
            "heatmap_matrix",
            {"color_bar": {"title": "count", "orientation": "vertical"}},
        ),
        (
            FermiSurfaceWidget,
            "fermi_surface",
            {"fermi_data": {"isosurfaces": []}, "color_bar": {"title": "v_F"}},
        ),
        (
            FermiSurfaceWidget,
            "fermi_surface",
            {"band_data": {"energies": []}, "color_bar": None},
        ),
        (
            ChemPotDiagramWidget,
            "chem_pot_diagram",
            {"entries": [{"name": "Li2O", "energy": -14.3}]},
        ),
        (SunburstWidget, "sunburst", {"data": [{"label": "leaf", "value": 3}]}),
        (
            TreemapWidget,
            "treemap",
            {"data": {"label": "root", "children": [{"label": "leaf", "value": 3}]}},
        ),
    ],
)
def test_widget_construction_and_type(
    widget_cls: type, expected_type: str, kwargs: dict[str, Any]
) -> None:
    """Constructors set widget_type and keep the last kwarg's state unchanged."""
    widget = widget_cls(**kwargs)
    assert widget.widget_type == expected_type
    state_key = list(kwargs)[-1]
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
    # the frontend loads remote bundles by URL, local builds as text
    esm, css = widget._resolve_assets()
    pinned_esm = matterviz._bundle_sources(MATTERVIZ_ANYWIDGET_VERSION)[0]
    assert widget._esm in (esm, pinned_esm)
    assert len(esm) > 1000
    assert len(css) > 100
    assert "export default" in esm or " as default" in esm
    assert "render" in esm


@pytest.mark.parametrize(
    ("widget_cls", "attr", "expected"),
    [
        (BarPlotWidget, "series", None),
        (ScatterPlotWidget, "x_axis", None),
        (BarPlotWidget, "orientation", "vertical"),
        (BarPlotWidget, "mode", "overlay"),
        (HistogramWidget, "bins", 100),
        (HistogramWidget, "mode", "single"),
        (HistogramWidget, "show_legend", True),
        (PeriodicTableWidget, "color_scale", None),  # component default (viridis)
        (PeriodicTableWidget, "color_bar", {}),  # shown by default
        (PeriodicTableWidget, "missing", None),
        (HeatmapMatrixWidget, "color_bar", {}),  # shown by default
        (HeatmapMatrixWidget, "missing", None),
        (HeatmapMatrixWidget, "values", None),  # None, not an empty list
        (HeatmapMatrixWidget, "x_items", []),
        (ChemPotDiagramWidget, "entries", []),
        (ScatterPlot3DWidget, "camera_projection", "perspective"),
        (HeatmapMatrixWidget, "color_scale", None),
        (HeatmapMatrixWidget, "tile_size", "50px"),
        (HeatmapMatrixWidget, "gap", "0px"),
        (SpacegroupBarPlotWidget, "data", {}),
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
        # scalar string axis items are one label
        (
            HeatmapMatrixWidget,
            {"x_items": "Fe"},
            "x_items",
            [{"key": "Fe", "label": "Fe"}],
        ),
        (
            HeatmapMatrixWidget,
            {"y_items": "O"},
            "y_items",
            [{"key": "O", "label": "O"}],
        ),
        (
            PeriodicTableWidget,
            {"heatmap_values": np.array([1.0, 4.0, 6.9])},
            "heatmap_values",
            [1.0, 4.0, 6.9],
        ),
        (  # an element-indexed Series keeps its labels (tolist() would map by Z)
            PeriodicTableWidget,
            {"heatmap_values": pd.Series({"Fe": 1.5, "O": np.int64(2)})},
            "heatmap_values",
            {"Fe": 1.5, "O": 2},
        ),
        (
            RdfPlotWidget,
            {"patterns": [{"r": np.array([1.0, 2.0]), "g_r": np.ones(2)}]},
            "patterns",
            [{"r": [1.0, 2.0], "g_r": [1.0, 1.0]}],
        ),
        (
            ChemPotDiagramWidget,
            {"entries": [{"composition": "Li2O", "energy": np.float64(-14.3)}]},
            "entries",
            [{"composition": {"Li": 2.0, "O": 1.0}, "energy": -14.3}],
        ),
        *[  # numpy DOS dicts sync as JSON lists (to_img's encoder rejects arrays)
            (
                widget_cls,
                {"doses": {"energies": np.array([0.0, 1.0]), "densities": np.ones(2)}},
                "doses",
                {"energies": [0.0, 1.0], "densities": [1.0, 1.0]},
            )
            for widget_cls in (DosWidget, BandsAndDosWidget)
        ],
        (  # numpy and pandas pattern dicts sync as JSON lists
            XrdWidget,
            {"patterns": {"x": pd.Series([10.0, 20.0]), "y": np.array([1, 2])}},
            "patterns",
            {"x": [10.0, 20.0], "y": [1, 2], "kind": "profile"},
        ),
        (
            SpacegroupBarPlotWidget,
            {"data": np.array([225, 166, 62])},
            "data",
            {"62": 1, "166": 1, "225": 1},
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
def test_widget_normalizes_inputs(
    widget_cls: type,
    kwargs: dict[str, Any],
    attr: str,
    expected: list[Any] | dict[str, Any],
) -> None:
    """Widgets normalize numpy arrays and scalar axis items to plain Python lists, on
    construction and on later assignment.
    """
    widget = widget_cls(**kwargs)
    result = getattr(widget, attr)
    assert result == expected
    assert type(result) is type(expected)
    setattr(widget, attr, kwargs[attr])
    assert getattr(widget, attr) == expected


_COLOR_SCHEME_CASES = [
    *[(name, name) for name in ELEMENT_COLOR_SCHEMES],
    ("dark mode", "Dark Mode"),  # caseless, stored canonical
    ("CPK", tl.TraitError),  # never a matterviz scheme
    ("", tl.TraitError),
]
_BONDING_STRATEGY_CASES = [
    *[(name, name) for name in StructureTraits.bonding_strategy.values],  # noqa: PD011
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
        (StructureWidget, "active_pane", "Flight", "flight"),
        (StructureWidget, "active_pane", None, None),
        (StructureWidget, "active_pane", "hotspots", tl.TraitError),  # trajectory-only
        (TrajectoryWidget, "active_pane", "hotspots", "hotspots"),
        (TrajectoryWidget, "active_pane", "nope", tl.TraitError),
        # the data inspector is a sidebar now (data_inspector_open), not a pane
        (TrajectoryWidget, "active_pane", "data-inspector", tl.TraitError),
        # show_controls takes matterviz's boolean, mode or {mode, hidden, style} forms
        (StructureWidget, "show_controls", "hover", "hover"),
        (TrajectoryWidget, "show_controls", False, False),
        (
            TrajectoryWidget,
            "show_controls",
            {"mode": "always", "hidden": ["export"]},
            {"mode": "always", "hidden": ["export"]},
        ),
        (StructureWidget, "show_controls", "ALWAYS", "always"),
        (StructureWidget, "show_controls", "sometimes", tl.TraitError),
        (StructureWidget, "show_controls", {"mode": "sometimes"}, tl.TraitError),
        (StructureWidget, "show_controls", {"hidden": "export"}, tl.TraitError),
        # color_scale takes a d3 interpolator name, which the renderer resolves
        *[
            (cls, "color_scale", *case)
            for cls in (ScatterPlotWidget, BarPlotWidget, ScatterPlot3DWidget)
            for case in [
                ("interpolatePlasma", "interpolatePlasma"),
                ("viridis", tl.TraitError),
                ({"scheme": "Blues"}, tl.TraitError),
            ]
        ],
        (ScatterPlotWidget, "size_legend", None, None),  # hides the legend
        (ScatterPlotWidget, "density_contours", True, True),
        (TrajectoryWidget, "data_inspector_open", False, False),
        (ChemPotDiagramWidget, "width", 0, tl.TraitError),
    ],
)
def test_matterviz_name_enums(
    widget_cls: type[MatterVizWidget],
    trait_name: str,
    value: str | None,
    expected: str | type[Exception] | None,
) -> None:
    """Scheme/strategy/pane/controls traits accept exactly matterviz's values."""
    widget = widget_cls()
    if isinstance(expected, type):
        with pytest.raises(expected):
            setattr(widget, trait_name, value)
    else:
        setattr(widget, trait_name, value)
        assert widget.to_dict()[trait_name] == expected


@pytest.mark.parametrize(
    "widget_cls",
    [
        PeriodicTableWidget,
        HeatmapMatrixWidget,
        ScatterPlotWidget,
        BarPlotWidget,
        ScatterPlot3DWidget,
    ],
)
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


def _widget_subclasses(cls: type = MatterVizWidget) -> list[type]:
    """All MatterVizWidget subclasses, recursively."""
    return [
        sub
        for child in cls.__subclasses__()
        for sub in (child, *_widget_subclasses(child))
    ]


@pytest.mark.parametrize(
    "make_widget",
    [
        *(
            pytest.param(cls, id=cls.__name__)
            for cls in sorted(set(_widget_subclasses()), key=lambda cls: cls.__name__)
            # validates its exactly-one-of data args before reaching the base init
            if cls.__name__ != "FermiSurfaceWidget"
        ),
        pytest.param(
            lambda **kw: pmv.ptable_hists({"Fe": [1.0]}, **kw), id="ptable_hists"
        ),
        pytest.param(
            lambda **kw: pmv.density_scatter([1.0, 2.0], [1.0, 2.0], **kw),
            id="density_scatter",
        ),
    ],
)
def test_unknown_kwargs_raise(make_widget: Any) -> None:
    """Typos and unsupported options raise instead of being dropped."""
    with pytest.raises(TypeError, match=r"Unknown \w+ options: \['bogus'\]"):
        make_widget(bogus=1)


@pytest.mark.parametrize("widget_cls", WIDGET_CASES, ids=lambda cls: cls.__name__)
def test_widget_json_round_trip(
    widget_cls: type[MatterVizWidget], request: pytest.FixtureRequest, tmp_path: Any
) -> None:
    """to_json/from_json restore every widget type with identical traits, binary
    array buffers included, from JSON text or a file.
    """
    widget = WIDGET_CASES[widget_cls][0](request)
    text = widget.to_json(f"{tmp_path}/widget.json")
    for source in (text, f"{tmp_path}/widget.json"):
        restored = from_json(source)
        # by name: other tests reload widget modules, which swaps the class objects
        assert type(restored).__name__ == widget_cls.__name__
        assert json.loads(restored.to_json()) == json.loads(text)  # key order may vary


def test_from_json_rejects_unknown_widget_type() -> None:
    """JSON without a known widget_type raises instead of guessing a class."""
    with pytest.raises(ValueError, match="Unknown widget_type='nope'"):
        from_json('{"widget_type": "nope"}')


@pytest.mark.parametrize(
    ("widget_cls", "trait", "make_value"),
    [
        (
            StructureWidget,
            "structure",
            lambda: kemio.Structure(np.eye(3) * 3, ["Fe"], [[0, 0, 0]]),
        ),
        (DosWidget, "doses", lambda: kemio.PhononDos([0, 1, 2], [0, 1, 0])),
        (
            BandStructureWidget,
            "band_structs",
            lambda: kemio.PhononBands(
                [[0, 0, 0], [0.5, 0, 0]], [[0, 0, 0], [1, 1, 2]], np.eye(3)
            ),
        ),
        (XrdWidget, "patterns", lambda: {"a": kemio.XrdPattern([10], [100])}),
        (
            ConvexHullWidget,
            "entries",
            lambda: kemio.PhaseDiagram(["Li", "O", "Li2O"], [-1.9, -4.2, -14.3 / 3]),
        ),
    ],
)
def test_widgets_normalize_assigned_objects(
    widget_cls: type[MatterVizWidget], trait: str, make_value: Any
) -> None:
    """Objects assigned after construction convert like constructor input."""
    widget = widget_cls()
    setattr(widget, trait, make_value())
    as_json = lambda val: json.dumps(val, default=np.ndarray.tolist, sort_keys=True)
    expected = getattr(widget_cls(**{trait: make_value()}), trait)
    assert as_json(getattr(widget, trait)) == as_json(expected)


def _is_plain_json(value: Any) -> bool:
    """Whether value holds only exact JSON types (no numpy scalars or arrays)."""
    if isinstance(value, dict):
        keys_ok = all(type(key) is str for key in value)
        return keys_ok and all(map(_is_plain_json, value.values()))
    if type(value) is list:
        return all(map(_is_plain_json, value))
    return type(value) in (type(None), bool, int, float, str)


@pytest.mark.parametrize("widget_cls", WIDGET_CASES, ids=lambda cls: cls.__name__)
def test_config_traits_normalize_on_assignment(
    widget_cls: type[MatterVizWidget], request: pytest.FixtureRequest
) -> None:
    """Every dict/list config trait without its own validator converts numpy values
    assigned after construction to plain JSON before syncing.
    """
    widget = WIDGET_CASES[widget_cls][0](request)
    assigned: dict[str, Any] = {}
    for name in widget.public_trait_names():
        trait = widget.traits()[name]
        sub_traits = getattr(trait, "trait_types", [trait])
        if name in widget._trait_validators or trait.read_only:
            continue  # own validators (e.g. series, data) are tested per widget
        if any(isinstance(sub, tl.Dict) for sub in sub_traits):
            value = {"range": np.array([0.0, 1.5]), "count": np.int64(2)}
            expected: Any = {"range": [0.0, 1.5], "count": 2}
        elif any(isinstance(sub, tl.List) and sub._trait is None for sub in sub_traits):
            value, expected = np.array([0.5, 1.0]), [0.5, 1.0]
        else:
            continue
        setattr(widget, name, value)
        assigned[name] = expected
    assert assigned  # every widget has config dicts
    state = widget.get_state(list(assigned))
    assert state == assigned
    assert _is_plain_json(state), {
        key: val for key, val in state.items() if not _is_plain_json(val)
    }
