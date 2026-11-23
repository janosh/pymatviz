"""Cross-widget tests: construction, widget_type, traitlet defaults, shared trait
validation, unknown kwargs and JSON round trips.
"""

from __future__ import annotations

import inspect
import json
from typing import Any
from unittest.mock import Mock

import kemio
import numpy as np
import pandas as pd
import pytest
import traitlets as tl

import pymatviz as pmv
from pymatviz.structures import structure_dict
from pymatviz.widgets import _traits, matterviz
from pymatviz.widgets._normalize import normalize_plot_json, structure_to_dict
from pymatviz.widgets._schema_traits import StructureTraits
from pymatviz.widgets._traits import ELEMENT_COLOR_SCHEMES
from pymatviz.widgets.band_structure import BandStructureWidget
from pymatviz.widgets.bands_and_dos import BandsAndDosWidget
from pymatviz.widgets.bar_plot import BarPlotWidget
from pymatviz.widgets.brillouin_zone import BrillouinZoneWidget
from pymatviz.widgets.chem_pot_diagram import ChemPotDiagramWidget
from pymatviz.widgets.composition import CompositionWidget
from pymatviz.widgets.convex_hull import ConvexHullWidget
from pymatviz.widgets.coordination_bar_plot import CoordinationBarPlotWidget
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
from tests.widgets.conftest import SI_STRUCT
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
    assert metadata == {}
    view = mime_data["application/vnd.jupyter.widget-view+json"]
    assert view == {"version_major": 2, "version_minor": 1, "model_id": widget.model_id}
    assert widget_cls.__name__ in str(mime_data["text/plain"])
    model_view = ("AnyModel", "AnyView", "anywidget", "anywidget")
    names = ("_model_name", "_view_name", "_model_module", "_view_module")
    assert tuple(getattr(widget, name) for name in names) == model_view
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
        (SpacegroupBarPlotWidget, "data", None),
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
        # dict items fill a missing key or label from the other (as a string)
        (
            HeatmapMatrixWidget,
            {"x_items": [{"label": "a"}, {"key": 5}, {"key": "k", "label": "L"}]},
            "x_items",
            [
                {"key": "a", "label": "a"},
                {"key": 5, "label": "5"},
                {"key": "k", "label": "L"},
            ],
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
        # NaN, ±inf and NA heat values become None (missing tiles), like tile_plots
        (
            PeriodicTableWidget,
            {"heatmap_values": pd.Series({"Fe": 1.5, "O": np.nan, "Na": -np.inf})},
            "heatmap_values",
            {"Fe": 1.5, "O": None, "Na": None},
        ),
        (
            PeriodicTableWidget,
            {"heatmap_values": np.array([1.0, np.nan, 3.0])},
            "heatmap_values",
            [1.0, None, 3.0],
        ),
        (  # split tiles keep their other segments
            PeriodicTableWidget,
            {"heatmap_values": {"Fe": [1.0, np.float32("nan")], "O": pd.NA}},
            "heatmap_values",
            {"Fe": [1.0, None], "O": None},
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


_TREE = [{"label": "a", "value": 1}, {"label": "b", "value": 2}]


def _seq(items: list[Any]) -> Any:
    """Factory of items as a list, or as a generator if lazy."""
    return lambda lazy: (item for item in items) if lazy else list(items)


_ITERATOR_CASES = [
    (TrajectoryWidget, "trajectory", _seq([SI_STRUCT, SI_STRUCT])),
    (CoordinationBarPlotWidget, "structures", _seq([SI_STRUCT, SI_STRUCT])),
    (RdfPlotWidget, "structures", _seq([SI_STRUCT])),
    (TreemapWidget, "data", _seq(_TREE)),
    (SunburstWidget, "data", _seq(_TREE)),
    (SpacegroupBarPlotWidget, "data", _seq([225, 225, 62])),
    (ScatterPlotWidget, "series", _seq([{"x": [1, 2], "y": [3, 4]}])),
    (HistogramWidget, "series", _seq([{"values": [1, 2]}, {"values": [3]}])),
    (BarPlotWidget, "series", _seq([{"x": [1, 2], "y": [3, 4]}])),
    (ConvexHullWidget, "entries", _seq([{"composition": {"Li": 1}, "energy": -1}])),
    (StructureWidget, "selected_sites", _seq([0, 1])),
    (PeriodicTableWidget, "active_elements", _seq(["Fe", "O"])),
    (PeriodicTableWidget, "heatmap_values", _seq([1.0, np.nan, 3.0])),
    # nested iterators: x/y of an XRD dict, a tree node's children, a tile's series
    (
        XrdWidget,
        "patterns",
        lambda lazy: {"x": _seq([10.0, 20.0])(lazy), "y": _seq([1.0, 2.0])(lazy)},
    ),
    (TreemapWidget, "data", lambda lazy: {"label": "r", "children": _seq(_TREE)(lazy)}),
    (
        PeriodicTableWidget,
        "tile_plots",
        lambda lazy: {"Fe": _seq([{"values": _seq([1.0, 2.0])(lazy)}])(lazy)},
    ),
]


@pytest.mark.parametrize(
    ("widget_cls", "attr", "make_value"),
    _ITERATOR_CASES,
    ids=[f"{cls.__name__}.{attr}" for cls, attr, _ in _ITERATOR_CASES],
)
def test_widgets_take_iterators(
    widget_cls: type[MatterVizWidget], attr: str, make_value: Any
) -> None:
    """Generators sync like the list of their items, on construction, assignment
    and in batches.
    """
    expected = getattr(widget_cls(**{attr: make_value(lazy=False)}), attr)
    assert getattr(widget_cls(**{attr: make_value(lazy=True)}), attr) == expected
    widget = widget_cls()
    setattr(widget, attr, make_value(lazy=True))
    assert getattr(widget, attr) == expected
    widget = widget_cls()
    with widget.hold_trait_notifications():
        setattr(widget, attr, make_value(lazy=True))
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
        with pytest.raises(expected, match=f"'{trait_name}' trait"):
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
    with pytest.raises(tl.TraitError, match="'color_scale' trait"):
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


def test_positional_traits() -> None:
    """Constructors take their _positional_traits in order, each once."""
    assert BrillouinZoneWidget(None, {"vertices": []}).bz_data == {"vertices": []}
    with pytest.raises(
        TypeError, match=r"Xrd\w+ takes positional \('patterns',\), got 2"
    ):
        XrdWidget(None, None)
    with pytest.raises(TypeError, match=r"got \['doses'\] positionally and as keyword"):
        DosWidget(None, doses=None)
    assert str(inspect.signature(BandsAndDosWidget)) == (
        "(band_structs=None, doses=None, structure=None, *, version_override=None, "
        "**kwargs)"
    )
    with pytest.raises(TypeError, match=r"_positional_traits \('nope',\) must be"):
        type("Bad", (MatterVizWidget,), {"_positional_traits": ("nope",)})


@pytest.mark.parametrize("widget_cls", WIDGET_CASES, ids=lambda cls: cls.__name__)
def test_widget_json_round_trip(
    widget_cls: type[MatterVizWidget], request: pytest.FixtureRequest, tmp_path: Any
) -> None:
    """to_json/from_json restore every widget type with identical traits, binary
    array buffers included, from JSON text or a file (in a new dir, like to_img).
    """
    widget = WIDGET_CASES[widget_cls][0](request)
    json_path = f"{tmp_path}/new_dir/widget.json"
    text = widget.to_json(json_path)
    for source in (text, json_path):
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


def test_constructor_converts_once_to_plain_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Constructor kwargs convert once (HasTraits.__init__ re-sets each validated
    kwarg, which reran every conversion), and structure dicts become plain JSON
    (numpy site properties used to sync as is).
    """
    struct_trait = StructureWidget.class_traits()["structure"]
    struct_calls: list[Any] = []
    monkeypatch.setattr(
        struct_trait,
        "normalize",
        lambda value: struct_calls.append(value) or structure_to_dict(value),
    )
    json_labels: list[str] = []

    def counting_normalize(value: Any, label: str) -> Any:
        """normalize_plot_json recording its label."""
        json_labels.append(label)
        return normalize_plot_json(value, label)

    monkeypatch.setattr(_traits, "normalize_plot_json", counting_normalize)
    struct = structure_dict(SI_STRUCT)
    struct["sites"][0]["properties"] = {"magmom": np.float32(1.5)}
    widget = StructureWidget(struct, slice_settings={"thickness": np.float32(0.5)})
    assert len(struct_calls) == 1
    assert json_labels.count("StructureWidget.slice_settings") == 1
    assert _is_plain_json(widget.get_state(["structure", "slice_settings"]))
    assert widget.structure["sites"][0]["properties"] == {"magmom": 1.5}
    # the skip holds only while constructing: reassigning the stored value after
    # editing it in place converts it again
    widget.structure["sites"][0]["properties"]["magmom"] = np.float32(2.5)
    widget.slice_settings["thickness"] = np.float32(0.25)
    widget.structure, widget.slice_settings = widget.structure, widget.slice_settings
    assert len(struct_calls) == 2
    assert json_labels.count("StructureWidget.slice_settings") == 2
    assert _is_plain_json(widget.get_state(["structure", "slice_settings"]))


@pytest.mark.parametrize(
    ("error", "expected_msg"),
    [
        (tl.TraitError("bad value"), "bad value"),
        (KeyError("key"), "KeyError: 'key'"),
        (FileNotFoundError("missing.xyz"), "FileNotFoundError: missing.xyz"),
    ],
)
def test_validator_raises_trait_errors(error: Exception, expected_msg: str) -> None:
    """validator() re-raises a TraitError as is and any other error as a TraitError
    naming its type, chained to the original.
    """

    class Checked(tl.HasTraits):
        """HasTraits with an always failing validator."""

        value = tl.Int()

        @_traits.validator("value")
        def _check_value(self, _proposal: tl.Bunch) -> Any:
            """Raise the parametrized error."""
            raise error

    with pytest.raises(tl.TraitError) as exc_info:
        Checked().value = 1
    assert str(exc_info.value) == expected_msg
    if isinstance(error, tl.TraitError):
        assert exc_info.value is error
    else:
        assert exc_info.value.__cause__ is error


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
    """Every dict/list config trait, those with validators included, converts numpy
    values assigned after construction to plain JSON before syncing, unless listed
    in ``_raw_json_traits``. A validator returning its raw input used to skip the
    conversion (e.g. ``StructureWidget.isosurface_settings``).
    """
    widget = WIDGET_CASES[widget_cls][0](request)
    # values the validators of these traits accept
    valid_values: dict[str, tuple[Any, Any]] = {
        "tile_props": ({"data-count": np.int64(2)}, {"data-count": 2}),
        "tile_plot_config": (
            {"x_range": np.array([0.0, 1.5]), "bins": np.int64(2)},
            {"x_range": [0.0, 1.5], "bins": 2},
        ),
        "data": (
            {"nodes": [{"label": "a"}], "links": [{"value": np.float32(3)}]},
            {"nodes": [{"label": "a"}], "links": [{"value": 3.0}]},
        ),
        "selected_point": (
            {"series_idx": np.int64(0), "point_idx": np.int64(1)},
            {"series_idx": 0, "point_idx": 1},
        ),
    }
    assigned: dict[str, Any] = {}
    for name in widget.public_trait_names():
        trait = widget.traits()[name]
        sub_traits = getattr(trait, "trait_types", [trait])
        if name in widget._raw_json_traits or trait.read_only:
            continue  # their validators convert them (e.g. series to binary)
        if name in valid_values and any(isinstance(sub, tl.Dict) for sub in sub_traits):
            value, expected = valid_values[name]
        elif any(isinstance(sub, tl.Dict) for sub in sub_traits):
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


@pytest.mark.parametrize("widget_cls", WIDGET_CASES, ids=lambda cls: cls.__name__)
def test_scalar_traits_take_numpy_reject_non_finite(
    widget_cls: type[MatterVizWidget], request: pytest.FixtureRequest
) -> None:
    """Float, Int and Bool traits take numpy scalars as Python ones; Float traits
    reject NaN and infinities (synced, they'd be invalid JSON).
    """
    widget = WIDGET_CASES[widget_cls][0](request)
    traits = {
        name: widget.traits()[name]
        for name in widget.public_trait_names()
        if name not in widget._raw_json_traits
    }
    floats = [name for name, trait in traits.items() if isinstance(trait, tl.Float)]
    ints = [name for name, trait in traits.items() if isinstance(trait, tl.Int)]
    bools = [name for name, trait in traits.items() if isinstance(trait, tl.Bool)]
    for name in floats:
        setattr(widget, name, np.float32(0.5))
        assert type(getattr(widget, name)) is float
        assert getattr(widget, name) == 0.5
        for bad in (np.nan, -np.inf, np.float32("nan")):
            with pytest.raises(
                tl.TraitError, match=f"'{name}' trait .* must be finite"
            ):
                setattr(widget, name, bad)
            assert getattr(widget, name) == 0.5
    for name in ints:  # 1 is in range of index traits (2 frames, 2 series)
        setattr(widget, name, np.int64(1))
        assert type(getattr(widget, name)) is int
        assert getattr(widget, name) == 1
    for name in bools:  # False passes validators like DosWidget's stack check
        setattr(widget, name, np.bool_(0))
        assert getattr(widget, name) is False


@pytest.mark.parametrize(
    ("make_widget", "trait", "bad_value", "match"),
    [
        (TrajectoryWidget, "trajectory", "bad", "Unsupported trajectory type"),
        (TreemapWidget, "data", {"label": "a", "value": np.nan}, "non-finite"),
        (HeatmapMatrixWidget, "values", [[np.inf]], "non-finite"),
        (lambda: CompositionWidget("Fe2O3"), "composition", "Xx9", "unknown element"),
        (ScatterPlotWidget, "y_axis", {"label": np.nan}, "non-finite"),
        (lambda: XrdWidget({"x": [1], "y": [1]}), "patterns", {"x": [1], "y": []}, "x"),
        (StructureWidget, "structure", 5, "structure must be a dict"),
        (BandStructureWidget, "fermi_level", np.nan, "must be finite"),
        (lambda: SpacegroupBarPlotWidget([225]), "data", [0], r"in \[1, 230\]"),
        # counts and cutoffs need positive values (0 bins or radii used to sync)
        (HistogramWidget, "bins", 0, "'bins' trait .* not be less than 1"),
        (RdfPlotWidget, "n_bins", -5, "'n_bins' trait .* not be less than 1"),
        (RdfPlotWidget, "cutoff", 0, "cutoff must be > 0, got 0"),
        (RdfPlotWidget, "cutoff", np.float32(-1), "cutoff must be > 0, got -1"),
        # a second data source made the state (and its saved JSON) unloadable
        (
            lambda: RdfPlotWidget(patterns=[{"r": [1], "g_r": [1]}]),
            "structures",
            {"lattice": {}, "sites": []},
            "not both",
        ),
        (
            lambda: FermiSurfaceWidget(fermi_data={"isosurfaces": []}),
            "band_data",
            {"energies": []},
            "exactly one of fermi_data or band_data",
        ),
        (
            lambda: FermiSurfaceWidget(fermi_data={"isosurfaces": []}),
            "fermi_data",
            None,
            "exactly one of fermi_data or band_data",
        ),
        # errors other than TypeError/ValueError (KeyError, FileNotFoundError) in
        # converters skipped the rollback and left the raw value synced
        (TrajectoryWidget, "trajectory", {"foo": 1}, "need a 'frames' list"),
        (
            TrajectoryWidget,
            "trajectory",
            [kemio.Structure(np.eye(3) * 3, ["Fe"], [[0, 0, 0]]), "missing.xyz"],
            "FileNotFoundError: .*missing.xyz",
        ),
        (DosWidget, "doses", "missing/vasprun.xml", "FileNotFoundError: .*missing"),
        # band and DOS labels were checked in the constructor only
        (
            lambda: BandsAndDosWidget(
                doses={"a": {"frequencies": [0], "densities": [1]}}
            ),
            "band_structs",
            {"b": {"qpoints": [], "frequencies": []}},
            r"band_keys=\{'b'\} and dos_keys=\{'a'\} must be identical",
        ),
        (
            lambda: BandsAndDosWidget(
                band_structs={"a": {"qpoints": [], "frequencies": []}}
            ),
            "doses",
            {"b": {"frequencies": [0], "densities": [1]}},
            r"band_keys=\{'a'\} and dos_keys=\{'b'\} must be identical",
        ),
        # labels equal as strings merged two structures into one
        *(
            (widget_cls, "structures", {1: SI_STRUCT, "1": SI_STRUCT}, "unique as str")
            for widget_cls in (CoordinationBarPlotWidget, RdfPlotWidget)
        ),
        # None skipped the validator outside batches, then broke isosurface checks
        (StructureWidget, "volumetric_data", None, "list of volumes or a {prefix"),
        (TreemapWidget, "zoom_root_id", True, "string or number, got True"),
        # index traits must address an item of the data
        (
            lambda: TrajectoryWidget([SI_STRUCT] * 2),
            "current_step_idx",
            2,
            r"current_step_idx=2 is out of range for 2 frames \(valid: 0 to 1\)",
        ),
        (
            lambda: HistogramWidget([{"values": [1]}]),
            "selected_series_idx",
            1,
            r"selected_series_idx=1 is out of range for 1 series \(valid: 0 to 0\)",
        ),
        (
            lambda: StructureWidget(SI_STRUCT),
            "selected_sites",
            [0, 1],
            r"selected_sites\[1\]=1 is out of range for 1 sites",
        ),
        (
            lambda: ScatterPlotWidget([{"x": [1, 2], "y": [3, 4]}]),
            "selected_point",
            {"series_idx": 0, "point_idx": 2},
            r"selected_point.point_idx=2 is out of range for 2 points",
        ),
        (
            lambda: ScatterPlotWidget([{"x": [1, 2], "y": [3, 4]}]),
            "selected_point",
            {"series_idx": 1, "point_idx": 0},
            r"selected_point.series_idx=1 is out of range for 1 series",
        ),
        (
            ScatterPlotWidget,
            "selected_point",
            {"point_idx": 0},
            "selected_point needs int series_idx and point_idx",
        ),
    ],
)
def test_invalid_batched_assignment_rolls_back(
    make_widget: Any, trait: str, bad_value: Any, match: str
) -> None:
    """An invalid value in a hold_trait_notifications() batch raises TraitError and
    rolls the whole batch back without notifying. Other errors used to leave earlier
    (or, for cross-validated traits, the invalid raw) values set and synced.
    """
    widget = make_widget()
    old_state = widget.to_dict()
    observer = Mock()
    widget.observe(observer, names=["style", trait])

    def assign_batch() -> None:
        """Set a valid style, then the bad value, in one batch."""
        with widget.hold_trait_notifications():
            widget.style = "height: 123px;"
            setattr(widget, trait, bad_value)

    with pytest.raises(tl.TraitError, match=match):
        assign_batch()
    assert widget.to_dict() == old_state
    observer.assert_not_called()
    with pytest.raises(tl.TraitError, match=match):  # same outside a batch
        setattr(widget, trait, bad_value)
    assert widget.to_dict() == old_state


@pytest.mark.parametrize(
    ("make_widget", "old_source", "new_source", "new_value"),
    [
        (
            lambda: RdfPlotWidget(patterns=[{"r": [1], "g_r": [1]}]),
            "patterns",
            "structures",
            {"lattice": {}, "sites": []},
        ),
        (
            lambda: FermiSurfaceWidget(fermi_data={"isosurfaces": []}),
            "fermi_data",
            "band_data",
            {"energies": []},
        ),
    ],
)
def test_data_source_swap_in_batch(
    make_widget: Any, old_source: str, new_source: str, new_value: Any
) -> None:
    """Swapping a widget's one data source inside hold_trait_notifications() passes
    the cross-trait check, and the result round-trips through to_json/from_json.
    """
    widget = make_widget()
    with widget.hold_trait_notifications():
        setattr(widget, old_source, None)
        setattr(widget, new_source, new_value)
    assert getattr(widget, old_source) is None
    assert getattr(widget, new_source) == new_value
    assert from_json(widget.to_json()).to_dict() == widget.to_dict()
