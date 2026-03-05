"""Tests for widget construction, traitlet defaults, to_dict, and validation."""

from __future__ import annotations

import importlib.util
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pymatgen.core import Composition

from pymatviz.widgets.band_structure import BandStructureWidget
from pymatviz.widgets.bands_and_dos import BandsAndDosWidget
from pymatviz.widgets.bar_plot import BarPlotWidget
from pymatviz.widgets.brillouin_zone import BrillouinZoneWidget
from pymatviz.widgets.chem_pot_diagram import ChemPotDiagramWidget
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
from pymatviz.widgets.xrd import XrdWidget


# === Construction and widget_type ===


@pytest.mark.parametrize(
    ("widget_cls", "kwargs", "expected_type", "state_key", "expected_state"),
    [
        (
            ConvexHullWidget,
            {"entries": [{"composition": {"Li": 1}, "energy": -1.5}]},
            "convex_hull",
            "entries",
            [{"composition": {"Li": 1}, "energy": -1.5}],
        ),
        (
            BandStructureWidget,
            {"band_structure": {"bands": []}},
            "band_structure",
            "band_structure",
            {"bands": []},
        ),
        (DosWidget, {"dos": {"energies": [0]}}, "dos", "dos", {"energies": [0]}),
        (
            BandsAndDosWidget,
            {"band_structure": {"bands": []}, "dos": {"energies": []}},
            "bands_and_dos",
            "dos",
            {"energies": []},
        ),
        (
            FermiSurfaceWidget,
            {"fermi_data": {"isosurfaces": []}},
            "fermi_surface",
            "fermi_data",
            {"isosurfaces": []},
        ),
        (
            FermiSurfaceWidget,
            {"band_data": {"energies": []}},
            "fermi_surface",
            "band_data",
            {"energies": []},
        ),
        (
            BrillouinZoneWidget,
            {"structure": {"lattice": {}, "sites": []}},
            "brillouin_zone",
            "structure",
            {"lattice": {}, "sites": []},
        ),
        (
            PhaseDiagramWidget,
            {"data": {"components": ["A", "B"]}},
            "phase_diagram",
            "data",
            {"components": ["A", "B"]},
        ),
        (
            XrdWidget,
            {"patterns": {"x": [10], "y": [100]}},
            "xrd",
            "patterns",
            {"x": [10], "y": [100]},
        ),
        (
            ScatterPlotWidget,
            {"series": [{"x": [0, 1], "y": [1, 2], "label": "curve"}]},
            "scatter_plot",
            "series",
            [{"x": [0.0, 1.0], "y": [1.0, 2.0], "label": "curve"}],
        ),
        (
            BarPlotWidget,
            {"series": [{"x": [0, 1], "y": [2, 3], "label": "bars"}]},
            "bar_plot",
            "series",
            [{"x": [0.0, 1.0], "y": [2.0, 3.0], "label": "bars"}],
        ),
        (
            HistogramWidget,
            {"series": [{"x": [0, 1], "y": [2, 2.5], "label": "hist"}]},
            "histogram",
            "series",
            [{"x": [0.0, 1.0], "y": [2.0, 2.5], "label": "hist"}],
        ),
        (
            PeriodicTableWidget,
            {"heatmap_values": {"Fe": 42, "O": 100}},
            "periodic_table",
            "heatmap_values",
            {"Fe": 42, "O": 100},
        ),
        (
            RdfPlotWidget,
            {"structures": {"lattice": {}, "sites": []}},
            "rdf_plot",
            "structures",
            {"lattice": {}, "sites": []},
        ),
        (
            ScatterPlot3DWidget,
            {"series": [{"x": [1], "y": [2], "z": [3], "label": "pt"}]},
            "scatter_plot_3d",
            "series",
            [{"x": [1], "y": [2], "z": [3], "label": "pt"}],
        ),
        (
            HeatmapMatrixWidget,
            {"x_items": ["A", "B"], "y_items": ["C", "D"], "values": [[1, 2], [3, 4]]},
            "heatmap_matrix",
            "values",
            [[1, 2], [3, 4]],
        ),
        (
            SpacegroupBarPlotWidget,
            {"data": [225, 166, 62]},
            "spacegroup_bar",
            "data",
            [225, 166, 62],
        ),
        (
            ChemPotDiagramWidget,
            {"entries": [{"name": "Li2O", "energy": -14.3}]},
            "chem_pot_diagram",
            "entries",
            [{"name": "Li2O", "energy": -14.3}],
        ),
    ],
)
def test_widget_construction_and_type(
    widget_cls: type,
    kwargs: dict[str, Any],
    expected_type: str,
    state_key: str,
    expected_state: Any,
) -> None:
    """Widget constructors preserve expected normalized state."""
    widget = widget_cls(**kwargs)
    assert widget.widget_type == expected_type
    assert getattr(widget, state_key) == expected_state


# === to_dict auto-discovery ===


@pytest.mark.parametrize(
    ("widget_cls", "kwargs", "expected_fields"),
    [
        (
            ScatterPlotWidget,
            {"series": [{"x": [0, 1], "y": [1, 2], "label": "s"}]},
            {
                "series",
                "x_axis",
                "x2_axis",
                "y_axis",
                "y2_axis",
                "display",
                "legend",
                "styles",
                "color_scale",
                "size_scale",
                "ref_lines",
                "fill_regions",
                "error_bands",
                "controls",
                "padding",
                "range_padding",
                "show_legend",
                "x_range",
                "x2_range",
                "y_range",
                "y2_range",
                "color_bar",
                "hover_config",
                "label_placement_config",
                "point_tween",
                "line_tween",
                "point_events",
            },
        ),
        (
            HistogramWidget,
            {"series": [{"x": [0, 1], "y": [1, 2]}], "bins": 50, "mode": "overlay"},
            {
                "series",
                "bins",
                "mode",
                "selected_property",
                "show_legend",
                "x_axis",
                "x2_axis",
                "y_axis",
                "y2_axis",
                "display",
                "legend",
                "bar",
                "ref_lines",
                "controls",
                "padding",
                "range_padding",
                "x_range",
                "x2_range",
                "y_range",
                "y2_range",
            },
        ),
        (
            BarPlotWidget,
            {"series": [{"x": [0], "y": [1]}], "mode": "grouped"},
            {
                "series",
                "orientation",
                "mode",
                "x_axis",
                "x2_axis",
                "y_axis",
                "y2_axis",
                "display",
                "legend",
                "bar",
                "line",
                "ref_lines",
                "controls",
                "padding",
                "range_padding",
                "show_legend",
                "x_range",
                "x2_range",
                "y_range",
                "y2_range",
                "color_scale",
                "size_scale",
                "point_tween",
            },
        ),
        (
            DosWidget,
            {"dos": {"energies": [0, 1]}, "sigma": 0.05},
            {
                "dos",
                "stack",
                "sigma",
                "normalize",
                "orientation",
                "show_legend",
                "spin_mode",
            },
        ),
        (
            ConvexHullWidget,
            {"entries": [{"composition": {"Li": 1}, "energy": -1.5}]},
            {
                "entries",
                "show_stable",
                "show_unstable",
                "show_hull_faces",
                "hull_face_opacity",
                "show_stable_labels",
                "show_unstable_labels",
                "max_hull_dist_show_labels",
                "max_hull_dist_show_phases",
                "temperature",
            },
        ),
        (
            PeriodicTableWidget,
            {"heatmap_values": {"Fe": 42}},
            {
                "heatmap_values",
                "color_scale",
                "color_scale_range",
                "color_overrides",
                "labels",
                "log_scale",
                "show_color_bar",
                "gap",
                "missing_color",
            },
        ),
        (
            ScatterPlot3DWidget,
            {"series": [{"x": [1], "y": [2], "z": [3]}]},
            {
                "series",
                "surfaces",
                "ref_lines",
                "ref_planes",
                "x_axis",
                "y_axis",
                "z_axis",
                "display",
                "styles",
                "color_scale",
                "size_scale",
                "legend",
                "controls",
                "camera_projection",
            },
        ),
        (
            HeatmapMatrixWidget,
            {"x_items": ["A"], "y_items": ["B"]},
            {
                "x_items",
                "y_items",
                "values",
                "color_scale",
                "color_scale_range",
                "log_scale",
                "missing_color",
                "x_axis",
                "y_axis",
                "tile_size",
                "gap",
                "show_values",
                "label_style",
            },
        ),
        (
            SpacegroupBarPlotWidget,
            {"data": [225]},
            {
                "data",
                "show_counts",
                "orientation",
                "x_axis",
                "y_axis",
            },
        ),
        (
            ChemPotDiagramWidget,
            {"entries": [{"name": "Li", "energy": -1.9}]},
            {
                "entries",
                "config",
                "temperature",
            },
        ),
    ],
)
def test_to_dict_includes_subclass_fields(
    widget_cls: type,
    kwargs: dict[str, Any],
    expected_fields: set[str],
) -> None:
    """to_dict returns base fields plus widget-specific synced traitlets."""
    widget = widget_cls(**kwargs)
    state = widget.to_dict()

    base_fields = {"widget_type", "style", "show_controls"}
    assert base_fields <= set(state), f"Missing base fields in {widget_cls.__name__}"

    all_expected = base_fields | expected_fields
    assert set(state) == all_expected, (
        f"{widget_cls.__name__}: to_dict keys mismatch.\n"
        f"  Extra: {set(state) - all_expected}\n"
        f"  Missing: {all_expected - set(state)}"
    )


def test_to_dict_reflects_constructor_values() -> None:
    """to_dict values match what was passed to the constructor."""
    widget = HistogramWidget(
        series=[{"x": [0, 1], "y": [2, 3], "label": "h"}],
        bins=25,
        mode="overlay",
        x_axis={"label": "Value"},
    )
    state = widget.to_dict()
    assert state["bins"] == 25
    assert state["mode"] == "overlay"
    assert state["series"] == [{"x": [0.0, 1.0], "y": [2.0, 3.0], "label": "h"}]
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


_MINIMAL_KWARGS: dict[type, dict[str, Any]] = {
    BarPlotWidget: {"series": [{"x": [0, 1], "y": [1, 2]}]},
    HistogramWidget: {"series": [{"x": [0, 1], "y": [1, 2]}]},
    HeatmapMatrixWidget: {"x_items": ["A"], "y_items": ["B"]},
    SpacegroupBarPlotWidget: {"data": [225]},
}


@pytest.mark.parametrize(
    ("widget_cls", "attr", "expected"),
    [
        (BarPlotWidget, "orientation", "vertical"),
        (BarPlotWidget, "mode", "overlay"),
        (HistogramWidget, "bins", 100),
        (HistogramWidget, "mode", "single"),
        (HistogramWidget, "show_legend", True),
        (PeriodicTableWidget, "color_scale", "interpolateViridis"),
        (PeriodicTableWidget, "log_scale", False),
        (PeriodicTableWidget, "show_color_bar", True),
        (PeriodicTableWidget, "missing_color", "element-category"),
        (ScatterPlot3DWidget, "camera_projection", "perspective"),
        (HeatmapMatrixWidget, "color_scale", "interpolateViridis"),
        (HeatmapMatrixWidget, "tile_size", "50px"),
        (HeatmapMatrixWidget, "gap", "0px"),
        (SpacegroupBarPlotWidget, "show_counts", True),
        (SpacegroupBarPlotWidget, "orientation", "vertical"),
        (RdfPlotWidget, "cutoff", 15),
        (RdfPlotWidget, "n_bins", 75),
        (RdfPlotWidget, "mode", "element_pairs"),
    ],
)
def test_widget_traitlet_defaults(widget_cls: type, attr: str, expected: Any) -> None:
    """Widgets expose correct default traitlet values."""
    widget = widget_cls(**_MINIMAL_KWARGS.get(widget_cls, {}))
    assert getattr(widget, attr) == expected


@pytest.mark.parametrize(
    ("widget_cls", "kwargs", "expected_attrs"),
    [
        (
            DosWidget,
            {"dos": {"energies": [0, 1]}, "sigma": 0.05, "spin_mode": "combined"},
            {
                "sigma": 0.05,
                "spin_mode": "combined",
                "show_controls": True,
                "stack": None,
            },
        ),
        (
            ConvexHullWidget,
            {
                "entries": [{"composition": {"Li": 1}, "energy": -1.5}],
                "show_unstable": False,
                "temperature": 300.0,
            },
            {
                "show_stable": True,
                "show_unstable": False,
                "temperature": 300.0,
                "hull_face_opacity": None,
            },
        ),
    ],
)
def test_widget_custom_and_default_traitlets(
    widget_cls: type,
    kwargs: dict[str, Any],
    expected_attrs: dict[str, Any],
) -> None:
    """Widgets preserve custom constructor values alongside correct defaults."""
    widget = widget_cls(**kwargs)
    for attr, expected in expected_attrs.items():
        assert getattr(widget, attr) == expected, f"{widget_cls.__name__}.{attr}"


# === Input normalization at widget level ===


def test_convex_hull_widget_from_phase_diagram() -> None:
    """ConvexHullWidget accepts a pymatgen PhaseDiagram directly."""
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

    widget = ConvexHullWidget(
        entries=PhaseDiagram(
            [
                PDEntry(Composition("Fe"), -4.0),
                PDEntry(Composition("O"), -3.0),
                PDEntry(Composition("Fe2O3"), -25.0),
            ]
        )
    )
    assert widget.widget_type == "convex_hull"
    assert len(widget.entries) == 3
    assert all("composition" in entry for entry in widget.entries)


def test_xrd_widget_from_diffraction_pattern() -> None:
    """XrdWidget normalizes DiffractionPattern to plain floats."""
    from pymatgen.analysis.diffraction.xrd import DiffractionPattern

    widget = XrdWidget(
        patterns=DiffractionPattern(
            x=np.array([10.0, 20.0]),
            y=np.array([100.0, 50.0]),
            hkls=[[{"hkl": (1, 0, 0)}], [{"hkl": (1, 1, 0)}]],
            d_hkls=np.array([2.5, 2.0]),
        )
    )
    assert widget.patterns["x"] == [10.0, 20.0]
    assert type(widget.patterns["x"][0]) is float


def test_histogram_widget_delegates_series_validation() -> None:
    """HistogramWidget constructor delegates to normalize_plot_series validation."""
    with pytest.raises(ValueError, match="must include keys 'x' and 'y'"):
        HistogramWidget(series=[{"x": [0, 1]}])


# === New widget specific behaviors ===


def test_structure_widget_isosurface_settings() -> None:
    """StructureWidget passes isosurface_settings through to_dict."""
    iso_settings = {"isovalue": 0.05, "opacity": 0.6, "positive_color": "#3b82f6"}
    widget = StructureWidget(isosurface_settings=iso_settings)
    state = widget.to_dict()
    assert state["isosurface_settings"] == iso_settings
    assert state["widget_type"] == "structure"


def test_periodic_table_accepts_list_input() -> None:
    """PeriodicTableWidget accepts list of values (not just dict)."""
    widget = PeriodicTableWidget(heatmap_values=[1.0, 4.0, 6.9])
    assert widget.heatmap_values == [1.0, 4.0, 6.9]
    assert widget.widget_type == "periodic_table"


@pytest.mark.parametrize(
    ("widget_cls", "kwargs", "expected_type"),
    [
        (PeriodicTableWidget, {}, "periodic_table"),
        (RdfPlotWidget, {}, "rdf_plot"),
        (ScatterPlot3DWidget, {}, "scatter_plot_3d"),
        (ChemPotDiagramWidget, {}, "chem_pot_diagram"),
        (HeatmapMatrixWidget, {}, "heatmap_matrix"),
        (SpacegroupBarPlotWidget, {}, "spacegroup_bar"),
    ],
)
def test_new_widgets_construct_with_no_data(
    widget_cls: type, kwargs: dict[str, Any], expected_type: str
) -> None:
    """New widgets construct successfully with empty/no data."""
    widget = widget_cls(**kwargs)
    assert widget.widget_type == expected_type
    assert isinstance(widget.to_dict(), dict)


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
    ],
)
def test_widget_normalizes_numpy_values(
    widget_cls: type, kwargs: dict[str, Any], attr: str, expected: list[Any]
) -> None:
    """Widgets normalize numpy arrays to plain Python lists."""
    widget = widget_cls(**kwargs)
    result = getattr(widget, attr)
    assert result == expected
    assert type(result) is list


def test_heatmap_matrix_values_none_passthrough() -> None:
    """HeatmapMatrixWidget passes None values through as None (not empty list)."""
    widget = HeatmapMatrixWidget(x_items=["A"], y_items=["B"])
    assert widget.values is None
    assert widget.to_dict()["values"] is None


def test_heatmap_matrix_accepts_nested_dict_values() -> None:
    """HeatmapMatrixWidget accepts dict-of-dicts as values."""
    widget = HeatmapMatrixWidget(
        x_items=["Fe", "O"],
        y_items=["Fe", "O"],
        values={"Fe": {"Fe": 1.0, "O": 0.5}, "O": {"Fe": 0.5, "O": 1.0}},
    )
    assert isinstance(widget.values, dict)
    assert widget.values["Fe"]["O"] == 0.5


def test_rdf_plot_accepts_pymatgen_structure() -> None:
    """RdfPlotWidget normalizes pymatgen Structure to dict via _to_dict."""
    from pymatgen.core import Lattice, Structure

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


def test_fermi_surface_widget_dual_input() -> None:
    """FermiSurfaceWidget accepts exactly one data source."""
    assert FermiSurfaceWidget(fermi_data={"iso": []}).band_data is None
    assert FermiSurfaceWidget(band_data={"e": []}).fermi_data is None


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


# === show() method ===


def test_show_calls_ipython_display_and_returns_none() -> None:
    """show() calls IPython.display.display and returns None."""
    with patch("pymatviz.widgets.matterviz.fetch_widget_asset", return_value="x"):
        widget = MatterVizWidget(widget_type="test")

    with patch("IPython.display.display") as mock_display:
        result = widget.show()

    assert result is None
    mock_display.assert_called_once_with(widget)


@pytest.mark.parametrize(
    "widget_cls",
    [ScatterPlotWidget, BarPlotWidget, HistogramWidget],
)
def test_show_not_shadowed_by_display_traitlet(widget_cls: type) -> None:
    """show() remains callable on widgets that define a 'display' traitlet."""
    widget = widget_cls(
        series=[{"x": [0, 1], "y": [1, 2]}],
        display={"x_grid": True},
    )
    assert callable(widget.show)
    assert isinstance(widget.display, dict)

    with patch("IPython.display.display") as mock_display:
        result = widget.show()

    assert result is None
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
        ("plot.PDF", None, "pdf"),
        ("plot.SVG", None, "svg"),
        ("plot.JPG", None, "jpeg"),
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
        widget.to_img(filename=resolved, fmt=explicit_fmt)  # type: ignore[arg-type]

    assert mock_render.call_args.kwargs["fmt"] == expected_capture_fmt


def test_to_img_rejects_unknown_extension(tmp_path: Any) -> None:
    """to_img raises ValueError for unrecognized file extensions."""
    widget = _make_widget()
    with pytest.raises(ValueError, match="Unsupported file extension"):
        widget.to_img(filename=f"{tmp_path}/plot.tiff")


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


def test_to_img_passes_pdf_format_directly() -> None:
    """to_img passes fmt='pdf' directly to the headless renderer."""
    widget = _make_widget()
    with patch(
        f"{_HEADLESS}.render_widget_headless", return_value=b"%PDF-fake"
    ) as mock_render:
        result = widget.to_img(fmt="pdf", dpi=200)

    assert mock_render.call_args.kwargs["fmt"] == "pdf"
    assert mock_render.call_args.kwargs["dpi"] == 200
    assert result == b"%PDF-fake"


def test_to_img_passes_quality() -> None:
    """to_img forwards quality parameter to the headless renderer."""
    widget = _make_widget()
    with patch(
        f"{_HEADLESS}.render_widget_headless", return_value=_FAKE_PNG
    ) as mock_render:
        widget.to_img(fmt="jpeg", quality=75)

    assert mock_render.call_args.kwargs["quality"] == 75


@pytest.mark.parametrize(
    ("width", "height"),
    [(1200, None), (None, 900), (1200, 900)],
)
def test_to_img_passes_width_height(width: int | None, height: int | None) -> None:
    """to_img forwards width/height overrides to the headless renderer."""
    widget = _make_widget()
    with patch(
        f"{_HEADLESS}.render_widget_headless", return_value=_FAKE_PNG
    ) as mock_render:
        widget.to_img(fmt="png", width=width, height=height)

    assert mock_render.call_args.kwargs["width"] == width
    assert mock_render.call_args.kwargs["height"] == height


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


def test_build_html_embeds_data_and_css() -> None:
    """_build_html injects widget JSON data and CSS into the page."""
    from pymatviz.widgets._headless import _build_html

    data = {"widget_type": "scatter_plot", "series": [{"x": [1]}]}
    css = "body { color: red; }"
    html = _build_html(data, "// esm", css)
    assert '"widget_type": "scatter_plot"' in html
    assert f"<style>{css}</style>" in html


def test_shutdown_browser_tolerates_none() -> None:
    """_shutdown_browser handles _browser=None and _pw=None without error."""
    import pymatviz.widgets._headless as headless_mod
    from pymatviz.widgets._headless import _shutdown_browser

    orig_browser = headless_mod._browser
    orig_pw = headless_mod._pw
    headless_mod._browser = None
    headless_mod._pw = None
    try:
        _shutdown_browser()
    finally:
        headless_mod._browser = orig_browser
        headless_mod._pw = orig_pw


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


def test_canvas_widget_types_frozenset() -> None:
    """_CANVAS_WIDGET_TYPES contains the expected WebGL widget types."""
    from pymatviz.widgets._headless import _CANVAS_WIDGET_TYPES

    expected = {
        "structure",
        "trajectory",
        "fermi_surface",
        "brillouin_zone",
        "scatter_plot_3d",
    }
    assert expected == _CANVAS_WIDGET_TYPES


@pytest.mark.parametrize(
    ("fmt", "dpi", "expected_scale"),
    [
        ("png", 72, 1.0),
        ("png", 144, 2.0),
        ("png", 216, 3.0),
        ("png", 36, 1.0),
        ("jpeg", 72, 1.0),
        ("jpeg", 144, 2.0),
        ("svg", 300, 1),
    ],
)
def test_render_widget_headless_dpi_to_scale(
    fmt: str,
    dpi: int,
    expected_scale: float,
) -> None:
    """render_widget_headless maps DPI to device_scale_factor correctly."""
    fake_svg = '<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg"/>'
    mock_page = MagicMock()
    mock_page.evaluate = MagicMock(side_effect=[None, fake_svg])
    mock_page.locator.return_value.screenshot = MagicMock(return_value=b"\x89PNG")

    mock_browser = MagicMock()
    mock_browser.new_page = MagicMock(return_value=mock_page)

    with patch(f"{_HEADLESS}._get_browser", return_value=mock_browser):
        from pymatviz.widgets._headless import render_widget_headless

        render_widget_headless(
            {"widget_type": "bar_plot"},
            "// esm",
            "/* css */",
            fmt=fmt,
            dpi=dpi,
        )
    scale = mock_browser.new_page.call_args.kwargs["device_scale_factor"]
    assert scale == expected_scale


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
def test_headless_bar_plot_jpeg(bar_plot_widget: BarPlotWidget) -> None:
    """BarPlotWidget.to_img(fmt='jpeg') produces a valid JPEG."""
    jpeg_bytes = bar_plot_widget.to_img(fmt="jpeg", dpi=72)
    assert jpeg_bytes[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert len(jpeg_bytes) > 1000


@_skip_no_playwright
def test_headless_bar_plot_jpeg_from_jpg_extension(
    bar_plot_widget: BarPlotWidget,
    tmp_path: Any,
) -> None:
    """to_img infers JPEG format from .jpg extension."""
    out_path = f"{tmp_path}/bar.jpg"
    result = bar_plot_widget.to_img(filename=out_path, dpi=72)
    assert result[:2] == b"\xff\xd8"
    with open(out_path, "rb") as fh:
        assert fh.read() == result


@_skip_no_playwright
def test_headless_bar_plot_pdf(bar_plot_widget: BarPlotWidget) -> None:
    """BarPlotWidget.to_img(fmt='pdf') produces valid PDF."""
    pdf_bytes = bar_plot_widget.to_img(fmt="pdf")
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 1000


@_skip_no_playwright
def test_headless_bar_plot_writes_file(
    bar_plot_widget: BarPlotWidget,
    tmp_path: Any,
) -> None:
    """to_img writes headless-captured PNG to disk."""
    out_path = f"{tmp_path}/bar.png"
    result = bar_plot_widget.to_img(filename=out_path, fmt="png", dpi=72)
    assert result[:4] == b"\x89PNG"
    with open(out_path, "rb") as fh:
        assert fh.read() == result


@_skip_no_playwright
def test_headless_structure_png() -> None:
    """StructureWidget (canvas/WebGL) produces a valid PNG."""
    from pymatgen.core import Lattice, Structure

    struct = Structure(Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    widget = StructureWidget(structure=struct, style="height: 400px;")
    png_bytes = widget.to_img(fmt="png", dpi=72)
    assert png_bytes[:4] == b"\x89PNG"
    assert len(png_bytes) > 5000


@_skip_no_playwright
def test_headless_structure_svg_raises() -> None:
    """StructureWidget (canvas) raises RuntimeError for SVG export."""
    from pymatgen.core import Lattice, Structure

    struct = Structure(Lattice.cubic(3.0), ["Si"], [[0, 0, 0]])
    widget = StructureWidget(structure=struct, style="height: 300px;")
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
