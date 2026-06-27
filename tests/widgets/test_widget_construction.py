"""Tests for widget construction, traitlet defaults, to_dict, and validation."""

from __future__ import annotations

import importlib.util
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

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


def _case(
    widget_cls: type,
    kwargs: dict[str, Any],
    expected_type: str,
    state_key: str,
    expected_state: Any = None,
) -> tuple[type, dict[str, Any], str, str, Any]:
    """Build a widget construction test case."""
    return (
        widget_cls,
        kwargs,
        expected_type,
        state_key,
        kwargs[state_key] if expected_state is None else expected_state,
    )


@pytest.mark.parametrize(
    ("widget_cls", "kwargs", "expected_type", "state_key", "expected_state"),
    [
        _case(
            ConvexHullWidget,
            {"entries": [{"composition": {"Li": 1}, "energy": -1.5}]},
            "convex_hull",
            "entries",
        ),
        _case(
            BandStructureWidget,
            {"band_structure": {"bands": []}},
            "band_structure",
            "band_structure",
        ),
        _case(DosWidget, {"dos": {"energies": [0]}}, "dos", "dos"),
        _case(
            BandsAndDosWidget,
            {"band_structure": {"bands": []}, "dos": {"energies": []}},
            "bands_and_dos",
            "dos",
        ),
        _case(
            FermiSurfaceWidget,
            {"fermi_data": {"isosurfaces": []}},
            "fermi_surface",
            "fermi_data",
        ),
        _case(
            FermiSurfaceWidget,
            {"band_data": {"energies": []}},
            "fermi_surface",
            "band_data",
        ),
        _case(
            BrillouinZoneWidget,
            {"structure": {"lattice": {}, "sites": []}},
            "brillouin_zone",
            "structure",
        ),
        _case(
            PhaseDiagramWidget,
            {"data": {"components": ["A", "B"]}},
            "phase_diagram",
            "data",
        ),
        _case(XrdWidget, {"patterns": {"x": [10], "y": [100]}}, "xrd", "patterns"),
        _case(
            ScatterPlotWidget,
            {"series": [{"x": [0, 1], "y": [1, 2], "label": "curve"}]},
            "scatter_plot",
            "series",
            [{"x": [0.0, 1.0], "y": [1.0, 2.0], "label": "curve"}],
        ),
        _case(
            BarPlotWidget,
            {"series": [{"x": [0, 1], "y": [2, 3], "label": "bars"}]},
            "bar_plot",
            "series",
            [{"x": [0.0, 1.0], "y": [2.0, 3.0], "label": "bars"}],
        ),
        _case(
            HistogramWidget,
            {"series": [{"x": [0, 1], "y": [2, 2.5], "label": "hist"}]},
            "histogram",
            "series",
            [{"x": [0.0, 1.0], "y": [2.0, 2.5], "label": "hist"}],
        ),
        _case(
            PeriodicTableWidget,
            {"heatmap_values": {"Fe": 42, "O": 100}},
            "periodic_table",
            "heatmap_values",
        ),
        _case(
            RdfPlotWidget,
            {"structures": {"lattice": {}, "sites": []}},
            "rdf_plot",
            "structures",
        ),
        _case(
            ScatterPlot3DWidget,
            {"series": [{"x": [1], "y": [2], "z": [3], "label": "pt"}]},
            "scatter_plot_3d",
            "series",
        ),
        _case(
            HeatmapMatrixWidget,
            {"x_items": ["A", "B"], "y_items": ["C", "D"], "values": [[1, 2], [3, 4]]},
            "heatmap_matrix",
            "values",
        ),
        _case(
            SpacegroupBarPlotWidget, {"data": [225, 166, 62]}, "spacegroup_bar", "data"
        ),
        _case(
            ChemPotDiagramWidget,
            {"entries": [{"name": "Li2O", "energy": -14.3}]},
            "chem_pot_diagram",
            "entries",
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
    ("widget_cls", "kwargs"),
    [
        (ScatterPlotWidget, {"series": [{"x": [0, 1], "y": [1, 2], "label": "s"}]}),
        (
            HistogramWidget,
            {"series": [{"x": [0, 1], "y": [1, 2]}], "bins": 50, "mode": "overlay"},
        ),
        (BarPlotWidget, {"series": [{"x": [0], "y": [1]}], "mode": "grouped"}),
        (DosWidget, {"dos": {"energies": [0, 1]}, "sigma": 0.05}),
        (ConvexHullWidget, {"entries": [{"composition": {"Li": 1}, "energy": -1.5}]}),
        (PeriodicTableWidget, {"heatmap_values": {"Fe": 42}}),
        (ScatterPlot3DWidget, {"series": [{"x": [1], "y": [2], "z": [3]}]}),
        (HeatmapMatrixWidget, {"x_items": ["A"], "y_items": ["B"]}),
        (SpacegroupBarPlotWidget, {"data": [225]}),
        (ChemPotDiagramWidget, {"entries": [{"name": "Li", "energy": -1.9}]}),
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
        (SpacegroupBarPlotWidget, "show_legend", False),
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


# === Input normalization at widget level ===


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


def test_structure_widget_vector_props() -> None:
    """StructureWidget exposes vector_configs and all vector props."""
    configs = {"force": {"visible": True, "color": "#e74c3c", "scale": None}}
    widget = StructureWidget(
        vector_configs=configs,
        vector_scale=3.0,
        vector_color="#00ff00",
        vector_normalize=True,
        vector_uniform_thickness=True,
        vector_origin_gap=0.25,
    )
    state = widget.to_dict()
    assert state["vector_configs"] == configs
    assert state["vector_scale"] == 3.0
    assert state["vector_color"] == "#00ff00"
    assert state["vector_normalize"] is True
    assert state["vector_uniform_thickness"] is True
    assert state["vector_origin_gap"] == 0.25


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
        (
            SpacegroupBarPlotWidget,
            {"data": np.array([225, 166, 62])},
            "data",
            [225, 166, 62],
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


def test_heatmap_matrix_scalar_string_axis_items() -> None:
    """HeatmapMatrixWidget treats scalar string axis items as one label."""
    widget = HeatmapMatrixWidget(x_items="Fe", y_items="O")
    assert widget.x_items == [{"key": "Fe", "label": "Fe"}]
    assert widget.y_items == [{"key": "O", "label": "O"}]


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


def test_html_widget_types_frozenset() -> None:
    """_HTML_WIDGET_TYPES holds the widgets that render plain HTML (no chart)."""
    from pymatviz.widgets._headless import _HTML_WIDGET_TYPES

    expected = {"periodic_table", "heatmap_matrix"}
    assert expected == _HTML_WIDGET_TYPES


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
    struct = Structure(Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    widget = StructureWidget(structure=struct, style="height: 400px;")
    png_bytes = widget.to_img(fmt="png", dpi=72)
    assert png_bytes[:4] == b"\x89PNG"
    assert len(png_bytes) > 5000


@_skip_no_playwright
def test_headless_structure_svg_raises() -> None:
    """StructureWidget (canvas) raises RuntimeError for SVG export."""
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
