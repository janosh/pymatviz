"""Real-browser exports per format (PNG, SVG, PDF, JPEG, inline HTML) and render
reports, using the matterviz bundle in headless Chromium.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
from kemio.interop import structure_to_pymatgen_dict

import pymatviz as pmv
from pymatviz.widgets import _headless as headless_mod
from pymatviz.widgets.bar_plot import BarPlotWidget
from pymatviz.widgets.chem_pot_diagram import ChemPotDiagramWidget
from pymatviz.widgets.convex_hull import ConvexHullWidget
from pymatviz.widgets.heatmap_matrix import HeatmapMatrixWidget
from pymatviz.widgets.periodic_table import PeriodicTableWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from pymatviz.widgets.structure import StructureWidget
from pymatviz.widgets.trajectory import TrajectoryWidget
from tests.conftest import cubic_structure
from tests.widgets.conftest import SI_STRUCT


if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatviz.widgets.matterviz import MatterVizWidget

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None, reason="needs playwright"
)

BCC_COORDS = [[0, 0, 0], [0.5, 0.5, 0.5]]
FE_STRUCT = cubic_structure(2.87, ["Fe", "Fe"], BCC_COORDS)
needs_pillow = pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None, reason="needs Pillow"
)


@pytest.fixture
def bar_plot_widget() -> BarPlotWidget:
    """A minimal BarPlotWidget for integration tests."""
    return BarPlotWidget(
        series=[{"x": [0, 1], "y": [3.0, 5.0], "label": "test"}],
        style="height: 400px;",
    )


@needs_pillow
def test_headless_bar_plot_png(bar_plot_widget: BarPlotWidget) -> None:
    """BarPlotWidget.to_img(fmt='png') produces a valid, non-trivial PNG."""
    import io

    from PIL import Image

    png_bytes = bar_plot_widget.to_img(fmt="png", dpi=72)
    assert png_bytes[:4] == b"\x89PNG"
    img = Image.open(io.BytesIO(png_bytes))
    assert img.width >= 400
    assert img.height >= 200


@pytest.mark.parametrize("chart", ["treemap", "sunburst"])
def test_hierarchy_vector_and_html_exports(
    chart: Literal["treemap", "sunburst"], tmp_path: Any
) -> None:
    """Chemical-system charts export vector SVG/PDF and interactive standalone HTML."""
    from playwright.sync_api import expect, sync_playwright

    widget = pmv.chem_sys_hierarchy(
        ["Fe2O3", "Li2O", "Si"],
        kind=chart,
        label_text="label+value",
        show_controls=False,
        style="width: 700px; height: 480px;",
    )
    svg = widget.to_img(fmt="svg").decode()
    assert all(label in svg for label in ("Fe-O", "Li-O", "Si"))
    pdf = widget.to_img(fmt="pdf")
    assert pdf.startswith(b"%PDF-")
    assert b"/Font" in pdf

    html_path = tmp_path / f"{chart}.html"
    widget.to_html(str(html_path), inline=True)
    with sync_playwright() as pw, pw.chromium.launch().new_page() as page:
        page.goto(html_path.as_uri())
        chart_nodes = page.locator(f"[data-{chart}-node-idx][aria-label]")
        chart_nodes.first.wait_for()
        assert chart_nodes.count() == widget.describe()["n_nodes"]
        shapes = page.locator(
            'svg[role="application"] :is(path, rect, circle, text, tspan)'
        )
        # invisible hit targets and measuring text don't reach the file
        paint = """nodes => nodes.map(node => getComputedStyle(node))
            .filter(style =>
                style.fill !== 'rgba(0, 0, 0, 0)' || style.stroke !== 'none')
            .map(style => {
            return [style.fill, style.stroke, style.strokeWidth, style.fontFamily,
                    style.fontSize, style.fontWeight, style.textAnchor,
                    style.dominantBaseline];
        })"""
        live_styles = shapes.evaluate_all(paint)
        page.get_by_role("button", name="binary: 2", exact=True).press("Enter")
        page.locator(".breadcrumbs").get_by_text("binary", exact=True).wait_for()
        expect(chart_nodes).to_have_count(2)  # retries until zoom-out cells leave
        svg_path = tmp_path / f"{chart}.svg"
        svg_path.write_text(svg)
        page.goto(svg_path.as_uri())
        # the exported file drops the live chart's role, so match its shapes directly
        file_shapes = page.locator("svg :is(path, rect, circle, text, tspan)")
        assert file_shapes.evaluate_all(paint) == live_styles


def test_headless_bar_plot_svg(bar_plot_widget: BarPlotWidget) -> None:
    """BarPlotWidget.to_img(fmt='svg') returns a standalone-sized chart SVG."""
    import re

    svg_text = bar_plot_widget.to_img(fmt="svg", width=600, height=450).decode()
    assert svg_text.startswith("<?xml")
    assert len(svg_text) > 2000, f"SVG too small ({len(svg_text)} chars)"
    assert "<g" in svg_text, "No <g> groups -- likely an icon"
    # CSS-sized charts get their rendered px size, else <img> shows them at 300x150
    root = re.search(r"<svg[^>]*>", svg_text)
    assert root is not None
    size = dict(re.findall(r' (width|height|viewBox)="([^"]+)"', root.group()))
    # the viewBox is padded by half the widest stroke so edge strokes aren't clipped
    # the chart fills the requested height too (it used to shrink to its 300 px
    # minimum when only the export root was sized)
    assert 600 <= float(size["width"]) <= 602
    assert 450 <= float(size["height"]) <= 452
    assert size["viewBox"].split()[2:] == [size["width"], size["height"]]
    assert "<title>" in svg_text  # the chart's aria-label as the file's name


@needs_pillow
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
        assert len(re.findall(r'<path [^>]*class="marker\b', svg_text)) == 3
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


@pytest.mark.parametrize(  # JPEG SOI marker, PNG and PDF headers
    ("filename", "fmt", "magic"),
    [
        ("bar.jpg", None, b"\xff\xd8"),
        ("bar.png", "png", b"\x89PNG"),
        ("bar.pdf", "pdf", b"%PDF-"),
    ],
    ids=["jpeg-from-jpg-extension", "explicit-png", "explicit-pdf"],
)
def test_headless_bar_plot_writes_file(
    bar_plot_widget: BarPlotWidget,
    tmp_path: Any,
    filename: str,
    fmt: str | None,
    magic: bytes,
) -> None:
    """to_img writes a valid, non-trivial capture to disk, inferring JPEG from .jpg."""
    out_path = f"{tmp_path}/{filename}"
    result = bar_plot_widget.to_img(filename=out_path, fmt=fmt, dpi=72)  # ty: ignore[invalid-argument-type]
    assert result.startswith(magic)
    assert len(result) > 1000
    with open(out_path, "rb") as fh:
        assert fh.read() == result


@pytest.mark.parametrize("with_plot", [False, True])
def test_headless_webgl_structure_exports(with_plot: bool) -> None:
    """WebGL structures export PNG, rasterize PDFs and raise for SVG, also beside an
    SVG plot.
    """
    if with_plot:
        frames = [
            {
                "structure": structure_to_pymatgen_dict(
                    cubic_structure(3 + 0.1 * idx, ["Si"], [0, 0, 0])
                ),
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
        widget = StructureWidget(
            structure=cubic_structure(3, ["Si", "Si"], BCC_COORDS),
            style="height: 400px;",
        )
    png_bytes = widget.to_img(fmt="png", dpi=72)
    assert png_bytes[:4] == b"\x89PNG"
    assert len(png_bytes) > 5000
    assert b"/Font" not in widget.to_img(fmt="pdf", dpi=72), "expected raster PDF"
    with pytest.raises(RuntimeError, match="no vector form"):
        widget.to_img(fmt="svg")


@needs_pillow
def test_headless_dpi_affects_png_dimensions() -> None:
    """Higher DPI produces larger pixel dimensions for the same widget."""
    import io

    from PIL import Image

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


@needs_pillow
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
    widget = widget_cls(style="height: 400px;", **kwargs)
    png_bytes = widget.to_img(fmt="png", dpi=72)
    assert png_bytes[:4] == b"\x89PNG"
    blank_frac = headless_mod.png_blank_fraction(png_bytes)
    assert blank_frac is not None
    assert blank_frac < max_blank


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
    with pytest.raises(RuntimeError, match="no vector form"):
        widget.to_img(fmt="svg")


def test_headless_repeated_exports_in_one_process() -> None:
    """Repeated to_img() calls succeed (1st sync render flips later ones to async)."""
    bar = BarPlotWidget(series=[{"x": [0, 1], "y": [1.0, 2.0]}], style="height: 300px;")
    scatter = ScatterPlotWidget(
        series=[{"x": [0, 1], "y": [1.0, 2.0]}], style="height: 300px;"
    )
    for widget in (bar, scatter, bar):
        assert widget.to_img(fmt="png", dpi=72)[:4] == b"\x89PNG"


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
                structure=SI_STRUCT,
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


@needs_pillow
@pytest.mark.parametrize(
    ("make_widget", "selector"),
    [
        (
            lambda: ScatterPlotWidget(
                series=[{"x": [0, 1, 2], "y": [0, 1, 4], "label": "s"}],
                style="width: 500px; height: 350px;",
            ),
            "svg",
        ),
        # default traits (no volumes, no isosurface settings) must not crash the scene
        (lambda: StructureWidget(structure=FE_STRUCT), "canvas"),
    ],
    ids=["scatter", "structure"],
)
def test_to_html_inline_renders_offline(
    tmp_path: Any, make_widget: Callable[[], MatterVizWidget], selector: str
) -> None:
    """to_html(inline=True) produces a self-contained page that actually renders.

    build_interactive_html has its own render path (controls on, no capture
    machinery) that to_img/render_report don't exercise, so load the file and
    confirm it paints without page errors.
    """
    from playwright.sync_api import sync_playwright

    out_path = tmp_path / "widget.html"
    make_widget().to_html(str(out_path), inline=True)

    with (
        sync_playwright() as pw,
        pw.chromium.launch().new_page(viewport={"width": 600, "height": 420}) as page,
    ):
        page_errors: list[str] = []
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        page.goto(out_path.as_uri(), wait_until="domcontentloaded")
        page.wait_for_selector(f"#widget-root {selector}", timeout=20000)
        page.wait_for_timeout(1000)
        png_bytes = page.locator("#widget-root").screenshot(type="png")

    assert page_errors == []
    assert png_bytes[:4] == b"\x89PNG"
    blank_frac = headless_mod.png_blank_fraction(png_bytes)
    assert blank_frac is not None
    assert blank_frac < 0.99  # real chart, not a blank frame
