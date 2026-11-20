"""Render every widget type in headless Chromium with the real MatterViz bundle.

Python-side tests can't see what only fails in the browser (a bundle rejecting a
payload, a blank canvas, an SVG export missing its size), so each widget type gets
one realistic instance that must render, paint and export. Runs against the bundle
pymatviz pins (matterviz main's docs-site build while that is "main") unless
MATTERVIZ_ANYWIDGET_DIR points at a local build.
"""

from __future__ import annotations

import importlib.util
import re
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure

import pymatviz as pmv
from pymatviz.widgets import (
    BandsAndDosWidget,
    BandStructureWidget,
    BarPlotWidget,
    BinnedScatterPlotWidget,
    BoxPlotWidget,
    BrillouinZoneWidget,
    ChemPotDiagramWidget,
    CompositionWidget,
    ConvexHullWidget,
    CoordinationBarPlotWidget,
    DosWidget,
    FermiSurfaceWidget,
    HeatmapMatrixWidget,
    HistogramWidget,
    PeriodicTableWidget,
    PhaseDiagramWidget,
    RdfPlotWidget,
    SankeyWidget,
    ScatterPlot3DWidget,
    ScatterPlotWidget,
    SpacegroupBarPlotWidget,
    StructureWidget,
    SunburstWidget,
    TrajectoryWidget,
    TreemapWidget,
    XrdWidget,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pymatviz.widgets.matterviz import MatterVizWidget

TIMEOUT = 60  # seconds per render, generous for slow CI runners
SVG_NS = "{http://www.w3.org/2000/svg}"
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("playwright") is None, reason="needs playwright"
)


@pytest.fixture(autouse=True, scope="module")
def _stop_sync_browser() -> Iterator[None]:
    """Shut the cached sync browser after this module: sync Playwright keeps an event
    loop running, which breaks later asyncio.run() tests in the same process.
    """
    yield
    from pymatviz.widgets._headless import _shutdown_browser

    _shutdown_browser()


NACL = Structure.from_spacegroup(
    "Fm-3m", Lattice.cubic(5.6), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
)
FE_O = Structure(Lattice.cubic(3), ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
RNG = np.random.default_rng(seed=0)
XS, YS = RNG.normal(size=(2, 300))
HULL_ENTRIES = [
    {"composition": {"Li": 1}, "energy": -1.9},
    {"composition": {"O": 1}, "energy": -3.0},
    {"composition": {"Li": 2, "O": 1}, "energy": -14.3},
    {"composition": {"Li": 1, "O": 1}, "energy": -9.0},
]


def _free_electron_bands() -> dict[str, Any]:
    """One parabolic band on an 8^3 k-grid, cut by the Fermi level into a sphere."""
    k_axis = np.linspace(-0.5, 0.5, 8)
    kx, ky, kz = np.meshgrid(k_axis, k_axis, k_axis, indexing="ij")
    return {
        "energies": [[(kx**2 + ky**2 + kz**2).tolist()]],
        "k_grid": [8, 8, 8],
        "k_lattice": np.eye(3).tolist(),
        "fermi_energy": 0.15,
    }


def _binary_phase_diagram() -> dict[str, Any]:
    """Liquid over a two-solid field: the smallest diagram MatterViz accepts."""
    phases = [
        {"id": id_, "model_phase": id_, "label": id_, "kind": kind}
        for id_, kind in (("L", "liquid"), ("a", "solid"), ("b", "solid"))
    ]
    lists = ("boundaries", "invariants", "congruent_points", "critical_points")
    return {
        "format": "binary-phase-diagram/2",
        "components": ["A", "B"],
        "pressure_gpa": 0.0001,
        "temperature_range": [300, 1500],
        "phases": phases,
        "regions": [
            {"phases": ["L"], "vertices": [[0, 900], [1, 900], [1, 1500], [0, 1500]]},
            {
                "phases": ["a", "b"],
                "vertices": [[0, 300], [1, 300], [1, 900], [0, 900]],
            },
        ],
        **{key: [] for key in lists},
        "unary_transitions": [],
        "polymorphic_transitions": [],
        "provenance": {"kind": "sketch"},
    }


def _phonons(request: pytest.FixtureRequest, key: str) -> dict[str, Any]:
    """DFT and MACE phonon ``bands`` or ``doses`` of mp-2758 (conftest fixture)."""
    return request.getfixturevalue("phonon_bands_doses_mp_2758")[key]


# widget class -> (factory, whether the figure has a vector form). Factories take the
# test request for conftest fixtures. WebGL viewers and 3D plots have no vector form,
# so their SVG export must fail with MatterViz's reason.
WIDGET_CASES: dict[
    type[MatterVizWidget],
    tuple[Callable[[pytest.FixtureRequest], MatterVizWidget], bool],
] = {
    BandStructureWidget: (
        lambda req: BandStructureWidget(band_structs=_phonons(req, "bands")),
        True,
    ),
    BandsAndDosWidget: (
        lambda req: BandsAndDosWidget(
            band_structs=_phonons(req, "bands"),
            doses=_phonons(req, "doses"),
        ),
        True,
    ),
    BarPlotWidget: (lambda _: pmv.elements_hist(["Fe2O3", "NaCl", "LiFePO4"]), True),
    BinnedScatterPlotWidget: (
        lambda _: BinnedScatterPlotWidget(series=[{"x": XS, "y": YS}]),
        True,
    ),
    BoxPlotWidget: (
        lambda _: BoxPlotWidget(series=[{"y": XS, "label": "a"}, {"y": YS + 1}]),
        True,
    ),
    BrillouinZoneWidget: (lambda _: BrillouinZoneWidget(structure=NACL), False),
    ChemPotDiagramWidget: (
        lambda _: ChemPotDiagramWidget(
            entries=[
                {"name": name, "energy": energy, "composition": comp}
                for name, energy, comp in (
                    ("Li", -1.9, {"Li": 1}),
                    ("O2", -9.9, {"O": 2}),
                    ("Li2O", -14.3, {"Li": 2, "O": 1}),
                )
            ]
        ),
        True,
    ),
    CompositionWidget: (lambda _: CompositionWidget(composition="LiFePO4"), True),
    ConvexHullWidget: (lambda _: ConvexHullWidget(entries=HULL_ENTRIES), True),
    CoordinationBarPlotWidget: (
        lambda _: CoordinationBarPlotWidget({"NaCl": NACL, "FeO": FE_O}),
        True,
    ),
    DosWidget: (lambda req: DosWidget(doses=_phonons(req, "doses")), True),
    FermiSurfaceWidget: (
        lambda _: FermiSurfaceWidget(band_data=_free_electron_bands()),
        False,
    ),
    HeatmapMatrixWidget: (
        lambda _: HeatmapMatrixWidget(
            x_items=["Fe", "O", "Na"],
            y_items=["Fe", "O"],
            values=[[1, 0.5, 0.1], [0.5, 1, 0.3]],
        ),
        True,
    ),
    HistogramWidget: (
        lambda _: HistogramWidget(series=[{"values": XS}, {"values": YS}]),
        True,
    ),
    PeriodicTableWidget: (
        lambda _: pmv.ptable_heatmap({"Fe": 3, "O": 7, "Na": 1, "Cl": 2}),
        True,
    ),
    PhaseDiagramWidget: (
        lambda _: PhaseDiagramWidget(data=_binary_phase_diagram()),
        True,
    ),
    RdfPlotWidget: (lambda _: RdfPlotWidget(structures=NACL), True),
    SankeyWidget: (
        lambda _: SankeyWidget(
            {
                "nodes": [{"label": "a"}, {"label": "b"}, {"label": "c"}],
                "links": [
                    {"source": 0, "target": 1, "value": 3},
                    {"source": 0, "target": 2, "value": 1},
                ],
            }
        ),
        True,
    ),
    ScatterPlotWidget: (lambda _: pmv.density_scatter(XS, YS), True),
    ScatterPlot3DWidget: (
        lambda _: ScatterPlot3DWidget(series=[{"x": XS, "y": YS, "z": XS * YS}]),
        False,
    ),
    SpacegroupBarPlotWidget: (lambda _: pmv.spacegroup_bar([225, 225, 194, 62]), True),
    StructureWidget: (lambda _: StructureWidget(structure=NACL), False),
    SunburstWidget: (lambda _: pmv.spacegroup_sunburst([225, 225, 194, 62]), True),
    TrajectoryWidget: (lambda _: TrajectoryWidget(trajectory=[NACL, FE_O]), False),
    TreemapWidget: (lambda _: pmv.chem_sys_treemap(["Fe2O3", "NaCl", "Fe"]), True),
    XrdWidget: (
        lambda _: XrdWidget(patterns=XRDCalculator().get_pattern(NACL)),
        True,
    ),
}


def test_cases_cover_every_widget_class() -> None:
    """Every exported MatterVizWidget subclass has a render case."""
    import inspect

    import pymatviz.widgets as widgets_mod
    from pymatviz.widgets.matterviz import MatterVizWidget

    widget_classes = {
        cls.__name__
        for _, cls in inspect.getmembers(widgets_mod, inspect.isclass)
        if issubclass(cls, MatterVizWidget) and cls is not MatterVizWidget
    }
    assert widget_classes == {cls.__name__ for cls in WIDGET_CASES}


@pytest.mark.parametrize("widget_cls", WIDGET_CASES, ids=lambda cls: cls.__name__)
def test_widget_renders_and_exports(
    widget_cls: type[MatterVizWidget], request: pytest.FixtureRequest
) -> None:
    """The widget mounts without error, paints, and exports SVG iff it is vector."""
    factory, vector = WIDGET_CASES[widget_cls]
    widget = factory(request)
    # by name: other tests reload widget modules, which swaps the class objects
    assert type(widget).__name__ == widget_cls.__name__

    report = widget.render_report(timeout=TIMEOUT)
    assert report.error is None, report.error
    assert not report.is_blank, f"{widget_cls.__name__} rendered blank: {report}"

    if not vector:
        with pytest.raises(RuntimeError, match="no vector form"):
            widget.to_img(fmt="svg", timeout=TIMEOUT)
        return
    markup = widget.to_img(fmt="svg", timeout=TIMEOUT).decode()
    svg = ET.fromstring(markup)  # noqa: S314 (our own export)
    assert svg.tag == f"{SVG_NS}svg"
    # sized like the live figure, else an <img> falls back to 300x150
    assert float(svg.get("width", 0)) > 100, svg.attrib
    assert float(svg.get("height", 0)) > 100, svg.attrib
    # real content, not just a background: labels and marks
    tags = {node.tag.removeprefix(SVG_NS) for node in svg.iter()}
    assert "text" in tags, f"{widget_cls.__name__} exported no text: {tags}"
    assert tags & {"path", "rect", "circle", "line", "polygon", "polyline"}, tags
    # every panel of multi-panel figures (bands + DOS: root plus both plots)
    min_svgs = 3 if widget_cls.__name__ == "BandsAndDosWidget" else 1
    assert markup.count("<svg") >= min_svgs
    # the page's sans font (on the root, or per text of redrawn HTML figures), not
    # the browser's serif default
    assert "system-ui" in markup
    assert "Times" not in markup
    # static files drop keyboard/pointer wiring and keep 0.01 px coordinates
    assert not re.search(r' (tabindex|role|data-[\w-]+)="', markup)
    assert not re.search(r' (d|x|y|cx|cy|points)="[^"]*\d\.\d{3}', markup)
