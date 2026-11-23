"""Tests for spectral and reciprocal-space widgets: DOS, band structure, Brillouin
zone, Fermi surface, XRD and RDF.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import kemio
import numpy as np
import pytest
import traitlets as tl
from kemio.interop import structure_to_pymatgen_dict

import pymatviz as pmv
from pymatviz.widgets.bands_and_dos import BandsAndDosWidget
from pymatviz.widgets.brillouin_zone import BrillouinZoneWidget
from pymatviz.widgets.dos import DosWidget
from pymatviz.widgets.fermi_surface import FermiSurfaceWidget
from pymatviz.widgets.rdf_plot import RdfPlotWidget
from pymatviz.widgets.xrd import XrdWidget
from tests.widgets.conftest import SI_STRUCT


if TYPE_CHECKING:
    from pymatviz.widgets.matterviz import MatterVizWidget


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
        with pytest.raises(expected, match=r"sigma.*not be less than 0\.0"):
            widget_cls(sigma=sigma)
    else:
        assert widget_cls(sigma=sigma).to_dict()["sigma"] == expected


@pytest.mark.parametrize(
    ("widget_cls", "trait"),
    [
        (RdfPlotWidget, "structures"),
        (BandsAndDosWidget, "structure"),
        (BrillouinZoneWidget, "structure"),
    ],
)
def test_widgets_accept_structure(widget_cls: type, trait: str) -> None:
    """Structure traits sync a kemio Structure as a dict."""
    synced = getattr(widget_cls(**{trait: SI_STRUCT}), trait)
    assert synced == structure_to_pymatgen_dict(SI_STRUCT)


H2_MOLECULE = kemio.Structure(
    None, ["H", "H"], [[0, 0, 0], [0, 0, 0.74]], pbc=(False,) * 3
)


@pytest.mark.parametrize(
    ("widget_cls", "kwargs", "match"),
    [
        # converted by the trait, not a TypeError from __init__ like before
        (
            BandsAndDosWidget,
            {"band_structs": "invalid"},
            "Unsupported type for band structure",
        ),
        # a Brillouin zone needs a lattice
        (BandsAndDosWidget, {"structure": H2_MOLECULE}, "periodic structure"),
        (BrillouinZoneWidget, {"structure": H2_MOLECULE}, "periodic structure"),
        (
            RdfPlotWidget,
            {
                "structures": {"lattice": {}, "sites": []},
                "patterns": [{"r": [1], "g_r": [1]}],
            },
            "not both",
        ),
        (
            FermiSurfaceWidget,
            {"fermi_data": None, "band_data": None},
            "exactly one of fermi_data or band_data",
        ),
        (
            FermiSurfaceWidget,
            {"fermi_data": {"isosurfaces": []}, "band_data": {"energies": []}},
            "exactly one of fermi_data or band_data",
        ),
        # normalized values must still be the dict the frontend reads
        (
            FermiSurfaceWidget,
            {"fermi_data": [1, 2]},
            r"'fermi_data' trait of a FermiSurfaceWidget .* dict or None, not the list",
        ),
        (
            BrillouinZoneWidget,
            {"bz_data": [1]},
            r"'bz_data' trait of a BrillouinZoneWidget .* dict or None, not the list",
        ),
        (
            RdfPlotWidget,
            {"patterns": "abc"},
            r"'patterns' trait of a RdfPlotWidget .* a list, not the str 'abc'",
        ),
    ],
)
def test_widgets_reject_invalid_data_sources(
    widget_cls: type, kwargs: dict[str, Any], match: str
) -> None:
    """Molecules for Brillouin zones and missing or conflicting data sources raise."""
    with pytest.raises(tl.TraitError, match=match):
        widget_cls(**kwargs)


def test_xrd_widget_labeled_patterns() -> None:
    """A {label: pattern} mapping normalizes each pattern, keeping label order."""
    computed = kemio.XrdPattern(
        [10, 20], [100, 50], d_spacings=[2, 1], hkls=[[([1, 0, 0], 6)]] * 2
    )
    widget = XrdWidget(patterns={"computed": computed, "scan": {"x": [5], "y": [1]}})
    assert list(widget.patterns) == ["computed", "scan"]
    first_hkls = widget.patterns["computed"]["hkls"][0]
    assert first_hkls == [{"hkl": [1, 0, 0], "multiplicity": 6}]
    assert widget.patterns["scan"] == {"x": [5], "y": [1], "kind": "profile"}
    with pytest.raises(tl.TraitError, match="patterns mapping must not be empty"):
        XrdWidget(patterns={})
    # labels sync as strings (JSON object keys)
    assert list(XrdWidget(patterns={1: {"x": [5], "y": [1]}}).patterns) == ["1"]


@pytest.mark.parametrize("labeled", [False, True])
def test_vasprun_paths(labeled: bool) -> None:
    """vasprun.xml paths (single or as dict values) give kemio's bands and DOS."""
    from kemio.io import read_vasprun

    from pymatviz.data import TEST_FILES

    path = f"{TEST_FILES}/vasprun/MgO.vasprun.xml.xz"
    vasprun = read_vasprun(path)
    ref_bands, ref_dos = vasprun.bands.as_dict(), vasprun.dos.as_dict()
    inputs = {"MgO": path} if labeled else path
    widget = pmv.bands_and_dos(band_structs=inputs, doses=inputs)
    bands = widget.band_structs["MgO"] if labeled else widget.band_structs
    doses = widget.doses["MgO"] if labeled else widget.doses
    np.testing.assert_array_equal(bands["bands"]["1"], ref_bands["bands"]["1"])
    np.testing.assert_array_equal(bands["kpoints"], ref_bands["kpoints"])
    assert bands["efermi"] == ref_bands["efermi"]
    np.testing.assert_array_equal(doses["energies"], ref_dos["energies"])
    np.testing.assert_array_equal(doses["densities"]["1"], ref_dos["densities"]["1"])
    assert pmv.dos(doses=path).doses["energies"] == doses["energies"]
    assert pmv.band_structure(band_structs=path).band_structs["bands"] == bands["bands"]


@pytest.mark.parametrize("widget_cls", [DosWidget, BandsAndDosWidget])
def test_stack_checks_dos_grids_in_either_order(widget_cls: type) -> None:
    """stack=True rejects DOS on different grids whether doses or stack comes last."""
    doses = {
        "a": kemio.PhononDos([0, 1, 2], [0, 1, 0]),
        "b": kemio.PhononDos([0, 1, 3], [0, 1, 0]),
    }
    match = r"one grid, but \['b'\]"
    with pytest.raises(tl.TraitError, match=match):
        widget_cls(doses=doses, stack=True)
    with pytest.raises(tl.TraitError, match=match):
        widget_cls(doses=doses).stack = True
    widget = widget_cls(stack=True)
    with pytest.raises(tl.TraitError, match=match):
        widget.doses = doses
    # a single (unlabeled) DOS and DOS on one grid stack fine
    widget.set_trait("doses", doses["a"])
    assert widget.doses["frequencies"] == [0, 1, 2]
    widget.set_trait(
        "doses", {"a": doses["a"], "c": kemio.PhononDos([0, 1, 2], [1, 1, 1])}
    )
    assert list(widget.doses) == ["a", "c"]
