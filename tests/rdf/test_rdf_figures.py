from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from pymatviz.rdf.figures import element_pair_rdfs, full_rdf
from pymatviz.rdf.helpers import calculate_rdf
from pymatviz.widgets.rdf_plot import RdfPlotWidget
from tests.conftest import SI_ATOMS, SI_STRUCTS


if TYPE_CHECKING:
    from collections.abc import Callable

EMPTY_STRUCT = Structure(Lattice.cubic(1), [], [])
NACL = Structure(Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.mark.parametrize("rdf_func", [element_pair_rdfs, full_rdf])
@pytest.mark.parametrize(
    ("structs", "kwargs", "err_cls", "match"),
    [
        (EMPTY_STRUCT, {}, ValueError, "input structure contains no sites"),
        ({"blank": EMPTY_STRUCT}, {}, ValueError, "input structure blank contains no"),
        ("not a structure", {}, TypeError, "Input must be a pymatgen Structure"),
        (
            SI_STRUCTS,
            {"n_bins": 100, "bin_size": 0.1},
            ValueError,
            "Cannot specify both n_bins=.* and bin_size=.*",
        ),
    ],
    ids=["empty", "empty_in_dict", "invalid_type", "n_bins_and_bin_size"],
)
def test_rdf_figures_invalid_input(
    rdf_func: Callable[..., RdfPlotWidget],
    structs: Any,
    kwargs: dict[str, Any],
    err_cls: type[Exception],
    match: str,
) -> None:
    """Both RDF figure functions reject empty/invalid structures and bin conflicts."""
    with pytest.raises(err_cls, match=match):
        rdf_func(structs, **kwargs)


def test_element_pair_rdfs_invalid_elements() -> None:
    """Element pairs must only contain elements present in the structures."""
    err_msg = "Elements .* in element_pairs not present in any structure"
    with pytest.raises(ValueError, match=err_msg):
        element_pair_rdfs(NACL, element_pairs=[("Zn", "Zn")])


@pytest.mark.parametrize(
    ("element_pairs", "expected_labels"),
    [(None, ["Cl-Cl", "Cl-Na", "Na-Na"]), ([("Na", "Na"), ("Cl", "Na")], None)],
)
def test_element_pair_rdfs_entries(
    element_pairs: list[tuple[str, str]] | None, expected_labels: list[str] | None
) -> None:
    """One pre-computed curve per sorted element pair, matching calculate_rdf."""
    widget = element_pair_rdfs(NACL, cutoff=6, n_bins=30, element_pairs=element_pairs)
    assert isinstance(widget, RdfPlotWidget)
    assert widget.structures is None
    assert widget.mode == "full"  # keeps all curves visible
    expected_labels = expected_labels or ["Cl-Na", "Na-Na"]
    assert [entry["label"] for entry in widget.patterns] == expected_labels
    for entry in widget.patterns:
        assert "legend_group" not in entry  # single structure
        el1, el2 = entry["pattern"]["element_pair"]
        radii, g_r = calculate_rdf(NACL, el1, el2, cutoff=6, n_bins=30)
        np.testing.assert_array_equal(entry["pattern"]["r"], radii)
        np.testing.assert_array_equal(entry["pattern"]["g_r"], g_r)


@pytest.mark.parametrize(
    ("kwargs", "expected_cutoff", "expected_bins"),
    [
        ({}, 2 * 5.64, 75),  # default: twice the longest lattice vector (<= 15 Å)
        ({"cutoff": 5}, 5, 75),
        ({"cutoff": -1.5}, 1.5 * 5.64, 75),
        ({"cutoff": 6, "bin_size": 0.2}, 6, 30),
    ],
)
def test_element_pair_rdfs_cutoff_and_bins(
    kwargs: dict[str, Any], expected_cutoff: float, expected_bins: int
) -> None:
    """Cutoff defaults to 2x the cell, negative cutoffs scale it, bin_size sets bins."""
    widget = element_pair_rdfs(NACL, **kwargs)
    for entry in widget.patterns:
        radii = entry["pattern"]["r"]
        assert len(radii) == expected_bins
        assert radii[-1] == pytest.approx(expected_cutoff, rel=1e-12)


@pytest.mark.parametrize("structs_type", ["dict", "list"])
def test_element_pair_rdfs_multiple_structures(structs_type: str) -> None:
    """Each structure's curves share a legend group, colors cycle per curve."""
    structs = SI_STRUCTS[:2]
    structures = (
        dict(zip(["first", "second"], structs, strict=True))
        if structs_type == "dict"
        else structs
    )
    colors = ["red", "blue", "green"]
    widget = element_pair_rdfs(structures, colors=colors)
    groups = [entry["legend_group"] for entry in widget.patterns]
    labels = list(structures) if structs_type == "dict" else sorted(set(groups))
    # Si2 has only the Si-Si pair, Pr2Si2Ru2 all 6 pairs of its 3 elements
    assert groups == [labels[0]] + [labels[1]] * 6
    assert [entry["color"] for entry in widget.patterns] == [
        colors[idx % 3] for idx in range(7)
    ]


@pytest.mark.parametrize("structs_type", ["single", "dict", "list", "ase"])
def test_full_rdf(structs_type: str) -> None:
    """One full RDF per structure, labeled by key, matching calculate_rdf."""
    structures = {
        "single": SI_STRUCTS[0],
        "dict": dict(zip(["a", "b"], SI_STRUCTS[:2], strict=True)),
        "list": SI_STRUCTS[:2],
        "ase": SI_ATOMS[:2],
    }[structs_type]
    widget = full_rdf(structures, cutoff=5, n_bins=20, colors=["red"])
    n_structs = 1 if structs_type == "single" else 2
    assert len(widget.patterns) == n_structs
    if structs_type == "dict":
        assert [entry["label"] for entry in widget.patterns] == ["a", "b"]
    radii, g_r = calculate_rdf(SI_STRUCTS[0], cutoff=5, n_bins=20)
    # ASE Atoms round-trip perturbs coordinates by ~1e-15 Å, rarely flipping a bin
    np.testing.assert_allclose(widget.patterns[0]["pattern"]["r"], radii, rtol=1e-12)
    if structs_type != "ase":
        np.testing.assert_array_equal(widget.patterns[0]["pattern"]["g_r"], g_r)
    assert {entry["color"] for entry in widget.patterns} == {"red"}


@pytest.mark.parametrize("rdf_func", [element_pair_rdfs, full_rdf])
def test_rdf_widget_kwargs(rdf_func: Callable[..., RdfPlotWidget]) -> None:
    """Extra kwargs reach the widget and may override the default mode."""
    widget = rdf_func(NACL, show_reference_line=False, style="height: 9em")
    assert (widget.show_reference_line, widget.style) == (False, "height: 9em")
    assert rdf_func(NACL, mode="element_pairs").mode == "element_pairs"


def test_element_pair_rdfs_skips_absent_pairs() -> None:
    """Pairs a structure lacks get no all-zero curve, and pairs are deduplicated."""
    kbr = Structure(Lattice.cubic(6.6), ["K", "Br"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    widget = element_pair_rdfs({"NaCl": NACL, "KBr": kbr}, cutoff=5)
    assert [(entry["legend_group"], entry["label"]) for entry in widget.patterns] == [
        ("NaCl", "Cl-Cl"),
        ("NaCl", "Cl-Na"),
        ("NaCl", "Na-Na"),
        ("KBr", "Br-Br"),
        ("KBr", "Br-K"),
        ("KBr", "K-K"),
    ]
    widget = element_pair_rdfs(NACL, element_pairs=[("Na", "Cl"), ("Cl", "Na")])
    assert [entry["label"] for entry in widget.patterns] == ["Cl-Na"]

    with pytest.raises(ValueError, match="No structure contains any of the element"):
        element_pair_rdfs({"NaCl": NACL, "KBr": kbr}, element_pairs=[("Na", "K")])
