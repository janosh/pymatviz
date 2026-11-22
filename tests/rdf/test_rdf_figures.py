from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest

import pymatviz.rdf.figures as rdf_figures
from pymatviz.rdf.figures import rdf_plot
from pymatviz.rdf.helpers import calculate_rdf
from pymatviz.structures import structure_dict
from pymatviz.widgets import RdfPlotWidget
from tests.conftest import SI_STRUCTS, cubic_structure


EMPTY_STRUCT = cubic_structure(1, [], [])
NACL = cubic_structure(5.64, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
KBR = cubic_structure(6.6, ["K", "Br"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def python_rdfs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compute every RDF in Python, as for structures too large for the browser."""
    monkeypatch.setattr(rdf_figures, "MAX_BROWSER_RDF_SITES", 0)


@pytest.mark.parametrize(
    ("structs", "kwargs", "err_cls", "match"),
    [
        (EMPTY_STRUCT, {}, ValueError, r"structures \['.*'\] contain no sites"),
        ({"blank": EMPTY_STRUCT}, {}, ValueError, r"\['blank'\] contain no sites"),
        ("not a structure", {}, FileNotFoundError, "no structure file at"),
        (None, {}, ValueError, "pass structures or patterns"),
        (NACL, {"patterns": []}, ValueError, "pass structures or patterns"),
        (NACL, {"element_pairs": [("Zn", "Zn")]}, ValueError, "not in any structure"),
        (
            pd.Series([NACL, NACL], index=[1, "1"]),
            {},
            ValueError,
            "labels must be unique as strings",
        ),
    ],
    ids=[
        *("empty", "empty_in_dict", "invalid_type", "neither", "both", "bad_pair"),
        "colliding_labels",
    ],
)
def test_rdf_plot_invalid_input(
    structs: Any, kwargs: dict[str, Any], err_cls: type[Exception], match: str
) -> None:
    """Reject empty/invalid structures, structures plus patterns, absent elements and
    labels equal as strings (1 and "1") instead of dropping a structure.
    """
    with pytest.raises(err_cls, match=match):
        rdf_plot(structs, **kwargs)


@pytest.mark.parametrize("mode", ["element_pairs", "full"])
def test_rdf_plot_in_browser(mode: Literal["element_pairs", "full"]) -> None:
    """Small structures go to the browser with the default cutoff (twice the longest
    lattice vector, at most 15 Å) and bins, which stay adjustable there.
    """
    widget = rdf_plot({"a": NACL, "b": SI_STRUCTS[0]}, mode=mode, bins=40)
    assert widget.patterns is None
    assert list(widget.structures) == ["a", "b"]
    assert widget.structures["a"] == structure_dict(NACL)
    assert (widget.mode, widget.n_bins) == (mode, 40)
    assert widget.cutoff == pytest.approx(2 * 5.64)  # NaCl's cell is the longest
    lone = rdf_plot(NACL, cutoff=4)  # a single structure is sent unlabeled
    assert lone.cutoff == 4
    assert lone.structures == structure_dict(NACL)
    # the widget itself takes structure objects alone, in lists or labeled
    for structs, expected in (
        ([NACL], [structure_dict(NACL)]),
        ({1: NACL}, {"1": structure_dict(NACL)}),
        (NACL, structure_dict(NACL)),
    ):
        assert RdfPlotWidget(structures=structs).structures == expected


def test_rdf_plot_patterns() -> None:
    """Precomputed patterns pass through."""
    patterns = [{"label": "x", "pattern": {"r": [1, 2], "g_r": [0, 1]}}]
    assert rdf_plot(patterns=patterns).patterns == patterns


@pytest.mark.parametrize(
    ("element_pairs", "expected_labels"),
    [(None, ["Cl-Cl", "Cl-Na", "Na-Na"]), ([("Na", "Na"), ("Cl", "Na")], None)],
)
@pytest.mark.usefixtures("python_rdfs")
def test_rdf_plot_python_pairs(
    element_pairs: list[tuple[str, str]] | None, expected_labels: list[str] | None
) -> None:
    """One pre-computed curve per sorted element pair, matching calculate_rdf."""
    widget = rdf_plot(NACL, cutoff=6, bins=30, element_pairs=element_pairs)
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


@pytest.mark.parametrize("cutoff", [4.0, 9.0])
@pytest.mark.usefixtures("python_rdfs")
def test_rdf_plot_one_search_matches_calculate_rdf(
    cutoff: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All pairs (incl. a disordered Fe:Co site) come from one neighbor search and
    match per-pair calculate_rdf bit for bit. Absent species skip the search.
    """
    import pymatviz.rdf.helpers as rdf_helpers

    rng = np.random.default_rng(seed=0)
    species: list[Any] = ["Si", "O", "Fe", "Li"] * 6 + [{"Fe": 0.5, "Co": 0.5}]
    struct = cubic_structure(6.0, species, rng.uniform(size=(25, 3)))
    n_searches = 0
    to_structure = rdf_helpers.to_structure

    class CountingStructure:
        """kemio Structure proxy counting neighbor searches."""

        def __init__(self, struct: Any) -> None:
            self._struct = struct

        def __len__(self) -> int:
            return len(self._struct)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._struct, name)

        def neighbor_list(self, cutoff: float) -> Any:
            nonlocal n_searches
            n_searches += 1
            return self._struct.neighbor_list(cutoff)

    monkeypatch.setattr(
        rdf_helpers,
        "to_structure",
        lambda obj: CountingStructure(to_structure(obj)),
    )
    widget = rdf_plot(struct, cutoff=cutoff, bins=40)
    assert n_searches == 1
    assert len(widget.patterns) == 15  # 5 elements -> 15 pairs
    for entry in widget.patterns:
        el1, el2 = entry["pattern"]["element_pair"]
        radii, g_r = calculate_rdf(struct, el1, el2, cutoff=cutoff, n_bins=40)
        np.testing.assert_array_equal(entry["pattern"]["r"], radii)
        np.testing.assert_array_equal(entry["pattern"]["g_r"], g_r)
    assert rdf_helpers.partial_rdfs(struct, [], cutoff=cutoff) == []
    ((_, absent_g_r),) = rdf_helpers.partial_rdfs(struct, [("Zn", "Si")], cutoff=cutoff)
    assert not absent_g_r.any()
    assert n_searches == 1 + len(widget.patterns)  # one per calculate_rdf call


@pytest.mark.parametrize("structs_type", ["dict", "list"])
def test_rdf_plot_multiple_structures(structs_type: str) -> None:
    """Each structure's curves share a legend group, colors cycle per curve."""
    structs = SI_STRUCTS[:2]
    structures = (
        dict(zip(["first", "second"], structs, strict=True))
        if structs_type == "dict"
        else structs
    )
    colors = ["red", "blue", "green"]
    widget = rdf_plot(structures, colors=colors)  # colors: computed in Python
    groups = [entry["legend_group"] for entry in widget.patterns]
    labels = list(structures) if structs_type == "dict" else sorted(set(groups))
    # Si2 has only the Si-Si pair, Pr2Si2Ru2 all 6 pairs of its 3 elements
    assert groups == [labels[0]] + [labels[1]] * 6
    assert [entry["color"] for entry in widget.patterns] == [
        colors[idx % 3] for idx in range(7)
    ]


@pytest.mark.parametrize("structs_type", ["single", "dict", "list"])
def test_rdf_plot_full_in_python(structs_type: str) -> None:
    """One full RDF per structure, labeled by key, matching calculate_rdf."""
    structures = {
        "single": SI_STRUCTS[0],
        "dict": dict(zip(["a", "b"], SI_STRUCTS[:2], strict=True)),
        "list": SI_STRUCTS[:2],
    }[structs_type]
    widget = rdf_plot(structures, mode="full", cutoff=5, bins=20, colors=["red"])
    n_structs = 1 if structs_type == "single" else 2
    assert len(widget.patterns) == n_structs
    if structs_type == "dict":
        assert [entry["label"] for entry in widget.patterns] == ["a", "b"]
    radii, g_r = calculate_rdf(SI_STRUCTS[0], cutoff=5, n_bins=20)
    np.testing.assert_array_equal(widget.patterns[0]["pattern"]["r"], radii)
    np.testing.assert_array_equal(widget.patterns[0]["pattern"]["g_r"], g_r)
    assert {entry["color"] for entry in widget.patterns} == {"red"}


@pytest.mark.parametrize("element_pairs", [None, [("Na", "Cl")]])
def test_rdf_plot_widget_kwargs(element_pairs: list[tuple[str, str]] | None) -> None:
    """Extra kwargs reach the widget, in the browser and Python paths."""
    widget = rdf_plot(
        NACL,
        element_pairs=element_pairs,
        show_reference_line=False,
        style="height: 9em",
    )
    assert (widget.show_reference_line, widget.style) == (False, "height: 9em")


@pytest.mark.usefixtures("python_rdfs")
def test_rdf_plot_skips_absent_pairs() -> None:
    """Pairs a structure lacks get no all-zero curve, and pairs are deduplicated."""
    widget = rdf_plot({"NaCl": NACL, "KBr": KBR}, cutoff=5)
    assert [(entry["legend_group"], entry["label"]) for entry in widget.patterns] == [
        ("NaCl", "Cl-Cl"),
        ("NaCl", "Cl-Na"),
        ("NaCl", "Na-Na"),
        ("KBr", "Br-Br"),
        ("KBr", "Br-K"),
        ("KBr", "K-K"),
    ]
    widget = rdf_plot(NACL, element_pairs=[("Na", "Cl"), ("Cl", "Na")])
    assert [entry["label"] for entry in widget.patterns] == ["Cl-Na"]
    with pytest.raises(ValueError, match="No structure contains any of the element"):
        rdf_plot({"NaCl": NACL, "KBr": KBR}, element_pairs=[("Na", "K")])
