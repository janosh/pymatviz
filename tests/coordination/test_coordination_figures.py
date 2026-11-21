from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import traitlets as tl
from pymatgen.analysis.local_env import (
    CrystalNN,
    CutOffDictNN,
    NearNeighbors,
    VoronoiNN,
)
from pymatgen.core import Lattice, Structure

from pymatviz.coordination import coordination_bar_plot, coordination_vs_cutoff_line
from pymatviz.coordination.figures import _element_colors
from pymatviz.coordination.helpers import CnSplitMode, coordination_nums_in_structure
from pymatviz.enums import ElemColorScheme
from pymatviz.widgets.bar_plot import BarPlotWidget
from pymatviz.widgets.coordination_bar_plot import CoordinationBarPlotWidget
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture
def nacl() -> Structure:
    """Conventional rocksalt NaCl: every site octahedral (CN 6 within 3 Å)."""
    return Structure.from_spacegroup(
        225, Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )


@pytest.mark.parametrize("split_mode", list(CnSplitMode))
def test_coordination_bar_plot_split_modes(
    structures: Sequence[Structure], split_mode: CnSplitMode
) -> None:
    """Series per split mode partition the same site CNs, on a contiguous CN axis."""
    struct_dict = {f"s{idx}": struct for idx, struct in enumerate(structures)}
    widget = coordination_bar_plot(struct_dict, strategy=3.0, split_mode=split_mode)
    assert isinstance(widget, BarPlotWidget)

    site_cns = {
        key: coordination_nums_in_structure(struct, strategy=3.0)
        for key, struct in struct_dict.items()
    }
    if split_mode == CnSplitMode.by_element:
        expected = {}
        for elem_cns in site_cns.values():
            for elem, cns in elem_cns.items():
                expected.setdefault(elem, Counter()).update(cns)
        expected = dict(sorted(expected.items()))
    elif split_mode == CnSplitMode.by_structure:
        expected = {
            key: Counter(cn for cns in elem_cns.values() for cn in cns)
            for key, elem_cns in site_cns.items()
        }
    else:
        expected = {
            "All sites": Counter(
                cn
                for elem_cns in site_cns.values()
                for cns in elem_cns.values()
                for cn in cns
            )
        }
    assert [srs["label"] for srs in widget.series] == list(expected)
    x_vals = widget.series[0]["x"]
    assert x_vals == list(range(int(x_vals[0]), int(x_vals[-1]) + 1))
    for srs, counts in zip(widget.series, expected.values(), strict=True):
        assert srs["x"] == x_vals
        assert dict(zip(x_vals, srs["y"], strict=True)) == {
            cn: counts.get(cn, 0) for cn in x_vals
        }
    total_sites = sum(len(struct) for struct in structures)
    assert sum(sum(srs["y"]) for srs in widget.series) == total_sites


@pytest.mark.parametrize(
    ("split_mode", "expected"),
    [
        ("by_element", {"Cl": [0, 4, 0], "Na": [0, 4, 0], "Si": [8, 0, 0]}),
        ("by_structure", {"NaCl": [0, 8, 0], "Si": [8, 0, 0]}),
        ("none", {"All sites": [8, 8, 0]}),
    ],
)
def test_coordination_bar_plot_literal_counts(
    nacl: Structure, split_mode: str, expected: dict[str, list[int]]
) -> None:
    """Rocksalt (CN 6) plus diamond Si (CN 4) at 3 Å, as hand-counted literals."""
    si = Structure.from_spacegroup(227, Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])
    widget = coordination_bar_plot(
        {"NaCl": nacl, "Si": si},
        strategy=3.0,
        split_mode=split_mode,
        x_axis={"range": [3, 7]},
    )
    assert isinstance(widget, BarPlotWidget)
    assert widget.series[0]["x"] == [4, 5, 6]
    assert {srs["label"]: srs["y"] for srs in widget.series} == {
        label: [counts[0], 0, counts[1]] for label, counts in expected.items()
    }
    # element colors for by_element, else uncolored so MatterViz cycles its palette
    has_colors = [("color" in srs) for srs in widget.series]
    assert all(has_colors) if split_mode == "by_element" else not any(has_colors)
    assert widget.mode == "stacked"
    assert widget.x_axis["range"] == [3, 7]


@pytest.mark.parametrize(
    "strategy", [3.0, 2.5, CrystalNN, CrystalNN(), VoronoiNN(tol=0.5)]
)
def test_coordination_bar_plot_strategies(nacl: Structure, strategy: Any) -> None:
    """Cutoffs and NearNeighbors are computed in Python; rocksalt is CN 6."""
    widget = coordination_bar_plot(nacl, strategy=strategy)
    assert isinstance(widget, BarPlotWidget)
    expected_cn = 0 if strategy == 2.5 else 6
    for srs in widget.series:
        assert dict(zip(srs["x"], srs["y"], strict=True)) == {expected_cn: 4}


def test_coordination_bar_plot_style_options(nacl: Structure) -> None:
    """Colors, bar labels, bar mode and axis options reach the BarPlotWidget."""
    widget = coordination_bar_plot(
        nacl,
        strategy=3.0,
        element_color_scheme={"Na": "red"},
        annotate_bars=True,
        mode="grouped",
        y_axis={"label": "Sites"},
        orientation="horizontal",
    )
    assert isinstance(widget, BarPlotWidget)
    cl_srs, na_srs = widget.series
    assert na_srs["color"] == "red"
    assert cl_srs["color"] == _element_colors(ElemColorScheme.jmol)["Cl"]
    assert na_srs["labels"] == ["Na"]
    assert widget.mode == "grouped"
    assert widget.orientation == "horizontal"
    # horizontal: CN categories on y (user label merged in), counts on x
    assert widget.y_axis == {"label": "Sites", "format": "d"}
    assert widget.x_axis == {"label": "Count", "format": "d", "range": [0, None]}


@pytest.mark.parametrize(
    ("structs", "kwargs", "err_cls", "match"),
    [
        ([], {}, ValueError, "Cannot plot empty set of structures"),
        ("invalid input", {}, TypeError, "Input must be a pymatgen Structure"),
        (None, {"strategy": "nope"}, tl.TraitError, "strategy"),
        (None, {"annotate_bars": True}, ValueError, "need a Python strategy"),
        (None, {"split_mode": "by_structure_and_element"}, ValueError, "not a valid"),
        (
            None,
            {"strategy": 3.0, "split_mode": "by_structure", "annotate_bars": True},
            ValueError,
            "annotate_bars needs",
        ),
        (None, {"strategy": 3, "element_color_scheme": "pastel"}, TypeError, "scheme"),
        (
            None,
            {"strategy": 3, "element_color_scheme": {"Na": (255, 0, 0)}},
            ValueError,
            "fractions",
        ),
    ],
)
def test_coordination_bar_plot_invalid_inputs(
    nacl: Structure,
    structs: Any,
    kwargs: dict[str, Any],
    err_cls: type[Exception],
    match: str,
) -> None:
    """coordination_bar_plot rejects invalid structures and options."""
    with pytest.raises(err_cls, match=match):
        coordination_bar_plot(nacl if structs is None else structs, **kwargs)


@pytest.mark.parametrize("split_mode", list(CnSplitMode))
def test_coordination_bar_plot_in_browser(
    nacl: Structure, split_mode: CnSplitMode
) -> None:
    """MatterViz strategy names bond in the browser, the default included."""
    widget = coordination_bar_plot({"NaCl": nacl}, split_mode=split_mode)
    assert isinstance(widget, CoordinationBarPlotWidget)
    assert (widget.strategy, widget.split_mode) == (
        "electroneg_ratio",
        split_mode.value,
    )
    assert list(widget.structures) == ["NaCl"]
    widget = coordination_bar_plot(nacl, strategy="explicit_only")
    assert isinstance(widget, CoordinationBarPlotWidget)
    assert widget.strategy == "explicit_only"


@pytest.mark.parametrize("scheme", [*ElemColorScheme, {"Na": "red", "Cl": "#00ff00"}])
def test_element_colors(scheme: ElemColorScheme | dict[str, str]) -> None:
    """Scheme RGB fractions become 0-255 CSS colors; custom colors pass through."""
    colors = _element_colors(scheme)
    if isinstance(scheme, dict):
        assert colors["Na"] == "red"
        assert colors["Cl"] == "#00ff00"
        assert colors["O"].startswith("rgb(")  # Jmol fills the rest
    else:
        for elem, color in scheme.color_map.items():
            expected = ", ".join(f"{255 * chan:.0f}" for chan in color)
            assert colors[elem] == f"rgb({expected})"


@pytest.mark.parametrize(
    ("strategy", "x_range"),
    [
        ((0, 3), (0, 3)),
        ((1.0, 5.0), (1, 5)),
        (CrystalNN, (0, 1)),
        (CrystalNN(distance_cutoffs=(0.5, 2)), (0, 2)),
    ],
)
def test_coordination_vs_cutoff_line(
    structures: Sequence[Structure],
    strategy: tuple[float, float] | NearNeighbors | type[NearNeighbors],
    x_range: tuple[float, float],
) -> None:
    """One line per structure and element, grouped and dashed per structure."""
    widget = coordination_vs_cutoff_line(
        structures[:2],
        strategy=strategy,
        num_points=7,
        element_color_scheme={"Si": "red"},
    )
    assert isinstance(widget, ScatterPlotWidget)
    elems = [
        sorted({site.specie.symbol for site in struct}) for struct in structures[:2]
    ]
    assert [srs["label"] for srs in widget.series] == [
        el for els in elems for el in els
    ]
    group_names = list(dict.fromkeys(srs["legend_group"] for srs in widget.series))
    assert [srs["legend_group"] for srs in widget.series] == [
        group_names[idx] for idx, els in enumerate(elems) for _ in els
    ]
    dashes = {srs["legend_group"]: srs["line_style"]["dash"] for srs in widget.series}
    assert len(set(dashes.values())) == 2
    for srs in widget.series:
        assert srs["markers"] == "line"
        assert (srs["x"][0], srs["x"][-1]) == pytest.approx(x_range)
        assert len(srs["x"]) == len(srs["y"]) == 7
        assert srs["y"] == sorted(srs["y"])  # CN grows with cutoff
        if srs["label"] == "Si":
            assert srs["line_style"]["color"] == "red"
    assert widget.y_axis["label"] == "Coordination Number"


def test_coordination_vs_cutoff_line_single_structure(nacl: Structure) -> None:
    """A single structure has no legend groups; rocksalt reaches CN 6 by 3 Å."""
    widget = coordination_vs_cutoff_line(nacl, strategy=(2, 3), num_points=3)
    assert all("legend_group" not in srs for srs in widget.series)
    assert [srs["y"] for srs in widget.series] == [[0, 0, 6], [0, 0, 6]]
    # descending cutoffs query neighbors out to the largest one
    widget = coordination_vs_cutoff_line(nacl, strategy=(3, 2), num_points=3)
    assert [srs["y"] for srs in widget.series] == [[6, 0, 0], [6, 0, 0]]


@pytest.mark.parametrize(
    ("structure", "strategy", "num_points"),
    [
        # periodic self-images at exactly the 2 and 4 A cutoffs
        (Structure(Lattice.cubic(2.0), ["Si"], [[0, 0, 0]]), (1, 5), 5),
        (
            Structure(
                Lattice.cubic(6.0),
                ["Si"] * 6 + ["O"] * 6,
                np.random.default_rng(seed=0).random((12, 3)),
            ),
            (0.5, 6),
            40,
        ),
    ],
)
def test_coordination_vs_cutoff_line_one_neighbor_search(
    structure: Structure,
    strategy: tuple[float, float],
    num_points: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One neighbor search for all sites gives the same mean CNs as one per site
    and cutoff.
    """
    cutoffs = np.linspace(*strategy, num_points)
    expected = {
        elem: [
            float(
                np.mean(
                    [
                        len(structure.get_neighbors(site, r=cutoff))
                        for site in structure
                        if site.specie.symbol == elem
                    ]
                )
            )
            for cutoff in cutoffs
        ]
        for elem in sorted({site.specie.symbol for site in structure})
    }
    n_calls = 0
    get_all_neighbors = Structure.get_all_neighbors

    def counting_get_all_neighbors(self: Structure, *args: Any, **kwargs: Any) -> Any:
        """Count calls to Structure.get_all_neighbors (get_neighbors calls it too)."""
        nonlocal n_calls
        n_calls += 1
        return get_all_neighbors(self, *args, **kwargs)

    monkeypatch.setattr(Structure, "get_all_neighbors", counting_get_all_neighbors)
    widget = coordination_vs_cutoff_line(
        structure, strategy=strategy, num_points=num_points
    )
    assert n_calls == 1
    assert {srs["label"]: srs["y"] for srs in widget.series} == expected


def test_coordination_vs_cutoff_line_disordered_structure() -> None:
    """A disordered Fe:C site contributes lines for both elements."""
    from pymatviz.utils.testing import fe3co4_disordered

    widget = coordination_vs_cutoff_line(fe3co4_disordered, strategy=(1, 5))
    assert {srs["label"] for srs in widget.series} == {"C", "Fe", "O"}
    assert all(min(srs["y"]) >= 0 for srs in widget.series)


@pytest.mark.parametrize(
    ("structs", "kwargs", "err_cls", "match"),
    [
        ([], {}, ValueError, "Cannot plot empty set of structures"),
        ("invalid input", {}, TypeError, "Input must be a pymatgen Structure"),
        (
            Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]]),
            {"strategy": "invalid"},
            TypeError,
            "Invalid strategy=",
        ),
        *(
            (
                Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]]),
                {"strategy": strategy},
                TypeError,
                "expected \\(min_cutoff, max_cutoff\\)",
            )
            for strategy in [(1, 2, 3), (True, 3), ("1", 3)]
        ),
        (
            Structure(Lattice.cubic(5), ["Si"], [[0, 0, 0]]),
            {"strategy": CutOffDictNN()},
            TypeError,
            "Could not determine cutoff",
        ),
    ],
    ids=[
        *("empty_list", "invalid_str", "invalid_strategy", "3_tuple", "bool", "str"),
        "nn_without_cutoff",
    ],
)
def test_coordination_vs_cutoff_line_invalid_inputs(
    structs: Any, kwargs: dict[str, Any], err_cls: type[Exception], match: str
) -> None:
    """coordination_vs_cutoff_line rejects invalid structures and strategies."""
    with pytest.raises(err_cls, match=match):
        coordination_vs_cutoff_line(structs, **kwargs)
