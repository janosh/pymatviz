"""Tests for widget interaction-state traits and Python-side reactive linking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from matscikit.interop import structure_to_pymatgen_dict

from pymatviz import link_selection
from pymatviz.widgets import (
    BarPlotWidget,
    PeriodicTableWidget,
    ScatterPlotWidget,
    StructureWidget,
    TrajectoryWidget,
)
from pymatviz.widgets.link import WidgetLink
from tests.conftest import cubic_structure


if TYPE_CHECKING:
    import matscikit as mk


def _frames(n_frames: int = 4) -> list[mk.Structure]:
    """Return n_frames distinct cubic structures (lattice grows per frame)."""
    return [
        cubic_structure(3.0 + 0.1 * idx, ["Fe"] * 2, [[0, 0, 0], [0.5] * 3])
        for idx in range(n_frames)
    ]


def _scatter(n_points: int = 4, **series: Any) -> ScatterPlotWidget:
    """Scatter widget with one series of n_points on the diagonal."""
    xs = list(range(n_points))
    return ScatterPlotWidget(series=[{"x": xs, "y": xs, **series}])


@pytest.mark.parametrize(
    ("widget_cls", "trait_name", "value", "default"),
    [
        (ScatterPlotWidget, "selected_point", {"series_idx": 0, "point_idx": 3}, None),
        (
            ScatterPlotWidget,
            "active_point",
            {"series_idx": 1, "point_idx": 2, "x": 1.0, "y": 2.0},
            None,
        ),
        (
            ScatterPlotWidget,
            "hovered_point",
            {"series_idx": 0, "point_idx": 0, "x": 0.0, "y": 0.0},
            None,
        ),
        (StructureWidget, "selected_sites", [1, 2], []),
        (StructureWidget, "highlighted_sites", [0], []),
        (StructureWidget, "hovered_site_idx", 3, None),
        (ScatterPlotWidget, "highlighted_ids", [0, 2], None),
        (PeriodicTableWidget, "active_elements", ["Fe", "O"], None),
    ],
)
def test_interaction_traits_sync(
    widget_cls: type[ScatterPlotWidget | StructureWidget | PeriodicTableWidget],
    trait_name: str,
    value: object,
    default: object,
) -> None:
    """Interaction traits default correctly, seed via init, round-trip and sync."""
    widget = widget_cls()
    assert getattr(widget, trait_name) == default
    setattr(widget, trait_name, value)
    assert getattr(widget, trait_name) == value
    assert widget.class_traits()[trait_name].metadata.get("sync") is True
    assert getattr(widget_cls(**{trait_name: value}), trait_name) == value


_NO_STRUCTS = "requires non-empty `structures`"


@pytest.mark.parametrize(
    ("sink_classes", "kwargs", "err_cls", "match"),
    [
        ([], {}, ValueError, "at least 2 widgets"),
        ([BarPlotWidget], {}, TypeError, r"unsupported widget type.*BarPlotWidget"),
        ([StructureWidget], {"structures": None}, ValueError, _NO_STRUCTS),
        ([StructureWidget], {"structures": []}, ValueError, _NO_STRUCTS),
        ([PeriodicTableWidget], {"structures": None}, ValueError, _NO_STRUCTS),
        ([PeriodicTableWidget], {"structures": []}, ValueError, _NO_STRUCTS),
    ],
)
def test_link_selection_rejects_invalid_args(
    sink_classes: list[type],
    kwargs: dict[str, Any],
    err_cls: type[Exception],
    match: str,
) -> None:
    """A lone widget, unsupported widget types and structure/ptable sinks without
    (non-empty) structures fail early.
    """
    sinks = [sink_cls() for sink_cls in sink_classes]
    with pytest.raises(err_cls, match=match):
        link_selection(_scatter(1), *sinks, **kwargs)


def test_link_scatter_click_updates_structure() -> None:
    """Scatter clicks show the matching frame in the structure view; observers see
    only real clicks (no synthetic None), hover-off (null active_point) and clicks
    after unlink() change nothing.
    """
    frames = _frames()
    scatter = _scatter()
    structure = StructureWidget(structure=frames[0])

    link = link_selection(scatter, structure, structures=frames)
    assert isinstance(link, WidgetLink)
    # initial sync shows frame 0
    assert structure.structure == structure_to_pymatgen_dict(frames[0])

    seen: list[dict[str, int] | None] = []
    scatter.observe(lambda change: seen.append(change["new"]), names="active_point")
    # simulate a frontend click writing back active_point
    scatter.active_point = {"series_idx": 0, "point_idx": 2, "event_id": 1}
    assert structure.structure == structure_to_pymatgen_dict(frames[2])
    assert seen == [{"series_idx": 0, "point_idx": 2, "event_id": 1}]
    assert scatter.active_point is not None  # trait keeps the click value

    scatter.active_point = None  # hover-off
    assert structure.structure == structure_to_pymatgen_dict(frames[2])
    link.unlink()
    scatter.active_point = {"series_idx": 0, "point_idx": 3}
    assert structure.structure == structure_to_pymatgen_dict(frames[2])


def test_link_scatter_click_highlights_source_scatter() -> None:
    """The clicked scatter highlights its own point too (source isn't skipped)."""
    scatter_a, scatter_b = _scatter(3), _scatter(3)

    link_selection(scatter_a, scatter_b, series_idx=0)
    scatter_a.active_point = {"series_idx": 0, "point_idx": 2}
    # both the clicked (source) and the other scatter get the persistent highlight
    assert scatter_a.selected_point == {"series_idx": 0, "point_idx": 2}
    assert scatter_b.selected_point == {"series_idx": 0, "point_idx": 2}


@pytest.mark.parametrize(
    ("link_kwargs", "series_idx"), [({}, 0), ({"series_idx": 2}, 2)]
)
def test_link_trajectory_step_drives_scatter_highlight(
    link_kwargs: dict[str, Any], series_idx: int
) -> None:
    """Stepping a trajectory highlights the matching scatter point in the configured
    series (default 0).
    """
    traj, scatter = TrajectoryWidget(trajectory=_frames()), _scatter()

    link_selection(traj, scatter, **link_kwargs)
    traj.current_step_idx = 3
    assert scatter.selected_point == {"series_idx": series_idx, "point_idx": 3}


def test_link_no_infinite_loop_between_two_trajectories() -> None:
    """Two linked trajectories converge without recursing."""
    frames = _frames()
    traj_a = TrajectoryWidget(trajectory=frames)
    traj_b = TrajectoryWidget(trajectory=frames)

    link_selection(traj_a, traj_b)
    traj_a.current_step_idx = 2
    assert traj_b.current_step_idx == 2
    traj_b.current_step_idx = 0
    assert traj_a.current_step_idx == 0


def test_link_repeated_scatter_click_resyncs() -> None:
    """Initial sync reflects the trajectory's starting step; re-clicking the same point
    re-syncs after the selection moved elsewhere.

    The frontend tags each click with a monotonic event_id, so repeated
    identical clicks are distinct trait values (no reset hack needed) and
    active_point keeps the last clicked point.
    """
    frames = _frames()
    traj = TrajectoryWidget(trajectory=frames, current_step_idx=3)
    structure = StructureWidget(structure=frames[0])
    scatter = _scatter()

    link_selection(traj, scatter, structure, structures=frames)
    assert structure.structure == structure_to_pymatgen_dict(frames[3])
    scatter.active_point = {"series_idx": 0, "point_idx": 1, "event_id": 1}
    assert traj.current_step_idx == 1
    assert scatter.active_point is not None  # not reset; retains the click

    traj.current_step_idx = 2  # move the shared selection elsewhere
    assert structure.structure == structure_to_pymatgen_dict(frames[2])

    # same point, new event_id (as the frontend would send) -> re-syncs
    scatter.active_point = {"series_idx": 0, "point_idx": 1, "event_id": 2}
    assert traj.current_step_idx == 1
    assert structure.structure == structure_to_pymatgen_dict(frames[1])


def test_link_out_of_range_index_warns_and_keeps_structure() -> None:
    """An out-of-range selection index warns and leaves the structure unchanged."""
    frames = _frames(3)
    structure = StructureWidget(structure=frames[0])
    # scatter has more points than there are structures
    scatter = _scatter(5)

    link_selection(scatter, structure, structures=frames)
    with pytest.warns(UserWarning, match=r"index=\d+ out of range for \d+ structures"):
        scatter.active_point = {"series_idx": 0, "point_idx": 4}  # beyond len(frames)
    # unchanged, no exception
    assert structure.structure == structure_to_pymatgen_dict(frames[0])


def test_link_periodic_table_highlights_and_filters_scatter() -> None:
    """Scatter clicks highlight the structure's elements in the periodic table,
    tile clicks dim scatter points whose structure lacks the clicked elements.
    """
    frames = [
        cubic_structure(3, species, [[0, 0, 0], [0.5] * 3])
        for species in (("Fe", "O"), ("Li", "O"), ("Li", "Li"))
    ]
    # without point_ids no point could dim, so linking refuses
    with pytest.raises(ValueError, match="point_ids"):
        link_selection(_scatter(3), PeriodicTableWidget(), structures=frames)
    scatter = _scatter(3, point_ids=[0, 1, 2])
    ptable = PeriodicTableWidget()

    # a structure dict takes the serialized path, Structures their composition
    link_selection(
        scatter, ptable, structures=[*frames[:2], structure_to_pymatgen_dict(frames[2])]
    )
    assert ptable.active_elements == ["Fe", "O"]  # initial sync to index 0
    # the frontend toggles clicks relative to the shown elements: clicking Li
    # writes [Fe, O, Li], which filters by Li alone, not by all three
    ptable.active_elements = ["Fe", "O", "Li"]
    assert (ptable.active_elements, scatter.highlighted_ids) == (["Li"], [1, 2])

    for clicked, expected_ids in (
        (["Li", "O"], [1]),
        (["O"], [0, 1]),
        ([], None),
        (None, None),
    ):
        ptable.active_elements = clicked  # as written by frontend tile clicks
        assert scatter.highlighted_ids == expected_ids

    scatter.active_point = {"series_idx": 0, "point_idx": 1}
    assert ptable.active_elements == ["Li", "O"]
    assert scatter.highlighted_ids is None  # index sync doesn't filter
    ptable.active_elements = ["Li"]  # un-clicking O after a broadcast filters by O
    assert (ptable.active_elements, scatter.highlighted_ids) == (["O"], [0, 1])

    # selecting a point after filtering shows that structure, so the filter clears
    scatter.active_point = {"series_idx": 0, "point_idx": 2}
    assert (ptable.active_elements, scatter.highlighted_ids) == (["Li"], None)
