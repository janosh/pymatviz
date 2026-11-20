"""Tests for widget interaction-state traits and Python-side reactive linking."""

from __future__ import annotations

from typing import Any

import pytest
from pymatgen.core import Lattice, Structure

from pymatviz import (
    PeriodicTableWidget,
    ScatterPlotWidget,
    StructureWidget,
    TrajectoryWidget,
    link_selection,
)
from pymatviz.widgets.link import WidgetLink


def _frames(n_frames: int = 4) -> list[Structure]:
    """Return n_frames distinct cubic structures (lattice grows per frame)."""
    return [
        Structure(Lattice.cubic(3.0 + 0.1 * idx), ("Fe", "Fe"), ((0, 0, 0), (0.5,) * 3))
        for idx in range(n_frames)
    ]


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
    """Interaction traits default correctly, round-trip, and sync to the frontend."""
    widget = widget_cls()
    assert getattr(widget, trait_name) == default
    setattr(widget, trait_name, value)
    assert getattr(widget, trait_name) == value
    assert widget.class_traits()[trait_name].metadata.get("sync") is True


def test_scatter_selected_point_via_constructor() -> None:
    """selected_point can be seeded at construction time."""
    point = {"series_idx": 0, "point_idx": 5}
    widget = ScatterPlotWidget(series=[{"x": [0], "y": [0]}], selected_point=point)
    assert widget.selected_point == point


def test_link_selection_requires_two_widgets() -> None:
    """A single widget cannot be linked."""
    scatter = ScatterPlotWidget(series=[{"x": [0], "y": [0]}])
    with pytest.raises(ValueError, match="at least 2 widgets"):
        link_selection(scatter)


@pytest.mark.parametrize("structures", [None, []])
@pytest.mark.parametrize("sink_cls", [StructureWidget, PeriodicTableWidget])
def test_link_selection_structure_requires_nonempty_structures(
    structures: list[Structure] | None,
    sink_cls: type[StructureWidget | PeriodicTableWidget],
) -> None:
    """Linking a structure/ptable sink without (non-empty) structures fails early."""
    scatter = ScatterPlotWidget(series=[{"x": [0], "y": [0]}])
    with pytest.raises(ValueError, match="requires non-empty `structures`"):
        link_selection(scatter, sink_cls(), structures=structures)


def test_link_selection_rejects_unsupported_widget() -> None:
    """Linking an unsupported widget type fails with a clear TypeError."""
    from pymatviz import BarPlotWidget

    scatter = ScatterPlotWidget(series=[{"x": [0], "y": [0]}])
    bar = BarPlotWidget(series=[{"x": [0], "y": [0]}])
    with pytest.raises(TypeError, match=r"unsupported widget type.*BarPlotWidget"):
        link_selection(scatter, bar)


def test_link_scatter_click_updates_structure() -> None:
    """Clicking a scatter point shows the matching frame in the structure view."""
    frames = _frames()
    scatter = ScatterPlotWidget(series=[{"x": list(range(4)), "y": list(range(4))}])
    structure = StructureWidget(structure=frames[0])

    link = link_selection(scatter, structure, structures=frames)
    assert isinstance(link, WidgetLink)
    # initial sync shows frame 0
    assert structure.structure == frames[0].as_dict()

    # simulate a frontend click writing back active_point
    scatter.active_point = {"series_idx": 0, "point_idx": 2}
    assert structure.structure == frames[2].as_dict()


def test_link_scatter_click_highlights_source_scatter() -> None:
    """The clicked scatter highlights its own point too (source isn't skipped)."""
    scatter_a = ScatterPlotWidget(series=[{"x": [0, 1, 2], "y": [0, 1, 2]}])
    scatter_b = ScatterPlotWidget(series=[{"x": [0, 1, 2], "y": [0, 1, 2]}])

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
    traj = TrajectoryWidget(trajectory=_frames())
    scatter = ScatterPlotWidget(series=[{"x": list(range(4)), "y": list(range(4))}])

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


def test_link_ignores_null_scatter_point() -> None:
    """A null active_point (hover-off) does not change linked widgets."""
    frames = _frames()
    scatter = ScatterPlotWidget(series=[{"x": list(range(4)), "y": list(range(4))}])
    structure = StructureWidget(structure=frames[0])

    link_selection(scatter, structure, structures=frames)
    scatter.active_point = {"series_idx": 0, "point_idx": 2}
    assert structure.structure == frames[2].as_dict()

    scatter.active_point = None  # hover-off
    assert structure.structure == frames[2].as_dict()  # unchanged


def test_unlink_stops_syncing() -> None:
    """After unlink, interactions no longer propagate."""
    frames = _frames()
    scatter = ScatterPlotWidget(series=[{"x": list(range(4)), "y": list(range(4))}])
    structure = StructureWidget(structure=frames[0])

    link = link_selection(scatter, structure, structures=frames)
    link.unlink()

    scatter.active_point = {"series_idx": 0, "point_idx": 3}
    assert structure.structure == frames[0].as_dict()  # unchanged after unlink


def test_link_repeated_scatter_click_resyncs() -> None:
    """Re-clicking the same point re-syncs after the selection moved elsewhere.

    The frontend tags each click with a monotonic event_id, so repeated
    identical clicks are distinct trait values (no reset hack needed) and
    active_point keeps the last clicked point.
    """
    frames = _frames()
    traj = TrajectoryWidget(trajectory=frames)
    structure = StructureWidget(structure=frames[0])
    scatter = ScatterPlotWidget(series=[{"x": list(range(4)), "y": list(range(4))}])

    link_selection(traj, scatter, structure, structures=frames)
    scatter.active_point = {"series_idx": 0, "point_idx": 1, "event_id": 1}
    assert traj.current_step_idx == 1
    assert scatter.active_point is not None  # not reset; retains the click

    traj.current_step_idx = 2  # move the shared selection elsewhere
    assert structure.structure == frames[2].as_dict()

    # same point, new event_id (as the frontend would send) -> re-syncs
    scatter.active_point = {"series_idx": 0, "point_idx": 1, "event_id": 2}
    assert traj.current_step_idx == 1
    assert structure.structure == frames[1].as_dict()


def test_link_does_not_emit_synthetic_active_point() -> None:
    """A user's own active_point observer sees only real clicks, no synthetic None."""
    frames = _frames()
    structure = StructureWidget(structure=frames[0])
    scatter = ScatterPlotWidget(series=[{"x": list(range(4)), "y": list(range(4))}])
    link_selection(scatter, structure, structures=frames)

    seen: list[dict[str, int] | None] = []
    scatter.observe(lambda change: seen.append(change["new"]), names="active_point")
    scatter.active_point = {"series_idx": 0, "point_idx": 2, "event_id": 1}

    assert seen == [{"series_idx": 0, "point_idx": 2, "event_id": 1}]
    assert scatter.active_point is not None  # trait keeps the click value


def test_link_out_of_range_index_warns_and_keeps_structure() -> None:
    """An out-of-range selection index warns and leaves the structure unchanged."""
    frames = _frames(3)
    structure = StructureWidget(structure=frames[0])
    # scatter has more points than there are structures
    scatter = ScatterPlotWidget(series=[{"x": list(range(5)), "y": list(range(5))}])

    link_selection(scatter, structure, structures=frames)
    with pytest.warns(UserWarning, match=r"index=\d+ out of range for \d+ structures"):
        scatter.active_point = {"series_idx": 0, "point_idx": 4}  # beyond len(frames)
    assert structure.structure == frames[0].as_dict()  # unchanged, no exception


def test_link_initial_sync_uses_trajectory_step() -> None:
    """Initial sync reflects a trajectory's starting step into other sinks."""
    frames = _frames()
    traj = TrajectoryWidget(trajectory=frames, current_step_idx=2)
    structure = StructureWidget(structure=frames[0])

    link_selection(traj, structure, structures=frames)
    assert structure.structure == frames[2].as_dict()


def test_link_periodic_table_highlights_and_filters_scatter() -> None:
    """Scatter clicks highlight the structure's elements in the periodic table,
    tile clicks dim scatter points whose structure lacks the clicked elements.
    """
    frames = [
        Structure(Lattice.cubic(3), species, ((0, 0, 0), (0.5,) * 3))
        for species in (("Fe", "O"), ("Li", "O"), ("Li", "Li"))
    ]
    xs = list(range(len(frames)))
    # without point_ids no point could dim, so linking refuses
    no_ids = ScatterPlotWidget(series=[{"x": xs, "y": xs}])
    with pytest.raises(ValueError, match="point_ids"):
        link_selection(no_ids, PeriodicTableWidget(), structures=frames)
    scatter = ScatterPlotWidget(series=[{"x": xs, "y": xs, "point_ids": xs}])
    ptable = PeriodicTableWidget()

    # a structure dict takes the serialized path, pymatgen ones their composition
    link_selection(scatter, ptable, structures=[*frames[:2], frames[2].as_dict()])
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
