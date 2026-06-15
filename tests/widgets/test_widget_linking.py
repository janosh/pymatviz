"""Tests for widget interaction-state traits and Python-side reactive linking."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from pymatviz import (
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
    ("trait_name", "value"),
    [
        ("selected_point", {"series_idx": 0, "point_idx": 3}),
        ("active_point", {"series_idx": 1, "point_idx": 2, "x": 1.0, "y": 2.0}),
        ("hovered_point", {"series_idx": 0, "point_idx": 0, "x": 0.0, "y": 0.0}),
    ],
)
def test_scatter_interaction_traits_sync(
    trait_name: str, value: dict[str, float]
) -> None:
    """Scatter interaction traits round-trip and are tagged for frontend sync."""
    widget = ScatterPlotWidget(series=[{"x": [0, 1], "y": [1, 2]}])
    assert getattr(widget, trait_name) is None  # default
    setattr(widget, trait_name, value)
    assert getattr(widget, trait_name) == value
    assert widget.class_traits()[trait_name].metadata.get("sync") is True


def test_scatter_selected_point_via_constructor() -> None:
    """selected_point can be seeded at construction time."""
    point = {"series_idx": 0, "point_idx": 5}
    widget = ScatterPlotWidget(series=[{"x": [0], "y": [0]}], selected_point=point)
    assert widget.selected_point == point


@pytest.mark.parametrize(
    ("trait_name", "value", "default"),
    [
        ("selected_sites", [1, 2], []),
        ("highlighted_sites", [0], []),
        ("hovered_site_idx", 3, None),
    ],
)
def test_structure_interaction_traits_sync(
    trait_name: str, value: object, default: object
) -> None:
    """Structure interaction traits round-trip and are tagged for frontend sync."""
    widget = StructureWidget()
    assert getattr(widget, trait_name) == default
    setattr(widget, trait_name, value)
    assert getattr(widget, trait_name) == value
    assert widget.class_traits()[trait_name].metadata.get("sync") is True


def test_link_selection_requires_two_widgets() -> None:
    """A single widget cannot be linked."""
    scatter = ScatterPlotWidget(series=[{"x": [0], "y": [0]}])
    with pytest.raises(ValueError, match="at least 2 widgets"):
        link_selection(scatter)


@pytest.mark.parametrize("structures", [None, []])
def test_link_selection_structure_requires_nonempty_structures(
    structures: list[Structure] | None,
) -> None:
    """Linking a StructureWidget without (non-empty) structures fails early."""
    scatter = ScatterPlotWidget(series=[{"x": [0], "y": [0]}])
    structure = StructureWidget()
    with pytest.raises(ValueError, match="requires non-empty `structures`"):
        link_selection(scatter, structure, structures=structures)


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


def test_link_trajectory_step_drives_scatter_highlight() -> None:
    """Stepping a trajectory highlights the matching scatter point."""
    frames = _frames()
    traj = TrajectoryWidget(trajectory=frames)
    scatter = ScatterPlotWidget(series=[{"x": list(range(4)), "y": list(range(4))}])

    link_selection(traj, scatter)
    traj.current_step_idx = 3
    assert scatter.selected_point == {"series_idx": 0, "point_idx": 3}


def test_link_scatter_click_drives_trajectory_step() -> None:
    """Clicking a scatter point steps a linked trajectory (two-way)."""
    frames = _frames()
    traj = TrajectoryWidget(trajectory=frames)
    scatter = ScatterPlotWidget(series=[{"x": list(range(4)), "y": list(range(4))}])

    link_selection(traj, scatter)
    scatter.active_point = {"series_idx": 0, "point_idx": 1}
    assert traj.current_step_idx == 1


def test_link_series_idx_used_for_highlight() -> None:
    """The configured series_idx is written into scatter selected_point."""
    frames = _frames()
    traj = TrajectoryWidget(trajectory=frames)
    scatter = ScatterPlotWidget(series=[{"x": [0, 1], "y": [0, 1]}])

    link_selection(traj, scatter, series_idx=2)
    traj.current_step_idx = 1
    assert scatter.selected_point == {"series_idx": 2, "point_idx": 1}


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

    scatter.active_point = None  # hover-off  # ty: ignore[invalid-assignment]
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


def test_link_ignores_event_id_for_selection() -> None:
    """Selection uses point_idx only; event_id does not affect which frame shows."""
    frames = _frames()
    structure = StructureWidget(structure=frames[0])
    scatter = ScatterPlotWidget(series=[{"x": list(range(4)), "y": list(range(4))}])
    link_selection(scatter, structure, structures=frames)

    scatter.active_point = {"series_idx": 0, "point_idx": 3, "event_id": 99}
    assert structure.structure == frames[3].as_dict()


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
