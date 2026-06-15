"""Reactively link a shared selection index across MatterViz widgets.

This is the Python-side answer to building ``structure+scatter``-style coordinated
views by composition instead of bundling the link at the JS level (see
https://github.com/janosh/pymatviz/issues/354). It builds on the interaction-state
traitlets that sync both ways with the frontend (``TrajectoryWidget``'s
``current_step_idx``, ``ScatterPlotWidget``'s ``active_point``/``selected_point``,
``StructureWidget``'s ``structure``): observing one widget's selection and writing
it to the others keeps a shared "current index" in sync.

Example:
    >>> import pymatviz as pmv
    >>> scatter = pmv.ScatterPlotWidget(series=[{"x": steps, "y": energies}])
    >>> structure = pmv.StructureWidget(structure=frames[0])
    >>> link = pmv.widgets.link_selection(scatter, structure, structures=frames)
    >>> # clicking a scatter point now shows the matching frame in `structure`
    >>> link.unlink()  # stop syncing
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from pymatviz.widgets.structure import StructureWidget, structure_to_dict
from pymatviz.widgets.trajectory import TrajectoryWidget


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatviz.widgets.matterviz import MatterVizWidget


_LINKABLE = (TrajectoryWidget, ScatterPlotWidget, StructureWidget)


class WidgetLink:
    """Keep a shared selection index in sync across linked MatterViz widgets.

    Each supported widget contributes a source (interaction it reports), a sink
    (how it reflects the shared index), or both:

    - ``TrajectoryWidget``: source + sink via ``current_step_idx``.
    - ``ScatterPlotWidget``: source via ``active_point`` (clicked point index),
      sink via ``selected_point`` (highlight).
    - ``StructureWidget``: sink only -- its ``structure`` is set to
      ``structures[index]`` (requires ``structures``).

    Re-entrancy is guarded so a sink update never echoes back into a new
    broadcast, and traitlets' own no-op-on-unchanged behavior prevents loops.
    """

    def __init__(
        self,
        widgets: Sequence[MatterVizWidget],
        *,
        structures: Sequence[Any] | None = None,
        series_idx: int = 0,
    ) -> None:
        """Link the given widgets on a shared selection index.

        Args:
            widgets: Widgets to keep in sync (need at least two).
            structures: Ordered structures indexed by the shared selection index.
                Required if any widget is a ``StructureWidget`` (its ``structure``
                is set to ``structures[index]`` on every change).
            series_idx: Series index written into ``ScatterPlotWidget`` sinks
                (``selected_point``). Defaults to 0 (the first series).

        Raises:
            ValueError: If fewer than two widgets are given, or a
                ``StructureWidget`` is included without ``structures``.
        """
        if len(widgets) < 2:
            raise ValueError(
                f"link_selection needs at least 2 widgets, got {len(widgets)}"
            )

        unsupported = [
            widget for widget in widgets if not isinstance(widget, _LINKABLE)
        ]
        if unsupported:
            kinds = ", ".join(sorted({type(widget).__name__ for widget in unsupported}))
            supported = ", ".join(cls.__name__ for cls in _LINKABLE)
            raise TypeError(
                f"link_selection got unsupported widget type(s): {kinds}. "
                f"Supported: {supported}."
            )

        self.widgets = list(widgets)
        self._structures = None if structures is None else list(structures)
        self._series_idx = series_idx
        self._syncing = False
        self._observers: list[tuple[Any, Any, str]] = []

        # structures back a StructureWidget sink; an empty/None list can't (every
        # index is out of range), so fail early rather than render a stale frame.
        if not self._structures and any(
            isinstance(widget, StructureWidget) for widget in self.widgets
        ):
            raise ValueError(
                "link_selection requires non-empty `structures` when a "
                "StructureWidget is linked (its structure is set to structures[index])."
            )

        for widget in self.widgets:
            if isinstance(widget, TrajectoryWidget):
                self._add_observer(widget, self._on_trajectory, "current_step_idx")
            elif isinstance(widget, ScatterPlotWidget):
                self._add_observer(widget, self._on_scatter, "active_point")

        # initial sync so all sinks reflect the starting index (a trajectory's
        # current step if present, else 0)
        initial = next(
            (
                int(widget.current_step_idx)
                for widget in self.widgets
                if isinstance(widget, TrajectoryWidget)
            ),
            0,
        )
        self._broadcast(initial)

    def _add_observer(self, widget: Any, handler: Any, trait_name: str) -> None:
        """Register a traitlets observer and remember it for ``unlink``."""
        widget.observe(handler, names=trait_name)
        self._observers.append((widget, handler, trait_name))

    def _on_trajectory(self, change: dict[str, Any]) -> None:
        """Broadcast a trajectory step change to the other widgets."""
        self._broadcast(int(change["new"]))

    def _on_scatter(self, change: dict[str, Any]) -> None:
        """Broadcast a scatter point click to the other widgets.

        Only ``point_idx`` drives selection; ``active_point``'s ``event_id`` (which
        makes repeated identical clicks distinct trait values) is ignored here.
        """
        point = change["new"]
        if not point or point.get("point_idx") is None:
            return
        self._broadcast(int(point["point_idx"]))

    def _broadcast(self, index: int) -> None:
        """Push ``index`` to every linked widget's sink (re-entrancy guarded).

        Applies to all widgets including the interaction source: a scatter source
        reports via ``active_point`` but its sink is ``selected_point`` (so the
        clicked point must be highlighted here too), and re-applying a trajectory
        source's own ``current_step_idx`` is a traitlets no-op. The ``_syncing``
        guard absorbs any resulting echo notification.
        """
        if self._syncing:
            return
        self._syncing = True
        try:
            for widget in self.widgets:
                self._apply(widget, index)
        finally:
            self._syncing = False

    def _apply(self, widget: Any, index: int) -> None:
        """Reflect the shared ``index`` in a single widget (its sink)."""
        if isinstance(widget, TrajectoryWidget):
            widget.current_step_idx = index
        elif isinstance(widget, ScatterPlotWidget):
            widget.selected_point = {"series_idx": self._series_idx, "point_idx": index}
        elif isinstance(widget, StructureWidget) and self._structures is not None:
            if 0 <= index < len(self._structures):
                widget.structure = structure_to_dict(self._structures[index])
            else:
                warnings.warn(
                    f"link_selection: {index=} out of range for {len(self._structures)}"
                    " structures; StructureWidget not updated.",
                    stacklevel=2,
                )

    def unlink(self) -> None:
        """Remove all observers, stopping further synchronization."""
        for widget, handler, trait_name in self._observers:
            widget.unobserve(handler, names=trait_name)
        self._observers.clear()


def link_selection(
    *widgets: MatterVizWidget,
    structures: Sequence[Any] | None = None,
    series_idx: int = 0,
) -> WidgetLink:
    """Two-way link a shared selection index across MatterViz widgets.

    Convenience wrapper around :class:`WidgetLink`. Clicking a scatter point or
    stepping a trajectory updates every other linked widget; a linked
    ``StructureWidget`` shows ``structures[index]``.

    Args:
        *widgets: Widgets to link (need at least two). Supported:
            ``TrajectoryWidget``, ``ScatterPlotWidget``, ``StructureWidget``.
        structures: Ordered structures indexed by the selection index. Required
            if a ``StructureWidget`` is linked.
        series_idx: Series index used when highlighting scatter points.

    Returns:
        A :class:`WidgetLink` whose ``unlink()`` stops synchronization.
    """
    return WidgetLink(widgets, structures=structures, series_idx=series_idx)
