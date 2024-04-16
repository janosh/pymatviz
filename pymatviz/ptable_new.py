"""Various periodic table heatmaps with matplotlib and plotly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from pymatgen.core import Element

from pymatviz.utils import df_ptable


if TYPE_CHECKING:
    from typing import Any, Callable


class PTableProjector:
    """Project (nest) a custom plot into a periodic table."""

    def __init__(
        self,
        colormap: str,
        **kwargs: Any,
    ) -> None:
        # Initialize a periodic table
        n_periods = df_ptable.row.max()
        n_groups = df_ptable.column.max()

        # Set figure size
        kwargs.setdefault("figsize", (0.75 * n_groups, 0.75 * n_periods))

        self.fig, self.axes = plt.subplots(n_periods, n_groups, **kwargs)

        # Get colormap
        self.cmap = plt.get_cmap(colormap)

    def set_style(self) -> None:
        """Set global styles."""

    def add_child_plots(self) -> None:
        """Add selected custom child plots."""

    def add_ele_symbols(
        self,
        symbol_text: str | Callable[[Element], str] = lambda elem: elem.symbol,
        symbol_pos: tuple[float, float] = (0.5, 0.5),
        symbol_kwargs: dict | None = None,
    ) -> None:
        """Add element symbols for each tile."""
        symbol_kwargs = symbol_kwargs or {}
        symbol_kwargs.setdefault("fontsize", 18)

        # Add symbol for each element
        for element in Element:
            symbol = element.symbol
            row, group = df_ptable.loc[symbol, ["row", "column"]]

            ax = self.axes[row - 1][group - 1]

            ax.text(
                *symbol_pos,
                symbol_text(element)
                if callable(symbol_text)
                else symbol_text.format(elem=element),
                ha="center",
                va="center",
                transform=ax.transAxes,
                **symbol_kwargs,
            )

    def add_colorbar(
        self,
        cbar_title: str,
        cbar_coords: tuple[float, float, float, float] = (0.18, 0.8, 0.42, 0.02),
        cbar_kwds: dict | None = None,
        title_kwds: dict | None = None,
    ) -> None:
        """Add a global colorbar."""
        # Update colorbar keyword args
        cbar_kwds = {"orientation": "horizontal"} | (cbar_kwds or {})

        cbar_ax = self.fig.add_axes(cbar_coords)

        self.fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=self.cmap),  # TODO:
            cax=cbar_ax,
            **cbar_kwds,
        )

        # Set colorbar title
        title_kwds = title_kwds or {}
        title_kwds.setdefault("fontsize", 12)
        title_kwds.setdefault("pad", 10)
        title_kwds["label"] = cbar_title

        cbar_ax.set_title(**title_kwds)
