"""Testing related utils."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt

from pymatviz.utils import ROOT


TEST_FILES: str = f"{ROOT}/tests/files"


def interactive_check(
    plot: plt.Figure | plt.Axes,
    /,
    *,
    elem_to_check: str = "figure",
) -> bool | None:
    """Ask a human to visually inspect a matplotlib Figure or Axes.

    Args:
        plot (plt.Figure | plt.Axes): Plot to inspect (either a Figure or Axes).
        elem_to_check (str): Prompt for what element in the figure to check.

    Returns:
        bool: whether this passes inspection.
        None: if running in CI.
    """
    # Skip CI runs
    if os.getenv("CI"):
        return None

    # Handle matplotlib Axes
    if isinstance(plot, plt.Axes):
        plot = plot.figure
    elif not isinstance(plot, plt.Figure):
        raise TypeError(f"plot {type(plot).__name__} not supported.")

    plot.show()

    # TODO: simplify prompt strategy, asking user to give feedback might be tedious?
    return (
        input(f"Please check the {elem_to_check}. Is this plot good? (y/n): ")
        .strip()
        .lower()
    ) == "y"
