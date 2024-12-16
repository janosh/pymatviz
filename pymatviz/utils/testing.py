"""Testing related utils."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pytest

from pymatviz.utils import ROOT


TEST_FILES: str = f"{ROOT}/tests/files"


def interactive_check(
    plot: plt.Figure | plt.Axes,
    /,
    *,
    elem_to_check: str = "figure",
) -> bool | None:
    """Ask a human to visually inspect a plot.

    Note:
        You would need to pass `-s` to `pytest` to release output.

    Todo:
        - `pytest` would capture output by default unless `pytest -s`, possible to
            modify it within?

    Args:
        plot (plt.Figure | plt.Axes): Plot to inspect.
        elem_to_check (str): Prompt for what element in the plot to check.

    Returns:
        bool: whether this passes inspection (KeyboardInterrupt would
            be treated as a quick "y").
        None: if running in CI.
    """
    # Skip CI runs
    if os.getenv("CI"):
        return None

    # Handle matplotlib Axes
    if isinstance(plot, plt.Axes):
        plot = plot.figure
    elif not isinstance(plot, plt.Figure):
        raise TypeError(f"plot type {type(plot).__name__} is not supported.")

    # TODO: scale the figure by screen resolution
    # TODO: display at the top middle of the screen
    plt.show(block=False)

    # print()  # TODO: print the fully qualified name of the test

    # KeyboardInterrupt (ctrl + C) would be treated as a quick "y"
    try:
        goodenough: bool = (
            input(f"Please check the {elem_to_check}. Is it looking good? (y/n): ")
            .strip()
            .lower()
        ) == "y"

    except KeyboardInterrupt:
        goodenough = True

    finally:
        plt.close()

    if not goodenough:
        pytest.fail(reason=f"{elem_to_check} would need attention.")

    return goodenough
