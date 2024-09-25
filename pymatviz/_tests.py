"""Unit test utils.

TODO: find a better home (AKA relocate this).
"""

import os

import matplotlib.pyplot as plt


def interactive_check(plot: plt.Figure) -> bool:
    """Ask a human to visually inspect a plot.

    Args:
        plot (plt.Figure): plot to inspect.
        TODO: handle other types (Axis...)

    Returns:
        bool: whether this passes inspection.
    """
    # Skip GitHub CI runs
    if os.getenv("GITHUB_ACTIONS") == "true":
        return True

    # placeholder
    assert isinstance(plot, plt.Figure)
