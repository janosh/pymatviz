from os.path import abspath, dirname

import matplotlib.pyplot as plt
import numpy as np

ROOT: str = dirname(dirname(abspath(__file__)))


def add_identity(ax=None, **line_kwargs) -> None:
    """Add a parity line (y = x) to the provided axis."""
    if ax is None:
        ax = plt.gca()

    # zorder=0 ensures other plotted data displays on top of line
    default_kwargs = dict(alpha=0.5, zorder=0, linestyle="dashed", color="black")
    (identity,) = ax.plot([], [], **default_kwargs, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(ax)
    # Update identity line when moving the plot in interactive
    # viewing mode to always extend to the plot's edges.
    ax.callbacks.connect("xlim_changed", callback)
    ax.callbacks.connect("ylim_changed", callback)


def with_hist(xs, ys, cell=None, bins: int = 100):
    fig = plt.gcf()

    gs = (cell.subgridspec if cell else fig.add_gridspec)(
        2, 2, width_ratios=(6, 1), height_ratios=(1, 5), wspace=0, hspace=0
    )

    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    # x_hist
    ax_histx.hist(xs, bins=bins, rwidth=0.8)
    ax_histx.axis("off")

    # y_hist
    ax_histy.hist(ys, bins=bins, rwidth=0.8, orientation="horizontal")
    ax_histy.axis("off")

    return ax_main


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x)
    return e_x / e_x.sum(axis=-1, keepdims=True)


def one_hot(targets: np.ndarray, n_classes: int = 2) -> np.ndarray:
    return np.eye(n_classes)[targets]
