"""Powerups such as parity lines, annotations, marginals, menu buttons, etc for
matplotlib and plotly figures.
"""

from pymatviz.powerups.both import (
    add_best_fit_line,
    add_identity_line,
    annotate_metrics,
)
from pymatviz.powerups.matplotlib import annotate_bars, with_marginal_hist
from pymatviz.powerups.plotly import (
    add_ecdf_line,
    select_colorscale,
    select_marker_mode,
    toggle_grid,
    toggle_log_linear_x_axis,
    toggle_log_linear_y_axis,
)
