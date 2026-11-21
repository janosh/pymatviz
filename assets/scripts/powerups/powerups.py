"""Parity plot and ECDF powerups for MatterViz scatter widgets."""

# %%
from __future__ import annotations

import numpy as np

import pymatviz as pmv
from pymatviz import powerups


np_rng = np.random.default_rng(seed=42)
x1 = np.arange(100.0)
y1 = x1 * 0.8 + 5 + np_rng.normal(0, 8, len(x1))  # positive slope with noise
y2 = 100 - x1 * 0.7 + np_rng.normal(0, 10, len(x1))  # negative slope with noise
y3 = x1 + np_rng.normal(0, 20, len(x1))  # perfect correlation with high noise


# %% Parity plot: identity line, best-fit line and metrics
widget = pmv.ScatterPlotWidget(series=[{"x": x1, "y": y3, "markers": "points"}])
powerups.enhance_parity_plot(widget)
widget.show()


# %% Several series: combined metrics plus one best-fit line per series
widget = pmv.ScatterPlotWidget(
    series=[
        {"x": x1, "y": y_vals, "label": label, "color": color, "markers": "points"}
        for y_vals, label, color in (
            (y1, "Positive slope", "steelblue"),
            (y2, "Negative slope", "crimson"),
            (y3, "Noisy identity", "seagreen"),
        )
    ]
)
powerups.enhance_parity_plot(widget, best_fit_line=False)
for series_idx in range(3):
    powerups.add_best_fit_line(widget, series=series_idx)
widget.show()


# %% Custom metrics with a title and sample count
widget = pmv.ScatterPlotWidget(series=[{"x": x1, "y": y1, "markers": "points"}])
powerups.annotate_metrics(
    widget,
    metrics={"MAE": "MAE", "RMSE": "RMSE", "Max error": "max_error"},
    prefix="Model performance:\n",
    suffix=f"\nN = {len(x1)}",
    fmt=".4f",
)
widget.show()


# %% ECDF lines of two distributions on a right y2 axis
gauss1, gauss2 = np_rng.normal(5, 4, 500), np_rng.normal(10, 2, 500)
widget = pmv.ScatterPlotWidget(
    series=[
        {
            "x": np.sort(vals),
            "y": np.zeros(len(vals)),
            "label": label,
            "markers": "none",
        }
        for vals, label in ((gauss1, "Gaussian 1"), (gauss2, "Gaussian 2"))
    ]
)
powerups.add_ecdf_line(widget)
widget.show()
