"""Density scatter examples."""

# %%
from __future__ import annotations

import numpy as np
import pandas as pd

import pymatviz as pmv


y_true, y_pred, _y_std = pmv.data.regression()


# %% Basic density scatter with regression data
widget = pmv.density_scatter(y_true, y_pred)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/density-scatter.svg", width=500, transparent=True
)


# %% Fine-grained density with marginal histograms
widget = pmv.density_hexbin(y_true, y_pred, bin_px=1.5, marginal_bins=100)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/density-scatter-with-hist.svg",
    width=500,
    transparent=True,
)


# %% ML model predictions: 20k points binned into 200x200 bins
np_rng = np.random.default_rng(seed=123)
true_values = np.linspace(0, 10, 20_000) + np_rng.normal(0, 0.5, 20_000)
df_ml = pd.DataFrame(
    {
        "True Values": true_values,
        "Predictions": true_values + np_rng.normal(0, 1, 20_000),
    }
)
widget = pmv.density_scatter(
    "True Values",
    "Predictions",
    df=df_ml,
    identity_line={"style": {"color": "black"}},
    best_fit_line={"style": {"color": "red"}},
    stats={"metrics": ["MAE", "RMSE", "R2"], "prefix": "20k points\n"},
)
widget.show()


# %% Structure-property relationship with log-scaled density colors
np_rng = np.random.default_rng(seed=456)
bond_length = np_rng.uniform(3.5, 6.0, 30_000) * 0.4 + np_rng.normal(0, 0.1, 30_000)
conductivity = np.exp(-bond_length) * 1000 + np_rng.normal(0, 50, 30_000)
widget = pmv.density_scatter(
    bond_length,
    conductivity,
    xlabel="Bond Length (Å)",
    ylabel="Conductivity (S/cm)",
    log=True,
    identity_line=False,
    best_fit_line=False,
    stats=False,
)
widget.show()


# %% 1 million points render fast as screen-space density bins
np_rng = np.random.default_rng(seed=999)
x_vals = np.concatenate(
    [
        np_rng.normal(-5, 1, 250_000),
        np_rng.normal(5, 1, 250_000),
        np_rng.normal(0, 5, 500_000),
    ]
)
y_vals = x_vals + np_rng.normal(0, 2, len(x_vals))
widget = pmv.density_hexbin(x_vals, y_vals, marginal_bins=100, identity_line=False)
widget.show()
