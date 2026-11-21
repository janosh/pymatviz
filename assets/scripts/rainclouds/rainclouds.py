"""Raincloud plots."""

# %%
import numpy as np

import pymatviz as pmv


# Set up the RNG with a seed for reproducibility
rng = np.random.default_rng(seed=0)


# %% Example 1: Two bimodal distributions
data_bimodal = {
    "Distribution A": np.concatenate(
        [rng.normal(-2, 0.5, 600), rng.normal(2, 0.5, 400)]
    ),
    "Distribution B": np.concatenate(
        [rng.normal(-1, 0.3, 400), rng.normal(3, 0.7, 600)]
    ),
}

widget_bi = pmv.rainclouds(
    data_bimodal,
    title="Raincloud Plot: Two Bimodal Distributions",
    x_axis={"label": "Value"},
)
widget_bi.show()
widget_bi.to_img(
    f"{pmv.utils.ROOT}/assets/svg/rainclouds-bimodal.svg", width=700, transparent=True
)


# %% Example 2: Three trimodal distributions as beeswarms
data_trimodal = {
    "Distribution X": np.concatenate(
        [rng.normal(-3, 0.4, 300), rng.normal(0, 0.3, 400), rng.normal(3, 0.5, 300)]
    ),
    "Distribution Y": np.concatenate(
        [rng.normal(-2, 0.3, 350), rng.normal(1, 0.4, 350), rng.normal(4, 0.6, 300)]
    ),
    "Distribution Z": np.concatenate(
        [rng.normal(-4, 0.5, 250), rng.normal(-1, 0.3, 450), rng.normal(2, 0.4, 300)]
    ),
}

widget_tri = pmv.rainclouds(
    data_trimodal,
    orientation="vertical",
    points="swarm",
    title="Raincloud Plot: Three Trimodal Distributions",
    y_axis={"label": "Value"},
)
widget_tri.show()
widget_tri.to_img(
    f"{pmv.utils.ROOT}/assets/svg/rainclouds-trimodal.svg", width=700, transparent=True
)
