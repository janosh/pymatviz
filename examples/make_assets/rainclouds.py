# %%
import numpy as np

import pymatviz as pmv


pmv.set_plotly_template("pymatviz_white")

# Set up the RNG with a seed for reproducibility
rng = np.random.default_rng(42)


# %% Example 1: Two bimodal distributions
data_bimodal = {
    "Distribution A": np.concatenate(
        [rng.normal(-2, 0.5, 600), rng.normal(2, 0.5, 400)]
    ),
    "Distribution B": np.concatenate(
        [rng.normal(-1, 0.3, 400), rng.normal(3, 0.7, 600)]
    ),
}

fig_bimodal = pmv.rainclouds(data_bimodal, figsize=(800, 400))
fig_bimodal.update_layout(
    title="Raincloud Plot: Two Bimodal Distributions",
    xaxis_title="Value",
    yaxis_title="Distribution",
)
fig_bimodal.layout.margin.t = 40
fig_bimodal.layout.title.x = 0.5
fig_bimodal.show()
pmv.io.save_and_compress_svg(fig_bimodal, "raincloud-bimodal")


# Example 2: Three trimodal distributions
data_trimodal = {
    "Distribution X": np.concatenate(
        [
            rng.normal(-3, 0.4, 300),
            rng.normal(0, 0.3, 400),
            rng.normal(3, 0.5, 300),
        ]
    ),
    "Distribution Y": np.concatenate(
        [
            rng.normal(-2, 0.3, 350),
            rng.normal(1, 0.4, 350),
            rng.normal(4, 0.6, 300),
        ]
    ),
    "Distribution Z": np.concatenate(
        [
            rng.normal(-4, 0.5, 250),
            rng.normal(-1, 0.3, 450),
            rng.normal(2, 0.4, 300),
        ]
    ),
}

fig_trimodal = pmv.rainclouds(data_trimodal, figsize=(1000, 600))
fig_trimodal.update_layout(
    title="Raincloud Plot: Three Trimodal Distributions",
    xaxis_title="Value",
    yaxis_title="Distribution",
)
fig_trimodal.layout.margin.t = 40
fig_trimodal.layout.title.x = 0.5
fig_trimodal.show()
pmv.io.save_and_compress_svg(fig_trimodal, "raincloud-trimodal")
