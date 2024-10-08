# %%
import numpy as np

import pymatviz as pmv


pmv.set_plotly_template("pymatviz_white")

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

fig_bi = pmv.rainclouds(data_bimodal)
fig_bi.layout.title.update(text="Raincloud Plot: Two Bimodal Distributions", x=0.5)
fig_bi.layout.xaxis.title = "Value"
fig_bi.layout.yaxis.title = "Distribution"
fig_bi.layout.margin.t = 40
fig_bi.show()
pmv.io.save_and_compress_svg(fig_bi, "rainclouds-bimodal")


# %% Example 2: Three trimodal distributions
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

fig_tri = pmv.rainclouds(data_trimodal)
fig_tri.layout.title.update(text="Raincloud Plot: Three Trimodal Distributions", x=0.5)
fig_tri.layout.xaxis.title = "Value"
fig_tri.layout.yaxis.title = "Distribution"
fig_tri.layout.margin.t = 40
fig_tri.show()
pmv.io.save_and_compress_svg(fig_tri, "rainclouds-trimodal")
