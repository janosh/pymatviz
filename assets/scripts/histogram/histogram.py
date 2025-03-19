# %%
import numpy as np

import pymatviz as pmv


pmv.set_plotly_template("pymatviz_white")


# %% plot 2 Gaussians and their cumulative distribution functions
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
gauss1 = np_rng.normal(5, 4, rand_regression_size)
gauss2 = np_rng.normal(10, 2, rand_regression_size)

fig = pmv.histogram({"Gaussian 1": gauss1, "Gaussian 2": gauss2}, bins=100)
pmv.powerups.add_ecdf_line(fig)
fig.show()
# pmv.io.save_and_compress_svg(fig, "histogram-ecdf")
