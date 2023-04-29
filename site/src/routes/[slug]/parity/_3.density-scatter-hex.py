import numpy as np

from pymatviz.parity import density_hexbin
from pymatviz.utils import save_and_compress_svg


rand_regression_size = 500
y_true = np.random.normal(5, 4, rand_regression_size)
y_pred = y_true + np.random.normal(0, 1, rand_regression_size)
y_std = (y_true - y_pred) * 10 * np.random.normal(0, 0.1, rand_regression_size)

ax = density_hexbin(y_pred, y_true)
save_and_compress_svg(ax, "density-scatter-hex")
