import numpy as np

from pymatviz.parity import scatter_with_err_bar
from pymatviz.utils import save_and_compress_svg


rand_regression_size = 500
y_true = np.random.normal(5, 4, rand_regression_size)
y_pred = y_true + np.random.normal(0, 1, rand_regression_size)
y_std = (y_true - y_pred) * 10 * np.random.normal(0, 0.1, rand_regression_size)


ax = scatter_with_err_bar(y_pred, y_true, yerr=y_std)
save_and_compress_svg(ax, "scatter-with-err-bar")
