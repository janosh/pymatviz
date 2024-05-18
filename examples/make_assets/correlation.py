# %%
import numpy as np

from pymatviz.correlation import marchenko_pastur
from pymatviz.io import save_and_compress_svg


# %% Correlation Plots
# Plot eigenvalue distribution of a pure-noise correlation matrix
# i.e. the correlation matrix contains no significant correlations
# beyond the spurious correlation that occurs randomly
n_rows, n_cols = 500, 1000
rand_wide_mat = np.random.normal(0, 1, size=(n_rows, n_cols))

corr_mat = np.corrcoef(rand_wide_mat)

ax = marchenko_pastur(corr_mat, gamma=n_cols / n_rows)
save_and_compress_svg(ax, "marchenko-pastur")

# Plot eigenvalue distribution of a correlation matrix with significant
# (i.e. non-noise) eigenvalue
n_rows, n_cols = 50, 400
linear_matrix = np.arange(n_rows * n_cols).reshape(n_rows, n_cols) / n_cols

corr_mat = np.corrcoef(linear_matrix + rand_wide_mat[:n_rows, :n_cols])

ax = marchenko_pastur(corr_mat, gamma=n_cols / n_rows)
save_and_compress_svg(ax, "marchenko-pastur-significant-eval")

# Plot eigenvalue distribution of a rank-deficient correlation matrix
n_rows, n_cols = 600, 500
rand_tall_mat = np.random.normal(0, 1, size=(n_rows, n_cols))

corr_mat_rank_deficient = np.corrcoef(rand_tall_mat)

ax = marchenko_pastur(corr_mat_rank_deficient, gamma=n_cols / n_rows)
save_and_compress_svg(ax, "marchenko-pastur-rank-deficient")
