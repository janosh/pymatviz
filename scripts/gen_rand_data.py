import numpy as np
import pandas as pd

from mlmatrics import ROOT

xs = np.random.rand(100)
y_pred = xs + 0.1 * np.random.normal(size=100)
y_true = xs + 0.1 * np.random.normal(size=100)

df_regr = pd.DataFrame({"xs": xs, "y_pred": y_pred, "y_true": y_true})
df_regr.to_csv(f"{ROOT}/data/rand_regr.csv", index=False, float_format="%g")

y_binary = np.random.choice([0, 1], 100)
y_proba = np.clip(y_binary - 0.1 * np.random.normal(scale=5, size=100), 0.2, 0.9)

df_clf = pd.DataFrame(
    {"y_binary": y_binary, "y_proba": y_proba, "y_pred": y_proba.round()}
)
df_clf.to_csv(f"{ROOT}/data/rand_clf.csv", index=False, float_format="%g")

r_rows, n_cols = 500, 1000
rand_mat = np.random.normal(0, 1, size=(r_rows, n_cols))
pd.DataFrame(rand_mat).to_csv(
    f"{ROOT}/data/rand_wide_matrix.csv", index=False, float_format="%g", header=False
)

r_rows, n_cols = 600, 500
rand_mat = np.random.normal(0, 1, size=(r_rows, n_cols))
pd.DataFrame(rand_mat).to_csv(
    f"{ROOT}/data/rand_tall_matrix.csv", index=False, float_format="%g", header=False
)
