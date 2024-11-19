# %%
import numpy as np

import pymatviz as pmv


# Random classification data
np_rng = np.random.default_rng(seed=0)
rand_clf_size = 100
y_binary = np_rng.choice([0, 1], size=rand_clf_size)
y_proba = np.clip(y_binary - 0.1 * np_rng.normal(scale=5, size=rand_clf_size), 0.2, 0.9)


# %% Relevance Plots
ax = pmv.precision_recall_curve(y_binary, y_proba)
pmv.io.save_and_compress_svg(ax, "precision-recall-curve")
