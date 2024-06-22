# %%
import numpy as np

from pymatviz.io import save_and_compress_svg
from pymatviz.relevance import precision_recall_curve, roc_curve


# Random classification data
np_rng = np.random.default_rng(seed=0)
rand_clf_size = 100
y_binary = np_rng.choice([0, 1], size=rand_clf_size)
y_proba = np.clip(y_binary - 0.1 * np_rng.normal(scale=5, size=rand_clf_size), 0.2, 0.9)


# %% Relevance Plots
ax = roc_curve(y_binary, y_proba)
save_and_compress_svg(ax, "roc-curve")


ax = precision_recall_curve(y_binary, y_proba)
save_and_compress_svg(ax, "precision-recall-curve")
