# %%
import numpy as np

from pymatviz.io import save_and_compress_svg
from pymatviz.relevance import precision_recall_curve, roc_curve


# Random classification data
np.random.seed(42)
rand_clf_size = 100
y_binary = np.random.choice([0, 1], size=rand_clf_size)
y_proba = np.clip(
    y_binary - 0.1 * np.random.normal(scale=5, size=rand_clf_size), 0.2, 0.9
)


# %% Relevance Plots
ax = roc_curve(y_binary, y_proba)
save_and_compress_svg(ax, "roc-curve")


ax = precision_recall_curve(y_binary, y_proba)
save_and_compress_svg(ax, "precision-recall-curve")
