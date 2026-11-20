"""Confusion matrix examples.

HeatmapMatrix renders HTML cells: export with widget.to_img(fmt="png"), not SVG.
"""

# %%
import numpy as np
from numpy.random import default_rng

import pymatviz as pmv


np_rng = default_rng(seed=0)


# %% Binary classification from raw labels
y_true = ["Negative"] * 6 + ["Positive"] * 2
# 4 TN, 2 FP, 1 FN, 1 TP
y_pred = ["Negative"] * 4 + ["Positive", "Positive", "Negative", "Positive"]
widget = pmv.confusion_matrix(y_true=y_true, y_pred=y_pred)
widget.show()


# %% Pre-computed matrix with binary classification metrics
widget = pmv.confusion_matrix(
    conf_mat=np.array([[0.7, 0.3], [0.15, 0.85]]),
    x_labels=("Stable", "Unstable"),
    normalize=False,
    float_fmt=".2f",
    color_scale="interpolateReds",
    metrics={"Prec": ".1%", "F1": ".0%", "MCC": ".2f"},
)
widget.show()


# %% Multi-class crystal system classification
n_samples = 300
crystal_systems = ["cubic", "hexagonal", "tetragonal", "orthorhombic"]
# Generate true labels with uneven class distribution
y_true = np_rng.choice(crystal_systems, n_samples, p=[0.4, 0.3, 0.2, 0.1])
# Simulate predictions with 75% accuracy and some systematic errors
y_pred = np.where(
    np_rng.random(n_samples) < 0.75,
    y_true,  # correct predictions
    np_rng.choice(crystal_systems, n_samples),  # random errors
)
widget = pmv.confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    x_labels=crystal_systems,
    color_scale="interpolateViridis",
    tile_size="90px",
)
widget.show()
