# %%
import numpy as np
from numpy.random import default_rng

import pymatviz as pmv


pmv.set_plotly_template("pymatviz_dark")

np_rng = default_rng(seed=0)


# %%
y_true = ["Negative"] * 6 + ["Positive"] * 2
# 4 TN, 2 FP, 1 FN, 1 TP
y_pred = ["Negative"] * 4 + ["Positive", "Positive", "Negative", "Positive"]

fig = pmv.confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    annotations=lambda cnt, total, *_: f"N={cnt:,}<br>{cnt / total:.1%}",
)
fig.show()


# %% Custom annotations for material stability with specific metrics
fig = pmv.confusion_matrix(
    # [[TP, FP], [FN, TN]]
    conf_mat=np.array([[0.7, 0.3], [0.15, 0.85]]),
    x_labels=("Stable", "Unstable"),
    y_labels=("Stable", "Unstable"),
    normalize=False,
    colorscale="Reds",
    metrics={"Prec": ".2%", "F1": ".0%", "MCC": ".2f"},
)
fig.layout.xaxis.title = "Predicted Stability"
fig.layout.yaxis.title = "True Stability"
fig.show()
# pmv.io.save_and_compress_svg(fig, "stability-confusion-matrix")


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
fig = pmv.confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    colorscale="Viridis",
)
fig.layout.xaxis.title = "Predicted System"
fig.layout.yaxis.title = "True System"
fig.layout.width = 650
fig.layout.height = 650
fig.show()
# pmv.io.save_and_compress_svg(fig, "crystal-system-confusion-matrix")


# %% Binary classification passing in raw labels and custom class names
n_samples = 200
# Generate synthetic stability data
y_true = np_rng.choice([True, False], n_samples, p=[0.3, 0.7])
# Model predicts unstable more often than it should
y_pred = np.where(
    np_rng.random(n_samples) < 0.8,  # 80% accuracy
    y_true,
    np_rng.choice([True, False], n_samples, p=[0.4, 0.6]),
)
fig = pmv.confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    x_labels=("Stable", "Unstable"),
    y_labels=("Stable", "Unstable"),
    colorscale="RdBu",
)
fig.layout.xaxis.title = "Predicted Stability"
fig.layout.yaxis.title = "True Stability"
fig.show()
pmv.io.save_and_compress_svg(fig, "stability-confusion-matrix-raw")
