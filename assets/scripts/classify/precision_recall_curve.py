"""Precision-recall curve examples."""

# %%
import numpy as np
import pandas as pd

import pymatviz as pmv


# Random classification data
np_rng = np.random.default_rng(seed=0)
rand_clf_size = 100
y_binary = np_rng.choice([0, 1], size=rand_clf_size)
y_proba = np.clip(y_binary - 0.1 * np_rng.normal(scale=5, size=rand_clf_size), 0.2, 0.9)
df_clf = pd.DataFrame({"target": y_binary, "probability": y_proba})


# %% Basic usage
widget = pmv.precision_recall_curve(y_binary, y_proba)
widget.show()


# %% With DataFrame
widget = pmv.precision_recall_curve("target", "probability", df=df_clf)
widget.show()


# %% Multiple PR curves on same plot
# Generate data for multiple classifiers
classifiers = {
    "Classifier A": (
        np.clip(y_binary - 0.1 * np_rng.normal(scale=5, size=rand_clf_size), 0.2, 0.9)
    ),
    "Classifier B": (
        np.clip(y_binary - 0.2 * np_rng.normal(scale=3, size=rand_clf_size), 0.1, 0.95)
    ),
    "Classifier C": (
        np.clip(
            y_binary - 0.15 * np_rng.normal(scale=4, size=rand_clf_size), 0.15, 0.85
        )
    ),
}

widget = pmv.precision_recall_curve(targets=y_binary, probs_positive=classifiers)
widget.show()
widget.to_img(
    f"{pmv.utils.ROOT}/assets/svg/precision-recall-curve-multiple.svg",
    width=500,
    transparent=True,
)
