# %%
import numpy as np
import pandas as pd

import pymatviz as pmv


pmv.set_plotly_template("pymatviz_dark")

# Random classification data
np_rng = np.random.default_rng(seed=0)
rand_clf_size = 100
y_binary = np_rng.choice([0, 1], size=rand_clf_size)
y_proba = np.clip(y_binary - 0.1 * np_rng.normal(scale=5, size=rand_clf_size), 0.2, 0.9)
df_in = pd.DataFrame({"target": y_binary, "probability": y_proba})


# %% Plotly version - basic usage
fig = pmv.roc_curve_plotly(y_binary, y_proba)
fig.show()
# pmv.io.save_and_compress_svg(fig, "roc-curve-plotly")


# %% Plotly version - with DataFrame
fig = pmv.roc_curve_plotly("target", "probability", df=df_in)
fig.show()


# %% Multiple ROC curves on same plot
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

fig = pmv.roc_curve_plotly(targets=y_binary, probs_positive=classifiers)
fig.show()
pmv.io.save_and_compress_svg(fig, "roc-curve-plotly-multiple")
