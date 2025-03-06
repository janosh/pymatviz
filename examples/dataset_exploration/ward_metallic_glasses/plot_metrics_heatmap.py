"""This script reads metrics from a YAML file and creates heatmap visualizations
to compare different feature sets across various metrics.
"""

import os

import pandas as pd
import plotly
import plotly.express as px
import yaml

import pymatviz as pmv
from pymatviz.utils.data import si_fmt_int


module_dir = os.path.dirname(__file__)
metrics_file = f"{module_dir}/gfa_metrics.yaml"
with open(metrics_file) as file:
    metrics_data = yaml.safe_load(file)


higher_better = ["accuracy", "precision", "recall", "f1", "roc_auc", "r2"]
lower_better = ["mae", "mse", "rmse"]
metric_labels = dict(
    roc_auc="ROC AUC",
    prc_auc="PRC AUC",
    f1="F1 Score",
    accuracy="Accuracy",
    precision="Precision",
    recall="Recall",
    mae="MAE",
    # n_features="Features",
    # train_size="Train Size",
    # test_size="Test Size",
)


def row_name(idx: str, df_metrics: pd.DataFrame) -> str:
    """Format heatmap row name with train/test size and number of features."""
    key_map = {
        "train_size": "Train",
        "test_size": "Test",
        "n_features": "Features",
    }
    return f"<b>{idx}</b> " + " ".join(
        f"{label}={si_fmt_int(df_metrics.loc[idx, key])}"
        for key, label in key_map.items()
    )


for target, models in metrics_data.items():
    for model_name, data_splits in models.items():
        for split_name, feature_sets in data_splits.items():
            df_metrics = pd.DataFrame(feature_sets).T.sort_values(
                by="roc_auc", ascending=False
            )
            df_metrics.index = [row_name(idx, df_metrics) for idx in df_metrics.index]
            df_plot = df_metrics[[k for k in metric_labels if k in df_metrics]].rename(
                columns=metric_labels
            )
            fig = px.imshow(
                df_plot,
                color_continuous_scale=plotly.colors.sequential.Viridis,
                text_auto=".3f",
                aspect="auto",
            )

            n_rows, n_cols = df_plot.shape
            title = f"{split_name} - {model_name} - {target}"
            fig.layout.title.update(text=title, x=0.5, y=0.97)
            fig.layout.height = 40 * n_rows
            fig.layout.width = 200 + 100 * n_cols
            fig.layout.font.size = 14
            fig.layout.yaxis.update(title="Input Features", ticklabelstandoff=5)
            # fig.layout.xaxis.title = "Metric"
            fig.layout.xaxis.update(side="top")
            fig.layout.margin.update(l=10, r=10, t=60, b=10)
            fig.layout.coloraxis.showscale = False  # hide colorbar

            fig.show()
            img_name = f"gfa-metrics-heatmap-{target}-{model_name}-{split_name}"
            os.makedirs(out_dir := f"{module_dir}/tmp/figs/{target}", exist_ok=True)
            pmv.save_fig(fig, f"{out_dir}/{img_name}.png", scale=2)
