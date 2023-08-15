"""pymatviz: A Python package for visualizing materials data."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pymatviz.correlation import marchenko_pastur, marchenko_pastur_pdf
from pymatviz.cumulative import cumulative_error, cumulative_residual
from pymatviz.histograms import (
    hist_elemental_prevalence,
    residual_hist,
    spacegroup_hist,
    true_pred_hist,
)
from pymatviz.parity import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)
from pymatviz.ptable import (
    count_elements,
    ptable_heatmap,
    ptable_heatmap_plotly,
    ptable_heatmap_ratio,
)
from pymatviz.relevance import precision_recall_curve, roc_curve
from pymatviz.sankey import sankey_from_2_df_cols
from pymatviz.structure_viz import plot_structure_2d
from pymatviz.sunburst import spacegroup_sunburst
from pymatviz.uncertainty import error_decay_with_uncert, qq_gaussian
from pymatviz.utils import ROOT, annotate_bars, annotate_metrics


try:
    __version__ = version("pymatviz")
except PackageNotFoundError:
    pass  # package not installed
