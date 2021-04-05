from .correlation import marchenko_pastur, marchenko_pastur_pdf
from .cumulative import add_dropdown, cum_err, cum_res
from .elements import (
    count_elements,
    hist_elemental_prevalence,
    ptable_elemental_prevalence,
    ptable_elemental_ratio,
)
from .histograms import residual_hist, spacegroup_hist, true_pred_hist
from .metrics import classification_metrics, regression_metrics
from .parity import (
    density_hexbin,
    density_hexbin_with_hist,
    density_scatter,
    density_scatter_with_hist,
    residual_vs_actual,
    scatter_with_err_bar,
)
from .quantile import qq_gaussian
from .ranking import err_decay
from .relevance import precision_recall_curve, roc_curve
from .utils import ROOT, add_identity, annotate_bar_heights
