# use redundant as-import following PEP 484, see
# https://github.com/microsoft/pylance-release/issues/856#issuecomment-763793949
# and https://peps.python.org/pep-0484/#stub-files 'Additional notes on stub files'
from pymatviz.correlation import marchenko_pastur as marchenko_pastur
from pymatviz.correlation import marchenko_pastur_pdf as marchenko_pastur_pdf
from pymatviz.cumulative import cumulative_error as cumulative_error
from pymatviz.cumulative import cumulative_residual as cumulative_residual
from pymatviz.histograms import hist_elemental_prevalence as hist_elemental_prevalence
from pymatviz.histograms import residual_hist as residual_hist
from pymatviz.histograms import spacegroup_hist as spacegroup_hist
from pymatviz.histograms import true_pred_hist as true_pred_hist
from pymatviz.parity import density_hexbin as density_hexbin
from pymatviz.parity import density_hexbin_with_hist as density_hexbin_with_hist
from pymatviz.parity import density_scatter as density_scatter
from pymatviz.parity import density_scatter_with_hist as density_scatter_with_hist
from pymatviz.parity import residual_vs_actual as residual_vs_actual
from pymatviz.parity import scatter_with_err_bar as scatter_with_err_bar
from pymatviz.ptable import count_elements as count_elements
from pymatviz.ptable import ptable_heatmap as ptable_heatmap
from pymatviz.ptable import ptable_heatmap_plotly as ptable_heatmap_plotly
from pymatviz.ptable import ptable_heatmap_ratio as ptable_heatmap_ratio
from pymatviz.relevance import precision_recall_curve as precision_recall_curve
from pymatviz.relevance import roc_curve as roc_curve
from pymatviz.sankey import sankey_from_2_df_cols as sankey_from_2_df_cols
from pymatviz.structure_viz import plot_structure_2d as plot_structure_2d
from pymatviz.sunburst import spacegroup_sunburst as spacegroup_sunburst
from pymatviz.uncertainty import error_decay_with_uncert as error_decay_with_uncert
from pymatviz.uncertainty import qq_gaussian as qq_gaussian
from pymatviz.utils import ROOT as ROOT
from pymatviz.utils import add_mae_r2_box as add_mae_r2_box
from pymatviz.utils import annotate_bars as annotate_bars
