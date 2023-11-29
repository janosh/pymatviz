"""pymatviz: A Python package for visualizing materials data."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import plotly.express as px

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
    ptable_hists,
)
from pymatviz.relevance import precision_recall_curve, roc_curve
from pymatviz.sankey import sankey_from_2_df_cols
from pymatviz.structure_viz import plot_structure_2d
from pymatviz.sunburst import spacegroup_sunburst
from pymatviz.uncertainty import error_decay_with_uncert, qq_gaussian
from pymatviz.utils import (
    PKG_DIR,
    ROOT,
    annotate_bars,
    annotate_metrics,
    styled_html_tag,
)


try:
    __version__ = version("pymatviz")
except PackageNotFoundError:
    pass  # package not installed


bandgap_col = "band_gap"
charge_col = "total_charge"
crystal_sys_col = "crystal_system"
e_above_hull_col = "energy_above_hull"
e_form_col = "formation_energy_per_atom"
energy_col = "energy"
formula_pretty_col = "formula_pretty"
formula_col = "formula"
id_col = "material_id"
n_elems_col = "nelements"
oxi_guesses_col = "oxi_state_guesses"
spg_col = "spacegroup_number"
structure_col = "structure"
symmetry_col = "symmetry"
volume_col = "volume"
volume_per_atom_col = "volume_per_atom"

small_font = "font-size: 0.9em; font-weight: lighter;"
eV_per_atom = styled_html_tag("(eV/atom)", tag="span", style=small_font)
eV = styled_html_tag("(eV)", tag="span", style=small_font)
elem_charge = styled_html_tag("(e)", tag="span", style=small_font)  # elementary charge
cubic_angstrom = styled_html_tag("(Å<sup>3</sup>)", tag="span", style=small_font)
angstrom = styled_html_tag("(Å)", tag="span", style=small_font)
angstrom_per_atom = styled_html_tag("(Å/atom)", tag="span", style=small_font)

px.defaults.labels |= {
    bandgap_col: "Band Gap",
    charge_col: f"Charge {elem_charge}",
    crystal_sys_col: "Crystal System",
    e_above_hull_col: f"E<sub>hull dist</sub> {eV_per_atom}",
    e_form_col: f"E<sub>form</sub> {eV_per_atom}",
    energy_col: f"Energy {eV}",
    formula_pretty_col: "Formula",
    formula_col: "Formula",
    id_col: "Material ID",
    n_elems_col: "N<sub>elements</sub>",
    oxi_guesses_col: "Oxidation State Guesses",
    spg_col: "Spacegroup",
    structure_col: "Structure",
    symmetry_col: "Symmetry",
    volume_col: f"Volume {cubic_angstrom}",
    volume_per_atom_col: f"Volume {angstrom_per_atom}",
}
px.defaults.template = "plotly_white"
