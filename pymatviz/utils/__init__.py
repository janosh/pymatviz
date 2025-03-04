"""pymatviz utility functions."""

# ruff: noqa: E402 (Module level import not at top of file)

from __future__ import annotations

import os


PKG_DIR: str = os.path.dirname(os.path.dirname(__file__))
ROOT: str = os.path.dirname(PKG_DIR)


class ExperimentalWarning(Warning):
    """Warning for experimental features."""


from pymatviz.utils.data import (
    atomic_numbers,
    bin_df_cols,
    df_ptable,
    df_to_arrays,
    element_symbols,
    html_tag,
    normalize_to_dict,
    patch_dict,
    si_fmt,
    si_fmt_int,
    spg_num_to_from_symbol,
    spg_to_crystal_sys,
)
from pymatviz.utils.plotting import (
    annotate,
    apply_matplotlib_template,
    get_fig_xy_range,
    get_font_color,
    luminance,
    pick_max_contrast_color,
    pretty_label,
    validate_fig,
)
from pymatviz.utils.testing import TEST_FILES
