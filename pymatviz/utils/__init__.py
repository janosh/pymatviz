"""pymatviz utility functions."""

from __future__ import annotations

import os


PKG_DIR = os.path.dirname(__file__)
ROOT = os.path.dirname(PKG_DIR)


class ExperimentalWarning(Warning):
    """Warning for experimental features."""


from pymatviz.utils.data import (
    atomic_numbers,
    bin_df_cols,
    crystal_sys_from_spg_num,
    df_ptable,
    df_to_arrays,
    element_symbols,
    html_tag,
    normalize_to_dict,
    patch_dict,
    si_fmt,
    si_fmt_int,
)
from pymatviz.utils.image import luminance, pick_bw_for_contrast
