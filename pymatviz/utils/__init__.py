"""pymatviz utility functions."""

from __future__ import annotations

import os


PKG_DIR: str = os.path.dirname(os.path.dirname(__file__))
ROOT: str = os.path.dirname(PKG_DIR)


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
from pymatviz.utils.misc import ExperimentalWarning
from pymatviz.utils.plotting import (
    annotate,
    apply_matplotlib_template,
    get_cbar_label_formatter,
    get_fig_xy_range,
    get_font_color,
    pretty_label,
    validate_fig,
)
from pymatviz.utils.testing import TEST_FILES
from pymatviz.utils.typing import (
    BACKENDS,
    MATPLOTLIB,
    PLOTLY,
    VALID_COLOR_ELEM_STRATEGIES,
    VALID_FIG_NAMES,
    VALID_FIG_TYPES,
    AxOrFig,
    Backend,
    ColorElemTypeStrategy,
    CrystalSystem,
    ElemValues,
    P,
    R,
    T,
)
