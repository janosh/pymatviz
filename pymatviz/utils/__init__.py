"""pymatviz utility functions."""

# ruff: noqa: E402 (Module level import not at top of file)

from __future__ import annotations

import os


PKG_DIR: str = os.path.dirname(os.path.dirname(__file__))
ROOT: str = os.path.dirname(PKG_DIR)


from pymatviz.utils.data import atomic_numbers, df_ptable, element_symbols
from pymatviz.utils.testing import TEST_FILES
