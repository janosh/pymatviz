"""Hierarchical multi-level pie charts (i.e. sunbursts).

E.g. for crystal symmetry distributions.
"""

from __future__ import annotations

import textwrap
import warnings
from typing import TYPE_CHECKING, get_args

import plotly.graph_objects as go

from pymatviz import chem_env
from pymatviz.process_data import normalize_structures
from pymatviz.sunburst.chem_env import chem_env_sunburst
from pymatviz.sunburst.chem_sys import chem_sys_sunburst
from pymatviz.sunburst.spacegroup import spacegroup_sunburst
from pymatviz.typing import ShowCounts


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from pymatgen.core import Structure

    from pymatviz.typing import FormulaGroupBy
