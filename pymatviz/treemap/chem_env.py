"""Chemical environments grouped by coordination number in a treemap."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pymatviz.chem_env import chem_env_nodes
from pymatviz.widgets.treemap import TreemapWidget


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatviz.chem_env import ChemEnvSettings
    from pymatviz.typing import PeriodicStructure


def chem_env_treemap(
    structures: PeriodicStructure | Sequence[PeriodicStructure],
    *,
    chem_env_settings: ChemEnvSettings = "crystal_nn",
    normalize: bool = False,
    max_coordination_numbers: int | None = None,
    max_environments: int | None = None,
    mode: Literal["other", "drop"] = "other",
    **widget_options: Any,
) -> TreemapWidget:
    """Show coordination numbers and their chemical environments as a treemap.

    Use CrystalNN (default), "chemenv", or a dict of ChemEnv settings.
    With normalize=True, each structure contributes one total count. Limit the
    most frequent coordination numbers and environments per coordination number
    with max_coordination_numbers and max_environments. Mode "other" combines
    omitted counts into one leaf per group; "drop" omits them. Additional
    options go to TreemapWidget.
    """
    nodes = chem_env_nodes(
        structures,
        chem_env_settings=chem_env_settings,
        normalize=normalize,
        max_coordination_numbers=max_coordination_numbers,
        max_environments=max_environments,
        mode=mode,
    )
    return TreemapWidget(data=nodes, **widget_options)
