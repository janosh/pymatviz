"""Chemical environments grouped by coordination number in a sunburst."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pymatviz.chem_env import chem_env_nodes
from pymatviz.widgets.sunburst import SunburstWidget


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pymatviz.chem_env import ChemEnvSettings
    from pymatviz.typing import PeriodicStructure


def chem_env_sunburst(
    structures: PeriodicStructure | Sequence[PeriodicStructure],
    *,
    chem_env_settings: ChemEnvSettings = "crystal_nn",
    normalize: bool = False,
    max_coordination_numbers: int | None = None,
    max_environments: int | None = None,
    mode: Literal["other", "drop"] = "other",
    **widget_options: Any,
) -> SunburstWidget:
    """Show coordination numbers and their chemical environments as a sunburst.

    Use CrystalNN (default), "chemenv", or a dict of ChemEnv settings.
    With normalize=True, each structure contributes one total count. Limit the
    most frequent coordination numbers and environments per coordination number
    with max_coordination_numbers and max_environments. Mode "other" combines
    omitted counts into one leaf per group; "drop" omits them. Additional
    options go to SunburstWidget.
    """
    nodes = chem_env_nodes(
        structures,
        chem_env_settings=chem_env_settings,
        normalize=normalize,
        max_coordination_numbers=max_coordination_numbers,
        max_environments=max_environments,
        mode=mode,
    )
    return SunburstWidget(data=nodes, **widget_options)
