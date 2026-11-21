"""Interactive 3D plots of crystal structures with the MatterViz structure widget."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymatviz.process_data import normalize_periodic_structures
from pymatviz.structure import helpers
from pymatviz.widgets.structure import StructureWidget


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import ipywidgets as ipw
    import pandas as pd

    from pymatviz.typing import AnyStructure


def structure_3d(
    struct: AnyStructure
    | Mapping[str, AnyStructure]
    | Sequence[AnyStructure]
    | pd.Series,
    *,
    n_cols: int = 3,
    **kwargs: Any,
) -> StructureWidget | ipw.GridBox:
    """Plot one or several periodic structures as interactive 3D scenes.

    Args:
        struct (AnyStructure | Mapping | Sequence | pd.Series): A pymatgen Structure,
            ASE Atoms or PhonopyAtoms, or a sequence, mapping or Series of them.
        n_cols (int): Number of grid columns when plotting several structures.
            Defaults to 3.
        **kwargs: Passed to every StructureWidget, e.g. ``show_bonds``,
            ``color_scheme``, ``atom_radius``, ``show_image_atoms``, ``style`` or
            ``cell_type="conventional"`` to show the conventional standard cell.

    Returns:
        StructureWidget | ipw.GridBox: The widget for a single structure. For several,
            an ipywidgets grid with one titled cell per structure, each cell a VBox
            whose ``children[1]`` is its StructureWidget.

    Raises:
        TypeError: If an input is not a periodic structure.
        ValueError: If no structures are given or n_cols is not positive.
    """
    structures = normalize_periodic_structures(struct)
    widgets = {
        key: StructureWidget(structure=struct_i, **kwargs)
        for key, struct_i in structures.items()
    }
    if len(widgets) == 1:
        return next(iter(widgets.values()))
    return helpers.widget_grid(widgets, n_cols)
