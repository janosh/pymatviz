"""Composition visualization widget for Jupyter notebooks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import traitlets as tl

from pymatviz.widgets._schema_traits import CompositionTraits
from pymatviz.widgets._traits import ELEMENT_COLOR_SCHEMES, JsonBool, normalized_trait
from pymatviz.widgets.matterviz import MatterVizWidget


if TYPE_CHECKING:
    from pymatviz.typing import FormulaLike


# MatterViz's species key: element, then an optional one-digit (decimal ok) charge
# magnitude and a sign ("Fe2+", "O-", "Fe2.5+"), as SPECIES_CHARGE in
# matterviz/src/lib/composition/parse.ts
_SPECIES_KEY = re.compile(r"([A-Z][a-z]?)((?:\d(?:\.\d+)?)?[+-])")


def _species_key(key: Any) -> str:
    """MatterViz key of a composition key: pymatgen ``Species`` (spin or other
    properties ignored) as ``{symbol}{charge}`` like ``"Fe2+"``, atomic numbers as
    element symbols, other keys as strings.
    """
    from pymatviz.utils import element_symbols

    if (oxi_state := getattr(key, "oxi_state", None)) is not None:
        magnitude = "" if abs(oxi_state) == 1 else f"{abs(oxi_state):g}"
        return f"{key.symbol}{magnitude}{'-' if oxi_state < 0 else '+'}"
    if isinstance(key, int) or (isinstance(key, str) and key.strip().isdigit()):
        return element_symbols[int(key)]
    return str(key)


def _composition_dict(composition: FormulaLike) -> dict[str, float]:
    """``{element or species: amount}`` of a formula, mapping, Composition or
    structure. Mappings (and structures' compositions) with oxidation-state keys
    (``"Fe2+"``, pymatgen ``Species``) keep them, summing keys that name the same
    species and dropping zero amounts; everything else goes through kemio, which
    sums oxidation states per element.
    """
    from pymatviz.compositions import to_composition
    from pymatviz.utils import df_ptable

    if isinstance(species_comp := getattr(composition, "composition", None), Mapping):
        composition = species_comp  # pymatgen structures with Species sites
    if isinstance(composition, Mapping):
        keys = [_species_key(key) for key in composition]
        matches = {key: _SPECIES_KEY.fullmatch(key) for key in keys}
        charged = [key for key in keys if key.endswith(("+", "-"))]
        if bad_keys := [key for key in charged if not matches[key]]:
            raise ValueError(f"species keys must look like Fe2+ or O-, got {bad_keys}")
        if any(matches.values()):
            elements = {match[1] if match else key for key, match in matches.items()}
            if unknown := sorted(elements - set(df_ptable.index)):
                raise ValueError(f"unknown element(s) {unknown} in {keys}")
            amounts: dict[str, float] = {}
            for key, amount in zip(keys, composition.values(), strict=True):
                amounts[key] = amounts.get(key, 0.0) + float(amount)
            non_finite = [key for key, val in amounts.items() if not math.isfinite(val)]
            if non_finite:
                raise ValueError(f"amounts must be finite, got {non_finite}")
            return {key: amount for key, amount in amounts.items() if amount != 0}
    return to_composition(composition, allow_negative=True).as_dict()


class CompositionWidget(CompositionTraits, MatterVizWidget):
    """MatterViz widget for visualizing compositions in Python notebooks.

    Examples:
        Basic usage:
        >>> import pymatviz as pmv
        >>> import kemio
        >>> comp = kemio.Composition("Fe2O3")
        >>> widget = pmv.composition(composition=comp)
        >>> widget

        With custom visualization options:
        >>> widget = pmv.composition(
        ...     composition=comp,
        ...     mode="bar",
        ...     show_percentages=True,
        ...     color_scheme="Jmol",
        ...     style="width: 400px; margin: 20px auto;",
        ... )
    """

    composition = normalized_trait(_composition_dict)  # on every assignment
    # Visualization options
    mode = tl.CaselessStrEnum(["pie", "bar", "bubble"], default_value="pie").tag(
        sync=True
    )
    show_percentages = JsonBool(default_value=False).tag(sync=True)
    color_scheme = tl.CaselessStrEnum(ELEMENT_COLOR_SCHEMES, default_value="Jmol").tag(
        sync=True
    )

    def __init__(self, composition: FormulaLike | None = None, **kwargs: Any) -> None:
        """Initialize the CompositionWidget.

        Args:
            composition: A formula string, {element: amount} dict, kemio or
                pymatgen Composition, or structure (negative amounts allowed).
                Mappings and pymatgen structures with oxidation states (``"Fe2+"``
                keys, pymatgen ``Species``) keep them: Fe2+ and Fe3+ show as separate
                segments colored by Fe. Formula strings sum them per element.
            **kwargs: Additional widget properties
        """
        super().__init__(composition=composition, **kwargs)
