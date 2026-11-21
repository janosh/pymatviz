"""Plot X-ray diffraction patterns with the MatterViz XRD widget."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import matscikit as mk
import numpy as np

from pymatviz.structures import is_structure_like, to_structure
from pymatviz.widgets._normalize import normalize_xrd_pattern
from pymatviz.widgets.xrd import XrdWidget


if TYPE_CHECKING:
    from pymatgen.analysis.diffraction.xrd import DiffractionPattern

    from pymatviz.typing import AnyStructure


type PatternOrStruct = mk.XrdPattern | DiffractionPattern | AnyStructure
cu_k_alpha_wavelength = 1.54184  # Angstroms


def _to_pattern_dict(
    label: str, pattern_or_struct: Any, wavelength: float
) -> dict[str, Any]:
    """Pattern dict of a structure (computed with matscikit) or pattern object, with
    its intensities checked to be present, finite and non-negative.
    """
    if is_structure_like(pattern_or_struct):
        struct = to_structure(pattern_or_struct)
        (pattern_or_struct,) = mk.xrd_patterns([struct], wavelength=wavelength)
    try:
        pattern = normalize_xrd_pattern(pattern_or_struct)
    except TypeError:
        pattern = None
    if pattern is None:
        raise TypeError(
            f"{label!r}: expected a structure (matscikit, pymatgen, ASE, phonopy), "
            f"XrdPattern or DiffractionPattern, got {type(pattern_or_struct).__name__}"
        )
    intensities = np.asarray(pattern.get("y", []), dtype=float)
    if len(intensities) == 0:
        raise ValueError(f"No intensities found in the diffraction pattern for {label}")
    if not np.isfinite(intensities).all() or np.any(intensities < 0):
        raise ValueError(f"Intensities for {label!r} must be finite and non-negative")
    return pattern


def xrd_plot(
    patterns: PatternOrStruct | Mapping[Any, PatternOrStruct],
    *,
    wavelength: float = cu_k_alpha_wavelength,
    **kwargs: Any,
) -> XrdWidget:
    """Plot XRD patterns of structures (computed with matscikit's xrd_patterns) or
    precomputed matscikit XrdPatterns / pymatgen DiffractionPatterns.

    Patterns of several inputs are overlaid with a legend and scaled to a shared
    maximum of 100. MatterViz labels each pattern's 5 strongest peaks with their
    Miller indices (and 2θ angles for up to 2 patterns).

    Args:
        patterns (PatternOrStruct | Mapping[str, PatternOrStruct]): A single
            structure (matscikit, pymatgen, ASE, phonopy; disordered sites scatter
            by occupancy), XrdPattern or DiffractionPattern, or a mapping from
            legend labels to them.
        wavelength (float): X-ray wavelength in Å used to compute patterns of
            structures. Defaults to 1.54184 (Cu K-alpha).
        **kwargs: Passed to XrdWidget, e.g. ``style`` or ``controls_open``.

    Raises:
        ValueError: If patterns are empty, an intensity is negative or not finite,
            or no intensity is positive.
        TypeError: If patterns is not a structure or pattern, or a mapping of them.

    Returns:
        XrdWidget: The MatterViz XRD plot.
    """
    is_single = not isinstance(patterns, Mapping)
    labeled = {"XRD Pattern": patterns} if is_single else patterns
    if not labeled:
        raise ValueError("patterns must not be empty")
    computed = {
        str(label): _to_pattern_dict(str(label), value, wavelength)
        for label, value in labeled.items()
    }
    if len(computed) != len(labeled):
        raise ValueError(f"Labels must be unique as strings, got {list(labeled)}")
    if not any(np.any(np.asarray(pat["y"]) > 0) for pat in computed.values()):
        raise ValueError("At least one diffraction intensity must be positive")
    return XrdWidget(
        patterns=computed.popitem()[1] if is_single else computed, **kwargs
    )
