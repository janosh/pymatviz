"""Plot X-ray diffraction patterns with the MatterViz XRD widget."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from pymatgen.core import IStructure

from pymatviz.structures import is_ase_atoms
from pymatviz.widgets.xrd import XrdWidget


if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.analysis.diffraction.xrd import DiffractionPattern


type PatternOrStruct = DiffractionPattern | IStructure | Atoms
cu_k_alpha_wavelength = 1.54184  # Angstroms


def _to_diffraction_pattern(
    label: str, pattern_or_struct: Any, wavelength: float
) -> DiffractionPattern:
    """Compute a structure's pattern (or pass a pattern through) and check its
    intensities are present, finite and non-negative.
    """
    from pymatgen.analysis.diffraction.xrd import DiffractionPattern, XRDCalculator

    if is_ase_atoms(pattern_or_struct):
        from pymatgen.io.ase import AseAtomsAdaptor

        pattern_or_struct = AseAtomsAdaptor().get_structure(pattern_or_struct)
    if isinstance(pattern_or_struct, IStructure):
        pattern = XRDCalculator(wavelength=wavelength).get_pattern(pattern_or_struct)
    elif isinstance(pattern_or_struct, DiffractionPattern):
        pattern = pattern_or_struct
    else:
        raise TypeError(
            f"{label!r}: expected a pymatgen Structure, ASE Atoms or DiffractionPattern"
            f", got {type(pattern_or_struct).__name__}"
        )
    intensities = pattern.y
    if intensities is None or len(intensities) == 0:
        raise ValueError(f"No intensities found in the diffraction pattern for {label}")
    if not np.isfinite(intensities).all() or np.any(intensities < 0):
        raise ValueError(f"Intensities for {label!r} must be finite and non-negative")
    return pattern


def xrd_pattern(
    patterns: PatternOrStruct | Mapping[Any, PatternOrStruct],
    *,
    wavelength: float = cu_k_alpha_wavelength,
    **kwargs: Any,
) -> XrdWidget:
    """Plot XRD patterns of DiffractionPatterns, pymatgen Structures or ASE Atoms.

    Patterns of several inputs are overlaid with a legend and scaled to a shared
    maximum of 100. MatterViz labels each pattern's 5 strongest peaks with their
    Miller indices (and 2θ angles for up to 2 patterns).

    Args:
        patterns (PatternOrStruct | Mapping[str, PatternOrStruct]): A single
            DiffractionPattern, Structure or Atoms, or a mapping from legend labels
            to them.
        wavelength (float): X-ray wavelength in Å used to compute patterns of
            structures. Defaults to 1.54184 (Cu K-alpha).
        **kwargs: Passed to XrdWidget, e.g. ``style`` or ``controls_open``.

    Raises:
        ValueError: If patterns are empty, an intensity is negative or not finite,
            or no intensity is positive.
        TypeError: If patterns is not a DiffractionPattern, Structure, ASE Atoms,
            or a mapping of them.

    Returns:
        XrdWidget: The MatterViz XRD plot.
    """
    is_single = not isinstance(patterns, Mapping)
    labeled = {"XRD Pattern": patterns} if is_single else patterns
    if not labeled:
        raise ValueError("patterns must not be empty")
    computed = {
        str(label): _to_diffraction_pattern(str(label), value, wavelength)
        for label, value in labeled.items()
    }
    if len(computed) != len(labeled):
        raise ValueError(f"Labels must be unique as strings, got {list(labeled)}")
    if not any(np.any(pattern.y > 0) for pattern in computed.values()):
        raise ValueError("At least one diffraction intensity must be positive")
    return XrdWidget(
        patterns=computed.popitem()[1] if is_single else computed, **kwargs
    )
