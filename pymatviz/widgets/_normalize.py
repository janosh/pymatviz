"""Normalization functions to convert pymatgen objects to JSON-serializable dicts
for matterviz Svelte components.
"""

from __future__ import annotations

from typing import Any


def _to_dict(obj: Any, label: str) -> dict[str, Any] | None:
    """Convert None/dict/MSONable object to a JSON-serializable dict.

    Args:
        obj: None, a dict (passthrough), or any object with .as_dict().
        label: Human-readable name for error messages (e.g. "band structure").

    Returns:
        The dict, or None if obj is None.

    Raises:
        TypeError: If obj has no .as_dict() method.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    raise TypeError(
        f"Unsupported type for {label}: {type(obj)}. "
        "Expected dict or object with .as_dict()."
    )


def normalize_structure_for_bz(obj: Any) -> dict[str, Any] | None:
    """Normalize a pymatgen Structure or ASE Atoms to a dict for BZ visualization."""
    if obj is None or isinstance(obj, dict) or hasattr(obj, "as_dict"):
        return _to_dict(obj, "Brillouin zone structure")

    # Handle ASE Atoms (no .as_dict but has .get_chemical_symbols)
    if hasattr(obj, "get_chemical_symbols"):
        from pymatviz.process_data import normalize_structures

        return next(iter(normalize_structures(obj).values())).as_dict()

    raise TypeError(
        f"Unsupported type for Brillouin zone structure: {type(obj)}. "
        "Expected dict, pymatgen Structure, or ASE Atoms."
    )


def normalize_convex_hull_entries(obj: Any) -> list[dict[str, Any]] | None:
    """Convert a pymatgen PhaseDiagram or list of entry dicts to convex hull entries.

    Args:
        obj: A pymatgen PhaseDiagram, a list of dicts, or None.

    Returns:
        List of entry dicts with composition, energy, and stability info, or None.
    """
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)):
        return obj if isinstance(obj, list) else list(obj)

    try:
        from pymatgen.analysis.phase_diagram import PhaseDiagram

        if isinstance(obj, PhaseDiagram):
            entries = []
            for entry in obj.all_entries:
                entry_dict: dict[str, Any] = {
                    "composition": entry.composition.as_dict(),
                    "energy": entry.energy,
                    "energy_per_atom": entry.energy_per_atom,
                }
                try:
                    entry_dict["e_above_hull"] = obj.get_e_above_hull(entry)
                    entry_dict["is_stable"] = entry in obj.stable_entries
                except (ValueError, KeyError):
                    pass
                entries.append(entry_dict)
            return entries
    except ImportError:
        pass

    raise TypeError(
        f"Unsupported type for convex hull entries: {type(obj)}. "
        "Expected list of dicts or pymatgen PhaseDiagram."
    )


def normalize_xrd_pattern(obj: Any) -> dict[str, Any] | None:
    """Convert a pymatgen DiffractionPattern to a JSON-serializable dict.

    Args:
        obj: A pymatgen DiffractionPattern, a dict, or None.

    Returns:
        XRD pattern dict with x, y, hkls, d_hkls keys, or None.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj

    try:
        from pymatgen.analysis.diffraction.xrd import DiffractionPattern

        if isinstance(obj, DiffractionPattern):
            result: dict[str, Any] = {"x": obj.x.tolist(), "y": obj.y.tolist()}
            if obj.hkls is not None:
                result["hkls"] = [
                    [
                        {
                            key: list(val) if key == "hkl" else val
                            for key, val in hkl_entry.items()
                        }
                        for hkl_entry in hkl_list
                    ]
                    for hkl_list in obj.hkls
                ]
            if obj.d_hkls is not None:
                result["d_hkls"] = (
                    obj.d_hkls.tolist()
                    if hasattr(obj.d_hkls, "tolist")
                    else list(obj.d_hkls)
                )
            return result
    except ImportError:
        pass

    raise TypeError(
        f"Unsupported type for XRD pattern: {type(obj)}. "
        "Expected dict or pymatgen DiffractionPattern."
    )
