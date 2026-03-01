"""Normalization functions to convert pymatgen objects to JSON-serializable dicts
for matterviz Svelte components.
"""

from __future__ import annotations

import math
from typing import Any


def _normalize_ferrox_hkls(hkls_data: Any) -> Any:
    """Normalize Ferrox HKL payload into widget-compatible shape.

    Args:
        hkls_data: HKL payload from Ferrox-style dict input.

    Returns:
        Normalized HKL payload where flat or grouped Miller indices are converted
        to `[[{"hkl": [h, k, l]}], ...]`. Unrecognized structures are returned
        unchanged.
    """
    if not isinstance(hkls_data, list):
        return hkls_data
    if hkls_data and all(
        isinstance(hkl_entry, list)
        and len(hkl_entry) == 3
        and all(isinstance(value, int) for value in hkl_entry)
        for hkl_entry in hkls_data
    ):
        return [[{"hkl": hkl_entry}] for hkl_entry in hkls_data]
    if hkls_data and all(
        isinstance(hkl_group, list)
        and hkl_group
        and isinstance(hkl_group[0], list)
        and len(hkl_group[0]) == 3
        and all(isinstance(value, int) for value in hkl_group[0])
        for hkl_group in hkls_data
    ):
        return [[{"hkl": hkl_group[0]}] for hkl_group in hkls_data]
    return hkls_data


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


def _normalize_numeric_sequence(
    values: list[Any], field_label: str, *, series_index: int
) -> list[float]:
    """Normalize a numeric sequence into finite floats.

    Args:
        values: Candidate numeric sequence.
        field_label: Field name for error context (usually ``x`` or ``y``).
        series_index: Zero-based series index for actionable error messages.

    Returns:
        List of finite floats.

    Raises:
        TypeError: If a value is not numeric.
        ValueError: If a numeric value is non-finite (NaN/inf).
    """
    normalized_values: list[float] = []
    for value_index, value in enumerate(values):
        if not isinstance(value, (int, float)):
            raise TypeError(
                "Plot series values must be numeric. "
                f"Got {field_label}[{value_index}]={value!r} in series index "
                f"{series_index}."
            )
        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(
                "Plot series values must be finite numbers. "
                f"Got {field_label}[{value_index}]={value!r} in series index "
                f"{series_index}."
            )
        normalized_values.append(numeric_value)
    return normalized_values


def normalize_plot_json(value: Any, label: str) -> Any:
    """Recursively normalize plot config values to JSON-safe Python primitives.

    Args:
        value: Arbitrary Python value used in widget props.
        label: Human-readable label included in error messages.

    Returns:
        JSON-safe value composed of dict/list/str/bool/int/float/None.

    Raises:
        TypeError: If value cannot be serialized to JSON-safe primitives.
        ValueError: If value contains non-finite numbers.
    """
    if value is None or isinstance(value, (bool, str)):
        return value

    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains non-finite float value: {value!r}.")
        return value

    if hasattr(value, "item") and callable(value.item):
        try:
            scalar_value = value.item()
        except (TypeError, ValueError):
            scalar_value = None
        if scalar_value is not None:
            return normalize_plot_json(scalar_value, label)

    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            list_value = value.tolist()
        except (TypeError, ValueError):
            list_value = None
        if list_value is not None:
            return normalize_plot_json(list_value, label)

    if isinstance(value, dict):
        return {
            str(key): normalize_plot_json(entry, f"{label}.{key}")
            for key, entry in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [normalize_plot_json(entry, label) for entry in value]

    if hasattr(value, "as_dict") and callable(value.as_dict):
        return normalize_plot_json(value.as_dict(), label)

    raise TypeError(
        f"Unsupported value in {label}: {type(value)}. "
        "Expected JSON-serializable primitives, lists, dicts, numpy arrays, or "
        "MSONable objects."
    )


def normalize_plot_series(
    series_data: Any, *, component_name: str
) -> list[dict[str, Any]] | None:
    """Normalize and validate generic plot series payloads.

    Args:
        series_data: Sequence of series dicts or None.
        component_name: Widget/component name used in error messages.

    Returns:
        Normalized list of series dictionaries or ``None``.

    Raises:
        TypeError: If input type or series item types are invalid.
        ValueError: If required fields are missing, malformed, or length-mismatched.
    """
    if series_data is None:
        return None
    if not isinstance(series_data, (list, tuple)):
        raise TypeError(
            f"{component_name} 'series' must be a list/tuple of dicts, got "
            f"{type(series_data)}."
        )

    normalized_series: list[dict[str, Any]] = []
    for series_index, series_entry in enumerate(series_data):
        if not isinstance(series_entry, dict):
            raise TypeError(
                f"{component_name} series entries must be dicts. "
                f"Got type {type(series_entry)} at index {series_index}."
            )
        if "x" not in series_entry or "y" not in series_entry:
            available_keys = sorted(str(key) for key in series_entry)
            raise ValueError(
                f"{component_name} series entry must include keys 'x' and 'y'. "
                f"Got keys at index {series_index}: {available_keys}."
            )

        normalized_entry = normalize_plot_json(series_entry, f"{component_name}.series")
        x_values = normalized_entry["x"]
        y_values = normalized_entry["y"]
        if not isinstance(x_values, list) or not isinstance(y_values, list):
            raise TypeError(
                f"{component_name} series x/y must be list-like after normalization. "
                f"Got types x={type(x_values)}, y={type(y_values)} at index "
                f"{series_index}."
            )
        if len(x_values) != len(y_values):
            raise ValueError(
                f"{component_name} series x/y lengths must match at index "
                f"{series_index}, got len(x)={len(x_values)} and "
                f"len(y)={len(y_values)}."
            )

        normalized_entry["x"] = _normalize_numeric_sequence(
            x_values, "x", series_index=series_index
        )
        normalized_entry["y"] = _normalize_numeric_sequence(
            y_values, "y", series_index=series_index
        )
        normalized_series.append(normalized_entry)

    return normalized_series


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
        has_canonical_keys = all(key in obj for key in ("x", "y"))
        has_ferrox_keys = all(key in obj for key in ("two_theta", "intensities"))
        has_partial_canonical = any(key in obj for key in ("x", "y"))
        has_partial_ferrox = any(key in obj for key in ("two_theta", "intensities"))

        if has_canonical_keys:
            if len(obj["x"]) != len(obj["y"]):
                raise ValueError(
                    "XRD pattern dict has mismatched canonical lengths: "
                    f"len(x)={len(obj['x'])}, len(y)={len(obj['y'])}."
                )
            return obj

        if has_ferrox_keys:
            if len(obj["two_theta"]) != len(obj["intensities"]):
                raise ValueError(
                    "XRD pattern dict has mismatched Ferrox lengths: "
                    f"len(two_theta)={len(obj['two_theta'])}, "
                    f"len(intensities)={len(obj['intensities'])}."
                )

            normalized: dict[str, Any] = {
                "x": obj["two_theta"],
                "y": obj["intensities"],
            }
            if "d_spacings" in obj:
                normalized["d_hkls"] = obj["d_spacings"]

            if "hkls" in obj:
                normalized["hkls"] = _normalize_ferrox_hkls(obj["hkls"])

            return normalized

        if has_partial_canonical:
            missing_keys = [key for key in ("x", "y") if key not in obj]
            raise ValueError(
                "XRD pattern dict missing required key(s) for canonical schema: "
                f"{missing_keys}. Expected keys: ['x', 'y']."
            )

        if has_partial_ferrox:
            missing_keys = [
                key for key in ("two_theta", "intensities") if key not in obj
            ]
            raise ValueError(
                "XRD pattern dict missing required key(s) for Ferrox schema: "
                f"{missing_keys}. Expected keys: ['two_theta', 'intensities']."
            )

        available_keys = sorted(str(key) for key in obj)
        raise ValueError(
            "Unsupported XRD dict schema. Expected either canonical keys "
            "['x', 'y'] or Ferrox keys ['two_theta', 'intensities'], but got "
            f"keys: {available_keys}."
        )

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
