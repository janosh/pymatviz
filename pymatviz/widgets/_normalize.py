"""Normalization functions to convert pymatgen objects to JSON-serializable dicts
for matterviz Svelte components.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any, Literal


if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd


def with_defaults(defaults: dict[str, Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return widget kwargs over a plot function's defaults, merging dict-valued
    options one level deep, so ``x_axis={"scale_type": "log"}`` keeps the default
    axis label instead of replacing the whole dict. Any other user value (including
    None) replaces its default.
    """
    return (
        defaults
        | kwargs
        | {
            key: default | kwargs[key]
            for key, default in defaults.items()
            if isinstance(default, dict) and isinstance(kwargs.get(key), dict)
        }
    )


def _normalize_ferrox_hkls(hkls_data: Any) -> Any:
    """Normalize Ferrox HKL payload into widget-compatible shape.

    Args:
        hkls_data: HKL payload from Ferrox-style dict input.

    Returns:
        Normalized HKL payload where flat or grouped Miller indices are converted
        to `[[{"hkl": [h, k, l]}], ...]`. Unrecognized structures are returned
        unchanged.
    """
    if not isinstance(hkls_data, list) or not hkls_data:
        return hkls_data

    def is_miller(value: Any) -> bool:
        """Whether value is a ``[h, k, l]`` list of ints."""
        return (
            isinstance(value, list)
            and len(value) == 3
            and all(isinstance(idx, int) for idx in value)
        )

    if all(is_miller(hkl_entry) for hkl_entry in hkls_data):
        return [[{"hkl": hkl_entry}] for hkl_entry in hkls_data]
    if all(
        isinstance(hkl_group, list) and hkl_group and is_miller(hkl_group[0])
        for hkl_group in hkls_data
    ):
        return [[{"hkl": hkl_group[0]}] for hkl_group in hkls_data]
    return hkls_data


# pymatgen as_dict() fields MatterViz never draws, up to ~100 MB on projected band
# structures: band projections, CompleteDos.pdos, phonon eigendisplacements
_UNREAD_KEYS = frozenset({"projections", "pdos", "eigendisplacements"})


# Empty stand-ins set on a shallow copy so as_dict() never builds those lists
_UNREAD_ATTRS = {
    "projections": {},
    "pdos": {},
    "eigendisplacements": (),
    "nac_eigendisplacements": [],
}


def _as_dict(obj: Any) -> dict[str, Any]:
    """``obj.as_dict()`` without the fields MatterViz never reads, serialized from a
    shallow copy with those fields emptied (the input is not mutated). Emptying them
    also drops what pymatgen derives from them: ``atom_dos``/``spd_dos`` of a
    CompleteDos and the ``structure`` of a projected band structure.
    """
    import copy

    view = copy.copy(obj)
    for attr, empty in _UNREAD_ATTRS.items():
        if attr in vars(view):
            setattr(view, attr, empty)
    return {key: val for key, val in view.as_dict().items() if key not in _UNREAD_KEYS}


# phonopy objects matscikit.interop converts to PhononBands / PhononDos
_PHONOPY_BANDS = ("phonopy.phonon.band_structure.BandStructure",)
_PHONOPY_DOS = ("phonopy.phonon.dos.TotalDos", "phonopy.api_phonopy.Phonopy")


def labeled_keys(data: dict[str, Any] | None) -> set[str]:
    """Legend labels of a converted ``{label: obj_dict}`` mapping (all values are
    dicts), ``{""}`` for a single object's dict, which also holds arrays.
    """
    if not data or not all(isinstance(val, dict) for val in data.values()):
        return {""}
    return set(data)


def check_stack_grids(doses: dict[str, Any] | None) -> None:
    """Raise ValueError unless all labeled DOS share one frequency or energy grid,
    since MatterViz stacks curves by index.
    """
    if doses is None or labeled_keys(doses) == {""}:
        return
    import numpy as np

    grids = {
        label: dos.get("frequencies", dos.get("energies"))
        for label, dos in doses.items()
    }
    first_label, first_grid = next(iter(grids.items()))
    # array_equal so list and tuple grids with equal values match
    if mismatched := [
        label for label, grid in grids.items() if not np.array_equal(grid, first_grid)
    ]:
        raise ValueError(
            f"stack=True needs all DOS on one grid, but {mismatched} differ from "
            f"{first_label!r}"
        )


def _object_dict(obj: Any) -> dict[str, Any] | None:
    """Dict of a structure-like, matscikit/phonopy phonon, matscikit electronic or
    MSONable object (pymatgen's ``as_dict()`` layout), None for others.
    """
    import matscikit as mk
    from matscikit.electronic import ElectronicBands, ElectronicDos
    from matscikit.interop import has_class, to_phonon_bands, to_phonon_dos

    from pymatviz.structures import is_structure_like, structure_dict

    if is_structure_like(obj):
        return structure_dict(obj)
    if has_class(obj, *_PHONOPY_BANDS):
        obj = to_phonon_bands(obj)
    elif has_class(obj, *_PHONOPY_DOS):
        obj = to_phonon_dos(obj)
    if isinstance(obj, (mk.PhononBands, mk.PhononDos, ElectronicBands, ElectronicDos)):
        return {
            key: val for key, val in obj.as_dict().items() if key not in _UNREAD_KEYS
        }
    return _as_dict(obj) if callable(getattr(obj, "as_dict", None)) else None


_VASPRUN_SUFFIXES = (".xml", ".xml.gz", ".xml.xz", ".xml.bz2")


def _vasprun_part(obj: Any, part: Literal["bands", "dos"] | None) -> Any:
    """The bands or DOS (matscikit ElectronicBands / ElectronicDos) of a vasprun.xml
    path (str or PathLike, optionally compressed) when ``part`` is set, else obj.
    """
    import os

    is_path = isinstance(obj, os.PathLike) or (
        isinstance(obj, str) and obj.endswith(_VASPRUN_SUFFIXES)
    )
    if part is None or not is_path:
        return obj
    from matscikit.io import read_vasprun

    return getattr(read_vasprun(os.fspath(obj)), part)


def _to_dict(
    obj: Any, label: str, *, vasprun: Literal["bands", "dos"] | None = None
) -> dict[str, Any] | None:
    """Convert None/dict/MSONable object to a JSON-serializable dict.

    Args:
        obj: None, a dict (MSONable values converted via ``as_dict``), or any
            object with .as_dict(). Structure-like objects (matscikit, pymatgen, ASE,
            phonopy) become structure dicts, phonopy BandStructure, TotalDos and
            Phonopy objects go through matscikit's PhononBands / PhononDos. Converted
            dicts drop ``_UNREAD_KEYS``.
        label: Human-readable name for error messages (e.g. "band structure").
        vasprun: Read vasprun.xml paths (obj or dict values) for this part of the
            run, "bands" or "dos". None leaves paths unconverted.

    Returns:
        The dict, or None if obj is None.

    Raises:
        TypeError: If obj has no .as_dict() method.
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        converted = {
            key: dct
            for key, val in obj.items()
            if not isinstance(val, dict)
            and (dct := _object_dict(_vasprun_part(val, vasprun))) is not None
        }
        return {**obj, **converted} if converted else obj
    if (dct := _object_dict(_vasprun_part(obj, vasprun))) is not None:
        return dct
    raise TypeError(
        f"Unsupported type for {label}: {type(obj)}. "
        "Expected dict, vasprun.xml path or object with .as_dict()."
    )


def _normalize_numeric_sequence(
    values: list[Any], field_label: str, *, series_index: int
) -> list[float]:
    """Convert JSON-normalized series values to floats, rejecting non-numeric items."""
    normalized_values: list[float] = []
    for value_index, value in enumerate(values):
        if not isinstance(value, (int, float)):
            raise TypeError(
                "Plot series values must be numeric. "
                f"Got {field_label}[{value_index}]={value!r} in series index "
                f"{series_index}."
            )
        normalized_values.append(float(value))
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
        return int(value)

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains non-finite float value: {value!r}.")
        return float(value)

    # Convert numpy/pandas arrays (.tolist()) and numpy scalars (.item()) to native
    # Python. tolist first: a 1-element array also has .item(), which would collapse
    # [0.5] to 0.5; numpy scalars' .tolist() returns the Python scalar anyway.
    for method_name in ("tolist", "item"):
        method = getattr(value, method_name, None)
        if not callable(method):
            continue
        try:
            converted = method()
        except (TypeError, ValueError):
            continue
        return normalize_plot_json(converted, label)

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


def normalize_json_kwargs(
    kwargs: dict[str, Any], keys: Sequence[str], component: str
) -> None:
    """In place, run normalize_plot_json on each of ``keys`` present in ``kwargs``.

    Args:
        kwargs: Widget constructor keyword arguments, modified in place.
        keys: Names of the config kwargs to normalize (e.g. numpy bounds in dicts).
        component: MatterViz component name used to label error messages.
    """
    for key in keys:
        if key in kwargs:
            kwargs[key] = normalize_plot_json(kwargs[key], f"{component}.{key}")


def check_metadata_frame(frame: pd.DataFrame, n_points: int, label: str) -> None:
    """Validate a per-point metadata DataFrame: one row per point and at least one,
    uniquely named column (a duplicate name would select a 2D block). Names must
    stay unique as the strings they sync as (columns 1 and "1" would merge).

    Raises:
        ValueError: For a wrong row count, no columns or duplicate column names.
    """
    if len(frame) != n_points:
        raise ValueError(
            f"{label}.metadata DataFrame needs one row per point ({n_points}), got "
            f"{len(frame)}"
        )
    if frame.columns.empty or frame.columns.map(str).has_duplicates:
        raise ValueError(
            f"{label}.metadata DataFrame needs uniquely named columns, got "
            f"{frame.columns.tolist()}"
        )


def metadata_records(
    frame: pd.DataFrame, n_points: int, label: str
) -> list[dict[str, Any]]:
    """Per-point metadata dicts from a DataFrame with one row per point. Missing and
    non-finite numbers become None, as JSON metadata carries them. Values must be
    JSON-serializable (e.g. no timestamps).

    Raises:
        ValueError: If check_metadata_frame rejects the frame.
    """
    import numpy as np

    check_metadata_frame(frame, n_points, label)
    missing = frame.isna()
    numeric = frame.select_dtypes("number").columns
    missing[numeric] |= ~np.isfinite(frame[numeric].to_numpy(dtype=float))
    for col in frame.select_dtypes(object).columns:  # e.g. [inf, "unavailable"]
        missing[col] |= frame[col].map(
            lambda val: isinstance(val, float | np.floating) and not np.isfinite(val)
        )
    return frame.astype(object).mask(missing, None).to_dict("records")


def normalize_plot_series(
    series_data: Any,
    *,
    component_name: str,
    samples_key: str | None = None,
    binary: bool = False,
) -> list[dict[str, Any]] | None:
    """Normalize and validate plot series payloads (``None`` passes through).

    Entries are dicts with ``x`` and ``y`` lists of equal length (``x`` all strings
    for categorical axes or all numeric, ``y`` numeric, both finite). Per-point
    ``metadata`` may be a pandas DataFrame with one row per point (columnar, so
    binary series send numeric columns as arrays). With
    ``samples_key`` (histograms bin ``values``), entries must instead hold raw
    samples under that key. Raises TypeError for wrong container/entry/sample types and
    ValueError for missing keys, length mismatches and non-finite numbers.
    """
    if series_data is None:
        return None
    if not isinstance(series_data, (list, tuple)):
        raise TypeError(
            f"{component_name} 'series' must be a list/tuple of dicts, got "
            f"{type(series_data)}."
        )
    import pandas as pd

    # lazy: _binary imports this module
    from pymatviz.widgets._binary import is_binary_envelope, plot_series_envelope

    normalized_series: list[dict[str, Any]] = []
    keys = (samples_key,) if samples_key else ("x", "y")
    for series_index, series_entry in enumerate(series_data):
        # with binary=True, large numeric series (and already encoded ones) skip the
        # per-value JSON conversion
        if binary and is_binary_envelope(series_entry, "plot-series"):
            normalized_series.append(series_entry)
            continue
        name = f"{component_name}.series[{series_index}]"
        if (
            binary
            and isinstance(series_entry, dict)
            and (envelope := plot_series_envelope(series_entry, name, keys)) is not None
        ):
            normalized_series.append(envelope)
            continue
        if not isinstance(series_entry, dict):
            raise TypeError(
                f"{component_name} series entries must be dicts. "
                f"Got type {type(series_entry)} at index {series_index}."
            )
        raw_entry = series_entry
        if isinstance(meta := series_entry.get("metadata"), pd.DataFrame):
            n_points = len(series_entry.get(keys[0], ()))
            records = metadata_records(meta, n_points, name)
            raw_entry = {**series_entry, "metadata": records}
        entry = normalize_plot_json(raw_entry, f"{component_name}.series")
        if not all(key in entry for key in keys):
            what = f"key '{samples_key}'" if samples_key else "keys 'x' and 'y'"
            raise ValueError(
                f"{component_name} series entry must include {what}. "
                f"Got keys at index {series_index}: {sorted(entry)}."
            )
        for key in keys:
            if not isinstance(entry[key], list):
                raise TypeError(
                    f"{component_name} series {key} must be list-like after "
                    f"normalization. Got {type(entry[key])} at index {series_index}."
                )
        if len(keys) == 2 and len(entry["x"]) != len(entry["y"]):
            raise ValueError(
                f"{component_name} series x/y lengths must match at index "
                f"{series_index}, got len(x)={len(entry['x'])} and "
                f"len(y)={len(entry['y'])}."
            )
        for key in keys:
            values = entry[key]
            # x can be strings (categorical) or numeric; samples/y are always numeric
            if key == "x" and values and all(isinstance(val, str) for val in values):
                continue
            if key == "x" and any(isinstance(val, str) for val in values):
                raise TypeError(
                    f"{component_name} series x values must be all strings "
                    f"(categorical) or all numeric, got mixed types at index "
                    f"{series_index}."
                )
            entry[key] = _normalize_numeric_sequence(
                values, key, series_index=series_index
            )
        normalized_series.append(entry)
    return normalized_series


def structure_to_dict(
    structure: Any, *, periodic: bool = False
) -> dict[str, Any] | None:
    """Convert one structure-like object to the dict structure widgets sync to JS.

    Args:
        structure: ``None`` (passed through), a structure dict with a ``sites`` list
            (passed through unchanged, schema errors are reported by the renderer),
            or a single pymatgen ``Structure``/``Molecule``, ASE ``Atoms`` or
            ``PhonopyAtoms`` (converted via ``structure_dict``).
        periodic: Require a lattice, for widgets drawing the Brillouin zone.

    Returns:
        The structure dict, or None if structure is None.

    Raises:
        TypeError: If a dict has no ``sites`` list, for any other type (including
            sequences of structures), or for a lattice-less one when ``periodic``.
    """
    if structure is None:
        return None
    if isinstance(structure, dict):
        if not isinstance(structure.get("sites"), list):
            raise TypeError(
                "structure dicts need a 'sites' list (e.g. from Structure.as_dict()), "
                f"got keys={sorted(structure)}"
            )
        struct = structure
    else:
        from pymatviz.structures import is_structure_like, structure_dict

        if not is_structure_like(structure):
            raise TypeError(
                "structure must be a dict, matscikit or pymatgen Structure/Molecule, "
                f"ASE Atoms or PhonopyAtoms, got {type(structure).__name__}"
            )
        struct = structure_dict(structure)
    if periodic and "lattice" not in struct:
        raise TypeError(
            f"a periodic structure (with a lattice) is needed, got {sorted(struct)=}"
        )
    return struct


def _parse_formula_to_dict(formula: str) -> dict[str, float]:
    """Parse a chemical formula string into an element->amount dict.

    Handles simple formulas like "Li2O", "Fe2O3", "LiFePO4".
    For formulas that are just an element symbol (e.g. "Li"), returns {"Li": 1}.

    Args:
        formula: Chemical formula string.

    Returns:
        Dict mapping element symbols to their counts.
    """
    if not formula.strip():
        raise ValueError("Formula must not be empty.")
    if any(char in formula for char in "()[]{}"):
        raise ValueError(f"Unsupported grouping in formula: {formula!r}.")

    composition: dict[str, float] = {}
    matches = list(re.finditer(r"([A-Z][a-z]?)(\d*\.?\d*)", formula))
    if "".join(match.group(0) for match in matches) != formula:
        raise ValueError(f"Invalid formula syntax: {formula!r}.")
    for match in matches:
        element = match.group(1)
        count_str = match.group(2)
        count = float(count_str) if count_str else 1.0
        composition[element] = composition.get(element, 0) + count
    return composition


def _normalize_entry_compositions(
    entries: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Ensure entry compositions are dicts, parsing formula strings as needed.

    The matterviz frontend expects compositions as ``{element: amount}`` dicts.
    This converts any string compositions (e.g. ``"Li2O"``) to that format,
    leaving dict compositions unchanged.
    """
    return [
        {**entry, "composition": _parse_formula_to_dict(entry["composition"])}
        if isinstance(entry.get("composition"), str)
        else entry
        for entry in entries
    ]


def normalize_convex_hull_entries(obj: Any) -> list[dict[str, Any]] | None:
    """Convex hull entries (composition, energy, stability) of entry dicts, a matscikit
    or pymatgen PhaseDiagram or an iterable of entries (pymatgen ComputedEntry,
    PDEntry, entry dicts); energies above hull and stability from matscikit's
    PhaseDiagram, total energies as energy per atom times atoms.

    Args:
        obj: Entry dicts (passed through, formula strings parsed), a PhaseDiagram,
            an iterable of entries, or None.

    Returns:
        List of entry dicts with composition, energy, and stability info, or None.

    Raises:
        TypeError: For other objects or entries without composition and energy.
    """
    if obj is None:
        return None
    if isinstance(obj, (list, tuple)) and all(isinstance(entry, dict) for entry in obj):
        return _normalize_entry_compositions(obj)
    import matscikit as mk
    from matscikit.interop import to_phase_diagram

    try:  # materialize generators, keep phase diagrams (all_entries) as they are
        is_diagram = isinstance(obj, mk.PhaseDiagram) or hasattr(obj, "all_entries")
        phase_diagram = to_phase_diagram(obj if is_diagram else list(obj))
    except TypeError as exc:
        raise TypeError(
            f"Unsupported type for convex hull entries: {type(obj)}. Expected list of "
            f"dicts, a PhaseDiagram (matscikit, pymatgen) or entry objects ({exc})."
        ) from exc
    return [
        {
            "composition": composition.as_dict(),
            "energy": energy_per_atom * composition.num_atoms,
            "energy_per_atom": energy_per_atom,
            "e_above_hull": e_above_hull,
            "is_stable": is_stable,
        }
        for composition, energy_per_atom, e_above_hull, is_stable in zip(
            phase_diagram.compositions,
            phase_diagram.energies_per_atom.tolist(),
            phase_diagram.e_above_hull().tolist(),
            phase_diagram.is_stable(),
            strict=True,
        )
    ]


XRD_PATTERN_KINDS = ("sticks", "profile")


def _with_xrd_kind(pattern: dict[str, Any]) -> dict[str, Any]:
    """Tag a pattern without Miller indices as a measured ``kind="profile"`` scan.

    matterviz draws ``profile`` patterns as a line and never broadens them, while
    ``sticks`` (its default) are discrete reflections drawn as bars. Computed patterns
    (pymatgen, Ferrox) carry ``hkls`` and stay untagged; an explicit ``kind`` wins.

    Raises:
        ValueError: If ``pattern["kind"]`` is not one of ``XRD_PATTERN_KINDS``.
    """
    if "kind" in pattern:
        if pattern["kind"] not in XRD_PATTERN_KINDS:
            raise ValueError(
                f"XRD pattern kind={pattern['kind']!r} must be one of "
                f"{list(XRD_PATTERN_KINDS)}."
            )
        return pattern
    if pattern.get("hkls"):
        return pattern
    return {**pattern, "kind": "profile"}


def normalize_xrd_pattern(obj: Any) -> dict[str, Any] | None:
    """Convert an XRD pattern to a JSON-serializable dict for matterviz.

    Patterns without ``hkls`` (measured scans) get ``kind="profile"`` so matterviz
    draws them as a continuous line and skips peak broadening.

    Args:
        obj: A pymatgen DiffractionPattern, a dict, or None.

    Returns:
        XRD pattern dict with x, y, and optional hkls, d_hkls, kind keys, or None.
    """
    if obj is None:
        return None
    return _with_xrd_kind(_xrd_pattern_to_dict(obj))


def _xrd_pattern_to_dict(obj: Any) -> dict[str, Any]:
    """Convert a canonical/Ferrox dict, matscikit XrdPattern or pymatgen
    DiffractionPattern to a dict.

    Args:
        obj: A matscikit XrdPattern, pymatgen DiffractionPattern or a dict.

    Returns:
        XRD pattern dict with x, y, and optional hkls, d_hkls keys.
    """
    if isinstance(obj, dict):
        if all(key in obj for key in ("x", "y")):
            if len(obj["x"]) != len(obj["y"]):
                raise ValueError(
                    "XRD pattern dict has mismatched canonical lengths: "
                    f"len(x)={len(obj['x'])}, len(y)={len(obj['y'])}."
                )
            return obj

        if all(key in obj for key in ("two_theta", "intensities")):
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
            if "kind" in obj:
                normalized["kind"] = obj["kind"]

            return normalized

        for schema, keys in (
            ("canonical", ["x", "y"]),
            ("Ferrox", ["two_theta", "intensities"]),
        ):
            if any(key in obj for key in keys):
                missing_keys = [key for key in keys if key not in obj]
                raise ValueError(
                    f"XRD pattern dict missing required key(s) for {schema} schema: "
                    f"{missing_keys}. Expected keys: {keys}."
                )

        available_keys = sorted(str(key) for key in obj)
        raise ValueError(
            "Unsupported XRD dict schema. Expected either canonical keys "
            "['x', 'y'] or Ferrox keys ['two_theta', 'intensities'], but got "
            f"keys: {available_keys}."
        )

    import matscikit as mk
    from matscikit.interop import has_class, to_xrd_pattern

    if has_class(obj, "pymatgen.analysis.diffraction.core.DiffractionPattern"):
        obj = to_xrd_pattern(obj)
    if isinstance(obj, mk.XrdPattern):
        result: dict[str, Any] = {
            "x": obj.two_theta.tolist(),
            "y": obj.intensities.tolist(),
        }
        if obj.hkls is not None:
            result["hkls"] = [
                [{"hkl": list(hkl), "multiplicity": mult} for hkl, mult in family]
                for family in obj.hkls
            ]
        if obj.d_spacings is not None:
            result["d_hkls"] = obj.d_spacings.tolist()
        return result

    raise TypeError(
        f"Unsupported type for XRD pattern: {type(obj)}. "
        "Expected dict, matscikit XrdPattern or pymatgen DiffractionPattern."
    )
