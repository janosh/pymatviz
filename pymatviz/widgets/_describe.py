"""Structured, machine-parseable summaries of widget state (no browser).

``describe_widget`` returns a dict of facts derived purely from a widget's
synced traitlets -- useful for agents to reason about output, and as the basis
for ``to_html`` titles and ``render_report`` summaries. ``check_inputs`` returns
cheap warnings about empty/degenerate data before a (slow) headless render.
"""

from __future__ import annotations

import functools
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

from pymatviz.utils.data import element_symbols
from pymatviz.widgets._binary import is_binary_envelope, series_column


def _minmax(values: Any) -> list[float] | None:
    """Find finite extrema in nested lists/tuples/mappings/arrays, ignoring bools."""
    minimum, maximum = math.inf, -math.inf
    stack = [values]
    while stack:
        item = stack.pop()
        if isinstance(item, bool):
            continue
        if isinstance(item, (int, float)):
            try:  # a huge int overflows float(); skip rather than raise
                num = float(item)
            except OverflowError:
                continue
            if math.isfinite(num):
                minimum = min(minimum, num)
                maximum = max(maximum, num)
        elif isinstance(item, np.ndarray) and item.dtype.kind in "iuf":
            if (finite := item[np.isfinite(item)]).size:
                minimum = min(minimum, float(finite.min()))
                maximum = max(maximum, float(finite.max()))
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, Mapping):
            stack.extend(item.values())
    return [minimum, maximum] if minimum != math.inf else None


def _fmt_amount(amount: float) -> str:
    """Omit unit amounts and keep summed fractional occupancies readable."""
    text = f"{amount:.12g}" if isinstance(amount, float) else str(amount)
    return "" if text == "1" else text


def _composition_facts(amounts: Mapping[str, float]) -> dict[str, Any]:
    """Sorted elements and alphabetical formula, omitting unit amounts."""
    elements = sorted(amounts)
    return {
        "elements": elements,
        "formula": "".join(f"{el}{_fmt_amount(amounts[el])}" for el in elements),
    }


def _species_facts(species: Iterable[Any], n_sites: int) -> dict[str, Any]:
    """Sum occupancies across site species in structure and trajectory JSON."""
    counts: defaultdict[str, float] = defaultdict(float)
    for entry in species:
        if isinstance(entry, str):
            counts[entry] += 1
        elif isinstance(entry, Mapping) and isinstance(
            element := entry.get("element"), str
        ):
            occu = entry.get("occu")
            counts[element] += 1 if occu is None else occu
    return {"n_sites": n_sites, **(_composition_facts(counts) if counts else {})}


def _structure_facts(struct: Any) -> dict[str, Any]:
    """Element counts, formula, and site count from a serialized structure dict."""
    if not isinstance(struct, Mapping):
        return {}
    sites = struct.get("sites") or []
    species = (
        entry
        for site in sites
        if isinstance(site, Mapping)
        and isinstance(entries := site.get("species"), list)
        for entry in entries
    )
    return _species_facts(species, len(sites))


def _series_list(widget_data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """The dict-valued entries of a plot widget's ``series`` traitlet."""
    series = widget_data.get("series") or []
    return [series_item for series_item in series if isinstance(series_item, Mapping)]


def _xy_facts(
    widget_data: Mapping[str, Any], axes: tuple[str, ...], *, histogram: bool = False
) -> dict[str, Any]:
    """Series count, point count, per-axis ranges, and labels for plot widgets.

    Histograms bin their samples along x, so those set ``n_points``/``x_range``; bin
    counts are computed in the renderer, so no ``y_range`` is reported.
    """
    series = _series_list(widget_data)
    samples = {
        axis: [series_column(s, "values" if histogram else axis) for s in series]
        for axis in axes
    }
    facts: dict[str, Any] = {"n_series": len(series)}
    facts["n_points"] = sum(0 if vals is None else len(vals) for vals in samples["x"])
    if labels := [s.get("label") for s in series if s.get("label")]:
        facts["series_labels"] = labels
    for axis, axis_samples in samples.items():
        if (axis_range := _minmax(axis_samples)) is not None:
            facts[f"{axis}_range"] = axis_range
    return facts


def _entries_facts(widget_data: Mapping[str, Any]) -> dict[str, Any]:
    """Entry count and chemical system for convex-hull / chem-pot widgets."""
    entries = widget_data.get("entries") or []
    elements: set[str] = set()
    for entry in entries:
        comp = entry.get("composition") if isinstance(entry, Mapping) else None
        if isinstance(comp, Mapping):
            elements |= {str(key) for key in comp}
    facts: dict[str, Any] = {"n_entries": len(entries)}
    if elements:
        facts["chemical_system"] = sorted(elements)
    return facts


def _source_flags(
    widget_data: Mapping[str, Any], keys: tuple[str, ...]
) -> dict[str, Any]:
    """Report which optional data sources are populated (e.g. structures/patterns)."""
    return {f"has_{key}": widget_data.get(key) is not None for key in keys}


def _describe_structure(data: Mapping[str, Any]) -> dict[str, Any]:
    """Facts for a single structure."""
    return _structure_facts(data.get("structure"))


def _describe_trajectory(data: Mapping[str, Any]) -> dict[str, Any]:
    """Frame count plus first-frame structure facts."""
    traj = data.get("trajectory")
    if not isinstance(traj, Mapping):
        return {"n_frames": 0}
    if is_binary_envelope(traj, "trajectory"):
        arrays = traj.get("arrays") or {}
        if missing := sorted({"positions", "numbers"} - set(arrays)):
            raise ValueError(
                f"binary trajectory envelope lacks arrays {missing}, has "
                f"{sorted(arrays)}"
            )
        positions, numbers = arrays["positions"], arrays["numbers"]
        atomic_numbers = np.frombuffer(
            numbers["data"], dtype=np.dtype(numbers["dtype"]).newbyteorder("<")
        )
        counts = {
            element_symbols[int(num)]: count
            for num, count in Counter(atomic_numbers.tolist()).items()
        }
        return {
            "n_frames": positions["shape"][0],
            "n_sites": len(atomic_numbers),
            **_composition_facts(counts),
        }
    if traj.get("@class") == "Trajectory":
        species = traj.get("species") or []
        coords = traj.get("coords", [])
        if isinstance(coords, Mapping) and coords.get("@class") == "array":
            coords = coords["data"]
        return {"n_frames": len(coords), **_species_facts(species, len(species))}
    frames = traj.get("frames") or []
    facts: dict[str, Any] = {"n_frames": len(frames)}
    if frames and isinstance(frames[0], Mapping):
        facts |= _structure_facts(frames[0].get("structure"))
    return facts


def _describe_periodic_table(data: Mapping[str, Any]) -> dict[str, Any]:
    """Element count and value range for a periodic-table heatmap, plus the number
    of elements with tile plots.
    """
    facts: dict[str, Any] = {"n_elements": 0}
    if tile_plots := data.get("tile_plots"):
        facts["n_tile_plots"] = len(tile_plots)
    values = data.get("heatmap_values")
    if not isinstance(values, (Mapping, list, tuple)):
        return facts
    facts["n_elements"] = len(values)
    if (value_range := _minmax(values)) is not None:
        facts["value_range"] = value_range
    return facts


def _describe_heatmap_matrix(data: Mapping[str, Any]) -> dict[str, Any]:
    """Grid shape and value range for a heatmap matrix."""
    facts: dict[str, Any] = {
        "shape": [len(data.get("x_items") or []), len(data.get("y_items") or [])]
    }
    if (value_range := _minmax(data.get("values"))) is not None:
        facts["value_range"] = value_range
    return facts


def _describe_composition(data: Mapping[str, Any]) -> dict[str, Any]:
    """Elements and formula for a composition."""
    comp = data.get("composition")
    if not isinstance(comp, Mapping):
        return {}
    amounts = {
        str(el): amt for el, amt in comp.items() if isinstance(amt, (int, float))
    }
    return _composition_facts(amounts) if amounts else {}


def _payload_sequences(
    payloads: Any, value_key: str, *, spin_keyed: bool = False
) -> list[list[Any]] | None:
    """The ``value_key`` lists of one payload or of a ``{label: payload}`` mapping
    (None if neither). With ``spin_keyed``, a value may also be a ``{spin: list}``
    mapping (pymatgen band structures key ``bands`` by spin), contributing each list.
    """
    if not isinstance(payloads, Mapping):
        return None

    def lists_of(value: Any) -> list[list[Any]] | None:
        """A value's lists: itself if a list, its spin channels' lists, else None."""
        if isinstance(value, (list, tuple)):
            return [value]
        if (
            spin_keyed
            and isinstance(value, Mapping)
            and value
            and all(isinstance(channel, (list, tuple)) for channel in value.values())
        ):
            return list(value.values())
        return None

    if (single := lists_of(payloads.get(value_key))) is not None:
        return single
    sequences = [
        seq
        for payload in payloads.values()
        if isinstance(payload, Mapping)
        and (seqs := lists_of(payload.get(value_key))) is not None
        for seq in seqs
    ]
    return sequences or None


def _describe_xrd(data: Mapping[str, Any]) -> dict[str, Any]:
    """Peak count and 2-theta range for one XRD pattern or ``{label: pattern}``
    overlays (peaks summed over patterns).
    """
    two_thetas = _payload_sequences(data.get("patterns"), "x") or []
    facts: dict[str, Any] = {"n_peaks": sum(map(len, two_thetas))}
    if (two_theta_range := _minmax(two_thetas)) is not None:
        facts["two_theta_range"] = two_theta_range
    return facts


def _describe_spacegroup_bar(data: Mapping[str, Any]) -> dict[str, Any]:
    """Total and unique space-group entry counts."""
    counts = data.get("data") or {}
    return {"n_entries": sum(counts.values()), "n_unique": len(counts)}


def _describe_spectral(
    data: Mapping[str, Any],
    *,
    data_key: str,
    value_key: str,
    count_key: str,
    range_key: str | None = None,
    spin_keyed: bool = False,
) -> dict[str, Any]:
    """Count (and optionally range) a field from one payload or named payloads,
    summing spin channels of ``{spin: list}`` values when ``spin_keyed``.
    """
    sequences = _payload_sequences(data.get(data_key), value_key, spin_keyed=spin_keyed)
    if not sequences:
        return {}
    facts: dict[str, Any] = {count_key: sum(map(len, sequences))}
    if range_key is not None and (value_range := _minmax(sequences)) is not None:
        facts[range_key] = value_range
    return facts


_describe_dos = functools.partial(
    _describe_spectral,
    data_key="doses",
    value_key="energies",
    count_key="n_energies",
    range_key="energy_range",
)
_describe_band_structure = functools.partial(
    _describe_spectral,
    data_key="band_structs",
    value_key="bands",
    count_key="n_bands",
    spin_keyed=True,  # n_bands sums band lines over spin channels
)


def _describe_bands_and_dos(data: Mapping[str, Any]) -> dict[str, Any]:
    """Combined band-structure and DOS facts."""
    return {**_describe_band_structure(data), **_describe_dos(data)}


def _describe_hierarchy(data: Mapping[str, Any]) -> dict[str, Any]:
    """Node/leaf counts and depth for a chart hierarchy."""
    tree = data.get("data")
    roots = tree if isinstance(tree, (list, tuple)) else [tree]
    n_nodes = n_leaves = max_depth = 0
    stack = [(node, 1) for node in roots if isinstance(node, Mapping)]
    seen_ids: set[int] = set()  # cycle guard: post-init trait mutation can
    while stack:  # produce self-referential dicts that would hang the walk
        node, depth = stack.pop()
        if id(node) in seen_ids:
            continue
        seen_ids.add(id(node))
        n_nodes += 1
        max_depth = max(max_depth, depth)
        children = node.get("children")
        children = children if isinstance(children, (list, tuple)) else []
        child_nodes = [child for child in children if isinstance(child, Mapping)]
        if child_nodes:
            stack.extend((child, depth + 1) for child in child_nodes)
        else:  # no valid child dicts (incl. malformed children entries) -> leaf
            n_leaves += 1
    facts: dict[str, Any] = {"n_nodes": n_nodes, "n_leaves": n_leaves}
    if max_depth:
        facts["max_depth"] = max_depth
    return facts


def _describe_brillouin_zone(data: Mapping[str, Any]) -> dict[str, Any]:
    """Data source flags plus structure facts for a Brillouin zone."""
    return {
        **_source_flags(data, ("structure", "bz_data")),
        **_describe_structure(data),
    }


# Facts for 2D scatter/bar plots (3D adds a z-axis range, histograms bin along x)
_describe_xy = functools.partial(_xy_facts, axes=("x", "y"))
_describe_xyz = functools.partial(_xy_facts, axes=("x", "y", "z"))
_describe_histogram = functools.partial(_xy_facts, axes=("x",), histogram=True)

# widget_type -> handler deriving structured facts from to_dict() output
_HANDLERS: dict[str, Any] = {
    "scatter_plot": _describe_xy,
    "bar_plot": _describe_xy,
    "histogram": _describe_histogram,
    "scatter_plot_3d": _describe_xyz,
    "structure": _describe_structure,
    "trajectory": _describe_trajectory,
    "periodic_table": _describe_periodic_table,
    "heatmap_matrix": _describe_heatmap_matrix,
    "convex_hull": _entries_facts,
    "chem_pot_diagram": _entries_facts,
    "composition": _describe_composition,
    "xrd": _describe_xrd,
    "spacegroup_bar": _describe_spacegroup_bar,
    "dos": _describe_dos,
    "band_structure": _describe_band_structure,
    "bands_and_dos": _describe_bands_and_dos,
    "sunburst": _describe_hierarchy,
    "treemap": _describe_hierarchy,
    # which optional data source is populated (structures/patterns, fermi/band)
    "rdf_plot": functools.partial(_source_flags, keys=("structures", "patterns")),
    "fermi_surface": functools.partial(_source_flags, keys=("fermi_data", "band_data")),
    "brillouin_zone": _describe_brillouin_zone,
}

# widget_type -> the primary data traitlet whose emptiness implies a blank render
_PRIMARY_DATA: dict[str, str | tuple[str, ...]] = {
    "scatter_plot": "series",
    "bar_plot": "series",
    "histogram": "series",
    "scatter_plot_3d": "series",
    "structure": "structure",
    "trajectory": "trajectory",
    "periodic_table": ("heatmap_values", "tile_plots"),
    "heatmap_matrix": "values",
    "convex_hull": "entries",
    "chem_pot_diagram": "entries",
    "composition": "composition",
    "xrd": "patterns",
    "spacegroup_bar": "data",
    "dos": "doses",
    "band_structure": "band_structs",
    "sunburst": "data",
    "treemap": "data",
    # can render with either input, so only warn when both are missing
    "bands_and_dos": ("doses", "band_structs"),
}


def describe_widget(widget_data: Mapping[str, Any]) -> dict[str, Any]:
    """Return structured, machine-parseable facts about a widget's data.

    Args:
        widget_data: Output of ``MatterVizWidget.to_dict()``.

    Returns:
        A dict that always contains ``"widget_type"`` plus type-specific facts
        (counts, ranges, formula, ...). Unknown types return just the type.
    """
    widget_type = widget_data.get("widget_type")
    handler = _HANDLERS.get(widget_type) if isinstance(widget_type, str) else None
    facts = handler(widget_data) if handler else {}
    return {"widget_type": widget_type, **facts}


def short_summary(report: Mapping[str, Any]) -> str:
    """Build a one-line human label from a ``describe_widget`` report.

    Used for ``to_html`` titles/meta descriptions.
    """
    widget_type = report.get("widget_type") or "widget"
    if "formula" in report:
        label = report["formula"]
        if "n_sites" in report:
            label += f" ({report['n_sites']} sites)"
        return f"{widget_type}: {label}"
    for key, noun in (
        ("n_series", "series"),
        ("n_frames", "frames"),
        ("n_entries", "entries"),
        ("n_tile_plots", "tile plots"),
        ("n_elements", "elements"),
        ("n_nodes", "nodes"),
    ):
        if key in report:
            return f"{widget_type}: {report[key]} {noun}"
    return str(widget_type)


def check_inputs(widget_data: Mapping[str, Any]) -> list[str]:
    """Return cheap warnings about empty/degenerate data (no browser).

    Flags likely-blank renders before paying for a headless capture.
    """
    warnings: list[str] = []
    widget_type = widget_data.get("widget_type")

    key = _PRIMARY_DATA.get(widget_type) if isinstance(widget_type, str) else None
    if key is not None:
        keys = key if isinstance(key, tuple) else (key,)

        def is_empty(value: Any) -> bool:
            return value is None or (hasattr(value, "__len__") and len(value) == 0)

        if all(is_empty(widget_data.get(k)) for k in keys):
            label = "/".join(f"'{k}'" for k in keys)
            warnings.append(
                f"{widget_type}: {label} is empty or None; widget may render blank"
            )

    if widget_type in ("scatter_plot", "bar_plot", "histogram", "scatter_plot_3d"):
        # Histograms bin `values`; the other plots need `x`.
        hist = widget_type == "histogram"
        series = _series_list(widget_data)
        if series and not any(
            (col := series_column(s, "values" if hist else "x")) is not None
            and len(col)
            for s in series
        ):
            what = "samples ('values')" if hist else "'x'"
            warnings.append(
                f"{widget_type}: all series have empty {what}; nothing to plot"
            )

    if widget_type == "rdf_plot" and not (
        widget_data.get("structures") or widget_data.get("patterns")
    ):
        warnings.append("rdf_plot: neither 'structures' nor 'patterns' provided")

    return warnings
