"""Structured, machine-parseable summaries of widget state (no browser).

``describe_widget`` returns a dict of facts derived purely from a widget's
synced traitlets -- useful for agents to reason about output, and as the basis
for ``to_html`` titles and ``render_report`` summaries. ``check_inputs`` returns
cheap warnings about empty/degenerate data before a (slow) headless render.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from typing import Any


def _numbers(values: Any) -> list[float]:
    """Flatten nested lists/tuples/mappings into finite floats (bools ignored)."""
    out: list[float] = []
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
            if math.isfinite(num):  # drop NaN/inf so they can't poison _minmax
                out.append(num)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, Mapping):
            stack.extend(item.values())
    return out


def _minmax(values: Any) -> list[float] | None:
    """Return ``[min, max]`` of flattened numeric values, or None if none found."""
    nums = _numbers(values)
    return [min(nums), max(nums)] if nums else None


def _fmt_amount(amount: float) -> str:
    """Format a composition amount, dropping a trailing ``1`` and ``.0``."""
    if isinstance(amount, float) and amount.is_integer():
        amount = int(amount)
    return "" if amount == 1 else str(amount)


def _format_formula(amounts: Mapping[str, float]) -> str:
    """Alphabetical formula string from element->amount counts (1s dropped)."""
    return "".join(f"{el}{_fmt_amount(amt)}" for el, amt in sorted(amounts.items()))


def _structure_facts(struct: Any) -> dict[str, Any]:
    """Element counts, formula, and site count from a serialized structure dict."""
    if not isinstance(struct, Mapping):
        return {}
    sites = struct.get("sites") or []
    counts: dict[str, int] = {}
    for site in sites:
        species = site.get("species") if isinstance(site, Mapping) else None
        if isinstance(species, list) and species and isinstance(species[0], Mapping):
            element = species[0].get("element")
            if element:
                counts[str(element)] = counts.get(str(element), 0) + 1
    facts: dict[str, Any] = {"n_sites": len(sites)}
    if counts:
        facts["elements"] = sorted(counts)
        facts["formula"] = _format_formula(counts)
    return facts


def _xy_facts(widget_data: Mapping[str, Any], axes: tuple[str, ...]) -> dict[str, Any]:
    """Series count, point count, per-axis ranges, and labels for plot widgets."""
    series = widget_data.get("series") or []
    series = [series_item for series_item in series if isinstance(series_item, Mapping)]
    facts: dict[str, Any] = {
        "n_series": len(series),
        "n_points": sum(len(series_item.get("x") or []) for series_item in series),
    }
    labels = [
        series_item.get("label") for series_item in series if series_item.get("label")
    ]
    if labels:
        facts["series_labels"] = labels
    for axis in axes:
        axis_range = _minmax([series_item.get(axis) for series_item in series])
        if axis_range is not None:
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
    frames = traj.get("frames") or [] if isinstance(traj, Mapping) else []
    facts: dict[str, Any] = {"n_frames": len(frames)}
    if frames and isinstance(frames[0], Mapping):
        facts |= _structure_facts(frames[0].get("structure"))
    return facts


def _describe_periodic_table(data: Mapping[str, Any]) -> dict[str, Any]:
    """Element count and value range for a periodic-table heatmap."""
    values = data.get("heatmap_values")
    if not isinstance(values, (Mapping, list, tuple)):
        return {"n_elements": 0}
    facts: dict[str, Any] = {"n_elements": len(values)}
    value_range = _minmax(values)
    if value_range is not None:
        facts["value_range"] = value_range
    return facts


def _describe_heatmap_matrix(data: Mapping[str, Any]) -> dict[str, Any]:
    """Grid shape and value range for a heatmap matrix."""
    facts: dict[str, Any] = {
        "shape": [len(data.get("x_items") or []), len(data.get("y_items") or [])]
    }
    value_range = _minmax(data.get("values"))
    if value_range is not None:
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
    if not amounts:
        return {}
    return {"elements": sorted(amounts), "formula": _format_formula(amounts)}


def _describe_xrd(data: Mapping[str, Any]) -> dict[str, Any]:
    """Peak count and 2-theta range for an XRD pattern."""
    patterns = data.get("patterns")
    two_theta = patterns.get("x") if isinstance(patterns, Mapping) else None
    facts: dict[str, Any] = {"n_peaks": len(two_theta or [])}
    two_theta_range = _minmax(two_theta)
    if two_theta_range is not None:
        facts["two_theta_range"] = two_theta_range
    return facts


def _describe_spacegroup_bar(data: Mapping[str, Any]) -> dict[str, Any]:
    """Total and unique space-group entry counts."""
    values = data.get("data") or []
    return {"n_entries": len(values), "n_unique": len(set(values))}


def _describe_dos(data: Mapping[str, Any]) -> dict[str, Any]:
    """Energy-grid facts for a density of states."""
    dos = data.get("dos")
    energies = dos.get("energies") if isinstance(dos, Mapping) else None
    facts: dict[str, Any] = {}
    if energies is not None:
        facts["n_energies"] = len(energies)
    energy_range = _minmax(energies)
    if energy_range is not None:
        facts["energy_range"] = energy_range
    return facts


def _describe_band_structure(data: Mapping[str, Any]) -> dict[str, Any]:
    """Band count for a band structure."""
    bands = data.get("band_structure")
    bands = bands.get("bands") if isinstance(bands, Mapping) else None
    return {"n_bands": len(bands)} if isinstance(bands, (list, tuple)) else {}


def _describe_bands_and_dos(data: Mapping[str, Any]) -> dict[str, Any]:
    """Combined band-structure and DOS facts."""
    return {**_describe_band_structure(data), **_describe_dos(data)}


def _describe_treemap(data: Mapping[str, Any]) -> dict[str, Any]:
    """Node/leaf counts and depth for a treemap hierarchy."""
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


# Facts for 2D scatter/bar/histogram plots (3D adds a z-axis range)
_describe_xy = functools.partial(_xy_facts, axes=("x", "y"))
_describe_xyz = functools.partial(_xy_facts, axes=("x", "y", "z"))

# widget_type -> handler deriving structured facts from to_dict() output
_HANDLERS: dict[str, Any] = {
    "scatter_plot": _describe_xy,
    "bar_plot": _describe_xy,
    "histogram": _describe_xy,
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
    "treemap": _describe_treemap,
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
    "periodic_table": "heatmap_values",
    "heatmap_matrix": "values",
    "convex_hull": "entries",
    "chem_pot_diagram": "entries",
    "composition": "composition",
    "xrd": "patterns",
    "spacegroup_bar": "data",
    "dos": "dos",
    "band_structure": "band_structure",
    "treemap": "data",
    # can render with either input, so only warn when both are missing
    "bands_and_dos": ("dos", "band_structure"),
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
        series = [
            s for s in (widget_data.get("series") or []) if isinstance(s, Mapping)
        ]
        if series and all(not (series_item.get("x")) for series_item in series):
            warnings.append(
                f"{widget_type}: all series have empty 'x'; nothing to plot"
            )

    if widget_type == "rdf_plot" and not (
        widget_data.get("structures") or widget_data.get("patterns")
    ):
        warnings.append("rdf_plot: neither 'structures' nor 'patterns' provided")

    return warnings
