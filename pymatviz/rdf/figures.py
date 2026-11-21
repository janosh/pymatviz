"""Plot full and element-pair radial distribution functions of structures."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatviz.rdf.helpers import calculate_rdf, partial_rdfs
from pymatviz.structures import normalize_periodic_structures
from pymatviz.widgets.rdf_plot import RdfPlotWidget


if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence
    from typing import Any

    import numpy as np
    import pandas as pd
    from pymatgen.core import IStructure

    from pymatviz.typing import AnyStructure


def _resolve_n_bins(
    struct_dict: Mapping[Hashable, IStructure],
    cutoff: float,
    bins: int | None,
    bin_size: float | None,
) -> int:
    """Check structures have sites and that bins and bin_size aren't both set.
    Returns the number of bins: bins, derived from bin_size, or 75 by default.
    """
    for key, struct in struct_dict.items():
        if len(struct) == 0:
            raise ValueError(
                f"input structure{f' {key}' if key else ''} contains no sites"
            )
    if bin_size is None:
        return 75 if bins is None else bins
    if bins is not None:
        raise ValueError(
            f"Cannot specify both {bins=} and {bin_size=}. Pick one or the other."
        )
    return int(cutoff / bin_size)


def _rdf_entry(
    label: str,
    rdf: tuple[np.ndarray, np.ndarray],
    *,
    color: str | None,
    element_pair: tuple[str, str] | None = None,
    legend_group: str | None = None,
) -> dict[str, Any]:
    """MatterViz RdfEntry for one (radii, g(r)) curve."""
    radii, g_r = rdf
    pattern: dict[str, Any] = {"r": radii.tolist(), "g_r": g_r.tolist()}
    if element_pair:
        pattern["element_pair"] = list(element_pair)
    entry: dict[str, Any] = {"label": label, "pattern": pattern}
    if color:
        entry["color"] = color
    if legend_group:
        entry["legend_group"] = legend_group
    return entry


def _rdf_widget(entries: list[dict[str, Any]], **kwargs: Any) -> RdfPlotWidget:
    """Plot precomputed RDF entries. mode="full" keeps every curve visible: in its
    default "element_pairs" mode, MatterViz hides all but the first 3 curves.
    """
    return RdfPlotWidget(patterns=entries, **{"mode": "full"} | kwargs)


def element_pair_rdfs(
    structures: AnyStructure
    | Sequence[AnyStructure]
    | Mapping[str, AnyStructure]
    | pd.Series,
    *,
    cutoff: float | None = None,
    bins: int | None = None,
    bin_size: float | None = None,
    element_pairs: list[tuple[str, str]] | None = None,
    colors: Sequence[str] | None = None,
    **kwargs: Any,
) -> RdfPlotWidget:
    """Plot pairwise radial distribution functions (RDFs) for all (or a subset of)
    element pairs in one or multiple structures.

    Every element pair of every structure is one curve labeled by the pair. With
    multiple structures, each structure's curves share a legend group.

    Args:
        structures (AnyStructure | Sequence[AnyStructure] | dict[str, AnyStructure]):
            Can be one of the following:
            - single pymatgen Structure or ASE Atoms object
            - sequence (list, tuple) of pymatgen Structures or ASE Atoms objects
            - dictionary mapping labels to pymatgen Structures or ASE Atoms objects
            - pandas Series of pymatgen Structures or ASE Atoms objects
        cutoff (float | None, optional): Maximum distance for RDF calculation.
            If None, defaults to twice the longest lattice vector length across all
            structures (up to 15A). If negative, its absolute value is used as a scaling
            factor for the longest lattice vector length (e.g. -1.5 means 1.5x the
            longest lattice vector). Default is None.
        bins (int, optional): Number of bins for RDF calculation. Defaults to 75.
        bin_size (float, optional): Size of bins for RDF calculation, an alternative
            to bins. Default is None.
        element_pairs (list[tuple[str, str]], optional): Element pairs to plot.
            If None, all pairs present in any structure are plotted.
        colors (Sequence[str], optional): CSS colors cycled over the curves. Defaults
            to the MatterViz palette.
        **kwargs: Passed to RdfPlotWidget, e.g. ``show_reference_line=False`` to hide
            the g(r)=1 line, ``x_axis``, ``y_axis`` or ``style``.

    Returns:
        RdfPlotWidget: MatterViz RDF plot with one curve per structure and element
            pair.

    Raises:
        ValueError: If no structures are provided, if structures have no sites,
            if invalid element pairs are provided, or if both bins and bin_size are
            specified.
        TypeError: If input structures are not pymatgen Structures or ASE Atoms.
    """
    struct_dict = normalize_periodic_structures(structures)

    # Calculate dynamic cutoff if not specified or negative
    max_cell_len = max(max(struct.lattice.abc) for struct in struct_dict.values())
    if cutoff is None:
        cutoff = min(15, 2 * max_cell_len)
    elif cutoff < 0:
        cutoff = abs(cutoff) * max_cell_len

    n_bins = _resolve_n_bins(struct_dict, cutoff, bins, bin_size)

    all_elements = set.union(
        *(struct.chemical_system_set for struct in struct_dict.values())
    )
    if element_pairs is None:
        element_pairs = [
            (el1, el2) for el1 in all_elements for el2 in all_elements if el1 <= el2
        ]
    else:
        # Check if all elements in element_pairs are present in at least one structure
        pair_elements = {elem for pair in element_pairs for elem in pair}
        if extra_elems := pair_elements - set(all_elements):
            raise ValueError(
                f"Elements {extra_elems} in element_pairs not present in any structure"
            )

    multiple = len(struct_dict) > 1
    pairs = sorted({(min(el1, el2), max(el1, el2)) for el1, el2 in element_pairs})
    entries = []
    for label, struct in struct_dict.items():
        # skip pairs with elements this structure lacks (all-zero curves)
        struct_elems = struct.chemical_system_set
        struct_pairs = [pair for pair in pairs if set(pair) <= struct_elems]
        pair_rdfs = partial_rdfs(struct, struct_pairs, cutoff=cutoff, n_bins=n_bins)
        for (el1, el2), rdf in zip(struct_pairs, pair_rdfs, strict=True):
            color = colors[len(entries) % len(colors)] if colors else None
            entries.append(
                _rdf_entry(
                    f"{el1}-{el2}",
                    rdf,
                    color=color,
                    element_pair=(el1, el2),
                    legend_group=str(label) if multiple else None,
                )
            )
    if not entries:
        raise ValueError(f"No structure contains any of the element pairs {pairs}")
    return _rdf_widget(entries, **kwargs)


def full_rdf(
    structures: AnyStructure
    | Sequence[AnyStructure]
    | Mapping[str, AnyStructure]
    | pd.Series,
    *,
    cutoff: float = 15,
    bins: int | None = None,
    bin_size: float | None = None,
    colors: Sequence[str] | None = None,
    **kwargs: Any,
) -> RdfPlotWidget:
    """Plot the full radial distribution functions (RDFs) of one or multiple
    structures.

    Args:
        structures (AnyStructure | Sequence[AnyStructure] | dict[str, AnyStructure]):
            Can be one of the following:
            - single pymatgen Structure or ASE Atoms object
            - list of pymatgen Structures or ASE Atoms objects
            - dictionary mapping labels to pymatgen Structures or ASE Atoms objects.
        cutoff (float, optional): Maximum distance for RDF calculation. Default is 15 Å.
        bins (int, optional): Number of bins for RDF calculation. Defaults to 75.
        bin_size (float, optional): Size of bins for RDF calculation, an alternative
            to bins. Default is None.
        colors (Sequence[str], optional): CSS colors cycled over the structures'
            curves. Defaults to the MatterViz palette.
        **kwargs: Passed to RdfPlotWidget, e.g. ``show_reference_line=False`` to hide
            the g(r)=1 line, ``x_axis``, ``y_axis`` or ``style``.

    Returns:
        RdfPlotWidget: MatterViz RDF plot with one curve per structure.

    Raises:
        ValueError: If no structures are provided, if structures have no sites,
            or if both bins and bin_size are specified.
    """
    struct_dict = normalize_periodic_structures(structures)
    n_bins = _resolve_n_bins(struct_dict, cutoff, bins, bin_size)
    entries = [
        _rdf_entry(
            str(label),
            calculate_rdf(struct, cutoff=cutoff, n_bins=n_bins),
            color=colors[idx % len(colors)] if colors else None,
        )
        for idx, (label, struct) in enumerate(struct_dict.items())
    ]
    return _rdf_widget(entries, **kwargs)
