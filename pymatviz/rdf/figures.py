"""Plot full and element-pair radial distribution functions of structures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pymatviz.rdf.helpers import calculate_rdf, partial_rdfs
from pymatviz.structures import normalize_periodic_structures
from pymatviz.widgets.rdf_plot import RdfPlotWidget


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import Any

    import numpy as np
    import pandas as pd
    from pymatgen.core import IStructure

    from pymatviz.typing import AnyStructure

# Above this many sites in all structures, RDFs are computed in Python: the browser
# recomputes them (O(n_sites^2) pairs) whenever cutoff or bins change
MAX_BROWSER_RDF_SITES = 2_000


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
    optional = {"color": color, "legend_group": legend_group}
    return {"label": label, "pattern": pattern} | {
        key: val for key, val in optional.items() if val
    }


def _pair_entries(
    struct_dict: Mapping[str, IStructure],
    element_pairs: Sequence[tuple[str, str]] | None,
    cutoff: float,
    n_bins: int,
    colors: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """One curve per element pair of each structure, grouped by structure."""
    all_elems = set.union(
        *(struct.chemical_system_set for struct in struct_dict.values())
    )
    if element_pairs is None:
        element_pairs = [(el1, el2) for el1 in all_elems for el2 in all_elems]
    elif extra := {elem for pair in element_pairs for elem in pair} - all_elems:
        raise ValueError(f"Elements {extra} in element_pairs not in any structure")
    pairs = sorted({(min(el1, el2), max(el1, el2)) for el1, el2 in element_pairs})
    entries: list[dict[str, Any]] = []
    for label, struct in struct_dict.items():
        # skip pairs with elements this structure lacks (all-zero curves)
        struct_pairs = [
            pair for pair in pairs if set(pair) <= struct.chemical_system_set
        ]
        rdfs = partial_rdfs(struct, struct_pairs, cutoff=cutoff, n_bins=n_bins)
        for pair, rdf in zip(struct_pairs, rdfs, strict=True):
            entries.append(
                _rdf_entry(
                    "-".join(pair),
                    rdf,
                    color=colors[len(entries) % len(colors)] if colors else None,
                    element_pair=pair,
                    legend_group=label if len(struct_dict) > 1 else None,
                )
            )
    if not entries:
        raise ValueError(f"No structure contains any of the element pairs {pairs}")
    return entries


def rdf_plot(
    structures: AnyStructure
    | Sequence[AnyStructure]
    | Mapping[str, AnyStructure]
    | pd.Series
    | None = None,
    *,
    patterns: Sequence[dict[str, Any]] | None = None,
    mode: Literal["element_pairs", "full"] = "element_pairs",
    cutoff: float | None = None,
    bins: int = 75,
    element_pairs: Sequence[tuple[str, str]] | None = None,
    colors: Sequence[str] | None = None,
    **kwargs: Any,
) -> RdfPlotWidget:
    """Radial distribution functions (RDFs) of structures: one curve per element pair
    of each structure (``mode="element_pairs"``) or one all-atom curve per structure
    (``mode="full"``).

    MatterViz computes the RDFs in the browser, where the controls pane changes cutoff
    and bins live and hovering a curve highlights its sites. Selecting
    ``element_pairs``, setting ``colors`` or passing more than
    ``MAX_BROWSER_RDF_SITES`` sites computes them in Python instead.

    Args:
        structures: A pymatgen Structure or ASE Atoms, or a sequence, dict or
            pd.Series of them (dict/Series keys label the structures).
        patterns: Precomputed MatterViz RDF entries (``{label, pattern: {r, g_r}}``)
            to plot instead of structures.
        mode: "element_pairs" (in the browser, only the first 3 curves start visible)
            or "full".
        cutoff: Maximum distance in Å. Defaults to twice the longest lattice vector
            of all structures, capped at 15 Å.
        bins: Number of distance bins. Defaults to 75.
        element_pairs: Element pairs to plot in "element_pairs" mode, e.g.
            ``[("Si", "O")]``. Defaults to all pairs.
        colors: CSS colors cycled over the curves. Defaults to MatterViz's palette.
        **kwargs: RdfPlotWidget traits, e.g. ``show_reference_line=False`` to hide
            the g(r)=1 line, ``x_axis``, ``y_axis`` or ``style``.

    Returns:
        RdfPlotWidget: The RDF plot.

    Raises:
        ValueError: If neither or both of structures and patterns are given, a
            structure has no sites, or no structure has any of element_pairs.
    """
    if patterns is not None:
        if structures is not None:
            raise ValueError("pass structures or patterns, not both")
        return RdfPlotWidget(patterns=patterns, mode=mode, **kwargs)
    if structures is None:
        raise ValueError("pass structures or patterns")
    struct_dict = {
        str(key): struct
        for key, struct in normalize_periodic_structures(structures).items()
    }
    if empty := [key for key, struct in struct_dict.items() if len(struct) == 0]:
        raise ValueError(f"structures {empty} contain no sites")
    if cutoff is None:
        longest = max(max(struct.lattice.abc) for struct in struct_dict.values())
        cutoff = min(15, 2 * longest)
    n_sites = sum(map(len, struct_dict.values()))
    if element_pairs is None and colors is None and n_sites <= MAX_BROWSER_RDF_SITES:
        as_dicts = {key: struct.as_dict() for key, struct in struct_dict.items()}
        # a lone structure goes unlabeled: no legend group heading for one structure
        lone = next(iter(as_dicts.values())) if len(as_dicts) == 1 else as_dicts
        return RdfPlotWidget(
            structures=lone, mode=mode, cutoff=cutoff, n_bins=bins, **kwargs
        )
    if mode == "full":
        entries = [
            _rdf_entry(
                key,
                calculate_rdf(struct, cutoff=cutoff, n_bins=bins),
                color=colors[idx % len(colors)] if colors else None,
            )
            for idx, (key, struct) in enumerate(struct_dict.items())
        ]
    else:
        entries = _pair_entries(struct_dict, element_pairs, cutoff, bins, colors)
    # mode="full" keeps every precomputed curve visible (element_pairs shows 3)
    return RdfPlotWidget(patterns=entries, **{"mode": "full"} | kwargs)
