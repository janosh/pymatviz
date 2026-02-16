"""Plotting functions for pymatgen phonon band structures and density of states."""

from __future__ import annotations

import sys
from collections import defaultdict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.constants as const
from plotly.subplots import make_subplots
from pymatgen.phonon.dos import CompletePhononDos, PhononDos

from pymatviz.phonons.helpers import (
    AnyBandStructure,
    YMax,
    YMin,
    _shaded_range,
    pretty_sym_point,
)
from pymatviz.process_data import normalize_phonon_bands
from pymatviz.typing import (
    SET_INTERSECTION,
    SET_MODE,
    SET_STRICT,
    SET_UNION,
    AnyDos,
    SetMode,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Sequence
    from typing import Any

    from phonopy.phonon.band_structure import BandStructure as PhonopyBandStructure
    from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands


def phonon_bands(
    band_structs: AnyBandStructure
    | PhonopyBandStructure
    | Mapping[str, AnyBandStructure | PhonopyBandStructure],
    *,
    line_kwargs: (
        dict[str, Any]  # single dict for all lines
        # separate dicts for modes
        | dict[Literal["acoustic", "optical"], dict[str, Any]]
        # function taking (band_data, band_idx)
        | Callable[[np.ndarray, int], dict[str, Any]]
        | None
    ) = None,
    branches: Sequence[str] = (),
    path_mode: SetMode = SET_STRICT,
    shaded_ys: dict[tuple[YMin | YMax, YMin | YMax], dict[str, Any]]
    | bool
    | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Plot single or multiple pymatgen or phonopy band structures using Plotly.

    Warning: Only tested with phonon band structures so far but plan is to extend to
    electronic band structures.

    Args:
        band_structs (AnyBandStructure | dict[str, AnyBandStructure]): Single
            BandStructureSymmLine, PhononBandStructureSymmLine, or phonopy
            BandStructure object, or a dict with labels mapped to multiple such
            objects. If a dict, the keys are used as labels for the band structures.
        line_kwargs (dict | dict[str, dict] | Callable): Line style configuration. Can
            be one of:
            - A single dict applied to all lines with Plotly line properties
              (e.g. dict(width=2, color="red", dash="solid"))
            - A dict with keys "acoustic" and "optical" containing style dicts for each
              mode type
            - A callable taking (frequencies: np.ndarray, band_idx: int) and returning
              a style dict
            Common style options include color, width, dash. Defaults to None.
        branches (Sequence[str]): Branches to plot. Defaults to empty tuple, meaning all
            branches are plotted.
        path_mode ("union" | "intersection" | "strict"): How to handle band structures
            with different q-point paths. Defaults to "strict":
            - "union": Plot all path segments from all band structures
            - "intersection": Only plot segments common to all band structures
            - "strict": Raise error if paths don't match exactly (default)
        shaded_ys (dict[tuple[float | str, float | str], dict]): Keys are y-ranges
            (min, max) tuple and values are kwargs for shaded regions created by
            fig.add_hrect(). Defaults to single entry (0, "y_min"):
            dict(fillcolor="gray", opacity=0.07). "y_min" and "y_max" will be replaced
            with the figure's y-axis range. dict(layer="below", row="all", col="all") is
            always passed to add_hrect but can be overridden by the user. Set to False
            to disable.
        **kwargs: Passed to Plotly's Figure.add_scatter method.

    Returns:
        go.Figure: Plotly figure object.

    Raises:
        TypeError: If band_structs is not a PhononBandStructureSymmLine, phonopy
            BandStructure or dict of these.
    """
    # Convert input to dict of pymatgen PhononBands
    bs_dict: dict[str, PhononBands] = normalize_phonon_bands(band_structs)

    fig = go.Figure()
    line_kwargs = line_kwargs or {}

    if isinstance(branches, str):
        branches = [branches]

    # First, collect all unique path segments and their endpoints across all structures
    all_segments: dict[
        tuple[str | None, str | None], list[tuple[Hashable, PhononBands]]
    ] = defaultdict(list)

    for label, band_struct in bs_dict.items():
        for branch in band_struct.branches:
            start_idx = branch["start_index"]
            end_idx = branch["end_index"]

            # Get start and end q-point labels for this branch
            start_label = band_struct.qpoints[start_idx].label
            end_label = band_struct.qpoints[end_idx].label

            segment_key = (start_label, end_label)
            all_segments[segment_key] += [(label, band_struct)]

    # Now we have all_segments, determine which segments to plot based on path_mode
    if path_mode == SET_STRICT:
        # Check if all band structures have exactly the same segments
        first_bs = next(iter(bs_dict.values()))
        first_segments = {
            (start, end)
            for branch in first_bs.branches
            for start, end in [
                (
                    first_bs.qpoints[branch["start_index"]].label,
                    first_bs.qpoints[branch["end_index"]].label,
                )
            ]
        }
        for band_struct in bs_dict.values():
            these_segments = {
                (start, end)
                for branch in band_struct.branches
                for start, end in [
                    (
                        band_struct.qpoints[branch["start_index"]].label,
                        band_struct.qpoints[branch["end_index"]].label,
                    )
                ]
            }
            if these_segments != first_segments:
                raise ValueError(
                    "Band structures have different q-point paths. Use path_mode="
                    f"{SET_UNION} or {SET_INTERSECTION} to plot band structures with "
                    "different paths."
                )
        segments_to_plot = first_segments
    else:
        # Count how many band structures have each segment
        segment_counts = {
            segment: len(structs) for segment, structs in all_segments.items()
        }
        n_structures = len(bs_dict)

        if path_mode == SET_INTERSECTION:
            # Only keep segments present in all band structures
            segments_to_plot = {
                segment
                for segment, count in segment_counts.items()
                if count == n_structures
            }
            if len(segments_to_plot) == 0:
                raise ValueError(
                    f"{path_mode=} but no common path segments found between band "
                    "structures"
                )
        elif path_mode == SET_UNION:
            segments_to_plot = set(all_segments)
        else:
            raise ValueError(f"Invalid {path_mode=}, must be one of {SET_MODE}")

    # find common branches by normalized branch names
    if branches:
        common_branches: set[str] = set()
        for idx, band_struct in enumerate(bs_dict.values()):
            bs_branches = {branch["name"] for branch in band_struct.branches}
            common_branches = (
                bs_branches
                if idx == 0
                # calc set union/intersect (& or |) depending on path_mode
                else common_branches & bs_branches
                if path_mode in (SET_INTERSECTION, SET_STRICT)
                else common_branches | bs_branches  # path_mode == SET_UNION
            )
        missing_branches = set(branches) - common_branches
        avail_branches = "\n- ".join(common_branches)
        common_branches &= set(branches)
        if missing_branches:
            print(  # keep this warning after "No common branches" error  # noqa: T201
                f"Warning {missing_branches=}, available branches:\n- {avail_branches}",
                file=sys.stderr,
            )

    # Create a mapping of q-point pairs to x-axis positions
    x_positions: dict[tuple[str | None, str | None], tuple[float, float]] = {}
    current_x = 0.0

    for segment in sorted(segments_to_plot):  # Sort to ensure consistent ordering
        if segment not in x_positions:
            # Find the length of this segment in the first band structure that has it
            band_struct = all_segments[segment][0][1]
            segment_len = 0
            for branch in band_struct.branches:
                start_idx, end_idx = branch["start_index"], branch["end_index"]
                if (
                    band_struct.qpoints[start_idx].label == segment[0]
                    and band_struct.qpoints[end_idx].label == segment[1]
                ):
                    segment_len = (
                        band_struct.distance[end_idx] - band_struct.distance[start_idx]
                    )
                    break
            x_positions[segment] = (current_x, current_x + segment_len)
            current_x += segment_len

    # Now plot each band structure's segments at the correct x positions
    colors = px.colors.qualitative.Plotly
    line_styles = ("solid", "dot", "dash", "longdash", "dashdot", "longdashdot")

    for bs_idx, (label, band_struct) in enumerate(bs_dict.items()):
        color = colors[bs_idx % len(colors)]
        line_style = line_styles[bs_idx % len(line_styles)]

        for branch in band_struct.branches:
            start_idx = branch["start_index"]
            end_idx = branch["end_index"] + 1

            # Get the x-axis position for this segment
            start_label = band_struct.qpoints[start_idx].label
            end_label = band_struct.qpoints[end_idx - 1].label
            segment = (start_label, end_label)

            if segment not in segments_to_plot:
                continue  # Skip segments not in the segments_to_plot set

            x_start, x_end = x_positions[segment]

            # Scale the x-range corresponding to the current band's k-path segment
            segment_distances = np.array(band_struct.distance[start_idx:end_idx])
            segment_distances = (segment_distances - segment_distances[0]) * (
                x_end - x_start
            ) / (segment_distances[-1] - segment_distances[0]) + x_start

            for band_idx in range(band_struct.nb_bands):
                frequencies = band_struct.bands[band_idx][start_idx:end_idx]
                is_acoustic = band_idx < 3
                mode_type = "acoustic" if is_acoustic else "optical"

                # Default line style
                line_defaults = dict(
                    color=color, width=1.5 if is_acoustic else 1, dash=line_style
                )
                trace_name = label
                existing_names = {trace.name for trace in fig.data}

                # Apply line style based on line_kwargs type
                if callable(line_kwargs):
                    # Pass band data and index to callback
                    custom_style = line_kwargs(frequencies, band_idx)
                    line_defaults |= custom_style
                elif isinstance(line_kwargs, dict):
                    # check for custom line styles for one or both modes
                    if {"acoustic", "optical"} <= set(line_kwargs):
                        # mode_type guaranteed to be "acoustic" or "optical"
                        mode_styles = line_kwargs[mode_type]
                        # use custom trace name if provided (needs to be popped before
                        # passed to line kwargs)
                        if mode_name := mode_styles.pop("name", None):
                            trace_name = mode_name
                            # don't show default trace name in legend if got custom name
                            existing_names.add(trace_name)

                        # Use mode-specific styles
                        line_defaults |= mode_styles
                    else:  # Apply single style dict to all lines
                        line_defaults |= cast("dict[str, Any]", line_kwargs)

                is_new_name = trace_name not in existing_names
                fig.add_scatter(
                    x=segment_distances,
                    y=frequencies,
                    mode="lines",
                    line=line_defaults,
                    legendgroup=trace_name,
                    name=trace_name,
                    showlegend=is_new_name,
                    **kwargs,
                )

    # Update x-axis ticks to show all q-points
    x_ticks, x_labels = [], []
    for (start_label, end_label), (x_start, x_end) in sorted(
        x_positions.items(), key=lambda x: x[1][0]
    ):
        if x_start not in x_ticks:
            x_ticks.append(x_start)
            x_labels.append(start_label or "")
        x_ticks.append(x_end)
        x_labels.append(end_label or "")

    fig.layout.xaxis.update(
        tickvals=x_ticks,
        ticktext=[pretty_sym_point(label) for label in x_labels],
        tickangle=0,
    )

    # Add vertical lines at high-symmetry points. Remove 0 and last line to avoid
    # duplicate vertical line (they look like graphical artifacts)
    for x_pos in {*x_ticks} - {0, x_ticks[-1]}:
        fig.add_vline(x=x_pos, line=dict(color="black", width=1))

    fig.layout.xaxis.title = "Wave Vector"
    fig.layout.yaxis.title = "Frequency (THz)"
    fig.layout.margin = dict(t=5, b=5, l=5, r=5)

    # get y-axis range from all band structures
    y_min = min(min(bs.bands.ravel()) for bs in bs_dict.values())
    y_max = max(max(bs.bands.ravel()) for bs in bs_dict.values())

    if y_min < -0.1:  # no need for y=0 line if y_min = 0
        fig.add_hline(y=0, line=dict(color="black", width=1))
    if y_min >= -0.01:  # set y_min=0 if below tolerance for imaginary frequencies
        y_min = 0
    fig.layout.yaxis.range = (1.05 * y_min, 1.05 * y_max)

    axes_kwargs = dict(linecolor="black", gridcolor="lightgray")
    fig.layout.xaxis.update(**axes_kwargs)
    fig.layout.yaxis.update(**axes_kwargs)

    # move legend to top left corner
    fig.layout.legend.update(
        x=0.005,
        y=0.99,
        orientation="h",
        yanchor="top",
        bgcolor="rgba(255, 255, 255, 0.6)",
        tracegroupgap=0,
    )

    # scale font size with figure size
    fig.layout.font.size = 16 * (fig.layout.width or 800) / 800

    _shaded_range(fig, shaded_ys=shaded_ys)

    return fig


def phonon_dos(
    doses: AnyDos | CompletePhononDos | Mapping[str, AnyDos | CompletePhononDos],
    *,
    stack: bool = False,
    sigma: float = 0,
    units: Literal["THz", "eV", "meV", "Ha", "cm-1"] = "THz",
    normalize: Literal["max", "sum", "integral"] | None = None,
    last_peak_anno: str | None = None,
    project: Literal["element", "site"] | None = None,
    show_total: bool = True,
    **kwargs: Any,
) -> go.Figure:
    """Plot phonon DOS using Plotly.

    Args:
        doses (AnyDos | CompletePhononDos | dict[str, AnyDos | CompletePhononDos]):
            pymatgen PhononDos, CompletePhononDos, phonopy TotalDos, or dict of these.
        stack (bool): Whether to plot the DOS as a stacked area graph. Defaults to
            False.
        sigma (float): Standard deviation for Gaussian smearing. Defaults to 0.
        units (str): Units for the frequencies. Defaults to "THz".
        normalize (str | None): Normalization mode. One of "max", "sum", "integral",
            or None. Defaults to None.
        last_peak_anno (str): Annotation for last DOS peak with f-string placeholders
            for key (of dict containing multiple DOSes), last_peak frequency and units.
            Defaults to None, meaning last peak annotation is disabled. Set to "" to
            enable with a sensible default string.
        project (str | None): Projection mode for CompletePhononDos.
            "element" decomposes into per-element partial DOS, "site" into per-site
            partial DOS. Requires CompletePhononDos input. Defaults to None (plot
            total DOS only).
        show_total (bool): When projecting, overlay the total DOS as a dashed gray line.
            Only used when project is not None. Defaults to True.
        **kwargs: Passed to Plotly's Figure.add_scatter method.

    Returns:
        go.Figure: Plotly figure object.

    Raises:
        TypeError: If project is set but input is not CompletePhononDos.
    """
    valid_normalize = (None, "max", "sum", "integral")
    if normalize not in valid_normalize:
        raise ValueError(f"Invalid {normalize=}, must be one of {valid_normalize}.")
    raw_doses = (
        cast("Mapping[str, AnyDos | CompletePhononDos]", doses)
        if isinstance(doses, Mapping)
        else {"": doses}
    )

    dos_dict: dict[str, PhononDos] = {}
    total_overlay_dict: dict[str, PhononDos] = {}
    for label, raw_dos in raw_doses.items():
        label_prefix = f"{label} - " if label else ""
        if project is None:
            cls_name = f"{type(raw_dos).__module__}.{type(raw_dos).__qualname__}"
            if cls_name == "phonopy.phonon.dos.TotalDos":
                phonopy_total_dos = cast("Any", raw_dos)
                dos_dict[label] = PhononDos(
                    frequencies=phonopy_total_dos.frequency_points,
                    densities=phonopy_total_dos.dos,
                )
            elif isinstance(raw_dos, CompletePhononDos):
                dos_dict[label] = PhononDos(raw_dos.frequencies, raw_dos.densities)
            elif isinstance(raw_dos, PhononDos):
                dos_dict[label] = raw_dos
            else:
                raise TypeError(
                    f"Only {PhononDos.__name__}, {CompletePhononDos.__name__}, "
                    "phonopy TotalDos, or dict of these supported, "
                    f"got {type(raw_dos).__name__}"
                )
            continue
        if not isinstance(raw_dos, CompletePhononDos):
            raise TypeError(
                f"project={project!r} requires CompletePhononDos, "
                f"got {type(raw_dos).__name__} for key {label!r}"
            )
        if project == "element":
            dos_dict |= {
                f"{label_prefix}{element}": partial_dos
                for element, partial_dos in raw_dos.get_element_dos().items()
            }
        elif project == "site":
            dos_dict |= {
                f"{label_prefix}{site.specie}{site_idx}": raw_dos.get_site_dos(site)
                for site_idx, site in enumerate(raw_dos.structure)
            }
        else:
            raise ValueError(f"Invalid {project=}, must be 'element' or 'site'")
        if show_total:
            total_overlay_dict[f"{label_prefix}Total" if label_prefix else "Total"] = (
                PhononDos(raw_dos.frequencies, raw_dos.densities)
            )

    if not dos_dict:
        raise ValueError("Empty DOS dict")

    if last_peak_anno == "":
        last_peak_anno = "ω<sub>{key}</sub></span>={last_peak:.1f} {units}"

    def _prepare_dos(dos: PhononDos) -> tuple[np.ndarray, np.ndarray]:
        """Convert frequencies and apply smearing + normalization."""
        frequencies = convert_frequencies(dos.frequencies, units)
        densities = dos.get_smeared_densities(sigma)
        if normalize in ("max", "sum"):
            density_norm = densities.max() if normalize == "max" else densities.sum()
            if density_norm == 0:
                msg_key = "max density" if normalize == "max" else "sum density"
                raise ValueError(
                    f"Cannot normalize DOS with mode={normalize!r}: {msg_key} is 0."
                )
            densities /= density_norm
        elif normalize == "integral":
            if len(frequencies) < 2:
                raise ValueError(
                    "Cannot normalize DOS with mode='integral': "
                    "need >=2 frequency points."
                )
            bin_width = frequencies[1] - frequencies[0]
            if bin_width == 0:
                raise ValueError(
                    "Cannot normalize DOS with mode='integral': bin width is 0."
                )
            density_norm = densities.sum()
            if density_norm == 0:
                raise ValueError(
                    "Cannot normalize DOS with mode='integral': sum density is 0."
                )
            densities = densities / density_norm / bin_width
        return frequencies, densities

    fig = go.Figure()
    cumulative_density: np.ndarray | None = None
    for dos_name, dos_obj in dos_dict.items():
        frequencies, densities = _prepare_dos(dos_obj)
        scatter_kwargs: dict[str, Any] = {"mode": "lines"}
        if stack:
            cumulative_density = (
                densities
                if cumulative_density is None
                else cumulative_density + densities
            )
            densities = cumulative_density
            scatter_kwargs["fill"] = "tonexty"
        fig.add_scatter(
            x=frequencies, y=densities, name=dos_name, **scatter_kwargs | kwargs
        )

    if project is not None and show_total:
        for total_name, total_dos in total_overlay_dict.items():
            frequencies, densities = _prepare_dos(total_dos)
            fig.add_scatter(
                x=frequencies,
                y=densities,
                name=total_name,
                mode="lines",
                line=dict(dash="dash", color="gray", width=1.5),
                showlegend=True,
            )

    fig.layout.xaxis.update(title=f"Frequency ({units})")
    fig.layout.yaxis.update(title="Density of States", rangemode="tozero")
    fig.layout.margin = dict(t=5, b=5, l=5, r=5)
    fig.layout.font.size = 16 * (fig.layout.width or 800) / 800
    fig.layout.legend.update(x=0.005, y=0.99, orientation="h", yanchor="top")

    if last_peak_anno:
        qual_colors = px.colors.qualitative.Plotly
        for idx, (key, dos) in enumerate(dos_dict.items()):
            last_peak = dos.get_last_peak()
            color = (
                fig.data[idx].line.color
                or fig.data[idx].marker.color
                or qual_colors[idx % len(qual_colors)]
            )

            anno = dict(
                text=last_peak_anno.format(key=key, last_peak=last_peak, units=units),
                font=dict(color=color),
                xanchor="right",
                yshift=idx * -30,  # shift downward with increasing index
            )
            fig.add_vline(
                x=last_peak,
                line=dict(color=color, dash="dot"),
                name=f"last phDOS peak {key}",
                annotation=anno,
            )

    return fig


def convert_frequencies(
    frequencies: np.ndarray,
    unit: Literal["THz", "eV", "meV", "Ha", "cm-1"] = "THz",
) -> np.ndarray:
    """Convert frequencies from THz to specified units.

    Args:
        frequencies (np.ndarray): Frequencies in THz.
        unit (str): Target units. One of 'THz', 'eV', 'meV', 'Ha', 'cm-1'.

    Returns:
        np.ndarray: Converted frequencies.
    """
    conversion_factors = {
        "THz": 1,
        "eV": const.value("hertz-electron volt relationship") * const.tera,
        "meV": const.value("hertz-electron volt relationship")
        * const.tera
        / const.milli,
        "Ha": const.value("hertz-hartree relationship") * const.tera,
        "cm-1": const.value("hertz-inverse meter relationship")
        * const.tera
        * const.centi,
    }

    factor = conversion_factors.get(unit)
    if factor is None:
        raise ValueError(f"Invalid {unit=}, must be one of {list(conversion_factors)}")
    return frequencies * factor


def phonon_bands_and_dos(
    band_structs: PhononBands | Mapping[str, PhononBands],
    doses: PhononDos | Mapping[str, PhononDos],
    bands_kwargs: dict[str, Any] | None = None,
    dos_kwargs: dict[str, Any] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    all_line_kwargs: dict[str, Any] | None = None,
    per_line_kwargs: dict[str, dict[str, Any]] | None = None,
    path_mode: SetMode = SET_STRICT,
    **kwargs: Any,
) -> go.Figure:
    """Plot phonon DOS and band structure using Plotly.

    Args:
        doses (PhononDos | dict[str, PhononDos]): PhononDos or dict of multiple.
        band_structs (PhononBandStructureSymmLine | dict[str, PhononBandStructure]):
            Single BandStructureSymmLine or PhononBandStructureSymmLine object or a dict
            with labels mapped to multiple such objects.
        bands_kwargs (dict[str, Any]): Passed to Plotly's Figure.add_scatter method.
        dos_kwargs (dict[str, Any]): Passed to Plotly's Figure.add_scatter method.
        subplot_kwargs (dict[str, Any]): Passed to Plotly's make_subplots method.
            Defaults to dict(shared_yaxes=True, column_widths=(0.8, 0.2),
            horizontal_spacing=0.01).
        all_line_kwargs (dict[str, Any]): Passed to trace.update for each trace in
            fig.data. Modifies line appearance for all traces. Defaults to None.
        per_line_kwargs (dict[str, str]): Map of line labels to kwargs for trace.update.
            Modifies line appearance for specific traces. Defaults to None.
        path_mode (str): Mode for q-point path between band structures. Defaults to
            "strict". See phonon_bands for options.
        **kwargs: Passed to Plotly's Figure.add_scatter method.

    Returns:
        go.Figure: Plotly figure object.

    Raises:
        ValueError: If band_structs and doses keys don't match.
    """
    if not isinstance(band_structs, Mapping):  # normalize input to Mapping
        band_structs = {"": band_structs}
    if not isinstance(doses, Mapping):  # normalize input to Mapping
        doses = {"": doses}
    if (band_keys := set(band_structs)) != (dos_keys := set(doses)):
        raise ValueError(f"{band_keys=} and {dos_keys=} must be identical")

    subplot_defaults = dict(
        shared_yaxes=True, column_widths=(0.8, 0.2), horizontal_spacing=0.03
    )
    fig = make_subplots(rows=1, cols=2, **subplot_defaults | (subplot_kwargs or {}))

    # plot band structure
    bands_kwargs = bands_kwargs or {}
    shaded_ys = bands_kwargs.pop("shaded_ys", None)
    # disable shaded_ys for bands, would cause double shading due to _shaded_range below
    bands_kwargs["shaded_ys"] = False
    bands_fig = phonon_bands(band_structs, path_mode=path_mode, **kwargs | bands_kwargs)
    # import band structure layout to main figure
    fig.update_layout(bands_fig.layout)

    fig.add_traces(bands_fig.data, rows=1, cols=1)

    # plot density of states
    dos_fig = phonon_dos(doses, **kwargs | (dos_kwargs or {}))
    # swap DOS x and y axes (for 90 degrees rotation)
    for trace in dos_fig.data:
        trace.x, trace.y = trace.y, trace.x

    fig.add_traces(dos_fig.data, rows=1, cols=2)
    # transfer zero line from DOS to band structure
    if fig.layout.yaxis.range[0] < -0.1:
        fig.add_hline(y=0, line=dict(color="black", width=1), row=1, col=2)

    line_map: dict[str, dict[str, Any]] = {}
    for trace in fig.data:
        # put traces with same labels into the same legend group
        trace.legendgroup = trace.name
        # hide legend for all BS lines, show only DOS line
        trace.showlegend = trace.showlegend and trace.xaxis == "x2"
        # give all lines with same name the same appearance (esp. color)
        trace.line = line_map.setdefault(trace.name, trace.line)

        trace.update(all_line_kwargs or {})
        if trace_kwargs := (per_line_kwargs or {}).get(trace.name):
            trace.update(trace_kwargs)

    fig.layout.xaxis2.update(title="DOS")
    # transfer x-axis label from DOS fig to parent fig (since DOS may have custom units)
    fig.layout.yaxis.update(title=dos_fig.layout.xaxis.title.text)

    # set y-axis range to match band structure
    fig.layout.yaxis.update(range=bands_fig.layout.yaxis.range)

    _shaded_range(fig, shaded_ys=shaded_ys)

    return fig
