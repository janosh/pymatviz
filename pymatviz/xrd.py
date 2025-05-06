"""Module for plotting XRD patterns using plotly."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymatgen.analysis.diffraction.xrd import DiffractionPattern, XRDCalculator
from pymatgen.core import Structure

from pymatviz.process_data import is_ase_atoms


if TYPE_CHECKING:
    from typing import Any, TypeAlias


PatternOrStruct: TypeAlias = DiffractionPattern | Structure
HklFormat: TypeAlias = Literal["compact", "full", None]
ValidHklFormats = HklCompact, HklFull, HklNone = get_args(HklFormat)
cu_k_alpha_wavelength = 1.54184  # Angstroms


def format_hkl(hkl: tuple[int, int, int], format_type: HklFormat) -> str:
    """Format hkl indices as a string.

    Args:
        hkl (tuple[int, int, int]): The hkl indices to format.
        format_type ('compact' | 'full' | None): How to display the hkl indices.

    Raises:
        ValueError: If format_type is not one of 'compact', 'full', or None.
    """
    if format_type == "compact":
        return "".join(map(str, hkl))
    if format_type == "full":
        return f"({', '.join(map(str, hkl))})"
    if format_type is None:
        return ""
    raise ValueError(f"{format_type=} must be one of {ValidHklFormats}")


def xrd_pattern(  # noqa: D417
    patterns: PatternOrStruct
    | dict[str, PatternOrStruct | tuple[PatternOrStruct, dict[str, Any]]],
    *,
    peak_width: float = 0.5,
    annotate_peaks: float = 5,
    hkl_format: HklFormat = HklCompact,
    show_angles: bool | None = None,
    wavelength: float = cu_k_alpha_wavelength,
    stack: Literal["horizontal", "vertical"] | None = None,
    subplot_kwargs: dict[str, Any] | None = None,
    subtitle_kwargs: dict[str, Any] | None = None,
    axis_title_kwargs: dict[str, Any] | None = None,
) -> go.Figure:
    """Create a plotly figure of XRD patterns from a pymatgen DiffractionPattern,
    from a pymatgen Structure, or a dictionary of either of them.

    Args:
        patterns (PatternOrStruct | dict[str, PatternOrStruct | tuple[PatternOrStruct,
            dict[str, Any]]]): Either a single DiffractionPattern or Structure object,
            or a dictionary where keys are legend labels
            and values are either DiffractionPattern/Structure objects or tuples of
            (DiffractionPattern/Structure, kwargs) for customizing individual patterns.
        peak_width (float): Width of the diffraction peaks in degrees. Default is 0.5.
        annotate_peaks (float): Controls peak annotation. If int, annotates that many
            highest peaks. If float, should be in (0, 1) which will annotate peaks
            higher than that fraction of the highest peak. Default is 5.
        hkl_format (HklFormat): Format for hkl indices. One of 'compact' (ex: '100'),
            'full' (ex: '(1, 0, 0)'), or None for no hkl indices. Default is 'compact'.
        show_angles (bool | None): Whether to show angles in peak annotations. If None,
            it will default to True if plotting 1 or 2 patterns, False for 3 or more
            patterns.
        wavelength (float): X-ray wavelength for the XRD calculation (in Angstroms).
            Default is 1.54184 (Cu K-alpha). Only used if patterns argument contains
            Structures.
        stack ("horizontal" | "vertical" | None): If set to "horizontal" or
            "vertical", creates separate subplots for each pattern. Default is None
            (all patterns in one plot).
        subplot_kwargs (dict[str, Any] | None): Passed to make_subplots. Can be used to
            control spacing between subplots, e.g. {'vertical_spacing': 0.02}.
        subtitle_kwargs (dict[str, Any] | None): Override default subplot title
            settings. E.g. dict(font_size=14). Default is None.
        axis_title_kwargs (dict[str, Any] | None): Override default axis title
            settings. E.g. dict(font_size=14). Default is None.

    Raises:
        ValueError: If annotate_peaks is not a positive int or a float in (0, 1).
        TypeError: If patterns is not a DiffractionPattern, Structure or a dict of them.

    Returns:
        go.Figure: A plotly figure of the XRD pattern(s).
    """
    if (
        not isinstance(annotate_peaks, int | float)
        or annotate_peaks < 0
        or (isinstance(annotate_peaks, float) and annotate_peaks >= 1)
    ):
        raise ValueError(
            f"{annotate_peaks=} should be a positive int or a float in (0, 1)"
        )

    # Convert single object to dict for uniform processing
    if not isinstance(patterns, dict):
        patterns = {"XRD Pattern": patterns}
    elif not isinstance(patterns, dict):
        raise TypeError(
            f"{patterns=} should be a DiffractionPattern, Structure or a dict of them"
        )

    # Determine show_angles based on number of patterns
    if show_angles is None:
        show_angles = len(patterns) <= 2

    n_patterns = len(patterns)
    if stack:
        rows, cols = (n_patterns, 1) if stack == "vertical" else (1, n_patterns)
        subplot_defaults = dict(
            rows=rows,
            cols=cols,
            shared_xaxes=True,
            shared_yaxes=True,
            horizontal_spacing=0.08 / cols,
            vertical_spacing=0.08 / rows,
        )
        fig = make_subplots(**subplot_defaults | (subplot_kwargs or {}))
        # increase peak width for horizontal stacking
        if stack == "horizontal":
            peak_width *= 3
    else:
        fig = go.Figure()

    max_intensity = max_two_theta = 0
    plotted_patterns: list[DiffractionPattern] = []

    for trace_idx, (label, pattern_data) in enumerate(patterns.items()):
        if isinstance(pattern_data, tuple):
            pattern_or_struct, trace_kwargs = pattern_data
        else:
            pattern_or_struct, trace_kwargs = pattern_data, {}

        if is_ase_atoms(pattern_or_struct):
            from pymatgen.io.ase import AseAtomsAdaptor

            pattern_or_struct = AseAtomsAdaptor().get_structure(pattern_or_struct)

        if isinstance(pattern_or_struct, Structure):
            xrd_calculator = XRDCalculator(wavelength=wavelength)
            diffraction_pattern = xrd_calculator.get_pattern(pattern_or_struct)
        elif isinstance(pattern_or_struct, DiffractionPattern):
            diffraction_pattern = pattern_or_struct
        else:
            value = pattern_or_struct
            raise TypeError(
                "expecting a pymatgen Structure or DiffractionPattern, "
                f"got {type(value).__name__}: {value}"
            )

        plotted_patterns.append(diffraction_pattern)
        two_theta, intensities = diffraction_pattern.x, diffraction_pattern.y
        hkls, d_hkls = diffraction_pattern.hkls, diffraction_pattern.d_hkls

        if intensities is None or len(intensities) == 0:
            raise ValueError(
                f"No intensities found in the diffraction pattern for {label}"
            )

        # get max intensity and two_theta across all patterns
        max_intensity = max(max_intensity, *intensities)
        max_two_theta = max(max_two_theta, *two_theta)

        tooltips = [
            f"<b>{label}</b><br>2θ: {x:.2f}°<br>Intensity: {y:.2f}<br>hkl: "
            f"{'<br>'.join(format_hkl(h['hkl'], HklFull) for h in hkl)}<br>d: {d:.3f} Å"
            for x, y, hkl, d in zip(two_theta, intensities, hkls, d_hkls, strict=True)
        ]

        if stack:
            row = trace_idx + 1 if stack == "vertical" else 1
            col = trace_idx + 1 if stack == "horizontal" else 1
            trace_kwargs.setdefault("row", row)
            trace_kwargs.setdefault("col", col)
        fig.add_bar(
            x=two_theta,
            y=intensities,
            width=peak_width,
            name=label,
            hovertext=tooltips,
            hoverinfo="text",
            **trace_kwargs,
        )

    # Normalize intensities to 100 and add annotations
    for trace_idx, trace in enumerate(fig.data):
        trace.y = [y / max_intensity * 100 for y in trace.y]

        if isinstance(annotate_peaks, int) and annotate_peaks > 0:
            peak_indices = np.argsort(trace.y)[-annotate_peaks:]
        elif 0 < annotate_peaks < 1:
            peak_indices = [
                idx
                for idx, intensity in enumerate(trace.y)
                if intensity > annotate_peaks * 100
            ]
        else:
            peak_indices = []

        for idx in peak_indices:
            x_pos, y_pos = trace.x[idx], trace.y[idx]

            if hkl_format:
                hkl_formatted = "<br>".join(
                    format_hkl(h["hkl"], hkl_format)
                    for h in plotted_patterns[trace_idx].hkls[idx]
                )
                annotation_text = f"{hkl_formatted}"
                if show_angles:
                    annotation_text += f"<br>{x_pos:.2f}°"
            elif show_angles:
                annotation_text = f"{x_pos:.2f}°"
            else:
                continue  # Skip annotation if neither hkl nor angle is shown

            # Determine annotation direction
            ax, ay = (20, -20) if trace_idx % 2 == 0 else (-20, -20)

            # Adjust position for edges of the plot
            if x_pos > max_two_theta * 0.9:
                ax = -abs(ax)
            elif y_pos > 90:
                ay = abs(ay)

            anno_kwargs = dict(
                x=x_pos,
                y=y_pos,
                text=annotation_text,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=ax,
                ay=ay,
                xanchor="left" if ax > 0 else "right",
                yanchor="bottom" if ay < 0 else "top",
            )

            if stack:
                row = trace_idx + 1 if stack == "vertical" else 1
                col = trace_idx + 1 if stack == "horizontal" else 1
                anno_kwargs.setdefault("row", row)
                anno_kwargs.setdefault("col", col)
            fig.add_annotation(**anno_kwargs)

        if stack:
            # Add trace name annotation at the top of each subplot
            row = trace_idx + 1 if stack == "vertical" else 1
            col = trace_idx + 1 if stack == "horizontal" else 1
            subtitle_defaults = dict(
                x=1, y=1, showarrow=False, font_size=12, xanchor="right", yanchor="top"
            )
            kwargs = subtitle_defaults | (subtitle_kwargs or {})
            xref = f"x{trace_idx + 1} domain".replace("x1 ", "x ")
            yref = f"y{trace_idx + 1} domain".replace("y1 ", "y ")
            fig.add_annotation(
                text=trace.name, xref=xref, yref=yref, row=row, col=col, **kwargs
            )

    # Add axis titles (not using plotly's built-in axis titles because they don't
    # position correctly in stacked mode)
    axis_title_kwargs = dict(
        font_size=12, xref="paper", yref="paper", showarrow=False
    ) | (axis_title_kwargs or {})
    fig.add_annotation(  # X-axis title
        text="2θ (degrees)", x=0.5, y=-0.12, **axis_title_kwargs
    )
    fig.add_annotation(  # Y-axis title
        text="Intensity (a.u.)", x=-0.07, y=0.5, textangle=-90, **axis_title_kwargs
    )
    fig.layout.margin.update(l=50, b=50)  # add bottom/left margin to fit axes titles

    fig.layout.update(hovermode="x", barmode="overlay")
    fig.layout.showlegend = stack is None and len(patterns) > 1
    fig.layout.legend.update(x=1, y=1, xanchor="right", yanchor="top")

    # move tick marks inside
    fig.update_xaxes(ticks="inside")
    fig.update_yaxes(ticks="inside")

    if stack:
        fig.update_xaxes(matches="x")
        fig.update_yaxes(matches="y")

    return fig
