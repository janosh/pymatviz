"""Module for plotting XRD patterns using plotly."""

from __future__ import annotations

from typing import Any, Literal, Union, get_args

import numpy as np
import plotly.graph_objects as go
from pymatgen.analysis.diffraction.xrd import DiffractionPattern, XRDCalculator
from pymatgen.core import Structure


PatternOrStruct = Union[DiffractionPattern, Structure]
HklFormat = Literal["compact", "full", None]
ValidHklFormats = HklCompact, HklFull, HklNone = get_args(HklFormat)


def format_hkl(hkl: tuple[int, int, int], format_type: HklFormat) -> str:
    """Format hkl indices as a string."""
    if format_type == "compact":
        return "".join(map(str, hkl))
    if format_type == "full":
        return f"({', '.join(map(str, hkl))})"
    if format_type is None:
        return ""
    raise ValueError(f"{format_type=} must be one of {ValidHklFormats}")


def plot_xrd_pattern(
    patterns: PatternOrStruct
    | dict[str, PatternOrStruct | tuple[PatternOrStruct, dict[str, Any]]],
    *,
    peak_width: float = 0.5,
    annotate_peaks: float = 5,
    hkl_format: HklFormat = HklCompact,
    show_angles: bool | None = None,
    wavelength: float = 1.54184,  # Cu K-alpha wavelength
) -> go.Figure:
    """Create a plotly figure of XRD patterns from DiffractionPattern, Structure
    objects, or a dictionary of them.

    Args:
        patterns: Either a single DiffractionPattern or Structure object, or a
            dictionary where keys are legend labels
            and values are either DiffractionPattern/Structure objects or tuples of
            (DiffractionPattern/Structure, kwargs) for customizing individual patterns.
        peak_width: Width of the diffraction peaks in degrees. Default is 0.5.
        annotate_peaks: Controls peak annotation. If int, annotates that many highest
            peaks. If float, should be in (0, 1) which will annotate peaks higher than
            that fraction of the highest peak. Default is 5.
        hkl_format: Format for hkl indices. One of 'compact' ('100'), 'full'
            ('(1, 0, 0)'), or None for no hkl indices. Default is 'compact' for 3 or
            fewer patterns, None for 4 or more patterns.
        show_angles: Whether to show angles in peak annotations. If None, it will
            default to True if plotting 1 or 2 patterns, False for 3 or more patterns.
        wavelength: X-ray wavelength for the XRD calculation (in Angstroms). Default is
            1.54184 (Cu K-alpha). Only used if patterns contains Structure objects.

    Raises:
        ValueError: If annotate_peaks is not a positive int or a float in (0, 1).
        TypeError: If patterns is not a DiffractionPattern, Structure or a dict of them.

    Returns:
        go.Figure: A plotly figure of the XRD pattern(s).
    """
    if (
        not isinstance(annotate_peaks, (int, float))
        or annotate_peaks <= 0
        or (isinstance(annotate_peaks, float) and annotate_peaks >= 1)
    ):
        raise ValueError(
            f"{annotate_peaks=} should be a positive int or a float in (0, 1)"
        )

    layout = dict(
        xaxis=dict(title="2θ (degrees)", tickmode="linear", tick0=0, dtick=10),
        yaxis=dict(title="Intensity (a.u.)", range=[0, 105]),
        hovermode="x",
        barmode="overlay",
    )
    fig = go.Figure(layout=layout)
    max_intensity = max_two_theta = 0

    # Convert single object to dict for uniform processing
    if isinstance(patterns, (DiffractionPattern, Structure)):
        patterns = {"XRD Pattern": patterns}
    elif not isinstance(patterns, dict):
        raise TypeError(
            f"{patterns=} should be a DiffractionPattern, Structure or a dict of them"
        )

    # Determine show_angles based on number of patterns
    if show_angles is None:
        show_angles = len(patterns) <= 2

    plotted_patterns: list[DiffractionPattern] = []
    for label, pattern_data in patterns.items():
        if isinstance(pattern_data, tuple):
            pattern_or_struct, trace_kwargs = pattern_data
        else:
            pattern_or_struct, trace_kwargs = pattern_data, {}

        if isinstance(pattern_or_struct, Structure):
            xrd_calculator = XRDCalculator(wavelength=wavelength)
            diffraction_pattern = xrd_calculator.get_pattern(pattern_or_struct)
        elif isinstance(pattern_or_struct, DiffractionPattern):
            diffraction_pattern = pattern_or_struct
        else:
            value = pattern_or_struct
            raise TypeError(
                f"{value=} should be a pymatgen Structure or DiffractionPattern"
            )

        plotted_patterns += [diffraction_pattern]
        two_theta = diffraction_pattern.x
        intensities = diffraction_pattern.y
        hkls = diffraction_pattern.hkls
        d_hkls = diffraction_pattern.d_hkls

        if intensities is None or len(intensities) == 0:
            raise ValueError(
                f"No intensities found in the diffraction pattern for {label}"
            )

        # Update max intensity and two_theta across all patterns
        max_intensity = max(max_intensity, *intensities)
        max_two_theta = max(max_two_theta, *two_theta)

        tooltips = [
            f"<b>{label}</b><br>"
            f"2θ: {x:.2f}°<br>Intensity: {y:.2f}<br>hkl: "
            f"{'<br>'.join(format_hkl(h['hkl'], HklFull) for h in hkl)}<br>d: {d:.3f} Å"
            for x, y, hkl, d in zip(two_theta, intensities, hkls, d_hkls)
        ]
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
            raise ValueError(
                f"{annotate_peaks=} should be a positive int or a float in (0, 1)"
            )

        for idx in peak_indices:
            x_pos = trace.x[idx]
            y_pos = trace.y[idx]

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

            fig.add_annotation(
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

    fig.layout.xaxis.range = [0, max_two_theta + 5]
    fig.layout.legend.update(x=1, y=1, xanchor="right", yanchor="top")

    return fig
