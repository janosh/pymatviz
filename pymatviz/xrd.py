"""Module for plotting XRD patterns using plotly."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from pymatgen.analysis.diffraction.xrd import DiffractionPattern


def plot_xrd_pattern(
    patterns: DiffractionPattern
    | dict[str, DiffractionPattern | tuple[DiffractionPattern, dict[str, Any]]],
    peak_width: float = 0.5,
    annotate_peaks: float = 5,
) -> go.Figure:
    """Create a plotly figure of XRD patterns from a DiffractionPattern object or a
    dictionary of DiffractionPattern objects.

    Args:
        patterns (DiffractionPattern | dict[str, DiffractionPattern | tuple[DiffractionPattern, dict]]):
            Either a single DiffractionPattern object
            or a dictionary where keys are legend labels and values are either
            DiffractionPattern objects or tuples of (DiffractionPattern, kwargs) for
            customizing individual patterns.
        peak_width (float): Width of the diffraction peaks in degrees. Default is 0.5.
        annotate_peaks (int | float): Controls peak annotation. If int, annotates that
            many highest peaks. If float, should be in (0, 1) which will annotate peaks
            higher than that fraction of the highest peak. Default is 5.

    Raises:
        ValueError: If annotate_peaks is not a positive int or a float in (0, 1).
        TypeError: If patterns is not a DiffractionPattern or a dictionary of
        DiffractionPatterns or tuples of (DiffractionPattern, kwargs).

    Returns:
        go.Figure: A plotly figure of the XRD pattern(s).
    """  # noqa: E501
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

    # Convert single DiffractionPattern to dict for uniform processing
    if isinstance(patterns, DiffractionPattern):
        patterns = {"XRD Pattern": patterns}
    elif not isinstance(patterns, dict):
        raise TypeError(
            f"{patterns=} should be a DiffractionPattern object or a dictionary of them"
        )

    for label, pattern_data in patterns.items():
        if isinstance(pattern_data, tuple):
            diffraction_pattern, trace_kwargs = pattern_data
        else:
            diffraction_pattern, trace_kwargs = pattern_data, {}

        if not isinstance(diffraction_pattern, DiffractionPattern):
            raise TypeError(
                f"{diffraction_pattern=} should be a pymatgen DiffractionPattern object"
            )

        two_theta = diffraction_pattern.x
        intensities = diffraction_pattern.y
        hkls = diffraction_pattern.hkls
        d_hkls = diffraction_pattern.d_hkls

        if intensities is None or len(intensities) == 0:
            raise ValueError(
                f"No intensities found in the diffraction pattern for {label}"
            )

        # Update max intensity and two_theta
        max_intensity = max(max_intensity, *intensities)
        max_two_theta = max(max_two_theta, *two_theta)

        # Create the trace for this pattern
        trace = go.Bar(
            x=two_theta,
            y=intensities,
            width=peak_width,
            name=label,
            hovertext=[
                f"2θ: {x:.2f}°<br>Intensity: {y:.2f}<br>hkl: "
                f"{', '.join(str(h['hkl']) for h in hkl)}<br>d: {d:.3f} Å"
                for x, y, hkl, d in zip(two_theta, intensities, hkls, d_hkls)
            ],
            **trace_kwargs,
        )
        fig.add_trace(trace)

    # Normalize intensities to 100 and add annotations
    for trace in fig.data:
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
            hkl_str = trace.hovertext[idx].split("hkl: ")[1].split("<br>")[0]
            x_pos = trace.x[idx]
            y_pos = trace.y[idx]

            annotation_text = f"{hkl_str}<br>{x_pos:.2f}°"

            ax, ay = 20, -20
            xanchor, yanchor = "left", "bottom"
            if x_pos > max_two_theta * 0.9:
                ax, ay = -20, -20
                xanchor, yanchor = "right", "bottom"
            elif y_pos > 90:
                ax, ay = 80, 20
                xanchor, yanchor = "right", "bottom"

            fig.add_annotation(
                x=x_pos,
                y=y_pos,
                text=annotation_text,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                ax=ax,
                ay=ay,
                xanchor=xanchor,
                yanchor=yanchor,
            )

    fig.layout.xaxis.range = [0, max_two_theta + 5]
    fig.layout.legend.update(x=1, y=1, xanchor="right", yanchor="top")

    return fig
