"""Module for plotting XRD patterns using plotly."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from pymatgen.analysis.diffraction.xrd import DiffractionPattern


def plot_xrd_pattern(
    diffraction_pattern: DiffractionPattern,
    peak_width: float = 0.5,
    annotate_peaks: float = 5,
) -> go.Figure:
    """Create a plotly figure of an XRD pattern from a pymatgen DiffractionPattern
    object.

    Args:
        diffraction_pattern (DiffractionPattern): A pymatgen DiffractionPattern object.
        peak_width (float): Width of the diffraction peaks in degrees. Default is 0.5.
        annotate_peaks (int | float): Controls peak annotation. If int, annotates that
            many highest peaks. If float, should be in (0, 1) which will annotate peaks
            higher than that fraction of the highest peak. Default is 5.

    Raises:
        ValueError: If annotate_peaks is not a positive int or a float in (0, 1).

    Returns:
        go.Figure: A plotly figure of the XRD pattern.
    """
    if not isinstance(diffraction_pattern, DiffractionPattern):
        raise TypeError(
            f"{diffraction_pattern=} should be a pymatgen DiffractionPattern object"
        )

    two_theta = diffraction_pattern.x
    intensities = diffraction_pattern.y
    hkls = diffraction_pattern.hkls
    d_hkls = diffraction_pattern.d_hkls
    if intensities is None or len(intensities) == 0:
        raise ValueError("No intensities found in the diffraction pattern")

    # Normalize intensities to 100
    max_intensity = max(intensities)
    intensities = [i / max_intensity * 100 for i in intensities]

    # Create the main trace
    trace = go.Bar(
        x=two_theta,
        y=intensities,
        width=peak_width,
        name="XRD Pattern",
        marker_color="rgba(0, 0, 0, 0.7)",
        hovertext=[
            f"2θ: {x:.2f}°<br>Intensity: {y:.2f}<br>hkl: "
            f"{', '.join(str(h['hkl']) for h in hkl)}<br>d: {d:.3f} Å"
            for x, y, hkl, d in zip(two_theta, intensities, hkls, d_hkls)
        ],
    )

    # Prepare peak annotations
    annotations = []
    if isinstance(annotate_peaks, int) and annotate_peaks > 0:
        peak_indices = np.argsort(intensities)[-annotate_peaks:]
    elif 0 < annotate_peaks < 1:
        peak_indices = [
            idx
            for idx, intensity in enumerate(intensities)
            if intensity > annotate_peaks * 100
        ]
    else:
        raise ValueError(
            f"{annotate_peaks=} should be a positive int or a float in (0, 1)"
        )

    x_range = max(two_theta) + 5  # x-axis range
    for idx in peak_indices:
        hkl = hkls[idx]
        hkl_str = ", ".join(str(h["hkl"]) for h in hkl)
        x_pos = two_theta[idx]
        y_pos = intensities[idx]

        # add reflection angle to annotation
        annotation_text = f"{hkl_str}<br>{x_pos:.2f}°"

        # Determine arrow direction and text position
        ax, ay = 20, -20  # 45 degree angle inward
        xanchor, yanchor = "left", "bottom"
        if x_pos > x_range * 0.9:  # Right 10% of the plot
            ax, ay = -20, -20  # 45 degree angle inward
            xanchor, yanchor = "right", "bottom"
        elif y_pos > 90:  # move anno down to the right if it's too close to the top
            ax, ay = 80, 20  # 45 degree angle inward
            xanchor, yanchor = "right", "bottom"

        annotation = dict(
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
        annotations.append(annotation)

    # Set up the layout
    layout = dict(
        xaxis=dict(
            title="2θ (degrees)",
            range=[0, x_range],
            tickmode="linear",
            tick0=0,
            dtick=10,
        ),
        # set y_max > 100 to show full peaks
        yaxis=dict(title="Intensity (a.u.)", range=[0, 105]),
        hovermode="x",  # make tooltip show up for any y value, even above the peak
        annotations=annotations,
    )

    return go.Figure(data=[trace], layout=layout)
