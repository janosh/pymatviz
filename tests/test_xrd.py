from __future__ import annotations

import re
from typing import Any

import plotly.graph_objects as go
import pytest
from pymatgen.analysis.diffraction.xrd import DiffractionPattern, XRDCalculator
from pymatgen.core import Structure

from pymatviz import plot_xrd_pattern
from pymatviz.utils import TEST_FILES
from pymatviz.xrd import HklCompact, HklFormat, HklFull, HklNone


mock_diffraction_pattern = DiffractionPattern(
    x=[10, 20, 30, 40, 50],
    y=[100, 50, 75, 25, 60],
    hkls=[
        [{"hkl": (1, 0, 0)}],
        [{"hkl": (1, 1, 0)}],
        [{"hkl": (1, 1, 1)}],
        [{"hkl": (2, 0, 0)}],
        [{"hkl": (2, 1, 0)}],
    ],
    d_hkls=[2.5, 2.0, 1.8, 1.5, 1.3],
)

bi2_zr2_o7_struct = Structure.from_file(
    f"{TEST_FILES}/xrd/Bi2Zr2O7-Fm3m-experimental-sqs.cif"
)
bi2_zr2_o7_xrd = XRDCalculator().get_pattern(bi2_zr2_o7_struct)


@pytest.mark.parametrize(
    "input_data, expected_traces",
    [
        (mock_diffraction_pattern, 1),
        (bi2_zr2_o7_xrd, 1),
        (bi2_zr2_o7_struct, 1),
        ({"Structure": bi2_zr2_o7_struct, "Pattern": mock_diffraction_pattern}, 2),
    ],
)
def test_plot_xrd_pattern_input_types(input_data: Any, expected_traces: int) -> None:
    fig = plot_xrd_pattern(input_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces
    assert all(isinstance(trace, go.Bar) for trace in fig.data)


@pytest.mark.parametrize("peak_width", [0.3, 0.5, 0.8])
def test_plot_xrd_pattern_peak_width(peak_width: float) -> None:
    fig = plot_xrd_pattern(mock_diffraction_pattern, peak_width=peak_width)
    assert fig.data[0].width == peak_width


@pytest.mark.parametrize("annotate_peaks", [3, 5, 0.5, 0.8])
def test_plot_xrd_pattern_annotate_peaks(annotate_peaks: float) -> None:
    fig = plot_xrd_pattern(mock_diffraction_pattern, annotate_peaks=annotate_peaks)
    annotations = fig.layout.annotations
    if isinstance(annotate_peaks, int):
        assert len(annotations) == min(annotate_peaks, len(mock_diffraction_pattern.x))
    else:
        assert len(annotations) > 0  # At least some peaks should be annotated


def test_plot_xrd_pattern_hover_data() -> None:
    fig = plot_xrd_pattern(mock_diffraction_pattern)
    assert len(fig.data[0].hovertext) == len(mock_diffraction_pattern.x)
    assert all(
        all(key in text for key in ["2θ", "Intensity", "hkl", "d"])
        for text in fig.data[0].hovertext
    )


def test_plot_xrd_pattern_layout_and_range() -> None:
    fig = plot_xrd_pattern(mock_diffraction_pattern)
    assert fig.layout.xaxis.title.text == "2θ (degrees)"
    assert fig.layout.yaxis.title.text == "Intensity (a.u.)"
    assert fig.layout.hovermode == "x"
    assert fig.layout.xaxis.range[0] == 0
    assert fig.layout.xaxis.range[1] == max(mock_diffraction_pattern.x) + 5
    assert fig.layout.yaxis.range == (0, 105)


@pytest.mark.parametrize(
    "hkl_format, expected_format, show_angles",
    [
        (HklCompact, r"\d{3}", True),
        (HklFull, r"\(\d, \d, \d\)", True),
        (HklNone, r"\d+\.\d+°", True),
        (HklCompact, r"\d{3}", False),
        (HklNone, None, False),
    ],
)
def test_plot_xrd_pattern_annotation_format(
    hkl_format: HklFormat, expected_format: str | None, show_angles: bool
) -> None:
    fig = plot_xrd_pattern(
        mock_diffraction_pattern, hkl_format=hkl_format, show_angles=show_angles
    )
    if hkl_format is HklNone and not show_angles:
        assert len(fig.layout.annotations) == 0
    else:
        for annotation in fig.layout.annotations:
            if expected_format:
                assert re.search(
                    expected_format, annotation.text
                ), f"{annotation.text=}"
            if show_angles:
                assert re.search(r"\d+\.\d+°", annotation.text), f"{annotation.text=}"
            else:
                assert not re.search(
                    r"\d+\.\d+°", annotation.text
                ), f"{annotation.text=}"


def test_plot_xrd_pattern_empty_input() -> None:
    empty_pattern = DiffractionPattern([], [], [], [])
    with pytest.raises(
        ValueError, match="No intensities found in the diffraction pattern"
    ):
        plot_xrd_pattern(empty_pattern)


def test_plot_xrd_pattern_intensity_normalization() -> None:
    original_max = max(mock_diffraction_pattern.y)
    fig = plot_xrd_pattern(mock_diffraction_pattern)
    normalized_max = max(fig.data[0].y)
    assert normalized_max == 100
    assert normalized_max / original_max == pytest.approx(100 / original_max)


@pytest.mark.parametrize("wavelength", [1.54184, 0.7093])
def test_plot_xrd_pattern_wavelength(wavelength: float) -> None:
    fig = plot_xrd_pattern(bi2_zr2_o7_struct, wavelength=wavelength)
    first_peak_position = fig.data[0].x[0]
    reference_fig = plot_xrd_pattern(bi2_zr2_o7_struct, wavelength=1.54184)
    reference_first_peak = reference_fig.data[0].x[0]
    if wavelength != 1.54184:
        assert first_peak_position != reference_first_peak
    else:
        assert first_peak_position == reference_first_peak


def test_plot_xrd_pattern_tooltip_content() -> None:
    patterns = {
        "Pattern 1": mock_diffraction_pattern,
        "Pattern 2": mock_diffraction_pattern,
    }
    fig = plot_xrd_pattern(patterns, show_angles=False, hkl_format=None)

    for trace in fig.data:
        assert len(trace.hovertext) > 0
        for hover_text in trace.hovertext:
            assert all(key in hover_text for key in ["2θ", "Intensity", "hkl", "d"])
            assert "°" in hover_text  # Angles should always be in tooltips


def test_plot_xrd_pattern_tooltip_label() -> None:
    patterns = {
        "Pattern 1": mock_diffraction_pattern,
        "Pattern 2": mock_diffraction_pattern,
    }
    fig = plot_xrd_pattern(patterns)

    assert len(fig.data) == 2

    for trace in fig.data:
        assert all(
            hover_text.startswith(f"<b>{trace.name}</b>")
            for hover_text in trace.hovertext
        )
        assert all(
            all(key in hover_text for key in ["2θ", "Intensity", "hkl", "d"])
            for hover_text in trace.hovertext
        )
