from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import plotly.graph_objects as go
import pytest
from pymatgen.analysis.diffraction.xrd import DiffractionPattern, XRDCalculator
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.utils import TEST_FILES


if TYPE_CHECKING:
    from typing import Any

MOCK_DIFFRACTION_PATTERN = DiffractionPattern(
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

BI2_ZR2_O7_STRUCT = Structure.from_file(
    f"{TEST_FILES}/xrd/Bi2Zr2O7-Fm3m-experimental-sqs.cif"
)
BI2_ZR2_O7_XRD = XRDCalculator().get_pattern(BI2_ZR2_O7_STRUCT)


@pytest.mark.parametrize(
    ("input_data", "expected_traces"),
    [
        (MOCK_DIFFRACTION_PATTERN, 1),
        (BI2_ZR2_O7_XRD, 1),
        (BI2_ZR2_O7_STRUCT, 1),
        ({"Structure": BI2_ZR2_O7_STRUCT, "Pattern": MOCK_DIFFRACTION_PATTERN}, 2),
    ],
)
def test_xrd_pattern_input_types(input_data: Any, expected_traces: int) -> None:
    fig = pmv.xrd_pattern(input_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces
    assert all(isinstance(trace, go.Bar) for trace in fig.data)


@pytest.mark.parametrize("peak_width", [0.3, 0.5, 0.8])
def test_xrd_pattern_peak_width(peak_width: float) -> None:
    fig = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN, peak_width=peak_width)
    assert fig.data[0].width == peak_width


@pytest.mark.parametrize("annotate_peaks", [0, 3, 5, 0.5, 0.8, -0.1, 1.5])
def test_xrd_pattern_annotate_peaks(annotate_peaks: float) -> None:
    if annotate_peaks < 0 or (
        isinstance(annotate_peaks, float) and annotate_peaks >= 1
    ):
        err_msg = re.escape(
            f"{annotate_peaks=} should be a positive int or a float in (0, 1)"
        )
        with pytest.raises(ValueError, match=err_msg):
            pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN, annotate_peaks=annotate_peaks)
    else:
        fig = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN, annotate_peaks=annotate_peaks)
        annotations = fig.layout.annotations
        if isinstance(annotate_peaks, int):
            assert len(annotations) == min(
                annotate_peaks, len(MOCK_DIFFRACTION_PATTERN.x)
            )
        else:
            assert len(annotations) > 0  # At least some peaks should be annotated


def test_xrd_pattern_hover_data() -> None:
    fig = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN)
    assert len(fig.data[0].hovertext) == len(MOCK_DIFFRACTION_PATTERN.x)
    assert all(
        all(key in text for key in ["2θ", "Intensity", "hkl", "d"])
        for text in fig.data[0].hovertext
    )


def test_xrd_pattern_layout_and_range() -> None:
    fig = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN)
    assert fig.layout.xaxis.title.text == "2θ (degrees)"
    assert fig.layout.yaxis.title.text == "Intensity (a.u.)"
    assert fig.layout.hovermode == "x"
    assert fig.layout.xaxis.range[0] == 0
    assert fig.layout.xaxis.range[1] == max(MOCK_DIFFRACTION_PATTERN.x) + 5
    assert fig.layout.yaxis.range == (0, 105)


@pytest.mark.parametrize(
    ("hkl_format", "expected_format", "show_angles"),
    [
        (pmv.xrd.HklCompact, r"\d{3}", True),
        (pmv.xrd.HklFull, r"\(\d, \d, \d\)", True),
        (pmv.xrd.HklNone, r"\d+\.\d+°", True),
        (pmv.xrd.HklCompact, r"\d{3}", False),
        (pmv.xrd.HklNone, None, False),
    ],
)
def test_xrd_pattern_annotation_format(
    hkl_format: pmv.xrd.HklFormat, expected_format: str | None, show_angles: bool
) -> None:
    fig = pmv.xrd_pattern(
        MOCK_DIFFRACTION_PATTERN, hkl_format=hkl_format, show_angles=show_angles
    )
    if hkl_format is pmv.xrd.HklNone and not show_angles:
        assert len(fig.layout.annotations) == 0
    else:
        for anno in fig.layout.annotations:
            if expected_format:
                assert re.search(expected_format, anno.text), f"{anno.text=}"
            if show_angles:
                assert re.search(r"\d+\.\d+°", anno.text), f"{anno.text=}"
            else:
                assert not re.search(r"\d+\.\d+°", anno.text), f"{anno.text=}"


def test_xrd_pattern_empty_input() -> None:
    empty_pattern = DiffractionPattern([], [], [], [])
    with pytest.raises(
        ValueError, match="No intensities found in the diffraction pattern"
    ):
        pmv.xrd_pattern(empty_pattern)


def test_xrd_pattern_intensity_normalization() -> None:
    original_max = max(MOCK_DIFFRACTION_PATTERN.y)
    fig = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN)
    normalized_max = max(fig.data[0].y)
    assert normalized_max == 100
    assert normalized_max / original_max == pytest.approx(100 / original_max)


@pytest.mark.parametrize("wavelength", [1.54184, 0.7093])
def test_xrd_pattern_wavelength(wavelength: float) -> None:
    fig = pmv.xrd_pattern(BI2_ZR2_O7_STRUCT, wavelength=wavelength)
    first_peak_position = fig.data[0].x[0]
    reference_fig = pmv.xrd_pattern(BI2_ZR2_O7_STRUCT, wavelength=1.54184)
    reference_first_peak = reference_fig.data[0].x[0]
    if wavelength != 1.54184:
        assert first_peak_position != reference_first_peak
    else:
        assert first_peak_position == reference_first_peak


def test_xrd_pattern_tooltip_content() -> None:
    patterns = {
        "Pattern 1": MOCK_DIFFRACTION_PATTERN,
        "Pattern 2": MOCK_DIFFRACTION_PATTERN,
    }
    fig = pmv.xrd_pattern(patterns, show_angles=False, hkl_format=None)

    for trace in fig.data:
        assert len(trace.hovertext) > 0
        for hover_text in trace.hovertext:
            assert all(key in hover_text for key in ["2θ", "Intensity", "hkl", "d"])
            assert "°" in hover_text  # Angles should always be in tooltips


def test_xrd_pattern_tooltip_label() -> None:
    patterns = {
        "Pattern 1": MOCK_DIFFRACTION_PATTERN,
        "Pattern 2": MOCK_DIFFRACTION_PATTERN,
    }
    fig = pmv.xrd_pattern(patterns)

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


@pytest.mark.parametrize(
    ("stack", "expected_rows", "expected_cols", "subplot_kwargs", "subtitle_kwargs"),
    [
        ("horizontal", 1, 2, None, None),
        ("vertical", 2, 1, None, None),
        (None, 1, 1, None, None),
        (
            "horizontal",
            1,
            2,
            {"horizontal_spacing": 0.05},
            {"font": {"size": 16, "color": "red"}},
        ),
        ("vertical", 2, 1, {"vertical_spacing": 0.1}, {"x": 0.1, "y": 0.9}),
    ],
)
def test_xrd_pattern_stack_and_kwargs(
    stack: Literal["horizontal", "vertical"] | None,
    expected_rows: int,
    expected_cols: int,
    subplot_kwargs: dict[str, Any] | None,
    subtitle_kwargs: dict[str, Any] | None,
) -> None:
    patterns = {
        "Pattern 1": MOCK_DIFFRACTION_PATTERN,
        "Pattern 2": MOCK_DIFFRACTION_PATTERN,
    }
    fig = pmv.xrd_pattern(
        patterns,
        stack=stack,
        subplot_kwargs=subplot_kwargs,
        subtitle_kwargs=subtitle_kwargs,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2

    if stack:
        actual_rows = len(fig._grid_ref)
        actual_cols = len(fig._grid_ref[0])
        assert actual_rows == expected_rows
        assert actual_cols == expected_cols

        for i, trace in enumerate(fig.data):
            expected_xaxis = f"x{i+1 if i > 0 else ''}"
            expected_yaxis = f"y{i+1 if i > 0 else ''}"
            assert trace.xaxis == expected_xaxis, f"{trace.xaxis=}"
            assert trace.yaxis == expected_yaxis, f"{trace.yaxis=}"

        assert fig.layout.xaxis.matches == "x"
        assert fig.layout.yaxis.matches == "y"

        # Check subtitle annotations if subtitle_kwargs is provided
        if subtitle_kwargs:
            subtitle_annotations = [
                anno for anno in fig.layout.annotations if anno.text in patterns
            ]
            assert len(subtitle_annotations) == 2
            for anno in subtitle_annotations:
                if "font" in subtitle_kwargs:
                    assert anno.font.size == subtitle_kwargs["font"].get("size")
                    assert anno.font.color == subtitle_kwargs["font"].get("color")
                if "x" in subtitle_kwargs:
                    assert anno.x == subtitle_kwargs["x"]
                if "y" in subtitle_kwargs:
                    assert anno.y == subtitle_kwargs["y"]
    else:
        assert fig.layout.grid == go.layout.Grid()
        for trace in fig.data:
            assert trace.xaxis is None, f"{trace.xaxis=}"
            assert trace.yaxis is None, f"{trace.yaxis=}"
