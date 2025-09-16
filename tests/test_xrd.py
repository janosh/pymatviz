from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import plotly.graph_objects as go
import pytest
from pymatgen.analysis.diffraction.xrd import DiffractionPattern, XRDCalculator
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.utils.testing import TEST_FILES
from pymatviz.xrd import cu_k_alpha_wavelength


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

        n_peak_annotations_expected = 0
        if isinstance(annotate_peaks, int):
            if annotate_peaks > 0:
                n_peak_annotations_expected = min(
                    annotate_peaks, len(MOCK_DIFFRACTION_PATTERN.x)
                )
        elif 0 < annotate_peaks < 1:  # float case
            y_values = MOCK_DIFFRACTION_PATTERN.y
            if y_values.size > 0:
                max_intensity = max(y_values)
                if max_intensity > 0:
                    normalized_intensities = [
                        (y_val / max_intensity) * 100 for y_val in y_values
                    ]
                    n_peak_annotations_expected = sum(
                        1
                        for intensity in normalized_intensities
                        if intensity > annotate_peaks * 100
                    )
                # else: all intensities are 0 or less, so 0 peak annotations
            # else: y_values is empty, so 0 peak annotations

        # Total annotations = peak annotations + 2 for axis titles
        assert len(annotations) == n_peak_annotations_expected + 2
        # Axis title annotations are identified by not having an arrowhead
        axis_texts = {anno.text for anno in annotations if not anno.arrowhead}
        assert "2θ (degrees)" in axis_texts
        assert "Intensity (a.u.)" in axis_texts


def test_xrd_pattern_hover_data() -> None:
    fig = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN)
    assert len(fig.data[0].hovertext) == len(MOCK_DIFFRACTION_PATTERN.x)
    assert all(
        all(key in text for key in ["2θ", "Intensity", "hkl", "d"])
        for text in fig.data[0].hovertext
    )


def test_xrd_pattern_layout_and_range() -> None:
    fig = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN)
    # Axis titles are now annotations
    axes_annotations = [
        anno
        for anno in fig.layout.annotations
        if anno.text in ("2θ (degrees)", "Intensity (a.u.)")
    ]
    assert len(axes_annotations) == 2
    x_axis_anno = next(anno for anno in axes_annotations if anno.text == "2θ (degrees)")
    y_axis_anno = next(
        anno for anno in axes_annotations if anno.text == "Intensity (a.u.)"
    )

    assert x_axis_anno.xref == "paper"
    assert x_axis_anno.yref == "paper"
    assert y_axis_anno.xref == "paper"
    assert y_axis_anno.yref == "paper"
    assert y_axis_anno.textangle == -90

    assert fig.layout.hovermode == "x"
    # Axis ranges are no longer explicitly set in xrd.py; relying on Plotly defaults.


@pytest.mark.parametrize(
    ("hkl_format", "expected_format", "show_angles"),
    [
        (pmv.xrd.HklCompact, r"\d{3}", True),
        (pmv.xrd.HklFull, r"\(\d, \d, \d\)", True),
        (None, r"\d+\.\d+°", True),
        (pmv.xrd.HklCompact, r"\d{3}", False),
        (None, None, False),
    ],
)
def test_xrd_pattern_annotation_format(
    hkl_format: pmv.xrd.HklFormat, expected_format: str | None, show_angles: bool
) -> None:
    fig = pmv.xrd_pattern(
        MOCK_DIFFRACTION_PATTERN, hkl_format=hkl_format, show_angles=show_angles
    )

    # Filter out axis title annotations (no arrowheads).
    peak_annotations = [
        anno
        for anno in fig.layout.annotations
        if anno.arrowhead is not None and anno.ax is not None
    ]
    axis_annotations = [
        anno for anno in fig.layout.annotations if anno.arrowhead is None
    ]

    assert len(axis_annotations) == 2  # Expect 2 axis title annotations
    assert "2θ (degrees)" in [anno.text for anno in axis_annotations]
    assert "Intensity (a.u.)" in [anno.text for anno in axis_annotations]

    if hkl_format is None and not show_angles:
        # No peak annotations if hkl_format is None and show_angles is False.
        assert len(peak_annotations) == 0
    else:
        # Expect peak annotations (default annotate_peaks=5 for 5 mock peaks).
        assert len(peak_annotations) > 0, "Expected peak annotations"
        for anno in peak_annotations:
            if expected_format:  # Regex for HKL or angle (if HKL is None)
                assert re.search(expected_format, anno.text), (
                    f"Format '{expected_format}' not in '{anno.text}'"
                )

            angle_pattern = r"\d+\.\d+°"
            contains_angle = re.search(angle_pattern, anno.text) is not None

            if show_angles:
                assert contains_angle, (
                    f"Angle missing in '{anno.text}' when show_angles=True"
                )
            # If HKLs are shown (hkl_format is not None) and show_angles is False,
            # then angles should not be present in the peak annotation.
            elif hkl_format is not None:
                assert not contains_angle, (
                    f"Angle found in '{anno.text}' when show_angles=False and "
                    "hkl_format specified"
                )


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


@pytest.mark.parametrize("wavelength", [cu_k_alpha_wavelength, 0.7093])
def test_xrd_pattern_wavelength(wavelength: float) -> None:
    fig = pmv.xrd_pattern(BI2_ZR2_O7_STRUCT, wavelength=wavelength)
    first_peak_position = fig.data[0].x[0]
    reference_fig = pmv.xrd_pattern(BI2_ZR2_O7_STRUCT, wavelength=cu_k_alpha_wavelength)
    reference_first_peak = reference_fig.data[0].x[0]
    if wavelength != cu_k_alpha_wavelength:
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
        assert fig._grid_ref is not None
        actual_rows = len(fig._grid_ref)
        actual_cols = len(fig._grid_ref[0])
        assert actual_rows == expected_rows
        assert actual_cols == expected_cols

        for idx, trace in enumerate(fig.data):
            expected_xaxis = f"x{idx + 1 if idx > 0 else ''}"
            expected_yaxis = f"y{idx + 1 if idx > 0 else ''}"
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
                expected_font_size = subtitle_kwargs.get("font_size", 12)
                # Color comes from subtitle_kwargs if specified
                expected_font_color = None
                if "font" in subtitle_kwargs and isinstance(
                    subtitle_kwargs["font"], dict
                ):
                    expected_font_color = subtitle_kwargs["font"].get("color")

                assert anno.font.size == expected_font_size
                if expected_font_color:
                    assert anno.font.color == expected_font_color

                if "x" in subtitle_kwargs:
                    assert anno.x == subtitle_kwargs["x"]
                if "y" in subtitle_kwargs:
                    assert anno.y == subtitle_kwargs["y"]
    else:
        assert fig.layout.grid == go.layout.Grid()
        for trace in fig.data:
            assert trace.xaxis is None, f"{trace.xaxis=}"
            assert trace.yaxis is None, f"{trace.yaxis=}"


def test_xrd_pattern_axis_kwargs() -> None:
    """Test custom axis_kwargs."""
    custom_style = {"font_size": 18, "font_color": "blue"}
    fig = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN, axis_title_kwargs=custom_style)

    x_axis_anno = next(
        anno for anno in fig.layout.annotations if anno.text == "2θ (degrees)"
    )
    y_axis_anno = next(
        anno for anno in fig.layout.annotations if anno.text == "Intensity (a.u.)"
    )

    assert x_axis_anno.font.size == custom_style["font_size"]
    assert x_axis_anno.font.color == custom_style["font_color"]
    assert y_axis_anno.font.size == custom_style["font_size"]
    assert y_axis_anno.font.color == custom_style["font_color"]
    assert x_axis_anno.xref == "paper"
    assert y_axis_anno.textangle == -90
    assert x_axis_anno.x == 0.5
    assert y_axis_anno.x < 0  # Default is -0.07 in xrd.py

    # Ensure axis titles appear with empty axis_kwargs.
    fig_empty_kwargs = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN, axis_title_kwargs={})
    assert any(
        anno.text == "2θ (degrees)" for anno in fig_empty_kwargs.layout.annotations
    )
    assert any(
        anno.text == "Intensity (a.u.)" for anno in fig_empty_kwargs.layout.annotations
    )
    # Check default font size (12) with empty kwargs.
    default_x_title = next(
        anno
        for anno in fig_empty_kwargs.layout.annotations
        if anno.text == "2θ (degrees)"
    )
    assert default_x_title.font.size == 12

    # Check default font_size (12) used when not in axis_kwargs, while other props apply
    # (Note: x/y in axis_kwargs would cause TypeError due to xrd.py structure).
    fig_custom_color = pmv.xrd_pattern(
        MOCK_DIFFRACTION_PATTERN, axis_title_kwargs={"font_color": "green"}
    )
    x_custom_color_anno = next(
        anno
        for anno in fig_custom_color.layout.annotations
        if anno.text == "2θ (degrees)"
    )
    assert x_custom_color_anno.font.size == 12  # Default size from xrd.py
    assert x_custom_color_anno.font.color == "green"
    assert x_custom_color_anno.x == 0.5  # Default x position from xrd.py
