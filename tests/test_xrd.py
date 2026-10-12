from __future__ import annotations

import re
from collections import OrderedDict
from typing import TYPE_CHECKING, Literal

import numpy as np
import plotly.graph_objects as go
import pytest
from pymatgen.analysis.diffraction.xrd import DiffractionPattern, XRDCalculator
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

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
ORDERED_XRD_INPUT = OrderedDict(
    [("Structure", BI2_ZR2_O7_STRUCT), ("Pattern", MOCK_DIFFRACTION_PATTERN)]
)


@pytest.mark.parametrize(
    ("input_data", "expected_traces"),
    [
        (MOCK_DIFFRACTION_PATTERN, 1),
        (BI2_ZR2_O7_XRD, 1),
        (BI2_ZR2_O7_STRUCT, 1),
        (AseAtomsAdaptor.get_atoms(BI2_ZR2_O7_STRUCT), 1),
        ({"Structure": BI2_ZR2_O7_STRUCT, "Pattern": MOCK_DIFFRACTION_PATTERN}, 2),
        (ORDERED_XRD_INPUT, 2),
    ],
)
def test_xrd_pattern_input_types(input_data: Any, expected_traces: int) -> None:
    fig = pmv.xrd_pattern(input_data)
    assert len(fig.data) == expected_traces
    assert all(isinstance(trace, go.Bar) for trace in fig.data)


@pytest.mark.parametrize("peak_width", [0.3, 0.5, 0.8])
def test_xrd_pattern_peak_width(peak_width: float) -> None:
    fig = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN, peak_width=peak_width)
    assert fig.data[0].width == peak_width


@pytest.mark.parametrize(
    ("annotate_peaks", "expected_peaks"),
    [
        (0, 0),
        (3, 3),
        (5, 5),
        (0.5, 3),
        (0.8, 1),
        (-0.1, None),
        (1.5, None),
        (np.nan, None),
        (np.inf, None),
    ],
)
def test_xrd_pattern_annotate_peaks(
    annotate_peaks: float, expected_peaks: int | None
) -> None:
    """Select peaks by count or intensity and reject invalid limits."""
    if expected_peaks is None:
        err_msg = re.escape(
            f"{annotate_peaks=} should be a positive int or a float in (0, 1)"
        )
        with pytest.raises(ValueError, match=err_msg):
            pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN, annotate_peaks=annotate_peaks)
    else:
        fig = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN, annotate_peaks=annotate_peaks)
        annotations = fig.layout.annotations

        assert len(annotations) == expected_peaks + 2  # Includes both axis titles.
        # Axis title annotations are identified by not having an arrowhead
        axis_texts = {anno.text for anno in annotations if not anno.arrowhead}
        assert "2θ (degrees)" in axis_texts
        assert "Intensity (a.u.)" in axis_texts


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
    with pytest.raises(ValueError, match="patterns must not be empty"):
        pmv.xrd_pattern({})


def test_xrd_pattern_intensity_normalization() -> None:
    """Intensities scale to a max of 100 and invalid intensities raise."""
    fig = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN)
    original_y = np.asarray(MOCK_DIFFRACTION_PATTERN.y)
    np.testing.assert_array_equal(fig.data[0].y, original_y / original_y.max() * 100)

    zero_pattern = DiffractionPattern([10], [0], [[{"hkl": (1, 0, 0)}]], [1])
    combined = pmv.xrd_pattern(
        {"zero": zero_pattern, "nonzero": MOCK_DIFFRACTION_PATTERN}
    )
    assert list(combined.data[0].y) == [0]
    for intensity in [0, -1, np.nan, np.inf]:
        pattern = DiffractionPattern([10], [intensity], [[{"hkl": (1, 0, 0)}]], [1])
        with pytest.raises(ValueError, match=r"intensit|Intensit"):
            pmv.xrd_pattern(pattern)


def test_xrd_pattern_wavelength() -> None:
    """Default wavelength is Cu K-alpha and peaks shift per Bragg's law."""
    default_peaks = pmv.xrd_pattern(BI2_ZR2_O7_STRUCT).data[0].x
    first_peaks = {
        wavelength: pmv.xrd_pattern(BI2_ZR2_O7_STRUCT, wavelength=wavelength)
        .data[0]
        .x[0]
        for wavelength in (cu_k_alpha_wavelength, 0.7093)
    }
    assert default_peaks[0] == first_peaks[cu_k_alpha_wavelength]
    # same lattice plane d = λ / (2 sin θ) for both wavelengths
    d_spacings = [
        wavelength / (2 * np.sin(np.radians(two_theta / 2)))
        for wavelength, two_theta in first_peaks.items()
    ]
    # observed rel. difference ~3e-16 (1.5 eps)
    assert d_spacings[0] == pytest.approx(d_spacings[1], rel=1e-12)


def test_xrd_pattern_tooltips() -> None:
    """Tooltips show label, angle, intensity, hkl and d regardless of annotations."""
    patterns = {
        "Pattern 1": MOCK_DIFFRACTION_PATTERN,
        "Pattern 2": MOCK_DIFFRACTION_PATTERN,
    }
    fig = pmv.xrd_pattern(patterns, show_angles=False, hkl_format=None)

    assert len(fig.data) == 2
    for trace in fig.data:
        assert len(trace.hovertext) == len(MOCK_DIFFRACTION_PATTERN.x)
        for hover_text in trace.hovertext:
            assert hover_text.startswith(f"<b>{trace.name}</b>")
            assert all(key in hover_text for key in ["2θ", "Intensity", "hkl", "d"])
            assert "°" in hover_text  # Angles should always be in tooltips


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
    trace_kwargs = {"marker_color": "red"}
    patterns = {
        "Pattern 1": (MOCK_DIFFRACTION_PATTERN, trace_kwargs),
        "Pattern 2": MOCK_DIFFRACTION_PATTERN,
    }
    fig = pmv.xrd_pattern(
        patterns,
        stack=stack,
        subplot_kwargs=subplot_kwargs,
        subtitle_kwargs=subtitle_kwargs,
    )

    assert len(fig.data) == 2
    assert trace_kwargs == {"marker_color": "red"}

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


@pytest.mark.parametrize(
    ("axis_title_kwargs", "font_size", "font_color"),
    [
        (None, 12, None),
        ({}, 12, None),
        ({"font_color": "green"}, 12, "green"),
        ({"font_size": 18, "font_color": "blue"}, 18, "blue"),
    ],
)
def test_xrd_pattern_axis_titles(
    axis_title_kwargs: dict[str, Any] | None, font_size: int, font_color: str | None
) -> None:
    """Axis titles are paper-positioned annotations styled by axis_title_kwargs."""
    fig = pmv.xrd_pattern(MOCK_DIFFRACTION_PATTERN, axis_title_kwargs=axis_title_kwargs)
    annos = {anno.text: anno for anno in fig.layout.annotations}
    x_title, y_title = annos["2θ (degrees)"], annos["Intensity (a.u.)"]

    for anno in (x_title, y_title):
        assert (anno.xref, anno.yref) == ("paper", "paper")
        assert anno.font.size == font_size
        assert anno.font.color == font_color
    assert (x_title.x, y_title.x) == (0.5, -0.07)
    assert y_title.textangle == -90
    assert fig.layout.hovermode == "x"
