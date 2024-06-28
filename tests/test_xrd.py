import re

import plotly.graph_objects as go
import pytest
from pymatgen.analysis.diffraction.xrd import DiffractionPattern, XRDCalculator
from pymatgen.core import Structure

from pymatviz import plot_xrd_pattern
from pymatviz.utils import TEST_FILES


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

bi2_zr2_o7_struct = Structure.from_file(f"{TEST_FILES}/xrd/Bi2Zr2O7-Fm3m-sqs.cif")
bi2_zr2_o7_xrd = XRDCalculator().get_pattern(bi2_zr2_o7_struct)


@pytest.mark.parametrize("diffract_patt", [mock_diffraction_pattern, bi2_zr2_o7_xrd])
def test_plot_xrd_pattern_basic(diffract_patt: DiffractionPattern) -> None:
    fig = plot_xrd_pattern(diffract_patt)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert isinstance(fig.data[0], go.Bar)


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


@pytest.mark.parametrize("diffract_patt", [mock_diffraction_pattern, bi2_zr2_o7_xrd])
def test_plot_xrd_pattern_hover_data(diffract_patt: DiffractionPattern) -> None:
    fig = plot_xrd_pattern(diffract_patt)
    assert len(fig.data[0].hovertext) == len(diffract_patt.x)
    assert all(
        "2θ" in text and "Intensity" in text and "hkl" in text and "d" in text
        for text in fig.data[0].hovertext
    )


def test_plot_xrd_pattern_layout() -> None:
    fig = plot_xrd_pattern(mock_diffraction_pattern)
    assert fig.layout.xaxis.title.text == "2θ (degrees)"
    assert fig.layout.yaxis.title.text == "Intensity (a.u.)"
    assert fig.layout.hovermode == "x"


def test_plot_xrd_pattern_x_range() -> None:
    fig = plot_xrd_pattern(mock_diffraction_pattern)
    assert fig.layout.xaxis.range[0] == 0
    assert fig.layout.xaxis.range[1] == max(mock_diffraction_pattern.x) + 5


def test_plot_xrd_pattern_y_range() -> None:
    fig = plot_xrd_pattern(mock_diffraction_pattern)
    assert fig.layout.yaxis.range[0] == 0
    assert fig.layout.yaxis.range[1] == 105  # Slightly above 100 to show full peaks


@pytest.mark.parametrize(
    "x_pos, expected_direction",
    [
        (1, (80, 20)),  # Left 10% of the plot
        (95, (-20, -20)),  # Right 10% of the plot
        (50, (-20, -20)),  # Middle, but high intensity
    ],
)
def test_plot_xrd_pattern_annotation_directions(
    x_pos: float, expected_direction: tuple[int, int]
) -> None:
    # Modify mock data to test different scenarios
    mock_diffraction_pattern.x = [x_pos]
    mock_diffraction_pattern.y = [95]  # High intensity
    mock_diffraction_pattern.hkls = [[{"hkl": (1, 1, 1)}]]
    mock_diffraction_pattern.d_hkls = [2.0]

    fig = plot_xrd_pattern(mock_diffraction_pattern)
    annotation = fig.layout.annotations[0]
    assert (annotation.ax, annotation.ay) == expected_direction


def test_plot_xrd_pattern_annotation_text() -> None:
    fig = plot_xrd_pattern(mock_diffraction_pattern)
    for annotation in fig.layout.annotations:
        assert any(
            f"({h}, {k}, {l})" in annotation.text
            for h, k, l in [(1, 0, 0), (1, 1, 0), (1, 1, 1), (2, 0, 0), (2, 1, 0)]  # noqa: E741
        )
        assert re.match(
            r".*<br>\d+\.\d+°", annotation.text
        ), f"{annotation.text=}"  # Check for degree information on new line


def test_plot_xrd_pattern_empty_input() -> None:
    empty_pattern = DiffractionPattern([], [], [], [])
    with pytest.raises(
        ValueError, match="No intensities found in the diffraction pattern"
    ):
        plot_xrd_pattern(empty_pattern)


def test_plot_xrd_pattern_single_peak() -> None:
    mock_diffraction_pattern.x = [30]
    mock_diffraction_pattern.y = [100]
    mock_diffraction_pattern.hkls = [[{"hkl": (1, 1, 1)}]]
    mock_diffraction_pattern.d_hkls = [2.0]

    fig = plot_xrd_pattern(mock_diffraction_pattern)
    assert len(fig.data[0].x) == 1
    assert len(fig.layout.annotations) == 1


def test_plot_xrd_pattern_intensity_normalization() -> None:
    original_max = max(mock_diffraction_pattern.y)
    fig = plot_xrd_pattern(mock_diffraction_pattern)
    normalized_max = max(fig.data[0].y)
    assert normalized_max == 100
    assert normalized_max / original_max == pytest.approx(100 / original_max)
