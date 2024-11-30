from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import pytest

from pymatviz.brillouin import brillouin_zone_3d


if TYPE_CHECKING:
    from pymatgen.core import Structure


@pytest.mark.parametrize(
    ("kwargs", "expected_mode"),
    [
        ({}, None),
        ({"point_kwargs": False}, None),
        ({"axes_vectors": False}, None),
        ({"point_kwargs": {"size": 10, "color": "red"}}, "markers+text"),
    ],
)
def test_brillouin_zone_3d_basic(
    structures: list[Structure], kwargs: dict[str, Any], expected_mode: str | None
) -> None:
    """Test basic functionality and trace types of brillouin_zone_3d."""
    fig = brillouin_zone_3d(structures[0], **kwargs)
    assert isinstance(fig, go.Figure)
    assert fig.layout.scene.aspectmode == "cube"

    # Check that the figure contains the expected traces
    trace_types = {trace["type"] for trace in fig.data}
    assert trace_types <= {"scatter3d", "cone", "mesh3d"}, f"{trace_types=}"

    if expected_mode:
        text_traces = [
            trace
            for trace in fig.data
            if trace.type == "scatter3d" and trace.mode == expected_mode
        ]
        assert len(text_traces) == 1


@pytest.mark.parametrize(
    ("struct_idx", "expected_labels"),
    [
        (
            0,
            (
                "Γ",
                "Y<sub>2</sub>",
                "Y<sub>4</sub>",
                "A",
                "M<sub>2</sub>",
                "V",
                "V<sub>2</sub>",
                "L<sub>2</sub>",
                "C",
                "C<sub>2</sub>",
                "C<sub>4</sub>",
                "D",
                "D<sub>2</sub>",
                "E",
                "E<sub>2</sub>",
                "E<sub>4</sub>",
            ),
        ),
        (1, ("Γ", "Z", "M", "A", "R", "X")),  # Different structure, different points
    ],
)
def test_brillouin_zone_3d_high_symm_points(
    structures: list[Structure], struct_idx: int, expected_labels: tuple[str, ...]
) -> None:
    """Test high symmetry points visualization for different structures."""
    fig = brillouin_zone_3d(
        structures[struct_idx],
        point_kwargs={"size": 10, "color": "red"},
        label_kwargs={"size": 16},
    )

    text_traces = [
        trace
        for trace in fig.data
        if trace.type == "scatter3d" and trace.mode == "markers+text"
    ]
    assert len(text_traces) == 1
    assert text_traces[0].text == expected_labels


def test_brillouin_zone_3d_different_lattices(structures: list[Structure]) -> None:
    """Test visualization for different lattice systems."""
    figs = [brillouin_zone_3d(struct) for struct in structures]
    assert all(isinstance(fig, go.Figure) for fig in figs)

    # Compare volumes to verify different shapes
    def get_volume(fig: go.Figure) -> float:
        for trace in fig.data:
            if trace.type == "mesh3d":
                x, y, z = trace.x, trace.y, trace.z
                return abs(
                    np.mean([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])
                    ** 3
                )
        return 0.0

    volumes = [get_volume(fig) for fig in figs]
    # Volumes should differ by more than 1%
    for i, j in [(0, 1), (1, 0)]:
        assert abs(volumes[i] - volumes[j]) / max(volumes) > 0.01


def test_brillouin_zone_3d_reciprocal_vectors(structures: list[Structure]) -> None:
    """Test that reciprocal lattice vectors are correctly calculated."""
    fig = brillouin_zone_3d(structures[0])

    # Get reciprocal lattice vectors
    recip_latt = structures[0].lattice.reciprocal_lattice
    expected_vectors = recip_latt.matrix

    vector_traces = [
        trace
        for trace in fig.data
        if trace.type == "scatter3d"
        and trace.mode == "lines"
        and hasattr(trace, "showlegend")
        and not trace.showlegend
    ]

    assert len(vector_traces) > 0
    for trace in vector_traces:
        vector = np.array([trace.x[-1], trace.y[-1], trace.z[-1]])
        if np.allclose(vector, 0):
            continue

        vector_norm = np.linalg.norm(vector)
        assert vector_norm > 0, f"Zero-length vector found: {vector}"
        scaled_vector = vector / vector_norm

        # Compare with normalized expected vectors
        expected_directions = [
            vec / np.linalg.norm(vec)
            for vec in expected_vectors
            if not np.allclose(vec, 0)
        ]

        angles = [
            np.arccos(np.clip(np.dot(scaled_vector, expected_dir), -1.0, 1.0))
            for expected_dir in expected_directions
        ]
        min_angle = min(angles) if angles else np.pi
        assert min_angle == pytest.approx(0, abs=2)  # 2 degree tolerance
