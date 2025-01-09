from __future__ import annotations

from glob import glob
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest
from pymatgen.core import Structure

from pymatviz.brillouin import brillouin_zone_3d
from pymatviz.utils.testing import TEST_FILES


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
            tuple(
                "Γ Y<sub>2</sub> Y<sub>4</sub> A M<sub>2</sub> V V<sub>2</sub> "
                "L<sub>2</sub> C C<sub>2</sub> C<sub>4</sub> D D<sub>2</sub> "
                "E E<sub>2</sub> E<sub>4</sub>".split()
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


@pytest.mark.parametrize(
    "material_id",
    "mp-1183085 mp-686119 mp-1183057 mp-862690 mp-1183089 mp-10018 mp-1207297".split(),
)
def test_brillouin_zone_3d_trace_counts(material_id: str) -> None:
    """Test that brillouin_zone_3d produces the expected number of traces for each
    crystal system.

    Args:
        material_id: Materials Project ID
        formula: Chemical formula
        system: Crystal system
        expected_counts: Tuple of expected trace counts:
            (mesh3d, scatter3d, cone, lines, markers+text, text)
    """
    struct_path = glob(f"{TEST_FILES}/structures/{material_id}-*.json.gz")[0]
    struct = Structure.from_file(struct_path)

    fig = brillouin_zone_3d(struct)

    # (scatter3d, lines)
    exp_scatter3d, exp_lines = {
        "mp-1183085": (32, 25),
        "mp-686119": (34, 27),
        "mp-1183057": (30, 23),
        "mp-862690": (30, 23),
        "mp-1183089": (36, 29),
        "mp-10018": (33, 26),
        "mp-1207297": (28, 21),
    }[material_id]
    # Count different trace types
    trace_counts = {"scatter3d": 0, "cone": 0, "mesh3d": 0}
    scatter_modes = {"lines": 0, "markers+text": 0, "text": 0}

    for trace in fig.data:
        trace_counts[trace["type"]] += 1

        if trace["type"] == "scatter3d":
            scatter_modes[trace["mode"]] += 1

    # Assert exact counts for each trace type
    exp_mesh3d = 1
    assert trace_counts["mesh3d"] == exp_mesh3d, f"{trace_counts=}, {exp_mesh3d=}"
    assert trace_counts["scatter3d"] == exp_scatter3d, (
        f"{trace_counts=}, {exp_scatter3d=}"
    )
    exp_cone = 6
    assert trace_counts["cone"] == exp_cone, f"{trace_counts=}, {exp_cone=}"
    assert scatter_modes["lines"] == exp_lines, f"{scatter_modes=}, {exp_lines=}"
    exp_markers_text = 1
    assert scatter_modes["markers+text"] == exp_markers_text, (
        f"{scatter_modes=}, {exp_markers_text=}"
    )
    exp_text = 6
    assert scatter_modes["text"] == exp_text, f"{scatter_modes=}, {exp_text=}"
