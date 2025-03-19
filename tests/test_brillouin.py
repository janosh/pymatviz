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
    assert fig.layout.scene.aspectmode == "data"

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
                "Γ Y<sub>2</sub> Y<sub>4</sub> A M<sub>2</sub> V V<sub>2</sub> "  # noqa: SIM905
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
    "mp-1183085 mp-686119 mp-1183057 mp-862690 mp-1183089 mp-10018 mp-1207297".split(),  # noqa: SIM905
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
        "mp-1183085": (26, 22),  # Updated line counts to include reciprocal vectors
        "mp-686119": (28, 24),
        "mp-1183057": (24, 20),
        "mp-862690": (24, 20),
        "mp-1183089": (30, 26),
        "mp-10018": (27, 23),
        "mp-1207297": (22, 18),
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
    exp_cone = 3  # 3 cones for reciprocal vector arrow heads
    assert trace_counts["cone"] == exp_cone, f"{trace_counts=}, {exp_cone=}"
    assert scatter_modes["lines"] == exp_lines, f"{scatter_modes=}, {exp_lines=}"
    exp_markers_text = 1  # 1 trace for high symmetry points
    assert scatter_modes["markers+text"] == exp_markers_text, (
        f"{scatter_modes=}, {exp_markers_text=}"
    )
    exp_text = 3  # 3 labels for reciprocal vectors
    assert scatter_modes["text"] == exp_text, f"{scatter_modes=}, {exp_text=}"


def test_brillouin_zone_3d_subplot_grid(structures: list[Structure]) -> None:
    """Test subplot grid functionality for multiple structures."""
    # Test with list of structures
    fig = brillouin_zone_3d(structures, n_cols=2)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) == len(structures)  # subplot titles
    assert len(fig._grid_ref) == (len(structures) + 1) // 2  # number of rows
    assert len(fig._grid_ref[0]) == 2  # number of columns

    # Test with dict of structures
    struct_dict = {f"struct_{idx}": struct for idx, struct in enumerate(structures)}
    fig = brillouin_zone_3d(struct_dict, n_cols=1)
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) == len(struct_dict)
    assert len(fig._grid_ref) == len(struct_dict)  # number of rows
    assert len(fig._grid_ref[0]) == 1  # number of columns

    # Test custom subplot titles
    def subplot_title(struct: Structure, key: str | int) -> str:
        return f"Custom {key} - {struct.formula}"

    fig = brillouin_zone_3d(struct_dict, subplot_title=subplot_title)
    assert isinstance(fig, go.Figure)
    for idx, (key, struct) in enumerate(struct_dict.items()):
        assert fig.layout.annotations[idx].text == f"Custom {key} - {struct.formula}"

    # Test subplot spacing and layout
    fig = brillouin_zone_3d(structures, n_cols=2)
    assert fig.layout.height == 400 * ((len(structures) + 1) // 2)  # height per row
    assert fig.layout.width == 400 * 2  # width per column
    assert fig.layout.margin.l == 0
    assert fig.layout.margin.r == 0
    assert fig.layout.margin.t == 0
    assert fig.layout.margin.b == 0

    # Test that each subplot has correct scene properties
    for idx in range(1, len(structures) + 1):
        scene = getattr(fig.layout, f"scene{idx}")
        assert scene.xaxis.showticklabels is False
        assert scene.yaxis.showticklabels is False
        assert scene.zaxis.showticklabels is False
        assert scene.xaxis.visible is False
        assert scene.yaxis.visible is False
        assert scene.zaxis.visible is False
        assert scene.aspectmode == "data"
        assert scene.bgcolor == "rgba(90, 90, 90, 0.01)"

    # Test that each subplot has the expected number of traces
    n_traces_per_struct = dict(scatter3d=26, cone=3, mesh3d=1)
    trace_counts = {"scatter3d": 0, "cone": 0, "mesh3d": 0}
    for trace in fig.data:
        trace_counts[trace.type] += 1

    total_structs = len(structures)
    for trace_type, count in trace_counts.items():
        expected = n_traces_per_struct[trace_type] * total_structs
        assert count == expected, (
            f"Expected {expected} {trace_type} traces, got {count}"
        )


def test_brillouin_zone_3d_subplot_grid_options(structures: list[Structure]) -> None:
    """Test different subplot grid configurations and title options."""
    # Test disabling subplot titles
    fig = brillouin_zone_3d(structures, subplot_title=False)
    assert isinstance(fig, go.Figure)
    assert all(anno.text == " " for anno in fig.layout.annotations)

    # Test different grid configurations
    n_structs = len(structures)
    for n_cols in range(1, n_structs + 1):
        fig = brillouin_zone_3d(structures, n_cols=n_cols)
        assert isinstance(fig, go.Figure)
        expected_rows = (n_structs - 1) // n_cols + 1
        assert len(fig._grid_ref) == expected_rows
        assert len(fig._grid_ref[0]) == min(n_cols, n_structs)
        assert fig.layout.height == 400 * expected_rows
        assert fig.layout.width == 400 * min(n_cols, n_structs)

    # Test subplot domains and gaps
    fig = brillouin_zone_3d(structures, n_cols=2)
    gap = 0.01  # default gap in the function
    for idx in range(1, n_structs + 1):
        scene = getattr(fig.layout, f"scene{idx}")
        row = (idx - 1) // 2 + 1
        col = (idx - 1) % 2 + 1

        # Calculate expected domain coordinates
        x_start = (col - 1) / 2 + gap / 2
        x_end = col / 2 - gap / 2
        y_start = 1 - row / ((n_structs + 1) // 2) + gap / 2
        y_end = 1 - (row - 1) / ((n_structs + 1) // 2) - gap / 2

        assert scene.domain.x == pytest.approx([x_start, x_end], abs=1e-10)
        assert scene.domain.y == pytest.approx([y_start, y_end], abs=1e-10)

    # Test custom subplot title with dict return
    def subplot_title(_struct: Structure, key: str | int) -> dict[str, Any]:
        return dict(
            text=f"Custom {key}", font=dict(size=16, color="blue"), yanchor="bottom"
        )

    fig = brillouin_zone_3d(structures, subplot_title=subplot_title)
    assert isinstance(fig, go.Figure)
    for idx, (key, _struct) in enumerate(enumerate(structures)):
        anno = fig.layout.annotations[idx]
        assert anno.text == f"Custom {key}"
        assert anno.font.size == 16
        assert anno.font.color == "blue"
        assert anno.yanchor == "bottom"


def test_brillouin_zone_3d_axes_vectors(structures: list[Structure]) -> None:
    """Test customization of coordinate axes vectors."""
    # Test with custom axes vector styling
    custom_axes = {
        "shaft": {"color": "purple", "width": 8},
        "cone": {"sizeref": 0.4},  # Match the default value
    }
    fig = brillouin_zone_3d(structures[0], axes_vectors=custom_axes)  # type: ignore[arg-type]
    assert isinstance(fig, go.Figure)

    # Check shaft traces
    shaft_traces = [
        trace
        for trace in fig.data
        if trace.type == "scatter3d"
        and trace.mode == "lines"
        and trace.hoverinfo == "none"
    ]
    assert len(shaft_traces) == 3  # one for each axis
    for trace in shaft_traces:
        assert trace.line.color == "purple"
        assert trace.line.width == 8

    # Check cone traces
    cone_traces = [trace for trace in fig.data if trace.type == "cone"]
    assert len(cone_traces) == 3  # one for each axis
    for trace in cone_traces:
        assert trace.sizeref == 0.4

    # Test disabling axes vectors
    fig = brillouin_zone_3d(structures[0], axes_vectors=False)
    assert isinstance(fig, go.Figure)
    shaft_traces = [
        trace
        for trace in fig.data
        if trace.type == "scatter3d"
        and trace.mode == "lines"
        and trace.hoverinfo == "none"
    ]
    cone_traces = [trace for trace in fig.data if trace.type == "cone"]
    assert len(shaft_traces) == 0
    assert len(cone_traces) == 0


def test_brillouin_zone_3d_edge_cases(structures: list[Structure]) -> None:
    """Test edge cases and error handling."""
    with pytest.raises(TypeError, match="Subplot title must be a string or dict"):
        brillouin_zone_3d(structures, subplot_title=lambda *_args: 42)  # type: ignore[arg-type]

    # Test with invalid axes_vectors dict
    with pytest.raises(KeyError, match="axes_vectors must contain 'shaft' and 'cone'"):
        brillouin_zone_3d(structures[0], axes_vectors={"shaft": {}})


def test_brillouin_zone_3d_custom_subplot_titles(structures: list[Structure]) -> None:
    """Test different subplot title configurations."""

    # Test with dict return type for subplot titles
    def title_with_dict(_struct: Structure, key: str | int) -> dict[str, Any]:
        return {
            "text": f"Test {key}",
            "font": {"size": 20, "color": "red"},
            "x": 0.5,
            "y": 0.9,
        }

    fig = brillouin_zone_3d(structures, subplot_title=title_with_dict)
    assert isinstance(fig, go.Figure)
    for idx, anno in enumerate(fig.layout.annotations):
        assert anno.text == f"Test {idx}"
        assert anno.font.size == 20
        assert anno.font.color == "red"
        assert anno.x == 0.5
        assert anno.y == 0.9

    # Test with disabled subplot titles
    fig = brillouin_zone_3d(structures, subplot_title=False)
    assert isinstance(fig, go.Figure)
    assert all(anno.text == " " for anno in fig.layout.annotations)

    # Test with custom string return type
    def title_with_string(struct: Structure, key: str | int) -> str:
        return f"Structure {key} - {struct.formula}"

    fig = brillouin_zone_3d(structures, subplot_title=title_with_string)
    assert isinstance(fig, go.Figure)
    for idx, (key, struct) in enumerate(
        zip(range(len(structures)), structures, strict=False)
    ):
        assert fig.layout.annotations[idx].text == f"Structure {key} - {struct.formula}"
