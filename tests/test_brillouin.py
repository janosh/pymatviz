from __future__ import annotations

from glob import glob
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
from pymatgen.core import Structure
from scipy.spatial import ConvexHull

from pymatviz.brillouin import brillouin_zone_3d
from pymatviz.process_data import is_ase_atoms
from pymatviz.utils.testing import TEST_FILES
from tests.conftest import SI_ATOMS, SI_STRUCTS


if TYPE_CHECKING:
    from collections.abc import Hashable

    from pymatviz.typing import AnyStructure


@pytest.mark.parametrize(
    ("struct_to_plot", "kwargs", "expected_mode", "n_expected_subplots"),
    [
        (SI_STRUCTS[0], {}, None, 1),
        (SI_ATOMS[0], {"point_kwargs": False}, None, 1),
        ({"s0": SI_STRUCTS[0], "s1": SI_STRUCTS[1]}, {"axes_vectors": False}, None, 2),
        (
            {"a0": SI_ATOMS[0], "a1": SI_ATOMS[1]},
            {"point_kwargs": {"size": 10, "color": "red"}},
            "markers+text",
            2,
        ),
        ({"s0": SI_STRUCTS[0], "a1": SI_ATOMS[1]}, {}, None, 2),
        (
            SI_ATOMS[1],
            {"point_kwargs": {"size": 5, "color": "blue"}},
            "markers+text",
            1,
        ),
    ],
)
def test_brillouin_zone_3d_basic(
    struct_to_plot: AnyStructure | dict[str, AnyStructure],
    kwargs: dict[str, Any],
    expected_mode: str | None,
    n_expected_subplots: int,
) -> None:
    """Test basic functionality and trace types of brillouin_zone_3d for various
    inputs.
    """
    fig = brillouin_zone_3d(struct_to_plot, **kwargs)

    trace_types = {trace["type"] for trace in fig.data}
    assert trace_types <= {"scatter3d", "cone", "mesh3d"}, f"{trace_types=}"
    assert sum(1 for trace in fig.data if trace.type == "mesh3d") == n_expected_subplots

    if n_expected_subplots == 1:
        assert fig.layout.scene.aspectmode == "data"
        if expected_mode:
            text_traces = [
                trace
                for trace in fig.data
                if trace.type == "scatter3d" and trace.mode == expected_mode
            ]
            assert len(text_traces) == 1
    else:  # Dictionary input, check subplot properties
        assert len(fig.layout.annotations) == n_expected_subplots  # subplot titles
        for idx_subplot in range(1, n_expected_subplots + 1):
            scene = getattr(fig.layout, f"scene{idx_subplot}")
            assert scene.aspectmode == "data"
        if expected_mode:
            # For dict inputs, high-symm points are plotted for each subplot
            text_traces = [
                trace
                for trace in fig.data
                if trace.type == "scatter3d" and trace.mode == expected_mode
            ]
            # Each subplot should have its own set of high-symm points if enabled
            assert len(text_traces) == n_expected_subplots


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


@pytest.mark.parametrize("struct_idx", [0, 1])
def test_brillouin_zone_3d_volume(structures: list[Structure], struct_idx: int) -> None:
    """BZ mesh and reciprocal vectors span (2π)³ / primitive cell volume."""
    struct = structures[struct_idx]
    fig = brillouin_zone_3d(struct)
    expected_volume = (2 * np.pi) ** 3 / struct.get_primitive_structure().volume

    mesh = next(trace for trace in fig.data if trace.type == "mesh3d")
    hull = ConvexHull(np.column_stack([mesh.x, mesh.y, mesh.z]))
    # vector shafts are the only scatter3d traces without hover info
    shafts = [
        trace
        for trace in fig.data
        if trace.type == "scatter3d" and trace.hoverinfo == "none"
    ]
    recip_vectors = [[trace.x[1], trace.y[1], trace.z[1]] for trace in shafts]
    # observed rel. error <5e-16, rtol leaves headroom for primitive-cell finders
    assert hull.volume == pytest.approx(expected_volume, rel=1e-9)
    assert abs(np.linalg.det(recip_vectors)) == pytest.approx(expected_volume, rel=1e-9)


@pytest.mark.parametrize(
    "material_id",
    "mp-1183085 mp-686119 mp-1183057 mp-862690 mp-1183089 mp-10018 mp-1207297".split(),  # noqa: SIM905
)
def test_brillouin_zone_3d_trace_counts(material_id: str) -> None:
    """Test that brillouin_zone_3d produces the expected number of traces for each
    crystal system.
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
    """Test subplot layout, scenes and custom titles for multiple structures."""
    # Test custom subplot titles with dict of structures
    struct_dict = {f"struct_{idx}": struct for idx, struct in enumerate(structures)}

    def subplot_title(struct: AnyStructure, key: Hashable) -> str:
        assert not is_ase_atoms(struct)  # narrow out ase Atoms for ty
        formula = struct.formula
        return f"Custom {key} - {formula}"

    fig = brillouin_zone_3d(struct_dict, subplot_title=subplot_title)
    assert [anno.text for anno in fig.layout.annotations] == [
        f"Custom {key} - {struct.formula}" for key, struct in struct_dict.items()
    ]

    fig = brillouin_zone_3d(structures, n_cols=2)
    assert len(fig.layout.annotations) == len(structures)  # subplot titles
    assert fig.layout.margin.to_plotly_json() == dict(l=0, r=0, t=0, b=0)

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
        assert count == expected, f"{expected=} {trace_type} traces, got {count}"


def test_brillouin_zone_3d_subplot_grid_options(structures: list[Structure]) -> None:
    """Test different subplot grid configurations and title options."""
    # Test disabling subplot titles
    fig = brillouin_zone_3d(structures, subplot_title=False)
    assert all(anno.text == " " for anno in fig.layout.annotations)

    # Test different grid configurations, subplot domains and gaps
    n_structs = len(structures)
    gap = 0.01  # default gap in the function
    for n_cols in range(1, n_structs + 1):
        fig = brillouin_zone_3d(structures, n_cols=n_cols)
        n_rows = (n_structs - 1) // n_cols + 1
        assert fig._grid_ref is not None
        assert len(fig._grid_ref) == n_rows
        assert len(fig._grid_ref[0]) == min(n_cols, n_structs)
        assert fig.layout.height == 400 * n_rows
        assert fig.layout.width == 400 * min(n_cols, n_structs)

        for idx in range(1, n_structs + 1):
            scene = getattr(fig.layout, f"scene{idx}")
            row, col = (idx - 1) // n_cols + 1, (idx - 1) % n_cols + 1
            x_domain = [(col - 1) / n_cols + gap / 2, col / n_cols - gap / 2]
            y_domain = [1 - row / n_rows + gap / 2, 1 - (row - 1) / n_rows - gap / 2]
            assert scene.domain.x == pytest.approx(x_domain, abs=1e-10)
            assert scene.domain.y == pytest.approx(y_domain, abs=1e-10)

    # Test custom subplot title with dict return
    def subplot_title(_struct: AnyStructure, key: Hashable) -> dict[str, Any]:
        return dict(
            text=f"Custom {key}", font=dict(size=16, color="blue"), yanchor="bottom"
        )

    fig = brillouin_zone_3d(structures, subplot_title=subplot_title)
    for idx, struct in enumerate(structures):
        anno = fig.layout.annotations[idx]
        expected_text = f"Custom {idx + 1} {struct.formula}"
        assert anno.text == expected_text
        assert anno.font.size == 16
        assert anno.font.color == "blue"
        assert anno.yanchor == "bottom"


def test_brillouin_zone_3d_axes_vectors(structures: list[Structure]) -> None:
    """Test customization of coordinate axes vectors."""
    # Test with custom axes vector styling
    custom_axes: dict[Literal["shaft", "cone"], dict[str, Any]] = {
        "shaft": {"color": "purple", "width": 8},
        "cone": {"sizeref": 0.99},
    }
    fig = brillouin_zone_3d(structures[0], axes_vectors=custom_axes)
    # axes_vectors must not be mutated (pop() used to empty the cone dict, so
    # repeated calls with the same dict silently fell back to defaults)
    assert custom_axes["cone"] == {"sizeref": 0.99}
    fig = brillouin_zone_3d(structures[0], axes_vectors=custom_axes)

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
        assert trace.sizeref == 0.99

    # Test disabling axes vectors
    fig = brillouin_zone_3d(structures[0], axes_vectors=False)
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
        brillouin_zone_3d(structures, subplot_title=lambda *_args: 42)  # ty: ignore[invalid-argument-type]

    # Test with invalid axes_vectors dict
    with pytest.raises(KeyError, match="axes_vectors must contain 'shaft' and 'cone'"):
        brillouin_zone_3d(structures[0], axes_vectors={"shaft": {}})
