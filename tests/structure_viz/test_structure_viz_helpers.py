import re
from collections.abc import Callable
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Lattice, PeriodicSite, Structure

from pymatviz.enums import ElemColorScheme, SiteCoords
from pymatviz.structure_viz.helpers import (
    NO_SYM_MSG,
    UNIT_CELL_EDGES,
    _angles_to_rotation_matrix,
    draw_bonds,
    draw_site,
    draw_unit_cell,
    draw_vector,
    get_atomic_radii,
    get_elem_colors,
    get_first_matching_site_prop,
    get_image_sites,
    get_site_hover_text,
    get_subplot_title,
)
from pymatviz.typing import RgbColorType, Xyz


@pytest.fixture
def mock_figure() -> Any:
    class MockFigure:
        def __init__(self) -> None:
            self.last_trace_kwargs: dict[str, Any] = {}

        def add_scatter(self, *_args: Any, **kwargs: Any) -> None:
            self.last_trace_kwargs = kwargs

        def add_scatter3d(self, *_args: Any, **kwargs: Any) -> None:
            self.last_trace_kwargs = kwargs

    return MockFigure()


@pytest.mark.parametrize(
    ("angles", "expected_shape"),
    [("10x,8y,3z", (3, 3)), ("30x,45y,60z", (3, 3))],
)
def test_angles_to_rotation_matrix(
    angles: str, expected_shape: tuple[int, int]
) -> None:
    rot_matrix = _angles_to_rotation_matrix(angles)
    assert rot_matrix.shape == expected_shape
    assert_allclose(np.linalg.det(rot_matrix), 1.0)


def test_angles_to_rotation_matrix_invalid_input() -> None:
    with pytest.raises(ValueError, match="could not convert string to float: "):
        _angles_to_rotation_matrix("invalid_input")


@pytest.mark.parametrize(
    ("elem_colors", "expected_result"),
    [
        (ElemColorScheme.jmol, dict),
        (ElemColorScheme.vesta, dict),
        ({"Si": "#FF0000", "O": "#00FF00"}, dict),
    ],
)
def test_get_elem_colors(
    elem_colors: ElemColorScheme | dict[str, RgbColorType], expected_result: type
) -> None:
    colors = get_elem_colors(elem_colors)
    assert isinstance(colors, expected_result)
    if isinstance(elem_colors, dict):
        assert colors == elem_colors
    else:
        assert "Si" in colors
        assert "O" in colors


def test_get_elem_colors_invalid_input() -> None:
    err_msg = f"colors must be a dict or one of ('{', '.join(ElemColorScheme)}')"
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        get_elem_colors("invalid_input")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("atomic_radii", "expected_type"),
    [(None, dict), (1.5, dict), ({"Si": 0.3, "O": 0.2}, dict)],
)
def test_get_atomic_radii(
    atomic_radii: float | dict[str, float] | None, expected_type: type
) -> None:
    radii = get_atomic_radii(atomic_radii)
    assert isinstance(radii, expected_type)

    if atomic_radii is None or isinstance(atomic_radii, float):
        assert "Si" in radii
        assert "O" in radii
        if isinstance(atomic_radii, float):
            assert radii["Si"] == pytest.approx(1.11 * atomic_radii)
    elif isinstance(atomic_radii, dict):
        assert radii == atomic_radii


def test_get_image_sites(structures: list[Structure]) -> None:
    structure = structures[0]
    site = structure[0]
    lattice = structure.lattice

    # Test with default tolerance
    image_atoms = get_image_sites(site, lattice)
    assert isinstance(image_atoms, np.ndarray)
    assert image_atoms.ndim in (1, 2)  # Allow both 1D and 2D arrays
    if image_atoms.size > 0:
        assert image_atoms.shape[1] == 3  # Each image atom should have 3 coordinates

    # Test with custom tolerance
    image_atoms = get_image_sites(site, lattice, tol=0.1)
    assert isinstance(image_atoms, np.ndarray)
    assert image_atoms.ndim in (1, 2)
    if image_atoms.size > 0:
        assert image_atoms.shape[1] == 3


@pytest.mark.parametrize(
    ("lattice", "site_coords", "expected_images"),
    [
        (Lattice.cubic(1), [0, 0, 0], 7),
        (Lattice.cubic(1), [0, 0.5, 0], 3),
        (Lattice.hexagonal(3, 5), [0, 0, 0], 7),
        (Lattice.rhombohedral(3, 5), [0, 0, 0], 7),
        (Lattice.rhombohedral(3, 5), [0.1, 0, 0], 3),
        (Lattice.rhombohedral(3, 5), [0.5, 0, 0], 3),
    ],
)
def test_get_image_sites_lattices(
    lattice: Lattice, site_coords: list[float], expected_images: int
) -> None:
    site = PeriodicSite("Si", site_coords, lattice)
    image_atoms = get_image_sites(site, lattice)
    assert len(image_atoms) == expected_images


@pytest.mark.parametrize("is_3d", [True, False])
@pytest.mark.parametrize("is_image", [True, False])
@pytest.mark.parametrize(
    "hover_text",
    [*SiteCoords, lambda site: f"Custom: {site.species_string}"],
)
def test_draw_site(
    structures: list[Structure],
    mock_figure: Any,
    is_3d: bool,
    is_image: bool,
    hover_text: SiteCoords | Callable[[PeriodicSite], str],
) -> None:
    """Test site drawing with various settings."""
    structure = structures[0]
    site = structure[0]
    coords = site.coords
    elem_colors = get_elem_colors(ElemColorScheme.jmol)
    atomic_radii = get_atomic_radii(None)

    draw_site(  # Test with default site labels
        mock_figure,
        site,
        coords,
        0,
        "symbol",
        elem_colors,
        atomic_radii,
        atom_size=40,
        scale=1,
        site_kwargs={},
        is_3d=is_3d,
        is_image=is_image,
        hover_text=hover_text,
    )

    custom_labels = {site.species_string: "Custom"}
    draw_site(  # Test with custom site labels
        mock_figure,
        site,
        coords,
        0,
        custom_labels,
        elem_colors,
        atomic_radii,
        atom_size=40,
        scale=1,
        site_kwargs={},
        is_3d=is_3d,
        is_image=is_image,
        hover_text=hover_text,
    )

    # Verify hover text format
    expected_text = ""
    if callable(hover_text):
        expected_text = "Custom:"
    elif hover_text == SiteCoords.cartesian:
        expected_text = "Coordinates (0, 0, 0)"
    elif hover_text == SiteCoords.fractional:
        expected_text = "Coordinates [0, 0, 0]"
    elif hover_text == SiteCoords.cartesian_fractional:
        expected_text = "Coordinates (0, 0, 0) [0, 0, 0]"

    assert expected_text in mock_figure.last_trace_kwargs["hovertext"]


@pytest.mark.parametrize(
    ("struct_key", "subplot_title", "expected_text"),
    [
        ("key", None, "key"),
        (1, None, "1. Si2 (spg="),  # Partial match due to dynamic spg number
        ("key", lambda _struct, key: f"Custom title for {key}", "Custom title for key"),
        (
            "key",
            lambda _struct, key: {"text": f"Custom for {key}", "font": {"size": 14}},
            "Custom for key",
        ),
    ],
)
def test_get_subplot_title(
    structures: list[Structure],
    struct_key: Any,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]] | None,
    expected_text: str,
) -> None:
    structure = structures[0]
    title = get_subplot_title(structure, struct_key, 1, subplot_title)
    assert isinstance(title, dict)
    assert "text" in title
    assert expected_text in title["text"]

    if callable(subplot_title) and isinstance(
        subplot_title(structure, struct_key), dict
    ):
        assert "font" in title
        assert title["font"]["size"] == 14


def test_constants() -> None:
    assert isinstance(NO_SYM_MSG, str)
    assert NO_SYM_MSG.startswith("Symmetry could not be determined")
    assert isinstance(UNIT_CELL_EDGES, tuple)
    assert len(UNIT_CELL_EDGES) == 12
    assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in UNIT_CELL_EDGES)


@pytest.mark.parametrize(
    ("start", "vector", "is_3d", "arrow_kwargs", "expected_traces"),
    [
        (
            [0, 0, 0],
            [1, 1, 1],
            True,
            {"color": "red", "width": 2, "arrow_head_length": 0.5},
            2,  # One for the line, one for the cone
        ),
        # One scatter trace for 2D
        ([0, 0], [1, 1], False, {"color": "blue", "width": 3}, 1),
        # One for the line, one for the cone
        ([1, 1, 1], [2, 2, 2], True, {"color": "green", "scale": 0.5}, 2),
        # One scatter trace for 2D
        ([1, 1], [2, 2], False, {"color": "yellow", "scale": 2}, 1),
    ],
)
def test_draw_vector(
    start: list[float],
    vector: list[float],
    is_3d: bool,
    arrow_kwargs: dict[str, Any],
    expected_traces: int,
) -> None:
    """Test vector drawing with various settings."""
    fig = go.Figure()
    start, vector = np.array(start), np.array(vector)

    # Draw vector and check trace count
    initial_trace_count = len(fig.data)
    draw_vector(fig, start, vector, is_3d=is_3d, arrow_kwargs=arrow_kwargs)
    assert len(fig.data) - initial_trace_count == expected_traces

    # Check trace properties based on dimensionality
    if is_3d:
        # 3D arrow properties (line + cone)
        line_trace = fig.data[-2]
        cone_trace = fig.data[-1]
        assert line_trace.mode == "lines"
        assert line_trace.line.color == arrow_kwargs["color"]
        assert line_trace.line.width == arrow_kwargs.get("width", 5)
        assert cone_trace.type == "cone"
        assert cone_trace.colorscale[0][1] == arrow_kwargs["color"]
        assert cone_trace.sizeref == arrow_kwargs.get("arrow_head_length", 0.8)
    else:
        # 2D arrow properties
        scatter_trace = fig.data[-1]
        assert scatter_trace.mode == "lines+markers"
        assert scatter_trace.marker.color == arrow_kwargs["color"]
        assert scatter_trace.line.width == arrow_kwargs.get("width", 5)

    # Check scaling
    scale = arrow_kwargs.get("scale", 1.0)
    end_point = start + vector * scale
    if is_3d:
        assert_allclose(fig.data[-2].x[1], end_point[0])
        assert_allclose(fig.data[-2].y[1], end_point[1])
        assert_allclose(fig.data[-2].z[1], end_point[2])
    else:
        assert_allclose(fig.data[-1].x[1], end_point[0])
        assert_allclose(fig.data[-1].y[1], end_point[1])


def test_draw_vector_default_values() -> None:
    """Test vector drawing with default parameters."""
    fig = go.Figure()
    start, vector = np.zeros(3), np.ones(3)
    draw_vector(fig, start=start, vector=vector, is_3d=True)
    assert len(fig.data) == 2

    line_trace, cone_trace = fig.data
    assert line_trace.line.color == "white"
    assert line_trace.line.width == 5
    assert cone_trace.sizeref == 0.8
    assert_allclose(cone_trace.x, [1])
    assert_allclose(cone_trace.y, [1])
    assert_allclose(cone_trace.z, [1])


@pytest.fixture
def test_structures() -> list[Structure]:
    """Create test structures with various site properties."""
    lattice = Lattice.cubic(5.0)
    struct1 = Structure(lattice, ["Si", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    struct1.add_site_property("force", [[1, 1, 1], [-1, -1, -1]])
    struct1.add_site_property("charge", [1, -1])

    struct2 = Structure(lattice, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    struct2.add_site_property("magmom", [5, -5])
    struct2.properties["energy"] = -10.0

    return [struct1, struct2]


@pytest.mark.parametrize(
    ("prop_keys", "expected_result", "filter_callback", "warn_if_none"),
    [
        (["force", "magmom"], "force", None, False),
        (["magmom", "force"], "magmom", None, False),
        (["energy"], "energy", None, False),
        (["non_existent"], None, None, False),
        ([], None, None, False),
        (
            ["charge", "magmom"],
            "charge",
            lambda _p, v: isinstance(v, int) and v > 0,
            False,
        ),
        (["non_existent"], None, None, True),
    ],
)
def test_get_first_matching_site_prop(
    test_structures: list[Structure],
    prop_keys: list[str],
    expected_result: str | None,
    filter_callback: Callable[[str, Any], bool] | None,
    warn_if_none: bool,
) -> None:
    """Test finding the first matching site property with various parameters."""
    if warn_if_none and expected_result is None:
        with pytest.warns(UserWarning, match="None of prop_keys="):
            result = get_first_matching_site_prop(
                test_structures,
                prop_keys,
                warn_if_none=warn_if_none,
                filter_callback=filter_callback,
            )
    else:
        result = get_first_matching_site_prop(
            test_structures,
            prop_keys,
            warn_if_none=warn_if_none,
            filter_callback=filter_callback,
        )

    assert result == expected_result


def test_get_first_matching_site_prop_edge_cases() -> None:
    """Test edge cases for get_first_matching_site_prop."""
    # Empty structure list
    assert get_first_matching_site_prop([], ["force"]) is None

    # Structure with multiple properties and complex filtering
    lattice = Lattice.cubic(5.0)
    empty_struct = Structure(lattice, ["Si"], [[0, 0, 0]])
    assert get_first_matching_site_prop([empty_struct], ["force"]) is None

    multi_prop_struct = Structure(lattice, ["Si", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    multi_prop_struct.add_site_property("force", [[1, 1, 1], [-1, -1, -1]])
    multi_prop_struct.add_site_property(
        "velocity", [[0.1, 0.1, 0.1], [-0.1, -0.1, -0.1]]
    )
    assert (
        get_first_matching_site_prop([multi_prop_struct], ["force", "velocity"])
        == "force"
    )

    # Test with complex filter
    def complex_filter(prop: str, value: Any) -> bool:
        if prop == "force":
            return all(abs(v) > 0.5 for v in value)
        if prop == "velocity":
            return all(abs(v) < 0.5 for v in value)
        return False

    assert (
        get_first_matching_site_prop(
            [multi_prop_struct], ["velocity", "force"], filter_callback=complex_filter
        )
        == "velocity"
    )


@pytest.mark.parametrize(
    ("hover_text", "expected_output"),
    [
        (SiteCoords.cartesian, "<b>Site: Si1</b><br>Coordinates (0, 0, 0)"),
        (SiteCoords.fractional, "<b>Site: Si1</b><br>Coordinates [0, 0, 0]"),
        (
            SiteCoords.cartesian_fractional,
            "<b>Site: Si1</b><br>Coordinates (0, 0, 0) [0, 0, 0]",
        ),
        (lambda site: f"Custom: {site.species_string}", "Custom: Si"),
    ],
)
def test_get_site_hover_text(
    hover_text: SiteCoords | Callable[[PeriodicSite], str], expected_output: str
) -> None:
    """Test hover text generation for sites with various formats."""
    lattice = Lattice.cubic(1.0)
    site = PeriodicSite("Si", [0, 0, 0], lattice)
    result = get_site_hover_text(site, hover_text, site.species)
    assert result == expected_output


def test_get_site_hover_text_invalid_template() -> None:
    """Test error handling for invalid hover text templates."""
    lattice = Lattice.cubic(1.0)
    site = PeriodicSite("Si", [0, 0, 0], lattice)
    with pytest.raises(ValueError, match="Invalid hover_text="):
        get_site_hover_text(site, "invalid_template", site.species)  # type: ignore[arg-type]


@pytest.mark.parametrize("is_3d", [True, False])
@pytest.mark.parametrize(
    "unit_cell_kwargs",
    [
        {},
        {"edge": {"color": "red", "width": 2, "dash": "solid"}},
        {"node": {"size": 5, "color": "blue"}},
        {
            "edge": {"color": "green", "width": 3},
            "node": {"size": 4, "color": "yellow"},
        },
    ],
)
def test_draw_unit_cell(
    structures: list[Structure], is_3d: bool, unit_cell_kwargs: dict[str, Any]
) -> None:
    """Test unit cell drawing with various parameters."""
    structure = structures[0]
    fig = go.Figure()

    draw_unit_cell(fig, structure, unit_cell_kwargs, is_3d=is_3d)

    # Check trace count and types
    n_edge_traces, n_node_traces = 12, 8
    assert len(fig.data) == n_edge_traces + n_node_traces
    expected_trace_type = go.Scatter3d if is_3d else go.Scatter
    assert all(isinstance(trace, expected_trace_type) for trace in fig.data)

    # Check trace modes
    edge_traces = fig.data[:n_edge_traces]
    node_traces = fig.data[n_edge_traces:]
    assert all(trace.mode == "lines" for trace in edge_traces)
    assert all(trace.mode == "markers" for trace in node_traces)

    # Check if custom properties were applied
    if "edge" in unit_cell_kwargs:
        for trace in edge_traces:
            for key, value in unit_cell_kwargs["edge"].items():
                assert getattr(trace.line, key) == value

    if "node" in unit_cell_kwargs:
        for trace in node_traces:
            for key, value in unit_cell_kwargs["node"].items():
                assert getattr(trace.marker, key) == value


def test_draw_unit_cell_hover_text(structures: list[Structure]) -> None:
    """Test hover text for unit cell elements."""
    structure = structures[0]
    fig = go.Figure()
    draw_unit_cell(fig, structure, {}, is_3d=True)

    # Check hover text content
    edge_trace = fig.data[0]  # First edge
    assert "Length:" in edge_trace.hovertext[1]
    assert "Start:" in edge_trace.hovertext[1]
    assert "End:" in edge_trace.hovertext[1]

    corner_trace = fig.data[12]  # First corner
    assert "α =" in corner_trace.hovertext  # noqa: RUF001
    assert "β =" in corner_trace.hovertext
    assert "γ =" in corner_trace.hovertext  # noqa: RUF001


@pytest.fixture
def bond_test_structure() -> Structure:
    """Simple cubic structure for bond tests."""
    lattice = Lattice.cubic(3.0)
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])


@pytest.fixture
def bond_test_structure_extended() -> Structure:
    """Structure with atoms closer to the cell boundary."""
    lattice = Lattice.cubic(3.0)
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.9, 0.9, 0.9]])


@pytest.mark.parametrize(
    ("is_3d", "bond_kwargs", "plotted_sites_coords_param"),
    [
        (True, None, None),
        (True, {"color": "rgb(0, 0, 255)", "width": 0.2}, {(1, 1, 1)}),
        (False, {"color": "rgb(255, 0, 0)", "width": 2}, {(3.0, 3.0, 3.0)}),
        (
            True,
            {"color": "rgb(0, 0, 255)", "width": 3, "dash": "dot"},
            {(3.0, 3.0, 3.0), (-3.0, -3.0, -3.0)},
        ),
        (False, {"color": "rgb(0, 128, 0)", "width": 1}, None),
    ],
)
def test_draw_bonds(
    bond_test_structure: Structure,
    is_3d: bool,
    bond_kwargs: dict[str, Any] | None,
    plotted_sites_coords_param: set[Xyz] | None,
) -> None:
    """Test basic bond drawing functionality with various parameters."""
    fig = go.Figure()
    nn_strategy = CrystalNN()

    draw_bonds(
        fig=fig,
        structure=bond_test_structure,
        nn=nn_strategy,
        is_3d=is_3d,
        bond_kwargs=bond_kwargs,
        plotted_sites_coords=plotted_sites_coords_param,
    )

    # Verify bonds were created with correct properties
    assert len(fig.data) > 0
    expected_trace_type = go.Scatter3d if is_3d else go.Scatter
    assert all(isinstance(trace, expected_trace_type) for trace in fig.data)

    if bond_kwargs:  # Check custom bond styling
        for trace in fig.data:
            for key, value in bond_kwargs.items():
                assert getattr(trace.line, key) == value

    # Check bonds to image atoms
    if plotted_sites_coords_param:
        if is_3d:
            max_coords = max(
                max(list(trace.x) + list(trace.y) + list(trace.z)) for trace in fig.data
            )
        else:
            max_coords = max(max(list(trace.x) + list(trace.y)) for trace in fig.data)
        assert max_coords >= 1.5
        assert len(fig.data) > 1  # More than just central bond

    for trace in fig.data:  # Verify trace properties
        assert trace.mode == "lines"
        assert trace.showlegend is False
        assert trace.hoverinfo == "skip"


@pytest.mark.parametrize(
    ("test_case", "is_3d", "rotation_angle", "image_coords", "check_outside_cell"),
    [
        ("rotation_only", False, np.pi / 4, None, False),
        ("image_filtering", True, None, (3.0, 3.0, 3.0), True),
        ("rotation_with_image", False, np.pi / 6, (3.0, 3.0, 3.0), True),
        ("no_image_atoms", True, None, None, False),
    ],
)
def test_draw_bonds_advanced(
    bond_test_structure: Structure,
    bond_test_structure_extended: Structure,
    test_case: str,
    is_3d: bool,
    rotation_angle: float | None,
    image_coords: Xyz | None,
    check_outside_cell: bool,
) -> None:
    """Test advanced bond drawing scenarios with rotation and image atoms."""
    fig = go.Figure()
    nn_strategy = CrystalNN()

    # Use extended structure for cases that need to check bonds to image atoms
    structure = (
        bond_test_structure_extended if check_outside_cell else bond_test_structure
    )

    # Setup rotation matrix if needed
    rotation_matrix = None
    if rotation_angle is not None:
        rotation_matrix = [
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle), np.cos(rotation_angle), 0],
            [0, 0, 1],
        ]

    # Setup visible image atoms
    plotted_sites_coords = None
    if image_coords is not None:
        if rotation_matrix is not None and rotation_angle is not None:
            # Apply rotation to image coordinates
            rotated_coords = np.dot(np.array(image_coords), rotation_matrix)
            plotted_sites_coords = {tuple(rotated_coords)}
        else:
            plotted_sites_coords = {image_coords}
    elif test_case == "no_image_atoms":
        plotted_sites_coords = set()  # Empty set

    draw_bonds(
        fig=fig,
        structure=structure,
        nn=nn_strategy,
        is_3d=is_3d,
        rotation_matrix=rotation_matrix,
        plotted_sites_coords=plotted_sites_coords,
    )

    # Common assertions
    assert len(fig.data) > 0
    expected_trace_type = go.Scatter3d if is_3d else go.Scatter
    assert all(isinstance(trace, expected_trace_type) for trace in fig.data)

    # Test-specific assertions
    if test_case == "rotation_only":
        # Check rotation was applied
        original_x, original_y = structure[0].coords[0], structure[0].coords[1]
        for trace in fig.data:
            assert not all(x == original_x for x in trace.x)
            assert not all(y == original_y for y in trace.y)

    elif test_case == "image_filtering" and image_coords is not None:
        # Check bonds only go to specified image atoms
        for trace in fig.data:
            max_x, max_y, max_z = max(trace.x), max(trace.y), max(trace.z)
            if max_x > 3.0 or max_y > 3.0 or max_z > 3.0:
                assert any(
                    abs(x - image_coords[0]) < 0.01
                    and abs(y - image_coords[1]) < 0.01
                    and abs(z - image_coords[2]) < 0.01
                    for x, y, z in zip(trace.x, trace.y, trace.z, strict=False)
                )

    elif test_case == "rotation_with_image":
        # Check bonds are rotated and only go to visible image atoms
        rotated_coords = tuple(np.dot(np.array(image_coords), rotation_matrix))
        for trace in fig.data:
            if max(trace.x) > 3.0 or max(trace.y) > 3.0:
                assert any(
                    abs(x - rotated_coords[0]) < 0.01
                    and abs(y - rotated_coords[1]) < 0.01
                    for x, y in zip(trace.x, trace.y, strict=False)
                )

    elif test_case == "no_image_atoms":
        # Check no bonds go outside unit cell
        for trace in fig.data:
            assert max(trace.x) <= 3.1
            assert max(trace.y) <= 3.1
            assert max(trace.z) <= 3.1
