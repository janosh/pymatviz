import re
from collections.abc import Callable, Hashable
from typing import Any

import numpy as np
import plotly.graph_objects as go
import pytest
from numpy.testing import assert_allclose
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import (
    Composition,
    Element,
    Lattice,
    PeriodicSite,
    Species,
    Structure,
)
from pymatgen.electronic_structure.core import Magmom

from pymatviz.enums import ElemColorScheme, SiteCoords
from pymatviz.structure import helpers
from pymatviz.structure.helpers import (
    NO_SYM_MSG,
    _angles_to_rotation_matrix,
    _coerce_vector,
    _get_site_vector,
    _is_3d_vector,
    _standardize_struct,
    draw_bonds,
    draw_cell,
    draw_disordered_site,
    draw_site,
    draw_vector,
    generate_site_label,
    get_atomic_radii,
    get_disordered_site_legend_name,
    get_elem_colors,
    get_elem_radius,
    get_first_matching_site_prop,
    get_image_sites,
    get_site_elements,
    get_site_hover_text,
    get_site_species,
    get_struct_prop,
    get_subplot_title,
    missing_covalent_radius,
    normalize_elem_color,
    resolve_per_struct_params,
)
from pymatviz.typing import AnyStructure, RgbColorType, Xyz


@pytest.fixture
def mock_figure() -> Any:
    """Figure stub recording the last trace kwargs."""

    class MockFigure:
        def __init__(self) -> None:
            self.last_trace_kwargs: dict[str, Any] = {}

        def add_scatter(self, *_args: Any, **kwargs: Any) -> None:
            self.last_trace_kwargs = kwargs

        def add_scatter3d(self, *_args: Any, **kwargs: Any) -> None:
            self.last_trace_kwargs = kwargs

    return MockFigure()


@pytest.mark.parametrize(
    ("species_input", "expected_type", "expected_elements"),
    [
        ("Si", Element, {"Si"}),
        (Element("O"), Element, {"O"}),
        (Species("Fe", 2), Species, {"Fe"}),
        (Composition({"Fe": 0.75, "Co": 0.25}), Composition, {"Fe", "Co"}),
        (
            Composition({"Fe": 0.5, "Co": 0.3, "Ni": 0.2}),
            Composition,
            {"Fe", "Co", "Ni"},
        ),
    ],
)
def test_get_site_species_and_elements(
    species_input: Any, expected_type: type, expected_elements: set[str]
) -> None:
    """Test get_site_species and get_site_elements for all site types."""
    lattice = Lattice.cubic(3.0)
    site = PeriodicSite(species_input, [0, 0, 0], lattice)

    species_result = get_site_species(site)
    assert isinstance(species_result, expected_type)

    elements_result = get_site_elements(site)
    assert elements_result == expected_elements


@pytest.mark.parametrize("angles", ["10x,8y,3z", "30x,45y,60z"])
def test_angles_to_rotation_matrix(angles: str) -> None:
    """Euler angles give a proper 3x3 rotation matrix."""
    rot_matrix = _angles_to_rotation_matrix(angles)
    assert rot_matrix.shape == (3, 3)
    assert_allclose(np.linalg.det(rot_matrix), 1.0)


@pytest.mark.parametrize(
    ("angles", "err_msg"),
    [
        ("invalid_input", "Invalid rotation angle='invalid_input'"),
        ("10x,abcy", "Invalid rotation angle='abcy'"),
        ("10x,20w", "Invalid rotation axis 'w' in angle='20w'"),
        ("10", "Invalid rotation axis '0' in angle='10'"),
    ],
)
def test_angles_to_rotation_matrix_invalid_input(angles: str, err_msg: str) -> None:
    """Malformed angles or unknown axes raise ValueError naming the bad term."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        _angles_to_rotation_matrix(angles)


@pytest.mark.parametrize(
    "elem_colors",
    [ElemColorScheme.jmol, ElemColorScheme.vesta, {"Si": "#FF0000", "O": "#00FF00"}],
)
def test_get_elem_colors(
    elem_colors: ElemColorScheme | dict[str, RgbColorType],
) -> None:
    """Color schemes resolve to element maps and custom dicts pass through."""
    colors = get_elem_colors(elem_colors)
    assert isinstance(colors, dict)
    if isinstance(elem_colors, dict):
        assert colors == elem_colors
    else:
        assert "Si" in colors
        assert "O" in colors


def test_get_elem_colors_invalid_input() -> None:
    """Unknown color schemes raise ValueError."""
    err_msg = f"colors must be a dict or one of ('{', '.join(ElemColorScheme)}')"
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        get_elem_colors("invalid_input")  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("atomic_radii", [None, 1.5, {"Si": 0.3, "O": 0.2}])
def test_get_atomic_radii(atomic_radii: float | dict[str, float] | None) -> None:
    """Radii default to covalent radii, scale with floats, pass through dicts."""
    radii = get_atomic_radii(atomic_radii)
    assert isinstance(radii, dict)

    if atomic_radii is None or isinstance(atomic_radii, float):
        assert "Si" in radii
        assert "O" in radii
        if isinstance(atomic_radii, float):
            assert radii["Si"] == pytest.approx(1.11 * atomic_radii)
    elif isinstance(atomic_radii, dict):
        assert radii == atomic_radii
        assert get_elem_radius(radii, "Si") == 0.3
        with pytest.raises(KeyError, match="No radius for elem_symbol='Fe'"):
            get_elem_radius(radii, "Fe")
        # dummy species (no physical radius) get the fallback, also in default radii
        assert get_elem_radius(radii, "X") == missing_covalent_radius
        assert get_elem_radius(get_atomic_radii(None), "X") == missing_covalent_radius


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
    """Image site counts for corner, edge and face sites in various lattices."""
    site = PeriodicSite("Si", site_coords, lattice)
    image_atoms = get_image_sites(site, lattice)
    assert len(image_atoms) == expected_images


@pytest.mark.parametrize(
    ("cell_boundary_tol", "site_coords", "expected_min_images", "expected_max_images"),
    [
        # Corner site - maximum image sites
        (0.0, [0, 0, 0], 7, 7),  # Strict boundaries
        (0.1, [0, 0, 0], 7, 15),  # Small buffer
        (0.2, [0, 0, 0], 7, 20),  # Medium buffer
        # Edge site - fewer images
        (0.0, [0.5, 0, 0], 3, 3),  # Strict boundaries
        (0.1, [0.5, 0, 0], 3, 8),  # Small buffer
        # Center site - minimal images
        (0.0, [0.5, 0.5, 0.5], 0, 0),  # Strict boundaries
        (0.2, [0.5, 0.5, 0.5], 0, 6),  # Medium buffer
    ],
)
def test_get_image_sites_cell_boundary_tol(
    cell_boundary_tol: float,
    site_coords: list[float],
    expected_min_images: int,
    expected_max_images: int,
) -> None:
    """Test cell_boundary_tol parameter controls image site inclusion correctly."""
    lattice = Lattice.cubic(3.0)
    site = PeriodicSite("Si", site_coords, lattice)

    image_atoms = get_image_sites(site, lattice, cell_boundary_tol=cell_boundary_tol)

    # Check expected number of image sites
    n_images = len(image_atoms)
    assert expected_min_images <= n_images <= expected_max_images, (
        f"Expected {expected_min_images}-{expected_max_images} images for "
        f"tol={cell_boundary_tol}, coords={site_coords}, got {n_images}"
    )

    # Verify all image sites are within tolerance bounds
    if n_images > 0:
        image_frac_coords = [
            lattice.get_fractional_coords(img_cart) for img_cart in image_atoms
        ]

        for img_frac in image_frac_coords:
            for coord in img_frac:
                assert -cell_boundary_tol <= coord <= 1 + cell_boundary_tol, (
                    f"Image site coordinate {coord} outside tolerance bounds"
                )


@pytest.mark.parametrize(("tol1", "tol2"), [(0.0, 0.1), (0.1, 0.2)])
def test_get_image_sites_tolerance_ordering(tol1: float, tol2: float) -> None:
    """Test that higher tolerances include at least as many sites as lower ones."""
    lattice = Lattice.cubic(2.5)
    site = PeriodicSite("Si", [0, 0, 0], lattice)  # Corner site for maximum effect

    images_low_tol = get_image_sites(site, lattice, cell_boundary_tol=tol1)
    images_high_tol = get_image_sites(site, lattice, cell_boundary_tol=tol2)

    assert len(images_high_tol) >= len(images_low_tol), (
        f"Higher tolerance {tol2} should include at least as many sites as {tol1}"
    )


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
    subplot_title: Callable[[AnyStructure, Hashable], str | dict[str, Any]] | None,
    expected_text: str,
) -> None:
    """Default and custom (str or dict) subplot titles."""
    structure = structures[0]
    title = get_subplot_title(structure, struct_key, 1, subplot_title)
    assert expected_text in title["text"]

    if callable(subplot_title) and isinstance(
        subplot_title(structure, struct_key), dict
    ):
        assert "font" in title
        assert title["font"]["size"] == 14


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
    start_arr, vector_arr = np.array(start), np.array(vector)

    # Draw vector and check trace count
    initial_trace_count = len(fig.data)
    draw_vector(fig, start_arr, vector_arr, is_3d=is_3d, arrow_kwargs=arrow_kwargs)
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
    end_point = start_arr + vector_arr * scale
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
        """Accept large forces and small velocities."""
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
        (
            SiteCoords.cartesian,
            "<b>Site: Si1</b><br>Coordinates (0, 0, 0)",
        ),
        (
            SiteCoords.fractional,
            "<b>Site: Si1</b><br>Coordinates [0, 0, 0]",
        ),
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


def test_get_site_hover_text_float_formatting() -> None:
    """Test float formatting in hover text generation."""
    # non-finite coords must not crash (int(nan) used to raise ValueError)
    nan_site = PeriodicSite("Si", (float("nan"), 0, 0), Lattice.cubic(3.0))
    assert "nan" in get_site_hover_text(nan_site, SiteCoords.cartesian, Element("Si"))

    lattice = Lattice.cubic(1.0)
    # Create a site with coordinates that would show scientific notation with .3g
    site = PeriodicSite("Si", [1.23456789, 1e-17, 2.3456], lattice)

    # Test default format (.4)
    result_default = get_site_hover_text(site, SiteCoords.cartesian, site.species)
    assert "1.235" in result_default
    assert ("1e-17" in result_default) or (
        "0" in result_default
    )  # Very small number in scientific notation or zero
    assert "2.346" in result_default

    # Test custom string format (.2f)
    result_2f = get_site_hover_text(site, SiteCoords.cartesian, site.species, ".2f")
    assert "1.23" in result_2f
    assert "0.00" in result_2f
    assert "2.35" in result_2f

    # Test with fractional coordinates
    result_frac = get_site_hover_text(site, SiteCoords.fractional, site.species, ".6f")
    assert "1.234568" in result_frac
    assert "0.000000" in result_frac
    assert "2.345600" in result_frac

    # Test with custom callable formatter
    def custom_formatter(val: float) -> str:
        """Custom formatter that shows rounded values with prefix."""
        return f"~{val:.1f}"

    result_custom = get_site_hover_text(
        site,
        SiteCoords.cartesian,
        site.species,
        custom_formatter,
    )
    assert "~1.2" in result_custom
    assert "~0.0" in result_custom  # 1e-17 rounds to 0
    assert "~2.3" in result_custom

    # Test with both cartesian and fractional
    result_both = get_site_hover_text(
        site,
        SiteCoords.cartesian_fractional,
        site.species,
        ".3f",
    )
    assert "1.235" in result_both  # cartesian
    assert "0.000" in result_both  # 1e-17 rounded
    assert "2.346" in result_both


def test_get_site_hover_text_invalid_template() -> None:
    """Test error handling for invalid hover text templates."""
    lattice = Lattice.cubic(1.0)
    site = PeriodicSite("Si", [0, 0, 0], lattice)
    with pytest.raises(ValueError, match="Invalid hover_text="):
        get_site_hover_text(site, "invalid_template", site.species)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("is_3d", [True, False])
@pytest.mark.parametrize(
    "cell_kwargs",
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
def test_draw_cell(
    structures: list[Structure], is_3d: bool, cell_kwargs: dict[str, Any]
) -> None:
    """Test cell drawing with various parameters."""
    structure = structures[0]
    fig = go.Figure()

    draw_cell(fig, structure, cell_kwargs, is_3d=is_3d)

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
    if "edge" in cell_kwargs:
        for trace in edge_traces:
            for key, value in cell_kwargs["edge"].items():
                assert getattr(trace.line, key) == value

    if "node" in cell_kwargs:
        for trace in node_traces:
            for key, value in cell_kwargs["node"].items():
                assert getattr(trace.marker, key) == value


@pytest.mark.parametrize(
    ("cell_kwargs", "show_faces", "err_msg"),
    [
        # top-level line kwargs and per-structure dicts used to be silently ignored
        (
            {"color": "red", "width": 2},
            False,
            "Unknown cell style keys ['color', 'width']",
        ),
        ({"struct1": {"edge": {}}}, False, "Unknown cell style keys ['struct1']"),
        ({}, {"opacity": 0.5}, "Unknown cell face style keys ['opacity']"),
    ],
)
def test_draw_cell_invalid_style_keys(
    structures: list[Structure],
    cell_kwargs: dict[str, Any],
    show_faces: bool | dict[str, Any],
    err_msg: str,
) -> None:
    """Unknown show_cell/show_cell_faces style keys raise instead of being ignored."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        draw_cell(go.Figure(), structures[0], cell_kwargs, show_faces=show_faces)


def test_draw_cell_hover_text(structures: list[Structure]) -> None:
    """Test hover text for cell elements."""
    structure = structures[0]
    fig = go.Figure()
    draw_cell(fig, structure, {}, is_3d=True)

    # Check hover text content
    edge_trace = fig.data[0]  # First edge
    assert "Length:" in edge_trace.hovertext[1]
    assert "Start:" in edge_trace.hovertext[1]
    assert "End:" in edge_trace.hovertext[1]

    corner_trace = fig.data[12]  # First corner
    assert "α =" in corner_trace.hovertext  # noqa: RUF001
    assert "β =" in corner_trace.hovertext
    assert "γ =" in corner_trace.hovertext  # noqa: RUF001


@pytest.mark.parametrize("is_3d", [True, False])
@pytest.mark.parametrize(
    ("show_faces", "expected_surface_traces"),
    [
        (False, 0),  # No surfaces
        (True, 12),  # Default surfaces for 3D (6 faces x 2 triangles), 1 for 2D
        ({"color": "rgba(255,0,0,0.2)"}, 12),  # Custom surface styling for 3D, 1 for 2D
        ({"color": "rgba(0,255,0,0.15)", "showscale": False}, 12),
    ],
)
def test_draw_cell_faces(
    structures: list[Structure],
    is_3d: bool,
    show_faces: bool | dict[str, Any],
    expected_surface_traces: int,
) -> None:
    """Test cell face drawing with various parameters."""
    structure = structures[0]
    fig = go.Figure()

    if not is_3d and show_faces:  # Adjust expected traces for 2D
        expected_surface_traces = 1  # 2D only has 1 filled polygon

    draw_cell(fig, structure, {}, is_3d=is_3d, show_faces=show_faces)

    if is_3d:  # Count surface traces
        surface_traces = [
            trace
            for trace in fig.data
            if trace.type == "mesh3d"
            and trace.name
            and trace.name.startswith("surface-face")
        ]
    else:
        surface_traces = [
            trace
            for trace in fig.data
            if trace.name == "cell-face"
            and hasattr(trace, "fill")
            and trace.fill == "toself"
        ]

    assert len(surface_traces) == expected_surface_traces

    # Test custom styling if provided
    if isinstance(show_faces, dict) and expected_surface_traces > 0:
        if expected_color := show_faces.get("color"):
            for trace in surface_traces:
                if is_3d:
                    assert trace.color == expected_color
                else:
                    assert trace.fillcolor == expected_color

        if is_3d and (showscale := show_faces.get("showscale")):
            for trace in surface_traces:
                assert trace.showscale == showscale


@pytest.fixture
def bond_test_structure() -> Structure:
    """BCC Si (a=3 A) where CrystalNN gives each site 8 neighbors."""
    lattice = Lattice.cubic(3.0)
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])


BCC_PRIMARY_COORDS: set[Xyz] = {(0.0, 0.0, 0.0), (1.5, 1.5, 1.5)}


# 8 bonds from the corner site + 8 from the body center, minus the shared in-cell
# bond (get_nn_info reports every bond from both ends, draw_bonds dedupes) = 15.
# plotted_sites_coords keeps only bonds with both ends plotted. draw_bonds only
# starts bonds at sites of the passed structure (structure_2d/3d pass one augmented
# with image sites), so filter sets lacking those sites legitimately yield 0 bonds.
@pytest.mark.parametrize(
    ("is_3d", "rotation", "bond_kwargs", "plotted_sites_coords", "n_bonds"),
    [
        (True, None, None, None, 15),
        (False, None, {"color": "rgb(0, 128, 0)", "width": 1}, None, 15),
        (False, "30z", None, None, 15),
        # only the in-cell bond between the two primary sites
        (
            True,
            None,
            {"color": "rgb(0, 0, 255)", "width": 3, "dash": "dot"},
            BCC_PRIMARY_COORDS,
            1,
        ),
        # plus the bond from the body center to the (3, 3, 3) corner image
        (
            False,
            "30z,10x",
            {"color": "rgb(255, 0, 0)", "width": 2},
            BCC_PRIMARY_COORDS | {(3.0, 3.0, 3.0)},
            2,
        ),
        (True, None, None, BCC_PRIMARY_COORDS | {(3.0, 3.0, 3.0)}, 2),
        # (0,0,0) and (3,3,3) aren't bonded and the body center isn't plotted
        (True, None, {"color": "rgb(9, 9, 9)"}, {(0.0, 0.0, 0.0), (3.0, 3.0, 3.0)}, 0),
        (False, None, None, set(), 0),  # hidden sites -> no bonds
    ],
)
def test_draw_bonds(
    bond_test_structure: Structure,
    is_3d: bool,
    rotation: str | None,
    bond_kwargs: dict[str, Any] | None,
    plotted_sites_coords: set[Xyz] | None,
    n_bonds: int,
) -> None:
    """Bonds are deduped, filtered to plotted sites, rotated and styled."""
    fig = go.Figure()
    rotation_matrix = None if rotation is None else _angles_to_rotation_matrix(rotation)
    draw_bonds(
        fig=fig,
        structure=bond_test_structure,
        nn=CrystalNN(),
        is_3d=is_3d,
        bond_kwargs=bond_kwargs,
        rotation_matrix=rotation_matrix,
        plotted_sites_coords=plotted_sites_coords,
    )

    is_gradient = (bond_kwargs or {}).get("color", True) is True
    segments_per_bond = 10 if is_gradient else 1
    assert len(fig.data) == n_bonds * segments_per_bond
    expected_trace_type = go.Scatter3d if is_3d else go.Scatter
    assert all(isinstance(trace, expected_trace_type) for trace in fig.data)
    for trace in fig.data:
        assert trace.mode == "lines"
        assert trace.showlegend is False
        assert trace.hoverinfo == "skip"
        for key, value in (bond_kwargs or {}).items():
            assert getattr(trace.line, key) == value

    n_dims = 3 if is_3d else 2

    def trace_point(trace: Any, point_idx: int) -> tuple[float, ...]:
        """Rounded (x, y[, z]) of a trace's start (0) or end (-1) point."""
        axes = (trace.x, trace.y, trace.z) if is_3d else (trace.x, trace.y)
        return tuple(round(float(axis[point_idx]), 5) for axis in axes)

    bond_ends = [
        frozenset((trace_point(segments[0], 0), trace_point(segments[-1], -1)))
        for segments in (
            fig.data[idx : idx + segments_per_bond]
            for idx in range(0, len(fig.data), segments_per_bond)
        )
    ]
    if is_3d:  # no bond drawn twice (2D projections of distinct bonds can overlap)
        assert len(set(bond_ends)) == n_bonds

    if plotted_sites_coords is not None:  # both ends at (rotated) plotted sites
        plotted_points = np.array(sorted(plotted_sites_coords)).reshape(-1, 3)
        if rotation_matrix is not None:
            plotted_points = plotted_points @ rotation_matrix
        allowed_ends = {
            tuple(round(float(val), 5) for val in point[:n_dims])
            for point in plotted_points
        }
        assert {end for ends in bond_ends for end in ends} <= allowed_ends


def test_get_struct_prop(fe3co4_disordered: Structure) -> None:
    """Test the property precedence helper function."""
    from pymatviz.structure import fe3co4_disordered as reference_structure

    struct = fe3co4_disordered
    reference_properties = reference_structure.properties.copy()

    # Test 1: Structure property takes precedence over function parameter
    struct.properties["test_prop"] = "struct_value"
    result = get_struct_prop(struct, "key1", "test_prop", "func_value")
    assert result == "struct_value"

    # Test 2: Function parameter used when structure property doesn't exist
    result = get_struct_prop(struct, "key1", "missing_prop", "func_value")
    assert result == "func_value"

    # Test 3: Dict function parameter with matching key
    result = get_struct_prop(
        struct, "key1", "missing_prop", {"key1": "dict_value", "key2": "other"}
    )
    assert result == "dict_value"

    # Test 4: Dict function parameter with missing key returns None
    result = get_struct_prop(
        struct, "missing_key", "missing_prop", {"key1": "dict_value"}
    )
    assert result is None

    # Test 5: Structure property takes precedence over dict parameter
    result = get_struct_prop(struct, "key1", "test_prop", {"key1": "dict_value"})
    assert result == "struct_value"

    # Test 6: falsy struct properties like 0.0 take precedence in
    # resolve_per_struct_params (regression: `or` fallbacks discarded them)
    struct.properties["cell_boundary_tol"] = 0.0
    _, cell_boundary_tol, *_ = resolve_per_struct_params(
        struct, "key", ElemColorScheme.jmol, 0.5, None, 30, 1.0
    )
    assert cell_boundary_tol == 0.0
    assert reference_structure.properties == reference_properties


@pytest.mark.parametrize("is_3d", [True, False])
def test_draw_disordered_site_legend_functionality(is_3d: bool) -> None:
    """Disordered site draws one combined legend entry grouped per site."""
    structure = Structure(
        Lattice.cubic(5.0),
        [Composition({"Fe": 0.75, "Ni": 0.25}), "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    fig = go.Figure()
    draw_disordered_site(
        fig=fig,
        site=structure[0],
        coords=np.zeros(3),
        site_idx=0,
        site_labels="legend",
        elem_colors=get_elem_colors(ElemColorScheme.jmol),
        atomic_radii=get_atomic_radii(None),
        atom_size=30,
        scale=1.0,
        site_kwargs={},
        is_image=False,
        is_3d=is_3d,
        showlegend=True,
        legend="legend",
        legendgroup=None,
    )

    assert [trace.name for trace in fig.data if trace.showlegend] == ["Fe₀.₇₅Ni₀.₂₅"]
    assert {trace.legendgroup for trace in fig.data} == {"disordered_site_0"}

    # ordered sites used to silently draw nothing
    with pytest.raises(ValueError, match="needs a site with multiple species, got 'O'"):
        draw_disordered_site(
            fig=fig,
            site=structure[1],
            coords=np.zeros(3),
            site_idx=1,
            site_labels="legend",
            elem_colors=get_elem_colors(ElemColorScheme.jmol),
            atomic_radii=get_atomic_radii(None),
            atom_size=30,
            scale=1.0,
            site_kwargs={},
            is_3d=is_3d,
        )


@pytest.mark.parametrize(
    ("sorted_species", "is_image", "expected"),
    [
        ([(Species("Fe"), 0.6), (Species("Ni"), 0.4)], False, "Fe₀.₆Ni₀.₄"),
        ([(Species("Fe"), 0.75), (Species("C"), 0.25)], False, "Fe₀.₇₅C₀.₂₅"),
        ([(Species("Fe"), 0.5), (Species("Ni"), 0.5)], False, "Fe₀.₅Ni₀.₅"),
        (
            [(Species("Fe"), 0.5), (Species("Ni"), 0.3), (Species("Cr"), 0.2)],
            False,
            "Fe₀.₅Ni₀.₃Cr₀.₂",
        ),
        ([(Species("Fe"), 0.8), (Species("Co"), 0.2)], True, "Image of Fe₀.₈Co₀.₂"),
        ([(Species("Ni"), 0.666667), (Species("Fe"), 0.333333)], False, "Ni₀.₆₇Fe₀.₃₃"),
    ],
    ids=["binary", "quarters", "equal", "ternary", "image", "rounding"],
)
def test_get_disordered_site_legend_name(
    sorted_species: list[tuple[Species, float]], is_image: bool, expected: str
) -> None:
    """Legend names list species with subscript occupancies in the given order."""
    legend_name = get_disordered_site_legend_name(sorted_species, is_image=is_image)
    assert legend_name == expected


@pytest.mark.parametrize(
    ("raw_color", "expected"),
    [
        ((1.0, 0.5, 0.0), "rgb(255,127,0)"),  # RGB tuple (0-1 range)
        ((255, 127, 0), "rgb(255,127,0)"),  # RGB tuple (0-255 range)
        ("red", "red"),
        ("rgb(100,200,50)", "rgb(100,200,50)"),
        ([1.0, 0.5, 0.0], "rgb(255,127,0)"),  # RGB list
        ((1, 0.5, 0), "rgb(255,127,0)"),  # mixed int/float 0-1 channels
        ((1, 1, 1), "rgb(255,255,255)"),  # int 0-1 channels (was near-black)
        ((1, 2), ValueError),  # wrong tuple length used to fall back to gray
        (["red"], ValueError),
        (None, ValueError),
        (123, ValueError),
        ((True, False, True), ValueError),
    ],
)
def test_normalize_elem_color(raw_color: Any, expected: str | type[Exception]) -> None:
    """Element colors normalize to rgb strings, unsupported values raise."""
    if isinstance(expected, str):
        assert normalize_elem_color(raw_color) == expected
        return
    with pytest.raises(expected, match="Unsupported element color"):
        normalize_elem_color(raw_color)


@pytest.mark.parametrize(
    ("input_val", "expected"),
    [
        ([1, 2, 3], [1, 2, 3]),
        (np.array([0.1, 0.2, 0.3]), [0.1, 0.2, 0.3]),
        (Magmom([0, 0, 0.5]), [0, 0, 0.5]),
        (Magmom([0, 0, 0]), [0, 0, 0]),
        (Magmom([1.5, -0.5, 2.0]), [1.5, -0.5, 2.0]),
    ],
    ids=["list", "ndarray", "magmom_collinear", "magmom_zero", "magmom_non_collinear"],
)
def test_coerce_vector(input_val: Any, expected: list[float]) -> None:
    """Array-likes pass through and Magmom objects yield their global moment."""
    result = _coerce_vector(input_val)
    assert result.shape == (3,)
    assert_allclose(result, expected)


# === _is_3d_vector ===


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ([1, 2, 3], True),
        (np.zeros(3), True),
        (np.array([[1, 2, 3], [4, 5, 6]]), True),
        (5, False),
        ([1, 2], False),
        ([1, 2, 3, 4], False),
        (np.ones((3, 2)), False),
        (np.zeros((2, 3, 4)), False),
        ([], False),
        (Magmom([0, 0, 1]), True),
        (Magmom([0, 0, 0]), True),
    ],
    ids=[
        "vec_3",
        "zeros_3",
        "Nx3_matrix",
        "scalar",
        "vec_2",
        "vec_4",
        "3xN_matrix",
        "3d_tensor",
        "empty",
        "magmom",
        "zero_magmom",
    ],
)
def test_is_3d_vector(value: Any, expected: bool) -> None:
    """Test _is_3d_vector with various inputs."""
    assert _is_3d_vector(value) is expected


# === _get_site_vector ===


@pytest.mark.parametrize(
    ("use_site_prop", "expected"),
    [(True, [1, 0, 0]), (False, [0.5, 0.5, 0])],
    ids=["site_properties", "struct_properties"],
)
def test_get_site_vector_lookup(use_site_prop: bool, expected: list[float]) -> None:
    """Test _get_site_vector from site.properties and struct.properties fallback."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    forces = [expected, [0, 1, 0]]
    if use_site_prop:
        struct.add_site_property("force", forces)
    else:
        struct.properties["force"] = forces

    result = _get_site_vector(struct[0], struct, 0, "force")
    assert result is not None
    assert_allclose(result, expected)
    assert result.shape == (3,)


@pytest.mark.parametrize(
    "struct_force",
    [[[1, 0, 0]], [1, 0, 0], [[1, 0], [0, 1]], 5.0],
    ids=["too_few_vectors", "single_vector", "2d_vectors", "scalar"],
)
def test_get_site_vector_missing_or_malformed(struct_force: Any) -> None:
    """Missing props give None, struct props not shaped (n_sites, 3) raise."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert _get_site_vector(struct[0], struct, 0, "nonexistent") is None
    # used to silently return None (or scalar entries) for malformed struct props
    struct.properties["force"] = struct_force
    with pytest.raises(ValueError, match=r"must hold one 3D vector per site"):
        _get_site_vector(struct[1], struct, 1, "force")


def test_get_site_vector_with_magmom() -> None:
    """Test _get_site_vector coerces Magmom objects to (3,) arrays."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    struct.add_site_property("magmom", [Magmom([0, 0, 3]), Magmom([0, 0, -3])])

    result = _get_site_vector(struct[0], struct, 0, "magmom")
    assert result is not None
    assert result.shape == (3,)
    assert_allclose(result, [0, 0, 3])


def test_get_site_vector_site_takes_precedence() -> None:
    """Test site.properties takes precedence over struct.properties."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Fe"], [[0, 0, 0]])
    struct.add_site_property("force", [[1, 0, 0]])
    struct.properties["force"] = [[99, 99, 99]]  # should be ignored

    result = _get_site_vector(struct[0], struct, 0, "force")
    assert result is not None
    assert_allclose(result, [1, 0, 0])


@pytest.mark.parametrize(
    "use_site_prop", [True, False], ids=["site_level_magmom", "struct_level"]
)
def test_get_first_matching_site_prop_with_3d_filter(
    use_site_prop: bool,
) -> None:
    """Test get_first_matching_site_prop with _is_3d_vector filter."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    if use_site_prop:
        struct.add_site_property("magmom", [Magmom([0, 0, 1]), Magmom([0, 0, -1])])
        prop_key = "magmom"
    else:
        struct.properties["force"] = [[1, 0, 0], [0, 1, 0]]
        prop_key = "force"

    result = get_first_matching_site_prop(
        [struct],
        [prop_key],
        warn_if_none=False,
        filter_callback=lambda _prop, val: _is_3d_vector(val),
    )
    assert result == prop_key


@pytest.mark.parametrize(
    ("bond_color", "expected_colors"),
    [
        # False leaked into plotly color validation -> TypeError
        (False, {"rgb(169, 169, 169)"}),
        # CSS names crashed find_intermediate_color, 3-digit hex was misparsed
        ("red", {"rgb(255, 0, 0)"}),
        ("#abc", {"rgb(170, 187, 204)"}),
        ((255, 0, 0), {"rgb(255, 0, 0)"}),
        (("red", "blue"), None),
        ("not-a-color", ValueError),
    ],
)
def test_draw_bonds_colors(
    bond_test_structure: Structure,
    bond_color: Any,
    expected_colors: set[str] | type[ValueError] | None,
) -> None:
    """Bond colors accept False, CSS names, hex and RGB tuples, solid or gradient."""
    fig = go.Figure()
    if expected_colors is ValueError:
        with pytest.raises(ValueError, match="Unsupported color format"):
            draw_bonds(
                fig=fig,
                structure=bond_test_structure,
                nn=CrystalNN(),
                bond_kwargs={"color": bond_color},
            )
        return
    draw_bonds(
        fig=fig,
        structure=bond_test_structure,
        nn=CrystalNN(),
        is_3d=True,
        bond_kwargs={"color": bond_color},
    )
    colors = {trace.line.color for trace in fig.data}
    assert len(fig.data) > 0
    if expected_colors is None:  # 10-segment red -> blue gradient
        assert len(colors) == 10
        assert "rgb(242.25, 0.0, 12.75)" in colors  # first segment midpoint t=0.05
    else:
        assert colors == expected_colors


@pytest.mark.parametrize(
    ("site_labels", "site_idx", "expected"),
    [
        ("symbol", 0, "Fe"),
        ("species", 0, "Fe:0.75, C:0.25"),
        ("legend", 0, None),
        (False, 0, None),
        ({"Fe": "Iron"}, 0, "Iron"),
        ({"Ni": "Nickel"}, 1, "O"),  # unmapped elements keep their symbol
        ({}, 1, ""),
        (["A", "B"], 1, "B"),
        (("A", "B"), 0, "A"),
        # used to silently return the symbol or an empty label
        (["A"], 1, "site_labels sequence has 1 labels but needs one per site"),
        ("symbols", 0, "Invalid site_labels='symbols'"),
        (True, 0, "Invalid site_labels=True"),
        (None, 0, "Invalid site_labels=None"),
    ],
)
def test_generate_site_label(
    fe3co4_disordered: Structure, site_labels: Any, site_idx: int, expected: str | None
) -> None:
    """Site labels follow the labeling strategy and unknown strategies raise."""
    site = fe3co4_disordered[site_idx]
    if isinstance(expected, str) and expected.startswith(("Invalid", "site_labels")):
        with pytest.raises(ValueError, match=re.escape(expected)):
            generate_site_label(site_labels, site_idx, site)
        return
    assert generate_site_label(site_labels, site_idx, site) == expected


@pytest.mark.parametrize("standardize_struct", [True, None, False])
def test_standardize_struct(
    standardize_struct: bool | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Standardization keeps site properties and warns if symmetry is undetermined."""
    # negative frac coords trigger auto-standardization for standardize_struct=None
    struct = Structure(Lattice.cubic(4), ["Fe", "O"], [[-0.1, 0, 0], [0.5, 0.5, 0.5]])
    struct.add_site_property("force", [[1, 0, 0], [0, 1, 0]])

    std_struct = _standardize_struct(struct, standardize_struct=standardize_struct)
    if standardize_struct is False:
        assert std_struct is struct
        return
    assert std_struct is not struct
    assert all(site.frac_coords.min() >= 0 for site in std_struct)
    # site properties used to be dropped, silently removing force/magmom vectors
    assert sorted(map(tuple, std_struct.site_properties["force"])) == [
        (0, 1, 0),
        (1, 0, 0),
    ]

    class FailingSpacegroupAnalyzer:
        """Stand-in for SpacegroupAnalyzer when spglib can't find symmetry."""

        def __init__(self, _struct: Structure) -> None:
            raise ValueError("Unable to determine symmetry")

    monkeypatch.setattr(helpers, "SpacegroupAnalyzer", FailingSpacegroupAnalyzer)
    with pytest.warns(
        UserWarning, match=f"{NO_SYM_MSG} for Fe1 O1: Unable to determine symmetry"
    ):
        no_sym_struct = _standardize_struct(
            struct, standardize_struct=standardize_struct
        )
    assert no_sym_struct is struct
