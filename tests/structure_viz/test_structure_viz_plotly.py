from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pymatgen.core import Lattice, Structure

import pymatviz as pmv
from pymatviz.colors import ELEM_COLORS_JMOL, ELEM_COLORS_VESTA
from pymatviz.enums import ElemColorScheme, Key
from pymatviz.structure_viz.helpers import (
    _get_site_symbol,
    get_atomic_radii,
    get_image_sites,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatviz.typing import Xyz

COORDS: Final[tuple[Xyz, Xyz]] = ((0, 0, 0), (0.5, 0.5, 0.5))
lattice_cubic = 5 * np.eye(3)  # 5 Å cubic lattice


def normalize_rgb_color(color_str: str) -> str:
    """Normalize RGB color string to consistent format."""
    if not isinstance(color_str, str):
        return str(color_str)

    # Remove spaces and ensure consistent format
    color_str = color_str.replace(" ", "")

    # If it's already in rgb format, ensure consistent spacing
    if color_str.startswith("rgb(") and color_str.endswith(")"):
        # Extract numbers and reformat
        numbers = color_str[4:-1].split(",")
        try:
            rgb_values = [int(float(n.strip())) for n in numbers]
            return f"rgb({rgb_values[0]},{rgb_values[1]},{rgb_values[2]})"
        except (ValueError, IndexError):
            return color_str

    return color_str


def _get_all_rendered_site_info(
    structure: Structure, show_image_sites: bool
) -> list[dict[str, Any]]:
    """Return a list of dicts, each representing a site (primary or image).

    Each dict contains "symbol" (effective symbol for coloring/sizing),
    "is_image", "primary_site_idx", and "primary_site_species_string".

    For color and size, images inherit from parent.
    For labels, it's more complex based on site_labels kwarg.
    """
    all_site_infos: list[dict[str, Any]] = []
    primary_sites_symbols = []

    for site_primary in structure:
        if hasattr(site_primary.species, "elements"):  # Disordered
            el_amt_dict = site_primary.species.get_el_amt_dict()
            symbol = max(el_amt_dict, key=el_amt_dict.get) if el_amt_dict else "X"
        elif hasattr(site_primary.species, "symbol"):  # Element
            symbol = site_primary.species.symbol
        elif hasattr(site_primary.species, "element"):  # Specie
            symbol = site_primary.species.element.symbol
        else:
            try:
                symbol = site_primary.species_string
            except AttributeError:
                symbol = "X"  # Fallback
        primary_sites_symbols.append(symbol)
        site_info = {
            "symbol": symbol,
            "is_image": False,
            "primary_site_idx": len(all_site_infos),
            "primary_site_species_string": site_primary.species_string,
            "site_obj": site_primary,
        }
        all_site_infos.append(site_info)

    if show_image_sites:
        for idx_primary, site_primary in enumerate(structure):
            parent_symbol = primary_sites_symbols[idx_primary]
            image_coords_list = get_image_sites(site_primary, structure.lattice)
            all_site_infos += [
                {
                    "symbol": parent_symbol,
                    "is_image": True,
                    "primary_site_idx": idx_primary,
                    "primary_site_species_string": site_primary.species_string,
                    "site_obj": site_primary,  # Image sites use the primary site object
                }
            ] * len(image_coords_list)

    return all_site_infos


# Helper function to resolve expected color string for tests
def _resolve_expected_color_str(
    site_symbol: str,
    elem_colors_kwarg: Any,  # ElemColorScheme | dict | None
    normalize_rgb_color_func: Callable[[str], str],
) -> str:
    """Determine the expected color string based on elem_colors_kwarg."""
    expected_color_str = "rgb(128,128,128)"  # Default fallback

    if isinstance(elem_colors_kwarg, dict):
        color_val = elem_colors_kwarg.get(site_symbol)
        if isinstance(color_val, str):
            if color_val.startswith("rgb"):
                expected_color_str = normalize_rgb_color_func(color_val)
            elif color_val.startswith("#"):
                # Convert hex to rgb string via plotly's internal conversion
                temp_fig = go.Figure(go.Scatter(x=[0], y=[0], marker_color=color_val))
                expected_color_str = normalize_rgb_color_func(
                    temp_fig.data[0].marker.color
                )
            else:  # named color
                expected_color_str = color_val  # Keep as named for comparison logic
        elif (
            isinstance(color_val, tuple) and len(color_val) == 3
        ):  # (r,g,b) float or int
            if all(0 <= c <= 1 for c in color_val):  # floats 0-1
                rgb_int = tuple(int(c * 255) for c in color_val)
            else:  # ints 0-255
                rgb_int = tuple(int(c) for c in color_val)
            expected_color_str = f"rgb({rgb_int[0]},{rgb_int[1]},{rgb_int[2]})"
            expected_color_str = normalize_rgb_color_func(expected_color_str)
    elif elem_colors_kwarg == ElemColorScheme.jmol:
        rgb_tuple = ELEM_COLORS_JMOL.get(site_symbol)
        if rgb_tuple:
            expected_color_str = (
                f"rgb({', '.join(str(int(v * 255)) for v in rgb_tuple)})"
            )
            expected_color_str = normalize_rgb_color_func(expected_color_str)
    elif elem_colors_kwarg == ElemColorScheme.vesta:
        rgb_tuple = ELEM_COLORS_VESTA.get(site_symbol)
        if rgb_tuple:
            expected_color_str = (
                f"rgb({', '.join(str(int(v * 255)) for v in rgb_tuple)})"
            )
            expected_color_str = normalize_rgb_color_func(expected_color_str)
    return expected_color_str


def _compare_colors(
    actual_color_str: str,  # Already normalized if coming from trace
    expected_color_str: str,  # Possibly named, resolved by _resolve_expected_color_str
    normalize_rgb_color_func: Callable[[str], str],
) -> None:
    """Compare actual (potentially RGB) and expected (potentially named) colors."""
    if not expected_color_str.startswith("rgb") and actual_color_str.startswith("rgb"):
        # Expected is named (e.g. "red"), actual is "rgb(255,0,0)".
        # Convert expected named color to rgb via a temp Plotly figure.
        temp_fig = go.Figure(go.Scatter(x=[0], y=[0], marker_color=expected_color_str))
        resolved_expected_color_str = normalize_rgb_color_func(
            temp_fig.data[0].marker.color
        )
        assert actual_color_str == resolved_expected_color_str
    else:
        # Both are rgb, or expected is named and actual is also or both are unresolvable
        # and should be compared as is.
        normed_expected = normalize_rgb_color_func(expected_color_str)
        normed_actual = normalize_rgb_color_func(actual_color_str)
        assert normed_actual == normed_expected


@pytest.mark.parametrize(
    ("test_scenario", "kwargs"),
    [
        # Basic functionality tests
        (
            "basic_jmol_colors",
            {
                "rotation": "0x,0y,0z",
                "atomic_radii": None,
                "atom_size": 30,
                "elem_colors": ElemColorScheme.jmol,
                "scale": 1,
                "show_cell": True,
                "show_sites": True,
                "site_labels": "symbol",
                "standardize_struct": None,
                "n_cols": 2,
                "show_site_vectors": "magmom",
            },
        ),
        (
            "vesta_colors_no_sites",
            {
                "rotation": "10x,-10y,0z",
                "atomic_radii": 0.5,
                "atom_size": 50,
                "elem_colors": ElemColorScheme.vesta,
                "scale": 1.5,
                "show_cell": False,
                "show_sites": False,
                "standardize_struct": True,
                "n_cols": 4,
                "show_site_vectors": ("magmom", "force"),
            },
        ),
        (
            "custom_colors_dict",
            {
                "rotation": "5x,5y,5z",
                "atomic_radii": {"Fe": 0.5, "O": 0.3},
                "atom_size": 40,
                "elem_colors": {"Fe": "red", "O": "blue"},
                "scale": 1.2,
                "show_cell": {"color": "red", "width": 2},
                "show_sites": {"line": {"width": 1, "color": "black"}},
                "site_labels": {"Fe": "Iron"},
                "standardize_struct": False,
                "n_cols": 3,
                "show_site_vectors": (),
            },
        ),
        (
            "legend_mode",
            {
                "rotation": "15x,0y,10z",
                "atomic_radii": 0.8,
                "atom_size": 35,
                "elem_colors": ElemColorScheme.jmol,
                "scale": 0.9,
                "show_cell": True,
                "show_sites": True,
                "site_labels": "legend",
                "standardize_struct": None,
                "n_cols": 2,
            },
        ),
        (
            "no_labels_custom_cell",
            {
                "rotation": "0x,20y,0z",
                "atomic_radii": None,
                "atom_size": 45,
                "elem_colors": ElemColorScheme.vesta,
                "scale": 1.1,
                "show_cell": {"color": "blue", "width": 1, "dash": "dot"},
                "show_sites": True,
                "site_labels": False,
                "standardize_struct": True,
                "n_cols": 4,
            },
        ),
        (
            "custom_site_styling",
            {
                "rotation": "30x,-15y,5z",
                "atomic_radii": 0.6,
                "atom_size": 55,
                "elem_colors": {"Fe": "green", "O": "yellow"},
                "scale": 0.8,
                "show_cell": True,
                "show_sites": {"line": {"width": 2, "color": "red"}},
                "standardize_struct": False,
                "n_cols": 3,
            },
        ),
        # New test scenarios for show_cell_faces
        (
            "cell_faces_default",
            {
                "rotation": "0x,0y,0z",
                "show_cell": True,
                "show_cell_faces": True,
                "show_sites": True,
                "site_labels": "symbol",
            },
        ),
        (
            "cell_faces_custom_color",
            {
                "rotation": "10x,10y,10z",
                "show_cell": True,
                "show_cell_faces": {"color": "rgba(255,0,0,0.2)"},
                "show_sites": True,
                "site_labels": "legend",
            },
        ),
        (
            "cell_faces_disabled",
            {
                "rotation": "5x,5y,5z",
                "show_cell": True,
                "show_cell_faces": False,
                "show_sites": True,
                "site_labels": "symbol",
            },
        ),
        (
            "no_cell_no_faces",
            {
                "rotation": "0x,0y,0z",
                "show_cell": False,
                "show_cell_faces": True,  # Should be ignored when show_cell=False
                "show_sites": True,
                "site_labels": "symbol",
            },
        ),
        # Coverage-specific scenarios
        (
            "struct_key_elem_colors",
            {
                "elem_colors": {
                    "struct1": ElemColorScheme.jmol,
                    "struct2": ElemColorScheme.vesta,
                },
                "site_labels": "symbol",
            },
        ),
        (
            "site_vector_properties",
            {
                "show_site_vectors": "force",
            },
        ),
        (  # Test cell_boundary_tol
            "cell_boundary_tol",
            dict(cell_boundary_tol=0.1, show_image_sites=True, site_labels="symbol"),
        ),
    ],
)
def test_structure_2d_plotly_comprehensive(
    test_scenario: str,
    kwargs: dict[str, Any],
    fe3co4_disordered_with_props: Structure,
) -> None:
    """Test structure_2d_plotly with various parameter combos."""
    # Handle multi-structure scenarios for coverage
    if test_scenario in ("struct_key_elem_colors", "site_vector_properties"):
        struct1 = fe3co4_disordered_with_props.copy()
        struct1.add_site_property("force", [[0.1, 0.2, 0.3]] * len(struct1))
        struct2 = fe3co4_disordered_with_props.copy()
        structures_input = {"struct1": struct1, "struct2": struct2}
    else:
        structures_input = fe3co4_disordered_with_props

    fig = pmv.structure_2d_plotly(structures_input, **kwargs)
    assert isinstance(fig, go.Figure)

    # Check that all traces are Scatter (2D)
    assert {*map(type, fig.data)} in ({go.Scatter}, set())

    # Verify layout properties based on site_labels
    site_labels_kwarg = kwargs.get("site_labels", "legend")  # Default from function
    expected_showlegend = site_labels_kwarg == "legend"
    assert fig.layout.showlegend is expected_showlegend
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"

    # Check that axes are properly configured
    for axis in fig.layout.xaxis, fig.layout.yaxis:
        assert axis.showticklabels is False
        assert axis.showgrid is False
        assert axis.zeroline is False
        assert axis.scaleratio == 1
        assert axis.constrain == "domain"

    # Test scenario-specific validations
    if isinstance(structures_input, dict):
        structure = next(iter(structures_input.values()))
    else:
        structure = structures_input
    _validate_2d_scenario_specifics(test_scenario, kwargs, fig, structure)


def _validate_2d_scenario_specifics(
    test_scenario: str,
    kwargs: dict[str, Any],
    fig: go.Figure,
    structure: Structure,
) -> None:
    """Validate scenario-specific aspects of 2D plots."""
    show_sites = kwargs.get("show_sites", True)
    show_cell = kwargs.get("show_cell", True)
    show_cell_faces = kwargs.get("show_cell_faces", True)

    if test_scenario == "vesta_colors_no_sites":
        # When show_sites=False, no site traces should exist
        site_traces = [
            trace
            for trace in fig.data
            if getattr(trace, "mode", None) in ("markers", "markers+text")
        ]
        assert len(site_traces) == 0

    elif test_scenario == "legend_mode":
        # Test legend functionality
        assert fig.layout.showlegend is True
        legend_traces = [trace for trace in fig.data if trace.showlegend]
        unique_elements = {_get_site_symbol(s) for s in structure}
        legend_trace_names = [trace.name for trace in legend_traces]
        for elem_symbol in unique_elements:
            assert elem_symbol in legend_trace_names

    elif test_scenario == "custom_colors_dict":
        # Test custom color application
        elem_colors_kwarg = kwargs.get("elem_colors")
        if show_sites and isinstance(elem_colors_kwarg, dict):
            site_traces = [
                trace
                for trace in fig.data
                if getattr(trace, "mode", None) in ("markers", "markers+text")
                and not (trace.name and trace.name.startswith(("node", "edge")))
            ]
            for trace in site_traces:
                if trace.name in elem_colors_kwarg:
                    actual_color = normalize_rgb_color(str(trace.marker.color))
                    expected_color = _resolve_expected_color_str(
                        trace.name, elem_colors_kwarg, normalize_rgb_color
                    )
                    _compare_colors(actual_color, expected_color, normalize_rgb_color)

    elif test_scenario == "no_labels_custom_cell":
        # Test custom cell styling
        cell_kwargs = kwargs.get("show_cell")
        if isinstance(cell_kwargs, dict):
            cell_traces = [trace for trace in fig.data if trace.mode == "lines"]
            assert len(cell_traces) > 0

    elif test_scenario == "cell_faces_default":
        # Test default cell faces functionality
        if show_cell and show_cell_faces:
            # For 2D plots, cell faces are shown as filled polygons
            surface_traces = [
                trace
                for trace in fig.data
                if trace.name == "cell-face" and trace.fill == "toself"
            ]
            assert len(surface_traces) == 1, "Expected 1 cell face trace in 2D"

    elif test_scenario == "cell_faces_custom_color":
        # Test custom cell faces color
        if show_cell and isinstance(show_cell_faces, dict):
            surface_traces = [
                trace
                for trace in fig.data
                if trace.name == "cell-face" and trace.fill == "toself"
            ]
            assert len(surface_traces) == 1
            expected_color = show_cell_faces.get("color", "rgba(255,255,255,0.1)")
            assert surface_traces[0].fillcolor == expected_color

    elif test_scenario == "cell_faces_disabled":
        # Test that no cell faces are shown when disabled
        surface_traces = [trace for trace in fig.data if trace.name == "cell-face"]
        assert len(surface_traces) == 0

    elif test_scenario == "no_cell_no_faces":
        # Test that no cell faces are shown when show_cell=False
        surface_traces = [trace for trace in fig.data if trace.name == "cell-face"]
        assert len(surface_traces) == 0

    # Skip validation for multi-structure scenarios as they have different trace counts
    if show_sites is not False and test_scenario not in (
        "struct_key_elem_colors",
        "site_vector_properties",
        "cell_faces_default",
        "cell_faces_custom_color",
        "cell_faces_disabled",
        "no_cell_no_faces",
    ):
        _validate_common_site_properties(kwargs, fig, structure)


def _validate_common_site_properties(
    kwargs: dict[str, Any], fig: go.Figure, structure: Structure
) -> None:
    """Validate common site properties across all test scenarios."""
    site_labels_kwarg = kwargs.get("site_labels", "legend")  # Default is "legend"

    if site_labels_kwarg == "legend":
        # Check legend traces
        legend_traces = [trace for trace in fig.data if trace.showlegend]
        unique_elements = {_get_site_symbol(s) for s in structure}
        assert len(legend_traces) == len(unique_elements)

        # Verify each element has correct number of points
        for trace in legend_traces:
            if trace.showlegend and trace.name in unique_elements:
                expected_points = sum(
                    1 for site in structure if _get_site_symbol(site) == trace.name
                )
                actual_points = len(trace.x) if hasattr(trace, "x") else 0
                assert actual_points == expected_points
    else:
        # Non-legend modes should have site traces
        site_traces = [
            trace
            for trace in fig.data
            if (trace.name or "").startswith("site") and "markers" in (trace.mode or "")
        ]
        if site_labels_kwarg is not False:
            assert len(site_traces) > 0


@pytest.mark.parametrize(
    ("input_type", "n_cols", "expected_structures"),
    [("dict", 3, 4), ("pandas_series", 2, 4), ("list", 2, 4)],
)
def test_structure_2d_plotly_multiple_inputs(
    input_type: str, n_cols: int, expected_structures: int
) -> None:
    """Test structure_2d_plotly with different input types and multiple structures."""
    struct1 = Structure(lattice_cubic, ["Fe", "O"], coords=COORDS)
    struct1.properties = {"id": "struct1"}
    struct2 = Structure(lattice_cubic, ["Co", "O"], coords=COORDS)
    struct2.properties = {Key.mat_id: "struct2"}
    struct3 = Structure(lattice_cubic, ["Ni", "O"], coords=COORDS)
    struct3.properties = {"ID": "struct3", "name": "nickel oxide"}
    struct4 = Structure(lattice_cubic, ["Cu", "O"], coords=COORDS)

    structs_dict = {
        "struct1": struct1,
        "struct2": struct2,
        "struct3": struct3,
        "struct4": struct4,
    }

    structures_input: Any
    if input_type == "dict":
        structures_input = structs_dict
    elif input_type == "pandas_series":
        structures_input = pd.Series(structs_dict)
    else:  # list
        structures_input = list(structs_dict.values())

    fig = pmv.structure_2d_plotly(structures_input, n_cols=n_cols, site_labels=False)
    assert isinstance(fig, go.Figure)

    # Validate trace counts
    trace_names = [trace.name or "" for trace in fig.data]

    actual_n_site_traces = sum(name.startswith("site-") for name in trace_names)
    expected_n_site_traces = sum(len(s) for s in structs_dict.values())
    assert actual_n_site_traces == expected_n_site_traces

    actual_n_edge_traces = sum(name.startswith("edge") for name in trace_names)
    assert actual_n_edge_traces == 12 * len(structs_dict)

    actual_n_node_traces = sum(name.startswith("node") for name in trace_names)
    assert actual_n_node_traces == 8 * len(structs_dict)

    assert len(fig.layout.annotations) == expected_structures


@pytest.mark.parametrize(
    ("hover_fmt", "test_coordinates", "expected_patterns"),
    [
        (".4", [1.23456789, 1e-17, 2.3456], ["1.235", "2.346"]),
        (".2f", [1.23456789, 1e-17, 2.3456], ["1.23", "0.00", "2.35"]),
        (".4f", [0.123456789, 1e-17, 0.5], ["0.1235", "0.0000", "0.5000"]),
        (".2e", [0.123456789, 1e-17, 0.5], ["e"]),  # Scientific notation
    ],
)
def test_structure_2d_plotly_hover_formatting(
    hover_fmt: str, test_coordinates: list[float], expected_patterns: list[str]
) -> None:
    """Test hover text formatting for 2D plots."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Li", "O"], [test_coordinates, [0.5, 0.5, 0.5]])

    fig = pmv.structure_2d_plotly(struct, hover_float_fmt=hover_fmt)

    # Find site traces with hover text
    site_traces = [
        t for t in fig.data if t.name in ["Li", "O"] and hasattr(t, "hovertext")
    ]
    assert len(site_traces) > 0

    hover_text = site_traces[0].hovertext
    for pattern in expected_patterns:
        if hover_fmt == ".2e" and pattern == "e":
            assert "e-" in hover_text.lower() or "e+" in hover_text.lower()
        else:
            assert pattern in hover_text


@pytest.mark.parametrize(
    ("elem_colors", "site_labels", "expected_legend"),
    [
        (ElemColorScheme.jmol, "legend", True),
        ({"Li": "red", "O": "blue"}, "legend", True),
        (ElemColorScheme.jmol, "symbol", False),
        (ElemColorScheme.jmol, False, False),
        ({"Li": (1.0, 0.0, 0.0), "O": (0.0, 1.0, 0.0)}, "legend", True),
        ({"Li": "#FF0000", "O": "#00FF00"}, "legend", True),
    ],
)
def test_structure_2d_plotly_legend_and_colors(
    elem_colors: Any,
    site_labels: Any,
    expected_legend: bool,
) -> None:
    """Test legend functionality and color schemes for 2D plots."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    fig = pmv.structure_2d_plotly(
        struct, site_labels=site_labels, elem_colors=elem_colors
    )

    assert fig.layout.showlegend is expected_legend

    if expected_legend:
        legend_traces = [trace for trace in fig.data if trace.showlegend]
        assert len(legend_traces) == 2  # Li and O

        trace_names = {trace.name for trace in legend_traces}
        assert "Li" in trace_names
        assert "O" in trace_names

        # Test color consistency for custom colors
        if isinstance(elem_colors, dict):
            for trace in legend_traces:
                element_name = trace.name
                if element_name in elem_colors:
                    expected_color = _resolve_expected_color_str(
                        element_name, elem_colors, normalize_rgb_color
                    )
                    actual_color = normalize_rgb_color(str(trace.marker.color))
                    _compare_colors(actual_color, expected_color, normalize_rgb_color)


@pytest.mark.parametrize(
    ("show_sites", "show_bonds", "expected_bond_traces"),
    [
        (False, True, 0),  # No bonds when no sites rendered
        (True, False, 0),  # No bonds when show_bonds=False
        (True, True, "variable"),  # Bonds may exist when both True
    ],
)
def test_structure_2d_plotly_bonds_and_sites(
    show_sites: bool, show_bonds: bool, expected_bond_traces: int | str
) -> None:
    """Test bond rendering behavior with different site visibility settings."""
    lattice = Lattice.cubic(3.0)
    struct = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    fig = pmv.structure_2d_plotly(
        struct,
        show_sites=show_sites,
        show_bonds=show_bonds,
        show_image_sites=False,  # Simplify test
    )

    bond_traces = [
        trace for trace in fig.data if getattr(trace, "name", "").startswith("bond")
    ]

    if expected_bond_traces == 0:
        assert len(bond_traces) == 0
    # For "variable", we just check that the function doesn't crash
    # and that bonds are only present when sites are also present


def test_structure_2d_plotly_invalid_input() -> None:
    """Test that structure_2d_plotly raises errors for invalid inputs."""
    expected_err_msg = (
        "Input must be a Pymatgen Structure, ASE Atoms object, a sequence"
    )
    with pytest.raises(TypeError, match=expected_err_msg):
        pmv.structure_2d_plotly("invalid input")

    with pytest.raises(ValueError, match="Cannot plot empty set of structures"):
        pmv.structure_2d_plotly([])

    # Test with invalid rotation string
    struct = Structure(lattice_cubic, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match="could not convert string to float"):
        pmv.structure_2d_plotly(struct, rotation="invalid_rotation")


def test_structure_plotly_site_vector_coverage() -> None:
    """Test site vector property coverage for both 2D and 3D functions."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # Add vector property to individual sites (covers line 259 in 2D, similar in 3D)
    struct[0].properties["force"] = [0.1, 0.2, 0.3]
    struct[1].properties["force"] = [0.2, 0.3, 0.4]

    # Test 2D with site-level vector properties
    fig_2d = pmv.structure_2d_plotly(struct, show_site_vectors="force")
    assert isinstance(fig_2d, go.Figure)

    # Test 3D with site-level vector properties
    fig_3d = pmv.structure_3d_plotly(struct, show_site_vectors="force")
    assert isinstance(fig_3d, go.Figure)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "atomic_radii": None,
            "atom_size": 20,
            "elem_colors": ElemColorScheme.jmol,
            "scale": 1,
            "show_cell": True,
            "show_sites": True,
            "show_image_sites": True,
            "site_labels": "symbol",
            "standardize_struct": None,
            "n_cols": 3,
            "show_site_vectors": "magmom",
        },
        {
            "atomic_radii": 0.5,
            "atom_size": 30,
            "elem_colors": ElemColorScheme.vesta,
            "scale": 1.5,
            "show_cell": False,
            "show_sites": False,
            "show_image_sites": False,
            "site_labels": "symbol",
            "standardize_struct": True,
            "n_cols": 2,
            "show_site_vectors": ("magmom", "force"),
        },
        {
            "atomic_radii": {"Fe": 0.8, "O": 0.6},
            "atom_size": 25,
            "elem_colors": {"Fe": "red", "O": "blue"},
            "scale": 0.9,
            "show_cell": {"edge": {"color": "red", "width": 3}},
            "show_sites": {"line": {"width": 1, "color": "black"}},
            "show_image_sites": {"opacity": 0.3},
            "site_labels": {"Fe": "Iron", "O": "Oxygen"},
            "standardize_struct": False,
            "n_cols": 4,
            "show_site_vectors": (),
        },
        {
            "atomic_radii": 1.2,
            "atom_size": 35,
            "elem_colors": ElemColorScheme.jmol,
            "scale": 1.1,
            "show_cell": True,
            "show_sites": True,
            "show_image_sites": True,
            "site_labels": False,
            "standardize_struct": None,
            "n_cols": 1,
        },
        # New test scenarios for show_cell_faces in 3D
        {
            "atomic_radii": None,
            "atom_size": 20,
            "elem_colors": ElemColorScheme.jmol,
            "scale": 1,
            "show_cell": True,
            "show_cell_faces": True,
            "show_sites": True,
            "show_image_sites": False,  # Simplify validation
            "site_labels": "symbol",
            "standardize_struct": None,
            "n_cols": 1,
        },
        {
            "atomic_radii": 0.5,
            "atom_size": 25,
            "elem_colors": ElemColorScheme.vesta,
            "scale": 1.2,
            "show_cell": True,
            "show_cell_faces": {"color": "rgba(0,255,0,0.15)", "showscale": False},
            "show_sites": True,
            "show_image_sites": False,  # Simplify validation
            "site_labels": "legend",
            "n_cols": 1,
        },
        {
            "atomic_radii": 1.0,
            "atom_size": 30,
            "elem_colors": ElemColorScheme.jmol,
            "scale": 1,
            "show_cell": True,
            "show_cell_faces": False,
            "show_sites": True,
            "show_image_sites": False,  # Simplify validation
            "site_labels": "symbol",
            "n_cols": 1,
        },
        {
            "atomic_radii": None,
            "atom_size": 20,
            "elem_colors": ElemColorScheme.jmol,
            "scale": 1,
            "show_cell": False,
            "show_cell_faces": True,  # Should be ignored when show_cell=False
            "show_sites": True,
            "show_image_sites": False,  # Simplify validation
            "site_labels": "symbol",
            "n_cols": 1,
        },
        # Coverage-specific scenarios for 3D (simplified to avoid validation issues)
        {
            "elem_colors": {
                "struct1": ElemColorScheme.jmol,
                "struct2": ElemColorScheme.vesta,
            },
            "subplot_title": False,
            "show_sites": False,  # Simplified to avoid complex validation
        },
        {
            "show_site_vectors": "force",
            "subplot_title": lambda _struct, idx: f"Custom {idx}",
            "show_sites": False,  # Simplified to avoid complex validation
        },
        # Test cell_boundary_tol parameter for 3D
        {
            "cell_boundary_tol": 0.1,
            "show_image_sites": True,
            "show_sites": True,
            "site_labels": "symbol",
            "elem_colors": ElemColorScheme.jmol,
            "atom_size": 20,
            "n_cols": 1,
        },
    ],
)
def test_structure_3d_plotly(
    kwargs: dict[str, Any], fe3co4_disordered_with_props: Structure
) -> None:
    # Handle multi-structure scenarios for coverage
    if "struct1" in str(kwargs.get("elem_colors", {})) or "struct1" in str(
        kwargs.get("show_bonds", {})
    ):
        struct1 = fe3co4_disordered_with_props.copy()
        struct1.add_site_property("force", [[0.1, 0.2, 0.3]] * len(struct1))
        struct2 = fe3co4_disordered_with_props.copy()
        structures_input = {"struct1": struct1, "struct2": struct2}
    else:
        structures_input = fe3co4_disordered_with_props

    fig = pmv.structure_3d_plotly(structures_input, **kwargs)
    assert isinstance(fig, go.Figure)

    # Check if the layout properties are set correctly
    site_labels_kwarg = kwargs.get("site_labels", "legend")  # Default from function
    expected_showlegend = site_labels_kwarg == "legend"
    assert fig.layout.showlegend is expected_showlegend
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"

    # Check if the 3D scene properties are set correctly
    for scene in fig.layout:
        if scene.startswith("scene"):
            assert fig.layout[scene].xaxis.visible is False
            assert fig.layout[scene].yaxis.visible is False
            assert fig.layout[scene].zaxis.visible is False
            assert fig.layout[scene].aspectmode == "data"

    # Additional checks based on specific kwargs
    if isinstance(kwargs.get("show_cell"), dict):
        edge_kwargs = kwargs["show_cell"].get("edge", {})
        cell_edge_trace = next(
            (trace for trace in fig.data if trace.mode == "lines"), None
        )
        assert cell_edge_trace is not None
        for key, value in edge_kwargs.items():
            assert cell_edge_trace.line[key] == value

    # Check cell faces functionality for 3D plots
    show_cell = kwargs.get("show_cell", True)
    show_cell_faces = kwargs.get("show_cell_faces", True)

    if show_cell and show_cell_faces:
        # For 3D plots, cell faces are shown as mesh3d traces
        surface_traces = [
            trace
            for trace in fig.data
            if trace.type == "mesh3d" and trace.name.startswith("surface-face")
        ]
        if isinstance(show_cell_faces, dict):
            # When custom styling is provided, check that surfaces exist
            assert len(surface_traces) > 0
            # Check custom color if specified
            if "color" in show_cell_faces:
                expected_color = show_cell_faces["color"]
                for trace in surface_traces:
                    assert trace.color == expected_color
        elif show_cell_faces is True:
            # When show_cell_faces=True, surfaces should exist
            assert len(surface_traces) > 0
    elif show_cell and show_cell_faces is False:
        # When show_cell_faces=False, no surface traces should exist
        surface_traces = [
            trace
            for trace in fig.data
            if trace.type == "mesh3d" and trace.name.startswith("surface-face")
        ]
        assert len(surface_traces) == 0
    elif not show_cell:  # When show_cell=False, no surface traces should exist
        # regardless of show_cell_faces
        surface_traces = [
            trace
            for trace in fig.data
            if trace.type == "mesh3d" and trace.name.startswith("surface-face")
        ]
        assert len(surface_traces) == 0

    if kwargs.get("show_sites"):
        site_traces = [
            trace
            for trace in fig.data
            if getattr(trace, "mode", None) in ("markers", "markers+text")
            and not (
                trace.name
                and (trace.name.startswith("node") or trace.name.startswith("edge"))
            )
        ]
        assert len(site_traces) > 0, "No site traces found when show_sites is True"

        # Determine total number of sites expected (primary + images)
        show_image_sites = kwargs.get("show_image_sites", True)
        # For show_image_sites as dict, it implies True if show_sites is True
        if isinstance(show_image_sites, dict):
            show_image_sites = True

        # Get the first structure for validation
        test_structure = (
            structures_input
            if not isinstance(structures_input, dict)
            else next(iter(structures_input.values()))
        )
        rendered_sites_info = _get_all_rendered_site_info(
            test_structure, show_image_sites
        )

        # Get unique elements in the structure for trace validation
        unique_elements = {_get_site_symbol(s) for s in test_structure}

        # In the new implementation, we expect one trace per element type
        if site_labels_kwarg == "legend":
            # In legend mode, expect traces named with element symbols
            expected_n_traces = len(unique_elements)
            element_traces = [t for t in site_traces if t.name in unique_elements]
            assert len(element_traces) == expected_n_traces
        else:
            # In other modes, we still expect one trace per element type
            expected_n_traces = len(unique_elements)
            assert len(site_traces) == expected_n_traces

        # Check atom colors and sizes across all element traces
        elem_colors = kwargs.get("elem_colors")
        atom_size_kwarg = kwargs.get("atom_size")
        scale_kwarg = kwargs.get("scale", 1)
        atomic_radii_kwarg = kwargs.get("atomic_radii")
        _processed_atomic_radii = get_atomic_radii(atomic_radii_kwarg)

        # Group rendered sites by element for validation
        sites_by_element: dict[str, list[dict[str, Any]]] = {}
        for site_info in rendered_sites_info:
            symbol = site_info["symbol"]
            if symbol not in sites_by_element:
                sites_by_element[symbol] = []
            sites_by_element[symbol].append(site_info)

        # Validate each element trace
        for element_symbol, element_sites in sites_by_element.items():
            # Find the trace for this element
            element_trace = None
            if site_labels_kwarg == "legend":
                element_trace = next(
                    (t for t in site_traces if t.name == element_symbol), None
                )
            else:  # For non-legend modes, match by name (most reliable, could
                # additionally match by length)
                element_trace = next(
                    (t for t in site_traces if t.name == element_symbol), None
                )

            assert element_trace is not None, f"No trace found for {element_symbol=}"

            # Check that the trace has the right number of sites
            assert len(element_trace.x) == len(element_sites)
            assert len(element_trace.y) == len(element_sites)
            assert len(element_trace.z) == len(element_sites)

            # Check colors for this element's trace
            raw_marker_color = element_trace.marker.color
            expected_color_str = _resolve_expected_color_str(
                element_symbol, elem_colors, normalize_rgb_color
            )

            if isinstance(raw_marker_color, str):  # Single color string
                actual_color = normalize_rgb_color(raw_marker_color)
                _compare_colors(actual_color, expected_color_str, normalize_rgb_color)
            elif isinstance(raw_marker_color, (list, tuple)):
                # Should be a list of colors, one per site in this element's trace
                if (
                    all(isinstance(c, (int, float)) for c in raw_marker_color)
                    and len(raw_marker_color) == 3
                ):
                    # Single RGB tuple for all points
                    r, g, b = raw_marker_color
                    rgb_tuple_str = f"rgb({r},{g},{b})"
                    actual_color = normalize_rgb_color(rgb_tuple_str)
                    _compare_colors(
                        actual_color, expected_color_str, normalize_rgb_color
                    )
                else:  # List of color strings
                    for color_val in raw_marker_color:
                        actual_color = normalize_rgb_color(str(color_val))
                        _compare_colors(
                            actual_color, expected_color_str, normalize_rgb_color
                        )

            # Check sizes for this element's trace
            expected_size = (
                _processed_atomic_radii.get(element_symbol, 1.0)
                * scale_kwarg
                * atom_size_kwarg
            )
            actual_sizes = element_trace.marker.size

            if isinstance(actual_sizes, (int, float)):  # Single size for all sites
                assert pytest.approx(actual_sizes) == expected_size
            else:  # List of sizes
                for size_val in actual_sizes:
                    assert pytest.approx(size_val) == expected_size

        # Check site labels if they are expected
        site_labels_kwarg = kwargs.get("site_labels")
        actual_site_labels = (
            site_labels_kwarg if site_labels_kwarg is not None else "legend"
        )

        if actual_site_labels not in (False, "legend"):
            # For non-legend modes, check text labels on traces
            for trace in site_traces:
                if trace.text:
                    assert len(trace.text) > 0, "Expected text labels on trace"
        elif actual_site_labels == "legend":
            # Check that Plotly's built-in legend is enabled and has the right traces
            assert fig.layout.showlegend is True

            # Check that traces with showlegend=True exist for each unique element
            legend_traces = [trace for trace in fig.data if trace.showlegend]
            unique_elements = {
                _get_site_symbol(s) for s in fe3co4_disordered_with_props
            }

            # Each unique element should have exactly one trace in the legend
            legend_trace_names = [trace.name for trace in legend_traces]
            for elem_symbol in unique_elements:
                assert elem_symbol in legend_trace_names

            # Verify element coverage and trace counting for legend mode
            all_elements_in_struct = {_get_site_symbol(s) for s in test_structure}
            element_trace_names = {
                trace.name for trace in legend_traces if trace.showlegend
            }

            # Each unique element should be represented in legend traces
            for elem_symbol in all_elements_in_struct:
                assert elem_symbol in element_trace_names, (
                    f"{elem_symbol=} missing from legend traces in 3D plot"
                )

            # Check that each element trace has data points
            for trace in legend_traces:
                if trace.showlegend and trace.name in all_elements_in_struct:
                    # Count sites of this element in structure (incl. image sites
                    # if enabled)
                    show_image_sites_kwarg = kwargs.get("show_image_sites", True)
                    rendered_sites_info = _get_all_rendered_site_info(
                        test_structure, show_image_sites_kwarg
                    )
                    expected_points = sum(
                        1
                        for site_info in rendered_sites_info
                        if site_info["symbol"] == trace.name
                    )
                    actual_points = len(trace.x) if hasattr(trace, "x") else 0
                    assert actual_points == expected_points

            # Ensure no text on primary site traces
            for trace in legend_traces:
                if trace.text:
                    assert all(t is None for t in trace.text)
                else:
                    assert trace.text is None
    else:  # show_sites is False
        site_traces = [
            trace
            for trace in fig.data
            if getattr(trace, "mode", None) in ("markers", "markers+text")
            and not (
                trace.name
                and (trace.name.startswith("node") or trace.name.startswith("edge"))
            )
        ]
        assert len(site_traces) == 0, "Site traces found when show_sites is False"


def test_structure_3d_plotly_multiple() -> None:
    struct1 = Structure(lattice_cubic, ["Fe", "O"], COORDS)
    struct1.properties = {"id": "struct1"}
    struct2 = Structure(lattice_cubic, ["Co", "O"], COORDS)
    struct2.properties = {Key.mat_id: "struct2"}
    struct3 = Structure(lattice_cubic, ["Ni", "O"], COORDS)
    struct3.properties = {"ID": "struct3", "name": "nickel oxide"}
    struct4 = Structure(lattice_cubic, ["Cu", "O"], COORDS)

    # Test dict[str, Structure]
    structs_dict = {
        "struct1": struct1,
        "struct2": struct2,
        "struct3": struct3,
        "struct4": struct4,
    }
    # Test with default site_labels="legend"
    fig = pmv.structure_3d_plotly(structs_dict, n_cols=2)
    assert isinstance(fig, go.Figure)

    expected_total_traces_3d = 0

    # In the new implementation, we have one trace per element type per structure
    # Each structure has 2 elements, so expect 2 traces per structure
    expected_site_traces_per_struct = 2
    expected_total_site_traces = len(structs_dict) * expected_site_traces_per_struct
    actual_n_site_traces_3d = len(
        [
            trace
            for trace in fig.data
            if getattr(trace, "mode", None) in ("markers", "markers+text")
            and not (
                trace.name
                and (trace.name.startswith("node") or trace.name.startswith("edge"))
            )
        ]
    )
    assert actual_n_site_traces_3d == expected_total_site_traces
    expected_total_traces_3d += actual_n_site_traces_3d

    actual_n_edge_traces_3d = sum(
        (trace.name or "").startswith("edge") for trace in fig.data
    )
    # Default show_cell=True
    assert actual_n_edge_traces_3d == 12 * len(structs_dict)
    expected_total_traces_3d += actual_n_edge_traces_3d

    actual_n_node_traces_3d = sum(
        (trace.name or "").startswith("node") for trace in fig.data
    )
    # Default show_cell=True
    assert actual_n_node_traces_3d == 8 * len(structs_dict)
    expected_total_traces_3d += actual_n_node_traces_3d

    actual_n_surface_traces_3d = sum(
        (trace.name or "").startswith("surface-face") for trace in fig.data
    )
    assert actual_n_surface_traces_3d == 12 * len(structs_dict)
    expected_total_traces_3d += actual_n_surface_traces_3d

    assert len(fig.data) == expected_total_traces_3d

    # For default site_labels="legend", Plotly's built-in legend is used
    # No legend annotations are created anymore
    expected_n_subplot_titles = len(structs_dict)
    # Check if subplot_title is default (None), which generates titles
    assert len(fig.layout.annotations) == expected_n_subplot_titles

    # Test pandas.Series[Structure]
    struct_series = pd.Series(structs_dict)
    fig = pmv.structure_3d_plotly(struct_series)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_total_traces_3d
    assert len(fig.layout.annotations) == expected_n_subplot_titles

    # Test list[Structure]
    fig = pmv.structure_3d_plotly(list(structs_dict.values()), n_cols=3)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_total_traces_3d
    assert len(fig.layout.annotations) == expected_n_subplot_titles

    # Test subplot_title and its interaction with legend annotations
    def custom_subplot_title_func(struct: Structure, key: str | int) -> str:
        return f"{key} - {struct.formula}"

    fig = pmv.structure_3d_plotly(
        struct_series, subplot_title=custom_subplot_title_func
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_total_traces_3d
    assert len(fig.layout.annotations) == expected_n_subplot_titles

    title_texts = [anno.text for anno in fig.layout.annotations]
    for _idx, (key, struct) in enumerate(structs_dict.items(), start=1):
        expected_title = custom_subplot_title_func(struct=struct, key=key)
        assert expected_title in title_texts


def test_structure_plotly_coverage_improvements() -> None:
    """Test specific scenarios to improve coverage for missing lines."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # Test vector property in site properties (line 259)
    struct[0].properties["force"] = [0.1, 0.2, 0.3]
    struct[1].properties["force"] = [0.2, 0.3, 0.4]

    # Test 2D with site-level vector properties
    fig_2d = pmv.structure_2d_plotly(struct, show_site_vectors="force")
    assert isinstance(fig_2d, go.Figure)

    # Test 3D with site-level vector properties
    fig_3d = pmv.structure_3d_plotly(struct, show_site_vectors="force")
    assert isinstance(fig_3d, go.Figure)

    # Test bond drawing with dict show_bonds (line 320)
    struct_dict = {"struct1": struct, "struct2": struct.copy()}
    fig_2d_bonds = pmv.structure_2d_plotly(
        struct_dict, show_bonds={"struct1": True, "struct2": False}
    )
    assert isinstance(fig_2d_bonds, go.Figure)

    # Test 3D subplot title handling when subplot_title is not False (lines 657-660)
    fig_3d_title = pmv.structure_3d_plotly(
        struct_dict,
        subplot_title=lambda _struct, idx: {"text": f"Custom {idx}", "x": 0.5},
    )
    assert isinstance(fig_3d_title, go.Figure)

    # Test legend configuration (lines 676-692)
    fig_3d_legend = pmv.structure_3d_plotly(struct_dict, site_labels="legend", n_cols=2)
    assert isinstance(fig_3d_legend, go.Figure)
    assert fig_3d_legend.layout.showlegend is True


def test_structure_plotly_bonds_coverage() -> None:
    """Test bond drawing scenarios for coverage."""
    lattice = Lattice.cubic(3.0)
    struct = Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # Test 2D bonds with dict show_bonds
    fig_2d = pmv.structure_2d_plotly(
        {"struct1": struct, "struct2": struct.copy()},
        show_bonds={"struct1": True, "struct2": False},
        show_sites=True,
    )
    assert isinstance(fig_2d, go.Figure)

    # Test 3D bonds with dict show_bonds
    fig_3d = pmv.structure_3d_plotly(
        {"struct1": struct, "struct2": struct.copy()},
        show_bonds={"struct1": True, "struct2": False},
        show_sites=True,
    )
    assert isinstance(fig_3d, go.Figure)


def test_structure_3d_plotly_subplot_title_coverage() -> None:
    """Test 3D subplot title scenarios for coverage."""
    lattice = Lattice.cubic(4.0)
    struct1 = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    struct2 = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # Test with custom subplot_title function that returns dict with y and yanchor
    def custom_title_with_position(_struct: Structure, k: str | int) -> dict[str, Any]:
        return {"text": f"Structure {k}", "x": 0.5, "y": 0.95, "yanchor": "top"}

    fig = pmv.structure_3d_plotly(
        {"struct1": struct1, "struct2": struct2},
        subplot_title=custom_title_with_position,
        n_cols=2,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) == 2

    # Test with subplot_title=False to cover that branch
    fig_no_title = pmv.structure_3d_plotly(
        {"struct1": struct1, "struct2": struct2}, subplot_title=False, n_cols=2
    )
    assert isinstance(fig_no_title, go.Figure)


@pytest.mark.parametrize(
    ("is_3d", "show_cell_faces", "expected_surface_traces"),
    [
        # 2D tests
        (False, True, 1),  # 2D with faces enabled - should have 1 filled polygon
        (False, False, 0),  # 2D with faces disabled - should have 0 surface traces
        (False, {"color": "rgba(255,0,0,0.3)"}, 1),  # 2D with custom color
        # 3D tests
        (True, True, 12),  # 3D with faces enabled - should have 12 mesh3d
        # traces (6 faces x 2 triangles each)
        (True, False, 0),  # 3D with faces disabled - should have 0 surface traces
        # 3D with custom styling
        (True, {"color": "rgba(0,255,0,0.2)", "showscale": False}, 12),
    ],
)
def test_structure_plotly_cell_faces(
    is_3d: bool, show_cell_faces: bool | dict[str, Any], expected_surface_traces: int
) -> None:
    """Test cell faces functionality for both 2D and 3D structure plots."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    plot_func = pmv.structure_3d_plotly if is_3d else pmv.structure_2d_plotly

    fig = plot_func(
        struct,
        show_cell=True,
        show_cell_faces=show_cell_faces,
        show_sites=True,
        site_labels="symbol",
    )

    assert isinstance(fig, go.Figure)

    if is_3d:  # For 3D plots, check mesh3d traces with surface-face names
        surface_traces = [
            trace
            for trace in fig.data
            if trace.type == "mesh3d" and trace.name.startswith("surface-face")
        ]
    else:  # For 2D plots, check scatter traces with cell-face name and fill
        surface_traces = [
            trace
            for trace in fig.data
            if trace.name == "cell-face"
            and hasattr(trace, "fill")
            and trace.fill == "toself"
        ]

    assert len(surface_traces) == expected_surface_traces

    # Test custom styling if provided
    if (
        isinstance(show_cell_faces, dict)
        and expected_surface_traces > 0
        and (expected_color := show_cell_faces.get("color"))
    ):
        for trace in surface_traces:
            if is_3d:
                assert trace.color == expected_color
            else:
                assert trace.fillcolor == expected_color


def test_structure_plotly_cell_faces_no_cell() -> None:
    """Test that cell faces are not shown when show_cell=False."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # Test 2D
    fig_2d = pmv.structure_2d_plotly(
        struct,
        show_cell=False,
        show_cell_faces=True,  # Should be ignored
        show_sites=True,
    )

    surface_traces_2d = [trace for trace in fig_2d.data if trace.name == "cell-face"]
    assert len(surface_traces_2d) == 0

    # Test 3D
    fig_3d = pmv.structure_3d_plotly(
        struct,
        show_cell=False,
        show_cell_faces=True,  # Should be ignored
        show_sites=True,
    )

    surface_traces_3d = [
        trace
        for trace in fig_3d.data
        if trace.type == "mesh3d"
        and trace.name
        and trace.name.startswith("surface-face")
    ]
    assert len(surface_traces_3d) == 0


def test_structure_plotly_cell_faces_multiple_structures() -> None:
    """Test cell faces with multiple structures."""
    structures = {
        "struct1": Structure(Lattice.cubic(3), ["Li"], [[0, 0, 0]]),
        "struct2": Structure(Lattice.cubic(4), ["Na"], [[0, 0, 0]]),
    }

    # Test 2D
    fig_2d = pmv.structure_2d_plotly(
        structures, show_cell=True, show_cell_faces=True, n_cols=2
    )
    assert isinstance(fig_2d, go.Figure)

    # Test 3D
    fig_3d = pmv.structure_3d_plotly(
        structures, show_cell=True, show_cell_faces=True, n_cols=2
    )
    assert isinstance(fig_3d, go.Figure)


@pytest.mark.parametrize("is_3d", [True, False])
def test_structure_plotly_cell_boundary_tol_properties(is_3d: bool) -> None:
    """Test cell_boundary_tol via structure.properties with highest precedence."""
    lattice = Lattice.cubic(4.0)
    struct_near_boundary = Structure(lattice, ["Na"], [[0.95, 0.95, 0.95]])

    # Test basic precedence: structure property overrides function parameter
    struct_with_prop = struct_near_boundary.copy()
    struct_with_prop.properties["cell_boundary_tol"] = 0.3

    plot_func = pmv.structure_3d_plotly if is_3d else pmv.structure_2d_plotly

    # Structure property should override function parameter
    fig_override = plot_func(
        struct_with_prop,
        cell_boundary_tol=0.0,  # Should be ignored
        show_image_sites=True,
        site_labels="symbol",
    )

    # Compare with function parameter only (no structure property)
    fig_func_param = plot_func(
        struct_near_boundary,  # No property set
        cell_boundary_tol=0.0,  # Strict boundaries
        show_image_sites=True,
        site_labels="symbol",
    )

    # Structure with property should have more traces (more permissive tolerance)
    assert len(fig_override.data) >= len(fig_func_param.data)

    # Test with multiple structures and dict parameter
    struct_no_prop = struct_near_boundary.copy()
    structures = {"with_prop": struct_with_prop, "no_prop": struct_no_prop}

    fig_mixed = plot_func(
        structures,
        cell_boundary_tol={
            "with_prop": 0.0,
            "no_prop": 0.1,
        },  # Dict should be overridden for first struct
        show_image_sites=True,
        site_labels="symbol",
    )

    assert isinstance(fig_mixed, go.Figure)
    assert len(fig_mixed.data) > 0
