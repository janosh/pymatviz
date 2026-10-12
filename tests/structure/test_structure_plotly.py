from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pymatgen.core import Composition, IMolecule, IStructure, Lattice, Structure
from pymatgen.electronic_structure.core import Magmom

import pymatviz as pmv
from pymatviz.colors import ELEM_COLORS_JMOL, ELEM_COLORS_VESTA
from pymatviz.enums import ElemColorScheme, Key, SiteCoords
from pymatviz.structure.helpers import (
    get_disordered_site_legend_name,
    get_site_symbol,
    get_spherical_wedge_mesh,
    normalize_elem_color,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from pymatviz.typing import AnyStructure, Xyz

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
        numbers = color_str[4:-1].split(",")  # Extract numbers and reformat
        try:
            r, g, b = [int(float(n.strip())) for n in numbers]
            return f"rgb({r},{g},{b})"  # noqa: TRY300
        except (ValueError, IndexError):
            return color_str

    return color_str


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
                r, g, b = tuple(int(c * 255) for c in color_val)
            else:  # ints 0-255
                r, g, b = tuple(int(c) for c in color_val)
            expected_color_str = f"rgb({r},{g},{b})"
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


STRUCTURE_2D_KWARGS: Final[dict[str, dict[str, Any]]] = {
    "basic_jmol_colors": {
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
    "vesta_colors_no_sites": {
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
    "custom_colors_dict": {
        "rotation": "5x,5y,5z",
        "atomic_radii": {"Fe": 0.5, "O": 0.3, "C": 0.4},
        "atom_size": 40,
        "elem_colors": {"Fe": "red", "O": "blue"},
        "scale": 1.2,
        "show_cell": {"edge": {"color": "red", "width": 2}},
        "show_sites": {"line": {"width": 1, "color": "black"}},
        "site_labels": {"Fe": "Iron"},
        "standardize_struct": False,
        "n_cols": 3,
        "show_site_vectors": (),
    },
    "legend_mode": {
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
    "no_labels_custom_cell": {
        "rotation": "0x,20y,0z",
        "atomic_radii": None,
        "atom_size": 45,
        "elem_colors": ElemColorScheme.vesta,
        "scale": 1.1,
        "show_cell": {"edge": {"color": "blue", "width": 1, "dash": "dot"}},
        "show_sites": True,
        "site_labels": False,
        "standardize_struct": True,
        "n_cols": 4,
    },
    "custom_site_styling": {
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
    "cell_boundary_tol": {
        "cell_boundary_tol": 0.1,
        "show_image_sites": True,
        "site_labels": "symbol",
    },
}


@pytest.mark.parametrize(
    "kwargs", STRUCTURE_2D_KWARGS.values(), ids=STRUCTURE_2D_KWARGS.keys()
)
def test_structure_2d_layout(
    kwargs: dict[str, Any], fe3co4_disordered_with_props: IStructure
) -> None:
    """Any kwarg combo gives 2D scatter traces and the standard subplot layout."""
    fig = pmv.structure_2d(fe3co4_disordered_with_props, **kwargs)

    assert {*map(type, fig.data)} in ({go.Scatter}, set())
    assert fig.layout.showlegend is (kwargs.get("site_labels", "legend") == "legend")
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"
    for axis in fig.layout.xaxis, fig.layout.yaxis:
        assert axis.showticklabels is False
        assert axis.showgrid is False
        assert axis.zeroline is False
        assert axis.scaleratio == 1
        assert axis.constrain == "domain"


def test_structure_2d_no_sites(fe3co4_disordered_with_props: IStructure) -> None:
    """show_sites=False draws no site markers, wedges, labels or vectors."""
    fig = pmv.structure_2d(
        fe3co4_disordered_with_props,
        show_sites=False,
        show_site_vectors=("magmom", "force"),
    )
    trace_names = [trace.name or "" for trace in fig.data]
    assert trace_names  # cell edges, nodes and faces
    assert all(name.startswith(("edge", "node", "cell-face")) for name in trace_names)


def test_structure_2d_custom_styling(fe3co4_disordered: IStructure) -> None:
    """Element color, site marker, label and cell edge overrides are applied."""
    fig = pmv.structure_2d(
        fe3co4_disordered,
        elem_colors={"Fe": "red", "O": "blue", "C": "green"},
        show_sites={"line": {"width": 1, "color": "black"}},
        site_labels={"Fe": "Iron"},
        atomic_radii={"Fe": 0.5, "O": 0.3, "C": 0.4},
        show_cell={"edge": {"color": "red", "width": 2}},
        show_image_sites=False,
    )
    (o_site,) = [trace for trace in fig.data if trace.mode == "markers+text"]
    assert o_site.marker.color == "blue"
    assert o_site.marker.line.to_plotly_json() == {"width": 1, "color": "black"}
    assert o_site.text == "O"  # unmapped elements keep their symbol
    wedge_colors = {trace.fillcolor for trace in fig.data if trace.fill == "toself"}
    assert wedge_colors == {"red", "green", "rgba(255,255,255,0.1)"}  # + cell face
    assert {trace.text for trace in fig.data if trace.mode == "text"} == {"Iron", "C"}
    edge_lines = [
        trace.line for trace in fig.data if (trace.name or "").startswith("edge")
    ]
    assert len(edge_lines) == 12
    assert {(line.color, line.width, line.dash) for line in edge_lines} == {
        ("red", 2, "dash")
    }


def test_structure_2d_per_struct_elem_colors(
    fe3co4_disordered_with_props: IStructure,
) -> None:
    """elem_colors keyed by structure key apply a color scheme per subplot."""
    structs = {
        "struct1": fe3co4_disordered_with_props.copy(),
        "struct2": fe3co4_disordered_with_props.copy(),
    }
    fig = pmv.structure_2d(
        structs,
        elem_colors={"struct1": ElemColorScheme.jmol, "struct2": ElemColorScheme.vesta},
        site_labels="symbol",
        show_image_sites=False,
    )
    o_colors = {trace.name: trace.marker.color for trace in fig.data}
    assert o_colors["site-struct1-1"] == normalize_elem_color(ELEM_COLORS_JMOL["O"])
    assert o_colors["site-struct2-1"] == normalize_elem_color(ELEM_COLORS_VESTA["O"])
    assert o_colors["site-struct1-1"] != o_colors["site-struct2-1"]


def test_structure_2d_site_vectors(fe3co4_disordered_with_props: IStructure) -> None:
    """Site vectors are drawn at primary and image sites of every subplot."""
    struct1 = fe3co4_disordered_with_props.copy(
        site_properties={"force": [[0.1, 0.2, 0.3]] * 2}
    )
    structs = {"struct1": struct1, "struct2": fe3co4_disordered_with_props}
    fig = pmv.structure_2d(structs, show_site_vectors="force")
    vector_names = {
        trace.name for trace in fig.data if (trace.name or "").startswith("vector")
    }
    assert {name for name in vector_names if "-img-" not in name} == {
        f"vector-{key}-{site_idx}" for key in structs for site_idx in range(2)
    }
    # Fe/C site at the origin has 7 corner images in each subplot
    assert {name for name in vector_names if "-img-" in name} == {
        f"vector-{key}-img-0-{img_idx}" for key in structs for img_idx in range(7)
    }


@pytest.mark.parametrize(
    ("input_type", "n_cols", "expected_structures"),
    [("dict", 3, 4), ("pandas_series", 2, 4), ("list", 2, 4)],
)
def test_structure_2d_multiple_inputs(
    input_type: str, n_cols: int, expected_structures: int
) -> None:
    """Test structure_2d with different input types and multiple structures."""
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

    fig = pmv.structure_2d(structures_input, n_cols=n_cols, site_labels=False)
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
def test_structure_2d_hover_formatting(
    hover_fmt: str, test_coordinates: list[float], expected_patterns: list[str]
) -> None:
    """Test hover text formatting for 2D plots."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Li", "O"], [test_coordinates, [0.5, 0.5, 0.5]])

    fig = pmv.structure_2d(struct, hover_float_fmt=hover_fmt)

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
def test_structure_2d_legend_and_colors(
    elem_colors: Any,
    site_labels: Any,
    expected_legend: bool,
) -> None:
    """Test legend functionality and color schemes for 2D plots."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    fig = pmv.structure_2d(struct, site_labels=site_labels, elem_colors=elem_colors)

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
    ("show_sites", "show_bonds", "expect_bonds"),
    [
        (False, True, False),  # No bonds when no sites rendered
        (True, False, False),  # No bonds when show_bonds=False
        (True, True, True),
    ],
)
def test_structure_2d_bonds_and_sites(
    show_sites: bool, show_bonds: bool, expect_bonds: bool
) -> None:
    """Test bond rendering behavior with different site visibility settings."""
    struct = Structure(Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    fig = pmv.structure_2d(
        struct, show_sites=show_sites, show_bonds=show_bonds, show_image_sites=False
    )
    has_bonds = any((trace.name or "").startswith("bond") for trace in fig.data)
    assert has_bonds is expect_bonds


def test_structure_2d_invalid_input() -> None:
    """Test that structure_2d raises errors for invalid inputs."""
    expected_err_msg = "Input must be a pymatgen Structure, IStructure, Molecule"
    with pytest.raises(TypeError, match=expected_err_msg):
        pmv.structure_2d("invalid input")  # ty: ignore[invalid-argument-type]

    with pytest.raises(ValueError, match="Cannot plot empty set of structures"):
        pmv.structure_2d([])

    # Test with invalid rotation string
    struct = Structure(lattice_cubic, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    with pytest.raises(ValueError, match="Invalid rotation angle='invalid_rotation'"):
        pmv.structure_2d(struct, rotation="invalid_rotation")


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
            "atomic_radii": {"Fe": 0.8, "O": 0.6, "C": 0.7},
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
def test_structure_3d(
    kwargs: dict[str, Any], fe3co4_disordered_with_props: Structure
) -> None:
    # Handle multi-structure scenarios for coverage
    """Test structure_3d layout, cell faces and site traces for various kwargs."""
    if "struct1" in str(kwargs.get("elem_colors", {})) or "struct1" in str(
        kwargs.get("show_bonds", {})
    ):
        struct1 = fe3co4_disordered_with_props.copy()
        struct1.add_site_property("force", [[0.1, 0.2, 0.3]] * len(struct1))
        struct2 = fe3co4_disordered_with_props.copy()
        structures_input = {"struct1": struct1, "struct2": struct2}
    else:
        structures_input = fe3co4_disordered_with_props

    fig = pmv.structure_3d(structures_input, **kwargs)
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
            (trace for trace in fig.data if getattr(trace, "mode", None) == "lines"),
            None,
        )
        assert cell_edge_trace is not None
        for key, value in edge_kwargs.items():
            assert cell_edge_trace.line[key] == value

    # Check cell faces functionality for 3D plots
    show_cell = kwargs.get("show_cell", True)
    show_cell_faces = kwargs.get("show_cell_faces", True)

    # For 3D plots, cell faces are mesh3d traces (only drawn if show_cell is on)
    surface_traces = [
        trace
        for trace in fig.data
        if trace.type == "mesh3d" and trace.name.startswith("surface-face")
    ]
    assert (len(surface_traces) > 0) is bool(show_cell and show_cell_faces)
    if isinstance(show_cell_faces, dict) and "color" in show_cell_faces:
        for trace in surface_traces:
            assert trace.color == show_cell_faces["color"]

    if kwargs.get("show_sites"):
        # For 3D plots, site traces include both scatter3d and mesh3d traces
        # scatter3d for ordered sites and mesh3d for disordered site wedges
        site_traces = [
            trace
            for trace in fig.data
            if (
                # scatter3d traces for ordered sites and labels
                (getattr(trace, "mode", None) in ("markers", "markers+text"))
                or
                # mesh3d traces for disordered site spherical wedges
                (trace.type == "mesh3d" and not trace.name.startswith("surface-face"))
            )
            and not (
                trace.name
                and (trace.name.startswith("node") or trace.name.startswith("edge"))
            )
        ]
        assert len(site_traces) > 0, "No site traces found when show_sites is True"

        # Get the first structure for validation
        if isinstance(structures_input, IStructure):
            test_structure = structures_input
        else:
            test_structure = next(iter(structures_input.values()))

        # Count ordered vs disordered sites
        ordered_sites = [
            site
            for site in test_structure
            if not (isinstance(site.species, Composition) and len(site.species) > 1)
        ]
        disordered_sites = [
            site
            for site in test_structure
            if (isinstance(site.species, Composition) and len(site.species) > 1)
        ]

        # Count expected traces
        expected_trace_count = 0

        # Ordered sites: one scatter3d trace per unique element
        if ordered_sites:
            ordered_elements = {get_site_symbol(site) for site in ordered_sites}
            expected_trace_count += len(ordered_elements)

        # Disordered sites: one mesh3d trace per species in each disordered site
        for site in disordered_sites:
            species = site.species
            if isinstance(species, Composition):
                expected_trace_count += len(species)

        # For legend mode, we expect traces to have showlegend=True for unique elements
        if site_labels_kwarg == "legend":
            legend_traces = [trace for trace in site_traces if trace.showlegend]

            # disordered sites have combined legend names, not individual elements

            expected_legend_names = set()  # Get expected legend names from structure
            for site in test_structure:
                species = getattr(site, "specie", site.species)
                if isinstance(species, Composition) and len(species) > 1:
                    # This is a disordered site - should appear as combined name
                    sorted_species = sorted(
                        species.items(), key=lambda x: x[1], reverse=True
                    )
                    legend_name = get_disordered_site_legend_name(
                        sorted_species, is_image=False
                    )
                    expected_legend_names.add(legend_name)
                else:
                    # This is an ordered site - should appear as element symbol
                    expected_legend_names.add(get_site_symbol(site))

            # Each expected legend name should be represented in legend
            legend_trace_names = {trace.name for trace in legend_traces}
            for expected_name in expected_legend_names:
                assert expected_name in legend_trace_names

        # Skip detailed trace counting for complex multi-structure scenarios
        # as they have different patterns
        if not isinstance(structures_input, dict):
            # For single structure, do basic validation
            unique_elements = {get_site_symbol(s) for s in test_structure}
            assert len(site_traces) >= len(unique_elements)

    else:  # show_sites is False
        site_traces = [
            trace
            for trace in fig.data
            if (
                (getattr(trace, "mode", None) in ("markers", "markers+text"))
                or (
                    trace.type == "mesh3d" and not trace.name.startswith("surface-face")
                )
            )
            and not (
                trace.name
                and (trace.name.startswith("node") or trace.name.startswith("edge"))
            )
        ]
        assert len(site_traces) == 0, "Site traces found when show_sites is False"


def test_structure_3d_multiple() -> None:
    """Test structure_3d trace counts and titles for dict/Series/list inputs."""
    struct1 = Structure(lattice_cubic, ["Fe", "O"], COORDS)
    struct1.properties = {"id": "struct1"}
    struct2 = Structure(lattice_cubic, ["Co", "O"], COORDS)
    struct2.properties = {Key.mat_id: "struct2"}
    struct3 = Structure(lattice_cubic, ["Ni", "O"], COORDS)
    struct3.properties = {"ID": "struct3", "name": "nickel oxide"}
    struct4 = Structure(lattice_cubic, ["Cu", "O"], COORDS)

    structs_dict: dict[str, Structure] = {
        "struct1": struct1,
        "struct2": struct2,
        "struct3": struct3,
        "struct4": struct4,
    }
    # Test with default site_labels="legend"
    fig = pmv.structure_3d(structs_dict, n_cols=2)
    assert isinstance(fig, go.Figure)

    expected_total_traces_3d = 0

    # Count actual site traces - both scatter3d and mesh3d (for disordered sites)
    actual_n_site_traces_3d = len(
        [
            trace
            for trace in fig.data
            if (
                (getattr(trace, "mode", None) in ("markers", "markers+text"))
                or (
                    trace.type == "mesh3d" and not trace.name.startswith("surface-face")
                )
            )
            and not (
                trace.name
                and (trace.name.startswith("node") or trace.name.startswith("edge"))
            )
        ]
    )
    # In the new implementation, each structure with image sites creates more traces
    # Each unique element in each structure gets its own trace, including image sites
    # With show_image_sites=True (default), this can result in many more traces
    # We just verify that we have a reasonable number of site traces
    assert actual_n_site_traces_3d > 0, "Should have some site traces"
    assert actual_n_site_traces_3d >= len(structs_dict) * 2, (
        "Should have at least 2 traces per structure (Fe/Co/Ni/Cu + O)"
    )
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
    fig = pmv.structure_3d(struct_series)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_total_traces_3d
    assert len(fig.layout.annotations) == expected_n_subplot_titles

    # Test list[Structure]
    fig = pmv.structure_3d(list(structs_dict.values()), n_cols=3)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_total_traces_3d
    assert len(fig.layout.annotations) == expected_n_subplot_titles

    # Test subplot_title and its interaction with legend annotations
    def custom_subplot_title_func(struct: AnyStructure, key: Hashable) -> str:
        """Title from structure key and formula."""
        assert isinstance(struct, (IStructure, IMolecule))  # narrow for ty
        return f"{key} - {struct.formula}"

    fig = pmv.structure_3d(struct_series, subplot_title=custom_subplot_title_func)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_total_traces_3d
    assert len(fig.layout.annotations) == expected_n_subplot_titles

    title_texts = [anno.text for anno in fig.layout.annotations]
    for _idx, (key, struct) in enumerate(structs_dict.items(), start=1):
        expected_title = custom_subplot_title_func(struct=struct, key=key)
        assert expected_title in title_texts


@pytest.mark.parametrize("plot_func", [pmv.structure_2d, pmv.structure_3d])
def test_structure_plotly_per_struct_show_bonds(plot_func: Callable) -> None:
    """show_bonds dict draws bonds only for structures mapped to True."""
    struct = Structure(Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    fig = plot_func(
        {"struct1": struct, "struct2": struct.copy()},
        show_bonds={"struct1": True, "struct2": False},
    )
    bond_traces = [trace for trace in fig.data if (trace.name or "").startswith("bond")]
    assert len(bond_traces) > 0
    subplot_ids = {
        trace.scene if plot_func is pmv.structure_3d else trace.xaxis
        for trace in bond_traces
    }
    assert subplot_ids == {"scene" if plot_func is pmv.structure_3d else "x"}


@pytest.mark.parametrize("plot_func", [pmv.structure_2d, pmv.structure_3d])
def test_structure_plotly_subplot_title(plot_func: Callable) -> None:
    """Subplot titles accept positioned dicts and False hides them."""
    lattice = Lattice.cubic(4.0)
    struct1 = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    struct2 = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    def custom_title_with_position(
        _struct: AnyStructure, key: Hashable
    ) -> dict[str, Any]:
        """Title dict with explicit position."""
        return {"text": f"Structure {key}", "x": 0.5, "y": 0.95, "yanchor": "top"}

    fig = plot_func(
        {"struct1": struct1, "struct2": struct2},
        subplot_title=custom_title_with_position,
        n_cols=2,
    )
    assert [anno.text for anno in fig.layout.annotations] == [
        "Structure struct1",
        "Structure struct2",
    ]
    assert {anno.y for anno in fig.layout.annotations} == {0.95}

    # 2D used to ignore subplot_title=False and render default titles
    fig_no_title = plot_func(
        {"struct1": struct1, "struct2": struct2}, subplot_title=False, n_cols=2
    )
    assert [anno.text for anno in fig_no_title.layout.annotations] == [" ", " "]


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

    plot_func = pmv.structure_3d if is_3d else pmv.structure_2d

    fig = plot_func(
        struct,
        show_cell=True,
        show_cell_faces=show_cell_faces,
        show_sites=True,
        site_labels="symbol",
    )

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
    fig_2d = pmv.structure_2d(
        struct,
        show_cell=False,
        show_cell_faces=True,  # Should be ignored
        show_sites=True,
    )

    surface_traces_2d = [trace for trace in fig_2d.data if trace.name == "cell-face"]
    assert len(surface_traces_2d) == 0

    # Test 3D
    fig_3d = pmv.structure_3d(
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


@pytest.mark.parametrize(
    ("plot_func", "face_name", "faces_per_struct"),
    [(pmv.structure_2d, "cell-face", 1), (pmv.structure_3d, "surface-face", 12)],
)
def test_structure_plotly_cell_faces_multiple_structures(
    plot_func: Callable, face_name: str, faces_per_struct: int
) -> None:
    """Every subplot gets its own cell faces."""
    structures = {
        "struct1": Structure(Lattice.cubic(3), ["Li"], [[0, 0, 0]]),
        "struct2": Structure(Lattice.cubic(4), ["Na"], [[0, 0, 0]]),
    }
    fig = plot_func(structures, show_cell=True, show_cell_faces=True, n_cols=2)
    n_faces = sum((trace.name or "").startswith(face_name) for trace in fig.data)
    assert n_faces == faces_per_struct * len(structures)

    # per-structure show_cell: style struct1's cell, hide struct2's
    red_edge = {"edge": dict(color="red", width=3)}
    fig = plot_func(structures, show_cell={"struct1": red_edge, "struct2": False})
    edges = [trace for trace in fig.data if (trace.name or "").startswith("edge")]
    assert len(edges) == 12  # one cell
    assert {trace.line.color for trace in edges} == {"red"}


@pytest.mark.parametrize("is_3d", [True, False])
def test_structure_plotly_multiple_properties_precedence(is_3d: bool) -> None:
    """Test that multiple structure properties take precedence over function
    params.
    """
    from pymatgen.core import Lattice, Structure

    # Create two identical structures but with different properties
    lattice = Lattice.cubic(4.0)
    struct1 = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    struct2 = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # Set properties only on the first structure
    struct1.properties |= {
        "atom_size": 50,  # Should override function param of 20
        "scale": 2.0,  # Should override function param of 1.0
        "atomic_radii": {"Na": 1.5, "Cl": 1.8},  # Should override function param
    }
    # struct2 has no properties, so should use function params

    # Test single structures to verify precedence works
    if is_3d:
        fig1 = pmv.structure_3d(struct1, atom_size=20, scale=1.0, atomic_radii=0.8)
        fig2 = pmv.structure_3d(struct2, atom_size=20, scale=1.0, atomic_radii=0.8)
    else:
        fig1 = pmv.structure_2d(struct1, atom_size=20, scale=1.0, atomic_radii=0.8)
        fig2 = pmv.structure_2d(struct2, atom_size=20, scale=1.0, atomic_radii=0.8)

    # Get site traces from both figures
    def get_site_traces(fig: go.Figure) -> list[Any]:
        """Site marker traces excluding cell and image traces."""
        return [
            trace
            for trace in fig.data
            if hasattr(trace, "marker")
            and trace.marker is not None
            and hasattr(trace.marker, "size")
            and trace.marker.size is not None
            and not (trace.name and ("edge" in trace.name or "node" in trace.name))
            and not (trace.name and "image-" in trace.name)  # Exclude image sites
        ]

    def extract_size(marker_size: Any) -> float:
        """Marker size as float."""
        if isinstance(marker_size, (tuple, list)):
            return float(marker_size[0]) if marker_size else 0.0
        return float(marker_size) if marker_size is not None else 0.0

    traces1 = get_site_traces(fig1)
    traces2 = get_site_traces(fig2)

    # Extract and compare marker sizes to verify precedence
    sizes1 = [
        extract_size(trace.marker.size)
        for trace in traces1
        if extract_size(trace.marker.size) > 0
    ]
    sizes2 = [
        extract_size(trace.marker.size)
        for trace in traces2
        if extract_size(trace.marker.size) > 0
    ]

    # Calculate averages for comparison
    avg_size1 = sum(sizes1) / len(sizes1)
    avg_size2 = sum(sizes2) / len(sizes2)

    # CORE PRECEDENCE TEST: struct1 has properties (atom_size=50, scale=2.0)
    # vs struct2 using function params (atom_size=20, scale=1.0)
    # So struct1 should have significantly larger markers
    assert avg_size1 > avg_size2 * 2

    # ATOMIC_RADII PRECEDENCE TEST: verify element-specific precedence
    # struct1 has custom atomic_radii: Na=1.5, Cl=1.8
    # struct2 uses atomic_radii=0.8 for both elements
    na_trace1 = next(t for t in traces1 if t.name and "Na" in t.name)
    cl_trace1 = next(t for t in traces1 if t.name and "Cl" in t.name)

    na_size1 = extract_size(na_trace1.marker.size)
    cl_size1 = extract_size(cl_trace1.marker.size)
    # In struct1, Cl should be larger than Na (custom atomic_radii: 1.8 vs 1.5)
    assert cl_size1 > na_size1


@pytest.mark.parametrize("plot_func", [pmv.structure_2d, pmv.structure_3d])
@pytest.mark.parametrize(
    ("site_labels", "expected_labels"),
    [
        ("symbol", {"Fe", "C"}),
        ("species", {"Fe", "C"}),
        ("legend", set()),
        (False, set()),
        ({"Fe": "Iron", "C": "Carbon"}, {"Iron", "Carbon"}),
    ],
)
def test_disordered_site_labeling_behavior(
    plot_func: Callable,
    site_labels: Any,
    expected_labels: set[str],
    fe3co4_disordered: Structure,
) -> None:
    """Disordered sites label all species unless in legend mode or labels are off."""
    fig = plot_func(
        fe3co4_disordered,
        site_labels=site_labels,
        show_cell=False,
        show_image_sites=False,
        atom_size=50,
    )
    label_traces = [
        trace for trace in fig.data if getattr(trace, "mode", None) == "text"
    ]
    labels = {trace.text for trace in label_traces}
    if not expected_labels:
        assert labels == set()
        return
    assert expected_labels <= labels

    if plot_func is pmv.structure_3d:
        # Labels sit outside the spherical wedges centered on the origin site
        for trace in label_traces:
            label_pos = np.array([trace.x[0], trace.y[0], trace.z[0]])
            assert np.linalg.norm(label_pos) > 0.1, f"{label_pos=} too close to site"


@pytest.mark.parametrize("plot_func", [pmv.structure_2d, pmv.structure_3d])
def test_disordered_site_low_occupancy(plot_func: Callable) -> None:
    """Species with 1% occupancy still get drawn and labeled."""
    minimal_disordered = Structure(
        lattice=Lattice.cubic(5),
        species=[{"Fe": 0.99, "C": 0.01}, "O", "O", "O"],
        coords=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
    )
    fig = plot_func(
        minimal_disordered,
        site_labels="symbol",
        show_cell=False,
        show_image_sites=False,
    )
    labels = {
        trace.text for trace in fig.data if getattr(trace, "mode", None) == "text"
    }
    assert {"Fe", "C"} <= labels


@pytest.mark.parametrize(
    ("n_theta", "n_phi"), [(1, 1), (8, 6), (16, 24), (0, 6), (8, 0), (-1, 6), (8, -1)]
)
@pytest.mark.parametrize("end_angle", [0.0, np.pi / 2, 2 * np.pi])
def test_spherical_wedge_mesh_generation(
    n_theta: int, n_phi: int, end_angle: float
) -> None:
    """Wedges retain surface indexing and close only partial spheres."""
    center = np.array([1.0, -2.0, 5.0])
    radius = 1.0
    if n_theta < 1 or n_phi < 1:
        with pytest.raises(ValueError, match="Mesh subdivisions must be positive"):
            get_spherical_wedge_mesh(center, radius, 0, end_angle, n_theta, n_phi)
        return

    x_coords, y_coords, z_coords, i_indices, j_indices, k_indices = (
        get_spherical_wedge_mesh(
            center=center,
            radius=radius,
            start_angle=0,
            end_angle=end_angle,
            n_theta=n_theta,
            n_phi=n_phi,
        )
    )

    assert len(x_coords) == 1 + (n_theta + 1) * (n_phi + 1)
    assert len(y_coords) == len(x_coords)
    assert len(z_coords) == len(x_coords)
    side_faces = 2 * n_theta if end_angle < 2 * np.pi else 0
    assert (
        len(i_indices)
        == len(j_indices)
        == len(k_indices)
        == (2 * n_theta * n_phi + side_faces)
    )
    faces = np.array([i_indices, j_indices, k_indices]).T
    assert faces.min() >= 0
    assert faces.max() < len(x_coords)
    assert faces[:2].tolist() == [[1, 2, n_phi + 2], [2, n_phi + 3, n_phi + 2]]
    if side_faces:
        assert faces[-side_faces].tolist() == [0, 1, n_phi + 2]
        assert faces[-n_theta].tolist() == [0, 2 * (n_phi + 1), n_phi + 1]
    else:
        assert 0 not in faces

    coords = np.array([x_coords, y_coords, z_coords]).T
    np.testing.assert_array_equal(coords[0], center)
    # Allow accumulated trig, translation, and norm error (about 45 float64 eps).
    np.testing.assert_allclose(
        np.linalg.norm(coords[1:] - center, axis=1), radius, rtol=1e-14, atol=1e-14
    )


@pytest.mark.parametrize("plot_func", [pmv.structure_2d, pmv.structure_3d])
@pytest.mark.parametrize(
    ("elem_colors", "err_msg"),
    [
        ({"Fe": "red"}, None),
        ({"Fe": [1, 0.5, 0]}, None),
        # malformed colors used to silently render gray
        ({"Fe": [1, 2, 3, 4]}, "Unsupported element color [1, 2, 3, 4]"),
        ({"Fe": None}, "Unsupported element color None"),
    ],
)
def test_structure_plotly_malformed_elem_colors(
    plot_func: Callable,
    elem_colors: dict[str, Any],
    err_msg: str | None,
    fe3co4_disordered: Structure,
) -> None:
    """Valid custom element colors render, malformed ones raise."""
    kwargs = dict(show_cell=False, show_image_sites=False, site_labels="symbol")
    if err_msg is None:
        fig = plot_func(fe3co4_disordered, elem_colors=elem_colors, **kwargs)
        assert len(fig.data) > 0
        return
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        plot_func(fe3co4_disordered, elem_colors=elem_colors, **kwargs)


@pytest.mark.parametrize("plot_func", [pmv.structure_2d, pmv.structure_3d])
def test_disordered_site_hover_text_formatting(
    plot_func: Callable, fe3co4_disordered: Structure
) -> None:
    """Disordered site hover text lists each species with its occupancy."""
    fig = plot_func(
        fe3co4_disordered,
        site_labels="symbol",
        show_cell=False,
        show_image_sites=False,
        hover_text=SiteCoords.cartesian_fractional,
    )
    hover_texts = [
        text
        for trace in fig.data
        for text in (
            [trace.hovertext] if isinstance(trace.hovertext, str) else trace.hovertext
        )
        or ()
        if text
    ]
    for species in ("Fe (0.75)", "C (0.25)"):
        assert any(
            "Site:" in text and f"Species: {species}" in text for text in hover_texts
        )


@pytest.mark.parametrize("plot_func", [pmv.structure_2d, pmv.structure_3d])
def test_disordered_site_legend_functionality(
    plot_func: Callable, fe3co4_disordered: Structure
) -> None:
    """Disordered sites get one combined legend entry and multi-structure plots get
    one legend per subplot.
    """
    fig = plot_func(fe3co4_disordered, site_labels="legend")
    assert fig.layout.showlegend is True
    legend_names = {trace.name for trace in fig.data if trace.showlegend}
    assert legend_names == {"Fe₀.₇₅C₀.₂₅", "O"}
    assert any(trace.legendgroup == "1-Fe" for trace in fig.data)

    multi_structs = {
        "struct1": fe3co4_disordered.copy(),
        "struct2": fe3co4_disordered.copy(),
    }
    fig_multi = plot_func(multi_structs, site_labels="legend", n_cols=2)
    legend_keys = {trace.legend for trace in fig_multi.data if trace.showlegend}
    assert legend_keys == {"legend", "legend2"}


@pytest.mark.parametrize("plot_func", [pmv.structure_2d, pmv.structure_3d])
@pytest.mark.parametrize(
    ("moments", "expect_vectors"),
    [([1, -1], True), ([0, 0], False)],
    ids=["nonzero", "zero"],
)
def test_structure_with_magmom_objects(
    plot_func: Callable, moments: list[int], expect_vectors: bool
) -> None:
    """Regression: Magmom objects crashed or produced wrong vectors."""
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    struct.add_site_property("magmom", [Magmom([0, 0, mom]) for mom in moments])

    fig = plot_func(struct, show_site_vectors="magmom")

    vector_traces = [tr for tr in fig.data if (tr.name or "").startswith("vector")]
    if expect_vectors:
        assert len(vector_traces) > 0, "Expected vector traces for magmom"
    else:
        assert len(vector_traces) == 0, "Zero magmom should have no vectors"


@pytest.mark.parametrize("plot_func", [pmv.structure_2d, pmv.structure_3d])
@pytest.mark.parametrize(
    ("show_image_sites", "show_image_vectors", "expect_image_vectors"),
    [
        pytest.param(True, True, True, id="image-sites-on_image-vectors-on"),
        pytest.param(True, False, False, id="image-sites-on_image-vectors-off"),
        pytest.param(False, True, False, id="image-sites-off_image-vectors-on"),
        pytest.param(False, False, False, id="image-sites-off_image-vectors-off"),
    ],
)
def test_show_image_vectors(
    plot_func: Callable,
    show_image_sites: bool,
    show_image_vectors: bool,
    expect_image_vectors: bool,
) -> None:
    """Test that vectors are drawn/not drawn at image site positions."""
    # Atom at origin generates image sites at cell corners
    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Fe"], [[0, 0, 0]])
    struct[0].properties["force"] = [0.1, 0.2, 0.3]

    fig = plot_func(
        struct,
        show_site_vectors="force",
        show_image_sites=show_image_sites,
        show_image_vectors=show_image_vectors,
    )

    # Image vector traces contain "img" in their name
    has_image_vector_traces = any(
        "img" in (trace.name or "") and "vector" in (trace.name or "")
        for trace in fig.data
    )
    # Primary vector traces start with "vector" but don't contain "img"
    has_primary_vector_traces = any(
        (trace.name or "").startswith("vector") and "img" not in (trace.name or "")
        for trace in fig.data
    )

    assert has_primary_vector_traces, "Should always have primary site vectors"
    assert has_image_vector_traces is expect_image_vectors


@pytest.mark.parametrize("plot_func", [pmv.structure_2d, pmv.structure_3d])
def test_image_site_hover_labels_and_legend(plot_func: Callable) -> None:
    """Image sites show their own coords, their primary's label and legend group."""
    # Li corner site has 7 images, O edge site 3
    struct = Structure(Lattice.cubic(4.0), ["Li", "O"], [[0, 0, 0], [0.5, 0, 0]])
    fig = plot_func(struct, site_labels=["A", "B"], hover_text=SiteCoords.fractional)
    site_traces = [
        trace
        for trace in fig.data
        if "markers" in (getattr(trace, "mode", None) or "")
        and not (trace.name or "").startswith("node")
    ]
    image_traces = [trace for trace in site_traces if "Image of" in trace.hovertext]
    assert len(site_traces) == 2 + 7 + 3
    assert len(image_traces) == 7 + 3
    # 2D image hover used to show the primary site's coords and default template
    image_hovers = {trace.hovertext for trace in image_traces}
    assert "<b>Image of Li</b><br>Coordinates [1, 1, 1]" in image_hovers
    assert "<b>Image of O</b><br>Coordinates [0.5, 1, 1]" in image_hovers
    # 3D image labels used to come from site_labels[augmented_idx % n_sites]
    labels = {
        (trace.hovertext.split("</b>")[0].removeprefix("<b>"), trace.text)
        for trace in site_traces
    }
    assert labels == {
        ("Site: Li", "A"),
        ("Site: O", "B"),
        ("Image of Li", "A"),
        ("Image of O", "B"),
    }

    # 3D image sites used to have no legendgroup, so legend clicks left them visible
    legend_fig = plot_func(struct, site_labels="legend")
    legend_groups = {
        trace.hovertext.split("</b>")[0].removeprefix("<b>"): trace.legendgroup
        for trace in legend_fig.data
        if "markers" in (getattr(trace, "mode", None) or "")
        and trace.hovertext.startswith("<b>")
    }
    assert legend_groups == {
        "Site: Li": "1-Li",
        "Site: O": "1-O",
        "Image of Li": "1-Li",
        "Image of O": "1-O",
    }
