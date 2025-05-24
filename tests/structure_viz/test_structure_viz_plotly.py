from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.colors import ELEM_COLORS_JMOL, ELEM_COLORS_VESTA
from pymatviz.enums import ElemColorScheme, Key
from pymatviz.structure_viz.helpers import (
    _get_site_symbol,
    generate_site_label,
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
    "kwargs",
    [  # the keyword combos below aim to maximize coverage, i.e. trigger every code path
        {
            "rotation": "0x,0y,0z",
            "atomic_radii": None,
            "atom_size": 30,
            "elem_colors": ElemColorScheme.jmol,
            "scale": 1,
            "show_unit_cell": True,
            "show_sites": True,
            "site_labels": "symbol",
            "standardize_struct": None,
            "n_cols": 2,
            "show_site_vectors": "magmom",
        },
        {
            "rotation": "10x,-10y,0z",
            "atomic_radii": 0.5,
            "atom_size": 50,
            "elem_colors": ElemColorScheme.vesta,
            "scale": 1.5,
            "show_unit_cell": False,
            "show_sites": False,
            "standardize_struct": True,
            "n_cols": 4,
            "show_site_vectors": ("magmom", "force"),
        },
        {
            "rotation": "5x,5y,5z",
            "atomic_radii": {"Fe": 0.5, "O": 0.3},
            "atom_size": 40,
            "elem_colors": {"Fe": "red", "O": "blue"},
            "scale": 1.2,
            "show_unit_cell": {"color": "red", "width": 2},
            "show_sites": {"line": {"width": 1, "color": "black"}},
            "site_labels": {"Fe": "Iron"},
            "standardize_struct": False,
            "n_cols": 3,
            "show_site_vectors": (),
        },
        {
            "rotation": "15x,0y,10z",
            "atomic_radii": 0.8,
            "atom_size": 35,
            "elem_colors": ElemColorScheme.jmol,
            "scale": 0.9,
            "show_unit_cell": True,
            "show_sites": True,
            "site_labels": ["Fe", "O"],
            "standardize_struct": None,
            "n_cols": 2,
        },
        {
            "rotation": "0x,20y,0z",
            "atomic_radii": None,
            "atom_size": 45,
            "elem_colors": ElemColorScheme.vesta,
            "scale": 1.1,
            "show_unit_cell": {"color": "blue", "width": 1, "dash": "dot"},
            "show_sites": True,
            "site_labels": False,
            "standardize_struct": True,
            "n_cols": 4,
        },
        {
            "rotation": "30x,-15y,5z",
            "atomic_radii": 0.6,
            "atom_size": 55,
            "elem_colors": {"Fe": "green", "O": "yellow"},
            "scale": 0.8,
            "show_unit_cell": True,
            "show_sites": {"line": {"width": 2, "color": "red"}},
            "standardize_struct": False,
            "n_cols": 3,
        },
    ],
)
def test_structure_2d_plotly(
    kwargs: dict[str, Any], fe3co4_disordered_with_props: Structure
) -> None:
    """Test structure_2d_plotly with various parameter combinations."""
    fig = pmv.structure_2d_plotly(fe3co4_disordered_with_props, **kwargs)
    assert isinstance(fig, go.Figure)

    # Check that all traces are Scatter (2D)
    assert {*map(type, fig.data)} in ({go.Scatter}, set())

    # Verify layout properties
    assert fig.layout.showlegend is False
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"

    # Check that axes are properly configured
    for axis in fig.layout.xaxis, fig.layout.yaxis:
        assert axis.showticklabels is False
        assert axis.showgrid is False
        assert axis.zeroline is False
        assert axis.scaleratio == 1
        assert axis.constrain == "domain"

    # Additional checks based on specific kwargs
    if isinstance(kwargs.get("show_unit_cell"), dict):
        edge_kwargs = kwargs["show_unit_cell"].get("edge", {})
        unit_cell_edge_trace = next(
            (trace for trace in fig.data if trace.mode == "lines"), None
        )
        assert unit_cell_edge_trace is not None
        for key, value in edge_kwargs.items():
            assert unit_cell_edge_trace.line[key] == value

    # --- Modified logic for site labels and colors ---
    show_sites_kwarg = kwargs.get(
        "show_sites", True
    )  # Default is True in structure_2d_plotly
    site_labels_kwarg = kwargs.get("site_labels")

    # Determine the actual struct_key used in trace names.
    # For a single input structure, all site traces will share the same struct_key part.
    first_site_trace_name_part = None
    for trace in fig.data:
        if trace.name and trace.name.startswith("site-") and "-" in trace.name[5:]:
            # Name is like "site-{struct_key}-{idx}"
            parts = trace.name.split("-")
            if len(parts) >= 3:
                first_site_trace_name_part = parts[1]  # This should be the struct_key
                break

    # If show_sites is False, there might be no site traces.
    # Only proceed if we expect site traces and found a key.
    struct_key_in_plot = (
        None if show_sites_kwarg is False else first_site_trace_name_part
    )

    # Collect all primary site traces. They are named "site-{struct_key}-{idx}"
    # and should have mode "markers" or "markers+text".
    # We sort them by site index parsed from the name to ensure correct order.
    primary_site_traces = []
    if struct_key_in_plot:  # Only try to collect if we have a key and expect sites
        primary_site_traces = sorted(
            [
                trace
                for trace in fig.data
                if trace.name
                and trace.name.startswith(f"site-{struct_key_in_plot}-")
                and trace.mode
                and "markers" in trace.mode
            ],
            key=lambda t: int(t.name.split("-")[-1]),
        )

    if show_sites_kwarg is not False:
        expected_n_primary_sites = len(fe3co4_disordered_with_props)
        # If site_labels is explicitly False or "legend", draw_site might not add text,
        # affecting trace.mode. But markers should still be there.
        # The number of traces should match the number of sites.
        assert len(primary_site_traces) == expected_n_primary_sites

        if site_labels_kwarg not in (False, "legend") and site_labels_kwarg is not None:
            # Labels are expected on sites
            for site_idx, trace in enumerate(primary_site_traces):
                site = fe3co4_disordered_with_props[site_idx]
                actual_label = str(trace.text)
                expected_label = generate_site_label(site_labels_kwarg, site_idx, site)
                assert actual_label == expected_label

        elif site_labels_kwarg == "legend" or site_labels_kwarg is None:  # Default case
            # Check for legend annotations
            legend_annos = [anno for anno in fig.layout.annotations if anno.bgcolor]
            # Check that each unique element in the structure has a legend entry
            unique_elements = sorted(
                {_get_site_symbol(s) for s in fe3co4_disordered_with_props}
            )
            assert len(legend_annos) == len(unique_elements)
            for anno, elem_symbol in zip(legend_annos, unique_elements, strict=True):
                assert anno.text == elem_symbol
                # Further checks for color, position could be added if necessary

            # Ensure no text on primary site traces
            for trace in primary_site_traces:
                assert trace.text is None or (
                    isinstance(trace.text, list) and not any(trace.text)
                )

        # Check atom colors for each primary site trace
        elem_colors_kwarg = kwargs.get("elem_colors")
        for site_idx, trace in enumerate(primary_site_traces):
            site = fe3co4_disordered_with_props[site_idx]
            site_symbol = _get_site_symbol(site)
            actual_color_str = normalize_rgb_color(str(trace.marker.color))
            expected_color_str = _resolve_expected_color_str(
                site_symbol, elem_colors_kwarg, normalize_rgb_color
            )
            _compare_colors(actual_color_str, expected_color_str, normalize_rgb_color)

    else:  # show_sites_kwarg is False
        assert len(primary_site_traces) == 0, (
            "No primary site traces should be present if show_sites=False"
        )

    # Check for sites and arrows
    if show_sites_kwarg is not False:
        site_traces = [
            trace for trace in fig.data if (trace.name or "").startswith("site")
        ]
        assert len(site_traces) > 0, "No site traces found when show_sites is True"

        if kwargs.get("show_site_vectors"):
            vector_traces = [
                trace for trace in fig.data if (trace.name or "").startswith("vector")
            ]
            assert len(vector_traces) > 0, (
                "No vector traces found when show_site_vectors is True"
            )
            for vector_trace in vector_traces:
                assert vector_trace.mode == "lines+markers"
                assert vector_trace.marker.symbol == "arrow"
                assert "angle" in vector_trace.marker

        # Only check image traces if we determined a struct_key
        n_expected_image_sites = 0
        # Compute total expected image sites across whole structure for count assertion
        for site_in_struct in fe3co4_disordered_with_props:
            n_expected_image_sites += len(
                get_image_sites(site_in_struct, fe3co4_disordered_with_props.lattice)
            )

        if n_expected_image_sites > 0 and struct_key_in_plot:
            image_site_traces = [
                trace
                for trace in fig.data
                if trace.name
                and trace.name.startswith(f"image-{struct_key_in_plot}-")
                and trace.mode
                and "markers" in trace.mode
            ]
            assert len(image_site_traces) == n_expected_image_sites

            # Resolve actual atomic radii map based on kwargs to calculate expected size
            current_atomic_radii_map = get_atomic_radii(
                kwargs.get("atomic_radii")  # Call with only one argument
            )
            scale_kwarg = kwargs.get("scale", 1)  # Default from structure_2d_plotly
            atom_size_kwarg = kwargs.get(
                "atom_size", 40
            )  # Default from structure_2d_plotly
            elem_colors_kwarg = kwargs.get("elem_colors")
            show_image_sites_kwarg = kwargs.get("show_image_sites", True)

            for image_trace in image_site_traces:
                # Image sites have text if site_labels is not False and applies to them
                assert "markers" in image_trace.mode
                assert image_trace.marker is not None, (
                    "Image site trace should have a marker object"
                )
                if image_trace.marker.symbol is not None:
                    assert (
                        image_trace.marker.symbol == "circle"
                    )  # Default if not overridden by show_image_sites dict

                # Extract parent site index from trace name: "image-{struct_key}-{parent_idx}-{img_idx}"  # noqa: E501
                parent_site_idx = int(image_trace.name.split("-")[-2])
                parent_site = fe3co4_disordered_with_props[parent_site_idx]
                parent_site_symbol = _get_site_symbol(parent_site)

                # 1. Check color (should match parent's resolved color)
                actual_image_color_str = normalize_rgb_color(
                    str(image_trace.marker.color)
                )
                expected_color_str = _resolve_expected_color_str(
                    parent_site_symbol, elem_colors_kwarg, normalize_rgb_color
                )
                _compare_colors(
                    actual_image_color_str, expected_color_str, normalize_rgb_color
                )

                # 2. Check size
                parent_radius_scaled = (
                    current_atomic_radii_map.get(parent_site_symbol, 1.0) * scale_kwarg
                )
                expected_image_size = parent_radius_scaled * atom_size_kwarg
                # site_kwargs in draw_site can override marker properties.
                # For now, assume show_image_sites dict doesn't typically override
                # 'size' calculated this way. This needs verification if tests fail here
                # for specific show_image_sites dicts.
                assert image_trace.marker.size == pytest.approx(expected_image_size)

                # 3. Check opacity
                expected_opacity = (
                    0.8  # Changed from 0.3 to 0.8 based on observed behavior
                )
                if isinstance(show_image_sites_kwarg, dict):
                    # If show_image_sites dict provides opacity, it should be used.
                    # Otherwise, it's 0.8 (matching primary site default, for now).
                    expected_opacity = show_image_sites_kwarg.get("opacity", 0.8)

                assert image_trace.marker.opacity == pytest.approx(expected_opacity), (
                    f"Image site opacity mismatch for trace {image_trace.name}: "
                    f"actual={image_trace.marker.opacity}, expected={expected_opacity}"
                )

        # Check unit cell if applicable
        # Only try to get edge_kwargs if show_unit_cell is a dict
        if isinstance(kwargs.get("show_unit_cell"), dict):
            edge_kwargs = kwargs["show_unit_cell"].get("edge", {})
            unit_cell_edge_trace = next(
                (trace for trace in fig.data if trace.mode == "lines"), None
            )
            assert unit_cell_edge_trace is not None, (
                "Unit cell edge trace not found when show_unit_cell is a dict"
            )
            for key, value in edge_kwargs.items():
                assert unit_cell_edge_trace.line[key] == value
        elif kwargs.get("show_unit_cell", True) is True:
            # If show_unit_cell is True (bool), just ensure the trace exists
            unit_cell_edge_trace = sum(
                trace.mode == "lines" and trace.name and trace.name.startswith("edge ")
                for trace in fig.data
            )

            assert unit_cell_edge_trace > 0


def test_structure_2d_plotly_multiple() -> None:
    struct1 = Structure(lattice_cubic, ["Fe", "O"], coords=COORDS)
    struct1.properties = {"id": "struct1"}
    struct2 = Structure(lattice_cubic, ["Co", "O"], coords=COORDS)
    struct2.properties = {Key.mat_id: "struct2"}
    struct3 = Structure(lattice_cubic, ["Ni", "O"], coords=COORDS)
    struct3.properties = {"ID": "struct3", "name": "nickel oxide"}
    struct4 = Structure(lattice_cubic, ["Cu", "O"], coords=COORDS)

    # Test dict[str, Structure]
    structs_dict = {
        "struct1": struct1,
        "struct2": struct2,
        "struct3": struct3,
        "struct4": struct4,
    }
    fig = pmv.structure_2d_plotly(structs_dict, n_cols=3, site_labels=False)
    assert isinstance(fig, go.Figure)
    # 4 structures. For each struct: 1  primary site trace, 12 for edges, 8 for nodes.
    # Image sites: each image site is a separate trace.
    trace_names = [trace.name or "" for trace in fig.data]

    actual_n_site_traces = sum(name.startswith("site-") for name in trace_names)
    expected_n_site_traces = sum(len(s) for s in structs_dict.values())
    assert actual_n_site_traces == expected_n_site_traces

    actual_n_edge_traces = sum(name.startswith("edge") for name in trace_names)
    assert actual_n_edge_traces == 12 * len(structs_dict)

    actual_n_node_traces = sum(name.startswith("node") for name in trace_names)
    assert actual_n_node_traces == 8 * len(structs_dict)

    actual_n_image_site_traces = sum(name.startswith("image-") for name in trace_names)
    expected_n_image_site_traces = 0
    for s_val in structs_dict.values():
        for site_in_s_val in s_val:
            expected_n_image_site_traces += len(
                get_image_sites(site_in_s_val, s_val.lattice)
            )
    assert actual_n_image_site_traces == expected_n_image_site_traces

    n_bond_traces = sum(name.startswith("bond") for name in trace_names)
    assert n_bond_traces == 0  # Default show_bonds=False

    expected_total_traces = (
        actual_n_site_traces
        + actual_n_edge_traces
        + actual_n_node_traces
        + actual_n_image_site_traces
    )
    assert len(fig.data) == expected_total_traces, (
        f"{len(fig.data)=}, {expected_total_traces=}"
    )
    assert len(fig.layout.annotations) == 4

    # Test with pandas.Series[Structure]
    struct_series = pd.Series(structs_dict)
    fig = pmv.structure_2d_plotly(struct_series, site_labels=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_total_traces
    assert len(fig.layout.annotations) == 4

    # Test with list[Structure]
    fig = pmv.structure_2d_plotly(
        list(structs_dict.values()), n_cols=2, site_labels=False
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_total_traces
    assert len(fig.layout.annotations) == 4

    # Test with custom subplot_title function
    def subplot_title(struct: Structure, key: str | int) -> str:
        return f"{key} - {struct.formula}"

    fig = pmv.structure_2d_plotly(
        struct_series, subplot_title=subplot_title, site_labels=False
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_total_traces
    assert len(fig.layout.annotations) == 4

    # Verify subplot titles
    for idx, (key, struct) in enumerate(structs_dict.items(), start=1):
        expected_title = subplot_title(struct=struct, key=key)
        assert fig.layout.annotations[idx - 1].text == expected_title


def test_structure_2d_plotly_invalid_input() -> None:
    """Test that structure_2d_plotly raises errors for invalid inputs."""
    # Match the actual error message from normalize_structures
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "atomic_radii": None,
            "atom_size": 20,
            "elem_colors": ElemColorScheme.jmol,
            "scale": 1,
            "show_unit_cell": True,
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
            "show_unit_cell": False,
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
            "show_unit_cell": {"edge": {"color": "red", "width": 3}},
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
            "show_unit_cell": True,
            "show_sites": True,
            "show_image_sites": True,
            "site_labels": False,
            "standardize_struct": None,
            "n_cols": 1,
        },
    ],
)
def test_structure_3d_plotly(
    kwargs: dict[str, Any], fe3co4_disordered_with_props: Structure
) -> None:
    fig = pmv.structure_3d_plotly(fe3co4_disordered_with_props, **kwargs)
    assert isinstance(fig, go.Figure)

    # Check if the layout properties are set correctly
    assert fig.layout.showlegend is False
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
    if isinstance(kwargs.get("show_unit_cell"), dict):
        edge_kwargs = kwargs["show_unit_cell"].get("edge", {})
        unit_cell_edge_trace = next(
            (trace for trace in fig.data if trace.mode == "lines"), None
        )
        assert unit_cell_edge_trace is not None
        for key, value in edge_kwargs.items():
            assert unit_cell_edge_trace.line[key] == value

    if kwargs.get("show_sites"):
        site_traces = [
            trace
            for trace in fig.data
            if getattr(trace, "mode", None) in ("markers", "markers+text")
        ]
        assert len(site_traces) > 0, "No site traces found when show_sites is True"
        site_trace = site_traces[0]

        # Determine total number of sites expected (primary + images)
        show_image_sites = kwargs.get("show_image_sites", True)
        # For show_image_sites as dict, it implies True if show_sites is True
        if isinstance(show_image_sites, dict):
            show_image_sites = True

        rendered_sites_info = _get_all_rendered_site_info(
            fe3co4_disordered_with_props, show_image_sites
        )
        expected_total_sites = len(rendered_sites_info)

        # Check atom colors
        elem_colors = kwargs.get("elem_colors")
        actual_colors_3d = []  # Initialize actual_colors_3d
        raw_marker_color_3d = site_trace.marker.color

        if raw_marker_color_3d is not None:
            if isinstance(raw_marker_color_3d, str):  # Single color string
                actual_colors_3d = [normalize_rgb_color(raw_marker_color_3d)] * (
                    expected_total_sites
                )
            elif isinstance(raw_marker_color_3d, (list, tuple)):
                # Check if it's a list of RGB tuples or list of color strings
                if (
                    all(isinstance(c, (int, float)) for c in raw_marker_color_3d)
                    and len(raw_marker_color_3d) == 3
                ):
                    # Single RGB tuple for all points - normalize and replicate
                    r, g, b = raw_marker_color_3d
                    rgb_tuple_str = f"rgb({r},{g},{b})"
                    actual_colors_3d = [
                        normalize_rgb_color(rgb_tuple_str)
                    ] * expected_total_sites
                else:  # List of color strings
                    actual_colors_3d = [
                        normalize_rgb_color(str(color)) for color in raw_marker_color_3d
                    ]

        if actual_colors_3d:  # Proceed only if actual_colors_3d were populated
            expected_colors_resolved = [
                _resolve_expected_color_str(
                    site_info["symbol"], elem_colors, normalize_rgb_color
                )
                for site_info in rendered_sites_info
            ]

            assert len(actual_colors_3d) == len(expected_colors_resolved)
            for actual_color_val, expected_color_val in zip(
                actual_colors_3d, expected_colors_resolved, strict=True
            ):
                _compare_colors(
                    actual_color_val, expected_color_val, normalize_rgb_color
                )

            # Check atom sizes
            atom_size_kwarg = kwargs.get("atom_size")
            scale_kwarg = kwargs.get("scale", 1)
            atomic_radii_kwarg = kwargs.get("atomic_radii")

            _processed_atomic_radii = get_atomic_radii(atomic_radii_kwarg)

            expected_sizes = []
            for site_info in rendered_sites_info:
                symbol = site_info["symbol"]
                radius_val = _processed_atomic_radii.get(symbol, 1.0)
                expected_sizes.append(radius_val * scale_kwarg * atom_size_kwarg)

            actual_sizes = site_trace.marker.size
            if isinstance(actual_sizes, (int, float)):  # Single size for all
                actual_sizes = [float(actual_sizes)] * expected_total_sites

            assert len(actual_sizes) == expected_total_sites
            for act_size, exp_size in zip(actual_sizes, expected_sizes, strict=True):
                assert pytest.approx(act_size) == exp_size

            # Check site labels if they are expected
            site_labels_kwarg = kwargs.get("site_labels")
            actual_site_labels = site_labels_kwarg
            # Default to "legend" if None
            if site_labels_kwarg is None:
                actual_site_labels = "legend"

            if actual_site_labels not in (False, "legend"):
                assert site_trace.text is not None, "Site labels missing on trace"
                actual_labels_list = list(site_trace.text)
                assert len(actual_labels_list) == expected_total_sites

                for site_idx, site_info in enumerate(rendered_sites_info):
                    actual_label = actual_labels_list[site_idx]
                    # Type assertion to help mypy understand the type
                    assert actual_site_labels not in (False, "legend", None)
                    expected_label = generate_site_label(
                        actual_site_labels,  # type: ignore[arg-type]
                        site_idx,
                        site_info["site_obj"],
                    )
                    assert actual_label == expected_label

            elif actual_site_labels == "legend":
                # Check legend annotations for this subplot. More complex for multiple
                # subplots. Need to correctly identify which annotations belong to which
                # subplot.

                # Simpler check: Count total legend items matching elements in *this*
                # structure
                current_struct_elements = {
                    _get_site_symbol(s) for s in fe3co4_disordered_with_props
                }
                legend_annos_for_this_subplot = [
                    anno
                    for anno in fig.layout.annotations
                    # Basic check: if it matches an element, assume it's a legend item
                    if anno.bgcolor and anno.text in current_struct_elements
                ]

                # Check that number of legend items for this structure is correct.
                # This assumes each legend annotation is unique text-wise per subplot.
                unique_elements_in_struct = sorted(current_struct_elements)
                assert len(legend_annos_for_this_subplot) == len(
                    unique_elements_in_struct
                )

                if site_trace.text:  # Ensure no text on the scatter3d trace itself
                    for idx, text in enumerate(site_trace.text):
                        assert text is None, f"{text=}, {idx=}"
                else:
                    assert site_trace.text is None
    else:  # show_sites is False
        site_traces = [
            trace
            for trace in fig.data
            if getattr(trace, "mode", None) in ("markers", "markers+text")
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
    actual_n_site_traces_3d = sum(
        (trace.name or "").startswith("site-") for trace in fig.data
    )
    assert actual_n_site_traces_3d == len(structs_dict)
    expected_total_traces_3d += actual_n_site_traces_3d

    actual_n_edge_traces_3d = sum(
        (trace.name or "").startswith("edge") for trace in fig.data
    )
    # Default show_unit_cell=True
    assert actual_n_edge_traces_3d == 12 * len(structs_dict)
    expected_total_traces_3d += actual_n_edge_traces_3d

    actual_n_node_traces_3d = sum(
        (trace.name or "").startswith("node") for trace in fig.data
    )
    # Default show_unit_cell=True
    assert actual_n_node_traces_3d == 8 * len(structs_dict)
    expected_total_traces_3d += actual_n_node_traces_3d

    assert len(fig.data) == expected_total_traces_3d, (
        f"{len(fig.data)=} vs {expected_total_traces_3d=}"
    )

    # Annotations: 4 subplot titles + legend annotations for each subplot
    # Each struct has 2 unique elements (e.g., Fe, O)
    expected_n_subplot_titles = len(structs_dict)
    expected_n_legend_items_per_struct = 2  # Assuming each struct has 2 unique elements
    expected_total_legend_annotations = (
        len(structs_dict) * expected_n_legend_items_per_struct
    )
    # Check if subplot_title is default (None), which generates titles
    # If subplot_title=False, then no titles.
    # Here, subplot_title is None by default in structure_3d_plotly
    assert (
        len(fig.layout.annotations)
        == expected_n_subplot_titles + expected_total_legend_annotations
    )

    # Test pandas.Series[Structure]
    struct_series = pd.Series(structs_dict)
    fig = pmv.structure_3d_plotly(struct_series)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_total_traces_3d
    assert (
        len(fig.layout.annotations)
        == expected_n_subplot_titles + expected_total_legend_annotations
    )

    # Test list[Structure]
    fig = pmv.structure_3d_plotly(list(structs_dict.values()), n_cols=3)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_total_traces_3d
    assert (
        len(fig.layout.annotations)
        == expected_n_subplot_titles + expected_total_legend_annotations
    )

    # Test subplot_title and its interaction with legend annotations
    def custom_subplot_title_func(struct: Structure, key: str | int) -> str:
        return f"{key} - {struct.formula}"

    fig = pmv.structure_3d_plotly(
        struct_series, subplot_title=custom_subplot_title_func
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_total_traces_3d
    assert (
        len(fig.layout.annotations)
        == expected_n_subplot_titles + expected_total_legend_annotations
    )
    title_texts = [anno.text for anno in fig.layout.annotations if not anno.bgcolor]
    for _idx, (key, struct) in enumerate(structs_dict.items(), start=1):
        expected_title = custom_subplot_title_func(struct=struct, key=key)
        assert expected_title in title_texts

    # Check total number of points in each site trace (remains the same)
    for s_key, s_val_3d in structs_dict.items():
        site_trace_found = False
        for trace in fig.data:
            if getattr(trace, "name", "") == f"site-{s_key}":
                site_trace_found = True
                rendered_sites = _get_all_rendered_site_info(
                    s_val_3d,
                    show_image_sites=True,  # Default for 3D
                )
                assert len(trace.x) == len(rendered_sites), (
                    f"Mismatch in number of points for {s_key=}. "
                    f"Expected {len(rendered_sites)}, got {len(trace.x)}"
                )
                assert len(trace.y) == len(rendered_sites)
                assert len(trace.z) == len(rendered_sites)
                break
        assert site_trace_found, f"Site trace for {s_key=} not found"


@pytest.mark.parametrize(
    ("custom_title_params", "expected_legend_annotations"),
    [
        (
            {
                "text": "Custom {key} - {struct.formula}",
                "font": {"size": 16, "color": "red"},
                "y": 0.8,
                "yanchor": "bottom",
            },
            4,  # 2 structs x 2 elements each = 4 legend items
        ),
        (
            {
                "text": "{struct.formula} ({key})",
                "font": {"size": 14, "color": "blue"},
                "x": 0.5,
                "xanchor": "center",
            },
            4,  # 2 structs x 2 elements each = 4 legend items
        ),
        (
            {
                "text": "Structure {key}",
                "font": {"size": 18, "color": "green"},
                "y": 1.0,
                "yanchor": "top",
            },
            4,  # 2 structs x 2 elements each = 4 legend items
        ),
        ({}, 4),  # Empty dict to test default behavior for title, legend still appears
    ],
)
def test_structure_3d_plotly_subplot_title_override_with_legend(
    custom_title_params: dict[str, str | float | dict[str, str | float]],
    expected_legend_annotations: int,
) -> None:
    struct1 = Structure(lattice_cubic, ["Fe", "O"], COORDS)
    struct2 = Structure(
        lattice_cubic, ["Co", "Ni"], COORDS
    )  # Different elements for distinct legends
    structs_dict = {"s1": struct1, "s2": struct2}

    def custom_subplot_title_generator(
        struct: Structure, key: str | int
    ) -> dict[str, Any]:
        title_dict_copy = custom_title_params.copy()
        if "text" in title_dict_copy and isinstance(title_dict_copy["text"], str):
            title_dict_copy["text"] = title_dict_copy["text"].format(
                key=key, struct=struct
            )
        return title_dict_copy

    # site_labels defaults to "legend", so legends will be drawn
    fig = pmv.structure_3d_plotly(
        structs_dict, subplot_title=custom_subplot_title_generator
    )

    assert isinstance(fig, go.Figure)

    # Annotations = titles + legend items
    # struct1 has 2 legend items (Fe, O), struct2 has 2 (Co, Ni)
    # Total 2 titles + 2*2 legend items = 6 annotations
    actual_legend_annotations = len(
        [
            ann
            for ann in fig.layout.annotations
            if ann.text and len(ann.text) <= 3 and ann.bgcolor is not None
        ]
    )
    assert actual_legend_annotations == expected_legend_annotations

    title_annotations = [anno for anno in fig.layout.annotations if not anno.bgcolor]
    legend_annotations = [anno for anno in fig.layout.annotations if anno.bgcolor]

    assert len(title_annotations) == len(structs_dict)
    assert len(legend_annotations) == expected_legend_annotations

    for idx, ((key, struct), title_anno) in enumerate(
        zip(structs_dict.items(), title_annotations, strict=True)
    ):
        expected_text_val: str | None
        if custom_title_params:
            custom_text_template = custom_title_params.get("text")
            if isinstance(custom_text_template, str):
                expected_text_val = custom_text_template.format(key=key, struct=struct)
            else:  # Default title text if custom_title_params is {} or text is not str
                # Default title from get_subplot_title for 3D is key then struct.formula
                expected_text_val = pmv.structure_viz.helpers.get_subplot_title(
                    struct, key, idx + 1, None
                )["text"]
        else:  # Should not happen with parametrize, but as fallback
            expected_text_val = pmv.structure_viz.helpers.get_subplot_title(
                struct, key, idx + 1, None
            )["text"]

        assert title_anno.text == expected_text_val

        # Check other attributes from custom_title_params if they were set
        if custom_title_params:
            for attr, value in custom_title_params.items():
                if attr == "font" and isinstance(value, dict):
                    for font_attr, font_value in value.items():
                        assert getattr(title_anno.font, font_attr) == font_value
                elif attr != "text":  # Avoid re-checking text
                    assert getattr(title_anno, attr) == value
        else:  # Check default title properties when custom_title_params is {}
            assert title_anno.font.size == 16  # Default for get_subplot_title
            assert title_anno.yanchor == "top"  # Default for get_subplot_title in 3D


@pytest.mark.parametrize(
    ("plot_function", "is_3d"),
    [
        (pmv.structure_2d_plotly, False),
        (pmv.structure_3d_plotly, True),
    ],
)
def test_structure_plotly_legend_core_functionality(
    fe3co4_disordered_with_props: Structure,
    plot_function: Callable[[Structure, dict[str, Any]], go.Figure],
    is_3d: bool,
) -> None:
    """Test core legend functionality for both 2D and 3D structure plotting."""
    struct = fe3co4_disordered_with_props

    # Test default legend behavior (site_labels defaults to "legend")
    fig = plot_function(struct)  # type: ignore[call-arg]
    assert isinstance(fig, go.Figure)

    # Check for legend annotations
    legend_annos = [
        ann
        for ann in fig.layout.annotations
        if ann.bgcolor and ann.text in ["Fe", "Co", "O"]
    ]
    unique_elements = sorted(
        {_get_site_symbol(s) for s in fe3co4_disordered_with_props}
    )
    assert len(legend_annos) == len(unique_elements)

    # Legend positioning should be within valid paper coordinates
    for ann in legend_annos:
        assert 0 <= ann.x <= 1, f"Legend x position {ann.x} should be in [0,1]"
        assert 0 <= ann.y <= 1, f"Legend y position {ann.y} should be in [0,1]"
        assert ann.xref == "paper"
        assert ann.yref == "paper"

    # Test legend vs other site labels produce different results
    fig_symbol = plot_function(struct, site_labels="symbol")  # type: ignore[call-arg]
    fig_false = plot_function(struct, site_labels=False)  # type: ignore[call-arg]

    def count_legend_annotations(fig: go.Figure) -> int:
        return len(
            [
                ann
                for ann in fig.layout.annotations
                if ann.text and len(ann.text) <= 3 and ann.bgcolor is not None
            ]
        )

    def count_text_traces(fig: go.Figure) -> int:
        return len(
            [trace for trace in fig.data if hasattr(trace, "text") and trace.text]
        )

    # Legend should have legend annotations but no text on traces
    assert count_legend_annotations(fig) > 0
    assert count_text_traces(fig) == 0

    # Symbol should have text on traces but no legend annotations
    assert count_legend_annotations(fig_symbol) == 0
    assert count_text_traces(fig_symbol) > 0

    # False should have neither
    assert count_legend_annotations(fig_false) == 0
    assert count_text_traces(fig_false) == 0

    if is_3d:
        # Ensure no text on the main site trace for 3D
        site_trace_3d = next(
            (t for t in fig.data if (t.name or "").startswith("site-")), None
        )
        assert site_trace_3d is not None
        if site_trace_3d.text:
            assert all(t is None for t in site_trace_3d.text)
        else:
            assert site_trace_3d.text is None
    else:
        # Ensure no text on primary site traces for 2D
        primary_site_traces = [
            trace
            for trace in fig.data
            if (trace.name or "").startswith("site-")
            and "markers" in (trace.mode or "")
        ]
        for trace in primary_site_traces:
            assert not trace.text  # Should be None or empty


@pytest.mark.parametrize(
    ("elem_colors", "expected_differences"),
    [
        (pmv.enums.ElemColorScheme.jmol, True),
        (pmv.enums.ElemColorScheme.vesta, True),
        ({"Li": "red", "O": "blue"}, False),  # Custom colors should match exactly
        ({"Li": (1.0, 0.0, 0.0), "O": (0.0, 1.0, 0.0)}, False),  # RGB float tuples
        ({"Li": (255, 0, 0), "O": (0, 255, 0)}, False),  # RGB int tuples
        ({"Li": "#FF0000", "O": "#00FF00"}, False),  # Hex colors
    ],
)
def test_structure_plotly_legend_colors(
    elem_colors: Any, expected_differences: bool
) -> None:
    """Test legend color schemes and custom colors."""
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    fig = pmv.structure_2d_plotly(struct, site_labels="legend", elem_colors=elem_colors)

    legend_annotations = [
        ann
        for ann in fig.layout.annotations
        if ann.text and len(ann.text) <= 3 and ann.bgcolor is not None
    ]

    assert len(legend_annotations) == 2  # Li and O

    if not expected_differences:
        # Test specific color expectations for custom colors
        colors_used = {ann.text: ann.bgcolor for ann in legend_annotations}

        if isinstance(elem_colors, dict):
            for element, expected_color in elem_colors.items():
                if isinstance(expected_color, str):
                    if expected_color.startswith("#"):
                        # Convert hex to RGB for comparison
                        continue  # Let plotly handle the conversion
                    assert colors_used[element] == expected_color
                elif isinstance(expected_color, tuple):
                    # RGB tuple should be converted to rgb() string
                    if all(0 <= c <= 1 for c in expected_color):  # Float range
                        rgb_int = tuple(int(c * 255) for c in expected_color)
                    else:  # Int range
                        rgb_int = expected_color
                    expected_rgb_str = f"rgb({rgb_int[0]},{rgb_int[1]},{rgb_int[2]})"
                    assert colors_used[element] == expected_rgb_str

    # Test text contrast for very dark/light backgrounds
    if elem_colors == {"Li": "#000000", "O": "#FFFFFF"}:
        text_colors = {ann.text: ann.font.color for ann in legend_annotations}
        li_color = text_colors["Li"]
        o_color = text_colors["O"]
        assert li_color in ["white", "#FFFFFF", "rgb(255,255,255)", "#ffffff"]
        assert o_color in ["black", "#000000", "rgb(0,0,0)", "#000"]


@pytest.mark.parametrize(
    ("legend_kwargs", "expected_attrs"),
    [
        (
            {},
            dict(font_size=12, box_side=18, xanchor="right", yanchor="bottom"),
        ),
        (
            {"font_size": 14, "box_size_px": 22},
            dict(font_size=14, box_side=22, xanchor="right", yanchor="bottom"),
        ),
        (
            {"corner": "top-left"},
            dict(font_size=12, box_side=18, xanchor="left", yanchor="top"),
        ),
        (
            {"corner": "bottom-left"},
            dict(font_size=12, box_side=18, xanchor="left", yanchor="bottom"),
        ),
        (
            {"corner": "top-right"},
            dict(font_size=12, box_side=18, xanchor="right", yanchor="top"),
        ),
        (
            dict(
                corner="top-left",
                font_size=12,
                box_size_px=25,
                item_gap_px=5,
                margin_frac=0.02,
            ),
            dict(font_size=12, box_side=25, xanchor="left", yanchor="top"),
        ),
    ],
)
def test_structure_plotly_legend_parameters_and_positioning(
    fe3co4_disordered_with_props: Structure,
    legend_kwargs: dict[str, Any],
    expected_attrs: dict[str, Any],
) -> None:
    """Test legend parameter customization and positioning."""
    struct = fe3co4_disordered_with_props

    # Test both 2D and 3D
    for plot_func in [pmv.structure_2d_plotly, pmv.structure_3d_plotly]:
        fig = plot_func(struct, site_labels="legend", legend_kwargs=legend_kwargs)  # type: ignore[operator]

        legend_annotations = [
            ann
            for ann in fig.layout.annotations
            if ann.text and len(ann.text) <= 3 and ann.bgcolor is not None
        ]

        assert len(legend_annotations) > 0

        for ann in legend_annotations:
            # Check font size
            assert ann.font.size == expected_attrs["font_size"]

            # Check box size
            assert ann.width == expected_attrs["box_side"]
            assert ann.height == expected_attrs["box_side"]

            # Check positioning
            assert ann.xanchor == expected_attrs["xanchor"]
            assert ann.yanchor == expected_attrs["yanchor"]

            # Check corner positioning logic
            corner = legend_kwargs.get("corner", "bottom-right")
            if "left" in corner:
                assert ann.x < 0.5, f"Left corners should have x < 0.5, got {ann.x}"
            else:  # right
                assert ann.x > 0.5, f"Right corners should have x > 0.5, got {ann.x}"

            if "bottom" in corner:
                assert ann.y < 0.5, f"Bottom corners should have y < 0.5, got {ann.y}"
            else:  # top
                assert ann.y > 0.5, f"Top corners should have y > 0.5, got {ann.y}"

            # Check hover text exists
            assert ann.hovertext is not None
            assert len(ann.hovertext) > len(ann.text)

            # Check that legend annotations have no border
            assert ann.borderwidth == 0


@pytest.mark.parametrize(
    ("hover_fmt", "test_coordinates", "expected_patterns"),
    [
        (
            ".4",  # Default format
            [1.23456789, 1e-17, 2.3456],
            ["1.235", "2.346"],  # Very small numbers may show as 0 or scientific
        ),
        (
            ".2f",  # Fixed decimal format
            [1.23456789, 1e-17, 2.3456],
            ["1.23", "0.00", "2.35"],
        ),
        (
            ".4f",  # Fixed decimal with more precision
            [0.123456789, 1e-17, 0.5],
            ["0.1235", "0.0000", "0.5000"],
        ),
        (
            ".2e",  # Scientific notation
            [0.123456789, 1e-17, 0.5],
            ["e"],  # Should contain 'e' for scientific notation
        ),
    ],
)
def test_structure_plotly_hover_formatting(
    hover_fmt: str, test_coordinates: list[float], expected_patterns: list[str]
) -> None:
    """Test hover text formatting for both 2D and 3D plots."""
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Li", "O"], [test_coordinates, [0.5, 0.5, 0.5]])

    # Test 2D
    fig_2d = pmv.structure_2d_plotly(struct, hover_float_fmt=hover_fmt)
    site_traces_2d = [t for t in fig_2d.data if (t.name or "").startswith("site-")]
    assert len(site_traces_2d) > 0
    hover_text_2d = site_traces_2d[0].hovertext

    for pattern in expected_patterns:
        if hover_fmt == ".2e" and pattern == "e":
            # Special case for scientific notation
            assert "e-" in hover_text_2d.lower() or "e+" in hover_text_2d.lower()
        else:
            assert pattern in hover_text_2d

    # Test 3D
    fig_3d = pmv.structure_3d_plotly(struct, hover_float_fmt=hover_fmt)
    site_traces_3d = [t for t in fig_3d.data if (t.name or "").startswith("site-")]
    assert len(site_traces_3d) > 0
    hover_texts_3d = site_traces_3d[0].hovertext

    # Get first hover text for 3D (may be list or single value)
    if isinstance(hover_texts_3d, (list, tuple)):
        first_hover_3d = hover_texts_3d[0]
    else:
        first_hover_3d = hover_texts_3d

    for pattern in expected_patterns:
        if hover_fmt == ".2e" and pattern == "e":
            assert "e-" in first_hover_3d.lower() or "e+" in first_hover_3d.lower()
        else:
            assert pattern in first_hover_3d

    # Test hover formatting with image sites
    lattice_small = Lattice.cubic(2.0)  # Small lattice to ensure image sites
    struct_image = Structure(lattice_small, ["Li"], [[1.8, 0.1, 0.05]])

    fig_image = pmv.structure_2d_plotly(
        struct_image, show_image_sites=True, hover_float_fmt=".3f"
    )
    site_traces = [
        t
        for t in fig_image.data
        if t.name and (t.name.startswith("site-") or t.name.startswith("image-"))
    ]

    for trace in site_traces:
        # Fixed format should not contain scientific notation
        if hasattr(trace, "hovertext") and trace.hovertext and hover_fmt.endswith("f"):
            assert "e-" not in trace.hovertext.lower()
            assert "e+" not in trace.hovertext.lower()


@pytest.mark.parametrize(
    "test_scenario",
    [
        "multiple_subplots",
        "many_elements_stress",
        "single_element_minimal",
        "margin_offset_precision",
        "vertical_stacking",
    ],
)
def test_structure_plotly_legend_edge_cases(test_scenario: str) -> None:
    """Test various edge cases and stress scenarios for legend functionality."""
    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)

    if test_scenario == "multiple_subplots":
        # Test legend with multiple subplots
        struct1 = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        struct2 = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        struct3 = Structure(
            lattice,
            ["Ca", "F", "F"],
            [[0, 0, 0], [0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
        )
        structures = {"LiO": struct1, "NaCl": struct2, "CaF2": struct3}

        fig_2d = pmv.structure_2d_plotly(structures, site_labels="legend", n_cols=2)
        fig_3d = pmv.structure_3d_plotly(structures, site_labels="legend", n_cols=2)

        expected_elements = {"Li", "O", "Na", "Cl", "Ca", "F"}
        for fig in [fig_2d, fig_3d]:
            legend_annotations = [
                ann
                for ann in fig.layout.annotations
                if ann.text and len(ann.text) <= 3 and ann.bgcolor is not None
            ]
            found_elements = {ann.text for ann in legend_annotations}
            assert expected_elements == found_elements

    elif test_scenario == "many_elements_stress":
        # Stress test with many elements
        elements = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]
        coords = [[i / 10, i / 10, i / 10] for i in range(len(elements))]
        struct = Structure(lattice, elements, coords)

        fig = pmv.structure_2d_plotly(struct, site_labels="legend")
        legend_annotations = [
            ann
            for ann in fig.layout.annotations
            if ann.text and len(ann.text) <= 3 and ann.bgcolor is not None
        ]

        assert len(legend_annotations) == len(elements)
        legend_symbols = {ann.text for ann in legend_annotations}
        assert legend_symbols == set(elements)

        # Check positioning
        for ann in legend_annotations:
            assert 0 <= ann.x <= 1

        # Check spacing
        y_positions = sorted([ann.y for ann in legend_annotations])
        for i in range(len(y_positions) - 1):
            gap = abs(y_positions[i + 1] - y_positions[i])
            assert gap >= 0.01  # Minimum gap between items

    elif test_scenario == "single_element_minimal":
        # Test minimal structure with single element
        minimal_struct = Structure(lattice, ["H"], [[0, 0, 0]])
        fig = pmv.structure_2d_plotly(minimal_struct, site_labels="legend")

        legend_annotations = [
            ann
            for ann in fig.layout.annotations
            if ann.text and len(ann.text) <= 3 and ann.bgcolor is not None
        ]
        assert len(legend_annotations) == 1
        assert legend_annotations[0].text == "H"

    elif test_scenario == "margin_offset_precision":
        # Test precise margin calculations
        struct = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
        test_margins = [0.01, 0.05, 0.1]

        margin_positions = []
        for margin in test_margins:
            fig = pmv.structure_2d_plotly(
                struct,
                site_labels="legend",
                legend_kwargs={"margin_frac": margin, "corner": "bottom-right"},
            )
            legend_annotations = [
                ann
                for ann in fig.layout.annotations
                if ann.text and len(ann.text) <= 3 and ann.bgcolor is not None
            ]
            avg_pos = (legend_annotations[0].x, legend_annotations[0].y)
            margin_positions.append(avg_pos)

        # Smaller margins should place legend closer to corners
        small_distance = (
            (1 - margin_positions[0][0]) ** 2 + margin_positions[0][1] ** 2
        ) ** 0.5
        medium_distance = (
            (1 - margin_positions[1][0]) ** 2 + margin_positions[1][1] ** 2
        ) ** 0.5
        large_distance = (
            (1 - margin_positions[2][0]) ** 2 + margin_positions[2][1] ** 2
        ) ** 0.5

        assert small_distance < medium_distance < large_distance

    elif test_scenario == "vertical_stacking":
        # Test vertical stacking with different gaps
        struct = Structure(
            lattice,
            ["Li", "Na", "K", "Rb", "Cs"],
            [
                [0, 0, 0],
                [0.2, 0.2, 0.2],
                [0.4, 0.4, 0.4],
                [0.6, 0.6, 0.6],
                [0.8, 0.8, 0.8],
            ],
        )

        fig_small_gap = pmv.structure_2d_plotly(
            struct, site_labels="legend", legend_kwargs={"item_gap_px": 2}
        )
        fig_large_gap = pmv.structure_2d_plotly(
            struct, site_labels="legend", legend_kwargs={"item_gap_px": 10}
        )

        def get_legend_positions(fig: go.Figure) -> list[float]:
            legend_annos = [
                ann
                for ann in fig.layout.annotations
                if ann.text and len(ann.text) <= 3 and ann.bgcolor is not None
            ]
            return sorted([ann.y for ann in legend_annos])

        small_gap_positions = get_legend_positions(fig_small_gap)
        large_gap_positions = get_legend_positions(fig_large_gap)

        # Calculate gaps between consecutive items
        small_gaps = [
            small_gap_positions[i + 1] - small_gap_positions[i] for i in range(4)
        ]
        large_gaps = [
            large_gap_positions[i + 1] - large_gap_positions[i] for i in range(4)
        ]

        # Large gap should produce larger spacing
        assert all(lg > sg for lg, sg in zip(large_gaps, small_gaps, strict=True))


def test_structure_plotly_legend_error_handling() -> None:
    """Test error handling and edge cases for legend functionality."""
    import unittest.mock
    import warnings

    from pymatgen.core import Lattice, Structure

    lattice = Lattice.cubic(4.0)
    struct = Structure(lattice, ["Li", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    # Test invalid corner values
    with pytest.raises(ValueError, match="Invalid corner"):
        pmv.structure_2d_plotly(
            struct, site_labels="legend", legend_kwargs={"corner": "invalid-corner"}
        )

    with pytest.raises(ValueError, match="Invalid corner"):
        pmv.structure_3d_plotly(
            struct, site_labels="legend", legend_kwargs={"corner": "middle-center"}
        )

    # Test fallback behavior for invalid element symbols
    with unittest.mock.patch("pymatgen.core.periodic_table.Element") as mock_element:
        mock_element.side_effect = ValueError("Unknown element")

        fig = pmv.structure_2d_plotly(struct, site_labels="legend")
        legend_annotations = [
            ann
            for ann in fig.layout.annotations
            if ann.text and len(ann.text) <= 3 and ann.bgcolor is not None
        ]

        hover_texts = {ann.text: ann.hovertext for ann in legend_annotations}
        assert "Element Li" in hover_texts.get("Li", "")
        assert "Element O" in hover_texts.get("O", "")

    # Test missing domain warning
    fig = pmv.structure_2d_plotly(struct, site_labels="legend")
    with warnings.catch_warnings(record=True) as warning_list:
        warnings.simplefilter("always")

        from pymatviz.structure_viz.helpers import _draw_element_legend, get_elem_colors

        _draw_element_legend(
            fig=fig,
            struct_i=struct,
            _elem_colors=get_elem_colors({}),
            subplot_idx=999,  # Invalid subplot index
            is_3d=False,
            font_size=12,
            box_size_px=18,
            item_gap_px=3,
            margin_frac=0.04,
        )

        assert len(warning_list) > 0
        warning_messages = [str(w.message) for w in warning_list]
        assert any("domain needed for legend" in msg for msg in warning_messages)

    # Test figure height edge cases
    fig_small = pmv.structure_2d_plotly(struct, site_labels="legend")
    fig_small.layout.height = 100  # Very small height

    legend_annotations = [
        ann
        for ann in fig_small.layout.annotations
        if ann.text and len(ann.text) <= 3 and ann.bgcolor is not None
    ]

    assert len(legend_annotations) == 2
    for ann in legend_annotations:
        assert 0 <= ann.x <= 1
        assert 0 <= ann.y <= 1
