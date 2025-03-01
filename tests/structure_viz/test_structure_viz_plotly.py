from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.enums import ElemColorScheme, Key, SiteCoords
from pymatviz.structure_viz.helpers import _angles_to_rotation_matrix, get_image_sites


if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatgen.core import PeriodicSite

COORDS = [[0, 0, 0], [0.5, 0.5, 0.5]]
lattice_cubic = 5 * np.eye(3)  # 5 Å cubic lattice


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
            "site_labels": "species",
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
            "site_labels": "species",
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

    site_trace = next(
        (trace for trace in fig.data if trace.mode == "markers+text"), None
    )
    show_sites = kwargs.get("show_sites")
    if (site_labels := kwargs.get("site_labels")) and show_sites is not False:
        assert site_trace is not None
        if isinstance(site_labels, dict):
            assert any(text in site_trace.text for text in site_labels.values())
        elif site_labels in ("symbol", "species"):
            assert len(site_trace.text) == len(fe3co4_disordered_with_props)

    # Check for sites and arrows
    if show_sites:
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
    fig = pmv.structure_2d_plotly(structs_dict, n_cols=3)
    assert isinstance(fig, go.Figure)
    # 4 structures, 2 sites each, 12 unit cell edges, 8 unit cell nodes
    trace_names = [trace.name or "" for trace in fig.data]
    n_site_traces = sum(name.startswith("site") for name in trace_names)
    assert n_site_traces == 8
    n_edge_traces = sum(name.startswith("edge") for name in trace_names)
    assert n_edge_traces == 48
    n_node_traces = sum(name.startswith("node") for name in trace_names)
    assert n_node_traces == 32
    n_bond_traces = sum(name.startswith("bond") for name in trace_names)
    assert n_bond_traces == 0
    n_image_site_traces = sum(name.startswith("Image of ") for name in trace_names)
    assert n_image_site_traces == 28
    expected_traces = (
        n_site_traces + n_edge_traces + n_node_traces + n_image_site_traces
    )
    assert len(fig.data) == expected_traces, f"{len(fig.data)=}, {expected_traces=}"
    assert len(fig.layout.annotations) == 4

    # Test with pandas.Series[Structure]
    struct_series = pd.Series(structs_dict)
    fig = pmv.structure_2d_plotly(struct_series)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces
    assert len(fig.layout.annotations) == 4

    # Test with list[Structure]
    fig = pmv.structure_2d_plotly(list(structs_dict.values()), n_cols=2)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces
    assert len(fig.layout.annotations) == 4

    # Test with custom subplot_title function
    def subplot_title(struct: Structure, key: str | int) -> str:
        return f"{key} - {struct.formula}"

    fig = pmv.structure_2d_plotly(struct_series, subplot_title=subplot_title)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces
    assert len(fig.layout.annotations) == 4

    # Verify subplot titles
    for idx, (key, struct) in enumerate(structs_dict.items(), start=1):
        assert fig.layout.annotations[idx - 1].text == f"{key} - {struct.formula}"


def test_structure_2d_plotly_invalid_input() -> None:
    """Test that structure_2d_plotly raises errors for invalid inputs."""
    with pytest.raises(
        TypeError, match="Expected pymatgen Structure or Sequence of them"
    ):
        pmv.structure_2d_plotly("invalid input")

    with pytest.raises(
        TypeError, match="Expected pymatgen Structure or Sequence of them, got"
    ):
        pmv.structure_2d_plotly([])

    # Test with invalid rotation string
    struct = Structure(lattice_cubic, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    with pytest.raises(ValueError, match="could not convert string to float: "):
        pmv.structure_2d_plotly(struct, rotation="invalid_rotation")

    # Test with invalid site_labels
    with pytest.raises(ValueError, match="Invalid site_labels=123. Must be one of "):
        pmv.structure_2d_plotly(struct, site_labels=123)  # type: ignore[arg-type]


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
            "site_labels": "species",
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

        if kwargs.get("site_labels"):
            if isinstance(kwargs["site_labels"], dict):
                assert any(
                    text in site_trace.text for text in kwargs["site_labels"].values()
                ), "Expected site labels not found in trace text"
            elif kwargs["site_labels"] in ("symbol", "species"):
                assert len(site_trace.text) == len(fe3co4_disordered_with_props), (
                    "Mismatch in number of site labels"
                )
        else:
            # If site_labels is False, ensure that the trace has no text
            assert site_trace.text is None or len(site_trace.text) == 0, (
                "Unexpected site labels found"
            )

    # Check for sites and arrows
    if kwargs.get("show_sites"):
        site_traces = [
            trace for trace in fig.data if (trace.name or "").startswith("site")
        ]
        assert len(site_traces) > 0, "No site traces found when show_sites is True"

        if show_site_vectors := kwargs.get("show_site_vectors"):
            vector_traces = [
                trace for trace in fig.data if (trace.name or "").startswith("vector")
            ]
            assert len(vector_traces) > 0, (
                f"No vector traces even though {show_site_vectors=}"
            )
            for vector_trace in vector_traces:
                if vector_trace.type == "scatter3d":
                    assert vector_trace.mode == "lines"
                    assert vector_trace.line.color == "white"
                    assert vector_trace.line.width == 5
                elif vector_trace.type == "cone":
                    assert vector_trace.sizemode == "absolute"
                    assert vector_trace.sizeref == 0.8


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
    fig = pmv.structure_3d_plotly(structs_dict, n_cols=2)
    assert isinstance(fig, go.Figure)

    expected_traces = 0
    for struct in structs_dict.values():
        expected_traces += len(struct)  # sites
        expected_traces += 12  # unit cell edges
        expected_traces += 8  # unit cell nodes
        expected_traces += sum(
            len(get_image_sites(site, struct.lattice)) for site in struct
        )  # image sites

    assert len(fig.data) == expected_traces

    assert len(fig.layout.annotations) == 4

    # Test pandas.Series[Structure]
    struct_series = pd.Series(structs_dict)
    fig = pmv.structure_3d_plotly(struct_series)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces
    assert len(fig.layout.annotations) == 4

    # Test list[Structure]
    fig = pmv.structure_3d_plotly(list(structs_dict.values()), n_cols=3)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces
    assert len(fig.layout.annotations) == 4

    # Test subplot_title
    def subplot_title(struct: Structure, key: str | int) -> str:
        return f"{key} - {struct.formula}"

    fig = pmv.structure_3d_plotly(struct_series, subplot_title=subplot_title)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces
    assert len(fig.layout.annotations) == 4
    for idx, (key, struct) in enumerate(structs_dict.items(), start=1):
        assert fig.layout.annotations[idx - 1].text == f"{key} - {struct.formula}"


def test_structure_3d_plotly_invalid_input() -> None:
    with pytest.raises(
        TypeError, match="Expected pymatgen Structure or Sequence of them"
    ):
        pmv.structure_3d_plotly("invalid input")


@pytest.mark.parametrize(
    "custom_title_dict",
    [
        {
            "text": "Custom {key} - {struct.formula}",
            "font": {"size": 16, "color": "red"},
            "y": 0.8,
            "yanchor": "bottom",
        },
        {
            "text": "{struct.formula} ({key})",
            "font": {"size": 14, "color": "blue"},
            "x": 0.5,
            "xanchor": "center",
        },
        {
            "text": "Structure {key}",
            "font": {"size": 18, "color": "green"},
            "y": 1.0,
            "yanchor": "top",
        },
        {},  # Empty dict to test default behavior
    ],
)
def test_structure_3d_plotly_subplot_title_override(
    custom_title_dict: dict[str, str | float | dict[str, str | float]],
) -> None:
    struct1 = Structure(lattice_cubic, ["Fe", "O"], COORDS)
    struct2 = Structure(lattice_cubic, ["Co", "O"], COORDS)
    structs_dict = {"struct1": struct1, "struct2": struct2}

    def custom_subplot_title(struct: Structure, key: str | int) -> dict[str, Any]:
        title_dict = custom_title_dict.copy()
        if "text" in title_dict:
            assert isinstance(title_dict["text"], str)  # for mypy
            title_dict["text"] = title_dict["text"].format(key=key, struct=struct)
        return title_dict

    fig = pmv.structure_3d_plotly(structs_dict, subplot_title=custom_subplot_title)

    assert isinstance(fig, go.Figure)
    assert len(fig.layout.annotations) == 2

    for idx, (key, struct) in enumerate(structs_dict.items()):
        annotation = fig.layout.annotations[idx]

        if custom_title_dict:
            custom_title = custom_title_dict.get("text", "")
            assert isinstance(custom_title, str)  # for mypy
            expected_text = custom_title.format(key=key, struct=struct)
            assert annotation.text == expected_text

            for attr, value in custom_title_dict.items():
                if attr == "font":
                    assert isinstance(value, dict)  # for mypy
                    for font_attr, font_value in value.items():
                        assert getattr(annotation.font, font_attr) == font_value
                elif attr != "text":
                    assert getattr(annotation, attr) == value
        else:
            # Check default behavior when an empty dict is provided
            assert annotation.text == key  # Changed this line
            assert annotation.font.size == 16
            assert annotation.font.color is None
            assert annotation.yanchor == "top"


@pytest.mark.parametrize(
    "plot_function", [pmv.structure_2d_plotly, pmv.structure_3d_plotly]
)
@pytest.mark.parametrize(
    "hover_text", [*SiteCoords, lambda site: f"<b>{site.frac_coords}</b>"]
)
def test_hover_text(
    plot_function: Callable[..., go.Figure],
    hover_text: SiteCoords | Callable[[PeriodicSite], str],
) -> None:
    struct = Structure(lattice_cubic, ["Fe", "O"], COORDS)
    fig = plot_function(struct, hover_text=hover_text)

    site_traces = [
        trace for trace in fig.data if trace.name and trace.name.startswith("site")
    ]
    assert len(site_traces) > 0

    # regex for a single site coordinate with optional decimal point
    re_coord = r"\d+\.?\d*"
    re_3_coords = rf"{re_coord}, {re_coord}, {re_coord}"
    for trace in site_traces:
        site_hover_text = trace.hovertext
        if callable(hover_text):
            assert "<b>" in site_hover_text, f"{site_hover_text=}"
            assert "</b>" in site_hover_text, f"{site_hover_text=}"
        elif hover_text == SiteCoords.cartesian:
            assert re.search(rf"Coordinates \({re_3_coords}\)", site_hover_text), (
                f"{site_hover_text=}"
            )
        elif hover_text == SiteCoords.fractional:
            assert re.search(rf"Coordinates \[{re_3_coords}\]", site_hover_text), (
                f"{site_hover_text=}"
            )
        elif hover_text == SiteCoords.cartesian_fractional:
            assert re.search(
                rf"Coordinates \({re_3_coords}\) \[{re_3_coords}\]", site_hover_text
            ), f"{site_hover_text=}"


@pytest.mark.parametrize(
    "plot_function", [pmv.structure_2d_plotly, pmv.structure_3d_plotly]
)
def test_structure_plotly_ase_atoms(
    plot_function: Callable[[Structure], go.Figure], structures: list[Structure]
) -> None:
    """Test that structure_2d_plotly works with ASE Atoms."""
    pytest.importorskip("ase")

    pmg_struct = structures[0]
    # Create a simple ASE Atoms object

    # Test single Atoms object
    fig_ase = plot_function(pmg_struct.to_ase_atoms())
    assert isinstance(fig_ase, go.Figure)

    # Test equivalence with pymatgen Structure
    fig_pmg = plot_function(pmg_struct)

    # Compare figures
    for trace_ase, trace_pmg in zip(fig_ase.data, fig_pmg.data, strict=True):
        assert trace_ase.type == trace_pmg.type, f"{trace_ase.type=}, {trace_pmg.type=}"
        assert trace_ase.mode == trace_pmg.mode, f"{trace_ase.mode=}, {trace_pmg.mode=}"
        assert trace_ase.name == trace_pmg.name, f"{trace_ase.name=}, {trace_pmg.name=}"
        n_x_ase, n_y_ase = len(trace_ase.x), len(trace_ase.y)
        n_x_pmg, n_y_pmg = len(trace_pmg.x), len(trace_pmg.y)
        assert n_x_ase == n_x_pmg, f"{n_x_ase=}, {n_x_pmg=}"
        assert n_y_ase == n_y_pmg, f"{n_y_ase=}, {n_y_pmg=}"

    # Test sequence of Atoms objects
    atoms_list = [struct.to_ase_atoms() for struct in structures]
    fig = plot_function(atoms_list)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    ("plot_function", "is_3d"),
    [(pmv.structure_2d_plotly, False), (pmv.structure_3d_plotly, True)],
)
def test_structure_plotly_show_bonds(
    plot_function: Callable[..., go.Figure], is_3d: bool
) -> None:
    """Test that bonds are drawn correctly when show_bonds is True."""
    # Create a simple structure with known bonding
    struct = Structure(lattice_cubic, ["Si", "O"], [[0, 0, 0], [0.2, 0.2, 0.2]])

    # Test with show_bonds=True (default CrystalNN)
    fig = plot_function(struct, show_bonds=True)

    # Check that bonds were drawn
    bond_traces = [trace for trace in fig.data if (trace.name or "").startswith("bond")]
    assert len(bond_traces) == 2

    # Check bond properties
    for trace in bond_traces:
        assert trace.mode == "lines"
        assert trace.showlegend is False
        assert trace.hoverinfo == "skip"
        assert trace.line.color == "white"  # Default color
        assert trace.line.width == 4  # Default width

    # Check that the trace type is correct based on dimension
    expected_trace_type = go.Scatter3d if is_3d else go.Scatter
    assert all(isinstance(trace, expected_trace_type) for trace in bond_traces)


@pytest.mark.parametrize(
    "plot_function", [pmv.structure_2d_plotly, pmv.structure_3d_plotly]
)
def test_structure_plotly_show_bonds_custom_nn(
    plot_function: Callable[..., go.Figure],
) -> None:
    """Test that bonds are drawn correctly with custom NearNeighbors."""
    from pymatgen.analysis.local_env import VoronoiNN

    # Create a simple structure with known bonding
    struct = Structure(lattice_cubic, ["Si", "O"], [[0, 0, 0], [0.2, 0.2, 0.2]])

    # Use VoronoiNN instead of default CrystalNN
    nn = VoronoiNN()

    # Test with custom NearNeighbors
    fig = plot_function(struct, show_bonds=nn)

    # Check that bonds were drawn
    bond_traces = [trace for trace in fig.data if (trace.name or "").startswith("bond")]
    assert len(bond_traces) == 12


@pytest.mark.parametrize(
    "plot_function", [pmv.structure_2d_plotly, pmv.structure_3d_plotly]
)
def test_structure_plotly_show_bonds_custom_kwargs(
    plot_function: Callable[..., go.Figure],
) -> None:
    """Test that bond_kwargs are applied correctly."""
    # Create a simple structure with known bonding
    struct = Structure(lattice_cubic, ["Si", "O"], [[0, 0, 0], [0.2, 0.2, 0.2]])

    # Custom bond styling
    bond_kwargs = {"color": "red", "width": 2, "dash": "dot"}

    # Test with custom bond styling
    fig = plot_function(struct, show_bonds=True, bond_kwargs=bond_kwargs)

    # Check that bonds were drawn with custom styling
    bond_traces = [trace for trace in fig.data if (trace.name or "").startswith("bond")]
    assert len(bond_traces) == 2

    # Check that custom styling was applied
    for trace in bond_traces:
        for key, value in bond_kwargs.items():
            assert getattr(trace.line, key) == value


@pytest.mark.parametrize(
    "plot_function", [pmv.structure_2d_plotly, pmv.structure_3d_plotly]
)
def test_structure_plotly_show_bonds_with_image_sites(
    plot_function: Callable[..., go.Figure],
) -> None:
    """Test that bonds are drawn correctly to visible image sites."""
    # Create a simple structure where bonds will cross unit cell boundaries
    struct = Structure(lattice_cubic, ["Si", "O"], [[0, 0, 0], [0.9, 0.9, 0.9]])

    # Test with both show_bonds and show_image_sites enabled
    fig = plot_function(struct, show_bonds=True, show_image_sites=True)

    # Check that bonds were drawn
    bond_traces = [trace for trace in fig.data if (trace.name or "").startswith("bond")]
    assert len(bond_traces) == 1

    # Check for image sites
    image_site_traces = [
        trace for trace in fig.data if (trace.name or "").startswith("Image of ")
    ]
    assert len(image_site_traces) == 7

    # Verify that at least one bond is drawn
    # The exact count will depend on the structure and NearNeighbors algorithm
    assert len(bond_traces) == 1


@pytest.mark.parametrize(
    ("plot_function", "is_3d"),
    [(pmv.structure_2d_plotly, False), (pmv.structure_3d_plotly, True)],
)
def test_structure_plotly_no_bonds_to_hidden_sites(
    plot_function: Callable[..., go.Figure], is_3d: bool
) -> None:
    """Test that bonds are not drawn to sites that aren't visible."""
    # Create a simple structure
    struct = Structure(lattice_cubic, ["Si", "O"], [[0, 0, 0], [0.9, 0.9, 0.9]])

    # Test with show_bonds but without show_image_sites
    fig = plot_function(struct, show_bonds=True, show_image_sites=False)

    # Check that bonds were drawn
    bond_traces = [trace for trace in fig.data if (trace.name or "").startswith("bond")]

    # Check for image sites (should be none)
    image_site_traces = [
        trace for trace in fig.data if (trace.name or "").startswith("Image of ")
    ]
    assert len(image_site_traces) == 0, "Image sites found when show_image_sites=False"

    # Verify that we only have bonds within the unit cell
    # For this structure, we should have fewer bonds when image sites are hidden
    if len(bond_traces) > 0:  # Some bonds might still be drawn within the unit cell
        # Get the maximum coordinate value in any bond
        max_coords = []
        for trace in bond_traces:
            if is_3d:
                max_coords.extend([max(trace.x), max(trace.y), max(trace.z)])
            else:
                max_coords.extend([max(trace.x), max(trace.y)])

        # All coordinates should be within or very close to the unit cell
        assert max(max_coords) <= 3.1, "Bonds drawn outside the unit cell"


def test_structure_plotly_visible_image_atoms_population() -> None:
    """Test that the visible_image_atoms set is correctly populated and used for bond
    drawing."""
    # Create a simple structure where bonds will cross unit cell boundaries
    struct = Structure(lattice_cubic, ["Si", "O"], [[0, 0, 0], [0.9, 0.9, 0.9]])

    # Test with both show_bonds and show_image_sites enabled
    fig = pmv.structure_3d_plotly(struct, show_bonds=True, show_image_sites=True)

    # Check that bonds were drawn
    bond_traces = [trace for trace in fig.data if (trace.name or "").startswith("bond")]
    assert len(bond_traces) == 1

    # Check for image sites
    image_site_traces = [
        trace for trace in fig.data if (trace.name or "").startswith("Image of ")
    ]
    assert len(image_site_traces) == 7

    # Now test the 2D case with rotation
    fig = pmv.structure_2d_plotly(
        struct, rotation="45x,30y,15z", show_bonds=True, show_image_sites=True
    )

    # Check that bonds were drawn
    bond_traces = [trace for trace in fig.data if (trace.name or "").startswith("bond")]
    assert len(bond_traces) == 1

    # Check for image sites
    image_site_traces = [
        trace for trace in fig.data if (trace.name or "").startswith("Image of ")
    ]
    assert len(image_site_traces) == 7


def test_unit_cell_rotation() -> None:
    """Test rotation is correctly applied to the unit cell in structure_2d_plotly."""
    struct = Structure(lattice_cubic, ["Fe", "O"], COORDS)
    rotations: list[str] = [
        "0x,0y,0z",
        "45x,0y,0z",
        "0x,45y,0z",
        "0x,0y,45z",
        "30x,30y,30z",
    ]

    for rotation in rotations:
        rotation_matrix = _angles_to_rotation_matrix(rotation)
        fig = pmv.structure_2d_plotly(
            struct,
            rotation=rotation,
            show_unit_cell={"edge": {"color": "red", "width": 2}},
        )

        edge_traces = [
            trace for trace in fig.data if trace.name and trace.name.startswith("edge")
        ]
        site_traces = [
            trace for trace in fig.data if trace.name and trace.name.startswith("site")
        ]

        assert len(edge_traces) == 12
        assert len(site_traces) == 2

        if rotation != "0x,0y,0z":
            corners = [
                (0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (1, 1, 0),
                (0, 0, 1),
                (1, 0, 1),
                (0, 1, 1),
                (1, 1, 1),
            ]
            cart_corners = struct.lattice.get_cartesian_coords(corners)
            rotated_corners = np.dot(cart_corners, rotation_matrix)

            edge_start = [edge_traces[0].x[0], edge_traces[0].y[0]]
            distances = np.sum((rotated_corners[:, :2] - edge_start) ** 2, axis=1)
            assert (np.min(distances) ** 0.5) < 1e-10


def test_consistent_rotation_atoms_and_unit_cell() -> None:
    """Test that rotation is applied consistently to both atoms and unit cell."""
    positions: list[list[float]] = [
        [0.0, 0.0, 0.0],  # origin
        [1.0, 0.0, 0.0],  # x-axis corner
        [0.0, 1.0, 0.0],  # y-axis corner
        [0.0, 0.0, 1.0],  # z-axis corner
    ]
    species: list[str] = ["Fe", "O", "Ni", "Co"]
    struct = Structure(lattice_cubic, species, positions)

    rotation = "30x,45y,60z"
    fig = pmv.structure_2d_plotly(
        struct,
        rotation=rotation,
        show_unit_cell={"edge": {"color": "red", "width": 2}},
    )

    edge_traces = [
        trace for trace in fig.data if trace.name and trace.name.startswith("edge")
    ]
    site_traces = [
        trace for trace in fig.data if trace.name and trace.name.startswith("site")
    ]

    # Check origin alignment
    origin_atom_pos = np.array([site_traces[0].x[0], site_traces[0].y[0]])
    origin_corner_pos = np.array([edge_traces[0].x[0], edge_traces[0].y[0]])
    assert np.sqrt(np.sum((origin_atom_pos - origin_corner_pos) ** 2)) < 1e-10

    # Check x-axis alignment
    x_atom_pos = np.array([site_traces[1].x[0], site_traces[1].y[0]])
    x_corner_candidates: list[np.ndarray] = []
    for trace in edge_traces:
        if np.allclose([trace.x[0], trace.y[0]], origin_corner_pos, atol=1e-10):
            x_corner_candidates.append(np.array([trace.x[2], trace.y[2]]))
        elif np.allclose([trace.x[2], trace.y[2]], origin_corner_pos, atol=1e-10):
            x_corner_candidates.append(np.array([trace.x[0], trace.y[0]]))

    distances = [
        np.sqrt(np.sum((x_atom_pos - candidate) ** 2))
        for candidate in x_corner_candidates
    ]
    assert min(distances) < 1e-10


def test_independent_subplot_zooming() -> None:
    """Test that each subplot in structure_2d_plotly can be zoomed independently."""
    structs: dict[str, Structure] = {
        f"struct{idx}": Structure(lattice_cubic, [elem, "O"], COORDS)
        for idx, elem in enumerate(["Fe", "Co", "Ni", "Cu"], 1)
    }

    fig = pmv.structure_2d_plotly(structs, n_cols=2)

    for idx in range(1, len(structs) + 1):
        x_axis = getattr(fig.layout, f"xaxis{idx if idx > 1 else ''}")
        y_axis = getattr(fig.layout, f"yaxis{idx if idx > 1 else ''}")

        # Check that axes are properly anchored
        assert x_axis.scaleanchor == f"y{idx if idx > 1 else ''}"
        assert y_axis.scaleanchor == f"x{idx if idx > 1 else ''}"

        # Set fixedrange to False to make axes zoomable
        assert x_axis.fixedrange is None
        assert y_axis.fixedrange is None


def test_rotation_matrix_passed_to_draw_unit_cell() -> None:
    """Test that the rotation matrix is passed to draw_unit_cell."""
    struct = Structure(lattice_cubic, ["Fe", "O"], COORDS)

    from pymatviz.structure_viz import plotly

    original_draw_unit_cell = plotly.draw_unit_cell
    passed_rotation_matrix: list[np.ndarray] = []

    def mock_draw_unit_cell(
        fig: go.Figure,
        structure: Structure,
        unit_cell_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> go.Figure:
        if "rotation_matrix" in kwargs:
            passed_rotation_matrix.append(kwargs["rotation_matrix"])
        return original_draw_unit_cell(fig, structure, unit_cell_kwargs, **kwargs)

    plotly.draw_unit_cell = mock_draw_unit_cell

    try:
        rotation = "30x,30y,30z"
        pmv.structure_2d_plotly(struct, rotation=rotation)

        assert len(passed_rotation_matrix) == 1
        expected_matrix = _angles_to_rotation_matrix(rotation)
        assert np.allclose(passed_rotation_matrix[0], expected_matrix)
    finally:
        plotly.draw_unit_cell = original_draw_unit_cell
