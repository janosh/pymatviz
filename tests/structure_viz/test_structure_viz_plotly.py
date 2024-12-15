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
from pymatviz.structure_viz.helpers import get_image_sites


if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatgen.core import PeriodicSite

COORDS = [[0, 0, 0], [0.5, 0.5, 0.5]]
DISORDERED_STRUCT = Structure(
    lattice := np.eye(3) * 5,
    species=[{"Fe": 0.75, "C": 0.25}, "O"],
    coords=COORDS,
    site_properties={
        "magmom": [[0, 0, 1], [0, 0, -1]],  # Vector values for magmom
        "force": [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]],
    },
)


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
def test_structure_2d_plotly(kwargs: dict[str, Any]) -> None:
    fig = pmv.structure_2d_plotly(DISORDERED_STRUCT, **kwargs)
    assert isinstance(fig, go.Figure)

    # Check if the layout properties are set correctly
    assert fig.layout.showlegend is False
    assert fig.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert fig.layout.plot_bgcolor == "rgba(0,0,0,0)"

    # Check if the axes properties are set correctly
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
            assert len(site_trace.text) == len(DISORDERED_STRUCT)

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
            assert (
                len(vector_traces) > 0
            ), "No vector traces found when show_site_vectors is True"
            for vector_trace in vector_traces:
                assert vector_trace.mode == "lines+markers"
                assert vector_trace.marker.symbol == "arrow"
                assert "angle" in vector_trace.marker


def test_structure_2d_plotly_multiple() -> None:
    struct1 = Structure(lattice, ["Fe", "O"], coords=COORDS)
    struct1.properties = {"id": "struct1"}
    struct2 = Structure(lattice, ["Co", "O"], coords=COORDS)
    struct2.properties = {Key.mat_id: "struct2"}
    struct3 = Structure(lattice, ["Ni", "O"], coords=COORDS)
    struct3.properties = {"ID": "struct3", "name": "nickel oxide"}
    struct4 = Structure(lattice, ["Cu", "O"], coords=COORDS)

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

    # Test pandas.Series[Structure]
    struct_series = pd.Series(structs_dict)
    fig = pmv.structure_2d_plotly(struct_series)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces, f"{len(fig.data)=}, {expected_traces=}"
    assert len(fig.layout.annotations) == 4

    # Test list[Structure]
    fig = pmv.structure_2d_plotly(list(structs_dict.values()), n_cols=2)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces, f"{len(fig.data)=}, {expected_traces=}"
    assert len(fig.layout.annotations) == 4

    # Test subplot_title
    def subplot_title(struct: Structure, key: str | int) -> str:
        return f"{key} - {struct.formula}"

    fig = pmv.structure_2d_plotly(struct_series, subplot_title=subplot_title)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces, f"{len(fig.data)=}, {expected_traces=}"
    assert len(fig.layout.annotations) == 4
    for idx, (key, struct) in enumerate(structs_dict.items(), start=1):
        assert fig.layout.annotations[idx - 1].text == f"{key} - {struct.formula}"


def test_structure_2d_plotly_invalid_input() -> None:
    with pytest.raises(
        TypeError, match="Expected pymatgen Structure or Sequence of them"
    ):
        pmv.structure_2d_plotly("invalid input")


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
def test_structure_3d_plotly(kwargs: dict[str, Any]) -> None:
    fig = pmv.structure_3d_plotly(DISORDERED_STRUCT, **kwargs)
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
                assert len(site_trace.text) == len(
                    DISORDERED_STRUCT
                ), "Mismatch in number of site labels"
        else:
            # If site_labels is False, ensure that the trace has no text
            assert (
                site_trace.text is None or len(site_trace.text) == 0
            ), "Unexpected site labels found"

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
            assert (
                len(vector_traces) > 0
            ), f"No vector traces even though {show_site_vectors=}"
            for vector_trace in vector_traces:
                if vector_trace.type == "scatter3d":
                    assert vector_trace.mode == "lines"
                    assert vector_trace.line.color == "white"
                    assert vector_trace.line.width == 5
                elif vector_trace.type == "cone":
                    assert vector_trace.sizemode == "absolute"
                    assert vector_trace.sizeref == 0.8


def test_structure_3d_plotly_multiple() -> None:
    struct1 = Structure(lattice, ["Fe", "O"], COORDS)
    struct1.properties = {"id": "struct1"}
    struct2 = Structure(lattice, ["Co", "O"], COORDS)
    struct2.properties = {Key.mat_id: "struct2"}
    struct3 = Structure(lattice, ["Ni", "O"], COORDS)
    struct3.properties = {"ID": "struct3", "name": "nickel oxide"}
    struct4 = Structure(lattice, ["Cu", "O"], COORDS)

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
    struct1 = Structure(lattice, ["Fe", "O"], COORDS)
    struct2 = Structure(lattice, ["Co", "O"], COORDS)
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
    plot_function: Callable[[Structure, Any], go.Figure],
    hover_text: SiteCoords | Callable[[PeriodicSite], str],
) -> None:
    struct = Structure(lattice, ["Fe", "O"], COORDS)
    fig = plot_function(struct, hover_text=hover_text)  # type: ignore[call-arg]

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
            assert re.search(
                rf"Coordinates \({re_3_coords}\)", site_hover_text
            ), f"{site_hover_text=}"
        elif hover_text == SiteCoords.fractional:
            assert re.search(
                rf"Coordinates \[{re_3_coords}\]", site_hover_text
            ), f"{site_hover_text=}"
        elif hover_text == SiteCoords.cartesian_fractional:
            assert re.search(
                rf"Coordinates \({re_3_coords}\) \[{re_3_coords}\]", site_hover_text
            ), f"{site_hover_text=}"
