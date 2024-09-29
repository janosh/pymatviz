from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pymatgen.core import Structure

import pymatviz as pmv
from pymatviz.enums import ElemColorScheme, Key


COORDS = [[0, 0, 0], [0.5, 0.5, 0.5]]
DISORDERED_STRUCT = Structure(
    lattice := np.eye(3) * 5, species=[{"Fe": 0.75, "C": 0.25}, "O"], coords=COORDS
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

    # Check if the figure has the correct number of traces
    expected_traces = 0
    if show_sites := kwargs.get("show_sites"):
        expected_traces += len(DISORDERED_STRUCT)
    if show_unit_cell := kwargs.get("show_unit_cell"):
        expected_traces += 12  # 12 edges in a cube
    assert len(fig.data) == expected_traces

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
    if isinstance(show_unit_cell, dict):
        unit_cell_trace = next(
            (trace for trace in fig.data if trace.mode == "lines"), None
        )
        assert unit_cell_trace is not None
        for key, value in show_unit_cell.items():
            assert unit_cell_trace.line[key] == value

    site_trace = next(
        (trace for trace in fig.data if trace.mode == "markers+text"), None
    )
    if (site_labels := kwargs.get("site_labels")) and show_sites is not False:
        assert site_trace is not None
        if isinstance(site_labels, dict):
            assert any(text in site_trace.text for text in site_labels.values())
        # elif isinstance(kwargs["site_labels"], list):
        #     assert all(label in site_trace.text for label in kwargs["site_labels"])
        elif site_labels in ("symbol", "species"):
            assert len(site_trace.text) == len(DISORDERED_STRUCT)


def test_structure_2d_plotly_multiple() -> None:
    struct1 = Structure(lattice, ["Fe", "O"], coords=COORDS)
    struct1.properties = {"id": "struct1"}
    struct2 = Structure(lattice, ["Co", "O"], coords=COORDS)
    struct2.properties = {Key.mat_id: "struct2"}
    struct3 = Structure(lattice, ["Ni", "O"], coords=COORDS)
    struct3.properties = {"ID": "struct3", "name": "nickel oxide"}
    struct4 = Structure(lattice, ["Cu", "O"], coords=COORDS)

    # Test dict[str, Structure]
    struct_dict = {
        "struct1": struct1,
        "struct2": struct2,
        "struct3": struct3,
        "struct4": struct4,
    }
    fig = pmv.structure_2d_plotly(struct_dict, n_cols=3)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4 * (
        len(COORDS) + 12
    )  # 4 structures, 2 sites each, 12 unit cell edges
    assert len(fig.layout.annotations) == 4

    # Test pandas.Series[Structure]
    struct_series = pd.Series(struct_dict)
    fig = pmv.structure_2d_plotly(struct_series)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4 * (len(COORDS) + 12)
    assert len(fig.layout.annotations) == 4

    # Test list[Structure]
    fig = pmv.structure_2d_plotly(list(struct_dict.values()), n_cols=2)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4 * (len(COORDS) + 12)
    assert len(fig.layout.annotations) == 4

    # Test subplot_title
    def subplot_title(struct: Structure, key: str | int) -> str:
        return f"{key} - {struct.formula}"

    fig = pmv.structure_2d_plotly(struct_series, subplot_title=subplot_title)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 4 * (len(COORDS) + 12)
    assert len(fig.layout.annotations) == 4
    for idx, (_key, struct) in enumerate(struct_dict.items(), start=1):
        assert fig.layout.annotations[idx - 1].text == f"{idx} - {struct.formula}"


def test_structure_2d_plotly_invalid_input() -> None:
    with pytest.raises(
        TypeError, match="Expected pymatgen Structure or Sequence of them"
    ):
        pmv.structure_2d_plotly("invalid input")
