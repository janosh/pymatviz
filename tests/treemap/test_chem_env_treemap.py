"""Coordination-environment hierarchies shared by treemap and sunburst widgets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import MagicMock

import pytest
from pymatgen.core import Molecule

import pymatviz as pmv
from pymatviz import chem_env


if TYPE_CHECKING:
    from pymatgen.core import Structure


@pytest.mark.parametrize("method", ["crystal_nn", "chemenv"])
@pytest.mark.parametrize("chart", ["treemap", "sunburst"])
@pytest.mark.parametrize("overflow", ["other", "drop"])
def test_chem_env_hierarchy(
    method: Literal["crystal_nn", "chemenv"],
    chart: Literal["treemap", "sunburst"],
    overflow: Literal["other", "drop"],
    structures: tuple[Structure, Structure],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate duplicate symbols, weight sites per structure, rank full CNs, break
    ties and limit both levels.
    """
    # normalized: 1/4 per site of the first structure, 1/2 of the second
    envs = [
        [(6, "O:6"), (4, "T:4"), (4, "see-saw-like:4"), (4, "see-saw-like:4")],
        [(4, "T:4"), (8, "C:8")],
    ]
    collect = MagicMock(return_value=envs)
    monkeypatch.setattr(chem_env, "site_environments", collect)
    widget = pmv.chem_env_hierarchy(
        structures,
        kind=chart,
        method=method,
        normalize=True,
        max_coordination_numbers=1,
        max_environments=1,
        overflow=overflow,
        value_format=".2f",
    )
    collect.assert_called_once()
    assert len(collect.call_args.args[0]) == 2
    assert collect.call_args.args[1] == method
    assert widget.widget_type == chart
    assert widget.value_format == ".2f"
    root, *omitted = widget.data
    children = [dict(id="CN 4/T:4", label="T:4", value=0.75)]
    if overflow == "other":
        children.append(
            dict(id="CN 4/__omitted__", label="Other (1 more not shown)", value=0.5)
        )
    assert root == dict(id="CN 4", label="CN 4", children=children)
    assert omitted == (
        [dict(id="__omitted__", label="Other (2 more not shown)", value=0.75)]
        if overflow == "other"
        else []
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"method": "unknown"}, "Invalid method"),
        ({"overflow": "keep"}, "overflow"),
    ],
)
def test_chem_env_treemap_invalid_options(
    structures: tuple[Structure, Structure],
    options: dict[str, Any],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid methods or omission modes fail even for an empty hierarchy."""
    collect = MagicMock()
    monkeypatch.setattr(chem_env, "site_environments", collect)
    with pytest.raises(ValueError, match=message):
        pmv.chem_env_hierarchy(structures, **options, kind="treemap")
    collect.assert_not_called()


@pytest.mark.parametrize("method", ["crystal_nn", "chemenv"])
def test_chem_env_requires_periodic_structure(
    method: Literal["crystal_nn", "chemenv"],
) -> None:
    """Both strategies reject nonperiodic inputs before neighbor analysis."""
    molecule = Molecule(["H", "H"], [[0, 0, 0], [0.7, 0, 0]])
    with pytest.raises(
        TypeError, match=r"Expected structures periodic along a, b and c"
    ):
        pmv.chem_env_hierarchy(molecule, method=method, kind="treemap")  # ty: ignore[invalid-argument-type]
