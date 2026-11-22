"""Coordination-environment hierarchies shared by treemap and sunburst widgets."""

from __future__ import annotations

from typing import Literal
from unittest.mock import MagicMock

import matscikit as mk
import pytest

import pymatviz as pmv
from pymatviz import chem_env


@pytest.mark.parametrize("chart", ["treemap", "sunburst"])
@pytest.mark.parametrize("overflow", ["other", "drop"])
def test_chem_env_hierarchy(
    chart: Literal["treemap", "sunburst"],
    overflow: Literal["other", "drop"],
    structures: tuple[mk.Structure, mk.Structure],
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
        normalize=True,
        max_coordination_numbers=1,
        max_environments=1,
        overflow=overflow,
        value_format=".2f",
    )
    collect.assert_called_once()
    assert len(collect.call_args.args[0]) == 2
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


def test_chem_env_treemap_invalid_overflow(
    structures: tuple[mk.Structure, mk.Structure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An invalid omission mode fails before any environment is classified."""
    collect = MagicMock()
    monkeypatch.setattr(chem_env, "site_environments", collect)
    with pytest.raises(ValueError, match="overflow"):
        pmv.chem_env_hierarchy(structures, overflow="keep", kind="treemap")  # ty: ignore[invalid-argument-type]
    collect.assert_not_called()


def test_chem_env_requires_periodic_structure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nonperiodic inputs are rejected before neighbor analysis."""
    collect = MagicMock()
    monkeypatch.setattr(chem_env, "site_environments", collect)
    molecule = mk.Structure(
        None, ["H", "H"], [[0, 0, 0], [0.7, 0, 0]], pbc=(False,) * 3
    )
    with pytest.raises(
        TypeError, match=r"Expected structures periodic along a, b and c"
    ):
        pmv.chem_env_hierarchy(molecule, kind="treemap")  # ty: ignore[invalid-argument-type]
    collect.assert_not_called()
