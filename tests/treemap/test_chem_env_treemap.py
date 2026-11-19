"""Coordination-environment hierarchies shared by treemap and sunburst widgets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import MagicMock, patch

import pytest
from pymatgen.core import Molecule

import pymatviz as pmv
from pymatviz import chem_env


if TYPE_CHECKING:
    from pymatgen.core import Structure


CHEMENV_MODULE = "pymatgen.analysis.chemenv.coordination_environments"


@pytest.mark.parametrize("chart", ["treemap", "sunburst"])
@pytest.mark.parametrize("mode", ["other", "drop"])
def test_chem_env_hierarchy(
    chart: str,
    mode: Literal["other", "drop"],
    structures: tuple[Structure, Structure],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate duplicate symbols, rank full CNs, break ties and limit both levels."""
    # CN 4 outranks CN 6 before pruning, though each individual CE is smaller.
    samples = [
        (6, "O:6", 1),
        (4, "T:4", 0.5),
        (4, "T:4", 0.25),
        (4, "see-saw-like:4", 0.75),
        (8, "C:8", 0.25),
    ]
    rows = [
        dict(coord_num=coord_num, chem_env_symbol=symbol, count=count)
        for coord_num, symbol, count in samples
    ]
    collect = MagicMock(return_value=rows)
    monkeypatch.setattr(chem_env, "collect_coord_envs_crystal_nn", collect)
    plot = pmv.chem_env_treemap if chart == "treemap" else pmv.chem_env_sunburst
    widget = plot(
        structures,
        normalize=True,
        max_coordination_numbers=1,
        max_environments=1,
        mode=mode,
        value_format=".2f",
    )
    collect.assert_called_once()
    assert collect.call_args.kwargs == {"normalize": True}
    assert widget.widget_type == chart
    assert widget.value_format == ".2f"
    root, *omitted = widget.data
    children = [dict(id="CN 4/T:4", label="T:4", value=0.75)]
    if mode == "other":
        children.append(
            dict(id="CN 4/__omitted__", label="Other (1 more not shown)", value=0.75)
        )
    assert root == dict(id="CN 4", label="CN 4", children=children)
    assert omitted == (
        [dict(id="__omitted__", label="Other (2 more not shown)", value=1.25)]
        if mode == "other"
        else []
    )


@pytest.mark.parametrize(
    ("settings", "envs", "expected"),
    [
        ("chemenv", None, {}),
        ("chemenv", [[]], {}),
        (
            {"custom": "setting"},
            [[{"ce_symbol": "T:4"}], [{"ce_symbol": "O:6"}]],
            {"T:4": 0.5, "O:6": 0.5},
        ),
    ],
)
def test_chem_env_treemap_chemenv(
    structures: tuple[Structure, Structure],
    settings: Any,
    envs: list[list[dict[str, str]]] | None,
    expected: dict[str, float],
) -> None:
    """Forward ChemEnv settings and normalize each classified structure."""
    with (
        patch(
            f"{CHEMENV_MODULE}.coordination_geometry_finder.LocalGeometryFinder"
        ) as finder,
        patch(
            f"{CHEMENV_MODULE}.coordination_geometries.AllCoordinationGeometries"
        ) as geometries,
        patch(
            f"{CHEMENV_MODULE}.structure_environments.LightStructureEnvironments.from_structure_environments",
            return_value=MagicMock(coordination_environments=envs),
        ),
    ):
        geometries.return_value.get_symbol_cn_mapping.return_value = {
            "T:4": 4,
            "O:6": 6,
        }
        widget = pmv.chem_env_treemap(
            structures[0], chem_env_settings=settings, normalize=True
        )
    finder.return_value.setup_parameters.assert_called_once_with(
        **({} if settings == "chemenv" else settings)
    )
    assert {
        child["label"]: child["value"]
        for root in widget.data
        for child in root["children"]
    } == expected


def test_chem_env_treemap_analysis_errors(
    structures: tuple[Structure, Structure],
) -> None:
    """Setup failures propagate; failed structures warn and contribute no data."""
    with patch(
        f"{CHEMENV_MODULE}.coordination_geometry_finder.LocalGeometryFinder"
    ) as finder:
        finder.side_effect = ImportError("ChemEnv unavailable")
        with pytest.raises(ImportError, match="ChemEnv unavailable"):
            pmv.chem_env_treemap(structures, chem_env_settings="chemenv")
        finder.side_effect = None
        finder.return_value.setup_structure.side_effect = RuntimeError("setup failed")
        with pytest.warns(
            UserWarning, match="ChemEnv analysis failed for structure 0.*setup failed"
        ):
            assert (
                pmv.chem_env_treemap(structures[0], chem_env_settings="chemenv").data
                == []
            )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"chem_env_settings": "unknown"}, "Invalid chem_env_settings"),
        ({"mode": "keep"}, "mode"),
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
    monkeypatch.setattr(chem_env, "collect_coord_envs_crystal_nn", collect)
    with pytest.raises(ValueError, match=message):
        pmv.chem_env_treemap(structures, **options)
    collect.assert_not_called()


@pytest.mark.parametrize("method", ["crystal_nn", "chemenv"])
def test_chem_env_requires_periodic_structure(
    method: Literal["crystal_nn", "chemenv"],
) -> None:
    """Both strategies reject nonperiodic inputs before neighbor analysis."""
    molecule = Molecule(["H", "H"], [[0, 0, 0], [0.7, 0, 0]])
    with pytest.raises(TypeError, match=r"Expected periodic structure.*Molecule"):
        pmv.chem_env_treemap(molecule, chem_env_settings=method)  # ty: ignore[invalid-argument-type]
