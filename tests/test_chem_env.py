"""Test chemical environment analysis utilities."""

from __future__ import annotations

from pymatgen.core import Structure

from pymatviz import chem_env
from pymatviz.structures import to_structure
from pymatviz.utils.testing import TEST_FILES


AC2BR2O = f"{TEST_FILES}/structures/mp-1207297-Ac2Br2O1-tetragonal.json.gz"


def test_site_environments_chemenv_matches_pymatgen() -> None:
    """ChemEnv symbols equal pymatgen's on the input cell (no refinement)."""
    from pymatgen.analysis.chemenv.coordination_environments import (
        chemenv_strategies,
        coordination_geometry_finder,
        structure_environments,
    )

    pmg_struct = Structure.from_file(AC2BR2O)
    finder = coordination_geometry_finder.LocalGeometryFinder()
    finder.setup_parameters(structure_refinement="none")
    finder.setup_structure(pmg_struct)
    lse = structure_environments.LightStructureEnvironments.from_structure_environments(
        chemenv_strategies.SimplestChemenvStrategy(),
        finder.compute_structure_environments(),
    )
    expected = [envs[0]["ce_symbol"] for envs in lse.coordination_environments or []]
    (envs,) = chem_env.site_environments([to_structure(pmg_struct)])
    assert [symbol for _, symbol in envs] == expected
    assert [cn for cn, _ in envs] == [int(sym.split(":")[1]) for sym in expected]


def test_site_environments_chemenv_without_reference_geometry() -> None:
    """Sites whose CN has no ChemEnv reference geometry are labeled CN:<n>."""
    struct = to_structure(Structure.from_file(f"{TEST_FILES}/structures/mp-12712.json"))
    (envs,) = chem_env.site_environments([struct])
    assert [envs[idx] for idx in (24, 32, 40)] == [(14, "CN:14")] * 3
