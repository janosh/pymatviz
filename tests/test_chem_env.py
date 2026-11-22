"""Test chemical environment analysis utilities."""

from __future__ import annotations

from pymatviz import chem_env
from pymatviz.structures import to_structure
from pymatviz.utils.testing import TEST_FILES


def test_site_environments_chemenv_without_reference_geometry() -> None:
    """Sites whose CN has no ChemEnv reference geometry are labeled CN:<n>."""
    struct = to_structure(f"{TEST_FILES}/structures/mp-12712.json")
    (envs,) = chem_env.site_environments([struct])
    assert [envs[idx] for idx in (24, 32, 40)] == [(14, "CN:14")] * 3
