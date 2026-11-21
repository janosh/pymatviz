from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from glob import glob
from importlib.metadata import version
from types import ModuleType

import pymatviz as pmv


def test_pkg_metadata() -> None:
    assert pmv.__version__ == version("pymatviz")

    # ensure __init__.py and site/package.json are in sync
    with open("site/package.json") as file:
        pkg_data = json.load(file)
    assert pkg_data["name"] == "pymatviz"

    with open("pyproject.toml", mode="rb") as file:
        pyproject = tomllib.load(file)

    assert pyproject["project"]["name"] == "pymatviz"
    assert pyproject["project"]["version"] == pmv.__version__
    assert pyproject["project"]["description"] == pkg_data["description"]


def test_all_modules_reexported() -> None:
    """Importing the package explicitly re-exports its public top-level modules."""
    # Test a fresh package namespace without leaking it into subsequent tests.
    original_module = sys.modules.pop("pymatviz")
    try:
        import pymatviz

        for file in glob(f"{pmv.utils.PKG_DIR}/[!_]*.py"):
            module_name = os.path.basename(file).removesuffix(".py")
            # These modules share their names with public plotting functions.
            if module_name in ("histogram", "rainclouds", "sankey"):
                continue

            assert isinstance(getattr(pymatviz, module_name, None), ModuleType), (
                f"{module_name} is not a module exported in pymatviz/__init__.py"
            )
    finally:
        sys.modules["pymatviz"] = original_module


HEAVY_MODULES = (
    "scipy.stats",
    "scipy.spatial",
    "matplotlib",
    "sklearn",
    "pymatgen.phonon",
    "pymatgen.electronic_structure",
    "pymatgen.analysis.local_env",
    "pymatgen.symmetry.analyzer",
    "pymatgen.analysis.diffraction.xrd",
)


def test_import_skips_heavy_modules() -> None:
    """Importing pymatviz skips heavy optional dependencies to stay fast."""
    loaded = f"sorted(set({HEAVY_MODULES!r}) & sys.modules.keys())"
    code = f"import sys, pymatviz; print({loaded})"
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", "heavy modules loaded on import"
