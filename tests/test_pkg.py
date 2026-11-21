from __future__ import annotations

import os
import subprocess
import sys
from glob import glob
from types import ModuleType

import pymatviz as pmv


def test_all_modules_reexported() -> None:
    """Importing the package explicitly re-exports its public top-level modules."""
    # Test a fresh package namespace without leaking it into subsequent tests.
    original_module = sys.modules.pop("pymatviz")
    try:
        import pymatviz

        for file in glob(f"{pmv.utils.PKG_DIR}/[!_]*.py"):
            module_name = os.path.basename(file).removesuffix(".py")
            attr = getattr(pymatviz, module_name, None)
            if callable(attr):  # name taken by a plot function of that module
                continue
            assert isinstance(attr, ModuleType), (
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
