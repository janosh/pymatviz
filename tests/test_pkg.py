from __future__ import annotations

import json
import os
import sys
from glob import glob
from importlib.metadata import version
from types import ModuleType

import pymatviz as pmv


def test_pkg_metadata() -> None:
    assert pmv.__version__ == version(pmv.PKG_NAME)

    # ensure __init__.py and site/package.json are in sync
    with open("site/package.json") as file:
        pkg_data = json.load(file)
    assert pkg_data["name"] == pmv.PKG_NAME

    try:
        import tomllib

        # check pyproject.toml matches package.json and __init__.py
        with open("pyproject.toml", mode="rb") as file:
            pyproject = tomllib.load(file)

        assert pyproject["project"]["name"] == pmv.PKG_NAME
        assert pyproject["project"]["version"] == pmv.__version__
        assert pyproject["project"]["description"] == pkg_data["description"]
    except ImportError:
        pass  # tomllib only available in 3.11+


def test_all_modules_reexported() -> None:
    # pytest seems to have special import behavior for the tested module making all
    # submodules importable, regardless of whether __init__ re-exports them, so we
    # override pymatviz importing for this test
    sys.modules.pop(pmv.PKG_NAME, None)

    import pymatviz  # manually re-import

    try:
        first_level_modules = [
            f"{pmv.PKG_NAME}.{os.path.basename(os.path.splitext(file)[0])}"
            for file in glob(f"{pmv.PKG_DIR}/*.py")
            if "__init__.py" not in file
        ]

        for full_module_name in first_level_modules:
            module_name = full_module_name.split(".")[-1]

            # Skip histogram module which name-clashes with its own histogram function
            if module_name in ("histogram", "rainclouds"):
                continue

            # Check if the module or subpackage is in the main package namespace
            assert hasattr(pymatviz, module_name), (
                f"{module_name} not exported in {pmv.PKG_NAME}/__init__.py"
            )

            reexported_submodule = getattr(pymatviz, module_name)

            # For subpackages, check if it's a module (subpackages are also modules)
            if "." in full_module_name.split(".", maxsplit=1)[1]:
                assert isinstance(reexported_submodule, ModuleType), (
                    f"{module_name} in {pmv.PKG_NAME}/__init__.py is not a "
                    "module/subpackage"
                )
            else:
                assert (  # For regular modules, check more strictly
                    type(reexported_submodule).__name__ == "module"
                ), f"{module_name} in {pmv.PKG_NAME}/__init__.py is not a module"

    finally:
        sys.modules[pmv.PKG_NAME] = pymatviz


def test_is_ipython() -> None:
    assert not pmv.IS_IPYTHON
