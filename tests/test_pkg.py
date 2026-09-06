from __future__ import annotations

import json
import os
import sys
import tomllib
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

    # check pyproject.toml matches package.json and __init__.py
    with open("pyproject.toml", mode="rb") as file:
        pyproject = tomllib.load(file)

    assert pyproject["project"]["name"] == pmv.PKG_NAME
    assert pyproject["project"]["version"] == pmv.__version__
    assert pyproject["project"]["description"] == pkg_data["description"]


def test_all_modules_reexported() -> None:
    """Top-level modules are explicitly re-exported when importing the package."""
    # Test a fresh package namespace without leaking it into subsequent tests.
    original_module = sys.modules.pop(pmv.PKG_NAME)
    try:
        import pymatviz

        for file in glob(f"{pmv.PKG_DIR}/*.py"):
            module_name = os.path.basename(file).removesuffix(".py")
            # These modules share their names with public plotting functions.
            if module_name in ("__init__", "histogram", "rainclouds"):
                continue

            assert isinstance(getattr(pymatviz, module_name, None), ModuleType), (
                f"{module_name} is not a module exported in {pmv.PKG_NAME}/__init__.py"
            )
    finally:
        sys.modules[pmv.PKG_NAME] = original_module
