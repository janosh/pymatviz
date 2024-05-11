from __future__ import annotations

import json
from importlib.metadata import version

from pymatviz import PKG_NAME, __version__


def test_pkg_metadata() -> None:
    assert __version__ == version(PKG_NAME)

    # ensure __init__.py and site/package.json are in sync
    with open("site/package.json") as file:
        pkg_data = json.load(file)
    assert pkg_data["name"] == PKG_NAME

    try:
        import tomllib

        # check pyproject.toml matches package.json and __init__.py
        with open("pyproject.toml", mode="rb") as file:
            pyproject = tomllib.load(file)

        assert pyproject["project"]["name"] == PKG_NAME
        assert pyproject["project"]["version"] == __version__
        assert pyproject["project"]["description"] == pkg_data["description"]
    except ImportError:
        pass  # tomllib only available in 3.11+
