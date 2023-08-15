from __future__ import annotations

from importlib.metadata import version

from pymatviz import __version__


def test_version() -> None:
    assert __version__ == version("pymatviz")


def test_convenience_exports() -> None:
    import pymatviz

    assert len(pymatviz.__dict__) >= 52
