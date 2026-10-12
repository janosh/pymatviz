"""Test notebook integration for automatic pymatgen object rendering."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from pymatgen.core import Lattice, Structure


if "IPython" not in sys.modules:
    # Mock IPython module to avoid import errors in CI
    mock_ipython = MagicMock()
    mock_ipython.display.publish_display_data = MagicMock()
    # Set version_info to a high version to satisfy matplotlib's version check
    mock_ipython.version_info = (8, 24, 0)  # >= (8, 24) to pass matplotlib check
    sys.modules["IPython"] = mock_ipython
    sys.modules["IPython.display"] = mock_ipython.display

import pymatviz as pmv
from pymatviz import notebook


@pytest.fixture
def test_objects(
    structures: tuple[Structure, Structure], ase_atoms: tuple[Any, Any]
) -> dict[str, Any]:
    """Create test objects for all supported types using existing fixtures."""
    from pymatgen.analysis.diffraction.xrd import DiffractionPattern

    diffraction_pattern = DiffractionPattern(
        x=[10, 20, 30],
        y=[100, 50, 75],
        hkls=[[{"hkl": (1, 0, 0)}], [{"hkl": (1, 1, 0)}], [{"hkl": (1, 1, 1)}]],
        d_hkls=[2.5, 2.0, 1.8],
    )
    return {
        "structure": structures[0],
        "ase_atoms": ase_atoms[0],
        "diffraction_pattern": diffraction_pattern,
    }


@pytest.mark.parametrize("enable", [True, False])
def test_notebook_mode_toggle(enable: bool) -> None:
    """Test enabling and disabling notebook integration."""
    pmv.notebook_mode(on=not enable)
    pmv.notebook_mode(on=enable)

    assert hasattr(Structure, "_ipython_display_") == enable
    assert hasattr(Structure, "_repr_mimebundle_") == enable

    if enable:
        assert callable(Structure._ipython_display_)  # ty: ignore[unresolved-attribute]
        assert callable(Structure._repr_mimebundle_)  # ty: ignore[unresolved-attribute]

    # Check optional classes
    for module_name, class_name in [
        ("ase.atoms", "Atoms"),
        ("pymatgen.analysis.diffraction.xrd", "DiffractionPattern"),
        ("pymatgen.phonon.bandstructure", "PhononBandStructureSymmLine"),
        ("pymatgen.phonon.dos", "PhononDos"),
        ("phonopy.phonon.dos", "TotalDos"),
    ]:
        try:
            cls = getattr(__import__(module_name, fromlist=[class_name]), class_name)
            assert hasattr(cls, "_ipython_display_") == enable
            assert hasattr(cls, "_repr_mimebundle_") == enable
        except ImportError:
            pass

    pmv.notebook_mode(on=False)


@pytest.mark.parametrize("attr", ["_ipython_display_", "_repr_mimebundle_"])
def test_notebook_mode_preserves_external_hooks(
    monkeypatch: pytest.MonkeyPatch, attr: str
) -> None:
    """Disabling restores prior hooks without overwriting third-party replacements."""
    pmv.notebook_mode(on=False)
    original = object()
    monkeypatch.setattr(Structure, attr, original, raising=False)
    pmv.notebook_mode(on=False)
    assert getattr(Structure, attr) is original
    try:
        pmv.notebook_mode(on=True)
        pmv.notebook_mode(on=True)
        pmv.notebook_mode(on=False)
        assert getattr(Structure, attr) is original

        pmv.notebook_mode(on=True)
        replacement = object()
        setattr(Structure, attr, replacement)
        pmv.notebook_mode(on=False)
        assert getattr(Structure, attr) is replacement
    finally:
        pmv.notebook_mode(on=False)


@pytest.mark.parametrize("obj_name", ["structure", "ase_atoms", "diffraction_pattern"])
def test_object_notebook_display(
    test_objects: dict[str, Any], obj_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test _ipython_display_ and _repr_mimebundle_ for supported types."""
    obj = test_objects[obj_name]
    pmv.notebook_mode(on=True)
    published_data: list[dict[str, str]] = []
    monkeypatch.setattr(
        "IPython.display.publish_display_data",
        published_data.append,
    )

    try:
        # Test _ipython_display_
        obj._ipython_display_()
        assert len(published_data) == 1
        assert "text/plain" in published_data[0]

        # Test _repr_mimebundle_
        mime_bundle = obj._repr_mimebundle_()
        assert isinstance(mime_bundle, dict)
        assert "text/plain" in mime_bundle
    finally:
        pmv.notebook_mode(on=False)


class UnknownType:
    """Type without a registered MatterViz widget."""


@pytest.mark.parametrize(
    ("obj", "has_widget"),
    [
        (Structure(Lattice.cubic(3), ["Fe", "Fe"], [[0, 0, 0], [0.5] * 3]), True),
        (UnknownType(), False),
    ],
)
def test_widget_mime_bundle(obj: object, has_widget: bool) -> None:
    """Registered types get widget MIME, others fall back to plain text."""
    result = notebook._widget_mime_bundle(obj)
    assert "text/plain" in result
    assert ("application/vnd.jupyter.widget-view+json" in result) is has_widget
    if not has_widget:
        assert result == {"text/plain": repr(obj)}


def test_fallback_without_widgets(
    structures: tuple[Structure, Structure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fall back to plain text when the widget module fails to import."""
    original_import = __import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("pymatviz.widgets"):
            raise ImportError(f"No module named {name!r}")
        return original_import(name, *args, **kwargs)

    published_data: list[dict[str, str]] = []
    monkeypatch.setattr(
        "IPython.display.publish_display_data",
        published_data.append,
    )

    pmv.notebook_mode(on=True)
    monkeypatch.setattr("builtins.__import__", mock_import)
    try:
        expected = {"text/plain": repr(structures[0])}
        structures[0]._ipython_display_()  # ty: ignore[unresolved-attribute]
        assert published_data == [expected]
        assert structures[0]._repr_mimebundle_() == expected  # ty: ignore[unresolved-attribute]
    finally:
        monkeypatch.undo()
        pmv.notebook_mode(on=False)


def test_phonopy_dos_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test phonopy TotalDos notebook integration."""
    pytest.importorskip("phonopy", reason="phonopy not available")

    from pymatviz.utils.testing import load_phonopy_nacl

    phonopy_nacl = load_phonopy_nacl()
    phonopy_nacl.run_mesh([10, 10, 10])
    phonopy_nacl.run_total_dos()

    pmv.notebook_mode(on=True)
    published_data: list[dict[str, str]] = []
    monkeypatch.setattr(
        "IPython.display.publish_display_data",
        published_data.append,
    )

    try:
        from phonopy.phonon.dos import TotalDos

        assert hasattr(TotalDos, "_ipython_display_")
        phonopy_nacl.total_dos._ipython_display_()  # ty: ignore[unresolved-attribute]
        assert len(published_data) == 1
        assert "text/plain" in published_data[0]
        assert "application/vnd.jupyter.widget-view+json" in published_data[0]
    finally:
        pmv.notebook_mode(on=False)
