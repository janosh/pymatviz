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
    objects: dict[str, Any] = {"structure": structures[0]}

    if ase_atoms is not None:
        objects["ase_atoms"] = ase_atoms[0]
    else:
        try:
            from ase.build import bulk

            objects["ase_atoms"] = bulk("Si", "diamond", a=5.43)
        except ImportError:
            objects["ase_atoms"] = None

    try:
        from pymatgen.analysis.diffraction.xrd import DiffractionPattern

        objects["diffraction_pattern"] = DiffractionPattern(
            x=[10, 20, 30],
            y=[100, 50, 75],
            hkls=[[{"hkl": (1, 0, 0)}], [{"hkl": (1, 1, 0)}], [{"hkl": (1, 1, 1)}]],
            d_hkls=[2.5, 2.0, 1.8],
        )
    except ImportError:
        objects["diffraction_pattern"] = None

    return objects


@pytest.mark.parametrize("enable", [True, False])
def test_notebook_mode_toggle(enable: bool) -> None:
    """Test enabling and disabling notebook integration."""
    pmv.notebook_mode(on=not enable)
    pmv.notebook_mode(on=enable)

    assert hasattr(Structure, "_ipython_display_") == enable
    assert hasattr(Structure, "_repr_mimebundle_") == enable

    if enable:
        assert callable(Structure._ipython_display_)
        assert callable(Structure._repr_mimebundle_)

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


def test_multiple_enable_disable_cycles() -> None:
    """Test that repeated enable/disable cycles are idempotent."""
    pmv.notebook_mode(on=False)
    assert not hasattr(Structure, "_ipython_display_")

    for _ in range(3):
        pmv.notebook_mode(on=True)
        assert hasattr(Structure, "_ipython_display_")
        assert hasattr(Structure, "_repr_mimebundle_")

        pmv.notebook_mode(on=False)
        assert not hasattr(Structure, "_ipython_display_")
        assert not hasattr(Structure, "_repr_mimebundle_")


@pytest.mark.parametrize(
    ("obj_name", "should_skip"),
    [("structure", False), ("ase_atoms", True), ("diffraction_pattern", True)],
)
def test_object_notebook_display(
    test_objects: dict[str, Any],
    obj_name: str,
    should_skip: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _ipython_display_ and _repr_mimebundle_ for supported types."""
    obj = test_objects[obj_name]
    if should_skip and obj is None:
        pytest.skip(f"{obj_name} not available")

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


def test_widget_mime_bundle() -> None:
    """Test _widget_mime_bundle produces widget MIME for registered types."""
    struct = Structure(Lattice.cubic(3), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    result = notebook._widget_mime_bundle(struct)
    assert "text/plain" in result
    assert "application/vnd.jupyter.widget-view+json" in result


def test_widget_mime_bundle_unregistered() -> None:
    """Test _widget_mime_bundle falls back for unregistered types."""

    class UnknownType:
        pass

    result = notebook._widget_mime_bundle(UnknownType())
    assert list(result.keys()) == ["text/plain"]
    assert "UnknownType" in result["text/plain"]


def test_fallback_without_pymatviz(
    structures: tuple[Structure, Structure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test graceful fallback when pymatviz import fails."""
    original_import = __import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pymatviz":
            raise ImportError("No module named 'pymatviz'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    published_data: list[dict[str, str]] = []
    monkeypatch.setattr(
        "IPython.display.publish_display_data",
        published_data.append,
    )

    pmv.notebook_mode(on=True)
    try:
        structures[0]._ipython_display_()
        assert len(published_data) == 1
        assert "text/plain" in published_data[0]

        mime = structures[0]._repr_mimebundle_()
        assert "text/plain" in mime
    finally:
        pmv.notebook_mode(on=False)


@pytest.mark.parametrize(
    "import_error_modules",
    [
        ["pymatgen.core"],
        ["ase.atoms"],
        ["pymatgen.analysis.diffraction.xrd"],
        ["pymatgen.phonon.bandstructure"],
        ["pymatgen.phonon.dos"],
        ["phonopy.phonon.dos"],
        ["ase.atoms", "pymatgen.phonon.dos"],
    ],
)
def test_import_error_handling(
    import_error_modules: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test notebook_mode when various imports fail."""
    original_import = __import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in import_error_modules:
            raise ImportError(f"No module with {name=}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)
    pmv.notebook_mode(on=True)
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
        phonopy_nacl.total_dos._ipython_display_()
        assert len(published_data) == 1
        assert "text/plain" in published_data[0]
    finally:
        pmv.notebook_mode(on=False)
