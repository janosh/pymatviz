"""Test notebook integration for automatic pymatgen object rendering."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
from pymatgen.core import Structure

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


WIDGET_MIME = "application/vnd.jupyter.widget-view+json"


@pytest.fixture
def shell() -> Any:
    """IPython shell with notebook_mode turned off again after each test."""
    from IPython.core.interactiveshell import InteractiveShell

    ipython = InteractiveShell.instance()
    yield ipython
    pmv.notebook_mode(on=False)


@pytest.mark.parametrize("obj_name", ["structure", "ase_atoms", "diffraction_pattern"])
def test_notebook_mode_formats_objects(
    shell: Any, test_objects: dict[str, Any], obj_name: str
) -> None:
    """With notebook_mode on, supported objects format as widgets; off, they don't."""
    obj = test_objects[obj_name]
    pmv.notebook_mode(on=True)
    pmv.notebook_mode(on=True)  # idempotent
    data, _metadata = shell.display_formatter.format(obj)
    assert WIDGET_MIME in data
    pmv.notebook_mode(on=False)
    data, _metadata = shell.display_formatter.format(obj)
    assert WIDGET_MIME not in data


def test_notebook_mode_preserves_third_party_printers(
    shell: Any, structures: tuple[Structure, Structure]
) -> None:
    """Turning off restores a prior printer and keeps one installed meanwhile."""
    formatter = shell.display_formatter.mimebundle_formatter
    key = ("pymatgen.core.structure", "Structure")
    prior = lambda _obj: {"text/plain": "prior"}
    formatter.for_type_by_name(*key, prior)
    pmv.notebook_mode(on=True)
    assert formatter.lookup(structures[0]) is notebook._widget_mime_bundle
    pmv.notebook_mode(on=False)
    assert formatter.lookup(structures[0]) is prior

    pmv.notebook_mode(on=True)
    replacement = lambda _obj: {"text/plain": "replacement"}
    formatter.for_type(Structure, replacement)
    pmv.notebook_mode(on=False)
    assert formatter.lookup(structures[0]) is replacement
    formatter.pop(Structure)


def test_notebook_mode_imports_nothing() -> None:
    """Importing pymatviz in a kernel registers display types by name, importing
    none of the optional or heavy modules they live in.
    """
    code = (
        "import sys\nfrom IPython.core.interactiveshell import InteractiveShell\n"
        "InteractiveShell.instance()\nimport pymatviz as pmv\n"
        "from pymatviz import notebook\n"
        "assert notebook._previous_printers, 'notebook_mode did not run'\n"
        "print(sorted(mod for mod in ('ase', 'pymatgen.phonon.dos', "
        "'pymatgen.analysis.diffraction.core', 'phonopy') if mod in sys.modules))"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_notebook_mode_needs_ipython(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside IPython, notebook_mode raises instead of silently doing nothing."""
    monkeypatch.setattr("IPython.get_ipython", lambda: None)
    with pytest.raises(RuntimeError, match="needs a running IPython session"):
        pmv.notebook_mode(on=True)


class UnknownType:
    """Type without a registered MatterViz widget."""


def test_widget_mime_bundle(structures: tuple[Structure, Structure]) -> None:
    """Registered types get widget MIME; others raise instead of a silent repr."""
    bundle = notebook._widget_mime_bundle(structures[0])
    assert isinstance(bundle, tuple)
    assert WIDGET_MIME in bundle[0]
    with pytest.raises(ValueError, match="No widget"):
        notebook._widget_mime_bundle(UnknownType())


def test_phonopy_dos_integration(shell: Any) -> None:
    """Test phonopy TotalDos notebook integration."""
    pytest.importorskip("phonopy", reason="phonopy not available")

    from pymatviz.utils.testing import load_phonopy_nacl

    phonopy_nacl = load_phonopy_nacl()
    phonopy_nacl.run_mesh([10, 10, 10])
    phonopy_nacl.run_total_dos()

    pmv.notebook_mode(on=True)
    data, _metadata = shell.display_formatter.format(phonopy_nacl.total_dos)
    assert WIDGET_MIME in data
