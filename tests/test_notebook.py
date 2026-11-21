"""Test notebook integration for automatic pymatgen object rendering."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest
from pymatgen.analysis.diffraction.xrd import DiffractionPattern
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition, Lattice, Structure

import pymatviz as pmv
from pymatviz import notebook
from pymatviz.notebook import create_widget


@pytest.fixture
def test_objects(
    structures: tuple[Structure, Structure], ase_atoms: tuple[Any, Any]
) -> dict[str, Any]:
    """Create test objects for all supported types using existing fixtures."""
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
    """Outside IPython and marimo, notebook_mode raises instead of doing nothing."""
    monkeypatch.setattr("IPython.get_ipython", lambda: None)
    monkeypatch.delitem(sys.modules, "marimo", raising=False)
    with pytest.raises(RuntimeError, match="needs a running IPython session or marimo"):
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


@pytest.mark.parametrize(
    ("obj_factory", "expected_widget", "expected_type"),
    [
        (lambda: Composition("Fe2O3"), "CompositionWidget", "composition"),
        (
            lambda: DiffractionPattern(x=[10], y=[100], hkls=[], d_hkls=[]),
            "XrdWidget",
            "xrd",
        ),
        (
            lambda: PhaseDiagram(
                [
                    PDEntry(Composition("Li"), -1.9),
                    PDEntry(Composition("O"), -4.2),
                    PDEntry(Composition("Li2O"), -14.3),
                ]
            ),
            "ConvexHullWidget",
            "convex_hull",
        ),
        # trajectory fallback for a list of frame dicts
        (
            lambda: (
                [
                    {
                        "structure": Structure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]]),
                        "energy": -1.0,
                    }
                ]
                * 2
            ),
            "TrajectoryWidget",
            "trajectory",
        ),
    ],
    ids=["composition", "xrd", "phase_diagram", "trajectory_dicts"],
)
def test_create_widget(
    obj_factory: Any, expected_widget: str, expected_type: str
) -> None:
    """create_widget produces correct widget for registered types."""
    widget = create_widget(obj_factory())
    assert type(widget).__name__ == expected_widget
    assert widget.widget_type == expected_type


@pytest.mark.parametrize(
    ("bad_input", "match"),
    [(42, "No widget registered"), ([], "empty sequence"), ("string", "No widget")],
)
def test_create_widget_unknown_type_raises(bad_input: Any, match: str) -> None:
    """create_widget raises ValueError for unregistered/empty types."""
    with pytest.raises(ValueError, match=match):
        create_widget(bad_input)


def test_create_widget_matches_subclasses() -> None:
    """Subclasses of registered types display like their base class."""

    class MyStructure(Structure):
        """A pymatgen Structure subclass."""

    widget = create_widget(MyStructure(Lattice.cubic(3), ["Fe"], [[0, 0, 0]]))
    assert widget.widget_type == "structure"


def test_notebook_mode_in_marimo(monkeypatch: pytest.MonkeyPatch) -> None:
    """In marimo, supported classes get a _display_ creating their widget, which
    turning notebook_mode off removes again.
    """
    monkeypatch.setattr("IPython.get_ipython", lambda: None)
    monkeypatch.setitem(sys.modules, "marimo", type(sys)("marimo"))
    pmv.notebook_mode(on=True)
    try:
        widget = Composition("Fe2O3")._display_()  # ty: ignore[missing-attribute]
        assert widget.widget_type == "composition"
    finally:
        pmv.notebook_mode(on=False)
    assert not hasattr(Composition, "_display_")
