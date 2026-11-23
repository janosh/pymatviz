"""Test notebook integration for automatic kemio object rendering."""

from __future__ import annotations

import subprocess
import sys
from typing import Any

import kemio
import numpy as np
import pytest
from kemio.electronic import ElectronicDos

import pymatviz as pmv
from pymatviz import notebook
from pymatviz.notebook import create_widget
from tests.conftest import cubic_structure


@pytest.fixture
def test_objects(structures: tuple[kemio.Structure, kemio.Structure]) -> dict[str, Any]:
    """Create test objects for all supported types using existing fixtures."""
    hkls = [[([1, 0, 0], 1)], [([1, 1, 0], 1)], [([1, 1, 1], 1)]]
    return {
        "structure": structures[0],
        "composition": kemio.Composition("Fe2O3"),
        "xrd_pattern": kemio.XrdPattern(
            [10, 20, 30], [100, 50, 75], [2.5, 2.0, 1.8], hkls
        ),
    }


WIDGET_MIME = "application/vnd.jupyter.widget-view+json"


@pytest.fixture
def shell() -> Any:
    """IPython shell with notebook_mode turned off again after each test."""
    from IPython.core.interactiveshell import InteractiveShell

    ipython = InteractiveShell.instance()
    yield ipython
    pmv.notebook_mode(on=False)


@pytest.mark.parametrize("obj_name", ["structure", "composition", "xrd_pattern"])
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
    shell: Any, structures: tuple[kemio.Structure, kemio.Structure]
) -> None:
    """Turning off restores a prior printer and keeps one installed meanwhile."""
    formatter = shell.display_formatter.mimebundle_formatter
    key = ("kemio", "Structure")
    prior = lambda _obj: {"text/plain": "prior"}
    formatter.for_type_by_name(*key, prior)
    pmv.notebook_mode(on=True)
    assert formatter.lookup(structures[0]) is notebook._widget_mime_bundle
    pmv.notebook_mode(on=False)
    assert formatter.lookup(structures[0]) is prior

    pmv.notebook_mode(on=True)
    replacement = lambda _obj: {"text/plain": "replacement"}
    formatter.for_type(kemio.Structure, replacement)
    pmv.notebook_mode(on=False)
    assert formatter.lookup(structures[0]) is replacement
    formatter.pop(kemio.Structure)


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


class SubclassDos(ElectronicDos):
    """ElectronicDos subclass, which displays like its registered base class."""


def electronic_dos(cls: type[ElectronicDos] = ElectronicDos) -> ElectronicDos:
    """Three-point electronic DOS of the given class."""
    return cls(np.array([-1, 0, 1]), np.array([[0, 1, 0]]), 0.0)


def test_widget_mime_bundle(
    structures: tuple[kemio.Structure, kemio.Structure],
) -> None:
    """Registered types get widget MIME; others raise instead of a silent repr."""
    bundle = notebook._widget_mime_bundle(structures[0])
    assert isinstance(bundle, tuple)
    assert WIDGET_MIME in bundle[0]
    with pytest.raises(ValueError, match="No widget"):
        notebook._widget_mime_bundle(UnknownType())


@pytest.mark.parametrize(
    ("obj_factory", "expected_widget", "expected_type"),
    [
        (lambda: kemio.Composition("Fe2O3"), "CompositionWidget", "composition"),
        (lambda: kemio.XrdPattern([10], [100]), "XrdWidget", "xrd"),
        (
            lambda: kemio.PhaseDiagram(["Li", "O", "Li2O"], [-1.9, -4.2, -14.3 / 3]),
            "ConvexHullWidget",
            "convex_hull",
        ),
        (lambda: kemio.PhononDos([0, 1, 2], [0, 1, 0]), "DosWidget", "dos"),
        (
            lambda: kemio.PhononBands(
                [[0, 0, 0], [0.5, 0, 0]], [[0, 0, 0], [1, 1, 2]], np.eye(3)
            ),
            "BandStructureWidget",
            "band_structure",
        ),
        (electronic_dos, "DosWidget", "dos"),
        (lambda: electronic_dos(SubclassDos), "DosWidget", "dos"),
        (  # trajectory fallback for a list of frame dicts
            lambda: (
                [{"structure": cubic_structure(3, ["Fe"], [0, 0, 0]), "energy": -1.0}]
                * 2
            ),
            "TrajectoryWidget",
            "trajectory",
        ),
    ],
    ids=[
        *(
            "composition",
            "xrd",
            "phase_diagram",
            "kemio_phonon_dos",
            "kemio_phonon_bands",
        ),
        *("kemio_electronic_dos", "electronic_dos_subclass", "trajectory_dicts"),
    ],
)
def test_create_widget(
    obj_factory: Any, expected_widget: str, expected_type: str
) -> None:
    """create_widget picks the widget registered for a type or its base class."""
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


def test_notebook_mode_in_marimo(monkeypatch: pytest.MonkeyPatch) -> None:
    """In marimo, supported classes get a _display_ creating their widget, which
    turning notebook_mode off removes again.
    """
    monkeypatch.setattr("IPython.get_ipython", lambda: None)
    monkeypatch.setitem(sys.modules, "marimo", type(sys)("marimo"))
    pmv.notebook_mode(on=True)
    try:
        widget = kemio.Composition("Fe2O3")._display_()  # ty: ignore[missing-attribute]
        assert widget.widget_type == "composition"
    finally:
        pmv.notebook_mode(on=False)
    assert not hasattr(kemio.Composition, "_display_")
