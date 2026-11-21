"""Display pymatgen, ASE and phonopy objects as MatterViz widgets in notebooks."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING, Any

from pymatviz.widgets.band_structure import BandStructureWidget
from pymatviz.widgets.composition import CompositionWidget
from pymatviz.widgets.convex_hull import ConvexHullWidget
from pymatviz.widgets.dos import DosWidget
from pymatviz.widgets.structure import StructureWidget
from pymatviz.widgets.trajectory import TrajectoryWidget
from pymatviz.widgets.xrd import XrdWidget


if TYPE_CHECKING:
    from pymatviz.widgets.matterviz import MatterVizWidget


# (module defining the classes, class names, widget class, constructor kwarg), matched
# by dotted class name so nothing is imported until such an object is displayed.
# pymatgen's PhaseDiagram is an energy convex hull, so it maps to ConvexHullWidget.
_DISPLAY_SPECS: tuple[tuple[str, tuple[str, ...], type[MatterVizWidget], str], ...] = (
    (
        "pymatgen.core.structure",
        ("Structure", "IStructure", "Molecule", "IMolecule"),
        StructureWidget,
        "structure",
    ),
    ("ase.atoms", ("Atoms",), StructureWidget, "structure"),
    ("phonopy.structure.atoms", ("PhonopyAtoms",), StructureWidget, "structure"),
    (
        "pymatgen.electronic_structure.bandstructure",
        ("BandStructure", "BandStructureSymmLine"),
        BandStructureWidget,
        "band_structs",
    ),
    (
        "pymatgen.phonon.bandstructure",
        ("PhononBandStructureSymmLine",),
        BandStructureWidget,
        "band_structs",
    ),
    ("pymatgen.electronic_structure.dos", ("Dos", "CompleteDos"), DosWidget, "doses"),
    ("pymatgen.phonon.dos", ("PhononDos",), DosWidget, "doses"),
    ("phonopy.phonon.dos", ("TotalDos",), DosWidget, "doses"),
    (
        "pymatgen.analysis.diffraction.core",
        ("DiffractionPattern",),
        XrdWidget,
        "patterns",
    ),
    ("pymatgen.analysis.phase_diagram", ("PhaseDiagram",), ConvexHullWidget, "entries"),
    ("pymatgen.core.composition", ("Composition",), CompositionWidget, "composition"),
)
_WIDGET_OF = {
    f"{module}.{name}": (widget_cls, param)
    for module, names, widget_cls, param in _DISPLAY_SPECS
    for name in names
}
# Printers notebook_mode replaced in IPython, restored when it is turned off
_previous_printers: dict[tuple[str, str], Any] = {}


def _display_spec(obj: Any) -> tuple[type[MatterVizWidget], str] | None:
    """Widget class and constructor kwarg for obj's class or its nearest base."""
    for cls in type(obj).__mro__:
        if spec := _WIDGET_OF.get(f"{cls.__module__}.{cls.__qualname__}"):
            return spec
    return None


def create_widget(obj: Any) -> MatterVizWidget:
    """MatterViz widget for a pymatgen/ASE/phonopy object (subclasses included), or
    a trajectory widget for a sequence of structures or frame dicts.

    Raises:
        ValueError: If no widget displays obj's type, or obj is an empty sequence.
    """
    if spec := _display_spec(obj):
        widget_cls, param = spec
        return widget_cls(**{param: obj})
    if isinstance(obj, (list, tuple)):
        if not obj:
            raise ValueError("Cannot create widget from empty sequence")
        first_spec = _display_spec(obj[0])
        is_struct = first_spec is not None and first_spec[0] is StructureWidget
        if is_struct or (isinstance(obj[0], dict) and "structure" in obj[0]):
            return TrajectoryWidget(trajectory=obj)
    raise ValueError(
        f"No widget registered for {type(obj).__name__}. Supported types: "
        f"{', '.join(name.rsplit('.', 1)[1] for name in _WIDGET_OF)}"
    )


def _widget_mime_bundle(obj: Any) -> dict[str, Any] | tuple[dict, dict] | None:
    """MIME bundle of the MatterViz widget for a registered object. Errors propagate:
    IPython shows the traceback and falls back to the object's own repr.
    """
    return create_widget(obj)._repr_mimebundle_()


def _set_marimo_display(
    module_path: str, class_names: tuple[str, ...], *, on: bool
) -> None:
    """Add (or remove) create_widget as _display_ on the importable classes."""
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        return
    for cls in filter(None, (getattr(module, name, None) for name in class_names)):
        if on and not hasattr(cls, "_display_"):
            cls._display_ = create_widget  # ty: ignore[invalid-assignment]
        elif not on and vars(cls).get("_display_") is create_widget:
            del cls._display_


def notebook_mode(*, on: bool) -> None:
    """Enable or disable rendering pymatgen/ASE/phonopy objects as MatterViz widgets.

    In IPython (Jupyter, JupyterLab, VS Code interactive windows), the types are
    registered by name with the MIME-bundle formatter, so nothing is imported until
    such an object is displayed. In marimo, their importable classes get a
    ``_display_`` method. Importing pymatviz turns this on in either.

    Args:
        on: If True, enable automatic rendering. If False, restore prior printers
            and remove the ``_display_`` methods.

    Raises:
        RuntimeError: Outside an IPython session and marimo.
    """
    from IPython import get_ipython

    shell = get_ipython()
    in_marimo = "marimo" in sys.modules
    if shell is None and not in_marimo:
        raise RuntimeError("notebook_mode needs a running IPython session or marimo")
    for module_path, class_names, _widget_cls, _param in _DISPLAY_SPECS:
        if in_marimo:
            _set_marimo_display(module_path, class_names, on=on)
        if shell is None:
            continue
        formatter = shell.display_formatter.mimebundle_formatter
        for class_name in class_names:
            key = (module_path, class_name)
            # pop by name also finds printers IPython already resolved to the type
            current = formatter.pop(f"{module_path}.{class_name}", None)
            if on:
                _previous_printers.setdefault(key, current)
                formatter.for_type_by_name(*key, _widget_mime_bundle)
            elif current is not _widget_mime_bundle and current is not None:
                formatter.for_type_by_name(*key, current)  # a third party's, keep it
            elif (previous := _previous_printers.get(key)) is not None:
                formatter.for_type_by_name(*key, previous)
    if not on:
        _previous_printers.clear()
