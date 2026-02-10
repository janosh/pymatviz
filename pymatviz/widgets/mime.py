"""Auto-display registration for pymatgen objects as MatterViz widgets."""

from __future__ import annotations

from typing import Any

from pymatviz.widgets.band_structure import BandStructureWidget
from pymatviz.widgets.composition import CompositionWidget
from pymatviz.widgets.convex_hull import ConvexHullWidget
from pymatviz.widgets.dos import DosWidget
from pymatviz.widgets.structure import StructureWidget
from pymatviz.widgets.trajectory import TrajectoryWidget
from pymatviz.widgets.xrd import XrdWidget


# Direct mapping: pymatgen class -> (widget class, constructor kwarg name)
_AUTO_DISPLAY: dict[type, tuple[type, str]] = {}

# (module_path, class_names, widget_class, param_name) for deferred import.
# Note: PhaseDiagramWidget is NOT auto-registered here — it visualizes isobaric
# binary phase diagrams (T vs x), NOT pymatgen's PhaseDiagram (energy convex hull).
# Use ConvexHullWidget for pymatgen PhaseDiagram objects.
_CLASS_SPECS: list[tuple[str, list[str], type, str]] = [
    # Structures (pymatgen, ASE, phonopy)
    (
        "pymatgen.core",
        ["Structure", "IStructure", "Molecule", "IMolecule"],
        StructureWidget,
        "structure",
    ),
    ("ase.atoms", ["Atoms"], StructureWidget, "structure"),
    ("phonopy.structure.atoms", ["PhonopyAtoms"], StructureWidget, "structure"),
    # Band structures
    (
        "pymatgen.electronic_structure.bandstructure",
        ["BandStructure", "BandStructureSymmLine"],
        BandStructureWidget,
        "band_structure",
    ),
    (
        "pymatgen.phonon.bandstructure",
        ["PhononBandStructureSymmLine"],
        BandStructureWidget,
        "band_structure",
    ),
    # DOS (pymatgen electronic + phonon, phonopy)
    ("pymatgen.electronic_structure.dos", ["Dos", "CompleteDos"], DosWidget, "dos"),
    ("pymatgen.phonon.dos", ["PhononDos"], DosWidget, "dos"),
    ("phonopy.phonon.dos", ["TotalDos"], DosWidget, "dos"),
    # XRD
    (
        "pymatgen.analysis.diffraction.xrd",
        ["DiffractionPattern"],
        XrdWidget,
        "patterns",
    ),
    # PhaseDiagram -> ConvexHull
    ("pymatgen.analysis.phase_diagram", ["PhaseDiagram"], ConvexHullWidget, "entries"),
    # Composition
    ("pymatgen.core", ["Composition"], CompositionWidget, "composition"),
]


def create_widget(obj: Any) -> Any:
    """Create the appropriate MatterViz widget for a pymatgen/ASE/phonopy object.

    Uses MRO-based type matching, so subclasses of registered types are handled
    automatically. Falls back to trajectory detection for list/tuple of structures.

    Args:
        obj: A pymatgen Structure, Composition, BandStructure, Dos, PhaseDiagram,
            DiffractionPattern, or similar object.

    Returns:
        The appropriate widget instance.

    Raises:
        ValueError: If no widget is registered for the given object type.
    """
    if not _AUTO_DISPLAY:
        register_matterviz_widgets()

    # MRO-based lookup: walks the inheritance chain for exact or parent match
    for cls in type(obj).__mro__:
        if cls in _AUTO_DISPLAY:
            widget_cls, param_name = _AUTO_DISPLAY[cls]
            return widget_cls(**{param_name: obj})

    # Fallback: trajectory heuristic (list/tuple of structures or dicts)
    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            raise ValueError("Cannot create widget from empty sequence")
        first = obj[0]
        is_struct = any(
            _AUTO_DISPLAY.get(cls, (None,))[0] is StructureWidget
            for cls in type(first).__mro__
        )
        is_frame_dict = isinstance(first, dict) and "structure" in first
        if is_struct or is_frame_dict:
            return TrajectoryWidget(trajectory=obj)

    raise ValueError(
        f"No widget registered for {type(obj).__name__}. Supported types: "
        f"{', '.join(cls.__name__ for cls in _AUTO_DISPLAY)}"
    )


def register_matterviz_widgets() -> None:
    """Register pymatgen/ASE/phonopy classes for auto-display in notebooks.

    Populates the _AUTO_DISPLAY registry and monkey-patches _display_ on
    registered classes for marimo auto-rendering.
    """
    if _AUTO_DISPLAY:
        return  # already registered

    # Load and register all class specs (silently skip unavailable packages)
    for module_path, class_names, widget_cls, param_name in _CLASS_SPECS:
        try:
            module = __import__(module_path, fromlist=class_names)
        except ImportError:
            continue
        for name in class_names:
            cls = getattr(module, name, None)
            if cls is not None:
                _AUTO_DISPLAY[cls] = (widget_cls, param_name)

    # Register _display_ for marimo auto-rendering
    for cls in _AUTO_DISPLAY:
        if not hasattr(cls, "_display_"):
            cls._display_ = create_widget  # type: ignore[attr-defined]
