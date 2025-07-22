"""MIME type handling for automatic structure and trajectory visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, get_args

from pymatviz.widgets.composition import CompositionWidget
from pymatviz.widgets.structure import StructureWidget
from pymatviz.widgets.trajectory import TrajectoryWidget


if TYPE_CHECKING:
    from collections.abc import Callable

WidgetType = Literal["structure", "trajectory", "composition"]
StructureType, TrajectoryType, CompositionType = get_args(WidgetType)
WIDGET_REGISTRY: dict[Callable[[Any], bool], WidgetType] = {}

WIDGET_MAP: dict[WidgetType, tuple[str, str, WidgetType]] = {
    StructureType: ("pymatviz", StructureWidget.__name__, StructureType),
    TrajectoryType: ("pymatviz", TrajectoryWidget.__name__, TrajectoryType),
    CompositionType: ("pymatviz", CompositionWidget.__name__, CompositionType),
}


def create_widget(obj: Any, widget_type: WidgetType | None = None) -> Any:
    """Create widget for object."""
    if widget_type is None:
        for predicate, wt in WIDGET_REGISTRY.items():
            if predicate(obj):
                widget_type = wt
                break
        else:
            raise ValueError(f"No widget type found for {obj=}")

    if widget_type not in get_args(WidgetType):
        raise ValueError(
            f"Unknown {widget_type=}, must be one of {get_args(WidgetType)}"
        )

    module_name, class_name, param_name = WIDGET_MAP[widget_type]
    module = __import__(module_name, fromlist=[class_name])
    widget_class = getattr(module, class_name)
    return widget_class(**{param_name: obj})


def _register_renderers() -> None:
    """Register renderers for all environments."""
    from pymatviz.process_data import STRUCTURE_CLASSES

    classes: list[type] = []  # collect structure classes
    for module_path, class_names in STRUCTURE_CLASSES:
        try:
            module = __import__(module_path, fromlist=class_names)
            classes.extend(
                getattr(module, cls) for cls in class_names if hasattr(module, cls)
            )
        except ImportError:
            continue

    try:  # Register for Jupyter using notebook.py system
        from pymatviz.notebook import set_renderer

        for cls in classes:
            set_renderer(cls, StructureWidget.__name__)
        try:
            from pymatgen.core import Composition

            set_renderer(Composition, CompositionWidget.__name__)
        except ImportError:
            pass
    except ImportError:
        pass

    try:  # Register for Marimo
        for cls in classes:
            if not hasattr(cls, "_display_"):
                cls._display_ = lambda self: create_widget(self)  # type: ignore[attr-defined]
        try:
            from pymatgen.core import Composition

            if not hasattr(Composition, "_display_"):
                Composition._display_ = lambda self: create_widget(self)
        except ImportError:
            pass
    except ImportError:
        pass


def register_matterviz_widgets() -> None:
    """Register widgets for automatic display."""
    from pymatviz import process_data as pd

    WIDGET_REGISTRY[pd.is_trajectory_like] = StructureType
    WIDGET_REGISTRY[pd.is_composition_like] = CompositionType
    WIDGET_REGISTRY[pd.is_structure_like] = StructureType
    _register_renderers()
