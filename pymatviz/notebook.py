"""Notebook integration for automatic rendering of pymatgen objects with pymatviz."""

from __future__ import annotations

from typing import Any


# Printers notebook_mode replaced, restored when it is turned off
_previous_printers: dict[tuple[str, str], Any] = {}


def _widget_mime_bundle(obj: Any) -> dict[str, Any] | tuple[dict, dict] | None:
    """MIME bundle of the MatterViz widget for a registered object. Errors propagate:
    IPython shows the traceback and falls back to the object's own repr.
    """
    from pymatviz.widgets.mime import create_widget

    return create_widget(obj)._repr_mimebundle_()


def notebook_mode(*, on: bool) -> None:
    """Enable or disable pymatviz notebook display for pymatgen/ASE/phonopy objects.

    When enabled, those objects returned from IPython cells (Jupyter, JupyterLab,
    VS Code interactive windows) render as interactive MatterViz widgets. Types are
    registered by name with IPython's MIME-bundle formatter, so nothing is imported
    until such an object is displayed. marimo renders them via ``_display_``.

    Args:
        on: If True, enable automatic rendering. If False, restore prior printers.

    Raises:
        RuntimeError: Outside an IPython session.
    """
    from IPython import get_ipython

    from pymatviz.widgets.mime import _CLASS_SPECS

    if (shell := get_ipython()) is None:
        raise RuntimeError("notebook_mode needs a running IPython session")
    formatter = shell.display_formatter.mimebundle_formatter
    for module_path, class_names, _widget_cls, _param in _CLASS_SPECS:
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
