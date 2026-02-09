"""Notebook integration for automatic rendering of pymatgen objects with pymatviz."""

from __future__ import annotations

from typing import Any


def _widget_mime_bundle(obj: Any) -> dict[str, Any]:
    """Create a MIME bundle by wrapping the object in a MatterViz widget.

    Args:
        obj: A pymatgen/ASE/phonopy object registered for auto-display.

    Returns:
        MIME bundle dict for IPython display.
    """
    try:
        from pymatviz.widgets.mime import create_widget

        widget = create_widget(obj)
        mime_bundle = widget._repr_mimebundle_()
    except Exception:  # noqa: BLE001
        return {"text/plain": repr(obj)}

    if isinstance(mime_bundle, tuple):
        mime_bundle = mime_bundle[0]
    if not isinstance(mime_bundle, dict):
        return {"text/plain": repr(obj)}
    result = mime_bundle.copy()
    if "text/plain" not in result:
        result["text/plain"] = repr(obj)
    return result


def _ipython_display(self: Any) -> None:
    """Generic IPython display method that renders via MatterViz widgets."""
    from IPython.display import publish_display_data

    publish_display_data(_widget_mime_bundle(self))


def _repr_mimebundle(
    self: Any,
    include: tuple[str, ...] = (),  # noqa: ARG001
    exclude: tuple[str, ...] = (),  # noqa: ARG001
) -> dict[str, Any]:
    """Generic MIME bundle method that renders via MatterViz widgets."""
    return _widget_mime_bundle(self)


def notebook_mode(*, on: bool) -> None:
    """Enable or disable pymatviz notebook display for pymatgen classes.

    When enabled, pymatgen objects returned from notebook cells automatically
    render as interactive MatterViz widgets.

    Works in Jupyter notebooks, JupyterLab, Marimo, VSCode interactive windows,
    and other notebook environments that support IPython display protocols.

    Args:
        on: If True, enable automatic rendering. If False, remove display methods.
    """
    from importlib import import_module

    from pymatviz.widgets.mime import _CLASS_SPECS

    # Derive (module, class_name) pairs from the canonical _CLASS_SPECS registry
    for module_path, class_name in (
        (mod, name)
        for mod, names, _widget_cls, _param in _CLASS_SPECS
        for name in names
    ):
        try:
            cls = getattr(import_module(module_path), class_name)
            if on:
                cls._ipython_display_ = _ipython_display
                cls._repr_mimebundle_ = _repr_mimebundle
            else:
                for attr in ("_ipython_display_", "_repr_mimebundle_"):
                    if hasattr(cls, attr):
                        delattr(cls, attr)
        except (ImportError, AttributeError):
            pass  # Module not available or crystal_toolkit conflict
