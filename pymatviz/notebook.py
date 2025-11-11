"""Notebook integration for automatic rendering of pymatgen objects with pymatviz."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymatviz.widgets.mime import _RENDERER_REGISTRY, _WIDGET_CLASS_TO_KEY, WIDGET_MAP


if TYPE_CHECKING:
    from collections.abc import Callable

    import plotly.graph_objects as go


def _hide_plotly_toolbar(fig: go.Figure) -> None:
    """Configure plotly figure to hide toolbar by default for cleaner display."""
    modebar_remove = (
        "pan select lasso zoomIn zoomOut autoScale resetScale3d".split()  # noqa: SIM905
    )
    fig.update_layout(modebar={"remove": modebar_remove})


def _create_widget_mime_bundle(widget_cls_name: str, obj: Any) -> dict[str, Any]:
    """Create MIME bundle for widget types."""
    # Find the widget key by looking up the widget class name in WIDGET_MAP
    widget_key = _WIDGET_CLASS_TO_KEY.get(widget_cls_name)

    if widget_key is None:
        return {"text/plain": repr(obj)}

    module_name, cls_name, param_name = WIDGET_MAP[widget_key]  # type: ignore[index]
    from importlib import import_module

    module = import_module(module_name)
    widget_class = getattr(module, cls_name)
    widget = widget_class(**{param_name: obj})
    mime_bundle = widget._repr_mimebundle_()
    if isinstance(mime_bundle, tuple):
        mime_bundle = mime_bundle[0]
    result = mime_bundle.copy()
    if "text/plain" not in result:
        result["text/plain"] = repr(obj)
    return result


def _render_mime(obj: Any, plot_func_name: str | None = None) -> dict[str, Any]:
    """Render object to MIME bundle using registry lookup or specific plot function."""
    try:
        if plot_func_name is None:
            # Use registry lookup
            renderer = _RENDERER_REGISTRY.get(type(obj))
            if renderer is None:
                return {"text/plain": repr(obj)}
            if isinstance(renderer, str) and renderer.endswith("Widget"):
                return _create_widget_mime_bundle(renderer, obj)
            if (
                hasattr(renderer, "__name__")
                and isinstance(renderer.__name__, str)
                and renderer.__name__.endswith("Widget")
            ):
                return _create_widget_mime_bundle(renderer.__name__, obj)
            fig = renderer(obj)  # type: ignore[operator]
            _hide_plotly_toolbar(fig)
            return {
                "application/vnd.plotly.v1+json": fig.to_plotly_json(),
                "text/plain": repr(obj),
            }
        # Use specific plot function
        import pymatviz as pmv

        plot_func = getattr(pmv, plot_func_name)
        fig = plot_func(obj)
        _hide_plotly_toolbar(fig)
        return {
            "application/vnd.plotly.v1+json": fig.to_plotly_json(),
            "text/plain": repr(obj),
        }
    except ImportError:
        return {"text/plain": repr(obj)}


def _create_display_methods(
    plot_func_name: str | None = None,
) -> tuple[Callable[..., None], Callable[..., dict[str, Any]]]:
    """Create IPython display and MIME bundle methods for a given plot function or
    registry lookup.
    """

    def ipython_display(self: Any) -> None:
        """Generic IPython display method."""
        from IPython.display import publish_display_data

        display_data = _render_mime(self, plot_func_name)
        publish_display_data(display_data)

    def repr_mimebundle(
        self: Any,
        include: tuple[str, ...] = (),  # noqa: ARG001
        exclude: tuple[str, ...] = (),  # noqa: ARG001
    ) -> dict[str, Any]:
        """Generic MIME bundle method."""
        return _render_mime(self, plot_func_name)

    return ipython_display, repr_mimebundle


# Create display methods for each object type
_structure_ipython_display_, _structure_repr_mimebundle_ = _create_display_methods()
_ase_atoms_ipython_display_, _ase_atoms_repr_mimebundle_ = _create_display_methods()
_diffraction_pattern_ipython_display_, _diffraction_pattern_repr_mimebundle_ = (
    _create_display_methods("xrd_pattern")
)
_phonon_bands_ipython_display_, _phonon_bands_repr_mimebundle_ = (
    _create_display_methods("phonon_bands")
)
_phonon_dos_ipython_display_, _phonon_dos_repr_mimebundle_ = _create_display_methods(
    "phonon_dos"
)
_phonopy_dos_ipython_display_, _phonopy_dos_repr_mimebundle_ = _create_display_methods(
    "phonon_dos"
)


def notebook_mode(*, on: bool) -> None:
    """Enable or disable pymatviz notebook display for pymatgen classes.

    This function adds or removes IPython display methods to/from various pymatgen
    classes so that when these objects are returned from a notebook cell, they will
    automatically render as interactive plots using the appropriate pymatviz functions.

    Works in Jupyter notebooks, JupyterLab, Marimo, VSCode interactive windows,
    and other notebook environments that support IPython display protocols.

    Args:
        on (bool): If True, enable automatic rendering.

    Supported classes:
    - Structure -> StructureWidget or structure_3d (configurable via
      pymatviz.widgets.mime.set_renderer)
    - ASE Atoms -> StructureWidget or structure_3d (configurable as above)
    - PhonopyAtoms -> StructureWidget or structure_3d (configurable as above)
    - Composition -> CompositionWidget (configurable as above)
    - PhononBandStructureSymmLine -> phonon_bands
    - PhononDos -> phonon_dos
    - phonopy TotalDos -> phonon_dos
    - DiffractionPattern -> xrd_pattern
    """
    class_configs = [  # Define (import_func, class_obj, display_methods)
        (
            lambda: __import__("pymatgen.core", fromlist=["Structure"]).Structure,  # type: ignore[unresolved-attribute]
            _structure_ipython_display_,
            _structure_repr_mimebundle_,
        ),
        (
            lambda: __import__("ase.atoms", fromlist=["Atoms"]).Atoms,  # type: ignore[unresolved-attribute]
            _ase_atoms_ipython_display_,
            _ase_atoms_repr_mimebundle_,
        ),
        (
            lambda: __import__(
                "phonopy.structure.atoms", fromlist=["PhonopyAtoms"]
            ).PhonopyAtoms,  # type: ignore[unresolved-attribute]
            _structure_ipython_display_,
            _structure_repr_mimebundle_,
        ),
        (
            lambda: __import__("pymatgen.core", fromlist=["Composition"]).Composition,  # type: ignore[unresolved-attribute]
            _structure_ipython_display_,
            _structure_repr_mimebundle_,
        ),
        (
            lambda: __import__(
                "pymatgen.analysis.diffraction.xrd", fromlist=["DiffractionPattern"]
            ).DiffractionPattern,  # type: ignore[unresolved-attribute]
            _diffraction_pattern_ipython_display_,
            _diffraction_pattern_repr_mimebundle_,
        ),
        (
            lambda: __import__(
                "pymatgen.phonon.bandstructure",
                fromlist=["PhononBandStructureSymmLine"],
            ).PhononBandStructureSymmLine,  # type: ignore[unresolved-attribute]
            _phonon_bands_ipython_display_,
            _phonon_bands_repr_mimebundle_,
        ),
        (
            lambda: __import__("pymatgen.phonon.dos", fromlist=["PhononDos"]).PhononDos,  # type: ignore[unresolved-attribute]
            _phonon_dos_ipython_display_,
            _phonon_dos_repr_mimebundle_,
        ),
        (
            lambda: __import__("phonopy.phonon.dos", fromlist=["TotalDos"]).TotalDos,  # type: ignore[unresolved-attribute]
            _phonopy_dos_ipython_display_,
            _phonopy_dos_repr_mimebundle_,
        ),
    ]

    for import_func, ipython_display, repr_mimebundle in class_configs:
        try:
            cls = import_func()
            if on:
                cls._ipython_display_ = ipython_display
                cls._repr_mimebundle_ = repr_mimebundle
            else:
                if hasattr(cls, "_ipython_display_"):
                    del cls._ipython_display_
                if hasattr(cls, "_repr_mimebundle_"):
                    del cls._repr_mimebundle_
        except (ImportError, AttributeError):
            pass  # ImportError: Module not available
            # AttributeError: when using crystal_toolkit, _ipython_display_ is monkey
            # patched to MSONable (https://github.com/janosh/pymatviz/pull/324)
