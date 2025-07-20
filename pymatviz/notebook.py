"""Notebook integration for automatic rendering of pymatgen objects with pymatviz."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pymatviz.widgets.mime import WIDGET_MAP


if TYPE_CHECKING:
    from collections.abc import Callable

    import plotly.graph_objects as go

# Configuration for structure rendering mode
_RENDERER_REGISTRY: dict[type, Callable[..., Any] | str] = {}


def set_renderer(
    cls: type, renderer: Callable[..., Any] | str
) -> Callable[..., Any] | str | None:
    """Set the renderer for a specific class.

    Args:
        cls: The class to register a renderer for (e.g., Structure, Atoms, Composition)
        renderer: The render function to use (e.g., structure_3d, StructureWidget)

    Returns:
        The previous renderer for this class, or None if none was set
    """
    previous = _RENDERER_REGISTRY.get(cls)
    _RENDERER_REGISTRY[cls] = renderer
    return previous


def _hide_plotly_toolbar(fig: go.Figure) -> None:
    """Configure plotly figure to hide toolbar by default for cleaner display."""
    modebar_remove = (
        "pan select lasso zoomIn zoomOut autoScale resetScale3d".split()  # noqa: SIM905
    )
    fig.update_layout(modebar=dict(remove=modebar_remove))


def _create_display_methods(
    plot_func_name: str | None = None,
) -> tuple[Callable[..., None], Callable[..., dict[str, Any]]]:
    """Create IPython display and MIME bundle methods for a given plot function or
    registry lookup.
    """

    def _create_widget_mime_bundle(widget_type: str, obj: Any) -> dict[str, Any]:
        """Create MIME bundle for widget types."""
        # Find the widget type by looking up the widget class name in WIDGET_MAP
        widget_type = ""
        for key, (_module_name, cls_name, _param_name) in WIDGET_MAP.items():
            if cls_name == widget_type:
                widget_type = key
                break

        if widget_type == "":
            return {"text/plain": repr(obj)}

        module_name, cls_name, param_name = WIDGET_MAP[widget_type]  # type: ignore[index]
        module = __import__(module_name, fromlist=[cls_name])
        widget_class = getattr(module, cls_name)
        widget = widget_class(**{param_name: obj})
        mime_bundle = widget._repr_mimebundle_()
        if isinstance(mime_bundle, tuple):
            mime_bundle = mime_bundle[0]
        result = mime_bundle.copy()
        if "text/plain" not in result:
            result["text/plain"] = repr(obj)
        return result

    def ipython_display(self: Any) -> None:
        """Generic IPython display method."""
        from IPython.display import publish_display_data

        try:
            if plot_func_name is None:
                # Use registry lookup
                renderer = _RENDERER_REGISTRY.get(type(self))
                if renderer is None:
                    display_data = {"text/plain": repr(self)}
                elif isinstance(renderer, str) and renderer.endswith("Widget"):
                    display_data = _create_widget_mime_bundle(renderer, self)
                elif hasattr(renderer, "__name__") and renderer.__name__.endswith(
                    "Widget"
                ):
                    display_data = _create_widget_mime_bundle(renderer.__name__, self)
                else:
                    fig = renderer(self)  # type: ignore[operator]
                    _hide_plotly_toolbar(fig)
                    display_data = {
                        "application/vnd.plotly.v1+json": fig.to_plotly_json(),
                        "text/plain": repr(self),
                    }
            else:
                # Use specific plot function
                import pymatviz as pmv

                plot_func = getattr(pmv, plot_func_name)
                fig = plot_func(self)
                _hide_plotly_toolbar(fig)
                display_data = {
                    "application/vnd.plotly.v1+json": fig.to_plotly_json(),
                    "text/plain": repr(self),
                }
        except ImportError:
            display_data = {"text/plain": repr(self)}

        publish_display_data(display_data)

    def repr_mimebundle(
        self: Any,
        include: tuple[str, ...] = (),  # noqa: ARG001
        exclude: tuple[str, ...] = (),  # noqa: ARG001
    ) -> dict[str, Any]:
        """Generic MIME bundle method."""
        try:
            if plot_func_name is None:
                # Use registry lookup
                renderer = _RENDERER_REGISTRY.get(type(self))
                if renderer is None:
                    return {"text/plain": repr(self)}
                if isinstance(renderer, str) and renderer.endswith("Widget"):
                    return _create_widget_mime_bundle(renderer, self)
                if hasattr(renderer, "__name__") and renderer.__name__.endswith(
                    "Widget"
                ):
                    return _create_widget_mime_bundle(renderer.__name__, self)
                fig = renderer(self)  # type: ignore[operator]
                _hide_plotly_toolbar(fig)
                return {
                    "application/vnd.plotly.v1+json": fig.to_plotly_json(),
                    "text/plain": repr(self),
                }
            # Use specific plot function
            import pymatviz as pmv

            plot_func = getattr(pmv, plot_func_name)
            fig = plot_func(self)
            _hide_plotly_toolbar(fig)
            return {
                "application/vnd.plotly.v1+json": fig.to_plotly_json(),
                "text/plain": repr(self),
            }
        except ImportError:
            return {"text/plain": repr(self)}

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
    - Structure -> StructureWidget or structure_3d (configurable via set_renderer)
    - ASE Atoms -> StructureWidget or structure_3d (configurable via set_renderer)
    - PhonopyAtoms -> StructureWidget or structure_3d (configurable via set_renderer)
    - Composition -> CompositionWidget (configurable via set_renderer)
    - PhononBandStructureSymmLine -> phonon_bands
    - PhononDos -> phonon_dos
    - phonopy TotalDos -> phonon_dos
    - DiffractionPattern -> xrd_pattern
    """
    class_configs = [  # Define (import_func, class_obj, display_methods)
        (
            lambda: __import__("pymatgen.core", fromlist=["Structure"]).Structure,
            _structure_ipython_display_,
            _structure_repr_mimebundle_,
        ),
        (
            lambda: __import__("ase.atoms", fromlist=["Atoms"]).Atoms,
            _ase_atoms_ipython_display_,
            _ase_atoms_repr_mimebundle_,
        ),
        (
            lambda: __import__(
                "phonopy.structure.atoms", fromlist=["PhonopyAtoms"]
            ).PhonopyAtoms,
            _structure_ipython_display_,
            _structure_repr_mimebundle_,
        ),
        (
            lambda: __import__("pymatgen.core", fromlist=["Composition"]).Composition,
            _structure_ipython_display_,
            _structure_repr_mimebundle_,
        ),
        (
            lambda: __import__(
                "pymatgen.analysis.diffraction.xrd", fromlist=["DiffractionPattern"]
            ).DiffractionPattern,
            _diffraction_pattern_ipython_display_,
            _diffraction_pattern_repr_mimebundle_,
        ),
        (
            lambda: __import__(
                "pymatgen.phonon.bandstructure",
                fromlist=["PhononBandStructureSymmLine"],
            ).PhononBandStructureSymmLine,
            _phonon_bands_ipython_display_,
            _phonon_bands_repr_mimebundle_,
        ),
        (
            lambda: __import__("pymatgen.phonon.dos", fromlist=["PhononDos"]).PhononDos,
            _phonon_dos_ipython_display_,
            _phonon_dos_repr_mimebundle_,
        ),
        (
            lambda: __import__("phonopy.phonon.dos", fromlist=["TotalDos"]).TotalDos,
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
        except ImportError:
            pass  # Module not available
