"""Notebook integration for automatic rendering of pymatgen objects with pymatviz."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable

    import plotly.graph_objects as go


def _hide_plotly_toolbar(fig: go.Figure) -> None:
    """Configure plotly figure to hide toolbar by default for cleaner display."""
    modebar_remove = (
        "pan select lasso zoomIn zoomOut autoScale resetScale3d".split()  # noqa: SIM905
    )
    fig.update_layout(modebar=dict(remove=modebar_remove))


def _create_display_methods(
    plot_func_name: str,
) -> tuple[Callable[..., None], Callable[..., dict[str, str]]]:
    """Create IPython display and MIME bundle methods for a given plot function."""

    def ipython_display(self: Any) -> None:
        """Generic IPython display method."""
        from IPython.display import publish_display_data

        try:
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
    ) -> dict[str, str]:
        """Generic MIME bundle method."""
        try:
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
_structure_ipython_display_, _structure_repr_mimebundle_ = _create_display_methods(
    "structure_3d_plotly"
)
_ase_atoms_ipython_display_, _ase_atoms_repr_mimebundle_ = _create_display_methods(
    "structure_3d_plotly"
)
_diffraction_pattern_ipython_display_, _diffraction_pattern_repr_mimebundle_ = (
    _create_display_methods("xrd_pattern")
)
_phonon_bands_ipython_display_, _phonon_bands_repr_mimebundle_ = (
    _create_display_methods("phonon_bands")
)
_phonon_dos_ipython_display_, _phonon_dos_repr_mimebundle_ = _create_display_methods(
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
        on: If True, enable automatic rendering. If False, disable it.

    Supported classes:
    - Structure -> structure_3d_plotly
    - ASE Atoms -> structure_3d_plotly
    - PhononBandStructureSymmLine -> phonon_bands
    - PhononDos -> phonon_dos
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
