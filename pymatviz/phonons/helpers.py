"""Plotting functions for pymatgen phonon band structures and density of states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pymatgen.electronic_structure.bandstructure import BandStructureSymmLine
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
from pymatgen.util.string import htmlify


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any, Self, TypeAlias

    import plotly.graph_objects as go
    from phonopy.phonon.band_structure import BandStructure as PhonopyBandStructure
    from pymatgen.core import Structure
    from pymatgen.phonon.dos import PhononDos

AnyBandStructure: TypeAlias = BandStructureSymmLine | PhononBands
YMin: TypeAlias = float | Literal["y_min"]
YMax: TypeAlias = float | Literal["y_max"]


@dataclass
class PhononDBDoc:
    """Dataclass for phonon DB docs."""

    structure: Structure
    phonon_bandstructure: PhononBands
    phonon_dos: PhononDos
    free_energies: list[float]  # vibrational part of free energies per formula unit
    internal_energies: list[float]  # vibrational part of internal energies per f.u.
    heat_capacities: list[float]
    entropies: list[float]
    temps: list[float] | None = None  # temperatures
    # whether imaginary modes are present in the BS
    has_imaginary_modes: bool | None = None
    primitive: Structure | None = None
    supercell: list[list[int]] | None = None  # 3x3 matrix
    # non-analytical corrections based on Born charges
    nac_params: dict[str, Any] | None = None
    thermal_displacement_data: dict[str, Any] | None = None
    mp_id: str | None = None  # material ID
    formula: str | None = None  # chemical formula

    def __new__(cls, **kwargs: Any) -> Self:
        """Ignore unexpected and initialize dataclass with known kwargs."""
        try:
            cls_init = cls.__initializer  # type: ignore[has-type]
        except AttributeError:
            # store original init on the class in a different place
            cls.__initializer = cls_init = cls.__init__
            # replace init with noop to avoid raising on unexpected kwargs
            cls.__init__ = lambda *args, **kwargs: None  # type: ignore[method-assign] # noqa: ARG005

        ret = object.__new__(cls)
        known_kwargs = {
            key: val for key, val in kwargs.items() if key in cls.__annotations__
        }
        cls_init(ret, **known_kwargs)

        return ret


def pretty_sym_point(symbol: str) -> str:
    """Convert a symbol to a pretty-printed version."""
    # htmlify maps S0 -> S<sub>0</sub> but leaves S_0 as is so we remove underscores
    return (
        htmlify(symbol.replace("_", ""))
        .replace("GAMMA", "Γ")
        .replace("DELTA", "Δ")
        .replace("SIGMA", "Σ")
    )


def get_band_xaxis_ticks(
    band_struct: PhononBands, branches: Sequence[str] | set[str] = ()
) -> tuple[list[float], list[str]]:
    """Get all ticks and labels for a band structure plot.

    Returns:
        tuple[list[float], list[str]]: Ticks and labels for the x-axis of a band
            structure plot.
        branches (Sequence[str]): Branches to plot. Defaults to empty tuple, meaning all
            branches are plotted.
    """
    ticks_x_pos: list[float] = []
    tick_labels: list[str] = []
    prev_label = band_struct.qpoints[0].label
    prev_branch = band_struct.branches[0]["name"]

    for idx, point in enumerate(band_struct.qpoints):
        if point.label is None:
            continue

        branch_names = (
            branch["name"]
            for branch in band_struct.branches
            if branch["start_index"] <= idx <= branch["end_index"]
        )
        this_branch = next(branch_names, None)

        if point.label != prev_label and prev_branch != this_branch:
            tick_labels.pop()
            ticks_x_pos.pop()
            tick_labels += [f"{prev_label or ''}|{point.label}"]
            ticks_x_pos += [band_struct.distance[idx]]
        elif this_branch in branches:
            tick_labels += [point.label]
            ticks_x_pos += [band_struct.distance[idx]]

        prev_label = point.label
        prev_branch = this_branch

    tick_labels = list(map(pretty_sym_point, tick_labels))
    return ticks_x_pos, tick_labels


def _shaded_range(
    fig: go.Figure,
    *,
    shaded_ys: dict[tuple[YMin | YMax, YMin | YMax], dict[str, Any]] | bool | None,
) -> go.Figure:
    """Add shaded regions to a figure.

    Args:
        fig (go.Figure): Plotly figure to add shaded regions to
        shaded_ys (dict[tuple[YMin | YMax, YMin | YMax], dict[str, Any]] | bool | None):
            Configuration for shaded regions. Can be:
            - False: No shading
            - None or True: Default shading (0 to y_min, gray at 0.07 opacity)
            - dict: Keys are (y0, y1) tuples and values are kwargs for add_hrect()

    Returns:
        go.Figure: Modified figure with shaded regions added
    """
    if shaded_ys is False:
        return fig

    shade_defaults = dict(layer="below", row="all", col="all")
    y_lim: dict[float | Literal["y_min", "y_max"], Any] = dict(
        zip(("y_min", "y_max"), fig.layout.yaxis.range, strict=True),
    )

    # If shaded_ys is True or None, use default shading
    if shaded_ys is True or shaded_ys is None:
        shaded_ys = {(0, "y_min"): dict(fillcolor="gray", opacity=0.07)}
    elif not isinstance(shaded_ys, dict):
        raise TypeError(f"expect shaded_ys as dict, got {type(shaded_ys).__name__}")

    for (y0, y1), kwargs in shaded_ys.items():
        for y_val in (y0, y1):
            if isinstance(y_val, str) and y_val not in y_lim:
                raise ValueError(f"Invalid {y_val=}, must be one of {[*y_lim]}")
        fig.add_hrect(
            y0=y_lim.get(y0, y0), y1=y_lim.get(y1, y1), **shade_defaults | kwargs
        )

    return fig


def phonopy_to_pymatgen_bands(band_struct: PhonopyBandStructure) -> PhononBands:
    """Convert phonopy BandStructure to pymatgen PhononBandStructureSymmLine.

    Args:
        band_struct (PhonopyBandStructure): Phonopy band structure object

    Returns:
        PhononBands: Converted pymatgen phonon band structure
    """
    import tempfile

    import yaml
    from pymatgen.io.phonopy import get_ph_bs_symm_line_from_dict

    # Write band structure to temporary YAML file
    with tempfile.NamedTemporaryFile() as tmp_file:
        # Use phonopy's band structure YAML writer
        band_struct.write_yaml(filename=tmp_file.name)
        # Load YAML and convert to pymatgen band structure
        with open(tmp_file.name) as file:
            bands_dict = yaml.safe_load(file)
        return get_ph_bs_symm_line_from_dict(bands_dict)
