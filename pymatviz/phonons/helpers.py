"""Plotting functions for pymatgen phonon band structures and density of states."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal

from pymatgen.electronic_structure.bandstructure import (
    BandStructure,
    BandStructureSymmLine,
)
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
from pymatgen.util.string import htmlify


if TYPE_CHECKING:
    from typing import Any, Self

    import plotly.graph_objects as go
    from phonopy.phonon.band_structure import BandStructure as PhonopyBandStructure
    from pymatgen.core import Structure
    from pymatgen.phonon.dos import PhononDos

type AnyBandStructure = BandStructure | BandStructureSymmLine | PhononBands
type YMin = float | Literal["y_min"]
type YMax = float | Literal["y_max"]


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Decode document fields, ignoring database metadata outside this schema."""
        from monty.json import MontyDecoder

        decoder = MontyDecoder()
        return cls(
            **{
                field.name: decoder.process_decoded(data[field.name])
                for field in fields(cls)
                if field.init and field.name in data
            }
        )


def pretty_sym_point(symbol: str) -> str:
    """Convert a symbol to a pretty-printed version."""
    # htmlify maps S0 -> S<sub>0</sub> but leaves S_0 as is so we remove underscores
    return (
        htmlify(symbol.replace("_", ""))
        .replace("GAMMA", "Γ")
        .replace("DELTA", "Δ")
        .replace("SIGMA", "Σ")
    )


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

    with tempfile.NamedTemporaryFile() as tmp_file:
        band_struct.write_yaml(filename=tmp_file.name)
        with open(tmp_file.name) as file:
            bands_dict = yaml.safe_load(file)
        return get_ph_bs_symm_line_from_dict(bands_dict)
