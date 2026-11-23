"""Phonon density of states plots (with element/site projections) and phonon
database helpers. Plot band structures with BandStructureWidget or
BandsAndDosWidget, which take kemio, pymatgen and phonopy objects and vasprun.xml
paths directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Literal

import kemio
from kemio.interop import decode_monty, to_phonon_bands, to_phonon_dos, to_structure

from pymatviz.structures import site_symbols
from pymatviz.widgets.dos import DosWidget


if TYPE_CHECKING:
    from typing import Self

    import numpy as np
    from kemio.results import PhononDosDict
    from numpy.typing import NDArray

    from pymatviz.typing import AnyDos


@dataclass
class PhononDBDoc:
    """Phonon database document: structure, phonon band structure and DOS as
    kemio objects, plus thermal properties.
    """

    structure: kemio.Structure
    phonon_bandstructure: kemio.PhononBands
    phonon_dos: kemio.PhononDos
    free_energies: list[float]  # vibrational part of free energies per formula unit
    internal_energies: list[float]  # vibrational part of internal energies per f.u.
    heat_capacities: list[float]
    entropies: list[float]
    temps: list[float] | None = None
    # whether imaginary modes are present in the BS
    has_imaginary_modes: bool | None = None
    primitive: kemio.Structure | None = None
    supercell: NDArray[np.int64] | None = None  # 3x3 matrix
    # non-analytical corrections based on Born charges
    nac_params: dict[str, Any] | None = None
    thermal_displacement_data: dict[str, Any] | None = None
    mp_id: str | None = None  # material ID
    formula: str | None = None  # chemical formula

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Decode a pymatgen/monty JSON doc without pymatgen: structures and phonon
        results become kemio objects, monty arrays numpy arrays of their
        dtype. Keys outside this schema (database metadata) are ignored.
        """
        decoders = {
            "structure": to_structure,
            "primitive": to_structure,
            "phonon_bandstructure": to_phonon_bands,
            "phonon_dos": to_phonon_dos,
        }
        values = {
            field.name: decode_monty(data[field.name])
            for field in fields(cls)
            if field.init and field.name in data
        }
        decoded: dict[str, Any] = {
            key: val if val is None or key not in decoders else decoders[key](val)
            for key, val in values.items()
        }
        return cls(**decoded)


type DosInput = AnyDos | Mapping[str, AnyDos]


def phonon_dos(
    doses: DosInput,
    *,
    stack: bool = False,
    sigma: float = 0,
    normalize: Literal["max", "sum", "integral"] | None = None,
    project: Literal["element", "site"] | None = None,
    show_total: bool = True,
    **kwargs: Any,
) -> DosWidget:
    """Plot one or several (overlaid) phonon densities of states.

    Args:
        doses (DosInput): kemio PhononDos, phonopy TotalDos or Phonopy (after
            run_total_dos), pymatgen PhononDos or CompletePhononDos, or a dict
            mapping legend labels to them.
        stack (bool): Whether to plot the DOS as a stacked area graph. Defaults to
            False.
        sigma (float): Standard deviation in THz of the Gaussian smearing MatterViz
            applies. Defaults to 0 (no smearing).
        normalize ("max" | "sum" | "integral" | None): Normalization mode MatterViz
            applies to each DOS. Defaults to None.
        project ("element" | "site" | None): Decompose each site-projected DOS
            (pymatgen CompletePhononDos, kemio PhononDos with projections,
            phonopy after run_total_dos and run_projected_dos on one mesh) into
            per-element or per-site partial DOS (sites labeled by symbol and 1-based
            index, e.g. ``Na1``, like coordination plots). Defaults to None (total
            DOS only).
        show_total (bool): When projecting, also plot each total DOS (labeled
            ``Total``). Defaults to True.
        **kwargs: Passed to DosWidget, e.g. ``orientation``, ``show_legend`` or
            ``style``.

    Returns:
        DosWidget: MatterViz density of states plot.

    Raises:
        ValueError: If doses is empty, project is invalid, or stacking a
            projection would also stack its total or several DOS onto each other,
            or stacked DOS have different frequency grids.
        TypeError: If a DOS is unsupported or project is set for a DOS without site
            projections.
        TraitError: If normalize is invalid.
    """
    if project not in (None, "element", "site"):
        raise ValueError(f"Invalid {project=}, must be 'element' or 'site'")
    raw_doses = dict(doses) if isinstance(doses, Mapping) else {"": doses}
    if not raw_doses:
        raise ValueError("Empty DOS dict")
    if stack and project and (show_total or len(raw_doses) > 1):
        # MatterViz stacks every curve onto one running sum
        raise ValueError(
            f"stack=True with {project=} needs show_total=False and a single DOS, "
            f"got {show_total=} and {len(raw_doses)} DOS"
        )

    dos_dict: dict[str, PhononDosDict] = {}
    for label, raw_dos in raw_doses.items():
        try:
            ph_dos = to_phonon_dos(raw_dos)
        except TypeError as exc:
            raise TypeError(f"{exc} (key {label!r})") from exc
        freqs = ph_dos.frequency_points
        total = kemio.PhononDos(freqs, ph_dos.total_dos).as_dict()
        if project is None:
            dos_dict[label] = total
            continue
        if ph_dos.projected_dos is None or ph_dos.structure is None:
            raise TypeError(
                f"project={project!r} needs a DOS with site projections (pymatgen "
                "CompletePhononDos, kemio Phonons.projected_dos, phonopy after "
                "run_total_dos and run_projected_dos on one mesh), got "
                f"{type(raw_dos).__name__} for key {label!r}"
            )
        if project == "element":
            projected = ph_dos.element_dos()
        else:
            symbols = site_symbols(ph_dos.structure)
            site_doses = zip(symbols, ph_dos.projected_dos, strict=True)
            projected = {
                f"{symbol}{site_idx}": densities
                for site_idx, (symbol, densities) in enumerate(site_doses, start=1)
            }
        prefix = f"{label} - " if label else ""
        dos_dict |= {
            f"{prefix}{key}": kemio.PhononDos(freqs, densities).as_dict()
            for key, densities in projected.items()
        }
        if show_total:
            dos_dict[f"{prefix}Total"] = total
    # an unlabeled single DOS is passed as itself so MatterViz shows no legend entry
    return DosWidget(
        doses=dos_dict[""] if list(dos_dict) == [""] else dos_dict,
        stack=stack,
        sigma=sigma,
        normalize=normalize,
        **kwargs,
    )
