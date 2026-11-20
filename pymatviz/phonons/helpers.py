"""Phonon database documents and phonopy to pymatgen conversion."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any, Self

    from phonopy.phonon.band_structure import BandStructure as PhonopyBandStructure
    from pymatgen.core import Structure
    from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
    from pymatgen.phonon.dos import PhononDos


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
