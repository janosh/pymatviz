"""Phonon database documents."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from typing import Self

    import matscikit as mk


def _plain_json(value: Any) -> Any:
    """Value with monty-encoded numpy arrays ({"@class": "array", "data": ...})
    replaced by their nested-list data, recursively.
    """
    if isinstance(value, dict):
        if value.get("@class") == "array" and "data" in value:
            return value["data"]
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_json(item) for item in value]
    return value


@dataclass
class PhononDBDoc:
    """Phonon database document: structure, phonon band structure and DOS as
    matscikit objects, plus thermal properties.
    """

    structure: mk.Structure
    phonon_bandstructure: mk.PhononBands
    phonon_dos: mk.PhononDos
    free_energies: list[float]  # vibrational part of free energies per formula unit
    internal_energies: list[float]  # vibrational part of internal energies per f.u.
    heat_capacities: list[float]
    entropies: list[float]
    temps: list[float] | None = None  # temperatures
    # whether imaginary modes are present in the BS
    has_imaginary_modes: bool | None = None
    primitive: mk.Structure | None = None
    supercell: list[list[int]] | None = None  # 3x3 matrix
    # non-analytical corrections based on Born charges
    nac_params: dict[str, Any] | None = None
    thermal_displacement_data: dict[str, Any] | None = None
    mp_id: str | None = None  # material ID
    formula: str | None = None  # chemical formula

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        """Decode a pymatgen/monty JSON doc without pymatgen: structures and phonon
        results become matscikit objects, monty arrays nested lists. Keys outside
        this schema (database metadata) are ignored.
        """
        from matscikit.interop import to_phonon_bands, to_phonon_dos, to_structure

        decoders = {
            "structure": to_structure,
            "primitive": to_structure,
            "phonon_bandstructure": to_phonon_bands,
            "phonon_dos": to_phonon_dos,
        }
        values = {
            field.name: _plain_json(data[field.name])
            for field in fields(cls)
            if field.init and field.name in data
        }
        return cls(
            **{
                key: val if val is None else decoders.get(key, lambda obj: obj)(val)
                for key, val in values.items()
            }
        )
