"""Enums used as keys/accessors for dicts and dataframes across Matbench Discovery."""

from __future__ import annotations

from enum import Enum, unique
from typing import TYPE_CHECKING

from pymatviz.utils import styled_html_tag


if TYPE_CHECKING:
    from typing import Any, Self


class ReprEnum(Enum):
    """Only changes the repr(), leaving str() and format()
    to the mixed-in type.
    """


class StrEnum(str, ReprEnum):
    """Enum where members are also (and must be) strings.

    Copied from std lib due to being 3.11+.
    """

    def __new__(cls, *values: Any) -> Self:
        """Values must already be str."""
        if len(values) > 3:
            raise TypeError(f"too many arguments for str(): {values!r}")
        if len(values) == 1 and not isinstance(values[0], str):
            # it must be a string
            raise TypeError(f"{values[0]!r} is not a string")
        if len(values) >= 2 and not isinstance(values[1], str):
            # check that encoding argument is a string
            raise TypeError(f"encoding must be a string, not {values[1]!r}")
        if len(values) == 3 and not isinstance(values[2], str):
            # check that errors argument is a string
            raise TypeError(f"errors must be a string, not {values[2]!r}")
        value = str(*values)
        member = str.__new__(cls, value)
        member._value_ = value
        return member

    def _generate_next_value_(  # type: ignore[override]
        self,
        start: int,  # noqa: ARG002
        count: int,  # noqa: ARG002
        last_values: list[str],  # noqa: ARG002
    ) -> str:
        """Return the lower-cased version of the member name."""
        return self.lower()


class LabelEnum(StrEnum):
    """StrEnum with optional label and description attributes plus dict() methods."""

    def __new__(
        cls, val: str, label: str | None = None, desc: str | None = None
    ) -> Self:
        """Create a new class."""
        member = str.__new__(cls, val)
        member._value_ = val
        member.__dict__ |= dict(label=label, desc=desc)
        return member

    @property
    def label(self) -> str:
        """Make label read-only."""
        return self.__dict__["label"]

    @property
    def description(self) -> str:
        """Make description read-only."""
        return self.__dict__["desc"]

    @classmethod
    def key_val_dict(cls) -> dict[str, str]:
        """Map of keys to values."""
        return {key: str(val) for key, val in cls.__members__.items()}

    @classmethod
    def val_label_dict(cls) -> dict[str, str | None]:
        """Map of values to labels."""
        return {str(val): val.label for val in cls.__members__.values()}

    @classmethod
    def val_desc_dict(cls) -> dict[str, str | None]:
        """Map of values to descriptions."""
        return {str(val): val.description for val in cls.__members__.values()}

    @classmethod
    def label_desc_dict(cls) -> dict[str | None, str | None]:
        """Map of labels to descriptions."""
        return {str(val.label): val.description for val in cls.__members__.values()}


small_font = "font-size: 0.9em; font-weight: lighter;"
eV_per_atom = styled_html_tag("(eV/atom)", style=small_font)  # noqa: N816
eV = styled_html_tag("(eV)", style=small_font)  # noqa: N816
cubic_angstrom = styled_html_tag("(Å<sup>3</sup>)", style=small_font)
angstrom = styled_html_tag("(Å)", style=small_font)
angstrom_per_atom = styled_html_tag("(Å/atom)", style=small_font)


@unique
class Key(LabelEnum):
    """Keys used to access dataframes columns."""

    arity = "arity", "N<sub>elements</sub>"  # unique elements in a chemical formula
    bandgap = "bandgap", "Band Gap"
    charge = "total_charge", "Total Charge"
    chem_sys = "chemical_system", "Chemical System"
    composition = "composition", "Composition"
    crystal_system = "crystal_system", "Crystal System"
    cse = "computed_structure_entry", "Computed Structure Entry"
    e_form_per_atom = "e_form_per_atom", f"E<sub>form</sub> {eV_per_atom}"
    e_form_pred = "e_form_per_atom_pred", f"Predicted E<sub>form</sub> {eV_per_atom}"
    e_form_true = "e_form_per_atom_true", f"Actual E<sub>form</sub> {eV_per_atom}"
    each = "energy_above_hull", f"E<sub>hull dist</sub> {eV_per_atom}"
    each_pred = "e_above_hull_pred", f"Predicted E<sub>hull dist</sub> {eV_per_atom}"
    each_true = "e_above_hull_true", f"Actual E<sub>MP hull dist</sub> {eV_per_atom}"
    element = "element", "Element"
    energy = "energy", f"Energy {eV}"
    energy_per_atom = "energy_per_atom", f"Energy {eV_per_atom}"
    final_struct = "final_structure", "Final Structure"
    forces = "forces", "Forces"
    form_energy = "formation_energy_per_atom", f"Formation Energy {eV_per_atom}"
    formula = "formula", "Formula"
    formula_pretty = "formula_pretty", "Pretty Formula"
    heat_val = "heat_val", "Heatmap Value"  # used by PTableProjector for ptable data
    init_struct = "initial_structure", "Initial Structure"
    magmoms = "magmoms", "Magnetic Moments"
    mat_id = "material_id", "Material ID"
    n_sites = "n_sites", "Number of Sites"
    oxi_state_guesses = "oxidation_state_guesses", "Oxidation State Guesses"
    spacegroup = "spacegroup", "Spacegroup Number"
    spacegroup_symbol = "spacegroup_symbol", "Spacegroup Symbol"
    stress = "stress", "Stress"
    structure = "structure", "Structure"
    task_id = "task_id", "Task ID"  # unique identifier for a compute task
    task_type = "task_type", "Task Type"
    volume = "volume", "Volume (Å³)"
    wyckoff = "wyckoff", "Aflow-style Wyckoff Label"  # crystallographic site symmetry


@unique
class Model(LabelEnum):
    """Model names."""

    # key, label, color
    m3gnet_ms = "m3gnet", "M3GNet-MS", "blue"
    chgnet_030 = "chgnet-v0.3.0", "CHGNet v0.3.0", "orange"
    mace_mp = "mace-mp-0-medium", "MACE-MP", "green"
    pbe = "pbe", "PBE", "gray"