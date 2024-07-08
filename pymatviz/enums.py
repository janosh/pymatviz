"""Enums used as keys/accessors for dicts and dataframes across Matbench Discovery."""

from __future__ import annotations

import sys
from enum import Enum, unique
from typing import TYPE_CHECKING

from pymatviz.utils import styled_html_tag


if TYPE_CHECKING:
    from typing import Any, Self

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
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
    """StrEnum with optional label and description attributes plus dict() methods.

    Simply add label and description as a tuple starting with the key's value.
    """

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
    """Keys used to access dataframes columns, organized by semantic groups."""

    # Structural
    crystal_system = "crystal_system", "Crystal System"
    spg_num = "space_group_number", "Space Group Number"
    spg_symbol = "space_group_symbol", "Space Group Symbol"
    wyckoff = "wyckoff", "Aflow-style Wyckoff Label"
    n_sites = "n_sites", "Number of Sites"
    structure = "structure", "Structure"
    init_struct = "initial_structure", "Initial Structure"
    final_struct = "final_structure", "Final Structure"
    init_volume = "initial_volume", f"Initial Volume {cubic_angstrom}"
    final_volume = "final_volume", f"Final Volume {cubic_angstrom}"
    volume = "volume", f"Volume {cubic_angstrom}"
    vol_per_atom = "volume_per_atom", f"Volume per Atom {cubic_angstrom}"
    density = "density", "Density (g/cm³)"
    symmetry = "symmetry", "Symmetry"
    point_group = "point_group", "Point Group"
    lattice_params = "lattice_parameters", "Lattice Parameters"
    supercell = "supercell", "Supercell"
    atom_nums = "atom_nums", "Atomic Numbers", "Atomic numbers for each crystal site"
    struct_proto = "structure_prototype", "Structure Prototype"
    iso_struct_proto = (
        "isopointal_structure_prototype",
        "Isopointal Structure Prototype",
    )
    uniq_proto = "unique_prototype", "Unique Prototype"
    coord_num = "coordination_number", "Coordination Number"
    bond_lens = "bond_lengths", f"Bond Lengths {angstrom}"
    bond_angles = "bond_angles", "Bond Angles (°)"
    packing_fraction = "packing_fraction", "Packing Fraction"

    # Composition and Chemical
    arity = "arity", "N<sub>elements</sub>"
    atomic_mass = "atomic_mass", "Atomic Mass (u)"
    atomic_number = "atomic_number", "Atomic Number"
    atomic_radius = "atomic_radius", f"Atomic Radius {angstrom}"
    atomic_symbol = "atomic_symbol", "Atomic Symbol"
    atomic_volume = "atomic_volume", f"Atomic Volume {cubic_angstrom}"
    block = "block", "Block"
    group = "group", "Group"
    period = "period", "Period"
    row = "row", "Row"
    series = "series", "Series"
    shell = "shell", "Shell"
    valence = "valence", "Valence"
    chem_sys = "chemical_system", "Chemical System"
    composition = "composition", "Composition"
    element = "element", "Element"
    formula = "formula", "Formula"
    formula_pretty = "formula_pretty", "Pretty Formula"
    reduced_formula = "reduced_formula", "Reduced chemical formula"
    charge = "total_charge", "Total Charge"
    oxi_states = "oxidation_states", "Oxidation States"
    oxi_state_guesses = "oxidation_state_guesses", "Oxidation State Guesses"
    n_val_electrons = "n_valence_electrons", "Number of Valence Electrons"
    n_electrons = "n_total_electrons", "Total Number of Electrons"
    isotope_masses = "isotope_masses", "Isotope Masses"
    natural_abundance = "natural_abundance", "Natural Abundance (%)"
    half_life = "half_life", "Half-life"
    electronegativity = "electronegativity", "Electronegativity (Pauling scale)"
    ionic_radius = "ionic_radius", f"Ionic Radius {angstrom}"
    covalent_radius = "covalent_radius", f"Covalent Radius {angstrom}"
    ionization_energy = "ionization_energy", f"Ionization Energy {eV}"

    # Thermodynamic
    energy = "energy", f"Energy {eV}"
    energy_per_atom = "energy_per_atom", f"Energy {eV_per_atom}"
    uncorrected_energy_per_atom = (
        "uncorrected_energy_per_atom",
        f"Uncorrected Energy {eV_per_atom}",
    )
    cohesive_energy_per_atom = (
        "cohesive_energy_per_atom",
        f"Cohesive Energy {eV_per_atom}",
    )
    e_form_per_atom = "e_form_per_atom", f"E<sub>form</sub> {eV_per_atom}"
    e_form_pred = "e_form_per_atom_pred", f"Predicted E<sub>form</sub> {eV_per_atom}"
    e_form_true = "e_form_per_atom_true", f"Actual E<sub>form</sub> {eV_per_atom}"
    each = "energy_above_hull", f"E<sub>hull dist</sub> {eV_per_atom}"
    each_pred = "e_above_hull_pred", f"Predicted E<sub>hull dist</sub> {eV_per_atom}"
    each_true = "e_above_hull_true", f"Actual E<sub>MP hull dist</sub> {eV_per_atom}"
    form_energy = "formation_energy_per_atom", f"Formation Energy {eV_per_atom}"
    cse = "computed_structure_entry", "Computed Structure Entry"
    melting_point = "melting_point", "Melting Point (K)"
    boiling_point = "boiling_point", "Boiling Point (K)"
    phase_transition_temp = (
        "phase_transition_temperature",
        "Phase Transition Temperature (K)",
    )
    critical_temp = "critical_temperature", "Critical Temperature (K)"
    critical_pressure = "critical_pressure", "Critical Pressure (Pa)"
    critical_vol = "critical_volume", "Critical Volume (m³/mol)"
    gibbs_free_energy = "gibbs_free_energy", f"Gibbs Free Energy {eV}"
    enthalpy = "enthalpy", f"Enthalpy {eV}"
    entropy = "entropy", "Entropy (eV/K)"
    lattice_energy = "lattice_energy", f"Lattice Energy {eV}"
    interface_energy = "interface_energy", "Interface Energy (J/m²)"

    # Electronic
    bandgap = "bandgap", f"Band Gap {eV}"
    bandgap_pbe = "bandgap_pbe", f"PBE Band Gap {eV}"
    bandgap_hse = "bandgap_hse", f"HSE Band Gap {eV}"
    bandgap_r2scan = "bandgap_r2scan", f"r2SCAN Band Gap {eV}"
    bandgap_ml = "bandgap_ml", f"ML Band Gap {eV}"
    bandgap_true = "bandgap_true", f"Actual Band Gap {eV}"
    bandgap_pred = "bandgap_pred", f"Predicted Band Gap {eV}"
    fermi_energy = "fermi_energy", f"Fermi Energy {eV}"
    electron_affinity = "electron_affinity", f"Electron Affinity {eV}"
    work_function = "work_function", f"Work Function {eV}"
    dos = "density_of_states", "Density of States"
    band_structure = "band_structure", "Band Structure"
    conductivity = "conductivity", "Electrical Conductivity (S/m)"
    seebeck_coeff = "seebeck_coefficient", "Seebeck Coefficient (μV/K)"
    hall_coeff = "hall_coefficient", "Hall Coefficient (m³/C)"
    supercon_crit_temp = (
        "superconducting_critical_temperature",
        "Superconducting Critical Temperature (K)",
    )
    carrier_concentration = "carrier_concentration", "Carrier Concentration (cm⁻³)"
    mobility = "mobility", "Carrier Mobility (cm²/V·s)"
    effective_mass = "effective_mass", "Effective Mass (m<sub>e</sub>)"
    # how easy it is to move an atom's cloud of electrons
    polarizability = "polarizability", "Polarizability (Å³)"
    # displacement of positive charges relative to negative charges
    polarization = "polarization", "Polarization (C/m²)"

    # Mechanical
    forces = "forces", "Forces"
    stress = "stress", "Stress"
    stress_trace = "stress_trace", "Stress Trace"
    voigt_stress = "voigt_stress", "Voigt Stress"
    bulk_modulus = "bulk_modulus", "Bulk Modulus (GPa)"
    shear_modulus = "shear_modulus", "Shear Modulus (GPa)"
    young_modulus = "young_modulus", "Young's Modulus (GPa)"
    poisson_ratio = "poisson_ratio", "Poisson's Ratio"
    hardness = "hardness", "Hardness (Mohs scale)"
    elastic_tensor = "elastic_tensor", "Elastic Tensor"
    yield_strength = "yield_strength", "Yield Strength (MPa)"
    tensile_strength = "tensile_strength", "Tensile Strength (MPa)"
    ductility = "ductility", "Ductility (%)"
    fracture_toughness = "fracture_toughness", "Fracture Toughness (MPa·m½)"
    bulk_sound_velocity = "bulk_sound_velocity", "Bulk Sound Velocity (m/s)"

    # Thermal
    temperature = "temperature", "Temperature (K)"
    thermal_conductivity = "thermal_conductivity", "Thermal Conductivity (W/m·K)"
    lattice_thermal_conductivity = (
        "lattice_thermal_conductivity",
        "Lattice Thermal Conductivity (W/m·K)",
    )
    electronic_thermal_conductivity = (
        "electronic_thermal_conductivity",
        "Electronic Thermal Conductivity (W/m·K)",
    )
    heat_capacity = "heat_capacity", "Heat Capacity (J/mol·K)"
    specific_heat_capacity = "specific_heat_capacity", "Specific Heat Capacity (J/kg·K)"
    thermal_expansion_coefficient = (
        "thermal_expansion_coefficient",
        "Thermal Expansion Coefficient (1/K)",
    )
    debye_temp = "debye_temperature", "Debye Temperature (K)"
    gruneisen_parameter = "gruneisen_parameter", "Grüneisen Parameter"
    thermal_diffusivity = "thermal_diffusivity", "Thermal Diffusivity (m²/s)"

    # Phonon
    ph_band_structure = "phonon_bandstructure", "Phonon Band Structure"
    ph_dos = "phonon_dos", "Phonon Density of States"
    ph_dos_mae = "ph_dos_mae", "Phonon DOS MAE"
    has_imag_ph_gamma_modes = (
        "has_imaginary_gamma_phonon_freq",
        "Has imaginary Γ phonon modes",
    )
    has_imag_ph_modes = "has_imag_phonon_freq", "Has imaginary phonon modes"
    last_ph_dos_peak = "last_ph_dos_peak_thz", "ω<sub>max</sub> (THz)"
    max_ph_freq = "max_freq_thz", "Ω<sub>max</sub> (THz)"  # highest phonon frequency
    min_ph_freq = "min_freq_thz", "Ω<sub>min</sub> (THz)"  # lowest phonon frequency

    # Optical
    refractive_index = "refractive_index", "Refractive Index"
    diel_const = "dielectric_constant", "Dielectric Constant"
    absorption_spectrum = "absorption_spectrum", "Absorption Spectrum"
    photoluminescence = "photoluminescence", "Photoluminescence"
    optical_conductivity = "optical_conductivity", "Optical Conductivity (S/m)"
    reflectivity = "reflectivity", "Reflectivity"
    transmittance = "transmittance", "Transmittance"
    absorption_coefficient = "absorption_coefficient", "Absorption Coefficient (cm⁻¹)"
    extinction_coefficient = "extinction_coefficient", "Extinction Coefficient"

    # Surface
    surface_energy = "surface_energy", "Surface Energy (J/m²)"
    wulff_shape = "wulff_shape", "Wulff Shape"
    surface_area = "surface_area", "Surface Area (m²)"
    surface_reconstruction = "surface_reconstruction", "Surface Reconstruction"
    adsorption_energy = "adsorption_energy", f"Adsorption Energy {eV}"
    work_of_adhesion = "work_of_adhesion", "Work of Adhesion (J/m²)"

    # Defect
    vacancy_formation_energy = (
        "vacancy_formation_energy",
        f"Vacancy Formation Energy {eV}",
    )
    interstitial_formation_energy = (
        "interstitial_formation_energy",
        f"Interstitial Formation Energy {eV}",
    )
    defect_concentration = "defect_concentration", "Defect Concentration (cm⁻³)"
    migration_energy = "migration_energy", f"Migration Energy {eV}"
    dislocation_energy = "dislocation_energy", "Dislocation Energy (eV/Å)"
    stacking_fault_energy = "stacking_fault_energy", "Stacking Fault Energy (mJ/m²)"

    # Magnetic
    magmoms = "magmoms", "Magnetic Moments"
    magnetic_moment = "magnetic_moment", "Magnetic Moment (μB)"
    curie_temperature = "curie_temperature", "Curie Temperature (K)"
    neel_temp = "neel_temperature", "Néel Temperature (K)"
    magnetocrystalline_anisotropy = (
        "magnetocrystalline_anisotropy",
        "Magnetocrystalline Anisotropy (meV)",
    )
    coercivity = "coercivity", "Coercivity (Oe)"

    # DFT
    dft = "dft", "DFT"
    pbe = "pbe", "PBE"
    pbe_sol = "pbe_sol", "PBEsol"
    r2scan = "r2scan", "r2SCAN"
    hse = "hse", "HSE"
    xc_functional = "xc_functional", "Exchange-Correlation Functional"
    convergence_electronic = "convergence_electronic", "Electronic Convergence"
    convergence_ionic = "convergence_ionic", "Ionic Convergence"
    kpoints = "kpoints", "K-points"
    pseudopotentials = "pseudopotentials", "Pseudopotentials"
    u_correction = "u_correction", "Hubbard U Correction"
    needs_u_correction = "needs_u_correction", "Needs Hubbard U correction"
    soc = "spin_orbit_coupling", "Spin-Orbit Coupling"

    # ML
    train_task = "train_task", "Training Task"
    test_task = "test_task", "Test Task"
    train_set = "training_set", "Training Set"
    targets = "targets", "Targets"
    model_name = "model_name", "Model Name"
    model_version = "model_version", "Model Version"
    model_type = "model_type", "Model Type"
    model_params = "model_parameters", "Model Parameters"
    hyperparams = "hyperparameters", "Hyperparameters"
    feature_importance = "feature_importance", "Feature Importance"
    daf = "DAF", "Discovery Acceleration Factor"

    # Computational Details
    run_time_sec = "run_time_sec", "Run Time (sec)"
    run_time_hr = "run_time_hr", "Run Time (hr)"
    cpu_hours = "cpu_hours", "CPU Hours"
    gpu_hours = "gpu_hours", "GPU Hours"
    start_time = "start_time", "Start Time"
    start_date = "start_date", "Start Date"
    end_time = "end_time", "End Time"
    end_date = "end_date", "End Date"
    step = "step", "Step"  # as in job/optimizer step
    state = "state", "State"  # as in job state
    job_type = "job_type", "Job Type"

    # Identifiers and Metadata
    id = "id", "ID"
    mat_id = "material_id", "Material ID"
    task = "task", "Task"
    task_id = "task_id", "Task ID"
    task_type = "task_type", "Task Type"
    model = "model", "Model"

    # Synthesis-related
    synthesis_temperature = "synthesis_temperature", "Synthesis Temperature (K)"
    synthesis_pressure = "synthesis_pressure", "Synthesis Pressure (Pa)"

    # Performance Indicators
    fom = "figure_of_merit", "Figure of Merit"  # codespell:ignore
    power_factor = "power_factor", "Power Factor"

    # Environmental Indicators
    toxicity = "toxicity", "Toxicity Index"
    recyclability = "recyclability", "Recyclability Score"

    # Economic Factors
    raw_material_cost = "raw_material_cost", "Raw Material Cost ($/kg)"
    abundance = "abundance", "Elemental Abundance (ppm)"

    # Chemical Properties
    corrosion_resistance = "corrosion_resistance", "Corrosion Resistance"
    viscosity = "viscosity", "Viscosity (Pa·s)"
    activation_energy = "activation_energy", f"Activation Energy {eV}"

    # Miscellaneous
    count = "count", "Count"  # type: ignore[assignment]
    heat_val = "heat_val", "Heatmap Value"
    piezoelectric_tensor = "piezoelectric_tensor", "Piezoelectric Tensor"


@unique
class Model(LabelEnum):
    """Model names."""

    # key, label, color
    m3gnet_ms = "m3gnet", "M3GNet-MS", "blue"
    chgnet_030 = "chgnet-v0.3.0", "CHGNet v0.3.0", "orange"
    mace_mp = "mace-mp-0-medium", "MACE-MP", "green"
    pbe = "pbe", "PBE", "gray"


@unique
class ElemCountMode(LabelEnum):
    """Mode of counting elements in a chemical formula."""

    # key, label, color
    composition = "composition", "Composition", "blue"
    fractional_composition = (
        "fractional_composition",
        "Fractional Composition",
        "orange",
    )
    reduced_composition = "reduced_composition", "Reduced Composition", "green"
    occurrence = "occurrence", "Occurrence", "gray"


@unique
class ElemColorMode(LabelEnum):
    """Mode of coloring elements in structure visualizations or periodic table
    plots.
    """

    # key, label, color
    element_types = "element-types", "Element Types", "blue"


@unique
class ElemColorScheme(LabelEnum):
    """Names of element color palettes.

    Used e.g. in structure visualizations and periodic table plots.
    """

    # key, label, color
    jmol = "jmol", "Jmol", "Java-based molecular visualization"
    # https://wikipedia.org/wiki/Jmol"
    vesta = "vesta", "VESTA", "Visualization for Electronic Structural Analysis"
    # https://jp-minerals.org/vesta
