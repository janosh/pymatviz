"""Enums used as keys/accessors for dicts and dataframes across Matbench Discovery."""

from __future__ import annotations

import sys
from enum import Enum, unique
from typing import TYPE_CHECKING

from pymatviz.utils import html_tag


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

        def __str__(self) -> str:
            """Return the lower-cased version of the member name."""
            """
            use enum_name instead of class.enum_name
            """
            if self._name_ is None:
                cls_name = type(self).__name__
                return f"{cls_name}({self._value_!r})"
            return self._name_.lower()

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
        """Create a new class from a value, label, and description. Label and
        description are optional.
        """
        member = str.__new__(cls, val)
        member._value_ = val
        member.__dict__ |= dict(label=label, desc=desc)
        return member

    def __repr__(self) -> str:
        """Return label if available, else type name and value."""
        return self.label or f"{type(self).__name__}.{self.name}"

    def __reduce_ex__(self, proto: object) -> tuple[type, tuple[str]]:
        """Return as a string when pickling. Overrides Enum.__reduce_ex__ which returns
        the tuple self.__class__, (self._value_,). self.__class can cause pickle
        failures if the corresponding Enum was moved or renamed in the meantime.
        """
        return str, (self.value,)

    @property
    def label(self) -> str | None:
        """Make label read-only."""
        return self.__dict__.get("label")

    @property
    def description(self) -> str | None:
        """Make description read-only."""
        return self.__dict__.get("desc")

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
    def label_desc_dict(cls) -> dict[str, str | None]:
        """Map of labels to descriptions. None-valued labels are omitted."""
        return {
            str(val.label): val.description
            for val in cls.__members__.values()
            if val.label
        }


eV_per_atom = html_tag("(eV/atom)", style="small")  # noqa: N816
eV = html_tag("(eV)", style="small")  # noqa: N816
eV_per_angstrom = html_tag("(eV/Å)", style="small")  # noqa: N816
eV_per_kelvin = html_tag("(eV/K)", style="small")  # noqa: N816
angstrom = html_tag("(Å)", style="small")
angstrom_per_atom = html_tag("(Å/atom)", style="small")
cubic_angstrom = html_tag("(Å<sup>3</sup>)", style="small")
gram_per_cm3 = html_tag("(g/cm³)", style="small")
kelvin = html_tag("(K)", style="small")
pascal = html_tag("(Pa)", style="small")
giga_pascal = html_tag("(GPa)", style="small")
joule = html_tag("(J)", style="small")
joule_per_mol = html_tag("(J/mol)", style="small")
joule_per_m2 = html_tag("(J/m²)", style="small")


@unique
class Key(LabelEnum):
    """Keys used to access dataframes columns, organized by semantic groups."""

    # Structural
    crystal_system = "crystal_system", "Crystal System"
    spg_num = "space_group_number", "Space Group Number"
    spg_symbol = "space_group_symbol", "Space Group Symbol"
    n_sites = "n_sites", "Number of Sites"
    n_wyckoff = "n_wyckoff", "Number of Wyckoff Positions"
    structure = "structure", "Structure"
    init_struct = "initial_structure", "Initial Structure"
    final_struct = "final_structure", "Final Structure"
    cell = "cell", "Cell"
    lattice = "lattice", "Lattice"
    lattice_vectors = "lattice_vectors", "Lattice Vectors"
    lattice_angles = "lattice_angles", "Lattice Angles (°)"
    lattice_lens = "lattice_lengths", f"Lattice Lengths {angstrom}"
    init_volume = "initial_volume", f"Initial Volume {cubic_angstrom}"
    final_volume = "final_volume", f"Final Volume {cubic_angstrom}"
    volume = "volume", f"Volume {cubic_angstrom}"
    vol_per_atom = "volume_per_atom", f"Volume per Atom {cubic_angstrom}"
    density = "density", f"Density {gram_per_cm3}"
    symmetry = "symmetry", "Symmetry"
    point_group = "point_group", "Point Group"
    lattice_params = "lattice_parameters", "Lattice Parameters"
    supercell = "supercell", "Supercell"
    atom_nums = "atom_nums", "Atomic Numbers", "Atomic numbers for each crystal site"
    coord_num = "coordination_number", "Coordination Number"
    bond_lens = "bond_lengths", f"Bond Lengths {angstrom}"
    bond_angles = "bond_angles", "Bond Angles (°)"
    packing_fraction = "packing_fraction", "Packing Fraction"

    # Structure Prototyping
    # AFLOW-style prototype label with appended chemical system
    protostructure = "protostructure", "Protostructure Label"
    # Deprecated name for the protostructure
    wyckoff = "wyckoff", "AFLOW-style Label with Chemical System"
    wyckoff_spglib = "wyckoff_spglib", "Wyckoff Label (spglib)"
    prototype = "prototype", "Prototype Label"
    aflow_prototype = "aflow_prototype", "AFLOW-style Prototype Label"
    # AFLOW-style prototype label that has been canonicalized
    canonical_proto = "canonical_prototype", "Canonical AFLOW-style Prototype"
    # Deprecated name for the canonical_proto
    uniq_proto = "unique_prototype", "Unique AFLOW-style Prototype"

    # Composition and Chemical
    arity = "arity", "N<sub>elements</sub>"
    atomic_mass = "atomic_mass", "Atomic Mass (u)"
    atomic_number = "atomic_number", "Atomic Number"
    atomic_radius = "atomic_radius", f"Atomic Radius {angstrom}"
    atomic_symbol = "atomic_symbol", "Atomic Symbol"
    elem_symbol = "element_symbol", "Element Symbol"
    atomic_volume = "atomic_volume", f"Atomic Volume {cubic_angstrom}"
    block = "block", "Block"
    group = "group", "Group"
    period = "period", "Period"
    row = "row", "Row"
    column = "column", "Column"
    series = "series", "Series"
    shell = "shell", "Shell"
    valence = "valence", "Valence"
    chem_sys = "chemical_system", "Chemical System"
    composition = "composition", "Composition"
    element = "element", "Element"
    formula = "formula", "Formula"
    formula_pretty = "formula_pretty", "Pretty Formula"
    reduced_formula = "reduced_formula", "Reduced chemical formula"
    anonymous_formula = "anonymous_formula", "Anonymous Formula"
    charge = "total_charge", "Total Charge"
    oxi_states = "oxidation_states", "Oxidation States"
    oxi_state_guesses = "oxidation_state_guesses", "Oxidation State Guesses"
    n_atoms = "n_atoms", "Number of Atoms"
    n_elements = "n_elements", "Number of Elements"
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
    enthalpy = "enthalpy", f"Enthalpy {eV}"
    entropy = "entropy", f"Entropy {eV_per_kelvin}"
    free_energy = "free_energy", f"Free Energy {eV}"
    gibbs_free_energy = "gibbs_free_energy", f"Gibbs Free Energy {eV}"
    helmholtz_free_energy = "helmholtz_free_energy", f"Helmholtz Free Energy {eV}"
    corrected_energy = "corrected_energy", f"Corrected Energy {eV}"
    uncorrected_energy = "uncorrected_energy", f"Uncorrected Energy {eV}"
    internal_energy = "internal_energy", f"Internal Energy {eV}"
    energy_per_atom = "energy_per_atom", f"Energy {eV_per_atom}"
    corrected_energy_per_atom = (
        "corrected_energy_per_atom",
        f"Corrected Energy {eV_per_atom}",
    )
    uncorrected_energy_per_atom = (
        "uncorrected_energy_per_atom",
        f"Uncorrected Energy {eV_per_atom}",
    )
    cohesive_energy_per_atom = (
        "cohesive_energy_per_atom",
        f"Cohesive Energy {eV_per_atom}",
    )
    heat_of_formation = "heat_of_formation", f"Heat of Formation {eV}"
    heat_of_reaction = "heat_of_reaction", f"Heat of Reaction {eV}"
    e_form_per_atom = "e_form_per_atom", f"E<sub>form</sub> {eV_per_atom}"
    e_form_pred = "e_form_per_atom_pred", f"Predicted E<sub>form</sub> {eV_per_atom}"
    e_form_true = "e_form_per_atom_true", f"Actual E<sub>form</sub> {eV_per_atom}"
    each = "energy_above_hull", f"E<sub>hull dist</sub> {eV_per_atom}"
    each_pred = "e_above_hull_pred", f"Predicted E<sub>hull dist</sub> {eV_per_atom}"
    each_true = "e_above_hull_true", f"Actual E<sub>MP hull dist</sub> {eV_per_atom}"
    form_energy = "formation_energy_per_atom", f"Formation Energy {eV_per_atom}"
    cse = "computed_structure_entry", "Computed Structure Entry"
    melting_point = "melting_point", f"Melting Point {kelvin}"
    boiling_point = "boiling_point", f"Boiling Point {kelvin}"
    phase_transition_temp = (
        "phase_transition_temperature",
        f"Phase Transition Temperature {kelvin}",
    )
    critical_temp = "critical_temperature", f"Critical Temperature {kelvin}"
    critical_pressure = "critical_pressure", f"Critical Pressure {pascal}"
    critical_vol = "critical_volume", "Critical Volume (m³/mol)"
    lattice_energy = "lattice_energy", f"Lattice Energy {eV}"
    interface_energy = "interface_energy", f"Interface Energy {joule_per_m2}"

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
        f"Superconducting Critical Temperature {kelvin}",
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
    bulk_modulus = "bulk_modulus", f"Bulk Modulus {giga_pascal}"
    shear_modulus = "shear_modulus", f"Shear Modulus {giga_pascal}"
    young_modulus = "young_modulus", f"Young's Modulus {giga_pascal}"
    poisson_ratio = "poisson_ratio", "Poisson's Ratio"
    hardness = "hardness", "Hardness (Mohs scale)"
    elastic_tensor = "elastic_tensor", "Elastic Tensor"
    elastic_tensor_voigt = "elastic_tensor_voigt", "Voigt Elastic Tensor"
    elastic_tensor_reuss = "elastic_tensor_reuss", "Reuss Elastic Tensor"
    elastic_tensor_vrh = "elastic_tensor_vrh", "Voigt-Reuss-Hill Elastic Tensor"
    toughness = "toughness", "Toughness (MPa)"
    yield_strength = "yield_strength", "Yield Strength (MPa)"
    tensile_strength = "tensile_strength", "Tensile Strength (MPa)"
    ductility = "ductility", "Ductility (%)"
    fracture_toughness = "fracture_toughness", "Fracture Toughness (MPa·m½)"
    bulk_sound_velocity = "bulk_sound_velocity", "Bulk Sound Velocity (m/s)"

    # Thermal
    temperature = "temperature", f"Temperature {kelvin}"
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
    debye_temp = "debye_temperature", f"Debye Temperature {kelvin}"
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
    surface_energy = "surface_energy", f"Surface Energy {joule_per_m2}"
    wulff_shape = "wulff_shape", "Wulff Shape"
    surface_area = "surface_area", "Surface Area (m²)"
    surface_reconstruction = "surface_reconstruction", "Surface Reconstruction"
    adsorption_energy = "adsorption_energy", f"Adsorption Energy {eV}"
    work_of_adhesion = "work_of_adhesion", f"Work of Adhesion {joule_per_m2}"

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
    dislocation_energy = "dislocation_energy", f"Dislocation Energy {eV_per_angstrom}"
    stacking_fault_energy = "stacking_fault_energy", "Stacking Fault Energy (mJ/m²)"

    # Magnetic
    magmoms = "magmoms", "Magnetic Moments"
    magnetic_moment = "magnetic_moment", "Magnetic Moment (μB)"
    curie_temperature = "curie_temperature", f"Curie Temperature {kelvin}"
    neel_temp = "neel_temperature", f"Néel Temperature {kelvin}"
    magnetocrystalline_anisotropy = (
        "magnetocrystalline_anisotropy",
        "Magnetocrystalline Anisotropy (meV)",
    )
    coercivity = "coercivity", "Coercivity (Oe)"

    # DFT
    dft = "dft", "DFT"
    xc = "exchange_correlation", "Exchange-Correlation"
    lda = "lda", "LDA"
    gga = "gga", "GGA"
    meta_gga = "meta_gga", "Meta-GGA"
    hybrid = "hybrid", "Hybrid"
    hartree_fock = "hartree_fock", "Hartree-Fock"
    pbe = "pbe", "PBE"
    pbe_sol = "pbe_sol", "PBEsol"
    scan = "scan", "SCAN"
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
    model_id = "model_id", "Model ID"
    model_version = "model_version", "Model Version"
    model_type = "model_type", "Model Type"
    model_params = "model_params", "Model Parameters"
    model_framework = "model_framework", "Model Framework"  # e.g. PyTorch, TensorFlow
    hyperparams = "hyperparameters", "Hyperparameters"
    feature_importance = "feature_importance", "Feature Importance"
    optimizer = "optimizer", "Optimizer"
    loss = "loss", "Loss"
    uncertainty = "uncertainty", "Uncertainty"
    epochs = "epochs", "Epochs"
    batch_size = "batch_size", "Batch Size"
    learning_rate = "learning_rate", "Learning Rate"
    momentum = "momentum", "Momentum"
    weight_decay = "weight_decay", "Weight Decay"
    early_stopping = "early_stopping", "Early Stopping"
    n_folds = "n_folds", "Number of Folds"
    n_estimators = "n_estimators", "Number of Estimators"
    n_features = "n_features", "Number of Features"
    n_targets = "n_targets", "Number of Targets"
    n_classes = "n_classes", "Number of Classes"
    n_layers = "n_layers", "Number of Layers"
    radial_cutoff = "radial_cutoff", "Radial Cutoff"  # for GNNs, usually in Å
    angular_cutoff = "angular_cutoff", "Angular Cutoff"  # max order spherical harmonics

    # Metrics
    accuracy = "accuracy", "Accuracy"
    auc = "AUC", "Area Under the Curve"
    confusion_matrix = "confusion_matrix", "Confusion Matrix"
    daf = "DAF", "Discovery Acceleration Factor"
    f1 = "F1", "F1 Score"
    fp = "FP", "False Positives"
    fn = "FN", "False Negatives"
    tp = "TP", "True Positives"
    tn = "TN", "True Negatives"
    tpr = "TPR", "True Positive Rate"
    fpr = "FPR", "False Positive Rate"
    tnr = "TNR", "True Negative Rate"
    fnr = "FNR", "False Negative Rate"
    mae = "MAE", "Mean Absolute Error"
    r2 = "R²", "R² Score"
    pearson = "Pearson", "Pearson Correlation"
    spearman = "Spearman", "Spearman Correlation"
    kendall = "Kendall", "Kendall Correlation"
    rmse = "RMSE", "Root Mean Squared Error"
    mape = "MAPE", "Mean Absolute Percentage Error"
    variance = "variance", "Variance"
    std_dev = "std_dev", "Standard Deviation"
    iqr = "IQR", "Interquartile Range"
    outlier = "outlier", "Outlier"
    error = "error", "Error"
    residuals = "residuals", "Residuals"
    prc = "PRC", "Precision-Recall Curve"
    prc_curve = "prc_curve", "PRC Curve"
    precision = "precision", "Precision"
    recall = "recall", "Recall"
    sensitivity = "sensitivity", "Sensitivity"  # same as recall
    specificity = "specificity", "Specificity"  # same as precision
    roc = "ROC", "Receiver Operating Characteristic"
    roc_curve = "roc_curve", "ROC Curve"
    roc_auc = "ROC_AUC", "ROC AUC"
    hit_rate = "hit_rate", "Hit Rate"

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
    output = "output", "Output"
    n_cores = "n_cores", "Number of Cores"
    n_nodes = "n_nodes", "Number of Nodes"
    n_gpus = "n_gpus", "Number of GPUs"
    n_tasks = "n_tasks", "Number of Tasks"
    n_processes = "n_processes", "Number of Processes"
    n_threads = "n_threads", "Number of Threads"
    core_hours = "core_hours", "Core Hours"
    memory = "memory", "Memory"
    n_steps = "n_steps", "Number of Steps"
    queue_name = "queue_name", "Queue Name"
    job_name = "job_name", "Job Name"
    job_type = "job_type", "Job Type"
    job_dir = "job_dir", "Job Directory"
    job_script = "job_script", "Job Script"
    job_log = "job_log", "Job Log"
    job_output = "job_output", "Job Output"
    job_status = "job_status", "Job Status"
    job_errors = "job_errors", "Job Errors"
    job_warnings = "job_warnings", "Job Warnings"
    job_comments = "job_comments", "Job Comments"
    job_metadata = "job_metadata", "Job Metadata"

    # Identifiers and Metadata
    id = "id", "ID"
    db_id = "db_id", "Database ID"
    uuid = "uuid", "UUID"
    mat_id = "material_id", "Material ID"
    # as in molecular dynamics or geometry optimization frame
    frame_id = "frame_id", "Frame ID"
    task = "task", "Task"
    job_id = "job_id", "Job ID"
    task_id = "task_id", "Task ID"
    task_type = "task_type", "Task Type"
    model = "model", "Model"

    # Synthesis-related
    synthesis_temperature = "synthesis_temperature", f"Synthesis Temperature {kelvin}"
    synthesis_pressure = "synthesis_pressure", f"Synthesis Pressure {pascal}"

    # Performance Indicators
    fom = "figure_of_merit", "Figure of Merit"  # codespell:ignore
    power_factor = "power_factor", "Power Factor"
    zt = "ZT", "ZT"
    efficiency = "efficiency", "Efficiency"
    capacity = "capacity", "Capacity"
    rate = "rate", "Rate"
    lifetime = "lifetime", "Lifetime"
    stability = "stability", "Stability"
    selectivity = "selectivity", "Selectivity"
    yield_ = "yield", "Yield"  # underscore because reserved keyword
    activity = "activity", "Activity"
    performance = "performance", "Performance"
    gain = "gain", "Gain"
    power = "power", "Power"
    current = "current", "Current"
    voltage = "voltage", "Voltage"
    resistance = "resistance", "Resistance"
    impedance = "impedance", "Impedance"
    capacitance = "capacitance", "Capacitance"

    # Environmental Indicators
    toxicity = "toxicity", "Toxicity Index"
    recyclability = "recyclability", "Recyclability Score"
    biodegradability = "biodegradability", "Biodegradability Score"
    sustainability = "sustainability", "Sustainability Index"

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
    dielectric_tensor = "dielectric_tensor", "Dielectric Tensor"


class Task(LabelEnum):
    """What kind of task is being performed."""

    static = "static", "Static"  # aka single-point
    relax = "relax", "Relaxation"  # aka geometry optimization
    double_relax = "double_relax", "Double Relaxation"
    phonon = "phonon", "Phonon"  # aka vibrational analysis
    eos = "eos", "Equation of State"  # aka volume optimization
    band_structure = "band_structure", "Band Structure"  # aka electronic structure
    dos = "dos", "Density of States"
    line_non_scf = "line_non_scf", "Non-SCF Line"
    defect = "defect", "Defect"
    point_defect = "point_defect", "Point Defect"
    line_defect = "line_defect", "Line Defect"
    adsorption = "adsorption", "Adsorption"  # aka surface adsorption
    surface = "surface", "Surface"  # aka surface energy
    reaction = "reaction", "Reaction"  # aka chemical reaction
    formation_energy = "formation_energy", "Formation Energy"
    bandgap = "bandgap", "Band Gap"
    elastic = "elastic", "Elastic"
    thermal = "thermal", "Thermal"
    magnetic = "magnetic", "Magnetic"
    magnetic_ordering = "magnetic_ordering", "Magnetic Ordering"
    optical = "optical", "Optical"
    dielectric = "dielectric", "Dielectric"  # aka optical properties
    electronic = "electronic", "Electronic"
    synthesis = "synthesis", "Synthesis"
    molecular_dynamics = "molecular_dynamics", "Molecular Dynamics"
    ion_diffusion = "ion_diffusion", "Ion Diffusion"
    electron_transport = "electron_transport", "Electron Transport"
    charge_transport = "charge_transport", "Charge Transport"
    thermal_transport = "thermal_transport", "Thermal Transport"


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
