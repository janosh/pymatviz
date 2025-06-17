"""Enums used as keys/accessors for dicts and dataframes across Matbench Discovery."""

from __future__ import annotations

import os
from enum import EnumType, StrEnum, _EnumDict, unique
from typing import TYPE_CHECKING, Final

import yaml

from pymatviz.utils import PKG_DIR, html_tag


if TYPE_CHECKING:
    from typing import Any, Self

    from pymatviz.typing import RgbColorType


# Load YAML data at module level
with open(f"{PKG_DIR}/keys.yml", encoding="utf-8") as file:
    _key_data = yaml.safe_load(file)

# Flatten nested structure and add group info
_keys: Final[dict[str, dict[str, str]]] = {}
for category, keys in _key_data.items():
    _keys |= {key: {"category": category} | val for key, val in keys.items()}  # type: ignore[misc]


# Map Unicode characters to their ASCII equivalents
SUPERSCRIPT_MAP: Final[dict[str, str]] = {
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "⁻": "-",
    "½": "1/2",
}
SUBSCRIPT_MAP: Final[dict[str, str]] = {
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "₋": "-",
    "½": "1/2",
}


class LabelEnum(StrEnum):
    """StrEnum with optional label and description attributes plus dict() methods.

    Simply add label and description as a tuple starting with the key's value.
    """

    def __new__(cls, val: str, label: str, desc: str = "") -> Self:
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
    def label(self) -> str:
        """Make label read-only."""
        return self.__dict__["label"]

    @property
    def description(self) -> str:
        """Make description read-only."""
        return self.__dict__["desc"]


eV_per_atom = html_tag("(eV/atom)", style="small")  # noqa: N816
eV = html_tag("(eV)", style="small")  # noqa: N816
eV_per_angstrom = html_tag("(eV/Å)", style="small")  # noqa: N816
eV_per_kelvin = html_tag("(eV/K)", style="small")  # noqa: N816
angstrom = html_tag("(Å)", style="small")
angstrom_per_atom = html_tag("(Å/atom)", style="small")
cubic_angstrom = html_tag("(Å<sup>3</sup>)", style="small")
degrees = html_tag("(°)", style="small")
gram_per_cm3 = html_tag("(g/cm³)", style="small")
kelvin = html_tag("(K)", style="small")
pascal = html_tag("(Pa)", style="small")
giga_pascal = html_tag("(GPa)", style="small")
joule = html_tag("(J)", style="small")
joule_per_mol = html_tag("(J/mol)", style="small")
joule_per_m2 = html_tag("(J/m²)", style="small")


@unique
class Key(StrEnum):
    """Keys used to access dataframes columns, organized by semantic groups."""

    @property
    def label(self) -> str:
        """Label associated with the key."""
        label = _keys[self.value]["label"]
        unit = _keys[self.value].get("unit")
        # needlessly verbose to include "dimensionless" in label
        if unit and unit != "dimensionless":
            label += f" ({unit})"
        return label

    @property
    def unit(self) -> str | None:
        """Unit associated with the key with HTML tags for sub/superscripts."""
        if not (unit := _keys[self.value].get("unit")):
            return None

        # Process character by character
        html_str = ""
        in_super = in_sub = False

        idx = 0
        while idx < len(unit):
            char = unit[idx]

            # Check if character is superscript
            if new_char := SUPERSCRIPT_MAP.get(char):
                if not in_super:
                    html_str += "<sup>"
                    in_super = True
                html_str += new_char
            # Check if character is subscript
            elif new_char := SUBSCRIPT_MAP.get(char):
                if not in_sub:
                    html_str += "<sub>"
                    in_sub = True
                html_str += new_char
            else:
                # Close any open tags
                if in_super:
                    html_str += "</sup>"
                    in_super = False
                if in_sub:
                    html_str += "</sub>"
                    in_sub = False
                html_str += char
            idx += 1

        # Close any remaining open tags
        if in_super:
            html_str += "</sup>"
        if in_sub:
            html_str += "</sub>"

        return html_str

    @property
    def category(self) -> str:
        """Category associated with the key."""
        return _keys[self.value]["category"]

    @property
    def symbol(self) -> str | None:
        """Symbol associated with the key."""
        return _keys[self.value].get("symbol")

    @property
    def desc(self) -> str | None:
        """Description associated with the key."""
        return _keys[self.value].get("description")

    def __reduce_ex__(self, proto: object) -> tuple[type, tuple[str]]:
        """Return as a string when pickling. Overrides Enum.__reduce_ex__ which returns
        the tuple self.__class__, (self._value_,). self.__class can cause pickle
        failures if the corresponding Enum was moved or renamed in the meantime.
        """
        return str, (self.value,)

    # Structural
    n_sites = "n_sites"
    structure = "structure"
    init_struct = "init_struct"
    initial_struct = "initial_struct"
    final_struct = "final_struct"
    cell = "cell"
    lattice = "lattice"
    lattice_vectors = "lattice_vectors"
    lattice_angles = "lattice_angles"
    lattice_lens = "lattice_lens"
    init_volume = "init_volume"
    final_volume = "final_volume"
    volume = "volume"
    vol_per_atom = "vol_per_atom"
    density = "density"
    pressure = "pressure"
    lattice_params = "lattice_params"
    supercell = "supercell"
    atom_nums = "atom_nums"
    coord_num = "coord_num"
    chem_env_symbol = "chem_env_symbol"
    bond_lens = "bond_lens"
    bond_angles = "bond_angles"
    packing_fraction = "packing_fraction"
    max_pair_dist = "max_pair_dist"
    conventional_cell = "conventional_cell"
    primitive_cell = "primitive_cell"
    reduced_cell = "reduced_cell"
    niggli_reduced_cell = "niggli_reduced_cell"

    # Crystal Symmetry Properties
    crystal_system = "crystal_system"
    spg_num = "spg_num"
    init_spg_num = "init_spg_num"
    final_spg_num = "final_spg_num"
    spg_symbol = "spg_symbol"
    choice_symbol = "choice_symbol"
    hall_num = "hall_num"
    hall_symbol = "hall_symbol"
    translations = "translations"
    rotations = "rotations"
    symmetry = "symmetry"
    symmetry_change = "symmetry_change"
    symmetry_decrease = "symmetry_decrease"
    symmetry_increase = "symmetry_increase"
    symmetry_match = "symmetry_match"
    symprec = "symprec"
    angle_tolerance = "angle_tolerance"
    point_group = "point_group"
    n_wyckoff_pos = "n_wyckoff_pos"
    n_rot_syms = "n_rot_syms"
    n_trans_syms = "n_trans_syms"
    n_sym_ops = "n_sym_ops"
    wyckoff = "wyckoff"
    wyckoff_symbol = "wyckoff_symbol"
    wyckoff_symbols = "wyckoff_symbols"

    # Structure Prototyping
    protostructure = "protostructure"
    protostructure_moyo = "protostructure_moyo"
    protostructure_spglib = "protostructure_spglib"
    prototype = "prototype"

    # Composition
    arity = "arity"
    chem_sys = "chem_sys"
    composition = "composition"
    reduced_composition = "reduced_composition"
    fractional_composition = "fractional_composition"
    composition_wt_perc = "composition_wt_perc"
    molar_composition = "molar_composition"
    weight_composition = "weight_composition"
    molar_fraction = "molar_fraction"
    weight_fraction = "weight_fraction"
    molar_mass = "molar_mass"
    element = "element"
    formula = "formula"
    reduced_formula = "reduced_formula"
    fractional_formula = "fractional_formula"
    formula_pretty = "formula_pretty"
    anonymous_formula = "anonymous_formula"
    charge = "charge"
    oxi_states = "oxi_states"
    oxi_state_guesses = "oxi_state_guesses"
    n_atoms = "n_atoms"
    n_elements = "n_elements"
    n_val_electrons = "n_val_electrons"
    n_electrons = "n_electrons"

    # Chemical
    atomic_mass = "atomic_mass"
    atomic_number = "atomic_number"
    atomic_radius = "atomic_radius"
    atomic_symbol = "atomic_symbol"
    elem_symbol = "elem_symbol"
    atomic_volume = "atomic_volume"
    block = "block"
    group = "group"
    period = "period"
    row = "row"
    column = "column"
    series = "series"
    shell = "shell"
    valence = "valence"
    isotope_masses = "isotope_masses"
    natural_abundance = "natural_abundance"
    half_life = "half_life"
    electronegativity = "electronegativity"
    ionic_radius = "ionic_radius"
    covalent_radius = "covalent_radius"
    ionization_energy = "ionization_energy"

    # Thermodynamic
    energy = "energy"
    enthalpy = "enthalpy"
    entropy = "entropy"
    free_energy = "free_energy"
    gibbs_free_energy = "gibbs_free_energy"
    helmholtz_free_energy = "helmholtz_free_energy"
    corrected_energy = "corrected_energy"
    uncorrected_energy = "uncorrected_energy"
    internal_energy = "internal_energy"
    energy_per_atom = "energy_per_atom"
    corrected_energy_per_atom = "corrected_energy_per_atom"
    uncorrected_energy_per_atom = "uncorrected_energy_per_atom"
    cohesive_energy_per_atom = "cohesive_energy_per_atom"
    heat_of_formation = "heat_of_formation"
    heat_of_reaction = "heat_of_reaction"
    e_form = "e_form"
    e_form_per_atom = "e_form_per_atom"
    e_form_pred = "e_form_pred"
    e_form_true = "e_form_true"
    form_energy = "form_energy"
    formation_energy = "formation_energy"
    formation_energy_per_atom = "formation_energy_per_atom"
    each = "each"
    each_pred = "each_pred"
    each_true = "each_true"
    computed_structure_entry = "computed_structure_entry"
    melting_point = "melting_point"
    boiling_point = "boiling_point"
    phase_transition_temp = "phase_transition_temp"
    critical_temp = "critical_temp"
    critical_pressure = "critical_pressure"
    critical_vol = "critical_vol"
    lattice_energy = "lattice_energy"
    interface_energy = "interface_energy"

    # Electronic
    bandgap = "bandgap"
    bandgap_pbe = "bandgap_pbe"
    bandgap_hse = "bandgap_hse"
    bandgap_r2scan = "bandgap_r2scan"
    bandgap_ml = "bandgap_ml"
    bandgap_true = "bandgap_true"
    bandgap_pred = "bandgap_pred"
    fermi_energy = "fermi_energy"
    electronic_structure = "electronic_structure"
    electron_affinity = "electron_affinity"
    work_function = "work_function"
    dos = "dos"
    band_structure = "band_structure"
    conductivity = "conductivity"
    seebeck_coefficient = "seebeck_coefficient"
    hall_coefficient = "hall_coefficient"
    supercon_crit_temp = "supercon_crit_temp"
    carrier_concentration = "carrier_concentration"
    mobility = "mobility"
    effective_mass = "effective_mass"
    polarizability = "polarizability"
    polarization = "polarization"
    dielectric_constant = "dielectric_constant"
    charge_density = "charge_density"
    electron_density = "electron_density"
    hole_density = "hole_density"
    electron_mobility = "electron_mobility"
    hole_mobility = "hole_mobility"
    quantum_capacitance = "quantum_capacitance"
    plasmon_frequency = "plasmon_frequency"
    spin_orbit_coupling = "spin_orbit_coupling"
    carrier_mobility = "carrier_mobility"

    # Mechanical
    forces = "forces"
    stress = "stress"
    max_stress = "max_stress"
    virial = "virial"
    stress_trace = "stress_trace"
    voigt_stress = "voigt_stress"
    bulk_modulus = "bulk_modulus"
    shear_modulus = "shear_modulus"
    young_modulus = "young_modulus"
    poisson_ratio = "poisson_ratio"
    hardness = "hardness"
    elastic_tensor = "elastic_tensor"
    elastic_tensor_voigt = "elastic_tensor_voigt"
    elastic_tensor_reuss = "elastic_tensor_reuss"
    elastic_tensor_vrh = "elastic_tensor_vrh"
    toughness = "toughness"
    yield_strength = "yield_strength"
    tensile_strength = "tensile_strength"
    ductility = "ductility"
    fracture_toughness = "fracture_toughness"
    sound_velocity = "sound_velocity"
    strain = "strain"
    strain_rate = "strain_rate"
    compliance = "compliance"

    # Thermal
    temperature = "temperature"
    thermal_conductivity = "thermal_conductivity"
    lattice_thermal_conductivity = "lattice_thermal_conductivity"
    electronic_thermal_conductivity = "electronic_thermal_conductivity"
    heat_capacity = "heat_capacity"
    specific_heat_capacity = "specific_heat_capacity"
    thermal_expansion_coefficient = "thermal_expansion_coefficient"
    debye_temp = "debye_temp"
    gruneisen_parameter = "gruneisen_parameter"
    thermal_diffusivity = "thermal_diffusivity"
    thermal_expansion = "thermal_expansion"
    thermal_expansion_coeff = "thermal_expansion_coeff"
    thermal_resistivity = "thermal_resistivity"
    thermal_time_constant = "thermal_time_constant"
    heat_flux = "heat_flux"

    # Phonon
    ph_band_structure = "ph_band_structure"
    ph_dos = "ph_dos"
    ph_freqs = "ph_freqs"
    mode_weights = "mode_weights"
    q_points = "q_points"
    n_q_points = "n_q_points"
    q_point_mesh = "q_point_mesh"
    fc2_supercell = "fc2_supercell"
    fc3_supercell = "fc3_supercell"
    ph_dos_mae = "ph_dos_mae"
    ph_dos_rmse = "ph_dos_rmse"
    has_imag_ph_gamma_modes = "has_imag_ph_gamma_modes"
    has_imag_ph_modes = "has_imag_ph_modes"
    last_ph_dos_peak = "last_ph_dos_peak"
    max_ph_freq = "max_ph_freq"
    min_ph_freq = "min_ph_freq"
    phonon_frequency = "phonon_frequency"

    # Optical
    refractive_index = "refractive_index"
    diel_const = "diel_const"
    absorption_spectrum = "absorption_spectrum"
    photoluminescence = "photoluminescence"
    optical_conductivity = "optical_conductivity"
    reflectivity = "reflectivity"
    transmittance = "transmittance"
    absorption_coefficient = "absorption_coefficient"
    extinction_coefficient = "extinction_coefficient"
    absorption_length = "absorption_length"
    quantum_efficiency = "quantum_efficiency"
    oscillator_strength = "oscillator_strength"
    group_velocity = "group_velocity"
    phase_velocity = "phase_velocity"

    # Surface
    surface_energy = "surface_energy"
    wulff_shape = "wulff_shape"
    surface_area = "surface_area"
    surface_reconstruction = "surface_reconstruction"
    adsorption_energy = "adsorption_energy"
    work_of_adhesion = "work_of_adhesion"

    # Defect
    vacancy_formation_energy = "vacancy_formation_energy"
    interstitial_formation_energy = "interstitial_formation_energy"
    defect_concentration = "defect_concentration"
    migration_energy = "migration_energy"
    dislocation_energy = "dislocation_energy"
    stacking_fault_energy = "stacking_fault_energy"
    defect_formation_volume = "defect_formation_volume"
    migration_barrier = "migration_barrier"
    diffusion_coefficient = "diffusion_coefficient"
    activation_volume = "activation_volume"

    # Magnetic
    magmoms = "magmoms"
    magnetic_moment = "magnetic_moment"
    curie_temperature = "curie_temperature"
    neel_temp = "neel_temp"
    anisotropy = "anisotropy"
    magnetocrystalline_anisotropy = "magnetocrystalline_anisotropy"
    coercivity = "coercivity"
    exchange_coupling = "exchange_coupling"
    magnetic_susceptibility = "magnetic_susceptibility"
    magnetic_anisotropy = "magnetic_anisotropy"

    # DFT
    dft = "dft"
    xc = "xc"
    lda = "lda"
    gga = "gga"
    meta_gga = "meta_gga"
    hybrid = "hybrid"
    hartree_fock = "hartree_fock"
    pbe = "pbe"
    pbe_sol = "pbe_sol"
    scan = "scan"
    r2scan = "r2scan"
    hse = "hse"
    xc_functional = "xc_functional"
    convergence_electronic = "convergence_electronic"
    convergence_ionic = "convergence_ionic"
    k_points = "k_points"
    pseudo_potential = "pseudo_potential"
    pseudo_potential_type = "pseudo_potential_type"
    u_correction = "u_correction"
    needs_u_correction = "needs_u_correction"
    soc = "soc"
    basis_set = "basis_set"
    basis_set_size = "basis_set_size"
    energy_cutoff = "energy_cutoff"
    fft_grid = "fft_grid"
    smearing = "smearing"
    max_scf_iter = "max_scf_iter"
    scf_tol = "scf_tol"
    wave_function = "wave_function"
    n_steps = "n_steps"
    n_elec_steps = "n_elec_steps"
    n_ionic_steps = "n_ionic_steps"
    n_md_steps = "n_md_steps"
    n_relax_steps = "n_relax_steps"
    n_scf_steps = "n_scf_steps"
    n_bands = "n_bands"
    n_kpoints = "n_kpoints"
    kpoint_density = "kpoint_density"
    kpoint_spacing = "kpoint_spacing"
    kpoint_mesh = "kpoint_mesh"
    kpoint_offset = "kpoint_offset"
    kpoint_symmetry = "kpoint_symmetry"
    kpoint_sampling = "kpoint_sampling"
    relativistic = "relativistic"
    spin_polarized = "spin_polarized"
    collinear = "collinear"
    non_collinear = "non_collinear"

    # Molecular Dynamics
    trajectory = "trajectory"
    frames = "frames"
    frame = "frame"
    diffusivity = "diffusivity"
    diffusion_tensor = "diffusion_tensor"
    msd = "msd"  # mean squared displacement
    velocity_autocorr = "velocity_autocorr"  # velocity autocorrelation function
    ensemble = "ensemble"
    nvt = "nvt"  # canonical ensemble
    nve = "nve"  # microcanonical ensemble
    npt = "npt"  # isothermal-isobaric ensemble
    nvp = "nvp"  # isoenthalpic-isobaric ensemble
    micro_canonical = "micro_canonical"
    canonical = "canonical"
    grand_canonical = "grand_canonical"
    isothermal_isobaric = "isothermal_isobaric"
    time_step = "time_step"
    time_steps = "time_steps"
    integration_time = "integration_time"
    equilibration_time = "equilibration_time"
    production_time = "production_time"
    thermostat = "thermostat"
    barostat = "barostat"
    langevin_damping = "langevin_damping"
    nose_hoover = "nose_hoover"
    berendsen = "berendsen"
    andersen = "andersen"
    velocity_verlet = "velocity_verlet"
    verlet = "verlet"
    leap_frog = "leap_frog"
    kinetic_energy = "kinetic_energy"
    potential_energy = "potential_energy"
    total_energy = "total_energy"
    conserved_energy = "conserved_energy"
    temperature_avg = "temperature_avg"
    pressure_avg = "pressure_avg"
    volume_avg = "volume_avg"
    density_avg = "density_avg"
    rdf = "rdf"  # radial distribution function
    velocity = "velocity"
    acceleration = "acceleration"
    momentum = "momentum"
    angular_momentum = "angular_momentum"
    gyration_radius = "gyration_radius"
    drift = "drift"
    flux = "flux"
    correlation_time = "correlation_time"
    correlation_length = "correlation_length"
    relaxation_time = "relaxation_time"

    # ML
    train_task = "train_task"
    test_task = "test_task"
    train_set = "train_set"
    targets = "targets"
    model_name = "model_name"
    model_id = "model_id"
    model_version = "model_version"
    model_type = "model_type"
    model_params = "model_params"
    model_framework = "model_framework"
    hyperparams = "hyperparams"
    feature_importance = "feature_importance"
    optimizer = "optimizer"
    loss = "loss"
    uncertainty = "uncertainty"
    epochs = "epochs"
    batch_size = "batch_size"
    learning_rate = "learning_rate"
    optimizer_momentum = "optimizer_momentum"
    weight_decay = "weight_decay"
    early_stopping = "early_stopping"
    n_folds = "n_folds"
    n_estimators = "n_estimators"
    n_features = "n_features"
    n_targets = "n_targets"
    n_classes = "n_classes"
    n_layers = "n_layers"
    radial_cutoff = "radial_cutoff"
    angular_cutoff = "angular_cutoff"

    # Metrics
    accuracy = "accuracy"
    auc = "auc"
    confusion_matrix = "confusion_matrix"
    daf = "daf"
    f1 = "f1"
    fp = "fp"
    fn = "fn"
    tp = "tp"
    tn = "tn"
    tpr = "tpr"
    fpr = "fpr"
    tnr = "tnr"
    fnr = "fnr"
    mae = "mae"
    r2 = "r2"
    pearson = "pearson"
    spearman = "spearman"
    kendall = "kendall"
    mse = "mse"
    rmse = "rmse"
    rmsd = "rmsd"
    n_sym_ops_mae = "n_sym_ops_mae"
    structure_rmsd = "structure_rmsd"
    mape = "mape"
    srme = "srme"
    sre = "sre"
    srd = "srd"
    smpe = "smpe"
    smape = "smape"
    sse = "sse"
    variance = "variance"
    std_dev = "std_dev"
    iqr = "iqr"
    outlier = "outlier"
    error = "error"
    energy_error = "energy_error"
    force_error = "force_error"
    stress_error = "stress_error"
    volume_error = "volume_error"
    max_force_error = "max_force_error"
    max_stress_error = "max_stress_error"
    max_volume_error = "max_volume_error"
    force_rmse = "force_rmse"
    stress_rmse = "stress_rmse"
    volume_rmse = "volume_rmse"
    force_mae = "force_mae"
    stress_mae = "stress_mae"
    volume_mae = "volume_mae"
    residuals = "residuals"
    prc = "prc"
    prc_curve = "prc_curve"
    precision = "precision"
    recall = "recall"
    sensitivity = "sensitivity"
    specificity = "specificity"
    roc = "roc"
    roc_curve = "roc_curve"
    roc_auc = "roc_auc"
    hit_rate = "hit_rate"
    n_structures = "n_structures"
    n_materials = "n_materials"
    n_molecules = "n_molecules"
    n_samples = "n_samples"
    n_configs = "n_configs"
    conservative = "conservative"
    non_conservative = "non_conservative"
    conservation_error = "conservation_error"
    conservativeness = "conservativeness"
    smoothness = "smoothness"
    tortuosity = "tortuosity"
    force_flips = "force_flips"
    energy_jumps = "energy_jumps"

    # Computational Details
    run_time_sec = "run_time_sec"
    run_time_hr = "run_time_hr"
    cpu_hours = "cpu_hours"
    gpu_hours = "gpu_hours"
    start_time = "start_time"
    start_date = "start_date"
    date_time = "date_time"
    date = "date"
    time = "time"
    date_added = "date_added"
    date_modified = "date_modified"
    end_time = "end_time"
    end_date = "end_date"
    step = "step"
    state = "state"
    output = "output"
    n_cores = "n_cores"
    n_nodes = "n_nodes"
    n_gpus = "n_gpus"
    n_tasks = "n_tasks"
    n_processes = "n_processes"
    n_threads = "n_threads"
    core_hours = "core_hours"
    memory = "memory"
    queue_name = "queue_name"
    job_name = "job_name"
    job_type = "job_type"
    job_dir = "job_dir"
    job_script = "job_script"
    job_log = "job_log"
    job_output = "job_output"
    job_status = "job_status"
    job_errors = "job_errors"
    job_warnings = "job_warnings"
    job_comments = "job_comments"
    job_metadata = "job_metadata"

    # Identifiers and Metadata
    id = "id"
    db_id = "db_id"
    uuid = "uuid"
    mat_id = "material_id"
    frame_id = "frame_id"
    task = "task"
    job_id = "job_id"
    task_id = "task_id"
    task_type = "task_type"
    model = "model"
    author = "author"
    authors = "authors"
    year = "year"
    journal = "journal"
    issue = "issue"
    pages = "pages"
    doi = "doi"
    url = "url"
    citation = "citation"
    description = "description"
    keywords = "keywords"
    license = "license"
    version = "version"
    abstract = "abstract"
    tags = "tags"
    categories = "categories"
    funding = "funding"
    acknowledgements = "acknowledgements"

    # Code
    repo = "repo"
    branch = "branch"
    commit = "commit"
    hash = "hash"
    tag = "tag"
    release = "release"
    synthesis_temperature = "synthesis_temperature"
    synthesis_pressure = "synthesis_pressure"
    synthesis_time = "synthesis_time"
    synthesis_method = "synthesis_method"
    synthesis_conditions = "synthesis_conditions"
    atmosphere = "atmosphere"
    heat_treatment = "heat_treatment"
    powder_preparation = "powder_preparation"

    # Performance Indicators
    fom = "fom"  # codespell:ignore
    power_factor = "power_factor"
    zt = "zt"
    efficiency = "efficiency"
    capacity = "capacity"
    rate = "rate"
    lifetime = "lifetime"
    stability = "stability"
    selectivity = "selectivity"
    purity = "purity"
    yield_ = "yield"
    activity = "activity"
    performance = "performance"
    gain = "gain"
    power = "power"
    current = "current"
    voltage = "voltage"
    resistance = "resistance"
    impedance = "impedance"
    capacitance = "capacitance"

    # Environmental Indicators
    toxicity = "toxicity"
    recyclability = "recyclability"
    biodegradability = "biodegradability"
    sustainability = "sustainability"

    # Economic Factors
    cost = "cost"
    raw_material_cost = "raw_material_cost"
    abundance = "abundance"
    corrosion_resistance = "corrosion_resistance"
    viscosity = "viscosity"
    activation_energy = "activation_energy"
    count = "count"  # type: ignore[assignment]
    heat_val = "heat_val"
    piezoelectric_tensor = "piezoelectric_tensor"
    dielectric_tensor = "dielectric_tensor"


class Task(LabelEnum):
    """What kind of task is being performed."""

    static = "static", "Static"  # aka single-point
    relax = "relax", "Relaxation"  # aka geometry optimization
    geo_opt = "geo_opt", "Geometry Optimization"
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
    md = "md", "Molecular Dynamics"
    ion_diffusion = "ion_diffusion", "Ion Diffusion"
    electron_transport = "electron_transport", "Electron Transport"
    charge_transport = "charge_transport", "Charge Transport"
    thermal_transport = "thermal_transport", "Thermal Transport"


@unique
class ElemCountMode(LabelEnum):
    """Mode of counting elements in a chemical formula."""

    # key, label, color
    composition = (
        "composition",
        "Composition",
        "Count elements by their amount in the composition",
    )
    fractional_composition = (
        "fractional_composition",
        "Fractional Composition",
        "Count elements by their fraction of the composition",
    )
    reduced_composition = (
        "reduced_composition",
        "Reduced Composition",
        "Count elements by their amount in the reduced composition",
    )
    occurrence = (
        "occurrence",
        "Occurrence",
        "Count elements by occurrence in composition, ignoring their amount",
    )


@unique
class ElemColorScheme(LabelEnum):
    """Names of element color palettes.

    Used e.g. in structure visualizations and periodic table plots.
    """

    # key, label, color
    # from https://wikipedia.org/wiki/Jmol"
    jmol = "jmol", "Jmol", "Java-based molecular visualization"
    # from https://jp-minerals.org/vesta
    vesta = "vesta", "VESTA", "Visualization for Electronic Structural Analysis"
    # custom made for pymatviz
    alloy = "alloy", "Alloy", "High-contrast color scheme optimized for metal alloys"
    pastel = "pastel", "Pastel", "Pastel color scheme"

    @property
    def color_map(self) -> dict[str, RgbColorType]:
        """Return map from element symbol to color."""
        import pymatviz.colors as pmv_colors

        return getattr(pmv_colors, f"ELEM_COLORS_{self.value.upper()}")


@unique
class SiteCoords(LabelEnum):
    """Site coordinate representations."""

    cartesian = "cartesian", "Cartesian"
    fractional = "fractional", "Fractional"
    cartesian_fractional = "cartesian+fractional", "Cartesian and Fractional"


class MetaFiles(EnumType):
    """Metaclass of Files enum that adds base_dir class kwarg."""

    _base_dir: str
    _auto_download: bool

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: _EnumDict,
        *,
        base_dir: str = "",
        auto_download: bool = False,
        **kwargs: Any,
    ) -> MetaFiles:
        """Create new Files enum with given base directory."""
        obj = super().__new__(cls, name, bases, namespace, **kwargs)
        obj._base_dir = base_dir
        obj._auto_download = auto_download
        return obj


class Files(StrEnum, metaclass=MetaFiles):
    """Enum of data files with associated file directories and URLs."""

    def __new__(
        cls, file_path: str, url: str = "", label: str = "", desc: str = ""
    ) -> Self:
        """Create a new member of the FileUrls enum with a given URL where to load the
        file from and directory where to save it to.
        """
        obj = str.__new__(cls)
        obj._value_ = file_path
        obj.__dict__ |= dict(file_path=file_path, url=url, label=label, desc=desc)
        return obj

    def __repr__(self) -> str:
        """String representation of the file."""
        return f"{type(self).__name__}.{self.name}"

    def __str__(self) -> str:
        """String representation of the file."""
        return self.name

    @property
    def url(self) -> str:
        """URL associated with the file."""
        return self.__dict__["url"]

    @property
    def file_path(self) -> str:
        """File path associated with the file."""
        path = f"{type(self)._base_dir}/{self.__dict__['file_path']}"
        if type(self)._auto_download and not os.path.isfile(path) and self.url:
            print(f"Downloading {self.url} to {path}")  # noqa: T201
            import requests

            response = requests.get(self.url)  # noqa: S113
            response.raise_for_status()
            with open(path, mode="wb") as file:
                file.write(response.content)
        return path

    @property
    def label(self) -> str:
        """Label associated with the file."""
        return self.__dict__["label"]

    @property
    def description(self) -> str:
        """Description associated with the file."""
        return self.__dict__["desc"]

    @property
    def rel_path(self) -> str:
        """Path of the file relative to the repo's ROOT directory."""
        return self.__dict__["file_path"]

    @classmethod
    def from_label(cls, label: str) -> Self:
        """Get enum member from pretty label."""
        file = next((attr for attr in cls if attr.label == label), None)
        if file is None:
            import difflib

            similar_labels = difflib.get_close_matches(label, [k.label for k in cls])
            raise ValueError(
                f"{label=} not found in {cls.__name__}. Did you mean one "
                f"of {similar_labels}?"
            )
        return file
