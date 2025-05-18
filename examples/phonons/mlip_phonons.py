"""Calculate and compare Phonon DOS and band structure using MACE and ORB v3."""

# /// script
# dependencies = [
#     "mace-torch>=0.3.11",
#     "orb-models>=0.5.2",
#     "phonopy>=2.35",
#     "pymatviz[export-figs]>=0.15.1",
#     "seekpath",
#     "ase",
# ]
# ///

import itertools
import os
import warnings
from collections import defaultdict

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.optimize import FIRE
from mace.calculators.foundations_models import mace_mp
from orb_models.forcefield.calculator import ORBCalculator
from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

import pymatviz as pmv


for category, msg in {
    RuntimeWarning: "Guards may run slower on Python",
    UserWarning: "There are no gridspecs with layoutgrids",
    DeprecationWarning: "dict interface is deprecated.",
}.items():
    warnings.filterwarnings("ignore", category=category, message=msg)

pmv.set_plotly_template("pymatviz_white")
module_dir = os.path.dirname(__file__)

# --- Configuration ---
device: str = "cuda" if torch.cuda.is_available() else "cpu"
dtype: str = "float32"
supercell_matrix = 2 * np.eye(3)
mesh: list[int] = [20, 20, 20]
n_relax: int = 100
displacement: float = 0.01
mace_mpa_0_url: str = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model"
mace_omat_0_url: str = "https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model"

models_to_run: dict[str, Calculator] = {
    "MACE-MPA-0": mace_mp(model=mace_mpa_0_url, default_dtype=dtype, device=device),
    "MACE-OMAT-0": mace_mp(model=mace_omat_0_url, default_dtype=dtype, device=device),
    "ORB_v3": ORBCalculator(
        model=ORB_PRETRAINED_MODELS["orb-v3-conservative-inf-mpa"](),
        max_num_neighbors=None,
        device=device,
    ),
}

results: dict[str, dict[str, Phonopy]] = defaultdict(
    dict
)  # bulk_kwargs -> model_name -> Phonopy
q_pts: list[np.ndarray] | None = None
connections: list[np.ndarray] | None = None
labels: list[str] | None = None
systems = (
    dict(name="Si", crystalstructure="diamond", a=5.431),
    dict(name="Al", crystalstructure="fcc", a=4.11),
)

for cubic, bulk_kwargs, (model_name, calc) in itertools.product(
    (True, False), systems, models_to_run.items()
):
    atoms = bulk(**bulk_kwargs, cubic=cubic)
    print(f"{atoms.get_chemical_formula()} {model_name} {cubic=}")
    relaxed_struct = atoms.copy()
    relaxed_struct.calc = calc
    # --- Relaxation ---
    optimizer = FIRE(relaxed_struct, logfile=None)
    optimizer.run(fmax=0.01, steps=n_relax)

    # --- Phonopy Setup ---
    unitcell_ph_atoms = PhonopyAtoms(
        symbols=relaxed_struct.symbols,
        cell=relaxed_struct.cell.array,
        scaled_positions=relaxed_struct.get_scaled_positions(),
    )
    ph = Phonopy(unitcell_ph_atoms, supercell_matrix)

    # Generate FC2 displacements
    ph.generate_displacements(distance=displacement)
    phonopy_supercells = ph.supercells_with_displacements

    # Convert PhonopyAtoms back to ASE Atoms for force calculation
    ase_supercells = []
    for ph_supercell in phonopy_supercells:
        ase_supercell = Atoms(
            symbols=ph_supercell.symbols,
            cell=ph_supercell.cell,
            scaled_positions=ph_supercell.scaled_positions,
            pbc=True,
        )
        ase_supercell.calc = calc
        ase_supercells.append(ase_supercell)

    # --- Force Calculation ---
    force_sets = [sc.get_forces() for sc in ase_supercells]

    # Produce force constants
    ph.forces = force_sets
    ph.produce_force_constants()

    # --- DOS Calculation ---
    ph.run_mesh(mesh)
    ph.run_total_dos()

    # --- Band Structure Calculation ---
    if len(results) == 0:
        # Use auto_band_structure on the first run to get path info
        ph.auto_band_structure(plot=False, write_yaml=False)
        bands_data = ph.get_band_structure_dict()
        # Extract path details from the BandStructure object for subsequent runs
        q_pts = ph.band_structure.qpoints
        connections = ph.band_structure.path_connections
        labels = ph.band_structure.labels  # Extract labels
        first_run = False
    else:  # For subsequent runs, use the stored q_pts and connections
        ph.run_band_structure(
            q_pts, path_connections=connections, labels=labels, is_band_connection=True
        )
        bands_data = ph.get_band_structure_dict()

    bulk_str = "|".join(f"{k}={v}" for k, v in bulk_kwargs.items()) + f"|{cubic=}"
    results[bulk_str][model_name] = ph

os.makedirs(fig_dir := f"{module_dir}/tmp/phonons", exist_ok=True)

for bulk_str, model_results in results.items():
    bands = {key: ph.band_structure for key, ph in model_results.items()}
    doses = {key: ph.total_dos for key, ph in model_results.items()}
    fig = pmv.phonon_bands_and_dos(bands, doses)
    fig.layout.title.update(text=f"{bulk_str}")
    fig.layout.margin.t = 60
    fig.show()
    pmv.save_fig(fig, f"{fig_dir}/{bulk_str}.pdf")
