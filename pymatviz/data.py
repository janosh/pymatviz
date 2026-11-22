"""Demo and test data: dummy regression data, disordered demo structures, the test
files directory and NaCl phonons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, NamedTuple

import kemio
import numpy as np

from pymatviz.utils import ROOT


if TYPE_CHECKING:
    from numpy.typing import NDArray


class RegressionData(NamedTuple):
    """Regression data containing: y_true, y_pred and y_std."""

    y_true: NDArray[np.float64]
    y_pred: NDArray[np.float64]
    y_std: NDArray[np.float64]


def regression(
    n_samples: int = 500,
    true_mean: float = 5,
    true_std: float = 4,
    pred_slope: float = 1.2,
    pred_intercept: float = -2,
    seed: int = 0,
) -> RegressionData:
    """Generate dummy regression data for testing and prototyping.

    This function creates synthetic data to simulate a regression task:
    - `y_true`: Sampled from a normal distribution with mean 5 and
        standard deviation 4.
    - `y_pred`: Linearly related to `y_true` with a slope of 1.2 and
        additional Gaussian noise.
    - `y_std`: Residuals scaled by random noise, representing variability
        in predictions.

    Parameters:
        n_samples (int): Number of samples to generate. Default is 500.

    Returns:
        RegressionData: A named tuple containing y_true, y_pred, and y_std.
    """
    np_rng = np.random.default_rng(seed=seed)

    y_true = np_rng.normal(loc=true_mean, scale=true_std, size=n_samples)

    noise = np_rng.normal(loc=0, scale=1, size=n_samples)
    y_pred = pred_slope * y_true + pred_intercept + noise

    # Generate realistic positive uncertainties
    # Base uncertainty that's correlated with absolute residuals but always positive
    abs_residuals = np.abs(y_true - y_pred)
    y_std = (
        0.5 + 0.3 * abs_residuals + 0.8 * np_rng.exponential(scale=1, size=n_samples)
    )

    return RegressionData(y_true=y_true, y_pred=y_pred, y_std=y_std)


TEST_FILES: str = f"{ROOT}/tests/files"

# === Disordered structures for tests and examples ===


fe3co4_disordered = kemio.Structure(
    np.eye(3) * 5,
    [{"Fe": 0.75, "C": 0.25}, "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5]],
    fractional=True,
)
batio3_disordered = kemio.Structure(
    np.eye(3) * 4.0,
    [
        "Ba",  # Pure Ba site
        {"Ti": 0.6, "Zr": 0.4},  # Disordered Ti/Zr site
        "O",
        "O",
        "O",  # Pure O sites
    ],
    [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
    fractional=True,
)

# Li-ion cathode with vacancy disorder
lico2_disordered = kemio.Structure(
    kemio.lattice_from_parameters(2.82, 2.82, 14.05, 90, 90, 120, c_along_z=True),
    [
        {"Li": 0.75},  # Li site with 25% vacancies
        "Co",
        "O",
        "O",
    ],
    [(0, 0, 0), (0, 0, 0.5), (0, 0, 0.25), (0, 0, 0.75)],
    fractional=True,
)

# High-entropy alloy with ternary disorder (3 species on one site)
hea_disordered = kemio.Structure(
    np.eye(3) * 3.6,
    [
        {"Fe": 0.5, "Ni": 0.3, "Cr": 0.2},  # Ternary disordered site
        "Al",  # Pure Al site
    ],
    [(0, 0, 0), (0.5, 0.5, 0.5)],
    fractional=True,
)

# Complex multisite disorder - solid solution with multiple disordered sites
solid_solution = kemio.Structure(
    np.eye(3) * 5.0,
    [
        {"Ca": 0.6, "Sr": 0.4},  # Disordered A-site
        {"Ti": 0.8, "Zr": 0.15, "Hf": 0.05},  # Ternary disordered B-site
        "O",  # Pure O sites
        "O",
        "O",
    ],
    [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
    fractional=True,
)

disordered_demo_structures: Final[dict[str, kemio.Structure]] = {
    "BaTi₀.₆Zr₀.₄O₃": batio3_disordered,
    "Li₀.₇₅CoO₂ (with vacancies)": lico2_disordered,
    "Fe₀.₅Ni₀.₃Cr₀.₂Al": hea_disordered,
    "Ca₀.₆Sr₀.₄Ti₀.₈Zr₀.₁₅Hf₀.₀₅O₃": solid_solution,
}


def load_nacl_phonons() -> kemio.Phonons:
    """Harmonic phonons of NaCl (2x2x2 supercell, force constants not symmetrized)
    from the test phonopy files, read by kemio, for tests and examples.

    Example:
        >>> phonons = load_nacl_phonons()
        >>> pmv.phonon_dos(phonons.projected_dos([10, 10, 10]), project="element")
    """
    return kemio.Phonons.from_phonopy_yaml(
        f"{TEST_FILES}/phonons/NaCl/phonopy_disp.yaml.xz",
        force_sets=f"{TEST_FILES}/phonons/NaCl/force_sets.dat",
        symmetrize=False,
    )
