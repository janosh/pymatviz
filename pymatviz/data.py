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

    Args:
        n_samples (int): Number of samples to generate. Defaults to 500.
        true_mean (float): Mean of the normally distributed y_true. Defaults to 5.
        true_std (float): Standard deviation of y_true. Defaults to 4.
        pred_slope (float): Slope of y_pred vs y_true. Defaults to 1.2.
        pred_intercept (float): Intercept of y_pred vs y_true. Defaults to -2.
        seed (int): Random seed. Defaults to 0.

    Returns:
        RegressionData: y_true, y_pred (linear in y_true plus unit Gaussian noise)
            and positive y_std that grows with the absolute residuals.
    """
    rng = np.random.default_rng(seed=seed)
    y_true = rng.normal(loc=true_mean, scale=true_std, size=n_samples)
    noise = rng.normal(loc=0, scale=1, size=n_samples)
    y_pred = pred_slope * y_true + pred_intercept + noise

    abs_residuals = np.abs(y_true - y_pred)
    random_spread = 0.8 * rng.exponential(scale=1, size=n_samples)
    y_std = 0.5 + 0.3 * abs_residuals + random_spread

    return RegressionData(y_true=y_true, y_pred=y_pred, y_std=y_std)


TEST_FILES: str = f"{ROOT}/tests/files"

# === Disordered structures for tests and examples ===


fe3co4_disordered = kemio.Structure(
    np.eye(3) * 5,
    [{"Fe": 0.75, "C": 0.25}, "O"],
    [[0, 0, 0], [0.5, 0.5, 0.5]],
    fractional=True,
)

disordered_demo_structures: Final[dict[str, kemio.Structure]] = {
    "BaTi₀.₆Zr₀.₄O₃": kemio.Structure(
        np.eye(3) * 4.0,
        ["Ba", {"Ti": 0.6, "Zr": 0.4}, "O", "O", "O"],
        [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        fractional=True,
    ),
    # Li-ion cathode with 25% Li vacancies
    "Li₀.₇₅CoO₂ (with vacancies)": kemio.Structure(
        kemio.lattice_from_parameters(2.82, 2.82, 14.05, 90, 90, 120, c_along_z=True),
        [{"Li": 0.75}, "Co", "O", "O"],
        [(0, 0, 0), (0, 0, 0.5), (0, 0, 0.25), (0, 0, 0.75)],
        fractional=True,
    ),
    # high-entropy alloy with 3 species on one site
    "Fe₀.₅Ni₀.₃Cr₀.₂Al": kemio.Structure(
        np.eye(3) * 3.6,
        [{"Fe": 0.5, "Ni": 0.3, "Cr": 0.2}, "Al"],
        [(0, 0, 0), (0.5, 0.5, 0.5)],
        fractional=True,
    ),
    # solid solution with disordered A and (ternary) B sites
    "Ca₀.₆Sr₀.₄Ti₀.₈Zr₀.₁₅Hf₀.₀₅O₃": kemio.Structure(
        np.eye(3) * 5.0,
        [{"Ca": 0.6, "Sr": 0.4}, {"Ti": 0.8, "Zr": 0.15, "Hf": 0.05}, "O", "O", "O"],
        [(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
        fractional=True,
    ),
}


def load_nacl_phonons() -> kemio.Phonons:
    """Harmonic phonons of NaCl (2x2x2 supercell, force constants not symmetrized)
    from the test phonopy files, read by kemio, for tests and examples.

    Example:
        >>> import pymatviz as pmv
        >>> from pymatviz.data import load_nacl_phonons
        >>> phonons = load_nacl_phonons()
        >>> pmv.phonon_dos(phonons.projected_dos([10, 10, 10]), project="element")
    """
    return kemio.Phonons.from_phonopy_yaml(
        f"{TEST_FILES}/phonons/NaCl/phonopy_disp.yaml.xz",
        force_sets=f"{TEST_FILES}/phonons/NaCl/force_sets.dat",
        symmetrize=False,
    )
