from __future__ import annotations

import subprocess
from shutil import which

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pytest
from pymatgen.core import Lattice, Structure

from pymatviz import ROOT


# random regression data
np.random.seed(42)
xs = np.random.rand(100)
y_pred = xs + 0.1 * np.random.normal(size=100)
y_true = xs + 0.1 * np.random.normal(size=100)

# random classification data
y_binary = np.random.choice([0, 1], 100)
y_proba = np.clip(y_binary - 0.1 * np.random.normal(scale=5, size=100), 0.2, 0.9)
y_clf = y_proba.round()


@pytest.fixture(autouse=True)
def run_around_tests():
    # runs before each test

    yield

    # runs after each test
    plt.close()


@pytest.fixture
def spg_symbols():
    symbols = "C2/m C2/m Fm-3m C2/m Cmc2_1 P4/nmm P-43m P-43m P6_3mc".split()
    symbols += "P-43m P6_3mc Cmcm P2_1/m I2_13 P-6m2".split()
    return symbols


@pytest.fixture
def structures():
    coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
    lattice = [[3.8, 0, 0], [1.9, 3.3, 0], [0, -2.2, 3.1]]
    Si2 = Structure(lattice, ["Si4+", "Si4+"], coords)

    coords = [
        [0.25, 0.25, 0.173],
        [0.75, 0.75, 0.827],
        [0.75, 0.25, 0],
        [0.25, 0.75, 0],
        [0.25, 0.25, 0.676],
        [0.75, 0.75, 0.324],
    ]
    lattice = Lattice.tetragonal(4.192, 6.88)
    Si2Ru2Pr2 = Structure(lattice, ["Si", "Si", "Ru", "Ru", "Pr", "Pr"], coords)
    return [Si2, Si2Ru2Pr2]


@pytest.fixture
def plotly_scatter():
    xs = np.arange(7)
    y1 = xs**2
    y2 = xs**0.5
    fig = px.scatter(x=xs, y=[y1, y2])
    return fig


def save_reference_img(save_to: str) -> None:
    """Save a matplotlib figure to a specified fixture path.

    Raises:
        ValueError: save_to is not inside 'tests/fixtures/' directory.
    """
    if not save_to.startswith((f"{ROOT}/tests/fixtures/", "tests/fixtures/")):
        raise ValueError(f"{save_to=} must point at 'tests/fixtures/'")

    pngquant, zopflipng = which("pngquant"), which("zopflipng")

    print(
        f"created new fixture {save_to=}, image comparison will run for real on "
        "subsequent test runs unless fixture is deleted"
    )
    plt.savefig(save_to)
    plt.close()

    if not pngquant:
        return print("Warning: pngquant not installed. Cannot compress new fixture.")
    if not zopflipng:
        return print("Warning: zopflipng not installed. Cannot compress new fixture.")

    subprocess.run(
        f"{pngquant} 32 --skip-if-larger --ext .png --force".split() + [save_to],
        check=False,
        capture_output=True,
    )
    subprocess.run(
        [zopflipng, "-y", save_to, save_to],
        check=True,
        capture_output=True,
    )
