import subprocess
from shutil import which

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from ml_matrics import ROOT


@pytest.fixture(autouse=True)
def run_around_tests():
    # Code that runs before each test

    yield

    # Code that runs after each test
    plt.close()


y_binary, y_proba, y_clf = pd.read_csv(f"{ROOT}/data/rand_clf.csv").to_numpy().T
xs, y_pred, y_true = pd.read_csv(f"{ROOT}/data/rand_regr.csv").to_numpy().T


pngquant, zopflipng = which("pngquant"), which("zopflipng")


def save_fixture(save_to: str) -> None:
    """Save a matplotlib figure to a specified fixture path.

    Raises:
        ValueError: save_to is not inside 'tests/fixtures/' directory.
    """
    if not save_to.startswith((f"{ROOT}/tests/fixtures/", "tests/fixtures/")):
        raise ValueError(f"{save_to=} must point at 'tests/fixtures/'")

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
