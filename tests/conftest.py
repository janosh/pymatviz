from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Generator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pytest
from pymatgen.core import Lattice, Structure


if TYPE_CHECKING:
    import plotly.graph_objects as go


# if platform is windows, set matplotlib backend to "Agg" to fix
# _tkinter.TclError: Can't find a usable init.tcl in the following directories
# https://github.com/orgs/community/discussions/26434
if platform.system() == "Windows":
    import matplotlib as mpl

    mpl.use("Agg")


# random regression data
np.random.seed(42)
xs = np.random.rand(100)
y_pred = xs + 0.1 * np.random.normal(size=100)
y_true = xs + 0.1 * np.random.normal(size=100)

# random classification data
y_binary = np.random.choice([0, 1], 100)
y_proba = np.clip(y_binary - 0.1 * np.random.normal(scale=5, size=100), 0.2, 0.9)


df = pd.DataFrame(dict(y_true=y_true, y_pred=y_pred))
df_x_y = [(None, y_true, y_pred), (df, *df.columns[:2])]

df_clf = pd.DataFrame(dict(y_binary=y_binary, y_proba=y_proba))
df_x_y_clf = [(None, y_binary, y_proba), (df_clf, *df_clf.columns[:2])]


@pytest.fixture(autouse=True)
def _run_around_tests() -> Generator[None, None, None]:
    # runs before each test

    yield

    # runs after each test
    plt.close()


@pytest.fixture()
def spg_symbols() -> list[str]:
    symbols = "C2/m C2/m Fm-3m C2/m Cmc2_1 P4/nmm P-43m P-43m P6_3mc".split()
    symbols += "P-43m P6_3mc Cmcm P2_1/m I2_13 P-6m2".split()
    return symbols


@pytest.fixture()
def structures() -> list[Structure]:
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
    Si2_Ru2_Pr2 = Structure(lattice, "Si Si Ru Ru Pr Pr".split(), coords)
    return [Si2, Si2_Ru2_Pr2]


@pytest.fixture()
def plotly_scatter() -> go.Figure:
    xs = np.arange(7)
    y1 = xs**2
    y2 = xs**0.5
    return px.scatter(x=xs, y=[y1, y2])
