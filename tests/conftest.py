from __future__ import annotations

import platform
from typing import TYPE_CHECKING, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest
from pymatgen.core import Lattice, Structure


if TYPE_CHECKING:
    from collections.abc import Generator


# if platform is windows, set matplotlib backend to "Agg" to fix
# _tkinter.TclError: Can't find a usable init.tcl in the following directories
# https://github.com/orgs/community/discussions/26434
if platform.system() == "Windows":
    import matplotlib as mpl

    mpl.use("Agg")


# random regression data
np_rng = np.random.default_rng(seed=0)
xs = np_rng.random(100)
y_pred = xs + 0.1 * np_rng.normal(size=100)
y_true = xs + 0.1 * np_rng.normal(size=100)

# random classification data
y_binary = np_rng.choice([0, 1], 100)
y_proba = np.clip(y_binary - 0.1 * np_rng.normal(scale=5, size=100), 0.2, 0.9)


df_regr = pd.DataFrame(dict(y_true=y_true, y_pred=y_pred))  # regression
DfOrArrays = tuple[
    Union[pd.DataFrame, None], Union[str, np.ndarray], Union[str, np.ndarray]
]


@pytest.fixture(params=[(None, y_true, y_pred), (df_regr, *df_regr.columns[:2])])
def df_or_arrays(request: pytest.FixtureRequest) -> DfOrArrays:
    return request.param


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
def plotly_scatter_two_ys() -> go.Figure:
    xs = np.arange(7)
    y1 = xs**2
    y2 = xs**0.5
    return px.scatter(x=xs, y=[y1, y2])


@pytest.fixture()
def plotly_scatter() -> go.Figure:
    fig = go.Figure(go.Scatter(x=[1, 10, 100], y=[10, 100, 1000]))
    fig.add_scatter(x=[1, 10, 100], y=[1, 10, 100])
    return fig


@pytest.fixture()
def matplotlib_scatter() -> plt.Figure:
    fig, ax = plt.subplots()
    ax.plot([1, 10, 100], [10, 100, 1000])
    ax.plot([1, 10, 100], [1, 10, 100])
    return fig


@pytest.fixture()
def glass_formulas() -> list[str]:
    """First 20 materials in the Matbench glass dataset.

    from matminer.datasets import load_dataset

    load_dataset("matbench_glass").composition.head(20)
    """
    return (
        "Al Al(NiB)2 Al10Co21B19 Al10Co23B17 Al10Co27B13 Al10Co29B11 Al10Co31B9 "
        "Al10Co33B7 Al10Cr3Si7 Al10Fe23B17 Al10Fe27B13 Al10Fe31B9 Al10Fe33B7 "
        "Al10Ni23B17 Al10Ni27B13 Al10Ni29B11 Al10Ni31B9 Al10Ni33B7 Al11(CrSi2)3"
    ).split()


@pytest.fixture()
def df_float() -> pd.DataFrame:
    return pd.DataFrame(np_rng.random(size=(30, 5)), columns=[*"ABCDE"])


@pytest.fixture()
def df_mixed() -> pd.DataFrame:
    floats = np_rng.random(size=30)
    bools = np_rng.choice([True, False], size=30)
    strings = np_rng.choice([*"abcdef"], size=30)
    return pd.DataFrame(dict(floats=floats, bools=bools, strings=strings))
