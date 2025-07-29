from __future__ import annotations

import copy
import json
from glob import glob
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytest
from monty.io import zopen
from monty.json import MontyDecoder, MSONable
from plotly.subplots import make_subplots
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands
from pymatgen.phonon.dos import PhononDos

from pymatviz.utils.testing import TEST_FILES, load_phonopy_nacl


if TYPE_CHECKING:
    import ase
    from phonopy import Phonopy


# random regression data
np_rng = np.random.default_rng(seed=0)
xs = np_rng.random(100)
y_pred = xs + 0.1 * np_rng.normal(size=100)
y_true = xs + 0.1 * np_rng.normal(size=100)

# random classification data
y_binary = np_rng.choice([0, 1], 100)
y_proba = np.clip(y_binary - 0.1 * np_rng.normal(scale=5, size=100), 0.2, 0.9)


df_regr = pd.DataFrame(dict(y_true=y_true, y_pred=y_pred))  # regression
DfOrArrays = tuple[pd.DataFrame | None, str | np.ndarray, str | np.ndarray]


@pytest.fixture(params=[(None, y_true, y_pred), (df_regr, *df_regr.columns[:2])])
def df_or_arrays(request: pytest.FixtureRequest) -> DfOrArrays:
    return request.param


df_clf = pd.DataFrame(dict(y_binary=y_binary, y_proba=y_proba))
df_x_y_clf = [(None, y_binary, y_proba), (df_clf, *df_clf.columns[:2])]


@pytest.fixture
def spg_symbols() -> list[str]:
    symbols = ["C2/m", "C2/m", "Fm-3m", "C2/m", "Cmc2_1", "P4/nmm", "P-43m", "P-43m"]
    symbols += ["P6_3mc", "P-43m", "P6_3mc", "Cmcm", "P2_1/m", "I2_13", "P-6m2"]
    return symbols


SI2_STRUCT = Structure(
    lattice=[[3.8, 0, 0], [1.9, 3.3, 0], [0, -2.2, 3.1]],
    species=["Si4+", "Si4+"],
    coords=[[0, 0, 0], [0.75, 0.5, 0.75]],
)

lattice = Lattice.tetragonal(4.192, 6.88)
si2_ru2_pr2_struct = Structure(
    lattice=Lattice.tetragonal(4.192, 6.88),
    species=["Si", "Si", "Ru", "Ru", "Pr", "Pr"],
    coords=[
        [0.25, 0.25, 0.173],
        [0.75, 0.75, 0.827],
        [0.75, 0.25, 0],
        [0.25, 0.75, 0],
        [0.25, 0.25, 0.676],
        [0.75, 0.75, 0.324],
    ],
)
SI_STRUCTS = (SI2_STRUCT, si2_ru2_pr2_struct)
SI_ATOMS = tuple(map(AseAtomsAdaptor.get_atoms, SI_STRUCTS))


@pytest.fixture
def structures() -> tuple[Structure, Structure]:
    return tuple(struct.copy() for struct in SI_STRUCTS)


@pytest.fixture
def ase_atoms() -> tuple[ase.Atoms, ase.Atoms]:
    return tuple(copy.copy(atoms) for atoms in SI_ATOMS)


@pytest.fixture
def plotly_scatter_two_ys() -> go.Figure:
    xs = np.arange(7)
    y1 = xs**2
    y2 = xs**0.5
    return px.scatter(x=xs, y=[y1, y2])


@pytest.fixture
def plotly_scatter() -> go.Figure:
    fig = go.Figure(go.Scatter(x=[1, 10, 100], y=np.array([10, 100, 1000]) + 1))
    fig.add_scatter(x=[1, 10, 100], y=[1, 10, 100])
    return fig


@pytest.fixture
def plotly_faceted_scatter() -> go.Figure:
    fig = make_subplots(rows=1, cols=2)
    fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6], row=1, col=1)
    fig.add_scatter(x=[1, 2, 3], y=[4, 5, 6], row=1, col=2)
    return fig


@pytest.fixture
def glass_formulas() -> list[str]:
    """First 20 materials in the MatBench glass dataset.

    Equivalent to:
    from matminer.datasets import load_dataset

    load_dataset("matbench_glass").composition.head(20)
    """
    return list(
        (  # noqa: SIM905
            "Al Al(NiB)2 Al10Co21B19 Al10Co23B17 Al10Co27B13 Al10Co29B11 Al10Co31B9 "
            "Al10Co33B7 Al10Cr3Si7 Al10Fe23B17 Al10Fe27B13 Al10Fe31B9 Al10Fe33B7 "
            "Al10Ni23B17 Al10Ni27B13 Al10Ni29B11 Al10Ni31B9 Al10Ni33B7 Al11(CrSi2)3"
        ).split()
    )


@pytest.fixture
def df_float() -> pd.DataFrame:
    np_rng = np.random.default_rng(seed=0)
    return pd.DataFrame(np_rng.random(size=(30, 5)), columns=[*"ABCDE"])


@pytest.fixture
def df_mixed() -> pd.DataFrame:
    floats = np_rng.random(size=30)
    bools = np_rng.choice([True, False], size=30)
    strings = np_rng.choice([*"abcdef"], size=30)
    return pd.DataFrame(dict(floats=floats, bools=bools, strings=strings))


@pytest.fixture(scope="session")
def phonopy_nacl() -> Phonopy:
    """Return Phonopy class instance of NaCl 2x2x2 without symmetrizing fc2."""
    return load_phonopy_nacl()


@pytest.fixture
def fe3co4_disordered() -> Structure:
    """Disordered Fe3C-O2 structure without site properties. This structure has
    disordered sites with Fe:C ratio of 3:1.
    """
    from pymatviz.structure import fe3co4_disordered

    return fe3co4_disordered


@pytest.fixture
def fe3co4_disordered_with_props(fe3co4_disordered: Structure) -> Structure:
    """Disordered Fe3C-O2 structure with magnetic moment and force site properties."""
    site_props = {
        "magmom": [[0, 0, 1], [0, 0, -1]],
        "force": [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]],
    }
    return fe3co4_disordered.copy(site_properties=site_props)


BandsDoses = dict[str, dict[str, PhononBands | PhononDos]]
bs_key, dos_key = "phonon_bandstructure", "phonon_dos"
# enable loading PhononDBDocParsed with @module set to uninstalled ffonons.dbs.phonondb
# by changing to identical dataclass in pymatviz.phonons module
MSONable.REDIRECT["ffonons.dbs.phonondb"] = {
    "PhononDBDocParsed": {
        "@class": "PhononDBDoc",
        "@module": "pymatviz.phonons.helpers",
    }
}
MSONable.REDIRECT["atomate2.common.schemas.phonons"] = {
    "PhononBSDOSDoc": {"@class": "PhononDBDoc", "@module": "pymatviz.phonons.helpers"}
}


@pytest.fixture
def phonon_bands_doses_mp_2758() -> BandsDoses:
    with zopen(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-pbe.json.xz") as file:
        dft_doc = json.loads(file.read(), cls=MontyDecoder)

    with zopen(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-mace-y7uhwpje.json.xz") as file:
        ml_doc = json.loads(file.read(), cls=MontyDecoder)

    bands = {"DFT": getattr(dft_doc, bs_key), "MACE": getattr(ml_doc, bs_key)}
    doses = {"DFT": getattr(dft_doc, dos_key), "MACE": getattr(ml_doc, dos_key)}
    return {"bands": bands, "doses": doses}


@pytest.fixture
def phonon_bands_doses_mp_2667() -> BandsDoses:
    # with zopen(f"{TEST_FILES}/phonons/mp-2691-Cd4Se4-pbe.json.xz") as file:
    with zopen(f"{TEST_FILES}/phonons/mp-2667-Cs1Au1-pbe.json.xz") as file:
        return json.loads(file.read(), cls=MontyDecoder)


@pytest.fixture
def phonon_doses() -> dict[str, PhononDos]:
    paths = glob(f"{TEST_FILES}/phonons/mp-*-pbe.json.xz")
    assert len(paths) >= 2
    return {
        path.split("/")[-1].split("-pbe")[0]: getattr(
            json.loads(zopen(path).read(), cls=MontyDecoder), dos_key
        )
        for path in paths
    }
