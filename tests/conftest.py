from __future__ import annotations

import json
import lzma
from glob import glob
from typing import TYPE_CHECKING, Any, TypedDict

import kemio
import numpy as np
import pandas as pd
import pytest

from pymatviz.data import TEST_FILES
from pymatviz.phonons import PhononDBDoc


if TYPE_CHECKING:
    from collections.abc import Sequence


# random regression data
np_rng = np.random.default_rng(seed=0)
xs = np_rng.random(100)
y_pred = xs + 0.1 * np_rng.normal(size=100)
y_true = xs + 0.1 * np_rng.normal(size=100)

df_regr = pd.DataFrame(dict(y_true=y_true, y_pred=y_pred))  # regression
DfOrArrays = tuple[pd.DataFrame | None, str | np.ndarray, str | np.ndarray]


@pytest.fixture(params=[(None, y_true, y_pred), (df_regr, *df_regr.columns[:2])])
def df_or_arrays(request: pytest.FixtureRequest) -> DfOrArrays:
    return request.param


@pytest.fixture
def spg_symbols() -> list[str]:
    symbols = ["C2/m", "C2/m", "Fm-3m", "C2/m", "Cmc2_1", "P4/nmm", "P-43m", "P-43m"]
    symbols += ["P6_3mc", "P-43m", "P6_3mc", "Cmcm", "P2_1/m", "I2_13", "P-6m2"]
    return symbols


FCC_SITES = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])


def cubic_structure(
    a_len: float, species: Sequence[Any], frac_coords: Any
) -> kemio.Structure:
    """Cubic kemio Structure from fractional coordinates."""
    coords = np.reshape(np.asarray(frac_coords, dtype=float), (-1, 3))
    return kemio.Structure(np.eye(3) * a_len, list(species), coords, fractional=True)


def rocksalt(a_len: float = 5.64) -> kemio.Structure:
    """Conventional 8-site rocksalt NaCl cell (every site octahedral)."""
    return cubic_structure(
        a_len, ["Na"] * 4 + ["Cl"] * 4, [*FCC_SITES, *(FCC_SITES + 0.5) % 1]
    )


SI2_STRUCT = kemio.Structure(
    [[3.8, 0, 0], [1.9, 3.3, 0], [0, -2.2, 3.1]],
    ["Si", "Si"],
    [[0, 0, 0], [0.75, 0.5, 0.75]],
    fractional=True,
)
si2_ru2_pr2_struct = kemio.Structure(
    np.diag([4.192, 4.192, 6.88]),
    ["Si", "Si", "Ru", "Ru", "Pr", "Pr"],
    [
        [0.25, 0.25, 0.173],
        [0.75, 0.75, 0.827],
        [0.75, 0.25, 0],
        [0.25, 0.75, 0],
        [0.25, 0.25, 0.676],
        [0.75, 0.75, 0.324],
    ],
    fractional=True,
)
SI_STRUCTS = (SI2_STRUCT, si2_ru2_pr2_struct)


@pytest.fixture
def structures() -> tuple[kemio.Structure, kemio.Structure]:
    """Two silicon-containing kemio structures (Si2, Pr2Ru2Si2)."""
    return SI_STRUCTS


@pytest.fixture
def ase_atoms() -> tuple[Any, Any]:
    """SI_STRUCTS as ASE Atoms, for tests of ASE inputs."""
    pytest.importorskip("ase")
    from kemio.interop import structure_to_ase

    return structure_to_ase(SI_STRUCTS[0]), structure_to_ase(SI_STRUCTS[1])


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
    return pd.DataFrame(np_rng.random(size=(30, 5)), columns=pd.Index([*"ABCDE"]))


@pytest.fixture(scope="session")
def phonopy_nacl() -> Any:
    """Phonopy object of NaCl 2x2x2 without symmetrizing fc2, for tests of phonopy
    inputs (everything else uses load_nacl_phonons).
    """
    phonopy = pytest.importorskip("phonopy")

    return phonopy.load(
        f"{TEST_FILES}/phonons/NaCl/phonopy_disp.yaml.xz",
        force_sets_filename=f"{TEST_FILES}/phonons/NaCl/force_sets.dat",
        symmetrize_fc=False,
        produce_fc=True,
    )


@pytest.fixture
def fe3co4_disordered() -> kemio.Structure:
    """Disordered Fe3C-O2 structure (Fe:C 3:1 on the first site)."""
    from pymatviz.data import fe3co4_disordered

    return fe3co4_disordered


class BandsDoses(TypedDict):
    """Phonon band structures and DOS by label."""

    bands: dict[str, kemio.PhononBands]
    doses: dict[str, kemio.PhononDos]


def load_phonon_doc(path: str) -> PhononDBDoc:
    """Phonon DB doc (xz-compressed pymatgen/monty JSON) with kemio objects."""
    with lzma.open(path, mode="rt") as file:
        return PhononDBDoc.from_dict(json.load(file))


@pytest.fixture
def phonon_bands_doses_mp_2758() -> BandsDoses:
    docs = {
        label: load_phonon_doc(f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-{model}.json.xz")
        for label, model in (("DFT", "pbe"), ("MACE", "mace-y7uhwpje"))
    }
    return {
        "bands": {label: doc.phonon_bandstructure for label, doc in docs.items()},
        "doses": {label: doc.phonon_dos for label, doc in docs.items()},
    }


@pytest.fixture
def phonon_doses() -> dict[str, kemio.PhononDos]:
    paths = glob(f"{TEST_FILES}/phonons/mp-*-pbe.json.xz")
    assert len(paths) >= 2
    return {
        path.split("/")[-1].split("-pbe")[0]: load_phonon_doc(path).phonon_dos
        for path in paths
    }


@pytest.fixture
def complete_phonon_dos() -> kemio.PhononDos:
    """Synthetic site-projected phonon DOS of NaCl."""
    struct = kemio.Structure(
        np.eye(3) * 5.64, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]], fractional=True
    )
    rng = np.random.default_rng(seed=0)
    site_densities = rng.random((2, 100))
    return kemio.PhononDos(
        np.linspace(0, 10, 100),
        site_densities.sum(axis=0),
        projected_dos=site_densities,
        structure=struct,
    )
