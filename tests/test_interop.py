"""pymatgen, ASE and phonopy inputs are routed through matscikit (and match pymatgen,
matminer or phonopy references where those are the ground truth). Every other test
uses matscikit inputs. Optional libraries are imported inside each test via
pytest.importorskip, so this module runs with any subset of them installed.
"""

from __future__ import annotations

import json
import lzma
from typing import TYPE_CHECKING, Any

import matscikit as mk
import numpy as np
import pandas as pd
import pytest

import pymatviz as pmv
from pymatviz import chem_env
from pymatviz.composition import count_elements, count_formulas
from pymatviz.coordination.helpers import coordination_nums_in_structure
from pymatviz.enums import ElemCountMode
from pymatviz.rdf.figures import rdf_plot
from pymatviz.rdf.helpers import calculate_rdf
from pymatviz.structures import (
    normalize_spacegroups,
    normalize_structures,
    site_symbols,
    structure_dict,
    to_structure,
)
from pymatviz.sunburst import spacegroup_sunburst
from pymatviz.utils.testing import TEST_FILES
from pymatviz.widgets import (
    BandsAndDosWidget,
    BrillouinZoneWidget,
    CompositionWidget,
    PeriodicTableWidget,
    RdfPlotWidget,
    ScatterPlotWidget,
    StructureWidget,
    TrajectoryWidget,
    XrdWidget,
)
from pymatviz.widgets._normalize import (
    normalize_convex_hull_entries,
    normalize_xrd_pattern,
    structure_to_dict,
)
from tests.conftest import SI_STRUCTS
from tests.widgets.conftest import decode_envelope


if TYPE_CHECKING:
    from collections.abc import Callable, Hashable
    from typing import Literal


# === Structures, compositions, symmetry, RDF, coordination, chemical environments ===


H2O_SPECIES = ["H", "H", "O"]
H2O_COORDS = [[0, 0, 0], [0, 0, 1.5], [0, 0, 0.75]]


def pmg_structure(struct: mk.Structure) -> Any:
    """Pymatgen Structure of a matscikit Structure (skips without pymatgen)."""
    pytest.importorskip("pymatgen")
    from matscikit.interop import structure_to_pymatgen_dict
    from pymatgen.core import Structure

    return Structure.from_dict(structure_to_pymatgen_dict(struct))


@pytest.mark.parametrize(
    ("kind", "expected_keys"),
    [
        ("structure", ["Si2"]),
        ("istructure", ["Si2"]),
        ("atoms", ["Si2"]),
        ("imolecule", ["H2 O1"]),
        ("list", ["1 Si2", "2 H2 O1"]),  # 1-based index and formula
        ("dict", ["si", "h2o"]),
        ("series", ["s"]),
    ],
)
def test_normalize_structures_pymatgen_ase(
    kind: str, expected_keys: list[Hashable], ase_atoms: tuple[Any, Any]
) -> None:
    """Pymatgen Structures/Molecules and ASE Atoms normalize to matscikit."""
    pytest.importorskip("pymatgen")
    from pymatgen.core import IMolecule, IStructure, Molecule

    si_istruct = IStructure.from_sites(pmg_structure(SI_STRUCTS[0]))
    h2o = Molecule(H2O_SPECIES, H2O_COORDS)
    input_raw = {
        "structure": pmg_structure(SI_STRUCTS[0]),
        "istructure": si_istruct,
        "atoms": ase_atoms[0],
        "imolecule": IMolecule.from_sites(h2o),
        "list": [ase_atoms[0], h2o],
        "dict": {"si": ase_atoms[1], "h2o": h2o},
        "series": pd.Series([si_istruct], index=["s"]),
    }[kind]
    struct_dict = normalize_structures(input_raw)
    assert list(struct_dict) == expected_keys
    assert all(isinstance(struct, mk.Structure) for struct in struct_dict.values())


@pytest.mark.parametrize("source", ["pymatgen", "ase"])
def test_to_structure_geometry(source: str, ase_atoms: tuple[Any, Any]) -> None:
    """Pymatgen and ASE inputs keep lattice, positions, species and pbc."""
    obj = pmg_structure(SI_STRUCTS[0]) if source == "pymatgen" else ase_atoms[0]
    struct = to_structure(obj)
    np.testing.assert_allclose(struct.lattice, SI_STRUCTS[0].lattice, atol=1e-12)
    np.testing.assert_allclose(struct.positions, SI_STRUCTS[0].positions, atol=1e-12)
    assert struct.symbols == ["Si", "Si"]
    assert struct.pbc == (True, True, True)


def test_to_structure_pymatgen_species_and_molecules() -> None:
    """Oxidation states sum per element, dummy species become vacancies, molecules
    turn off periodic boundaries.
    """
    pytest.importorskip("pymatgen")
    from pymatgen.core import (
        DummySpecies,
        Element,
        IStructure,
        Lattice,
        Molecule,
        Species,
    )

    site_species = [
        {Species("Fe", 2): 0.25, Species("Fe", 3): 0.5, DummySpecies("X"): 0.25},
        Element("O"),
    ]
    pmg_struct = IStructure(Lattice.cubic(3), site_species, [[0, 0, 0], [0.5] * 3])
    struct = to_structure(pmg_struct)
    assert struct.site_occupancies == [{"Fe": 0.75}, {"O": 1.0}]
    assert not struct.is_ordered

    h2o = Molecule(H2O_SPECIES, H2O_COORDS)
    mol = to_structure(h2o)
    assert mol.pbc == (False, False, False)
    np.testing.assert_allclose(mol.positions, h2o.cart_coords)

    # a vacancy-only site stays a (vacant) site, so site indices match pymatgen's
    vacant = to_structure(
        IStructure(Lattice.cubic(3), [DummySpecies("X"), "Fe"], [[0, 0, 0], [0.5] * 3])
    )
    assert vacant.site_occupancies == [{}, {"Fe": 1.0}]
    assert site_symbols(vacant) == ["X", "Fe"]
    assert structure_dict(vacant)["sites"][0]["label"] == "X0+"


def test_structure_dict_pymatgen_ase(ase_atoms: tuple[Any, Any]) -> None:
    """Pymatgen objects keep their as_dict (oxidation states included), ASE Atoms
    become periodic pymatgen-style dicts.
    """
    pmg_struct = pmg_structure(SI_STRUCTS[0]).add_oxidation_state_by_element({"Si": 4})
    assert structure_dict(pmg_struct) == pmg_struct.as_dict()  # keeps Si4+
    periodic = structure_dict(ase_atoms[0])
    assert periodic["lattice"]["pbc"] == [True, True, True]
    assert [site["species"] for site in periodic["sites"]] == [
        [{"element": "Si", "occu": 1}]
    ] * 2


@pytest.mark.parametrize("kind", ["pymatgen", "ase", "ase_subclass"])
@pytest.mark.parametrize("as_series", [False, True])
def test_normalize_spacegroups_pymatgen_ase(
    kind: str, as_series: bool, ase_atoms: tuple[Any, Any]
) -> None:
    """Space groups of pymatgen and ASE inputs (subclasses included) match moyopy."""
    pytest.importorskip("pymatgen")
    from ase import Atoms
    from moyopy import MoyoDataset
    from moyopy.interface import MoyoAdapter

    class DerivedAtoms(Atoms):
        """Exercise adapter-free conversion of caller-defined ASE subclasses."""

    pmg_structs = tuple(map(pmg_structure, SI_STRUCTS))
    structures = {
        "pymatgen": pmg_structs,
        "ase": ase_atoms,
        "ase_subclass": tuple(DerivedAtoms(atoms) for atoms in ase_atoms),
    }[kind]
    expected = pd.Series(
        [MoyoDataset(MoyoAdapter.from_py_obj(struct)).number for struct in pmg_structs]
    )
    data: tuple | pd.Series = structures
    if as_series:
        index = pd.Index(["sample-a", "sample-b"], name="sample")
        data = pd.Series(list(structures), index=index, name="spacegroup")
        expected.index, expected.name = index, "spacegroup"
    pd.testing.assert_series_equal(normalize_spacegroups(data), expected)


def test_normalize_spacegroups_disordered_matches_pymatgen() -> None:
    """Disordered structures get the space groups of pymatgen's SpacegroupAnalyzer."""
    pytest.importorskip("pymatgen")
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    from pymatviz.utils.testing import disordered_demo_structures

    structs = list(disordered_demo_structures.values())
    expected = [
        SpacegroupAnalyzer(pmg_structure(struct)).get_space_group_number()
        for struct in structs
    ]
    assert normalize_spacegroups(structs).tolist() == expected == [221, 191, 221, 221]


def test_spacegroup_sunburst_rejects_ase_default_pbc(
    structures: tuple[mk.Structure, mk.Structure],
) -> None:
    """ASE Atoms with the default pbc=False are rejected with how to fix it."""
    pytest.importorskip("ase")
    from ase import Atoms

    atoms = Atoms("Si", cell=[3, 3, 3])  # pbc defaults to False
    with pytest.raises(
        TypeError, match=r"periodic along a, b and c.*atoms\.pbc = True"
    ):
        spacegroup_sunburst([*structures, atoms])
    assert spacegroup_sunburst([*structures, Atoms("Si", cell=[3, 3, 3], pbc=True)])


def test_site_environments_chemenv_matches_pymatgen() -> None:
    """ChemEnv symbols equal pymatgen's on the input cell (no refinement)."""
    pytest.importorskip("pymatgen")
    from pymatgen.analysis.chemenv.coordination_environments import (
        chemenv_strategies,
        coordination_geometry_finder,
        structure_environments,
    )
    from pymatgen.core import Structure

    pmg_struct = Structure.from_file(
        f"{TEST_FILES}/structures/mp-1207297-Ac2Br2O1-tetragonal.json.gz"
    )
    finder = coordination_geometry_finder.LocalGeometryFinder()
    finder.setup_parameters(structure_refinement="none")
    finder.setup_structure(pmg_struct)
    lse = structure_environments.LightStructureEnvironments.from_structure_environments(
        chemenv_strategies.SimplestChemenvStrategy(),
        finder.compute_structure_environments(),
    )
    expected = [envs[0]["ce_symbol"] for envs in lse.coordination_environments or []]
    (envs,) = chem_env.site_environments([to_structure(pmg_struct)])
    assert [symbol for _, symbol in envs] == expected
    assert [cn for cn, _ in envs] == [int(sym.split(":")[1]) for sym in expected]


@pytest.mark.parametrize("kind", ["pymatgen", "istructure", "ase"])
def test_calculate_rdf_pymatgen_ase(kind: str, ase_atoms: tuple[Any, Any]) -> None:
    """Pymatgen (I)Structures and ASE Atoms give the matscikit Structure's RDFs."""
    if kind == "ase":
        obj = ase_atoms[0]
    else:
        obj = pmg_structure(SI_STRUCTS[0])
        if kind == "istructure":
            from pymatgen.core import IStructure

            obj = IStructure.from_sites(list(obj.sites))
    radii, rdf = calculate_rdf(obj, cutoff=10, n_bins=100)
    radii_partial, rdf_partial = calculate_rdf(obj, "Si", "Si", 10, 100)
    for arr in (rdf, rdf_partial):
        assert arr.shape == (100,)
        assert arr[0] == 0
        assert np.all(arr >= 0)
    np.testing.assert_array_equal(radii_partial, radii)
    np.testing.assert_array_equal(rdf_partial, rdf)  # Si2 has only Si
    ref_radii, ref_rdf = calculate_rdf(SI_STRUCTS[0], cutoff=10, n_bins=100)
    np.testing.assert_allclose(radii, ref_radii, rtol=1e-12)
    if kind != "ase":  # ASE round-trip perturbs coordinates by ~1e-15 Å
        np.testing.assert_array_equal(rdf, ref_rdf)


def test_rdf_plot_full_ase(ase_atoms: tuple[Any, Any]) -> None:
    """One full RDF per ASE Atoms, matching calculate_rdf of the matscikit cell."""
    widget = rdf_plot(ase_atoms, mode="full", cutoff=5, bins=20, colors=["red"])
    assert len(widget.patterns) == 2
    radii, _g_r = calculate_rdf(SI_STRUCTS[0], cutoff=5, n_bins=20)
    # ASE Atoms round-trip perturbs coordinates by ~1e-15 Å, rarely flipping a bin
    np.testing.assert_allclose(widget.patterns[0]["pattern"]["r"], radii, rtol=1e-12)
    assert {entry["color"] for entry in widget.patterns} == {"red"}


@pytest.mark.parametrize(
    ("strategy", "as_molecule"),
    [
        *((cutoff, False) for cutoff in (2.5, 3.5, 6.0)),
        (3.5, True),
        ("crystal_nn", False),
        ("voronoi", False),
    ],
)
def test_coordination_nums_match_pymatgen(
    strategy: float | Literal["crystal_nn", "voronoi"], as_molecule: bool
) -> None:
    """Per-site CNs equal pymatgen's get_neighbors / CrystalNN / VoronoiNN."""
    pytest.importorskip("pymatgen")
    from pymatgen.analysis.local_env import CrystalNN, VoronoiNN
    from pymatgen.core import Lattice, Molecule, Structure

    rng = np.random.default_rng(seed=0)
    struct = Structure(Lattice.cubic(6.0), ["Na", "Cl"] * 10, rng.uniform(size=(20, 3)))
    system = Molecule.from_sites(struct) if as_molecule else struct
    cns = coordination_nums_in_structure(system, strategy=strategy, group_by="site")
    if isinstance(strategy, float):
        expected = [len(system.get_neighbors(site, r=strategy)) for site in system]  # ty: ignore[invalid-argument-type]
    else:
        nn = CrystalNN() if strategy == "crystal_nn" else VoronoiNN()
        expected = [len(nn.get_nn_info(struct, idx)) for idx in range(len(struct))]
    assert [cn for (cn,) in cns.values()] == expected
    assert sum(expected) > 0


@pytest.mark.parametrize(
    ("a_len", "n_sites", "cutoff_range", "num_points"),
    [(2.0, 1, (1, 5), 5), (6.0, 12, (0.5, 6), 40)],  # self-images at 2 and 4 Å
)
def test_coordination_vs_cutoff_line_matches_pymatgen(
    a_len: float, n_sites: int, cutoff_range: tuple[float, float], num_points: int
) -> None:
    """Mean CNs per element equal pymatgen's get_neighbors per site and cutoff."""
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure

    species = ["Si"] if n_sites == 1 else ["Si"] * 6 + ["O"] * 6
    frac_coords = (
        [[0, 0, 0]]
        if n_sites == 1
        else np.random.default_rng(seed=0).random((n_sites, 3))
    )
    structure = Structure(Lattice.cubic(a_len), species, frac_coords)
    expected = {
        elem: [
            float(
                np.mean(
                    [
                        len(structure.get_neighbors(site, r=cutoff))
                        for site in structure
                        if site.specie.symbol == elem
                    ]
                )
            )
            for cutoff in np.linspace(*cutoff_range, num_points)
        ]
        for elem in sorted(set(species))
    }
    widget = pmv.coordination_vs_cutoff_line(
        structure, cutoff_range=cutoff_range, num_points=num_points
    )
    assert {srs["label"]: srs["y"] for srs in widget.series} == expected


@pytest.mark.parametrize("count_mode", list(ElemCountMode))
def test_count_elements_pymatgen_compositions(count_mode: ElemCountMode) -> None:
    """Pymatgen Compositions, charged species included, count like formula strings."""
    pytest.importorskip("pymatgen")
    from pymatgen.core import Composition

    formulas = ["Fe2O3"] * 5 + ["Fe4P4O16"] * 3
    pd.testing.assert_series_equal(
        count_elements(list(map(Composition, formulas)), count_mode=count_mode),
        count_elements(formulas, count_mode=count_mode),
    )
    pd.testing.assert_series_equal(
        count_elements([Composition({"Fe3+": 2, "O2-": 3})], count_mode=count_mode),
        count_elements(["Fe2O3"], count_mode=count_mode),
    )


def test_count_formulas_and_chem_sys_hierarchy_pymatgen() -> None:
    """Pymatgen Compositions and Structures group like formulas in count_formulas
    and chem_sys_hierarchy.
    """
    pytest.importorskip("pymatgen")
    from pymatgen.core import Composition, Lattice, Structure

    df_out = count_formulas(["Fe2O3", Composition("Fe4O6"), Composition("FeO")])
    assert list(df_out.itertuples(index=False, name=None)) == [("binary", "Fe-O", 3)]
    df_out = count_formulas([Composition("Fe2O3")], group_by="formula")
    assert df_out["formula"].tolist() == ["Fe2 O3"]  # str(Composition) has amounts

    structure = Structure(Lattice.cubic(3), ["Fe", "O"], [[0, 0, 0], [0.5] * 3])
    widget = pmv.chem_sys_hierarchy([structure, Composition("LiPO4"), "FeO"])
    assert [(root["label"], root["children"]) for root in widget.data] == [
        ("binary", [{"id": "binary/Fe-O", "label": "Fe-O", "value": 2}]),
        ("ternary", [{"id": "ternary/Li-O-P", "label": "Li-O-P", "value": 1}]),
    ]


# === Phonons, XRD, notebook display and composition embeddings ===


WIDGET_MIME = "application/vnd.jupyter.widget-view+json"
SR_SE_DOC = f"{TEST_FILES}/phonons/mp-2758-Sr4Se4-pbe.json.xz"
BI2_ZR2_O7_CIF = f"{TEST_FILES}/xrd/Bi2Zr2O7-Fm3m-experimental-sqs.cif"


def load_raw_doc(path: str = SR_SE_DOC) -> dict[str, Any]:
    """Raw (pymatgen/monty JSON) phonon DB doc."""
    with lzma.open(path, mode="rt") as file:
        return json.load(file)


# === Phonons ===


def test_phonopy_inputs(phonopy_nacl: Any) -> None:
    """Phonopy band structures, TotalDos and projected DOS go through matscikit."""
    phonopy_nacl.run_mesh([5, 5, 5])
    phonopy_nacl.run_total_dos(freq_pitch=1, use_tetrahedron_method=False)
    total_dos = phonopy_nacl.total_dos
    phonopy_nacl.run_band_structure(
        paths=[np.linspace([0, 0, 0], [1 / 2, 0, 0], 11)], labels=["Γ", "X"]
    )
    band_structure = phonopy_nacl.band_structure
    assert total_dos is not None
    assert band_structure is not None

    widget = pmv.phonon_dos({"DFT": total_dos, "ML": total_dos})
    np.testing.assert_array_equal(
        widget.doses["DFT"]["frequencies"], total_dos.frequency_points
    )
    np.testing.assert_array_equal(widget.doses["ML"]["densities"], total_dos.dos)

    widget = pmv.band_structure(band_structs={"NaCl": band_structure})
    assert np.array(widget.band_structs["NaCl"]["bands"]).shape == (6, 11)
    widget = pmv.bands_and_dos(band_structs=band_structure, doses=total_dos)
    np.testing.assert_array_equal(widget.doses["densities"], total_dos.dos)
    np.testing.assert_array_equal(  # exact frequencies, no YAML round trip
        widget.band_structs["bands"], np.concatenate(band_structure.frequencies).T
    )
    assert widget.band_structs["labels_dict"] == {"Γ": [0, 0, 0], "X": [0.5, 0, 0]}

    phonopy_nacl.run_mesh([5, 5, 5], with_eigenvectors=True, is_mesh_symmetry=False)
    phonopy_nacl.run_total_dos(freq_pitch=1, use_tetrahedron_method=False)
    phonopy_nacl.run_projected_dos(freq_pitch=1, use_tetrahedron_method=False)
    widget = pmv.phonon_dos(phonopy_nacl, project="element")
    projected = phonopy_nacl.projected_dos.projected_dos
    np.testing.assert_allclose(
        np.add(widget.doses["Na"]["densities"], widget.doses["Cl"]["densities"]),
        projected.sum(axis=0),
        rtol=1e-12,
    )


@pytest.mark.parametrize("project", ["element", "site"])
def test_pymatgen_phonon_dos(
    complete_phonon_dos: mk.PhononDos, project: Literal["element", "site"]
) -> None:
    """Pymatgen PhononDos plots its total, CompletePhononDos projects per element or
    site like the equivalent matscikit PhononDos.
    """
    pytest.importorskip("pymatgen")
    from pymatgen.core import Lattice, Structure
    from pymatgen.phonon.dos import CompletePhononDos, PhononDos

    freqs, total = complete_phonon_dos.frequency_points, complete_phonon_dos.total_dos
    site_densities = complete_phonon_dos.projected_dos
    assert site_densities is not None
    pmg_struct = Structure(
        Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    pmg_dos = CompletePhononDos(
        pmg_struct,
        PhononDos(freqs, total),
        dict(zip(pmg_struct, site_densities, strict=True)),
    )
    widget = pmv.phonon_dos({"A": PhononDos(freqs, total), "B": pmg_dos})
    for label in "AB":
        np.testing.assert_array_equal(widget.doses[label]["densities"], total)
        assert "pdos" not in widget.doses[label]

    widget = pmv.phonon_dos(pmg_dos, project=project)
    expected = pmv.phonon_dos(complete_phonon_dos, project=project)
    assert set(widget.doses) == set(expected.doses)
    for name, doc in widget.doses.items():
        np.testing.assert_array_equal(
            doc["densities"], expected.doses[name]["densities"]
        )
    expected_names = {"Na", "Cl"} if project == "element" else {"Na0", "Cl1"}
    assert set(widget.doses) == expected_names | {"Total"}


def test_pymatgen_band_widgets_drop_eigendisplacements() -> None:
    """Pymatgen band structures convert without serializing eigendisplacements and
    without mutating the input.
    """
    pytest.importorskip("pymatgen")
    from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
    from pymatgen.phonon.dos import PhononDos

    raw = load_raw_doc()
    bands = PhononBandStructureSymmLine.from_dict(raw["phonon_bandstructure"])
    bands.eigendisplacements = np.ones((*bands.bands.shape, bands.nb_bands // 3, 3))
    assert "eigendisplacements" in bands.as_dict()

    class Unserializable:
        """Eigendisplacements that fail when as_dict() splits them into lists."""

        @property
        def real(self) -> None:
            """Raise like a (slow) list conversion would happen."""
            raise AssertionError("eigendisplacements were serialized")

    bands.eigendisplacements = sentinel = Unserializable()  # ty: ignore[invalid-assignment]
    widget = pmv.band_structure(band_structs=bands)
    assert "eigendisplacements" not in widget.band_structs
    np.testing.assert_array_equal(widget.band_structs["bands"], bands.bands)
    assert bands.eigendisplacements is sentinel  # input not mutated
    combined = pmv.bands_and_dos(
        band_structs=bands, doses=PhononDos.from_dict(raw["phonon_dos"])
    )
    assert "eigendisplacements" not in combined.band_structs


def test_phonon_db_doc_via_monty() -> None:
    """MontyDecoder builds PhononDBDoc with matscikit fields equal to pymatgen's."""
    pytest.importorskip("monty")
    pytest.importorskip("pymatgen")
    from monty.json import MontyDecoder
    from pymatgen.core import Structure
    from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine
    from pymatgen.phonon.dos import PhononDos

    from pymatviz.phonons.helpers import PhononDBDoc

    raw = load_raw_doc()
    doc = MontyDecoder().process_decoded(raw)
    assert isinstance(doc, PhononDBDoc)
    assert isinstance(doc.structure, mk.Structure)
    assert isinstance(doc.phonon_bandstructure, mk.PhononBands)
    assert isinstance(doc.phonon_dos, mk.PhononDos)
    pmg_struct = Structure.from_dict(raw["structure"])
    np.testing.assert_array_equal(doc.structure.positions, pmg_struct.cart_coords)
    pmg_bands = PhononBandStructureSymmLine.from_dict(raw["phonon_bandstructure"])
    np.testing.assert_array_equal(
        doc.phonon_bandstructure.frequencies.T, pmg_bands.bands
    )
    np.testing.assert_array_equal(
        doc.phonon_dos.total_dos, PhononDos.from_dict(raw["phonon_dos"]).densities
    )


# === XRD ===


def pymatgen_xrd_inputs() -> dict[str, Any]:
    """Pymatgen Bi2Zr2O7 structure, its XRDCalculator pattern and a mock pattern."""
    pytest.importorskip("pymatgen")
    from pymatgen.analysis.diffraction.xrd import DiffractionPattern, XRDCalculator
    from pymatgen.core import Structure

    struct = Structure.from_file(BI2_ZR2_O7_CIF)
    mock = DiffractionPattern(
        x=[10, 20, 30, 40, 50],
        y=[100, 50, 75, 25, 60],
        hkls=[
            [{"hkl": (1, 0, 0)}],
            [{"hkl": (1, 1, 0)}],
            [{"hkl": (1, 1, 1)}],
            [{"hkl": (2, 0, 0)}],
            [{"hkl": (2, 1, 0)}],
        ],
        d_hkls=[2.5, 2.0, 1.8, 1.5, 1.3],
    )
    return {"struct": struct, "xrd": XRDCalculator().get_pattern(struct), "mock": mock}


@pytest.mark.parametrize(
    ("make_input", "expected_key"),
    [
        (lambda inputs: inputs["mock"], "mock"),
        (lambda inputs: inputs["xrd"], "xrd"),
        (lambda inputs: inputs["struct"], "xrd"),
        (
            lambda inputs: pytest.importorskip("pymatgen.core").IStructure.from_sites(
                inputs["struct"]
            ),
            "xrd",
        ),
        (
            lambda inputs: pytest.importorskip(
                "pymatgen.io.ase"
            ).AseAtomsAdaptor.get_atoms(inputs["struct"]),
            "xrd",
        ),
    ],
    ids=["pattern", "xrd_calculator", "structure", "istructure", "ase"],
)
def test_pymatgen_xrd_inputs(
    make_input: Callable[[dict[str, Any]], Any], expected_key: str
) -> None:
    """Pymatgen DiffractionPatterns pass through, pymatgen/ASE structures match
    pymatgen's XRDCalculator.
    """
    inputs = pymatgen_xrd_inputs()
    expected = inputs[expected_key]
    widget = pmv.xrd_plot(make_input(inputs), style="height: 300px")
    assert isinstance(widget, pmv.widgets.XrdWidget)
    # matscikit's xrd_patterns vs pymatgen's XRDCalculator: observed max rel. 2θ
    # difference 4.5e-16 (2 eps), ASE Atoms round trip perturbs coordinates by
    # ~1.5e-14 relative (~70 eps)
    for key in ("x", "y"):
        np.testing.assert_allclose(
            widget.patterns[key], getattr(expected, key), rtol=1e-12, atol=0
        )
    assert len(widget.patterns["hkls"]) == len(expected.hkls)

    widget = pmv.xrd_plot({"pmg": inputs["mock"], "mk": mk.XrdPattern([10], [1])})
    assert widget.patterns["pmg"]["d_hkls"] == inputs["mock"].d_hkls


def test_pymatgen_xrd_raises() -> None:
    """Empty and invalid-intensity pymatgen DiffractionPatterns raise."""
    pytest.importorskip("pymatgen")
    from pymatgen.analysis.diffraction.xrd import DiffractionPattern

    with pytest.raises(ValueError, match="No intensities found"):
        pmv.xrd_plot(DiffractionPattern([], [], [], []))
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        pmv.xrd_plot(DiffractionPattern([10], [-1], [[{"hkl": (1, 0, 0)}]], [1]))


# === Notebook ===


@pytest.fixture
def notebook_shell() -> Any:
    """IPython shell with notebook_mode turned off again after each test."""
    from IPython.core.interactiveshell import InteractiveShell

    yield InteractiveShell.instance()
    pmv.notebook_mode(on=False)


def make_pmg_structure() -> Any:
    """Simple cubic Fe pymatgen Structure."""
    pymatgen_core = pytest.importorskip("pymatgen.core")
    return pymatgen_core.Structure(pymatgen_core.Lattice.cubic(3), ["Fe"], [[0, 0, 0]])


def make_diffraction_pattern() -> Any:
    """Three-peak pymatgen DiffractionPattern."""
    xrd = pytest.importorskip("pymatgen.analysis.diffraction.xrd")
    return xrd.DiffractionPattern(
        x=[10, 20, 30],
        y=[100, 50, 75],
        hkls=[[{"hkl": (1, 0, 0)}], [{"hkl": (1, 1, 0)}], [{"hkl": (1, 1, 1)}]],
        d_hkls=[2.5, 2.0, 1.8],
    )


def make_phonopy_total_dos(phonopy_nacl: Any) -> Any:
    """Phonopy TotalDos of NaCl on a 10x10x10 mesh."""
    phonopy_nacl.run_mesh([10, 10, 10])
    phonopy_nacl.run_total_dos()
    return phonopy_nacl.total_dos


@pytest.mark.parametrize("obj_name", ["structure", "ase_atoms", "pattern", "dos"])
def test_notebook_mode_formats_third_party_objects(
    notebook_shell: Any, obj_name: str, request: pytest.FixtureRequest
) -> None:
    """With notebook_mode on, pymatgen/ASE/phonopy objects format as widgets."""
    if obj_name == "structure":
        obj = make_pmg_structure()
    elif obj_name == "ase_atoms":
        obj = request.getfixturevalue("ase_atoms")[0]
    elif obj_name == "pattern":
        obj = make_diffraction_pattern()
    else:
        obj = make_phonopy_total_dos(request.getfixturevalue("phonopy_nacl"))
    pmv.notebook_mode(on=True)
    data, _metadata = notebook_shell.display_formatter.format(obj)
    assert WIDGET_MIME in data
    pmv.notebook_mode(on=False)
    data, _metadata = notebook_shell.display_formatter.format(obj)
    assert WIDGET_MIME not in data


def make_pmg_phase_diagram() -> Any:
    """Li-O pymatgen PhaseDiagram."""
    pytest.importorskip("pymatgen")
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition

    return PhaseDiagram(
        [
            PDEntry(Composition("Li"), -1.9),
            PDEntry(Composition("O"), -4.2),
            PDEntry(Composition("Li2O"), -14.3),
        ]
    )


def make_pmg_structure_subclass() -> Any:
    """Instance of a pymatgen Structure subclass."""
    pymatgen_core = pytest.importorskip("pymatgen.core")

    class MyStructure(pymatgen_core.Structure):
        """A pymatgen Structure subclass."""

    return MyStructure(pymatgen_core.Lattice.cubic(3), ["Fe"], [[0, 0, 0]])


@pytest.mark.parametrize(
    ("obj_factory", "expected_widget", "expected_type"),
    [
        (
            lambda: pytest.importorskip("pymatgen.core").Composition("Fe2O3"),
            "CompositionWidget",
            "composition",
        ),
        (
            lambda: pytest.importorskip(
                "pymatgen.analysis.diffraction.xrd"
            ).DiffractionPattern(x=[10], y=[100], hkls=[], d_hkls=[]),
            "XrdWidget",
            "xrd",
        ),
        (make_pmg_phase_diagram, "ConvexHullWidget", "convex_hull"),
        (make_pmg_structure_subclass, "StructureWidget", "structure"),
        (
            lambda: [{"structure": make_pmg_structure(), "energy": -1.0}] * 2,
            "TrajectoryWidget",
            "trajectory",
        ),
    ],
    ids=["composition", "xrd", "phase_diagram", "structure_subclass", "trajectory"],
)
def test_create_widget_pymatgen(
    obj_factory: Callable[[], Any], expected_widget: str, expected_type: str
) -> None:
    """create_widget handles pymatgen objects (and subclasses) like matscikit's."""
    from pymatviz.notebook import create_widget

    widget = create_widget(obj_factory())
    assert type(widget).__name__ == expected_widget
    assert widget.widget_type == expected_type


# === Composition embeddings ===


def test_one_hot_encode_pymatgen_compositions() -> None:
    """Pymatgen Compositions (with oxidation states) encode like formula strings."""
    pytest.importorskip("pymatgen")
    from pymatgen.core import Composition

    from pymatviz.cluster.composition import one_hot_encode

    formulas = ["H2O", "CO2", "NaCl"]
    np.testing.assert_array_equal(
        one_hot_encode([Composition(formula) for formula in formulas]),
        one_hot_encode(formulas),
    )
    result = one_hot_encode(
        ["Fe2O3", Composition({"Fe3+": 2, "O2-": 3})],
        elements=["Fe", "O"],
        normalize=False,
    )
    np.testing.assert_array_equal(result, [[0.4, 0.6], [0.4, 0.6]])


def test_magpie_featurize_matches_matminer() -> None:
    """Magpie features are bit-identical to matminer's ElementProperty magpie preset."""
    pytest.importorskip("matminer")
    from matminer.featurizers.composition import ElementProperty
    from pymatgen.core import Composition

    from pymatviz.cluster.composition import magpie_featurize

    formulas = ["H2O", "CO2", "NaCl", "Fe2O3"]
    featurizer = ElementProperty.from_preset("magpie")
    assert featurizer.feature_labels() == mk.composition_feature_labels()
    expected = [featurizer.featurize(Composition(formula)) for formula in formulas]
    np.testing.assert_array_equal(magpie_featurize(formulas), np.nan_to_num(expected))
    np.testing.assert_array_equal(
        magpie_featurize([Composition(formula) for formula in formulas]),
        np.nan_to_num(expected),
    )


# === Widgets and trajectories ===


_CELL = np.eye(3) * 10
_HKLS = [[{"hkl": [1, 0, 0]}], [{"hkl": [1, 1, 0]}]]


def _meta(payload: dict[str, Any]) -> dict[str, Any]:
    """Envelope JSON fields (everything but the binary arrays and tags)."""
    return {k: v for k, v in payload.items() if k not in ("format", "kind", "arrays")}


# === structures and compositions ===


def test_composition_widget_pymatgen_composition() -> None:
    """A pymatgen Composition syncs as its element amounts."""
    pytest.importorskip("pymatgen.core")
    from pymatgen.core import Composition

    comp = Composition(
        "Li0.1Na0.05K0.05Mg0.1Ca0.05Ba0.05Al0.1Si0.2Ti0.05V0.05Cr0.05"
        "Mn0.1Fe0.15Co0.05Ni0.05Cu0.05Zn0.05O1.0"
    )
    assert CompositionWidget(composition=comp).composition == comp.as_dict()


def test_structure_widget_pymatgen_site_properties(
    fe3co4_disordered: Any,
) -> None:
    """Pymatgen structures (disordered, with magmom and force site properties) sync
    as their as_dict().
    """
    pytest.importorskip("pymatgen.core")
    from matscikit.interop import structure_to_pymatgen_dict
    from pymatgen.core import IStructure

    site_props = {
        "magmom": [[0, 0, 1], [0, 0, -1]],
        "force": [[1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]],
    }
    struct = IStructure.from_dict(structure_to_pymatgen_dict(fe3co4_disordered)).copy(
        site_properties=site_props
    )
    widget = StructureWidget(structure=struct)
    assert widget.structure == struct.as_dict()
    assert {
        key for site in widget.structure["sites"] for key in site["properties"]
    } == {"magmom", "force"}


@pytest.mark.parametrize(
    ("widget_cls", "trait"),
    [
        (RdfPlotWidget, "structures"),
        (BandsAndDosWidget, "structure"),
        (BrillouinZoneWidget, "structure"),
    ],
)
def test_widgets_accept_pymatgen_structure(widget_cls: type, trait: str) -> None:
    """Structure traits sync a pymatgen Structure as a dict, Brillouin zone widgets
    reject a pymatgen Molecule.
    """
    pytest.importorskip("pymatgen.core")
    from pymatgen.core import Lattice, Molecule, Structure

    struct = Structure(Lattice.cubic(3), ["Si"], [[0, 0, 0]])
    synced = getattr(widget_cls(**{trait: struct}), trait)
    assert isinstance(synced, dict)
    assert "lattice" in synced
    if widget_cls is not RdfPlotWidget:
        with pytest.raises(TypeError, match="periodic structure"):
            widget_cls(structure=Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.74]]))


def test_structure_to_dict_ase_atoms() -> None:
    """ASE Atoms convert to structure dicts with lattice and sites."""
    pytest.importorskip("ase")
    from ase.build import bulk

    result = structure_to_dict(bulk("Si", "diamond", a=5.43))
    assert isinstance(result, dict)
    assert "lattice" in result
    assert "sites" in result


def test_link_selection_pymatgen_structures() -> None:
    """Scatter clicks highlight the elements of linked pymatgen structures."""
    pytest.importorskip("pymatgen.core")
    from pymatgen.core import Lattice, Structure

    frames = [
        Structure(Lattice.cubic(3), species, ((0, 0, 0), (0.5,) * 3))
        for species in (("Fe", "O"), ("Li", "O"))
    ]
    scatter = ScatterPlotWidget(
        series=[{"x": [0, 1], "y": [0, 1], "point_ids": [0, 1]}]
    )
    ptable = PeriodicTableWidget()
    pmv.link_selection(scatter, ptable, structures=frames)
    assert ptable.active_elements == ["Fe", "O"]
    scatter.active_point = {"series_idx": 0, "point_idx": 1}
    assert ptable.active_elements == ["Li", "O"]


def test_widget_volumetric_data_from_pymatgen() -> None:
    """Pymatgen VolumetricData expands to one binary envelope per data key; Chgcar
    values are divided by the cell volume like matterviz's CHGCAR parser.
    """
    pytest.importorskip("pymatgen.core")
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.vasp.outputs import Chgcar, VolumetricData

    struct = Structure(Lattice.cubic(2.0), ["Na"], [[0, 0, 0]])
    rng = np.random.default_rng(seed=0)
    data = {"total": rng.random((2, 2, 3)), "diff": rng.random((2, 2, 3))}
    total, diff = StructureWidget(
        volumetric_data=[Chgcar(struct, data)]
    ).volumetric_data
    assert (diff["id"], diff["label"]) == ("diff", "diff")
    assert _meta(total) == {
        "id": "total",
        "lattice": struct.lattice.matrix.tolist(),
        "origin": [0.0, 0.0, 0.0],
        "periodic": True,
        "label": "total",
    }
    np.testing.assert_allclose(
        decode_envelope(total)["values"], data["total"] / 8.0, rtol=6e-8, atol=0
    )
    # generic VolumetricData (ELFCAR, LOCPOT, ...) is passed through unscaled
    volume = VolumetricData(struct, {"total": data["total"]})
    (raw,) = StructureWidget(volumetric_data=[volume]).volumetric_data
    np.testing.assert_array_equal(
        decode_envelope(raw)["values"], data["total"].astype(np.float32)
    )

    # colliding data keys ("total" twice) load side by side via a prefix mapping
    payloads = StructureWidget(
        volumetric_data={"relaxed": Chgcar(struct, data), "initial": volume}
    ).volumetric_data
    assert [(pl["id"], pl["label"]) for pl in payloads] == [
        ("relaxed/total", "relaxed/total"),
        ("relaxed/diff", "relaxed/diff"),
        ("initial/total", "initial/total"),
    ]
    assert payloads[0]["arrays"] == total["arrays"]
    with pytest.raises(ValueError, match=r"duplicates \['total'\].*prefix: volume"):
        StructureWidget(volumetric_data=[Chgcar(struct, data), volume])


# === phase diagrams, XRD, vasprun ===


@pytest.mark.parametrize("source", ["pymatgen", "entries", "generator", "matscikit"])
def test_normalize_convex_hull_entries_pymatgen(source: str) -> None:
    """Phase diagrams (pymatgen, matscikit), entry lists and generators give
    pymatgen's energies above hull and stability (computed by matscikit).
    """
    pytest.importorskip("pymatgen.core")
    from matscikit.interop import to_phase_diagram
    from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    entries = [
        PDEntry(Composition("Li"), -1.9),
        PDEntry(Composition("O"), -4.2),
        PDEntry(Composition("Li2O"), -14.3),
        PDEntry(Composition("Li2O2"), -16.0),
        ComputedEntry("Li3O", -16.0, correction=-0.3),
    ]
    phase_diag = PhaseDiagram(entries)
    if source == "pymatgen":  # in PhaseDiagram.all_entries order
        entries = phase_diag.all_entries
    obj = {
        "pymatgen": phase_diag,
        "entries": entries,
        "generator": (entry for entry in entries),
        "matscikit": to_phase_diagram(entries),
    }[source]
    result = normalize_convex_hull_entries(obj)
    assert result is not None
    assert [entry["composition"] for entry in result] == [
        entry.composition.as_dict() for entry in entries
    ]
    # total energy = energy per atom x atoms, within round-off of pymatgen's
    np.testing.assert_allclose(
        [entry["energy"] for entry in result],
        [entry.energy for entry in entries],
        rtol=4e-16,
        atol=0,
    )
    np.testing.assert_allclose(
        [entry["e_above_hull"] for entry in result],
        [phase_diag.get_e_above_hull(entry) for entry in entries],
        rtol=0,
        atol=1e-12,
    )
    assert [entry["is_stable"] for entry in result] == [
        entry in phase_diag.stable_entries for entry in entries
    ]


def test_normalize_xrd_pattern_diffraction_pattern() -> None:
    """DiffractionPatterns are sticks with hkls (plain floats/ints), profiles
    without.
    """
    pytest.importorskip("pymatgen.core")
    from pymatgen.analysis.diffraction.xrd import DiffractionPattern

    x_vals, y_vals = np.array([10.0, 20.0]), np.array([100.0, 50.0])
    computed = normalize_xrd_pattern(DiffractionPattern(x_vals, y_vals, _HKLS, None))
    measured = normalize_xrd_pattern(DiffractionPattern(x_vals, y_vals, None, None))
    assert computed is not None
    assert "kind" not in computed
    assert measured == {"x": [10.0, 20.0], "y": [100.0, 50.0], "kind": "profile"}

    result = normalize_xrd_pattern(
        DiffractionPattern(
            x=np.array([10.0, 20.0, 30.0]),
            y=np.array([100.0, 50.0, 25.0]),
            hkls=[[{"hkl": (1, 0, 0)}], [{"hkl": (1, 1, 0)}], [{"hkl": (1, 1, 1)}]],
            d_hkls=np.array([2.5, 2.0, 1.5]),
        )
    )
    assert result is not None
    assert result["x"] == [10.0, 20.0, 30.0]
    assert result["d_hkls"] == [2.5, 2.0, 1.5]
    assert type(result["x"][0]) is float
    assert type(result["d_hkls"][0]) is float
    assert result["hkls"][0][0]["hkl"] == [1, 0, 0]
    assert type(result["hkls"][0][0]["hkl"][0]) is int


def test_xrd_widget_labeled_diffraction_patterns() -> None:
    """A {label: pattern} mapping normalizes pymatgen DiffractionPatterns, keeping
    label order.
    """
    pytest.importorskip("pymatgen.core")
    from pymatgen.analysis.diffraction.xrd import DiffractionPattern

    computed = DiffractionPattern(
        [10, 20], [100, 50], [[{"hkl": (1, 0, 0)}]] * 2, [2, 1]
    )
    widget = XrdWidget(
        patterns={"computed": computed, "scan": {"two_theta": [5], "intensities": [1]}}
    )
    assert list(widget.patterns) == ["computed", "scan"]
    # matscikit's XrdPattern defaults missing multiplicities to 1
    assert widget.patterns["computed"]["hkls"][0] == [
        {"hkl": [1, 0, 0], "multiplicity": 1}
    ]
    assert widget.patterns["scan"] == {"x": [5], "y": [1], "kind": "profile"}


@pytest.mark.parametrize("labeled", [False, True])
def test_vasprun_paths_match_pymatgen(labeled: bool) -> None:
    """vasprun.xml paths (single or as dict values) give pymatgen's bands and DOS."""
    pytest.importorskip("pymatgen.core")
    from pymatgen.electronic_structure.core import Spin
    from pymatgen.io.vasp import Vasprun

    from pymatviz.utils.testing import TEST_FILES

    path = f"{TEST_FILES}/vasprun/MgO.vasprun.xml.xz"
    vasprun = Vasprun(path, parse_projected_eigen=False)
    pmg_bands, pmg_dos = vasprun.get_band_structure().as_dict(), vasprun.complete_dos
    inputs = {"MgO": path} if labeled else path
    widget = pmv.bands_and_dos(band_structs=inputs, doses=inputs)
    bands = widget.band_structs["MgO"] if labeled else widget.band_structs
    doses = widget.doses["MgO"] if labeled else widget.doses
    np.testing.assert_array_equal(bands["bands"]["1"], pmg_bands["bands"]["1"])
    np.testing.assert_array_equal(bands["kpoints"], pmg_bands["kpoints"])
    assert bands["efermi"] == pmg_bands["efermi"]
    np.testing.assert_array_equal(doses["energies"], pmg_dos.energies)
    np.testing.assert_array_equal(doses["densities"]["1"], pmg_dos.densities[Spin.up])


# === trajectories ===


@pytest.mark.parametrize("key", ["positions", "force", "lattice"])
@pytest.mark.parametrize("value", [np.nan, np.inf, 1e39])
def test_invalid_ase_frame_arrays(key: str, value: float) -> None:
    """Non-finite data and float32 overflow in ASE frames fail before a frame syncs."""
    pytest.importorskip("ase")
    from ase import Atoms

    atoms = Atoms("H", positions=[[0, 0, 0]], cell=[2, 2, 2], pbc=True)
    if key == "positions":
        atoms.positions[0, 0] = value
    elif key == "force":
        atoms.info["force"] = [[value, 0, 0]]
    else:
        atoms.cell[0, 0] = value
        if np.isfinite(value):  # Lattice uses float64.
            arrays = decode_envelope(TrajectoryWidget(trajectory=atoms).trajectory)
            assert arrays["lattice"][0, 0] == value
            return
    with pytest.raises(
        ValueError, match=f"{key} (has non-finite value|beyond float32 range)"
    ):
        TrajectoryWidget(trajectory=atoms)


def _fallback_cases() -> list[Any]:
    """pymatgen/ASE frame lists (and pymatgen Trajectories) the binary format can't
    hold.
    """
    from ase import Atoms
    from ase.constraints import FixAtoms
    from pymatgen.core import DummySpecies, Lattice, Molecule, Species, Structure
    from pymatgen.core.trajectory import Trajectory

    from pymatviz.structures import to_structure

    lattice = Lattice.cubic(3)
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    fe2 = Structure(lattice, ["Fe", "Fe"], coords)
    dummy = Structure(lattice, [DummySpecies("X"), "Si"], coords)
    zeros = np.zeros((2, 3))
    with_force = Structure(
        lattice, ["Fe", "Fe"], coords, site_properties={"force": zeros}
    )
    h2_pos = [[0, 0, 0], [0, 0, 0.7]]
    h2_bulk = Atoms("H2", h2_pos, cell=np.eye(3) * 5, pbc=True)
    return [
        [fe2, Structure(lattice, ["Fe", "Co"], coords)],  # varying species
        [{"structure": fe2, "label": "relaxed"}],  # string frame property
        [Structure(lattice, [{"Fe": 0.5, "Co": 0.5}], [[0, 0, 0]])],  # disordered
        [fe2.copy(site_properties={"aux": [[0, 1, 2], [3, 4]]})],
        [with_force.copy(site_properties={"force_max": zeros})],
        [Atoms("H2", h2_pos, cell=[5, 5, 5], constraint=FixAtoms(indices=[0]))],
        [Atoms("H2", h2_pos, cell=[5, 5, 5], magmoms=[[0, 0, 1], [0, 0, 2]])],
        [
            {
                "structure": Structure(
                    lattice, ["Fe", "Fe"], coords, properties={"phase": "alpha"}
                ),
                "energy": -1,
            }
        ],
        [dummy, dummy],
        Trajectory.from_structures([dummy, dummy]),
        [Structure(lattice, [], [])] * 2,  # zero atoms
        [Structure(lattice, [Species("Fe", 2), Species("O", -2)], coords)],
        # matscikit keeps oxidation states in info["pymatgen"]
        [to_structure(Structure(lattice, [Species("Fe", 2), "O"], coords))],
        Trajectory.from_structures(
            [Structure(lattice, [Species("Fe", 2), Species("O", -2)], coords)] * 2
        ),
        [Structure(lattice, ["Fe", "Fe"], coords, labels=["Fe1", "Fe2"])],
        [Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.7]], charge=1, spin_multiplicity=2)],
        [Molecule(["H", "H"], [[0, 0, 0], [0, 0, 0.7]], spin_multiplicity=3)],
        [Structure(lattice, ["Fe", "Fe"], coords, charge=1)],
        # singular cell (2D slab without c): the JSON path pads it with vacuum
        [Atoms("H2", h2_pos, cell=[0.009] * 3, pbc=True)],
        [Atoms("H2", h2_pos, cell=[5, 5, 0], pbc=[1, 1, 0])] * 2,
        [with_force.copy(site_properties={"forces": zeros})],  # force and forces
        [{"structure": fe2, "n_evals": 2**53 + 1}],  # float64 would round it
        [h2_bulk, Atoms("H2", h2_pos, cell=np.eye(3) * 5, pbc=[1, 1, 0])],  # pbc
        [Atoms("H2", h2_pos), h2_bulk],  # cell in only some frames
        [with_force, fe2],  # vector keys differ between frames
        # scalar/vector clash
        [{"structure": with_force.copy(site_properties={"vel": zeros}), "vel": 1.0}],
        [{"structure": fe2, "steps": 1.0}],  # reserved name
    ]


_N_FALLBACK_CASES = 27


@pytest.mark.parametrize("case_idx", range(_N_FALLBACK_CASES))
def test_unsupported_pymatgen_ase_frames_fall_back_to_json(case_idx: int) -> None:
    """pymatgen/ASE frames the binary format can't hold (or would silently strip,
    like oxidation states, labels, molecule charge/spin) sync as JSON.
    """
    pytest.importorskip("ase")
    pytest.importorskip("pymatgen.core")
    cases = _fallback_cases()
    assert len(cases) == _N_FALLBACK_CASES
    traj = TrajectoryWidget(trajectory=cases[case_idx]).trajectory
    assert "format" not in traj
    assert "frames" in traj


def test_pymatgen_trajectory_ragged_forces_raise() -> None:
    """Per-frame forces of the wrong shape raise instead of syncing."""
    pytest.importorskip("pymatgen.core")
    from pymatgen.core import Lattice, Structure
    from pymatgen.core.trajectory import Trajectory

    fe2 = Structure(Lattice.cubic(3), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    forces = [{"forces": np.zeros((2, 3))}, {"forces": np.zeros((3, 3))}]
    traj = Trajectory.from_structures([fe2, fe2], frame_properties=forces)
    with pytest.raises(ValueError, match="'forces' must be per-atom"):
        TrajectoryWidget(trajectory=traj)


@pytest.mark.parametrize("source", ["object", "dict", "json"])
def test_disordered_pymatgen_trajectory_frames(source: str) -> None:
    """Mixed sites of pymatgen Trajectories retain occupancies, unwrapped coordinates
    and properties.
    """
    pytest.importorskip("pymatgen.core")
    pytest.importorskip("monty")
    from monty.json import MontyEncoder
    from pymatgen.core import Composition, Species
    from pymatgen.core.trajectory import Trajectory

    frac = np.array([[[0.9, 0, 0], [0, 0, 0]], [[1.1, 0, 0], [0, 0, 0]]])
    traj = Trajectory(
        [Composition({"Fe": 0.5, "Co": 0.5}), Species("Fe", 3)],
        frac,
        lattice=_CELL,
        time_step=2,
        site_properties={"moment": [np.array(1), np.array(2)]},
        frame_properties=[{}, {"energy": -1, "forces": np.full((2, 3), [0, 3, 4])}],
    )
    traj.to_displacements()
    traj.charge = 2  # Explicit charge differs from the sum of oxidation states (3).
    serialized = traj.as_dict()
    value = {
        "object": traj,
        "dict": serialized,
        "json": json.loads(
            json.dumps(
                serialized
                | {
                    "coords": traj.coords,
                    "lattice": _CELL,
                    "site_properties": traj.site_properties,
                },
                cls=MontyEncoder,
            )
        ),
    }[source]
    widget = TrajectoryWidget(trajectory=value)
    assert widget.describe() == {
        "widget_type": "trajectory",
        "n_frames": 2,
        "n_sites": 2,
        "elements": ["Co", "Fe"],
        "formula": "Co0.5Fe1.5",
    }
    payload = widget.trajectory
    assert traj.coords_are_displacement
    assert payload["time_step"] == {"value": 2, "unit": "fs"}
    assert payload["frames"][1]["metadata"] == {
        "energy": -1,
        "force_max": 5,
        "force_norm": 5,
    }
    assert payload["frames"][1]["structure"]["sites"][0]["properties"]["force"] == [
        0,
        3,
        4,
    ]
    for idx, (frame, pos) in enumerate(zip(payload["frames"], frac[:, 0], strict=True)):
        # as pymatgen's frames: Trajectory drops an explicit charge on disordered ones
        assert frame["structure"]["charge"] == traj[idx].charge == 3
        site, iron = frame["structure"]["sites"]
        assert iron["species"][0]["oxidation_state"] == 3
        assert site["species"] == [
            {"element": "Fe", "occu": 0.5},
            {"element": "Co", "occu": 0.5},
        ]
        assert site["properties"]["moment"] == 1
        np.testing.assert_array_equal(site["abc"], pos)
        np.testing.assert_array_equal(site["xyz"], pos @ _CELL)


@pytest.mark.parametrize("displacement", [False, True])
def test_pymatgen_trajectory_per_frame_lattice(displacement: bool) -> None:
    """Displacement coords are summed back to unwrapped positions in each frame's
    cell, per-frame lattices ship as [n_frames, 3, 3], frame_properties become
    scalar and force channels.
    """
    pytest.importorskip("pymatgen.core")
    from pymatgen.core.trajectory import Trajectory

    frac = np.array([[[0.1, 0.2, 0.95]], [[0.15, 0.2, 1.02]], [[0.2, 0.25, 1.1]]])
    lattices = np.array([np.eye(3) * (4 + 0.1 * idx) for idx in range(3)])
    forces = np.array([[[0, 0, 300]], [[0, 4, 0]], [[0, 0, 0]]], dtype=np.float16)
    frame_props = [
        {"energy": -1.0 - idx, "forces": force} for idx, force in enumerate(forces)
    ]
    traj = Trajectory(
        ["Si"],
        frac,
        lattice=lattices,
        constant_lattice=False,
        frame_properties=frame_props,
        time_step=np.float64(0.5),
    )
    if displacement:
        traj.to_displacements()
    payload = TrajectoryWidget(trajectory=traj).trajectory
    assert json.loads(json.dumps(payload["time_step"])) == {"value": 0.5, "unit": "fs"}
    arrays = decode_envelope(payload)
    np.testing.assert_array_equal(arrays["lattice"], lattices)
    assert arrays["energy"].tolist() == [-1.0, -2.0, -3.0]
    np.testing.assert_array_equal(arrays["force"], forces)
    assert arrays["force_max"].tolist() == arrays["force_norm"].tolist() == [300, 4, 0]
    expected = np.einsum("fai,fij->faj", frac, lattices).astype(np.float32)
    np.testing.assert_array_equal(arrays["positions"], expected)
    traj.frame_properties = [None, {"energy": -2.0}, {}]
    np.testing.assert_array_equal(
        decode_envelope(TrajectoryWidget(trajectory=traj).trajectory)["energy"],
        [np.nan, -2.0, np.nan],
    )


@pytest.mark.parametrize("step", [0, -1, np.nan, np.inf, True, "2"])
def test_invalid_pymatgen_time_step(step: Any) -> None:
    """Live and serialized trajectories reject invalid time steps before encoding."""
    pytest.importorskip("pymatgen.core")
    from pymatgen.core.trajectory import Trajectory

    traj = Trajectory(["Si"], np.zeros((1, 1, 3)), lattice=_CELL, time_step=step)
    for value in (traj, traj.as_dict()):
        with pytest.raises(ValueError, match="time_step value must be finite and > 0"):
            TrajectoryWidget(trajectory=value)


@pytest.mark.parametrize("monty_coords", [False, True])
def test_serialized_pymatgen_trajectory_facts(monty_coords: bool) -> None:
    """Serialized pymatgen Trajectories (read without pymatgen) count frames, symbols
    and decorated site occupancies.
    """
    coords = [[[0, 0, 0]] * 4] * 3
    widget = pmv.trajectory(
        trajectory={
            "@class": "Trajectory",
            "coords": (
                {
                    "@module": "numpy",
                    "@class": "array",
                    "dtype": "float64",
                    "data": coords,
                }
                if monty_coords
                else coords
            ),
            "lattice": np.eye(3) * 3,
            "species": [
                "O",
                26,
                "Fe2+",
                {"element": "Fe", "oxidation_state": 3, "occu": 0.5},
            ],
        }
    )
    sites = widget.trajectory["frames"][0]["structure"]["sites"]
    oxygen, iron, ferrous, ferric = (site["species"] for site in sites)
    assert oxygen == [{"element": "O", "occu": 1}]
    assert iron == [{"element": "Fe", "occu": 1}]
    assert ferrous == [{"element": "Fe", "oxidation_state": 2, "occu": 1}]
    assert ferric == [{"element": "Fe", "oxidation_state": 3, "occu": 0.5}]
    assert widget.describe() == {
        "widget_type": "trajectory",
        "n_frames": 3,
        "n_sites": 4,
        "elements": ["Fe", "O"],
        "formula": "Fe2.5O",
    }


def test_serialized_pymatgen_molecule_trajectory() -> None:
    """Molecular Trajectory JSON without the constructor's charge syncs as binary
    (read without pymatgen).
    """
    serialized = {
        "@class": "Trajectory",
        "coords": [[[0, 0, 0]]],
        "species": ["H"],
        "lattice": None,
    }
    arrays = decode_envelope(TrajectoryWidget(trajectory=serialized).trajectory)
    assert arrays["numbers"].tolist() == [1]
    np.testing.assert_array_equal(arrays["positions"], [[[0, 0, 0]]])
    assert "lattice" not in arrays


@pytest.mark.parametrize("n_frames", [1, 3])
def test_pymatgen_structure_sequence_trajectory(n_frames: int) -> None:
    """Lists of pymatgen structures and pymatgen Trajectories sync as binary frames."""
    pytest.importorskip("pymatgen.core")
    from pymatgen.core import Lattice, Structure
    from pymatgen.core.trajectory import Trajectory

    structures = [
        Structure(
            lattice=Lattice.cubic(3.0 + idx * 0.1),
            species=("Fe", "Fe"),
            coords=((0, 0, 0), (0.5, 0.5, 0.5)),
        )
        for idx in range(n_frames)
    ]
    arrays = decode_envelope(TrajectoryWidget(trajectory=structures).trajectory)
    expected_pos = np.array([struct.cart_coords for struct in structures])
    np.testing.assert_allclose(arrays["positions"], expected_pos, rtol=0, atol=1e-6)
    assert arrays["numbers"].tolist() == [26, 26]
    expected_cells = [struct.lattice.matrix for struct in structures]
    # one shared [3, 3] cell when all frames agree, else [n_frames, 3, 3]
    expected_cells = expected_cells[0] if n_frames == 1 else expected_cells
    np.testing.assert_array_equal(arrays["lattice"], expected_cells)

    traj = Trajectory.from_structures(structures, constant_lattice=False)
    traj_arrays = decode_envelope(TrajectoryWidget(trajectory=traj).trajectory)
    np.testing.assert_array_equal(
        traj_arrays["positions"], expected_pos.astype("float32")
    )
    np.testing.assert_array_equal(traj_arrays["lattice"], expected_cells)


@pytest.mark.parametrize(
    ("metadata_field", "metadata_value"),
    [
        ("properties", {"energy": -1.23, "forces": [[0.0, 0.3, 0.4], [0.0, 0.0, 0.0]]}),
        ("info", {"temperature": 300, "pressure": 1.0}),
    ],
)
def test_trajectory_widget_with_pymatgen_structure_metadata(
    metadata_field: str, metadata_value: dict[str, Any]
) -> None:
    """TrajectoryWidget forwards pymatgen structure properties/info to frame metadata,
    per-atom forces to the sites.
    """
    pytest.importorskip("pymatgen.core")
    from pymatgen.core import Lattice, Structure

    structure = Structure(
        lattice=Lattice.cubic(3.0),
        species=("Fe", "Fe"),
        coords=((0, 0, 0), (0.5, 0.5, 0.5)),
    )
    if metadata_field == "properties":
        structure.properties = metadata_value
    else:
        object.__setattr__(structure, "info", metadata_value)

    arrays = decode_envelope(TrajectoryWidget(trajectory=structure).trajectory)
    scalars = {key: arr.tolist() for key, arr in arrays.items() if arr.ndim == 1}
    scalars.pop("numbers")
    if "forces" in metadata_value:
        assert scalars == {
            "energy": [-1.23],
            "force_max": [pytest.approx(0.5, rel=1e-15)],
            "force_norm": [pytest.approx(0.5 / np.sqrt(2), rel=1e-15)],
        }
        # float32 vectors: within 1 float32 ulp of the float64 input
        np.testing.assert_allclose(
            arrays["force"][0], metadata_value["forces"], rtol=1.2e-7, atol=0
        )
    else:
        assert scalars == {key: [val] for key, val in metadata_value.items()}


@pytest.mark.parametrize("shape", [(6,), (3, 3)])
def test_ase_stress_transfer(shape: tuple[int, ...]) -> None:
    """Transfer raw ASE stress losslessly as binary frame_stress."""
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    frames = [Atoms("H", positions=[[idx, 0, 0]]) for idx in range(2)]
    tensors = np.random.default_rng(seed=0).normal(size=(len(frames), *shape))
    for atoms, stress in zip(frames, tensors, strict=True):
        atoms.calc = SinglePointCalculator(atoms, stress=stress)

    widget = TrajectoryWidget()
    widget.trajectory = frames
    wire = decode_envelope(widget.trajectory)["frame_stress"]
    assert wire.dtype == np.dtype("<f8")
    np.testing.assert_array_equal(wire, tensors.reshape(2, -1))


@pytest.mark.parametrize(
    ("symbols", "positions", "cell", "energy"),
    [
        pytest.param(
            "H2O", [[0, 0, 0], [0, 0, 1], [0, 1, 0]], None, -1.5, id="molecule"
        ),
        pytest.param(
            "Fe2", [[0, 0, 0], [0.5, 0.5, 0.5]], [3, 3, 3], -2.0, id="periodic"
        ),
    ],
)
@pytest.mark.parametrize("mson_atoms", [False, True], ids=["Atoms", "MSONAtoms"])
def test_trajectory_widget_with_ase_atoms(
    symbols: str,
    positions: list[list[float]],
    cell: list[int] | None,
    energy: float,
    mson_atoms: bool,
) -> None:
    """ASE frames (incl. pymatgen MSONAtoms, whose as_dict() is not a structure dict)
    keep info and calculator results (which AseAtomsAdaptor drops) with and without a
    simulation cell; calculator forces become site vectors.
    """
    atoms_cls = pytest.importorskip("ase").Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    if mson_atoms:
        pytest.importorskip("pymatgen.core")
        from pymatgen.io.ase import MSONAtoms

        atoms_cls = MSONAtoms

    atoms = atoms_cls(symbols, positions=positions, cell=cell)
    atoms.info = {"temperature": 300}
    forces = np.zeros((len(atoms), 3))
    forces[0] = [0.0, 0.0, 2.0]
    atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
    payload = TrajectoryWidget(trajectory=atoms).trajectory
    arrays = decode_envelope(payload)
    np.testing.assert_allclose(arrays["positions"][0], positions, rtol=0, atol=1e-7)
    assert arrays["numbers"].tolist() == atoms.numbers.tolist()
    # a molecule has no cell (and so no pbc); a periodic frame keeps its cell
    assert ("lattice" in arrays) == (cell is not None)
    assert payload.get("pbc") == (atoms.pbc.tolist() if cell else None)
    assert {key: arrays[key].tolist() for key in ("temperature", "energy")} == {
        "temperature": [300],
        "energy": [energy],
    }
    assert arrays["force_max"].tolist() == [2.0]
    np.testing.assert_allclose(arrays["force_norm"], [2 / np.sqrt(len(atoms))])
    assert arrays["force"][0].tolist() == forces.tolist()
