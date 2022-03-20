from matminer.datasets import load_dataset
from plotly.graph_objs._figure import Figure

from pymatviz import spacegroup_sunburst


df_phonons = load_dataset("matbench_phonons")

df_phonons[["spg_symbol", "spg_num"]] = [
    struct.get_space_group_info() for struct in df_phonons.structure
]


def test_spacegroup_sunburst():
    fig = spacegroup_sunburst(df_phonons.spg_num)
    assert isinstance(fig, Figure)
    assert set(fig.data[0].parents) == {
        "",
        "cubic",
        "trigonal",
        "triclinic",
        "orthorhombic",
        "tetragonal",
        "hexagonal",
        "monoclinic",
    }
    assert fig.data[0].branchvalues == "total"

    spacegroup_sunburst(df_phonons, spg_col="spg_num")
    spacegroup_sunburst(df_phonons.spg_num, show_values="percent")
