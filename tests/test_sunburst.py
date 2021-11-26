import pandas as pd
from plotly.graph_objects import Figure

from ml_matrics import ROOT, spacegroup_sunburst


phonons = pd.read_csv(f"{ROOT}/data/matbench-phonons.csv")


def test_spacegroup_sunburst():
    fig = spacegroup_sunburst(phonons.sg_number)
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

    spacegroup_sunburst(phonons, sgp_col="sg_number")
    spacegroup_sunburst(phonons.sg_number, show_values="percent")
