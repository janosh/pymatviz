<h1 align="center">
<img src="https://github.com/janosh/pymatviz/raw/main/site/static/favicon.svg" alt="Logo" height="60px">
<br class="hide-in-docs">
pymatviz
</h1>

<h4 align="center" class="toc-exclude">

A toolkit for visualizations in materials informatics.

[![Tests](https://github.com/janosh/pymatviz/actions/workflows/test.yml/badge.svg)](https://github.com/janosh/pymatviz/actions/workflows/test.yml)
[![This project supports Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/pymatviz?logo=pypi&logoColor=white)](https://pypi.org/project/pymatviz)
[![codecov](https://codecov.io/gh/janosh/pymatviz/graph/badge.svg?token=7BG2TZVOBH)](https://codecov.io/gh/janosh/pymatviz)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pymatviz?logo=icloud&logoColor=white)](https://pypistats.org/packages/pymatviz)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281/zenodo.10456384-blue?logo=Zenodo&logoColor=white)](https://zenodo.org/records/10456384)

</h4>

[fig-icon]: https://api.iconify.design/lsicon:scatter-diagram-outline.svg?color=%234c8bf5&height=16 "View example code"

<slot name="how-to-cite">

> If you use `pymatviz` in your research, [see how to cite](#how-to-cite-pymatviz). Check out [23 existing papers using `pymatviz`](#papers-using-pymatviz) for inspiration!

</slot>

## Installation

```sh
pip install pymatviz
```

See `pyproject.toml` for available extras like `pip install 'pymatviz[brillouin]'` to render 3d Brillouin zones.

## API Docs

See the [/api][/api] page.

[/api]: https://janosh.github.io/pymatviz/api

## Usage

See the Jupyter notebooks under [`examples/`](examples) for how to use `pymatviz`. PRs with additional examples are welcome! 🙏

|                                                                           |                                                                                                                                                             |                                   |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| [matbench_dielectric_eda.ipynb](examples/matbench_dielectric_eda.ipynb)   | [![Open in Google Colab][Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/matbench_dielectric_eda.ipynb)  | [Launch Codespace][codespace url] |
| [mp_bimodal_e_form.ipynb](examples/mp_bimodal_e_form.ipynb)               | [![Open in Google Colab][Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/mp_bimodal_e_form.ipynb)        | [Launch Codespace][codespace url] |
| [matbench_perovskites_eda.ipynb](examples/matbench_perovskites_eda.ipynb) | [![Open in Google Colab][Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/matbench_perovskites_eda.ipynb) | [Launch Codespace][codespace url] |
| [mprester_ptable.ipynb](examples/mprester_ptable.ipynb)                   | [![Open in Google Colab][Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/mprester_ptable.ipynb)          | [Launch Codespace][codespace url] |

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg
[codespace url]: https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=340898532

## Periodic Table

See [`pymatviz/ptable/plotly.py`](pymatviz/ptable/plotly.py). The module supports heatmaps, heatmap splits (multiple values per element), histograms, scatter plots and line plots. All visualizations are interactive through [Plotly](https://plotly.com) and support displaying additional data on hover.

> [!WARNING]
> Version 0.16.0 of `pymatviz` dropped the matplotlib-based functions in `ptable_matplotlib.py` in https://github.com/janosh/pymatviz/pull/270. Please use the `plotly`-based functions shown below instead which have feature parity, interactivity and better test coverage.

|                                        [`ptable_heatmap_plotly(atomic_masses)`](pymatviz/ptable/plotly.py#L38)                                         | [`ptable_heatmap_plotly(compositions, log=True)`](pymatviz/ptable/plotly.py#L38) [![fig-icon]](assets/scripts/ptable/ptable_heatmap_plotly.py) |
| :----------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                        ![ptable-heatmap-plotly-more-hover-data]                                                        |                                                          ![ptable-heatmap-plotly-log]                                                          |
|               [`ptable_hists_plotly(data)`](pymatviz/ptable/plotly.py#L398) [![fig-icon]](assets/scripts/ptable/ptable_hists_plotly.py)                | [`ptable_scatter_plotly(data, mode="markers")`](pymatviz/ptable/plotly.py#L1359) [![fig-icon]](assets/scripts/ptable/ptable_scatter_plotly.py) |
|                                                                 ![ptable-hists-plotly]                                                                 |                                                        ![ptable-scatter-plotly-markers]                                                        |
| [`ptable_heatmap_splits_plotly(2_vals_per_elem)`](pymatviz/ptable/plotly.py#L778) [![fig-icon]](assets/scripts/ptable/ptable_heatmap_splits_plotly.py) |                               [`ptable_heatmap_splits_plotly(3_vals_per_elem)`](pymatviz/ptable/plotly.py#L778)                                |
|                                                           ![ptable-heatmap-splits-plotly-2]                                                            |                                                       ![ptable-heatmap-splits-plotly-3]                                                        |

[ptable-heatmap-plotly-log]: assets/svg/ptable-heatmap-plotly-log.svg
[ptable-heatmap-plotly-more-hover-data]: assets/svg/ptable-heatmap-plotly-more-hover-data.svg
[ptable-heatmap-splits-plotly-2]: assets/svg/ptable-heatmap-splits-plotly-2.svg
[ptable-heatmap-splits-plotly-3]: assets/svg/ptable-heatmap-splits-plotly-3.svg
[ptable-hists-plotly]: assets/svg/ptable-hists-plotly.svg
[ptable-scatter-plotly-markers]: assets/svg/ptable-scatter-plotly-markers.svg

### Dash app using `ptable_heatmap_plotly()`

See [`examples/mprester_ptable.ipynb`](examples/mprester_ptable.ipynb).

[https://user-images.githubusercontent.com/30958850/181644052-b330f0a2-70fc-451c-8230-20d45d3af72f.mp4](https://user-images.githubusercontent.com/30958850/181644052-b330f0a2-70fc-451c-8230-20d45d3af72f.mp4)

## Phonons

| [`phonon_bands(bands_dict)`](pymatviz/phonons/plotly.py#L35) [![fig-icon]](assets/scripts/phonons/phonon_bands.py) |                  [`phonon_dos(doses_dict)`](pymatviz/phonons/plotly.py#L381) [![fig-icon]](assets/scripts/phonons/phonon_dos.py)                  |
| :----------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                  ![phonon-bands]                                                   |                                                                   ![phonon-dos]                                                                   |
|                 [`phonon_bands_and_dos(bands_dict, doses_dict)`](pymatviz/phonons/plotly.py#L528)                  | [`phonon_bands_and_dos(single_bands, single_dos)`](pymatviz/phonons/plotly.py#L528) [![fig-icon]](assets/scripts/phonons/phonon_bands_and_dos.py) |
|                                          ![phonon-bands-and-dos-mp-2758]                                           |                                                         ![phonon-bands-and-dos-mp-23907]                                                          |

[phonon-bands]: assets/svg/phonon-bands-mp-2758.svg
[phonon-dos]: assets/svg/phonon-dos-mp-2758.svg
[phonon-bands-and-dos-mp-2758]: assets/svg/phonon-bands-and-dos-mp-2758.svg
[phonon-bands-and-dos-mp-23907]: assets/svg/phonon-bands-and-dos-mp-23907.svg

### Composition Clustering

| [`cluster_compositions(compositions, properties, embedding_method, projection_method, n_components=2)`](pymatviz/cluster/composition/plot.py#L32) [![fig-icon]](assets/scripts/cluster/composition/cluster_compositions_matbench.py) | [`cluster_compositions(compositions, properties, embedding_method, projection_method, n_components=3)`](pymatviz/cluster/composition/plot.py#L32) |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                                                ![matbench-perovskites-magpie-pca-2d]                                                                                                 |                                                      ![matbench-perovskites-magpie-tsne-3d]                                                       |

[matbench-perovskites-magpie-pca-2d]: assets/svg/matbench-perovskites-magpie-pca-2d.svg
[matbench-perovskites-magpie-tsne-3d]: assets/svg/matbench-perovskites-magpie-tsne-3d.svg

Visualize 2D or 3D relationships between compositions and properties using multiple embedding and dimensionality reduction techniques:

Embedding methods: **One-hot** encoding of element fractions, **Magpie** features (elemental properties), **Matscholar** element embeddings, **MEGNet** element embeddings

Dimensionality reduction methods: **PCA** (linear), **t-SNE** (non-linear), **UMAP** (non-linear), **Isomap** (non-linear), **Kernel PCA** (non-linear)

Example usage:

```py
import pymatviz as pmv
from pymatgen.core import Composition

compositions = ("Fe2O3", "Al2O3", "SiO2", "TiO2")

# Create embeddings
embeddings = pmv.cluster.composition.one_hot_encode(compositions)
comp_emb_map = dict(zip(compositions, embeddings, strict=True))

# Plot with optional property coloring
fig = pmv.cluster_compositions(
    compositions=comp_emb_map,
    properties=[1.0, 2.0, 3.0, 4.0],  # Optional property values
    prop_name="Property",  # Optional property label
    embedding_method="one-hot",  # or "magpie", "matscholar_el", "megnet_el", etc.
    projection_method="pca",  # or "tsne", "umap", "isomap", "kernel_pca", etc.
    show_chem_sys="shape",  # works best for small number of compositions; "color" | "shape" | "color+shape" | None
    n_components=2,  # or 3 for 3D plots
)
fig.show()
```

## Structure Clustering

On the roadmap but no ETA yet.

## Structure

See [`pymatviz/structure/plotly.py`](pymatviz/structure/plotly.py).

|                           [`structure_3d(hea_structure)`](pymatviz/structure/plotly.py#L318)                            | [`structure_3d(lco_supercell)`](pymatviz/structure/plotly.py#L318) [![fig-icon]](assets/scripts/structure/structure_3d.py) |
| :---------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------: |
|                                                   ![hea-structure-3d]                                                   |                                                    ![lco-structure-3d]                                                     |
| [`structure_2d(six_structs)`](pymatviz/structure/plotly.py#L42) [![fig-icon]](assets/scripts/structure/structure_2d.py) |  [`structure_3d(six_structs)`](pymatviz/structure/plotly.py#L318) [![fig-icon]](assets/scripts/structure/structure_3d.py)  |
|                                            ![matbench-phonons-structures-2d]                                            |                                             ![matbench-phonons-structures-3d]                                              |

[matbench-phonons-structures-2d]: assets/svg/matbench-phonons-structures-2d.svg
[matbench-phonons-structures-3d]: assets/svg/matbench-phonons-structures-3d.svg
[hea-structure-3d]: assets/svg/hea-structure-3d.svg
[lco-structure-3d]: assets/svg/lco-structure-3d.svg

## Interactive Widgets

See [`pymatviz/widgets`](pymatviz/widgets). Interactive 3D structure, molecular dynamics trajectory and composition visualization widgets for [Jupyter](https://jupyter.org), [Marimo](https://marimo.io), and VSCode notebooks, powered by [anywidget](https://anywidget.dev) and [MatterViz](https://matterviz.janosh.dev) (<https://github.com/janosh/matterviz>). Supports pymatgen `Structure`, ASE `Atoms`, and `PhonopyAtoms`, as well as ASE, `pymatgen` and plain Python trajectory formats.

```py
from pymatviz import StructureWidget, CompositionWidget, TrajectoryWidget
from pymatgen.core import Structure, Composition

# Interactive 3D structure visualization
structure = Structure.from_file("structure.cif")
struct_widget = StructureWidget(structure=structure)

# Interactive composition visualization
composition = Composition("Fe2O3")
comp_widget = CompositionWidget(composition=composition)

# Interactive trajectory visualization
trajectory1 = [struct1, struct2, struct3]  # List of structures
traj_widget1 = TrajectoryWidget(trajectory=trajectory1)

trajectory2 = [{"structure": struct1, "energy": 1.0}, {"structure": struct2, "energy": 2.0}, {"structure": struct3, "energy": 3.0}]  # dicts with "structure" and property values
traj_widget2 = TrajectoryWidget(trajectory=trajectory2)
```

**Examples:**

- [Jupyter notebook demo](examples/widgets/jupyter_demo.ipynb)
- [Marimo demo](examples/widgets/marimo_demo.py)
- [VSCode interactive demo](examples/widgets/vscode_interactive_demo.py)

> [!TIP]
> Checkout the **✅ MatterViz VSCode extension** for using the same viewers directly in VSCode/Cursor editor tabs for rendering local and remote files: [marketplace.visualstudio.com/items?itemName=janosh.matterviz](https://marketplace.visualstudio.com/items?itemName=janosh.matterviz)

Importing `pymatviz` auto-registers all widgets for their respective sets of supported objects via `register_matterviz_widgets()`. To customize the registration, use [`set_renderer()`](pymatviz/notebook.py).

## Brillouin Zone

See [`pymatviz/brillouin.py`](pymatviz/brillouin.py).

|   [`brillouin_zone_3d(cubic_struct)`](pymatviz/brillouin.py#L15) [![fig-icon]](assets/scripts/brillouin/brillouin_zone_3d.py)    |  [`brillouin_zone_3d(hexagonal_struct)`](pymatviz/brillouin.py#L15)   |
| :------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------: |
|                                                   ![brillouin-cubic-mp-10018]                                                    |                   ![brillouin-hexagonal-mp-862690]                    |
| [`brillouin_zone_3d(monoclinic_struct)`](pymatviz/brillouin.py#L15) [![fig-icon]](assets/scripts/brillouin/brillouin_zone_3d.py) | [`brillouin_zone_3d(orthorhombic_struct)`](pymatviz/brillouin.py#L15) |
|                                                ![brillouin-monoclinic-mp-1183089]                                                |                      ![brillouin-volumes-3-cols]                      |

[brillouin-cubic-mp-10018]: assets/svg/brillouin-cubic-mp-10018.svg
[brillouin-hexagonal-mp-862690]: assets/svg/brillouin-hexagonal-mp-862690.svg
[brillouin-monoclinic-mp-1183089]: assets/svg/brillouin-monoclinic-mp-1183089.svg
[brillouin-orthorhombic-mp-1183085]: assets/svg/brillouin-orthorhombic-mp-1183085.svg
[brillouin-volumes-3-cols]: assets/svg/brillouin-volumes-3-cols.svg

## X-Ray Diffraction

See [`pymatviz/xrd.py`](pymatviz/xrd.py).

|             [`xrd_pattern(pattern)`](pymatviz/xrd.py#L42) [![fig-icon]](assets/scripts/xrd/xrd_pattern.py)             |  [`xrd_pattern({key1: patt1, key2: patt2})`](pymatviz/xrd.py#L42)   |
| :--------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------: |
|                                                     ![xrd-pattern]                                                     |                       ![xrd-pattern-multiple]                       |
| [`xrd_pattern(struct_dict, stack="horizontal")`](pymatviz/xrd.py#L42) [![fig-icon]](assets/scripts/xrd/xrd_pattern.py) | [`xrd_pattern(struct_dict, stack="vertical")`](pymatviz/xrd.py#L42) |
|                                            ![xrd-pattern-horizontal-stack]                                             |                    ![xrd-pattern-vertical-stack]                    |

[xrd-pattern]: assets/svg/xrd-pattern.svg
[xrd-pattern-multiple]: assets/svg/xrd-pattern-multiple.svg
[xrd-pattern-horizontal-stack]: assets/svg/xrd-pattern-horizontal-stack.svg
[xrd-pattern-vertical-stack]: assets/svg/xrd-pattern-vertical-stack.svg

## Radial Distribution Functions

See [`pymatviz/rdf/plotly.py`](pymatviz/rdf/plotly.py).

| [`element_pair_rdfs(pmg_struct)`](pymatviz/rdf/plotly.py#L33) | [`element_pair_rdfs({"A": struct1, "B": struct2})`](pymatviz/rdf/plotly.py#L33) [![fig-icon]](assets/scripts/rdf/element_pair_rdfs.py) |
| :-----------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------: |
|                ![element-pair-rdfs-Na8Nb8O24]                 |                                               ![element-pair-rdfs-crystal-vs-amorphous]                                                |

[element-pair-rdfs-Na8Nb8O24]: assets/svg/element-pair-rdfs-Na8Nb8O24.svg
[element-pair-rdfs-crystal-vs-amorphous]: assets/svg/element-pair-rdfs-crystal-vs-amorphous.svg

## Coordination

See [`pymatviz/coordination/plotly.py`](pymatviz/coordination/plotly.py).

|              [`coordination_hist(struct_dict)`](pymatviz/coordination/plotly.py#L34)              |            [`coordination_hist(struct_dict, by_element=True)`](pymatviz/coordination/plotly.py#L34) [![fig-icon]](assets/scripts/coordination/coordination_hist.py)             |
| :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                    ![coordination-hist-single]                                    |                                                                  ![coordination-hist-by-structure-and-element]                                                                  |
| [`coordination_vs_cutoff_line(struct_dict, strategy=None)`](pymatviz/coordination/plotly.py#L337) | [`coordination_vs_cutoff_line(struct_dict, strategy=None)`](pymatviz/coordination/plotly.py#L337) [![fig-icon]](assets/scripts/coordination/coordination_vs_cutoff_line.py#L52) |
|                                 ![coordination-vs-cutoff-single]                                  |                                                                       ![coordination-vs-cutoff-multiple]                                                                        |

[coordination-hist-single]: assets/svg/coordination-hist-single.svg
[coordination-hist-by-structure-and-element]: assets/svg/coordination-hist-by-structure-and-element.svg
[coordination-vs-cutoff-single]: assets/svg/coordination-vs-cutoff-single.svg
[coordination-vs-cutoff-multiple]: assets/svg/coordination-vs-cutoff-multiple.svg

## Sunburst

See [`pymatviz/sunburst.py`](pymatviz/sunburst.py).

| [`spacegroup_sunburst([65, 134, 225, ...])`](pymatviz/sunburst.py#L111) [![fig-icon]](assets/scripts/sunburst/spacegroup_sunburst.py) | [`chem_sys_sunburst(["FeO", "Fe2O3", "LiPO4", ...])`](pymatviz/sunburst.py#L206) [![fig-icon]](assets/scripts/sunburst/chem_sys_sunburst.py) |
| :-----------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                          ![spg-num-sunburst]                                                          |                                                        ![chem-sys-sunburst-ward-bmg]                                                         |
|                                    [`chem_env_sunburst(single_struct)`](pymatviz/sunburst.py#L351)                                    |                                      [`chem_env_sunburst(multiple_structs)`](pymatviz/sunburst.py#L351)                                      |
|                                                      ![chem-env-sunburst-basic]                                                       |                                                        ![chem-env-sunburst-mp-carbon]                                                        |

[spg-num-sunburst]: assets/svg/spg-num-sunburst.svg
[chem-sys-sunburst-ward-bmg]: assets/svg/chem-sys-sunburst-ward-bmg.svg
[chem-env-sunburst-basic]: assets/svg/chem-env-sunburst-basic.svg
[chem-env-sunburst-mp-carbon]: assets/svg/chem-env-sunburst-mp-carbon.svg

## Treemap

See [`pymatviz/treemap/chem_sys.py`](pymatviz/treemap/chem_sys.py).

| [`chem_sys_treemap(["FeO", "Fe2O3", "LiPO4", ...])`](pymatviz/treemap/chem_sys.py#L36) [![fig-icon]](assets/scripts/treemap/chem_sys_treemap.py) | [`chem_sys_treemap(["FeO", "Fe2O3", "LiPO4", ...], group_by="formula")`](pymatviz/treemap/chem_sys.py#L36) |
| :----------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------: |
|                                                           ![chem-sys-treemap-formula]                                                            |                                        ![chem-sys-treemap-ward-bmg]                                        |
|           [`chem_env_treemap(structures)`](pymatviz/treemap/chem_env.py#L36) [![fig-icon]](assets/scripts/treemap/chem_env_treemap.py)           |     [`chem_env_treemap(structures, max_cells_cn=3, max_cells_ce=4)`](pymatviz/treemap/chem_env.py#L36)     |
|                                                            ![chem-env-treemap-basic]                                                             |                                     ![chem-env-treemap-large-dataset]                                      |
|              [`py_pkg_treemap("pymatviz")`](pymatviz/treemap/py_pkg.py#L36) [![fig-icon]](assets/scripts/treemap/py_pkg_treemap.py)              |         [`py_pkg_treemap(["pymatviz", "torch_sim", "pymatgen"])`](pymatviz/treemap/py_pkg.py#L36)          |
|                                                            ![py-pkg-treemap-pymatviz]                                                            |                                         ![py-pkg-treemap-multiple]                                         |
|   [`py_pkg_treemap("pymatviz", color_by="coverage")`](pymatviz/treemap/py_pkg.py#L36) [![fig-icon]](assets/scripts/treemap/py_pkg_treemap.py)    | [`py_pkg_treemap("pymatgen", color_by="coverage", color_range=(0, 100))`](pymatviz/treemap/py_pkg.py#L36)  |
|                                                       ![py-pkg-treemap-pymatviz-coverage]                                                        |                                    ![py-pkg-treemap-pymatgen-coverage]                                     |

> **Note:** For `color_by="coverage"` the package must have coverage data (e.g. run `pytest --cov=<pkg> --cov-report=xml` and pass the resulting `.coverage` file to `coverage_data_file`).

[chem-sys-treemap-formula]: assets/svg/chem-sys-treemap-formula.svg
[chem-sys-treemap-ward-bmg]: assets/svg/chem-sys-treemap-ward-bmg.svg
[py-pkg-treemap-pymatviz]: assets/svg/py-pkg-treemap-pymatviz.svg
[py-pkg-treemap-multiple]: assets/svg/py-pkg-treemap-multiple.svg
[py-pkg-treemap-pymatgen-coverage]: assets/svg/py-pkg-treemap-pymatgen-coverage.svg
[py-pkg-treemap-pymatviz-coverage]: assets/svg/py-pkg-treemap-pymatviz-coverage.svg
[chem-env-treemap-large-dataset]: assets/svg/chem-env-treemap-large-dataset.svg
[chem-env-treemap-basic]: assets/svg/chem-env-treemap-basic.svg

## Rainclouds

See [`pymatviz/rainclouds.py`](pymatviz/rainclouds.py).

| [`rainclouds(two_key_dict)`](pymatviz/rainclouds.py#L20) [![fig-icon]](assets/scripts/rainclouds/rainclouds.py) | [`rainclouds(three_key_dict)`](pymatviz/rainclouds.py#L20) |
| :-------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------: |
|                                              ![rainclouds-bimodal]                                              |                   ![rainclouds-trimodal]                   |

[rainclouds-bimodal]: assets/svg/rainclouds-bimodal.svg
[rainclouds-trimodal]: assets/svg/rainclouds-trimodal.svg

## Sankey

See [`pymatviz/sankey.py`](pymatviz/sankey.py).

| [`sankey_from_2_df_cols(df_perovskites)`](pymatviz/sankey.py#L16) [![fig-icon]](assets/scripts/sankey/sankey_from_2_df_cols.py) | [`sankey_from_2_df_cols(df_space_groups)`](pymatviz/sankey.py#L16) |
| :-----------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------: |
|                                              ![sankey-spglib-vs-aflow-spacegroups]                                              |                ![sankey-crystal-sys-to-spg-symbol]                 |

[sankey-spglib-vs-aflow-spacegroups]: assets/svg/sankey-spglib-vs-aflow-spacegroups.svg
[sankey-crystal-sys-to-spg-symbol]: assets/svg/sankey-crystal-sys-to-spg-symbol.svg

## Bar Plots

See [`pymatviz/bar.py`](pymatviz/bar.py).

| [`spacegroup_bar([65, 134, 225, ...])`](pymatviz/bar.py#L29) [![fig-icon]](assets/scripts/bar/spacegroup_bar.py) | [`spacegroup_bar(["C2/m", "P-43m", "Fm-3m", ...])`](pymatviz/bar.py#L29) |
| :--------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------: |
|                                              ![spg-num-hist-plotly]                                              |                        ![spg-symbol-hist-plotly]                         |

[spg-symbol-hist-plotly]: assets/svg/spg-symbol-hist-plotly.svg
[spg-num-hist-plotly]: assets/svg/spg-num-hist-plotly.svg

## Histograms

See [`pymatviz/histogram.py`](pymatviz/histogram.py).

| [`elements_hist(compositions, log=True, bar_values='count')`](pymatviz/histogram.py#L37) [![fig-icon]](assets/scripts/histogram/elements_hist.py) | [`histogram({'key1': values1, 'key2': values2})`](pymatviz/histogram.py#L108) [![fig-icon]](assets/scripts/histogram/histogram.py) |
| :-----------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: |
|                                                                 ![elements-hist]                                                                  |                                                         ![histogram-ecdf]                                                          |

[histogram-ecdf]: assets/svg/histogram-ecdf.svg

## Scatter Plots

See [`pymatviz/scatter.py`](pymatviz/scatter.py).

|                  [`density_scatter_plotly(df, x=x_col, y=y_col, ...)`](pymatviz/scatter.py#L149)                   | [`density_scatter_plotly(df, x=x_col, y=y_col, ...)`](pymatviz/scatter.py#L149) [![fig-icon]](assets/scripts/scatter/density_scatter_plotly.py) |
| :----------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
|                                             ![density-scatter-plotly]                                              |                                                         ![density-scatter-plotly-blobs]                                                         |
| [`density_scatter(xs, ys, ...)`](pymatviz/scatter.py#L71) [![fig-icon]](assets/scripts/scatter/density_scatter.py) |          [`density_scatter_with_hist(xs, ys, ...)`](pymatviz/scatter.py#L570) [![fig-icon]](assets/scripts/scatter/density_scatter.py)          |
|                                                 ![density-scatter]                                                 |                                                          ![density-scatter-with-hist]                                                           |
| [`density_hexbin(xs, ys, ...)`](pymatviz/scatter.py#L493) [![fig-icon]](assets/scripts/scatter/density_hexbin.py)  |           [`density_hexbin_with_hist(xs, ys, ...)`](pymatviz/scatter.py#L587) [![fig-icon]](assets/scripts/scatter/density_hexbin.py)           |
|                                                 ![density-hexbin]                                                  |                                                           ![density-hexbin-with-hist]                                                           |

[density-scatter-plotly]: assets/svg/density-scatter-plotly.svg
[density-scatter-plotly-blobs]: assets/svg/density-scatter-plotly-blobs.svg
[density-hexbin-with-hist]: assets/svg/density-hexbin-with-hist.svg
[density-hexbin]: assets/svg/density-hexbin.svg
[density-scatter-with-hist]: assets/svg/density-scatter-with-hist.svg
[density-scatter]: assets/svg/density-scatter.svg

## Uncertainty

See [`pymatviz/uncertainty.py`](pymatviz/uncertainty.py).

|             [`qq_gaussian(y_true, y_pred, y_std)`](pymatviz/uncertainty.py#L27) [![fig-icon]](assets/scripts/uncertainty/qq_gaussian.py)              |       [`qq_gaussian(y_true, y_pred, y_std: dict)`](pymatviz/uncertainty.py#L27)        |
| :---------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: |
|                                                                    ![qq-gaussian]                                                                     |                                ![qq-gaussian-multiple]                                 |
| [`error_decay_with_uncert(y_true, y_pred, y_std)`](pymatviz/uncertainty.py#L208) [![fig-icon]](assets/scripts/uncertainty/error_decay_with_uncert.py) | [`error_decay_with_uncert(y_true, y_pred, y_std: dict)`](pymatviz/uncertainty.py#L208) |
|                                                              ![error-decay-with-uncert]                                                               |                          ![error-decay-with-uncert-multiple]                           |

## Classification

See [`pymatviz/classify/confusion_matrix.py`](pymatviz/classify/confusion_matrix.py).

| [`confusion_matrix(conf_mat, ...)`](pymatviz/classify/confusion_matrix.py#L10) | [`confusion_matrix(y_true, y_pred, ...)`](pymatviz/classify/confusion_matrix.py#L10) [![fig-icon]](assets/scripts/classify/confusion_matrix.py) |
| :----------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
|                         ![stability-confusion-matrix]                          |                                                       ![crystal-system-confusion-matrix]                                                        |

See [`pymatviz/classify/curves.py`](pymatviz/classify/curves.py).

| [`roc_curve_plotly(targets, probs_positive)`](pymatviz/classify/curves.py#L76) [![fig-icon]](assets/scripts/classify/roc_curve.py) | [`precision_recall_curve_plotly(targets, probs_positive)`](pymatviz/classify/curves.py#L176) [![fig-icon]](assets/scripts/classify/precision_recall_curve.py) |
| :--------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                    ![roc-curve-plotly-multiple]                                                    |                                                           ![precision-recall-curve-plotly-multiple]                                                           |

[roc-curve-plotly-multiple]: assets/svg/roc-curve-plotly-multiple.svg
[precision-recall-curve-plotly-multiple]: assets/svg/precision-recall-curve-plotly-multiple.svg
[stability-confusion-matrix]: assets/svg/stability-confusion-matrix.svg
[crystal-system-confusion-matrix]: assets/svg/crystal-system-confusion-matrix.svg
[error-decay-with-uncert-multiple]: assets/svg/error-decay-with-uncert-multiple.svg
[error-decay-with-uncert]: assets/svg/error-decay-with-uncert.svg
[elements-hist]: assets/svg/elements-hist.svg
[qq-gaussian-multiple]: assets/svg/qq-gaussian-multiple.svg
[qq-gaussian]: assets/svg/qq-gaussian.svg

## How to cite `pymatviz`

See [`citation.cff`](citation.cff) or cite the [Zenodo record](https://zenodo.org/badge/latestdoi/340898532) using the following BibTeX entry:

```bib
@software{riebesell_pymatviz_2022,
  title = {Pymatviz: visualization toolkit for materials informatics},
  author = {Riebesell, Janosh and Yang, Haoyu and Goodall, Rhys and Baird, Sterling G.},
  date = {2022-10-01},
  year = {2022},
  doi = {10.5281/zenodo.7486816},
  url = {https://github.com/janosh/pymatviz},
  note = {10.5281/zenodo.7486816 - https://github.com/janosh/pymatviz},
  urldate = {2023-01-01}, % optional, replace with your date of access
  version = {0.8.2}, % replace with the version you use
}
```

## Papers using `pymatviz`

Sorted by number of citations, then year. Last updated 2025-05-07. Auto-generated [from Google Scholar](https://scholar.google.com/scholar?q=pymatviz). Manual additions [via PR](https://github.com/janosh/pymatviz/edit/main/readme.md) welcome.

1. C Zeni, R Pinsler, D Zügner et al. (2023). [Mattergen: a generative model for inorganic materials design](https://arxiv.org/abs/2312.03687) (cited by 134)
1. J Riebesell, REA Goodall, P Benner et al. (2023). [Matbench Discovery--A framework to evaluate machine learning crystal stability predictions](https://arxiv.org/abs/2308.14920) (cited by 53)
1. L Barroso-Luque, M Shuaibi, X Fu et al. (2024). [Open materials 2024 (omat24) inorganic materials dataset and models](https://www.rivista.ai/wp-content/uploads/2024/10/2410.12771v1.pdf) (cited by 48)
1. C Chen, DT Nguyen, SJ Lee et al. (2024). [Accelerating computational materials discovery with machine learning and cloud high-performance computing: from large-scale screening to experimental validation](https://pubs.acs.org/doi/abs/10.1021/jacs.4c03849) (cited by 43)
1. M Giantomassi, G Materzanini (2024). [Systematic assessment of various universal machine‐learning interatomic potentials](https://onlinelibrary.wiley.com/doi/abs/10.1002/mgea.58) (cited by 22)
1. AA Naik, C Ertural, P Benner et al. (2023). [A quantum-chemical bonding database for solid-state materials](https://www.nature.com/articles/s41597-023-02477-5) (cited by 15)
1. K Li, AN Rubungo, X Lei et al. (2025). [Probing out-of-distribution generalization in machine learning for materials](https://www.nature.com/articles/s43246-024-00731-w) (cited by 9)
1. A Kapeliukha, RA Mayo (2025). [MOSAEC-DB: a comprehensive database of experimental metal–organic frameworks with verified chemical accuracy suitable for molecular simulations](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d4sc07438f) (cited by 3)
1. N Tuchinda, CA Schuh (2025). [Grain Boundary Segregation and Embrittlement of Aluminum Binary Alloys from First Principles](https://arxiv.org/abs/2502.01579) (cited by 2)
1. A Onwuli, KT Butler, A Walsh (2024). [Ionic species representations for materials informatics](https://pubs.aip.org/aip/aml/article/2/3/036112/3313198) (cited by 2)
1. A Peng, MY Guo (2025). [The OpenLAM Challenges](https://arxiv.org/abs/2501.16358) (cited by 1)
1. F Therrien, JA Haibeh (2025). [OBELiX: A curated dataset of crystal structures and experimentally measured ionic conductivities for lithium solid-state electrolytes](https://arxiv.org/abs/2502.14234) (cited by 1)
1. Aaron D. Kaplan, Runze Liu, Ji Qi et al. (2025). [A Foundational Potential Energy Surface Dataset for Materials](http://arxiv.org/abs/2503.04070)
1. Fei Shuang, Zixiong Wei, Kai Liu et al. (2025). [Universal machine learning interatomic potentials poised to supplant DFT in modeling general defects in metals and random alloys](http://arxiv.org/abs/2502.03578)
1. Yingheng Tang, Wenbin Xu, Jie Cao et al. (2025). [MatterChat: A Multi-Modal LLM for Material Science](http://arxiv.org/abs/2502.13107)
1. Liming Wu, Wenbing Huang, Rui Jiao et al. (2025). [Siamese Foundation Models for Crystal Structure Prediction](http://arxiv.org/abs/2503.10471)
1. K Yan, M Bohde, A Kryvenko (2025). [A Materials Foundation Model via Hybrid Invariant-Equivariant Architectures](https://arxiv.org/abs/2503.05771)
1. N Tuchinda, CA Schuh (2025). [A Grain Boundary Embrittlement Genome for Substitutional Cubic Alloys](https://arxiv.org/abs/2502.06531)
1. Daniel W. Davies, Keith T. Butler, Adam J. Jackson et al. (2024). [SMACT: Semiconducting Materials by Analogy and Chemical Theory](https://github.com/WMD-group/SMACT)
1. Hui Zheng, Eric Sivonxay, Rasmus Christensen et al. (2024). [The ab initio non-crystalline structure database: empowering machine learning to decode diffusivity](https://www.nature.com/articles/s41524-024-01469-2)
1. HH Li, Q Chen, G Ceder (2024). [Voltage Mining for (De) lithiation-Stabilized Cathodes and a Machine Learning Model for Li-Ion Cathode Voltage](https://pubs.acs.org/doi/abs/10.1021/acsami.4c15742)
1. Janosh Riebesell, Ilyes Batatia, Philipp Benner et al. (2023). [A foundation model for atomistic materials chemistry](https://arxiv.org/abs/2401.00096v1)
1. Jack Douglas Sundberg (2022). [A New Framework for Material Informatics and Its Application Toward Electride-Halide Material Systems](https://cdr.lib.unc.edu/concern/dissertations/r494vw405)
