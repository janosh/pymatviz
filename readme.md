<h1 align="center">
<img src="https://github.com/janosh/pymatviz/raw/main/site/static/favicon.svg" alt="Logo" height="60px">
<br class="hide-in-docs">
pymatviz
</h1>

<h4 align="center" class="toc-exclude">

A toolkit for visualizations in materials informatics.

[![Tests](https://github.com/janosh/pymatviz/actions/workflows/test.yml/badge.svg)](https://github.com/janosh/pymatviz/actions/workflows/test.yml)
[![This project supports Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/pymatviz?logo=pypi&logoColor=white)](https://pypi.org/project/pymatviz)
[![codecov](https://codecov.io/gh/janosh/pymatviz/graph/badge.svg?token=7BG2TZVOBH)](https://codecov.io/gh/janosh/pymatviz)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pymatviz?logo=icloud&logoColor=white)](https://pypistats.org/packages/pymatviz)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281/zenodo.10456384-blue?logo=Zenodo&logoColor=white)](https://zenodo.org/records/10456384)

</h4>

<slot name="how-to-cite">

> If you use `pymatviz` in your research, [see how to cite](#how-to-cite-pymatviz). Check out [existing papers using `pymatviz`](#papers-using-pymatviz) for inspiration!

</slot>

## Installation

```sh
pip install pymatviz
```

Available extras include:

```sh
pip install 'pymatviz[pdf-export]' # save figures to PDF
pip install 'pymatviz[brillouin]' # render 3d Brillouin zones
```

## API Docs

See the [/api][/api] page.

[/api]: https://janosh.github.io/pymatviz/api

## Usage

See the Jupyter notebooks under [`examples/`](examples) for how to use `pymatviz`. PRs with additional examples are welcome! 🙏

|                                                                                                                        |                                                                                                                                                             |                                   |
| ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| [mlff_phonons.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/mlff_phonons.ipynb)                         | [![Open in Google Colab][Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/mlff_phonons.ipynb)             | [Launch Codespace][codespace url] |
| [matbench_dielectric_eda.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/matbench_dielectric_eda.ipynb)   | [![Open in Google Colab][Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/matbench_dielectric_eda.ipynb)  | [Launch Codespace][codespace url] |
| [mp_bimodal_e_form.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/mp_bimodal_e_form.ipynb)               | [![Open in Google Colab][Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/mp_bimodal_e_form.ipynb)        | [Launch Codespace][codespace url] |
| [matbench_perovskites_eda.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/matbench_perovskites_eda.ipynb) | [![Open in Google Colab][Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/matbench_perovskites_eda.ipynb) | [Launch Codespace][codespace url] |
| [mprester_ptable.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/mprester_ptable.ipynb)                   | [![Open in Google Colab][Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/mprester_ptable.ipynb)          | [Launch Codespace][codespace url] |

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg
[codespace url]: https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=340898532

## Periodic Table

See [`pymatviz/ptable/ptable_plotly.py`](pymatviz/ptable/ptable_plotly.py). The module supports heatmaps, heatmap splits (multiple values per element), histograms, scatter plots and line plots. All visualizations are interactive through [Plotly](https://plotly.com) and support displaying additional data on hover.

> [!WARNING]
> Version 0.16.0 of `pymatviz` dropped the matplotlib-based functions in `ptable_matplotlib.py` in https://github.com/janosh/pymatviz/pull/270. Please use the `plotly`-based functions shown below instead which have feature parity, interactivity and better test coverage.

|         [`ptable_heatmap_plotly(atomic_masses)`](assets/scripts/ptable_plotly/ptable_heatmap_plotly.py)         |    [`ptable_heatmap_plotly(compositions, log=True)`](assets/scripts/ptable_plotly/ptable_heatmap_plotly.py)     |
| :-------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: |
|                                    ![ptable-heatmap-plotly-more-hover-data]                                     |                                          ![ptable-heatmap-plotly-log]                                           |
|               [`ptable_hists_plotly(data)`](assets/scripts/ptable_plotly/ptable_hists_plotly.py)                |     [`ptable_scatter_plotly(data, mode="markers")`](assets/scripts/ptable_plotly/ptable_scatter_plotly.py)      |
|                                             ![ptable-hists-plotly]                                              |                                        ![ptable-scatter-plotly-markers]                                         |
| [`ptable_heatmap_splits_plotly(2_vals_per_elem)`](assets/scripts/ptable_plotly/ptable_heatmap_splits_plotly.py) | [`ptable_heatmap_splits_plotly(3_vals_per_elem)`](assets/scripts/ptable_plotly/ptable_heatmap_splits_plotly.py) |
|                                        ![ptable-heatmap-splits-plotly-2]                                        |                                        ![ptable-heatmap-splits-plotly-3]                                        |

[ptable-heatmap-plotly-log]: assets/svg/ptable-heatmap-plotly-log.svg
[ptable-heatmap-plotly-more-hover-data]: assets/svg/ptable-heatmap-plotly-more-hover-data.svg
[ptable-heatmap-splits-plotly-2]: assets/svg/ptable-heatmap-splits-plotly-2.svg
[ptable-heatmap-splits-plotly-3]: assets/svg/ptable-heatmap-splits-plotly-3.svg
[ptable-hists-plotly]: assets/svg/ptable-hists-plotly.svg
[ptable-scatter-plotly-markers]: assets/svg/ptable-scatter-plotly-markers.svg

### Dash app using `ptable_heatmap_plotly()`

See [`examples/mprester_ptable.ipynb`](https://github.com/janosh/pymatviz/blob/main/examples/mprester_ptable.ipynb).

[https://user-images.githubusercontent.com/30958850/181644052-b330f0a2-70fc-451c-8230-20d45d3af72f.mp4](https://user-images.githubusercontent.com/30958850/181644052-b330f0a2-70fc-451c-8230-20d45d3af72f.mp4)

## Phonons

See [`examples/mlff_phonons.ipynb`](https://github.com/janosh/pymatviz/blob/main/examples/mlff_phonons.ipynb) for usage example.

|               [`phonon_bands(bands_dict)`](assets/scripts/phonons/phonon_bands.py)               |                  [`phonon_dos(doses_dict)`](assets/scripts/phonons/phonon_dos.py)                  |
| :----------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
|                                         ![phonon-bands]                                          |                                           ![phonon-dos]                                            |
| [`phonon_bands_and_dos(bands_dict, doses_dict)`](assets/scripts/phonons/phonon_bands_and_dos.py) | [`phonon_bands_and_dos(single_bands, single_dos)`](assets/scripts/phonons/phonon_bands_and_dos.py) |
|                                 ![phonon-bands-and-dos-mp-2758]                                  |                                  ![phonon-bands-and-dos-mp-23907]                                  |

[phonon-bands]: assets/svg/phonon-bands-mp-2758.svg
[phonon-dos]: assets/svg/phonon-dos-mp-2758.svg
[phonon-bands-and-dos-mp-2758]: assets/svg/phonon-bands-and-dos-mp-2758.svg
[phonon-bands-and-dos-mp-23907]: assets/svg/phonon-bands-and-dos-mp-23907.svg

## Structure

See [`pymatviz/structure_viz/plotly.py`](pymatviz/structure_viz/plotly.py).

| [`structure_3d_plotly(hea_structure)`](assets/scripts/structure_viz/structure_3d_plotly.py) | [`structure_3d_plotly(lco_supercell)`](assets/scripts/structure_viz/structure_3d_plotly.py) |
| :-----------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
|                                 ![hea-structure-3d-plotly]                                  |                                 ![lco-structure-3d-plotly]                                  |
|  [`structure_2d_plotly(six_structs)`](assets/scripts/structure_viz/structure_2d_plotly.py)  |  [`structure_3d_plotly(six_structs)`](assets/scripts/structure_viz/structure_3d_plotly.py)  |
|                          ![matbench-phonons-structures-2d-plotly]                           |                          ![matbench-phonons-structures-3d-plotly]                           |

[matbench-phonons-structures-2d-plotly]: assets/svg/matbench-phonons-structures-2d-plotly.svg
[matbench-phonons-structures-3d-plotly]: assets/svg/matbench-phonons-structures-3d-plotly.svg
[hea-structure-3d-plotly]: assets/svg/hea-structure-3d-plotly.svg
[lco-structure-3d-plotly]: assets/svg/lco-structure-3d-plotly.svg

## Brillouin Zone

See [`pymatviz/brillouin.py`](pymatviz/brillouin.py).

|   [`brillouin_zone_3d(cubic_struct)`](assets/scripts/brillouin/brillouin_zone_3d.py)    |  [`brillouin_zone_3d(hexagonal_struct)`](assets/scripts/brillouin/brillouin_zone_3d.py)   |
| :-------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------: |
|                               ![brillouin-cubic-mp-10018]                               |                             ![brillouin-hexagonal-mp-862690]                              |
| [`brillouin_zone_3d(monoclinic_struct)`](assets/scripts/brillouin/brillouin_zone_3d.py) | [`brillouin_zone_3d(orthorhombic_struct)`](assets/scripts/brillouin/brillouin_zone_3d.py) |
|                           ![brillouin-monoclinic-mp-1183089]                            |                                ![brillouin-volumes-3-cols]                                |

[brillouin-cubic-mp-10018]: assets/svg/brillouin-cubic-mp-10018.svg
[brillouin-hexagonal-mp-862690]: assets/svg/brillouin-hexagonal-mp-862690.svg
[brillouin-monoclinic-mp-1183089]: assets/svg/brillouin-monoclinic-mp-1183089.svg
[brillouin-orthorhombic-mp-1183085]: assets/svg/brillouin-orthorhombic-mp-1183085.svg
[brillouin-volumes-3-cols]: assets/svg/brillouin-volumes-3-cols.svg

## X-Ray Diffraction

See [`pymatviz/xrd.py`](pymatviz/xrd.py).

|             [`xrd_pattern(pattern)`](assets/scripts/xrd/xrd_pattern.py)             |  [`xrd_pattern({key1: patt1, key2: patt2})`](assets/scripts/xrd/xrd_pattern.py)   |
| :---------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: |
|                                   ![xrd-pattern]                                    |                              ![xrd-pattern-multiple]                              |
| [`xrd_pattern(struct_dict, stack="horizontal")`](assets/scripts/xrd/xrd_pattern.py) | [`xrd_pattern(struct_dict, stack="vertical")`](assets/scripts/xrd/xrd_pattern.py) |
|                           ![xrd-pattern-horizontal-stack]                           |                           ![xrd-pattern-vertical-stack]                           |

[xrd-pattern]: assets/svg/xrd-pattern.svg
[xrd-pattern-multiple]: assets/svg/xrd-pattern-multiple.svg
[xrd-pattern-horizontal-stack]: assets/svg/xrd-pattern-horizontal-stack.svg
[xrd-pattern-vertical-stack]: assets/svg/xrd-pattern-vertical-stack.svg

## Radial Distribution Functions

See [`pymatviz/rdf/plotly.py`](pymatviz/rdf/plotly.py).

| [`element_pair_rdfs(pmg_struct)`](assets/scripts/rdf/element_pair_rdfs.py) | [`element_pair_rdfs({"A": struct1, "B": struct2})`](assets/scripts/rdf/element_pair_rdfs.py) |
| :------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
|                       ![element-pair-rdfs-Na8Nb8O24]                       |                          ![element-pair-rdfs-crystal-vs-amorphous]                           |

[element-pair-rdfs-Na8Nb8O24]: assets/svg/element-pair-rdfs-Na8Nb8O24.svg
[element-pair-rdfs-crystal-vs-amorphous]: assets/svg/element-pair-rdfs-crystal-vs-amorphous.svg

## Coordination

See [`pymatviz/coordination/plotly.py`](pymatviz/coordination/plotly.py).

|                  [`coordination_hist(struct_dict)`](assets/scripts/coordination/coordination_hist.py)                   |          [`coordination_hist(struct_dict, by_element=True)`](assets/scripts/coordination/coordination_hist.py)          |
| :---------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------: |
|                                               ![coordination-hist-single]                                               |                                      ![coordination-hist-by-structure-and-element]                                      |
| [`coordination_vs_cutoff_line(struct_dict, strategy=None)`](assets/scripts/coordination/coordination_vs_cutoff_line.py) | [`coordination_vs_cutoff_line(struct_dict, strategy=None)`](assets/scripts/coordination/coordination_vs_cutoff_line.py) |
|                                            ![coordination-vs-cutoff-single]                                             |                                           ![coordination-vs-cutoff-multiple]                                            |

[coordination-hist-single]: assets/svg/coordination-hist-single.svg
[coordination-hist-by-structure-and-element]: assets/svg/coordination-hist-by-structure-and-element.svg
[coordination-vs-cutoff-single]: assets/svg/coordination-vs-cutoff-single.svg
[coordination-vs-cutoff-multiple]: assets/svg/coordination-vs-cutoff-multiple.svg

## Sunburst

See [`pymatviz/sunburst.py`](pymatviz/sunburst.py).

| [`spacegroup_sunburst([65, 134, 225, ...])`](assets/scripts/sunburst/spacegroup_sunburst.py) | [`chem_sys_sunburst(["FeO", "Fe2O3", "LiPO4", ...])`](assets/scripts/sunburst/chem_sys_sunburst.py) |
| :------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------: |
|                                     ![spg-num-sunburst]                                      |                                    ![chem-sys-sunburst-ward-bmg]                                    |

[spg-num-sunburst]: assets/svg/spg-num-sunburst.svg
[chem-sys-sunburst-ward-bmg]: assets/svg/chem-sys-sunburst-ward-bmg.svg

## Treemap

See [`pymatviz/treemap.py`](pymatviz/treemap.py).

| [`chem_sys_treemap(["FeO", "Fe2O3", "LiPO4", ...])`](assets/scripts/treemap/chem_sys_treemap.py) | [`chem_sys_treemap(["FeO", "Fe2O3", "LiPO4", ...], group_by="formula")`](assets/scripts/treemap/chem_sys_treemap.py) |
| :----------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------: |
|                                   ![chem-sys-treemap-formula]                                    |                                             ![chem-sys-treemap-ward-bmg]                                             |

[chem-sys-treemap-formula]: assets/svg/chem-sys-treemap-formula.svg
[chem-sys-treemap-ward-bmg]: assets/svg/chem-sys-treemap-ward-bmg.svg

## Rainclouds

See [`pymatviz/rainclouds.py`](pymatviz/rainclouds.py).

| [`rainclouds(two_key_dict)`](assets/scripts/rainclouds/rainclouds.py) | [`rainclouds(three_key_dict)`](assets/scripts/rainclouds/rainclouds.py) |
| :-------------------------------------------------------------------: | :---------------------------------------------------------------------: |
|                         ![rainclouds-bimodal]                         |                         ![rainclouds-trimodal]                          |

[rainclouds-bimodal]: assets/svg/rainclouds-bimodal.svg
[rainclouds-trimodal]: assets/svg/rainclouds-trimodal.svg

## Sankey

See [`pymatviz/sankey.py`](pymatviz/sankey.py).

| [`sankey_from_2_df_cols(df_perovskites)`](assets/scripts/sankey/sankey_from_2_df_cols.py) | [`sankey_from_2_df_cols(df_space_groups)`](assets/scripts/sankey/sankey_from_2_df_cols.py) |
| :---------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: |
|                           ![sankey-spglib-vs-aflow-spacegroups]                           |                            ![sankey-crystal-sys-to-spg-symbol]                             |

[sankey-spglib-vs-aflow-spacegroups]: assets/svg/sankey-spglib-vs-aflow-spacegroups.svg
[sankey-crystal-sys-to-spg-symbol]: assets/svg/sankey-crystal-sys-to-spg-symbol.svg

## Bar Plots

See [`pymatviz/bar.py`](pymatviz/bar.py).

| [`spacegroup_bar([65, 134, 225, ...], backend="plotly")`](assets/scripts/bar/spacegroup_bar.py) | [`spacegroup_bar(["C2/m", "P-43m", "Fm-3m", ...], backend="plotly")`](assets/scripts/bar/spacegroup_bar.py) |
| :---------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------: |
|                                     ![spg-num-hist-plotly]                                      |                                          ![spg-symbol-hist-plotly]                                          |

[spg-symbol-hist-plotly]: assets/svg/spg-symbol-hist-plotly.svg
[spg-num-hist-plotly]: assets/svg/spg-num-hist-plotly.svg

## Histograms

See [`pymatviz/histogram.py`](pymatviz/histogram.py).

| [`elements_hist(compositions, log=True, bar_values='count')`](assets/scripts/histogram/elements_hist.py) | [`histogram({'key1': values1, 'key2': values2})`](assets/scripts/histogram/histogram.py) |
| :------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------: |
|                                             ![elements-hist]                                             |                                    ![histogram-ecdf]                                     |

[histogram-ecdf]: assets/svg/histogram-ecdf.svg

## Scatter Plots

See [`pymatviz/scatter.py`](pymatviz/scatter.py).

| [`density_scatter_plotly(df, x=x_col, y=y_col, ...)`](assets/scripts/scatter/density_scatter_plotly.py) | [`density_scatter_plotly(df, x=x_col, y=y_col, ...)`](assets/scripts/scatter/density_scatter_plotly.py) |
| :-----------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------: |
|                                        ![density-scatter-plotly]                                        |                                     ![density-scatter-plotly-blobs]                                     |
|               [`density_scatter(xs, ys, ...)`](assets/scripts/scatter/density_scatter.py)               |     [`density_scatter_with_hist(xs, ys, ...)`](assets/scripts/scatter/density_scatter_with_hist.py)     |
|                                           ![density-scatter]                                            |                                      ![density-scatter-with-hist]                                       |
|                [`density_hexbin(xs, ys, ...)`](assets/scripts/scatter/density_hexbin.py)                |      [`density_hexbin_with_hist(xs, ys, ...)`](assets/scripts/scatter/density_hexbin_with_hist.py)      |
|                                            ![density-hexbin]                                            |                                       ![density-hexbin-with-hist]                                       |

[density-scatter-plotly]: assets/svg/density-scatter-plotly.svg
[density-scatter-plotly-blobs]: assets/svg/density-scatter-plotly-blobs.svg
[density-hexbin-with-hist]: assets/svg/density-hexbin-with-hist.svg
[density-hexbin]: assets/svg/density-hexbin.svg
[density-scatter-with-hist]: assets/svg/density-scatter-with-hist.svg
[density-scatter]: assets/svg/density-scatter.svg

## Uncertainty

See [`pymatviz/uncertainty.py`](pymatviz/uncertainty.py).

|             [`qq_gaussian(y_true, y_pred, y_std)`](assets/scripts/uncertainty/qq_gaussian.py)             |             [`qq_gaussian(y_true, y_pred, y_std: dict)`](assets/scripts/uncertainty/qq_gaussian.py)             |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: |
|                                            ![normal-prob-plot]                                            |                                          ![normal-prob-plot-multiple]                                           |
| [`error_decay_with_uncert(y_true, y_pred, y_std)`](assets/scripts/uncertainty/error_decay_with_uncert.py) | [`error_decay_with_uncert(y_true, y_pred, y_std: dict)`](assets/scripts/uncertainty/error_decay_with_uncert.py) |
|                                        ![error-decay-with-uncert]                                         |                                       ![error-decay-with-uncert-multiple]                                       |

## Classification

See [`pymatviz/classify/confusion_matrix.py`](pymatviz/classify/confusion_matrix.py).

| [`confusion_matrix(conf_mat, ...)`](assets/scripts/classify/confusion_matrix.py) | [`confusion_matrix(y_true, y_pred, ...)`](assets/scripts/classify/confusion_matrix.py) |
| :------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------: |
|                          ![stability-confusion-matrix]                           |                           ![crystal-system-confusion-matrix]                           |

See [`pymatviz/classify/curves.py`](pymatviz/classify/curves.py).

| [`roc_curve_plotly(targets, probs_positive)`](assets/scripts/classify/roc_curve.py) | [`precision_recall_curve_plotly(targets, probs_positive)`](assets/scripts/classify/precision_recall_curve.py) |
| :---------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
|                            ![roc-curve-plotly-multiple]                             |                                   ![precision-recall-curve-plotly-multiple]                                   |

[roc-curve-plotly-multiple]: assets/svg/roc-curve-plotly-multiple.svg
[precision-recall-curve-plotly-multiple]: assets/svg/precision-recall-curve-plotly-multiple.svg
[stability-confusion-matrix]: assets/svg/stability-confusion-matrix.svg
[crystal-system-confusion-matrix]: assets/svg/crystal-system-confusion-matrix.svg
[error-decay-with-uncert-multiple]: assets/svg/error-decay-with-uncert-multiple.svg
[error-decay-with-uncert]: assets/svg/error-decay-with-uncert.svg
[elements-hist]: assets/svg/elements-hist.svg
[normal-prob-plot-multiple]: assets/svg/normal-prob-plot-multiple.svg
[normal-prob-plot]: assets/svg/normal-prob-plot.svg
[struct-2d-mp-12712-Hf9Zr9Pd24-disordered]: assets/svg/struct-2d-mp-12712-Hf9Zr9Pd24-disordered.svg
[struct-2d-mp-19017-Li4Mn0.8Fe1.6P4C1.6O16-disordered]: assets/svg/struct-2d-mp-19017-Li4Mn0.8Fe1.6P4C1.6O16-disordered.svg

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

Sorted by number of citations. Last updated 2025-03-15. Auto-generated [from Google Scholar](https://scholar.google.com/scholar?q=pymatviz). Manual additions via PR welcome.

1. C Zeni, R Pinsler, D Zügner et al. (2023). [Mattergen: a generative model for inorganic materials design](https://arxiv.org/abs/2312.03687) (cited by 119)
1. J Riebesell, REA Goodall, P Benner et al. (2023). [Matbench Discovery--A framework to evaluate machine learning crystal stability predictions](https://arxiv.org/abs/2308.14920) (cited by 41)
1. C Chen, DT Nguyen, SJ Lee et al. (2024). [Accelerating computational materials discovery with machine learning and cloud high-performance computing: from large-scale screening to experimental validation](https://pubs.acs.org/doi/abs/10.1021/jacs.4c03849) (cited by 35)
1. L Barroso-Luque, M Shuaibi, X Fu et al. (2024). [Open materials 2024 (omat24) inorganic materials dataset and models](https://www.rivista.ai/wp-content/uploads/2024/10/2410.12771v1.pdf) (cited by 26)
1. M Giantomassi, G Materzanini (2024). [Systematic assessment of various universal machine‐learning interatomic potentials](https://onlinelibrary.wiley.com/doi/abs/10.1002/mgea.58) (cited by 14)
1. AA Naik, C Ertural, P Benner et al. (2023). [A quantum-chemical bonding database for solid-state materials](https://www.nature.com/articles/s41597-023-02477-5) (cited by 11)
1. K Li, AN Rubungo, X Lei et al. (2025). [Probing out-of-distribution generalization in machine learning for materials](https://www.nature.com/articles/s43246-024-00731-w) (cited by 6)
1. N Tuchinda, CA Schuh (2025). [Grain Boundary Segregation and Embrittlement of Aluminum Binary Alloys from First Principles](https://arxiv.org/abs/2502.01579) (cited by 2)
1. A Onwuli, KT Butler, A Walsh (2024). [Ionic species representations for materials informatics](https://pubs.aip.org/aip/aml/article/2/3/036112/3313198) (cited by 1)
1. A Peng, MY Guo (2025). [The OpenLAM Challenges](https://arxiv.org/abs/2501.16358)
1. F Therrien, JA Haibeh (2025). [OBELiX: A Curated Dataset of Crystal Structures and Experimentally Measured Ionic Conductivities for Lithium Solid-State Electrolytes](https://arxiv.org/abs/2502.14234)
1. HH Li, Q Chen, G Ceder (2024). [Voltage Mining for (De) lithiation-Stabilized Cathodes and a Machine Learning Model for Li-Ion Cathode Voltage](https://pubs.acs.org/doi/abs/10.1021/acsami.4c15742)
1. A Kapeliukha, RA Mayo (2025). [MOSAEC-DB: a comprehensive database of experimental metal–organic frameworks with verified chemical accuracy suitable for molecular simulations](https://pubs.rsc.org/en/content/articlehtml/2025/sc/d4sc07438f)
1. N Tuchinda, CA Schuh (2025). [A Grain Boundary Embrittlement Genome for Substitutional Cubic Alloys](https://arxiv.org/abs/2502.06531)
1. K Yan, M Bohde, A Kryvenko (2025). [A Materials Foundation Model via Hybrid Invariant-Equivariant Architectures](https://arxiv.org/abs/2503.05771)
1. Daniel W. Davies, Keith T. Butler, Adam J. Jackson et al. (2024). [SMACT: Semiconducting Materials by Analogy and Chemical Theory](https://github.com/WMD-group/SMACT)
1. Aaron D. Kaplan, Runze Liu, Ji Qi et al. (2025). [A Foundational Potential Energy Surface Dataset for Materials](http://arxiv.org/abs/2503.04070)
1. Fei Shuang, Zixiong Wei, Kai Liu et al. (2025). [Universal machine learning interatomic potentials poised to supplant DFT in modeling general defects in metals and random alloys](http://arxiv.org/abs/2502.03578)
1. Yingheng Tang, Wenbin Xu, Jie Cao et al. (2025). [MatterChat: A Multi-Modal LLM for Material Science](http://arxiv.org/abs/2502.13107)
1. Hui Zheng, Eric Sivonxay, Rasmus Christensen et al. (2024). [The ab initio non-crystalline structure database: empowering machine learning to decode diffusivity](https://www.nature.com/articles/s41524-024-01469-2)
