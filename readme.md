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
[![PyPI Downloads](https://img.shields.io/pypi/dm/pymatviz?logo=icloud&logoColor=white)](https://pypistats.org/packages/pymatviz)
[![Zenodo](https://img.shields.io/badge/DOI-10.5281/zenodo.10456384-blue?logo=Zenodo&logoColor=white)](https://zenodo.org/records/10456384)

</h4>

<slot name="how-to-cite">

> If you use `pymatviz` in your research, [see how to cite](#how-to-cite-pymatviz).

</slot>

## Installation

```sh
pip install pymatviz
```

## API Docs

See the [/api] page.

[/api]: https://janosh.github.io/pymatviz/api

## Usage

See the Jupyter notebooks under [`examples/`](examples) for how to use `pymatviz`. PRs with additional examples are welcome! 🙏

|                                                                                                                        |                                                                                                                                       |                                      |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| [mlff_phonons.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/mlff_phonons.ipynb)                         | [![Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/mlff_phonons.ipynb)             | [![Launch Codespace]][codespace url] |
| [matbench_dielectric_eda.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/matbench_dielectric_eda.ipynb)   | [![Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/matbench_dielectric_eda.ipynb)  | [![Launch Codespace]][codespace url] |
| [mp_bimodal_e_form.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/mp_bimodal_e_form.ipynb)               | [![Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/mp_bimodal_e_form.ipynb)        | [![Launch Codespace]][codespace url] |
| [matbench_perovskites_eda.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/matbench_perovskites_eda.ipynb) | [![Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/matbench_perovskites_eda.ipynb) | [![Launch Codespace]][codespace url] |
| [mprester_ptable.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/mprester_ptable.ipynb)                   | [![Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/mprester_ptable.ipynb)          | [![Launch Codespace]][codespace url] |

[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg
[Launch Codespace]: https://img.shields.io/badge/Launch-Codespace-darkblue?logo=github
[codespace url]: https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=340898532

## Periodic Table

See [`pymatviz/ptable/ptable_matplotlib.py`](pymatviz/ptable/ptable_matplotlib.py) and [`pymatviz/ptable/ptable_plotly.py`](pymatviz/ptable/ptable_plotly.py). `matplotlib` supports heatmaps, heatmap ratios, heatmap splits (multiple values per element), histograms, scatter plots and line plots. `plotly` currently only supports heatmaps (PRs to port over other `matplotlib` `ptable` variants to `plotly` are very welcome!). The `plotly` heatmap supports displaying additional data on hover or full interactivity through [Dash](https://plotly.com/dash).

|                    [`ptable_heatmap(compositions, log=True)`](pymatviz/ptable/ptable_matplotlib.py)                    |                   [`ptable_heatmap_ratio(comps_a, comps_b)`](pymatviz/ptable/ptable_matplotlib.py)                    |
| :--------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: |
|                                                   ![ptable-heatmap]                                                    |                                                ![ptable-heatmap-ratio]                                                |
|                       [`ptable_heatmap_plotly(atomic_masses)`](pymatviz/ptable/ptable_plotly.py)                       |                [`ptable_heatmap_plotly(compositions, log=True)`](pymatviz/ptable/ptable_matplotlib.py)                |
|                                        ![ptable-heatmap-plotly-more-hover-data]                                        |                                             ![ptable-heatmap-plotly-log]                                              |
|                   [`ptable_hists(data, colormap="coolwarm")`](pymatviz/ptable/ptable_matplotlib.py)                    |                            [`ptable_hists_plotly(data)`](pymatviz/ptable/ptable_plotly.py)                            |
|                                                    ![ptable-hists]                                                     |                                                ![ptable-hists-plotly]                                                 |
|                  [`ptable_scatters(data, colormap="coolwarm")`](pymatviz/ptable/ptable_matplotlib.py)                  |                             [`ptable_lines(data)`](pymatviz/ptable/ptable_matplotlib.py)                              |
|                                               ![ptable-scatters-parity]                                                |                                                    ![ptable-lines]                                                    |
| [`ptable_heatmap_splits(2_vals_per_elem, colormap="coolwarm", start_angle=135)`](pymatviz/ptable/ptable_matplotlib.py) | [`ptable_heatmap_splits(3_vals_per_elem, colormap="coolwarm", start_angle=90)`](pymatviz/ptable/ptable_matplotlib.py) |
|                                               ![ptable-heatmap-splits-2]                                               |                                              ![ptable-heatmap-splits-3]                                               |
|                  [`ptable_heatmap_splits_plotly(2_vals_per_elem)`](pymatviz/ptable/ptable_plotly.py)                   |                  [`ptable_heatmap_splits_plotly(3_vals_per_elem)`](pymatviz/ptable/ptable_plotly.py)                  |
|                                           ![ptable-heatmap-splits-plotly-2]                                            |                                           ![ptable-heatmap-splits-plotly-3]                                           |

[ptable-hists]: assets/svg/ptable-hists.svg
[ptable-lines]: assets/svg/homo-nuclear-mace-medium.svg
[ptable-scatters-parity]: assets/svg/ptable-scatters-parity.svg
[ptable-scatters-parabola]: assets/svg/ptable-scatters-parabola.svg
[ptable-heatmap-splits-2]: assets/svg/ptable-heatmap-splits-2.svg
[ptable-heatmap-splits-3]: assets/svg/ptable-heatmap-splits-3.svg
[ptable-heatmap-splits-plotly-2]: assets/svg/ptable-heatmap-splits-plotly-2.svg
[ptable-heatmap-splits-plotly-3]: assets/svg/ptable-heatmap-splits-plotly-3.svg
[ptable-hists-plotly]: assets/svg/ptable-hists-plotly.svg

## Phonons

See [`examples/mlff_phonons.ipynb`](https://github.com/janosh/pymatviz/blob/main/examples/mlff_phonons.ipynb) for usage example.

|           [`phonon_bands(bands_dict)`](pymatviz/phonons.py)           |             [`phonon_dos(doses_dict)`](pymatviz/phonons.py)             |
| :-------------------------------------------------------------------: | :---------------------------------------------------------------------: |
|                            ![phonon-bands]                            |                              ![phonon-dos]                              |
| [`phonon_bands_and_dos(bands_dict, doses_dict)`](pymatviz/phonons.py) | [`phonon_bands_and_dos(single_bands, single_dos)`](pymatviz/phonons.py) |
|                    ![phonon-bands-and-dos-mp-2758]                    |                    ![phonon-bands-and-dos-mp-23907]                     |

[phonon-bands]: assets/svg/phonon-bands-mp-2758.svg
[phonon-dos]: assets/svg/phonon-dos-mp-2758.svg
[phonon-bands-and-dos-mp-2758]: assets/svg/phonon-bands-and-dos-mp-2758.svg
[phonon-bands-and-dos-mp-23907]: assets/svg/phonon-bands-and-dos-mp-23907.svg

### Dash app using `ptable_heatmap_plotly()`

See [`examples/mprester_ptable.ipynb`](https://github.com/janosh/pymatviz/blob/main/examples/mprester_ptable.ipynb).

<https://user-images.githubusercontent.com/30958850/181644052-b330f0a2-70fc-451c-8230-20d45d3af72f.mp4>

## Structure

See [`pymatviz/structure_viz/(mpl|plotly).py`](pymatviz/structure_viz/plotly.py). Currently structure plotting is only supported with `matplotlib` in 2d. 3d interactive plots (probably with `plotly`) are on the road map.

|       [`structure_2d(mp_19017)`](pymatviz/structure_viz/mpl.py)        |       [`structure_2d(mp_12712)`](pymatviz/structure_viz/mpl.py)        |
| :--------------------------------------------------------------------: | :--------------------------------------------------------------------: |
|        ![struct-2d-mp-19017-Li4Mn0.8Fe1.6P4C1.6O16-disordered]         |              ![struct-2d-mp-12712-Hf9Zr9Pd24-disordered]               |
| [`structure_2d_plotly(six_structs)`](pymatviz/structure_viz/plotly.py) | [`structure_3d_plotly(six_structs)`](pymatviz/structure_viz/plotly.py) |
|                ![matbench-phonons-structures-2d-plotly]                |                ![matbench-phonons-structures-3d-plotly]                |

[matbench-phonons-structures-2d-plotly]: assets/svg/matbench-phonons-structures-2d-plotly.svg
[matbench-phonons-structures-3d-plotly]: assets/svg/matbench-phonons-structures-3d-plotly.svg

## X-Ray Diffraction

See [`pymatviz/xrd.py`](pymatviz/xrd.py).

|             [`xrd_pattern(pattern)`](pymatviz/xrd.py)             |             [`xrd_pattern({key1: patt1, key2: patt2})`](pymatviz/xrd.py)              |
| :---------------------------------------------------------------: | :-----------------------------------------------------------------------------------: |
|                          ![xrd-pattern]                           |                                ![xrd-pattern-multiple]                                |
| [`xrd_pattern(struct_dict, stack="horizontal")`](pymatviz/xrd.py) | [`xrd_pattern(struct_dict, stack="vertical", title="Custom Title")`](pymatviz/xrd.py) |
|                  ![xrd-pattern-horizontal-stack]                  |                             ![xrd-pattern-vertical-stack]                             |

[xrd-pattern]: assets/svg/xrd-pattern.svg
[xrd-pattern-multiple]: assets/svg/xrd-pattern-multiple.svg
[xrd-pattern-horizontal-stack]: assets/svg/xrd-pattern-horizontal-stack.svg
[xrd-pattern-vertical-stack]: assets/svg/xrd-pattern-vertical-stack.svg

## Radial Distribution Functions

See [`pymatviz/rdf/plotly.py`](pymatviz/rdf/plotly.py).

| [`rdf_plot(rdf)`](pymatviz/rdf/plotly.py) | [`rdf_plot(rdf, rdf2)`](pymatviz/rdf/plotly.py) |
| :---------------------------------------: | :---------------------------------------------: |
|      ![element-pair-rdfs-Na8Nb8O24]       |    ![element-pair-rdfs-crystal-vs-amorphous]    |

[element-pair-rdfs-Na8Nb8O24]: assets/svg/element-pair-rdfs-Na8Nb8O24.svg
[element-pair-rdfs-crystal-vs-amorphous]: assets/svg/element-pair-rdfs-crystal-vs-amorphous.svg

## Coordination

See [`pymatviz/coordination/plotly.py`](pymatviz/coordination/plotly.py).

|             [`coordination_hist(struct_dict)`](pymatviz/coordination/plotly.py)              |     [`coordination_hist(struct_dict, by_element=True)`](pymatviz/coordination/plotly.py)     |
| :------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
|                                 ![coordination-hist-single]                                  |                        ![coordination-hist-by-structure-and-element]                         |
| [`coordination_vs_cutoff_line(struct_dict, strategy=None)`](pymatviz/coordination/plotly.py) | [`coordination_vs_cutoff_line(struct_dict, strategy=None)`](pymatviz/coordination/plotly.py) |
|                               ![coordination-vs-cutoff-single]                               |                              ![coordination-vs-cutoff-multiple]                              |

[coordination-hist-single]: assets/svg/coordination-hist-single.svg
[coordination-hist-by-structure-and-element]: assets/svg/coordination-hist-by-structure-and-element.svg
[coordination-vs-cutoff-single]: assets/svg/coordination-vs-cutoff-single.svg
[coordination-vs-cutoff-multiple]: assets/svg/coordination-vs-cutoff-multiple.svg

## Sunburst

See [`pymatviz/sunburst.py`](pymatviz/sunburst.py).

| [`spacegroup_sunburst([65, 134, 225, ...])`](pymatviz/sunburst.py) | [`spacegroup_sunburst(["C2/m", "P-43m", "Fm-3m", ...])`](pymatviz/sunburst.py) |
| :----------------------------------------------------------------: | :----------------------------------------------------------------------------: |
|                        ![spg-num-sunburst]                         |                             ![spg-symbol-sunburst]                             |

## Rainclouds

See [`pymatviz/rainclouds.py`](pymatviz/rainclouds.py).

| [`rainclouds(two_key_dict)`](pymatviz/rainclouds.py) | [`rainclouds(three_key_dict)`](pymatviz/rainclouds.py) |
| :--------------------------------------------------: | :----------------------------------------------------: |
|                ![rainclouds-bimodal]                 |                 ![rainclouds-trimodal]                 |

[rainclouds-bimodal]: assets/svg/rainclouds-bimodal.svg
[rainclouds-trimodal]: assets/svg/rainclouds-trimodal.svg

## Sankey

See [`pymatviz/sankey.py`](pymatviz/sankey.py).

| [`sankey_from_2_df_cols(df_perovskites)`](pymatviz/sankey.py) | [`sankey_from_2_df_cols(df_space_groups)`](pymatviz/sankey.py) |
| :-----------------------------------------------------------: | :------------------------------------------------------------: |
|             ![sankey-spglib-vs-aflow-spacegroups]             |              ![sankey-crystal-sys-to-spg-symbol]               |

[sankey-spglib-vs-aflow-spacegroups]: assets/svg/sankey-spglib-vs-aflow-spacegroups.svg
[sankey-crystal-sys-to-spg-symbol]: assets/svg/sankey-crystal-sys-to-spg-symbol.svg

## Histograms

See [`pymatviz/histogram.py`](pymatviz/histogram.py).

| [`spacegroup_bar([65, 134, 225, ...], backend="matplotlib")`](pymatviz/histogram.py) | [`spacegroup_bar(["C2/m", "P-43m", "Fm-3m", ...], backend="matplotlib")`](pymatviz/histogram.py) |
| :----------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: |
|                              ![spg-num-hist-matplotlib]                              |                                  ![spg-symbol-hist-matplotlib]                                   |
|   [`spacegroup_bar([65, 134, 225, ...], backend="plotly")`](pymatviz/histogram.py)   |   [`spacegroup_bar(["C2/m", "P-43m", "Fm-3m", ...], backend="plotly")`](pymatviz/histogram.py)   |
|                                ![spg-num-hist-plotly]                                |                                    ![spg-symbol-hist-plotly]                                     |
| [`elements_hist(compositions, log=True, bar_values='count')`](pymatviz/histogram.py) |             [`histogram({'key1': values1, 'key2': values2})`](pymatviz/histogram.py)             |
|                                   ![elements-hist]                                   |                                        ![histogram-ecdf]                                         |

[spg-symbol-hist-plotly]: assets/svg/spg-symbol-hist-plotly.svg
[spg-num-hist-plotly]: assets/svg/spg-num-hist-plotly.svg
[spg-num-hist-matplotlib]: assets/svg/spg-num-hist-matplotlib.svg
[spg-symbol-hist-matplotlib]: assets/svg/spg-symbol-hist-matplotlib.svg
[histogram-ecdf]: assets/svg/histogram-ecdf.svg

## Scatter Plots

See [`pymatviz/scatter.py`](pymatviz/scatter.py).

| [`density_scatter_plotly(df, x=x_col, y=y_col, ...)`](pymatviz/scatter.py) | [`density_scatter_plotly(df, x=x_col, y=y_col, ...)`](pymatviz/scatter.py) |
| :------------------------------------------------------------------------: | :------------------------------------------------------------------------: |
|                         ![density-scatter-plotly]                          |                      ![density-scatter-plotly-blobs]                       |
|           [`density_scatter(xs, ys, ...)`](pymatviz/scatter.py)            |      [`density_scatter_with_hist(xs, ys, ...)`](pymatviz/scatter.py)       |
|                             ![density-scatter]                             |                        ![density-scatter-with-hist]                        |
|            [`density_hexbin(xs, ys, ...)`](pymatviz/scatter.py)            |       [`density_hexbin_with_hist(xs, ys, ...)`](pymatviz/scatter.py)       |
|                             ![density-hexbin]                              |                        ![density-hexbin-with-hist]                         |

[density-scatter-plotly]: assets/svg/density-scatter-plotly.svg
[density-scatter-plotly-blobs]: assets/svg/density-scatter-plotly-blobs.svg
[density-hexbin-with-hist]: assets/svg/density-hexbin-with-hist.svg
[density-hexbin]: assets/svg/density-hexbin.svg
[density-scatter-with-hist]: assets/svg/density-scatter-with-hist.svg
[density-scatter]: assets/svg/density-scatter.svg

## Uncertainty

See [`pymatviz/uncertainty.py`](pymatviz/uncertainty.py).

|       [`qq_gaussian(y_true, y_pred, y_std)`](pymatviz/uncertainty.py)       |       [`qq_gaussian(y_true, y_pred, y_std: dict)`](pymatviz/uncertainty.py)       |
| :-------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: |
|                             ![normal-prob-plot]                             |                           ![normal-prob-plot-multiple]                            |
| [`error_decay_with_uncert(y_true, y_pred, y_std)`](pymatviz/uncertainty.py) | [`error_decay_with_uncert(y_true, y_pred, y_std: dict)`](pymatviz/uncertainty.py) |
|                         ![error-decay-with-uncert]                          |                        ![error-decay-with-uncert-multiple]                        |

## Cumulative Metrics

See [`pymatviz/cumulative.py`](pymatviz/cumulative.py).

| [`cumulative_error(preds, targets)`](pymatviz/cumulative.py) | [`cumulative_residual(preds, targets)`](pymatviz/cumulative.py) |
| :----------------------------------------------------------: | :-------------------------------------------------------------: |
|                     ![cumulative-error]                      |                     ![cumulative-residual]                      |

## Classification

See [`pymatviz/relevance.py`](pymatviz/relevance.py).

| [`roc_curve(targets, proba_pos)`](pymatviz/relevance.py) | [`precision_recall_curve(targets, proba_pos)`](pymatviz/relevance.py) |
| :------------------------------------------------------: | :-------------------------------------------------------------------: |
|                       ![roc-curve]                       |                       ![precision-recall-curve]                       |

[cumulative-error]: assets/svg/cumulative-error.svg
[cumulative-residual]: assets/svg/cumulative-residual.svg
[error-decay-with-uncert-multiple]: assets/svg/error-decay-with-uncert-multiple.svg
[error-decay-with-uncert]: assets/svg/error-decay-with-uncert.svg
[elements-hist]: assets/svg/elements-hist.svg
[matbench-phonons-structures-2d]: assets/svg/matbench-phonons-structures-2d.svg
[normal-prob-plot-multiple]: assets/svg/normal-prob-plot-multiple.svg
[normal-prob-plot]: assets/svg/normal-prob-plot.svg
[precision-recall-curve]: assets/svg/precision-recall-curve.svg
[ptable-heatmap-plotly-log]: assets/svg/ptable-heatmap-plotly-log.svg
[ptable-heatmap-plotly-more-hover-data]: assets/svg/ptable-heatmap-plotly-more-hover-data.svg
[ptable-heatmap-ratio]: assets/svg/ptable-heatmap-ratio.svg
[ptable-heatmap]: assets/svg/ptable-heatmap.svg
[residual-vs-actual]: assets/svg/residual-vs-actual.svg
[roc-curve]: assets/svg/roc-curve.svg
[scatter-with-err-bar]: assets/svg/scatter-with-err-bar.svg
[spg-num-sunburst]: assets/svg/spg-num-sunburst.svg
[spg-symbol-sunburst]: assets/svg/spg-symbol-sunburst.svg
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
