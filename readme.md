<h1 align="center">
<img src="https://raw.githubusercontent.com/janosh/pymatviz/main/site/static/favicon.svg" alt="Logo" height="60px">
<br class="hide-in-docs">
pymatviz
</h1>

<h4 align="center" class="toc-exclude">

A toolkit for visualizations in materials informatics.

[![Tests](https://github.com/janosh/pymatviz/actions/workflows/test.yml/badge.svg)](https://github.com/janosh/pymatviz/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/pymatviz/main.svg)](https://results.pre-commit.ci/latest/github/janosh/pymatviz/main)
[![This project supports Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/pymatviz?logo=pypi&logoColor=white)](https://pypi.org/project/pymatviz)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pymatviz?logo=icloud&logoColor=white)](https://pypistats.org/packages/pymatviz)

</h4>

**Note**: This project is not endorsed by [`pymatgen`](https://github.com/materialsproject/pymatgen), but aims to complement it with additional plotting functionality.

## Installation

```sh
pip install pymatviz
```

## API Docs

See the [/api] page.

[/api]: https://janosh.github.io/pymatviz/api

## Usage

See the Jupyter notebooks under [`examples/`](examples) for how to use `pymatviz`.

|                                                                                                                        |                                      |                                                                                                              |                                                                                                                                       |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| [matbench_dielectric_eda.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/matbench_dielectric_eda.ipynb)   | [![Launch Codespace]][codespace url] | [![Binder]](https://mybinder.org/v2/gh/janosh/pymatviz/main?labpath=examples/matbench_dielectric_eda.ipynb)  | [![Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/matbench_dielectric_eda.ipynb)  |
| [mp_bimodal_e_form.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/mp_bimodal_e_form.ipynb)               | [![Launch Codespace]][codespace url] | [![Binder]](https://mybinder.org/v2/gh/janosh/pymatviz/main?labpath=examples/mp_bimodal_e_form.ipynb)        | [![Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/mp_bimodal_e_form.ipynb)        |
| [matbench_perovskites_eda.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/matbench_perovskites_eda.ipynb) | [![Launch Codespace]][codespace url] | [![Binder]](https://mybinder.org/v2/gh/janosh/pymatviz/main?labpath=examples/matbench_perovskites_eda.ipynb) | [![Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/matbench_perovskites_eda.ipynb) |
| [mprester_ptable.ipynb](https://github.com/janosh/pymatviz/blob/main/examples/mprester_ptable.ipynb)                   | [![Launch Codespace]][codespace url] | [![Binder]](https://mybinder.org/v2/gh/janosh/pymatviz/main?labpath=examples/mprester_ptable.ipynb)          | [![Open in Google Colab]](https://colab.research.google.com/github/janosh/pymatviz/blob/main/examples/mprester_ptable.ipynb)          |

[Binder]: https://mybinder.org/badge_logo.svg
[Open in Google Colab]: https://colab.research.google.com/assets/colab-badge.svg
[Launch Codespace]: https://img.shields.io/badge/Launch-Codespace-darkblue?logo=github
[codespace url]: https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=340898532

## Periodic Table

See [`pymatviz/ptable.py`](pymatviz/ptable.py). Heat maps of the periodic table can be plotted both with `matplotlib` and `plotly`. `plotly` supports displaying additional data on hover or full interactivity through [Dash](https://plotly.com/dash).

| [`ptable_heatmap(compositions, log=True)`](pymatviz/ptable.py) |    [`ptable_heatmap_ratio(comps_a, comps_b)`](pymatviz/ptable.py)     |
| :------------------------------------------------------------: | :-------------------------------------------------------------------: |
|                       ![ptable-heatmap]                        |                        ![ptable-heatmap-ratio]                        |
|  [`ptable_heatmap_plotly(atomic_masses)`](pymatviz/ptable.py)  | [`ptable_heatmap_plotly(compositions, log=True)`](pymatviz/ptable.py) |
|            ![ptable-heatmap-plotly-more-hover-data]            |                     ![ptable-heatmap-plotly-log]                      |

### Dash app using `ptable_heatmap_plotly()`

See [`examples/mprester_ptable.ipynb`](https://github.com/janosh/pymatviz/blob/main/examples/mprester_ptable.ipynb).

<https://user-images.githubusercontent.com/30958850/181644052-b330f0a2-70fc-451c-8230-20d45d3af72f.mp4>

## Sunburst

See [`pymatviz/sunburst.py`](pymatviz/sunburst.py).

| [`spacegroup_sunburst([65, 134, 225, ...])`](pymatviz/sunburst.py) | [`spacegroup_sunburst(["C2/m", "P-43m", "Fm-3m", ...])`](pymatviz/sunburst.py) |
| :----------------------------------------------------------------: | :----------------------------------------------------------------------------: |
|                        ![spg-num-sunburst]                         |                             ![spg-symbol-sunburst]                             |

## Sankey

See [`pymatviz/sankey.py`](pymatviz/sankey.py).

| [`sankey_from_2_df_cols(df_perovskites)`](pymatviz/sankey.py) | [`sankey_from_2_df_cols(df_rand_ints)`](pymatviz/sankey.py) |
| :-----------------------------------------------------------: | :---------------------------------------------------------: |
|             ![sankey-spglib-vs-aflow-spacegroups]             |              ![sankey-from-2-df-cols-randints]              |

## Structure

See [`pymatviz/structure_viz.py`](pymatviz/structure_viz.py). Currently structure plotting is only supported with `matplotlib` in 2d. 3d interactive plots (probably with `plotly`) are on the road map.

| [`plot_structure_2d(mp_19017)`](pymatviz/structure_viz.py) | [`plot_structure_2d(mp_12712)`](pymatviz/structure_viz.py) |
| :--------------------------------------------------------: | :--------------------------------------------------------: |
|  ![struct-2d-mp-19017-Li4Mn0.8Fe1.6P4C1.6O16-disordered]   |        ![struct-2d-mp-12712-Hf9Zr9Pd24-disordered]         |

![matbench-phonons-structures-2d]

## Histograms

See [`pymatviz/histograms.py`](pymatviz/histograms.py).

| [`spacegroup_hist([65, 134, 225, ...])`](pymatviz/histograms.py) |         [`spacegroup_hist(["C2/m", "P-43m", "Fm-3m", ...])`](pymatviz/histograms.py)          |
| :--------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
|                         ![spg-num-hist]                          |                                      ![spg-symbol-hist]                                       |
|    [`residual_hist(y_true, y_pred)`](pymatviz/histograms.py)     | [`hist_elemental_prevalence(compositions, log=True, bar_values='count')`](pymatviz/ptable.py) |
|                         ![residual-hist]                         |                                 ![hist-elemental-prevalence]                                  |

## Parity Plots

See [`pymatviz/parity.py`](pymatviz/parity.py).

|      [`density_scatter(xs, ys, ...)`](pymatviz/parity.py)       | [`density_scatter_with_hist(xs, ys, ...)`](pymatviz/parity.py)  |
| :-------------------------------------------------------------: | :-------------------------------------------------------------: |
|                       ![density-scatter]                        |                  ![density-scatter-with-hist]                   |
|       [`density_hexbin(xs, ys, ...)`](pymatviz/parity.py)       |  [`density_hexbin_with_hist(xs, ys, ...)`](pymatviz/parity.py)  |
|                        ![density-hexbin]                        |                   ![density-hexbin-with-hist]                   |
| [`scatter_with_err_bar(xs, ys, yerr, ...)`](pymatviz/parity.py) | [`residual_vs_actual(y_true, y_pred, ...)`](pymatviz/parity.py) |
|                     ![scatter-with-err-bar]                     |                      ![residual-vs-actual]                      |

## Uncertainty Calibration

See [`pymatviz/uncertainty.py`](pymatviz/uncertainty.py).

|       [`qq_gaussian(y_true, y_pred, y_std)`](pymatviz/uncertainty.py)       |       [`qq_gaussian(y_true, y_pred, y_std: dict)`](pymatviz/uncertainty.py)       |
| :-------------------------------------------------------------------------: | :-------------------------------------------------------------------------------: |
|                             ![normal-prob-plot]                             |                           ![normal-prob-plot-multiple]                            |
| [`error_decay_with_uncert(y_true, y_pred, y_std)`](pymatviz/uncertainty.py) | [`error_decay_with_uncert(y_true, y_pred, y_std: dict)`](pymatviz/uncertainty.py) |
|                         ![error-decay-with-uncert]                          |                        ![error-decay-with-uncert-multiple]                        |

## Cumulative Error & Residual

See [`pymatviz/cumulative.py`](pymatviz/cumulative.py).

| [`cumulative_error(preds, targets)`](pymatviz/cumulative.py) | [`cumulative_residual(preds, targets)`](pymatviz/cumulative.py) |
| :----------------------------------------------------------: | :-------------------------------------------------------------: |
|                     ![cumulative-error]                      |                     ![cumulative-residual]                      |

## Classification Metrics

See [`pymatviz/relevance.py`](pymatviz/relevance.py).

| [`roc_curve(targets, proba_pos)`](pymatviz/relevance.py) | [`precision_recall_curve(targets, proba_pos)`](pymatviz/relevance.py) |
| :------------------------------------------------------: | :-------------------------------------------------------------------: |
|                       ![roc-curve]                       |                       ![precision-recall-curve]                       |

## Correlation

See [`pymatviz/correlation.py`](pymatviz/correlation.py).

| [`marchenko_pastur(corr_mat, gamma=ncols/nrows)`](pymatviz/correlation.py) | [`marchenko_pastur(corr_mat_significant_eval, gamma=ncols/nrows)`](pymatviz/correlation.py) |
| :------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
|                            ![marchenko-pastur]                             |                            ![marchenko-pastur-significant-eval]                             |

## Glossary

1. **Residual** `y_res = y_true - y_pred`: The difference between ground truth target and model prediction.
2. **Error** `y_err = abs(y_true - y_pred)`: Absolute error between target and model prediction.
3. **Uncertainty** `y_std`: The model's estimate for its error, i.e. how much the model thinks its prediction can be trusted (`std` for standard deviation).

[cumulative-error]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/cumulative-error.svg
[cumulative-residual]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/cumulative-residual.svg
[density-hexbin-with-hist]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/density-hexbin-with-hist.svg
[density-hexbin]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/density-hexbin.svg
[density-scatter-with-hist]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/density-scatter-with-hist.svg
[density-scatter]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/density-scatter.svg
[error-decay-with-uncert-multiple]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/error-decay-with-uncert-multiple.svg
[error-decay-with-uncert]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/error-decay-with-uncert.svg
[hist-elemental-prevalence]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/hist-elemental-prevalence.svg
[marchenko-pastur-significant-eval]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/marchenko-pastur-significant-eval.svg
[marchenko-pastur]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/marchenko-pastur.svg
[matbench-phonons-structures-2d]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/matbench-phonons-structures-2d.svg
[normal-prob-plot-multiple]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/normal-prob-plot-multiple.svg
[normal-prob-plot]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/normal-prob-plot.svg
[precision-recall-curve]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/precision-recall-curve.svg
[ptable-heatmap-plotly-log]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/ptable-heatmap-plotly-log.svg
[ptable-heatmap-plotly-more-hover-data]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/ptable-heatmap-plotly-more-hover-data.svg
[ptable-heatmap-ratio]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/ptable-heatmap-ratio.svg
[ptable-heatmap]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/ptable-heatmap.svg
[residual-hist]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/residual-hist.svg
[residual-vs-actual]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/residual-vs-actual.svg
[roc-curve]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/roc-curve.svg
[sankey-from-2-df-cols-randints]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/sankey-from-2-df-cols-randints.svg
[sankey-spglib-vs-aflow-spacegroups]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/sankey-spglib-vs-aflow-spacegroups.svg
[scatter-with-err-bar]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/scatter-with-err-bar.svg
[spg-num-hist]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/spg-num-hist.svg
[spg-num-sunburst]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/spg-num-sunburst.svg
[spg-symbol-hist]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/spg-symbol-hist.svg
[spg-symbol-sunburst]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/spg-symbol-sunburst.svg
[struct-2d-mp-12712-Hf9Zr9Pd24-disordered]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/struct-2d-mp-12712-Hf9Zr9Pd24-disordered.svg
[struct-2d-mp-19017-Li4Mn0.8Fe1.6P4C1.6O16-disordered]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/struct-2d-mp-19017-Li4Mn0.8Fe1.6P4C1.6O16-disordered.svg
