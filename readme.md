<h1 align="center">pymatviz</h1>

<h4 align="center">

A toolkit for visualizations in materials informatics. Works with [`pymatgen`](https://github.com/materialsproject/pymatgen).

[![Tests](https://github.com/janosh/pymatviz/actions/workflows/test.yml/badge.svg)](https://github.com/janosh/pymatviz/actions/workflows/test.yml)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/pymatviz/main.svg)](https://results.pre-commit.ci/latest/github/janosh/pymatviz/main)
[![This project supports Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/pymatviz)](https://pypi.org/project/pymatviz)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pymatviz)](https://pypistats.org/packages/pymatviz)

</h4>

## Installation

```sh
pip install pymatviz
```

## Elements

See [`pymatviz/elements.py`](pymatviz/elements.py).

|      [`ptable_heatmap(compositions)`](pymatviz/elements.py)       |                [`ptable_heatmap(compositions, log=True)`](pymatviz/elements.py)                 |
| :---------------------------------------------------------------: | :---------------------------------------------------------------------------------------------: |
|                         ![ptable_heatmap]                         |                                      ![ptable_heatmap_log]                                      |
| [`ptable_heatmap_ratio(comps_a, comps_b)`](pymatviz/elements.py)  |           [`ptable_heatmap_ratio(comps_b, comps_a, log=True)`](pymatviz/elements.py)            |
|                      ![ptable_heatmap_ratio]                      |                                 ![ptable_heatmap_ratio_inverse]                                 |
| [`hist_elemental_prevalence(compositions)`](pymatviz/elements.py) | [`hist_elemental_prevalence(compositions, log=True, bar_values='count')`](pymatviz/elements.py) |
|                   ![hist_elemental_prevalence]                    |                             ![hist_elemental_prevalence_log_count]                              |

## Sunburst

See [`pymatviz/sunburst.py`](pymatviz/sunburst.py).

| [`spacegroup_sunburst([65, 134, 225, ...])`](pymatviz/sunburst.py) | [`spacegroup_sunburst([65, 134, 225, ...], show_values="percent")`](pymatviz/sunburst.py) |
| :----------------------------------------------------------------: | :---------------------------------------------------------------------------------------: |
|                       ![spacegroup_sunburst]                       |                              ![spacegroup_sunburst_percent]                               |

## Structure

See [`pymatviz/struct_vis.py`](pymatviz/struct_vis.py).

| [`plot_structure_2d(pmg_struct)`](pymatviz/struct_vis.py) | [`plot_structure_2d(pmg_struct, show_unit_cell=False, site_labels=False)`](pymatviz/struct_vis.py) |
| :-------------------------------------------------------: | :------------------------------------------------------------------------------------------------: |
|             ![struct-2d-mp-19017-disordered]              |                                       ![struct-2d-mp-12712]                                        |

![mp-structures-2d]

## Histograms

See [`pymatviz/histograms.py`](pymatviz/histograms.py).

| [`spacegroup_hist([65, 134, 225, ...])`](pymatviz/histograms.py) | [`spacegroup_hist([65, 134, 225, ...], show_counts=False)`](pymatviz/histograms.py) |
| :--------------------------------------------------------------: | :---------------------------------------------------------------------------------: |
|                        ![spacegroup_hist]                        |                            ![spacegroup_hist_no_counts]                             |
|    [`residual_hist(y_true, y_pred)`](pymatviz/histograms.py)     |          [`true_pred_hist(y_true, y_pred, y_std)`](pymatviz/histograms.py)          |
|                         ![residual_hist]                         |                                  ![true_pred_hist]                                  |

## Parity Plots

See [`pymatviz/parity.py`](pymatviz/parity.py).

|      [`density_scatter(xs, ys, ...)`](pymatviz/parity.py)       | [`density_scatter_with_hist(xs, ys, ...)`](pymatviz/parity.py)  |
| :-------------------------------------------------------------: | :-------------------------------------------------------------: |
|                       ![density_scatter]                        |                  ![density_scatter_with_hist]                   |
|       [`density_hexbin(xs, ys, ...)`](pymatviz/parity.py)       |  [`density_hexbin_with_hist(xs, ys, ...)`](pymatviz/parity.py)  |
|                        ![density_hexbin]                        |                   ![density_hexbin_with_hist]                   |
| [`scatter_with_err_bar(xs, ys, yerr, ...)`](pymatviz/parity.py) | [`residual_vs_actual(y_true, y_pred, ...)`](pymatviz/parity.py) |
|                     ![scatter_with_err_bar]                     |                      ![residual_vs_actual]                      |

## Uncertainty Calibration

See [`pymatviz/quantile.py`](pymatviz/quantile.py).

| [`qq_gaussian(y_true, y_pred, y_std)`](pymatviz/quantile.py) | [`qq_gaussian(y_true, y_pred, y_std: dict)`](pymatviz/quantile.py) |
| :----------------------------------------------------------: | :----------------------------------------------------------------: |
|                     ![normal_prob_plot]                      |                    ![normal_prob_plot_multiple]                    |

## Ranking

See [`pymatviz/ranking.py`](pymatviz/ranking.py).

| [`err_decay(y_true, y_pred, y_std)`](pymatviz/ranking.py) | [`err_decay(y_true, y_pred, y_std: dict)`](pymatviz/ranking.py) |
| :-------------------------------------------------------: | :-------------------------------------------------------------: |
|                       ![err_decay]                        |                      ![err_decay_multiple]                      |

## Cumulative Error and Residual

See [`pymatviz/cumulative.py`](pymatviz/cumulative.py).

| [`cum_err(preds, targets)`](pymatviz/cumulative.py) | [`cum_res(preds, targets)`](pymatviz/cumulative.py) |
| :-------------------------------------------------: | :-------------------------------------------------: |
|                 ![cumulative_error]                 |               ![cumulative_residual]                |

## Classification Metrics

See [`pymatviz/relevance.py`](pymatviz/relevance.py).

| [`roc_curve(targets, proba_pos)`](pymatviz/relevance.py) | [`precision_recall_curve(targets, proba_pos)`](pymatviz/relevance.py) |
| :------------------------------------------------------: | :-------------------------------------------------------------------: |
|                       ![roc_curve]                       |                       ![precision_recall_curve]                       |

## Correlation

See [`pymatviz/correlation.py`](pymatviz/correlation.py).

| [`marchenko_pastur(corr_mat, gamma=ncols/nrows)`](pymatviz/correlation.py) | [`marchenko_pastur(corr_mat_significant_eval, gamma=ncols/nrows)`](pymatviz/correlation.py) |
| :------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------: |
|                            ![marchenko_pastur]                             |                            ![marchenko_pastur_significant_eval]                             |

## Migrating from `ml-matrics` to `pymatviz`

This library was renamed from `ml-matrics` to `pymatviz` between versions 0.3.0 and 0.4.0. To update existing Python files that import `ml-matrics` in place, run the following commands. On Linux:

```sh
find . -name '*.py' | xargs sed -i 's/^from ml_matrics import/from pymatviz import/g'
find . -name '*.py' | xargs sed -i 's/^from ml_matrics./from pymatviz./g'
find . -name '*.py' | xargs sed -i 's/^import ml_matrics/import pymatviz/g'
```

On Mac, replace `sed -i` with `sed -i ""`.

## Glossary

1. **Residual** `y_res = y_true - y_pred`: The difference between ground truth target and model prediction.
2. **Error** `y_err = abs(y_true - y_pred)`: Absolute error between target and model prediction.
3. **Uncertainty** `y_std`: The model's estimate for its error, i.e. how much the model thinks its prediction can be trusted. (`std` for standard deviation.)

[cumulative_error]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/cumulative_error.svg
[cumulative_residual]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/cumulative_residual.svg
[density_hexbin_with_hist]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/density_hexbin_with_hist.svg
[density_hexbin]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/density_hexbin.svg
[density_scatter_with_hist]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/density_scatter_with_hist.svg
[density_scatter]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/density_scatter.svg
[err_decay_multiple]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/err_decay_multiple.svg
[err_decay]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/err_decay.svg
[hist_elemental_prevalence_log_count]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/hist_elemental_prevalence_log_count.svg
[hist_elemental_prevalence]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/hist_elemental_prevalence.svg
[marchenko_pastur_significant_eval]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/marchenko_pastur_significant_eval.svg
[marchenko_pastur]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/marchenko_pastur.svg
[mp-structures-2d]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/mp-structures-2d.svg
[normal_prob_plot_multiple]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/normal_prob_plot_multiple.svg
[normal_prob_plot]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/normal_prob_plot.svg
[precision_recall_curve]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/precision_recall_curve.svg
[ptable_heatmap_log]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/ptable_heatmap_log.svg
[ptable_heatmap_ratio_inverse]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/ptable_heatmap_ratio_inverse.svg
[ptable_heatmap_ratio]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/ptable_heatmap_ratio.svg
[ptable_heatmap]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/ptable_heatmap.svg
[residual_hist]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/residual_hist.svg
[residual_vs_actual]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/residual_vs_actual.svg
[roc_curve]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/roc_curve.svg
[scatter_with_err_bar]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/scatter_with_err_bar.svg
[spacegroup_hist_no_counts]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/spacegroup_hist_no_counts.svg
[spacegroup_hist]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/spacegroup_hist.svg
[spacegroup_sunburst_percent]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/spacegroup_sunburst_percent.svg
[spacegroup_sunburst]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/spacegroup_sunburst.svg
[struct-2d-mp-12712]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/struct-2d-mp-12712.svg
[struct-2d-mp-19017-disordered]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/struct-2d-mp-19017-disordered.svg
[true_pred_hist]: https://raw.githubusercontent.com/janosh/pymatviz/main/assets/true_pred_hist.svg
