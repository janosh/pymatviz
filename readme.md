<h1 align="center">ML Matrics</h1>

<h4 align="center">

A toolkit of metrics and visualizations for model performance in data-driven materials discovery.

[![Tests](https://github.com/janosh/ml-matrics/workflows/Tests/badge.svg)](https://github.com/janosh/ml-matrics/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/ml-matrics/main.svg)](https://results.pre-commit.ci/latest/github/janosh/ml-matrics/main)
[![This project supports Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/downloads)
[![PyPI](https://img.shields.io/pypi/v/ml-matrics)](https://pypi.org/project/ml-matrics)
[![PyPI Downloads](https://img.shields.io/pypi/dm/ml-matrics)](https://pypistats.org/packages/ml-matrics)

</h4>

## Installation

```sh
pip install ml-matrics
```

For a locally editable install, use

```sh
git clone https://github.com/janosh/ml-matrics && pip install -e ml-matrics
```

## Elements

See [`ml_matrics/elements.py`](ml_matrics/elements.py).

|      [`ptable_heatmap(compositions)`](ml_matrics/elements.py)       |                [`ptable_heatmap(compositions, log=True)`](ml_matrics/elements.py)                 |
| :-----------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: |
|                          ![ptable_heatmap]                          |                                       ![ptable_heatmap_log]                                       |
| [`ptable_heatmap_ratio(comps_a, comps_b)`](ml_matrics/elements.py)  |           [`ptable_heatmap_ratio(comps_b, comps_a, log=True)`](ml_matrics/elements.py)            |
|                       ![ptable_heatmap_ratio]                       |                                  ![ptable_heatmap_ratio_inverse]                                  |
| [`hist_elemental_prevalence(compositions)`](ml_matrics/elements.py) | [`hist_elemental_prevalence(compositions, log=True, bar_values='count')`](ml_matrics/elements.py) |
|                    ![hist_elemental_prevalence]                     |                              ![hist_elemental_prevalence_log_count]                               |

## Histograms

See [`ml_matrics/histograms.py`](ml_matrics/histograms.py).

| [`spacegroup_hist([65, 134, 225, ...])`](ml_matrics/histograms.py) | [`spacegroup_hist([65, 134, 225, ...], show_counts=False)`](ml_matrics/histograms.py) |
| :----------------------------------------------------------------: | :-----------------------------------------------------------------------------------: |
|                         ![spacegroup_hist]                         |                             ![spacegroup_hist_no_counts]                              |
|    [`residual_hist(y_true, y_pred)`](ml_matrics/histograms.py)     |          [`true_pred_hist(y_true, y_pred, y_std)`](ml_matrics/histograms.py)          |
|                          ![residual_hist]                          |                                   ![true_pred_hist]                                   |

## Parity Plots

See [`ml_matrics/parity.py`](ml_matrics/parity.py).

|      [`density_scatter(xs, ys, ...)`](ml_matrics/parity.py)       | [`density_scatter_with_hist(xs, ys, ...)`](ml_matrics/parity.py)  |
| :---------------------------------------------------------------: | :---------------------------------------------------------------: |
|                        ![density_scatter]                         |                   ![density_scatter_with_hist]                    |
|       [`density_hexbin(xs, ys, ...)`](ml_matrics/parity.py)       |  [`density_hexbin_with_hist(xs, ys, ...)`](ml_matrics/parity.py)  |
|                         ![density_hexbin]                         |                    ![density_hexbin_with_hist]                    |
| [`scatter_with_err_bar(xs, ys, yerr, ...)`](ml_matrics/parity.py) | [`residual_vs_actual(y_true, y_pred, ...)`](ml_matrics/parity.py) |
|                      ![scatter_with_err_bar]                      |                       ![residual_vs_actual]                       |

## Uncertainty Calibration

See [`ml_matrics/quantile.py`](ml_matrics/quantile.py).

| [`qq_gaussian(y_true, y_pred, y_std)`](ml_matrics/quantile.py) | [`qq_gaussian(y_true, y_pred, y_std: dict)`](ml_matrics/quantile.py) |
| :------------------------------------------------------------: | :------------------------------------------------------------------: |
|                      ![normal_prob_plot]                       |                     ![normal_prob_plot_multiple]                     |

## Ranking

See [`ml_matrics/ranking.py`](ml_matrics/ranking.py).

| [`err_decay(y_true, y_pred, y_std)`](ml_matrics/ranking.py) | [`err_decay(y_true, y_pred, y_std: dict)`](ml_matrics/ranking.py) |
| :---------------------------------------------------------: | :---------------------------------------------------------------: |
|                        ![err_decay]                         |                       ![err_decay_multiple]                       |

## Cumulative Error and Residual

See [`ml_matrics/cumulative.py`](ml_matrics/cumulative.py).

| [`cum_err(preds, targets)`](ml_matrics/cumulative.py) | [`cum_res(preds, targets)`](ml_matrics/cumulative.py) |
| :---------------------------------------------------: | :---------------------------------------------------: |
|                  ![cumulative_error]                  |                ![cumulative_residual]                 |

## Classification Metrics

See [`ml_matrics/relevance.py`](ml_matrics/relevance.py).

| [`roc_curve(targets, proba_pos)`](ml_matrics/relevance.py) | [`precision_recall_curve(targets, proba_pos)`](ml_matrics/relevance.py) |
| :--------------------------------------------------------: | :---------------------------------------------------------------------: |
|                        ![roc_curve]                        |                        ![precision_recall_curve]                        |

## Correlation

See [`ml_matrics/correlation.py`](ml_matrics/correlation.py).

| [`marchenko_pastur(corr_mat, gamma=ncols/nrows)`](ml_matrics/correlation.py) | [`marchenko_pastur(corr_mat_significant_eval, gamma=ncols/nrows)`](ml_matrics/correlation.py) |
| :--------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------: |
|                             ![marchenko_pastur]                              |                             ![marchenko_pastur_significant_eval]                              |

## Adding Assets

When adding new SVG assets, please compress them before committing. This can either be done online without setup at <https://vecta.io/nano> or on the command line with [`svgo`](https://github.com/svg/svgo). Install it with `npm -g svgo` (or `yarn global add svgo`). Then compress all assets in one go with `svgo assets`. (`svgo` is safe for multiple compressions).

## Testing

This project uses `pytest` ([docs](https://docs.pytest.org/en/stable/usage.html)). To run tests, use:

```sh
pytest # full test suite
pytest tests/test_cumulative.py # single file
pytest **/test_*_metrics.py # multiple files
pytest -k test_precision_recall_curve # -k takes regex matching test names
```

## Glossary

1. **Residual** `y_res = y_true - y_pred`: The difference between ground truth target and model prediction.
2. **Error** `y_err = abs(y_true - y_pred)`: Absolute error between target and model prediction.
3. **Uncertainty** `y_std`: The model's estimate for its error, i.e. how much the model thinks its prediction can be trusted. (`std` for standard deviation.)

[density_scatter]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/density_scatter.svg
[density_scatter_with_hist]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/density_scatter_with_hist.svg
[density_hexbin]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/density_hexbin.svg
[density_hexbin_with_hist]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/density_hexbin_with_hist.svg
[scatter_with_err_bar]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/scatter_with_err_bar.svg
[residual_vs_actual]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/residual_vs_actual.svg
[ptable_heatmap]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/ptable_heatmap.svg
[ptable_heatmap_log]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/ptable_heatmap_log.svg
[hist_elemental_prevalence]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/hist_elemental_prevalence.svg
[hist_elemental_prevalence_log_count]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/hist_elemental_prevalence_log_count.svg
[ptable_heatmap_ratio]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/ptable_heatmap_ratio.svg
[ptable_heatmap_ratio_inverse]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/ptable_heatmap_ratio_inverse.svg
[normal_prob_plot]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/normal_prob_plot.svg
[normal_prob_plot_multiple]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/normal_prob_plot_multiple.svg
[err_decay]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/err_decay.svg
[err_decay_multiple]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/err_decay_multiple.svg
[cumulative_error]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/cumulative_error.svg
[cumulative_residual]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/cumulative_residual.svg
[roc_curve]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/roc_curve.svg
[precision_recall_curve]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/precision_recall_curve.svg
[marchenko_pastur]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/marchenko_pastur.svg
[marchenko_pastur_significant_eval]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/marchenko_pastur_significant_eval.svg
[residual_hist]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/residual_hist.svg
[true_pred_hist]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/true_pred_hist.svg
[spacegroup_hist]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/spacegroup_hist.svg
[spacegroup_hist_no_counts]: https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/spacegroup_hist_no_counts.svg
