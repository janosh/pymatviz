<h1 align="center">ML Matrics</h1>

<h4 align="center">

A toolkit of metrics and visualizations for model performance in data-driven materials discovery.

[![Tests](https://github.com/janosh/ml-matrics/workflows/Tests/badge.svg)](https://github.com/janosh/ml-matrics/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/ml-matrics/main.svg)](https://results.pre-commit.ci/latest/github/janosh/ml-matrics/main)
[![PyPI](https://img.shields.io/pypi/v/ml-matrics)](https://pypi.org/project/ml-matrics)
[![This project supports Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/downloads)
[![License](https://img.shields.io/github/license/janosh/ml-matrics?label=License)](/license)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/janosh/ml-matrics?label=Repo+Size)](https://github.com/janosh/ml-matrics/graphs/contributors)

</h4>

## Installation

```sh
pip install ml-matrics
```

For a locally editable install, use

```sh
git clone https://github.com/janosh/ml-matrics && pip install -e ml-matrics
```

## Parity Plots

See [`ml_matrics/parity.py`](ml_matrics/parity.py).

|                              [`density_scatter(xs, ys, ...)`](ml_matrics/parity.py)                               |                              [`density_scatter_with_hist(xs, ys, ...)`](ml_matrics/parity.py)                               |
| :---------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|      ![density_scatter](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/density_scatter.svg)      | ![density_scatter_with_hist](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/density_scatter_with_hist.svg) |
|                               [`density_hexbin(xs, ys, ...)`](ml_matrics/parity.py)                               |                               [`density_hexbin_with_hist(xs, ys, ...)`](ml_matrics/parity.py)                               |
|       ![density_hexbin](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/density_hexbin.svg)       |  ![density_hexbin_with_hist](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/density_hexbin_with_hist.svg)  |
|                         [`scatter_with_err_bar(xs, ys, yerr, ...)`](ml_matrics/parity.py)                         |                              [`residual_vs_actual(y_true, y_pred, ...)`](ml_matrics/parity.py)                              |
| ![scatter_with_err_bar](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/scatter_with_err_bar.svg) |        ![residual_vs_actual](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/residual_vs_actual.svg)        |

## Elements

See [`ml_matrics/elements.py`](ml_matrics/elements.py).

|                              [`ptable_elemental_prevalence(compositions)`](ml_matrics/elements.py)                              |                                 [`ptable_elemental_prevalence(compositions, log=True)`](ml_matrics/elements.py)                                 |
| :-----------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------: |
| ![ptable_elemental_prevalence](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/ptable_elemental_prevalence.svg) |     ![ptable_elemental_prevalence_log](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/ptable_elemental_prevalence_log.svg)     |
|                               [`hist_elemental_prevalence(compositions)`](ml_matrics/elements.py)                               |                        [`hist_elemental_prevalence(compositions, log=True, bar_values='count')`](ml_matrics/elements.py)                        |
|   ![hist_elemental_prevalence](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/hist_elemental_prevalence.svg)   | ![hist_elemental_prevalence_log_count](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/hist_elemental_prevalence_log_count.svg) |
|                              [`ptable_elemental_ratio(comps_a, comps_b)`](ml_matrics/elements.py)                               |                                 [`ptable_elemental_ratio(comps_b, comps_a, log=True)`](ml_matrics/elements.py)                                  |
|      ![ptable_elemental_ratio](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/ptable_elemental_ratio.svg)      |          ![ptable_elemental_ratio_log](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/ptable_elemental_ratio_log.svg)          |

## Uncertainty Calibration

See [`ml_matrics/quantile.py`](ml_matrics/quantile.py).

|                      [`qq_gaussian(y_true, y_pred, y_std)`](ml_matrics/quantile.py)                       |                            [`qq_gaussian(y_true, y_pred, y_std: dict)`](ml_matrics/quantile.py)                             |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
| ![normal_prob_plot](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/normal_prob_plot.svg) | ![normal_prob_plot_multiple](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/normal_prob_plot_multiple.svg) |

## Ranking

See [`ml_matrics/ranking.py`](ml_matrics/ranking.py).

|                 [`err_decay(y_true, y_pred, y_std)`](ml_matrics/ranking.py)                 |                       [`err_decay(y_true, y_pred, y_std: dict)`](ml_matrics/ranking.py)                       |
| :-----------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------: |
| ![err_decay](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/err_decay.svg) | ![err_decay_multiple](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/err_decay_multiple.svg) |

## Cumulative Error and Residual

See [`ml_matrics/cumulative.py`](ml_matrics/cumulative.py).

|                           [`cum_err(preds, targets)`](ml_matrics/cumulative.py)                           |                              [`cum_res(preds, targets)`](ml_matrics/cumulative.py)                              |
| :-------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------: |
| ![cumulative_error](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/cumulative_error.svg) | ![cumulative_residual](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/cumulative_residual.svg) |

## Classification Metrics

See [`ml_matrics/relevance.py`](ml_matrics/relevance.py).

|                 [`roc_curve(targets, proba_pos)`](ml_matrics/relevance.py)                  |                        [`precision_recall_curve(targets, proba_pos)`](ml_matrics/relevance.py)                        |
| :-----------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------: |
| ![roc_curve](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/roc_curve.svg) | ![precision_recall_curve](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/precision_recall_curve.svg) |

## Correlation

See [`ml_matrics/correlation.py`](ml_matrics/correlation.py).

|               [`marchenko_pastur(corr_mat, gamma=ncols/nrows)`](ml_matrics/correlation.py)                |                        [`marchenko_pastur(corr_mat_significant_eval, gamma=ncols/nrows)`](ml_matrics/correlation.py)                        |
| :-------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------: |
| ![marchenko_pastur](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/marchenko_pastur.svg) | ![marchenko_pastur_significant_eval](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/marchenko_pastur_significant_eval.svg) |

## Histograms

See [`ml_matrics/histograms.py`](ml_matrics/histograms.py).

|                       [`residual_hist(y_true, y_pred)`](ml_matrics/histograms.py)                       |                  [`true_pred_hist(y_true, y_pred, y_std)`](ml_matrics/histograms.py)                  |
| :-----------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------: |
|   ![residual_hist](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/residual_hist.svg)   | ![true_pred_hist](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/true_pred_hist.svg) |
|                      [`spacegroup_hist(y_true, y_pred)`](ml_matrics/histograms.py)                      |                                                                                                       |
| ![spacegroup_hist](https://raw.githubusercontent.com/janosh/ml-matrics/main/assets/spacegroup_hist.svg) |                                                                                                       |

## Adding Assets

When adding new SVG assets, please compress them before committing. This can either be done online without setup at <https://vecta.io/nano> or on the command line with [`svgo`](https://github.com/svg/svgo). Install it with `npm -g svgo` (or `yarn global add svgo`). Then compress all assets in one go with `svgo assets`. (`svgo` is safe for multiple compressions).

## Testing

This project uses [`pytest`](https://docs.pytest.org/en/stable/usage.html). To run the entire test suite:

```sh
python -m pytest
```

To run individual or groups of test files, pass `pytest` a path or glob pattern, respectively:

```sh
python -m pytest tests/test_cumulative.py
python -m pytest **/test_*_metrics.py
```

To run a single test, pass its name to the `-k` flag:

```sh
python -m pytest -k test_precision_recall_curve
```

Consult the [`pytest`](https://docs.pytest.org/en/stable/usage.html) docs for more details.

## Glossary

1. **Residual** `y_res = y_true - y_pred`: The difference between ground truth target and model prediction.
2. **Error** `y_err = abs(y_true - y_pred)`: Absolute error between target and model prediction.
3. **Uncertainty** `y_std`: The model's estimate for its error, i.e. how much the model thinks its prediction can be trusted. (`std` for standard deviation.)
