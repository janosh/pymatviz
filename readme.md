<h1 align="center">ML Matrics</h1>

<h4 align="center">

A toolkit of metrics and visualizations for model performance in data-driven materials discovery.

[![Tests](https://github.com/janosh/mlmatrics/workflows/Tests/badge.svg)](https://github.com/janosh/mlmatrics/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/mlmatrics/master.svg)](https://results.pre-commit.ci/latest/github/janosh/mlmatrics/master)
[![License](https://img.shields.io/github/license/janosh/mlmatrics?label=License)](/license)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/janosh/mlmatrics?label=Repo+Size)](https://github.com/janosh/mlmatrics/graphs/contributors)
[![This project supports Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org/downloads)

</h4>

## Installation

```sh
pip install -U git+https://github.com/janosh/mlmatrics
```

For a locally editable install, use

```sh
git clone https://github.com/janosh/mlmatrics && pip install -e mlmatrics
```

To specify a dependence on this package in `requirements.txt`, use

```txt
pandas==1.1.2
numpy==1.20.1
git+git://github.com/janosh/mlmatrics
```

To specify a specific branch or commit, append its name or hash, e.g.

```txt
git+git://github.com/janosh/mlmatrics@master # default
git+git://github.com/janosh/mlmatrics@41b95ec
```

## Parity Plots

See [`mlmatrics/parity.py`](mlmatrics/parity.py).

|      [`density_scatter(xs, ys, ...)`](mlmatrics/parity.py)       |  [`density_scatter_with_hist(xs, ys, ...)`](mlmatrics/parity.py)   |
| :--------------------------------------------------------------: | :----------------------------------------------------------------: |
|          ![density_scatter](assets/density_scatter.svg)          | ![density_scatter_with_hist](assets/density_scatter_with_hist.svg) |
|       [`density_hexbin(xs, ys, ...)`](mlmatrics/parity.py)       |   [`density_hexbin_with_hist(xs, ys, ...)`](mlmatrics/parity.py)   |
|           ![density_hexbin](assets/density_hexbin.svg)           |  ![density_hexbin_with_hist](assets/density_hexbin_with_hist.svg)  |
| [`scatter_with_err_bar(xs, ys, yerr, ...)`](mlmatrics/parity.py) |  [`residual_vs_actual(y_true, y_pred, ...)`](mlmatrics/parity.py)  |
|     ![scatter_with_err_bar](assets/scatter_with_err_bar.svg)     |        ![residual_vs_actual](assets/residual_vs_actual.svg)        |

## Elements

See [`mlmatrics/elements.py`](mlmatrics/elements.py).

|  [`ptable_elemental_prevalence(compositions)`](mlmatrics/elements.py)  |          [`ptable_elemental_prevalence(compositions, log=True)`](mlmatrics/elements.py)          |
| :--------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------: |
| ![ptable_elemental_prevalence](assets/ptable_elemental_prevalence.svg) |          ![ptable_elemental_prevalence_log](assets/ptable_elemental_prevalence_log.svg)          |
|   [`hist_elemental_prevalence(compositions)`](mlmatrics/elements.py)   | [`hist_elemental_prevalence(compositions, log=True, bar_values='count')`](mlmatrics/elements.py) |
|   ![hist_elemental_prevalence](assets/hist_elemental_prevalence.svg)   |      ![hist_elemental_prevalence_log_count](assets/hist_elemental_prevalence_log_count.svg)      |
|  [`ptable_elemental_ratio(comps_a, comps_b)`](mlmatrics/elements.py)   |          [`ptable_elemental_ratio(comps_b, comps_a, log=True)`](mlmatrics/elements.py)           |
|      ![ptable_elemental_ratio](assets/ptable_elemental_ratio.svg)      |               ![ptable_elemental_ratio_log](assets/ptable_elemental_ratio_log.svg)               |

## Uncertainty Calibration

See [`mlmatrics/quantile.py`](mlmatrics/quantile.py).

| [`qq_gaussian(y_true, y_pred, y_std)`](mlmatrics/quantile.py) | [`qq_gaussian(y_true, y_pred, y_std: dict)`](mlmatrics/quantile.py) |
| :-----------------------------------------------------------: | :-----------------------------------------------------------------: |
|       ![normal_prob_plot](assets/normal_prob_plot.svg)        | ![normal_prob_plot_multiple](assets/normal_prob_plot_multiple.svg)  |

## Ranking

See [`mlmatrics/ranking.py`](mlmatrics/ranking.py).

| [`err_decay(y_true, y_pred, y_std)`](mlmatrics/ranking.py) | [`err_decay(y_true, y_pred, y_std: dict)`](mlmatrics/ranking.py) |
| :--------------------------------------------------------: | :--------------------------------------------------------------: |
|             ![err_decay](assets/err_decay.svg)             |       ![err_decay_multiple](assets/err_decay_multiple.svg)       |

## Cumulative Error and Residual

See [`mlmatrics/cumulative.py`](mlmatrics/cumulative.py).

| [`cum_err(preds, targets)`](mlmatrics/cumulative.py) |  [`cum_res(preds, targets)`](mlmatrics/cumulative.py)  |
| :--------------------------------------------------: | :----------------------------------------------------: |
|   ![cumulative_error](assets/cumulative_error.svg)   | ![cumulative_residual](assets/cumulative_residual.svg) |

## Classification Metrics

See [`mlmatrics/relevance.py`](mlmatrics/relevance.py).

| [`roc_curve(targets, proba_pos)`](mlmatrics/relevance.py) | [`precision_recall_curve(targets, proba_pos)`](mlmatrics/relevance.py) |
| :-------------------------------------------------------: | :--------------------------------------------------------------------: |
|            ![roc_curve](assets/roc_curve.svg)             |      ![precision_recall_curve](assets/precision_recall_curve.svg)      |

## Correlation

See [`mlmatrics/correlation.py`](mlmatrics/correlation.py).

| [`marchenko_pastur(corr_mat, gamma=ncols/nrows)`](mlmatrics/correlation.py) | [`marchenko_pastur(corr_mat_significant_eval, gamma=ncols/nrows)`](mlmatrics/correlation.py) |
| :-------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
|              ![marchenko_pastur](assets/marchenko_pastur.svg)               |      ![marchenko_pastur_significant_eval](assets/marchenko_pastur_significant_eval.svg)      |

## Histograms

See [`mlmatrics/histograms.py`](mlmatrics/histograms.py).

| [`residual_hist(y_true, y_pred)`](mlmatrics/histograms.py) | [`true_pred_hist(y_true, y_pred, y_std)`](mlmatrics/histograms.py) |
| :--------------------------------------------------------: | :----------------------------------------------------------------: |
|         ![residual_hist](assets/residual_hist.svg)         |            ![true_pred_hist](assets/true_pred_hist.svg)            |

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
