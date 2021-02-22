<h1 align="center">ML Matrics</h1>

<h4 align="center">

A toolkit of metrics and visualizations for model performance in data-driven materials discovery.

[![Tests](https://github.com/janosh/mlmatrics/workflows/Tests/badge.svg)](https://github.com/janosh/mlmatrics/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/mlmatrics/master.svg)](https://results.pre-commit.ci/latest/github/janosh/mlmatrics/master)
[![License](https://img.shields.io/github/license/janosh/mlmatrics?label=License)](/license)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/janosh/mlmatrics?label=Repo+Size)](https://github.com/janosh/mlmatrics/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/janosh/mlmatrics?label=Last+Commit)](https://github.com/janosh/mlmatrics/commits)

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

To specify a certain branch or commit, append it's name or hash, e.g.

```txt
git+git://github.com/janosh/mlmatrics@master # default
git+git://github.com/janosh/mlmatrics@41b95ec
```

## Parity Plots

See [`mlmatrics/parity.py`](mlmatrics/parity.py).

| [`density_scatter(xs, ys, ...)`](mlmatrics/parity.py)  |      [`density_scatter_with_hist(xs, ys, ...)`](mlmatrics/parity.py)       |
| :----------------------------------------------------: | :------------------------------------------------------------------------: |
|     ![density_scatter](assets/density_scatter.svg)     |     ![density_scatter_with_hist](assets/density_scatter_with_hist.svg)     |
|  [`density_hexbin(xs, ys, ...)`](mlmatrics/parity.py)  |       [`density_hexbin_with_hist(xs, ys, ...)`](mlmatrics/parity.py)       |
| ![density_scatter_hex](assets/density_scatter_hex.svg) | ![density_scatter_hex_with_hist](assets/density_scatter_hex_with_hist.svg) |

## Elements

See [`mlmatrics/elements.py`](mlmatrics/elements.py).

|    [`ptable_elemental_prevalence(formulas)`](mlmatrics/elements.py)    |   [`hist_elemental_prevalence(formulas)`](mlmatrics/elements.py)   |
| :--------------------------------------------------------------------: | :----------------------------------------------------------------: |
| ![ptable_elemental_prevalence](assets/ptable_elemental_prevalence.svg) | ![hist_elemental_prevalence](assets/hist_elemental_prevalence.svg) |

## Uncertainty Calibration

See [`mlmatrics/quantile.py`](mlmatrics/quantile.py).

| [`qq_gaussian(y_test, y_pred, y_std)`](mlmatrics/quantile.py) | [`qq_gaussian(y_test, y_pred, y_std: dict)`](mlmatrics/quantile.py) |
| :-----------------------------------------------------------: | :-----------------------------------------------------------------: |
|       ![normal_prob_plot](assets/normal_prob_plot.svg)        | ![normal_prob_plot_multiple](assets/normal_prob_plot_multiple.svg)  |

## Ranking

See [`mlmatrics/ranking.py`](mlmatrics/ranking.py).

| [`err_decay(y_test, y_pred, y_std)`](mlmatrics/cumulative.py) | [`err_decay(y_test, y_pred, y_std: dict)`](mlmatrics/cumulative.py) |
| :-----------------------------------------------------------: | :-----------------------------------------------------------------: |
|              ![err_decay](assets/err_decay.svg)               |        ![err_decay_multiple](assets/err_decay_multiple.svg)         |

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

You can also run single tests by passing its name to the `-k` flag:

```sh
python -m pytest -k test_precision_recall_curve
```

Consult the [`pytest`](https://docs.pytest.org/en/stable/usage.html) docs for more details.
