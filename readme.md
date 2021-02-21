<h1 align="center">ML Matrics</h1>

<h4 align="center">

A toolkit of metrics and visualizations for model performance in data-driven materials discovery.

[![Tests](https://github.com/janosh/mlmatrics/workflows/Tests/badge.svg)](https://github.com/janosh/mlmatrics/actions)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/janosh/mlmatrics/master.svg)](https://results.pre-commit.ci/latest/github/janosh/mlmatrics/master)
[![License](https://img.shields.io/github/license/janosh/mlmatrics?label=License)](/license)
[![GitHub Repo Size](https://img.shields.io/github/repo-size/janosh/mlmatrics?label=Repo+Size)](https://github.com/janosh/mlmatrics/graphs/contributors)
[![GitHub last commit](https://img.shields.io/github/last-commit/janosh/mlmatrics?label=Last+Commit)](https://github.com/janosh/mlmatrics/commits)

</h4>

## Density Scatter

|     [`density_scatter.svg`](assets/density_scatter)     |     [`density_scatter_with_hist.svg`](assets/density_scatter_with_hist)     |
| :-----------------------------------------------------: | :-------------------------------------------------------------------------: |
|     ![density_scatter](assets/density_scatter.svg)      |     ![density_scatter_with_hist](assets/density_scatter_with_hist.svg)      |
| [`density_scatter_hex.svg`](assets/density_scatter_hex) | [`density_scatter_hex_with_hist.svg`](assets/density_scatter_hex_with_hist) |
| ![density_scatter_hex](assets/density_scatter_hex.svg)  | ![density_scatter_hex_with_hist](assets/density_scatter_hex_with_hist.svg)  |

## Elements

| [`ptable_elemental_prevalence.svg`](assets/ptable_elemental_prevalence) | [`hist_elemental_prevalence.svg`](assets/hist_elemental_prevalence) |
| :---------------------------------------------------------------------: | :-----------------------------------------------------------------: |
| ![ptable_elemental_prevalence](assets/ptable_elemental_prevalence.svg)  | ![hist_elemental_prevalence](assets/hist_elemental_prevalence.svg)  |

## Uncertainty Calibration

| [`std_calibration_single.svg`](assets/std_calibration_single) | [`std_calibration_multiple.svg`](assets/std_calibration_multiple) |
| :-----------------------------------------------------------: | :---------------------------------------------------------------: |
| ![std_calibration_single](assets/std_calibration_single.svg)  | ![std_calibration_multiple](assets/std_calibration_multiple.svg)  |

## Cumulative Error and Residual

|                     [`cumulative_error.svg`](assets/cumulative_error)                     | [`cumulative_residual.svg`](assets/cumulative_residual) |
| :---------------------------------------------------------------------------------------: | :-----------------------------------------------------: |
|                     ![cumulative_error](assets/cumulative_error.svg)                      | ![cumulative_residual](assets/cumulative_residual.svg)  |
| [`cumulative_error_cumulative_residual.svg`](assets/cumulative_error_cumulative_residual) |                                                         |
| ![cumulative_error_cumulative_residual](assets/cumulative_error_cumulative_residual.svg)  |                                                         |

## Adding `assets`

When adding new SVG assets, it's good to compress them before committing. To use [`svgo`](https://github.com/svg/svgo), install it with `npm -g svgo` and then compress all assets with `svgo assets`.
