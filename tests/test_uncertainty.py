from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from pymatviz.uncertainty import error_decay_with_uncert, qq_gaussian
from pymatviz.widgets.scatter_plot import ScatterPlotWidget
from tests.conftest import DfOrArrays, xs, y_pred, y_true


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike


@pytest.mark.parametrize("y_std", [y_true - y_pred, {"y_std_mock": y_true - y_pred}])
@pytest.mark.parametrize("percentiles", [True, False])
def test_error_decay_with_uncert(
    df_or_arrays: DfOrArrays,
    y_std: ArrayLike | dict[str, ArrayLike],
    percentiles: bool,
) -> None:
    """Error decay widget has uncertainty, optimal and random series plus a band."""
    df, x, y = df_or_arrays
    widget = error_decay_with_uncert(x, y, y_std, df=df, percentiles=percentiles)

    assert isinstance(widget, ScatterPlotWidget)
    assert widget.x_axis == {
        "label": "Excluded samples (%)" if percentiles else "Excluded samples"
    }
    assert widget.y_axis == {"label": "MAE"}
    assert widget.y_range == [0, None]
    labels = [srs["label"] for srs in widget.series]
    std_key = "y_std_mock" if isinstance(y_std, dict) else "std"
    assert labels == [std_key, "error", "random"]
    (band,) = widget.error_bands
    assert band["series"] == {"type": "series", "series_idx": 2}
    assert len(band["error"]) == len(widget.series[2]["x"])

    nonmonotonic = error_decay_with_uncert(
        np.zeros(3),
        np.array([9, 1, 8]),
        np.array([1, 2, 3]),
        percentiles=percentiles,
        n_rand=5,
    )
    uncert_series, _, random_series = nonmonotonic.series
    # Retaining 1, 2, then 3 samples must preserve the uncertainty order.
    np.testing.assert_array_equal(uncert_series["y"], [9, 5, 6])
    expected_x = np.array([2, 1, 0]) * (100 / 3 if percentiles else 1)
    np.testing.assert_allclose(uncert_series["x"], expected_x, rtol=1e-14, atol=0)
    np.testing.assert_array_equal(random_series["x"], uncert_series["x"])
    assert random_series["y"] == [6, 6, 6]
    # No sampling variance when all samples remain.
    assert nonmonotonic.error_bands[0]["error"][-1] == 0


@pytest.mark.parametrize("key", ["error", "random"])
def test_error_decay_reserved_keys(key: str) -> None:
    """y_std keys may not shadow the optimal and random baseline series."""
    with pytest.raises(ValueError, match=rf"y_std keys \['{key}'\] clash"):
        error_decay_with_uncert(y_true, y_pred, {key: np.ones(len(y_true))})


@pytest.mark.parametrize(("n_samples", "n_rand"), [(0, 5), (3, 0)])
def test_error_decay_invalid_sampling(n_samples: int, n_rand: int) -> None:
    """Reject empty data and missing random baseline samples."""
    with pytest.raises(ValueError, match="non-empty data and n_rand >= 1"):
        error_decay_with_uncert(
            np.zeros(n_samples), np.ones(n_samples), np.ones(n_samples), n_rand=n_rand
        )


@pytest.mark.parametrize("y_std", [xs, {"foo": xs, "bar": 0.1 * xs}])
@pytest.mark.parametrize("identity_line", [True, False, {"style": {"color": "blue"}}])
def test_qq_gaussian(
    df_or_arrays: DfOrArrays,
    y_std: ArrayLike | dict[str, ArrayLike],
    identity_line: bool | dict[str, dict[str, str]],
) -> None:
    """Q-Q widget has one shaded line per uncertainty and an optional identity."""
    df, x, y = df_or_arrays
    widget = qq_gaussian(x, y, y_std, df=df, identity_line=identity_line)

    assert isinstance(widget, ScatterPlotWidget)
    assert widget.x_axis == {"label": "Theoretical Quantile"}
    assert widget.y_axis == {"label": "Observed Quantile"}
    assert widget.x_range == widget.y_range == [0, 1]
    n_std = len(y_std) if isinstance(y_std, dict) else 1
    assert len(widget.series) == len(widget.fill_regions) == n_std
    for idx, (srs, fill) in enumerate(
        zip(widget.series, widget.fill_regions, strict=True)
    ):
        assert "miscal:" in srs["label"]
        assert fill["upper"] == {"type": "series", "series_idx": idx}
        assert fill["lower"]["values"] == srs["x"]
    if identity_line:
        (line,) = widget.ref_lines
        assert (line["type"], line["slope"], line["intercept"]) == ("diagonal", 1, 0)
        expected_color = "blue" if isinstance(identity_line, dict) else "red"
        assert line["style"]["color"] == expected_color
    else:
        assert widget.ref_lines is None


def test_qq_gaussian_multiple_uncertainties() -> None:
    """Q-Q plot labels each uncertainty and counts tied residuals in the CDF."""
    n_samples = 100
    np_rng = np.random.default_rng(seed=0)
    y_true = np_rng.normal(0, 1, n_samples)
    y_pred = y_true + np_rng.normal(0, 0.1, n_samples)
    uncertainties = {
        "epistemic": np.abs(np_rng.normal(0.2, 0.05, n_samples)),
        "aleatoric": np.abs(np_rng.normal(0.15, 0.03, n_samples)),
    }
    labels = [srs["label"] for srs in qq_gaussian(y_true, y_pred, uncertainties).series]
    assert [label.split(" (")[0] for label in labels] == list(uncertainties)

    extreme = qq_gaussian(np.zeros(2), np.full(2, 100), np.ones(2))
    assert extreme.series[-1]["label"] == "std (miscal: 0.50)"
    # Repeated residuals on a quantile boundary must all count toward the CDF.
    proportions = np.linspace(1e-10, 1 - 1e-10, 100)
    residuals = np.array([-1, *[norm.ppf(proportions[35])] * 2, 1])
    observed = qq_gaussian(np.zeros(4), residuals, np.ones(4)).series[-1]["y"]
    expected = [
        np.count_nonzero(residuals <= norm.ppf(prop)) / 4 for prop in proportions
    ]
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("plot", [qq_gaussian, error_decay_with_uncert])
@pytest.mark.parametrize("std_kind", ["array", "list", "mapping", "column", "columns"])
def test_uncertainty_dataframe_alignment(plot: Callable, std_kind: str) -> None:
    """Filter external uncertainties with the same row mask as DataFrame targets."""
    frame = pd.DataFrame(
        {"true": [0, np.nan, 0], "pred": [3, 99, 1], "std": [2, 10, 1]},
        index=pd.Index(["a", "b", "c"]),
    )
    uncertainties = {
        "array": frame["std"].to_numpy(),
        "list": frame["std"].tolist(),
        "mapping": {"std": frame["std"].to_numpy()},
        "column": "std",
        "columns": ["std"],
    }[std_kind]
    actual = plot("true", "pred", uncertainties, df=frame)
    expected = plot([0, 0], [3, 1], [2, 1])
    assert actual.series == expected.series


@pytest.mark.parametrize("plot", [qq_gaussian, error_decay_with_uncert])
@pytest.mark.parametrize(
    ("true_vals", "pred_vals", "std", "match"),
    [
        ([[0], [0]], [1, 2], [1, 1], "must be 1D"),
        ([0, 0], [1], [1, 1], "must be 1D"),
        ([0, 0], [1, 2], [1], "must be 1D"),
        ([0, 0], [1, 2], [[1], [1]], "must be 1D"),
        ([0, np.nan], [1, 2], [1, 1], "finite values"),
        ([0, 0], [1, 2], [1, np.inf], "finite values"),
    ],
)
def test_uncertainty_invalid_arrays(
    plot: Callable,
    true_vals: list,
    pred_vals: list,
    std: list,
    match: str,
) -> None:
    """Reject broadcasting and nonfinite data before plotting calibration metrics."""
    with pytest.raises(ValueError, match=match):
        plot(true_vals, pred_vals, std)


@pytest.mark.parametrize("std", [[], [0], [-1]])
def test_qq_gaussian_invalid_uncertainties(std: list[float]) -> None:
    """Calibration requires observations with positive standard deviations."""
    with pytest.raises(ValueError, match=r"non-empty|must be positive"):
        qq_gaussian(np.zeros(len(std)), np.ones(len(std)), std)
