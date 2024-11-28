from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from numpy._typing._array_like import NDArray
from numpy.typing import ArrayLike

from pymatviz.classify.curves import precision_recall_curve_plotly, roc_curve_plotly


@pytest.fixture
def binary_classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic binary classification data."""
    rng = np.random.default_rng(42)
    targets = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    probs = rng.random(size=len(targets))
    return targets, probs


TestInput: TypeAlias = np.ndarray | dict[str, Any] | pd.Series


@pytest.mark.parametrize(
    ("curve_func", "expected_x_label", "expected_y_label"),
    [
        (precision_recall_curve_plotly, "Recall", "Precision"),
        (roc_curve_plotly, "False Positive Rate", "True Positive Rate"),
    ],
)
@pytest.mark.parametrize(
    ("targets", "probs", "expected_traces"),
    [
        # Basic numpy arrays
        (
            np.array([0, 1, 1, 0]),
            np.array([0.1, 0.9, 0.8, 0.2]),
            2,  # 1 model trace + 1 baseline
        ),
        # Dict with multiple model predictions
        (
            np.array([0, 1, 1, 0]),
            {
                "Model A": np.array([0.1, 0.9, 0.8, 0.2]),
                "Model B": np.array([0.2, 0.8, 0.7, 0.3]),
            },
            3,  # 2 model traces + 1 baseline
        ),
    ],
)
def test_curve_plotting(
    curve_func: Callable[..., go.Figure],
    expected_x_label: str,
    expected_y_label: str,
    targets: ArrayLike,
    probs: TestInput,
    expected_traces: int,
) -> None:
    """Test curve plotting with various input types."""
    fig = curve_func(targets, probs)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces
    assert fig.layout.xaxis.title.text == expected_x_label
    assert fig.layout.yaxis.title.text == expected_y_label
    # Check value ranges
    assert all(0 <= y <= 1 for trace in fig.data for y in trace.y)
    assert all(0 <= x <= 1 for trace in fig.data for x in trace.x)


@pytest.mark.parametrize(
    "curve_func", [precision_recall_curve_plotly, roc_curve_plotly]
)
def test_df_input(
    binary_classification_data: tuple[np.ndarray, np.ndarray],
    curve_func: Callable[..., go.Figure],
) -> None:
    """Test curve plotting with DataFrame input."""
    targets, probs = binary_classification_data
    df_clf = pd.DataFrame({"target": targets, "prob": probs})

    fig = curve_func("target", "prob", df=df_clf)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2  # 1 model trace + 1 baseline


@pytest.mark.parametrize(
    "invalid_case",
    [
        pytest.param(([], []), id="empty-lists"),
        pytest.param((np.array([]), np.array([])), id="empty-arrays"),
        pytest.param(
            (np.array([1, 2, 3]), np.array([0.1, 0.2])), id="mismatched-lengths"
        ),
        pytest.param((np.array([1, 2, 3]), "not_an_array"), id="invalid-type"),
        pytest.param(
            (np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3])), id="non-binary-targets"
        ),
    ],
)
@pytest.mark.parametrize(
    "curve_func", [precision_recall_curve_plotly, roc_curve_plotly]
)
def test_invalid_inputs(
    invalid_case: tuple[list[Any], list[Any]]
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, Literal["not_an_array"]],
    curve_func: Callable[..., go.Figure],
) -> None:
    """Test error handling for various invalid inputs."""
    targets, probs = invalid_case
    with pytest.raises((ValueError, TypeError)):
        curve_func(targets, probs)


@pytest.mark.parametrize(
    "edge_case",
    [
        pytest.param(
            (np.ones(4), np.array([0.1, 0.9, 0.8, 0.2])), id="all-positive-targets"
        ),
        pytest.param(
            (np.zeros(4), np.array([0.1, 0.9, 0.8, 0.2])), id="all-negative-targets"
        ),
        pytest.param(
            (np.array([0, 1, 0, 1]), np.array([0.0, 1.0, 0.0, 1.0])),
            id="perfect-predictions",
        ),
        pytest.param(
            (np.array([0, 1, 0, 1]), np.array([1.0, 0.0, 1.0, 0.0])),
            id="worst-predictions",
        ),
        pytest.param(
            (np.array([0, 1, 0, 1]), np.full(4, 0.5)), id="constant-predictions"
        ),
    ],
)
def test_edge_cases(
    edge_case: tuple[NDArray[np.float64], NDArray[Any]]
    | tuple[NDArray[Any], NDArray[Any]],
) -> None:
    """Test edge cases for both curve types."""
    targets, probs = edge_case

    # PR curves should work for all cases
    fig_pr = precision_recall_curve_plotly(targets, probs)
    assert isinstance(fig_pr, go.Figure)

    # ROC curves should fail for single-class targets
    if len(np.unique(targets)) == 1:
        with pytest.raises(ValueError, match=".*ROC AUC score is not defined.*"):
            roc_curve_plotly(targets, probs)
    else:
        fig_roc = roc_curve_plotly(targets, probs)
        assert isinstance(fig_roc, go.Figure)


def test_custom_styling() -> None:
    """Test custom styling options."""
    targets = np.array([0, 1, 0, 1])
    probs = {
        "Model A": {
            "probs_positive": np.array([0.1, 0.9, 0.2, 0.8]),
            "line": {"color": "red"},
        },
        "Model B": {
            "probs_positive": np.array([0.2, 0.8, 0.3, 0.7]),
            "line": {"dash": "dash"},
        },
    }

    fig = precision_recall_curve_plotly(targets, probs)
    assert fig.data[0].line.color == "red"
    assert fig.data[1].line.dash == "dash"


def test_large_dataset() -> None:
    """Test performance with a larger dataset."""
    rng = np.random.default_rng(42)
    n_samples = 10_000
    targets = rng.binomial(n=1, p=0.5, size=n_samples)
    probs = rng.random(size=n_samples)

    for curve_func in [precision_recall_curve_plotly, roc_curve_plotly]:
        fig = curve_func(targets, probs)
        assert isinstance(fig, go.Figure)
