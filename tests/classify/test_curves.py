from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from numpy.typing import ArrayLike

from pymatviz.classify.curves import precision_recall_curve_plotly, roc_curve_plotly


@pytest.fixture
def binary_classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic binary classification data."""
    rng = np.random.default_rng(seed=0)
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
        (  # Basic numpy arrays
            np.array([0, 1, 1, 0]),
            np.array([0.1, 0.9, 0.8, 0.2]),
            1,  # 1 model trace
        ),
        (  # Dict with multiple model predictions
            np.array([0, 1, 1, 0]),
            {
                "Model A": np.array([0.1, 0.9, 0.8, 0.2]),
                "Model B": np.array([0.2, 0.8, 0.7, 0.3]),
            },
            2,  # 2 model traces
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
    # no_skill=False for consistent trace counting
    fig = curve_func(targets, probs, no_skill=False)

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

    # no_skill=False for consistent trace counting
    fig = curve_func("target", "prob", df=df_clf, no_skill=False)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1  # 1 model trace


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
        (np.ones(4), np.array([0.1, 0.9, 0.8, 0.2])),
        (np.zeros(4), np.array([0.1, 0.9, 0.8, 0.2])),
        (np.array([0, 1, 0, 1]), np.array([0.0, 1.0, 0.0, 1.0])),
        (np.array([0, 1, 0, 1]), np.array([1.0, 0.0, 1.0, 0.0])),
        (np.array([0, 1, 0, 1]), np.full(4, 0.5)),
    ],
)
def test_edge_cases(edge_case: tuple[Sequence[float], Sequence[float]]) -> None:
    """Test edge cases for both curve types."""
    targets, probs = edge_case

    # PR curves should work for all cases
    fig_pr = precision_recall_curve_plotly(targets, probs, no_skill=False)
    assert isinstance(fig_pr, go.Figure)

    # Pass no_skill=False to disable the no-skill line to avoid annotation errors
    fig_roc = roc_curve_plotly(targets, probs, no_skill=False)
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

    # Pass no_skill=False to disable the no-skill line to avoid annotation errors
    fig = precision_recall_curve_plotly(targets, probs, no_skill=False)
    assert fig.data[0].line.color == "red"
    assert fig.data[1].line.dash == "dash"


def test_large_dataset() -> None:
    """Test performance with a larger dataset."""
    rng = np.random.default_rng(seed=0)
    n_samples = 10_000
    targets = rng.binomial(n=1, p=0.5, size=n_samples)
    probs = rng.random(size=n_samples)

    for curve_func in [precision_recall_curve_plotly, roc_curve_plotly]:
        # Pass no_skill=False to disable the no-skill line to avoid annotation errors
        fig = curve_func(targets, probs, no_skill=False)
        assert isinstance(fig, go.Figure)


def test_no_skill_line() -> None:
    """Test that the no-skill line is added correctly."""
    targets = np.array([0, 1, 0, 1])
    probs = np.array([0.1, 0.9, 0.2, 0.8])

    # Test with no_skill=True (default)
    fig_pr = precision_recall_curve_plotly(targets, probs)
    assert isinstance(fig_pr, go.Figure)
    # PR curve adds the no-skill line as a shape, not a trace
    assert len(fig_pr.data) == 1  # 1 model trace
    assert len(fig_pr.layout.shapes) == 1  # 1 no-skill line shape
    assert len(fig_pr.layout.annotations) == 1  # 1 no-skill annotation

    fig_roc = roc_curve_plotly(targets, probs)
    assert isinstance(fig_roc, go.Figure)
    # ROC curve adds the no-skill line as a trace
    assert len(fig_roc.data) == 2  # 1 model trace + 1 no-skill line

    # Test with no_skill=False
    fig_pr_no_baseline = precision_recall_curve_plotly(targets, probs, no_skill=False)
    assert isinstance(fig_pr_no_baseline, go.Figure)
    assert len(fig_pr_no_baseline.data) == 1  # 1 model trace, no baseline
    assert len(fig_pr_no_baseline.layout.shapes) == 0  # No shapes

    fig_roc_no_baseline = roc_curve_plotly(targets, probs, no_skill=False)
    assert isinstance(fig_roc_no_baseline, go.Figure)
    assert len(fig_roc_no_baseline.data) == 1  # 1 model trace, no baseline

    # Test with custom no_skill options for PR curve
    custom_no_skill_pr = {"line": {"color": "red"}}
    fig_pr_custom = precision_recall_curve_plotly(
        targets, probs, no_skill=custom_no_skill_pr
    )
    assert isinstance(fig_pr_custom, go.Figure)
    assert len(fig_pr_custom.data) == 1  # 1 model trace
    assert len(fig_pr_custom.layout.shapes) == 1  # 1 no-skill line shape
    assert fig_pr_custom.layout.shapes[0].line.color == "red"  # Custom color

    # Test with custom no_skill options for ROC curve
    # For ROC curve, we need to pass the line color differently
    custom_no_skill_roc = {"line": {"color": "red"}}
    fig_roc_custom = roc_curve_plotly(targets, probs, no_skill=custom_no_skill_roc)
    assert isinstance(fig_roc_custom, go.Figure)
    assert len(fig_roc_custom.data) == 2  # 1 model trace + 1 no-skill line
    assert fig_roc_custom.data[1].line.color == "red"  # Custom color


@pytest.mark.parametrize(
    "curve_func", [precision_recall_curve_plotly, roc_curve_plotly]
)
def test_dict_with_dataframe_raises_error(
    curve_func: Callable[..., go.Figure],
) -> None:
    """Test that passing a dict with a DataFrame raises a clear TypeError."""
    df_clf = pd.DataFrame({"target": [0, 1, 0, 1], "probs": [0.1, 0.9, 0.2, 0.8]})
    probs_dict = {"Model A": np.array([0.1, 0.9, 0.2, 0.8])}

    with pytest.raises(
        TypeError,
        match="when passing a DataFrame, probs_positive must be a column name",
    ):
        curve_func("target", probs_dict, df=df_clf)
