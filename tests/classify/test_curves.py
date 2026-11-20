from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm
from numpy.typing import ArrayLike

from pymatviz.classify.curves import precision_recall_curve, roc_curve
from pymatviz.widgets.scatter_plot import ScatterPlotWidget


@pytest.fixture
def binary_classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic binary classification data."""
    rng = np.random.default_rng(seed=0)
    targets = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 1])
    probs = rng.random(size=len(targets))
    return targets, probs


type TestInput = np.ndarray | dict[str, Any] | pd.Series


@pytest.mark.parametrize(
    ("curve_func", "expected_x_label", "expected_y_label"),
    [
        (precision_recall_curve, "Recall", "Precision"),
        (roc_curve, "False Positive Rate", "True Positive Rate"),
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
    curve_func: Callable[..., ScatterPlotWidget],
    expected_x_label: str,
    expected_y_label: str,
    targets: ArrayLike,
    probs: TestInput,
    expected_traces: int,
) -> None:
    """Test curve plotting with various input types."""
    # no_skill=False for consistent trace counting
    widget = curve_func(targets, probs, no_skill=False)

    assert len(widget.series) == expected_traces
    assert widget.x_axis == {"label": expected_x_label}
    assert widget.y_axis == {"label": expected_y_label}
    for srs in widget.series:
        assert all(0 <= val <= 1 for val in (*srs["x"], *srs["y"]))
        assert len(srs["metadata"]) == len(srs["x"])
        assert {*srs["metadata"][-1]} >= {"threshold"}
    # series are sorted by AUC/F1 (descending)
    scores = [float(srs["label"].split("=")[-1]) for srs in widget.series]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("curve_func", [precision_recall_curve, roc_curve])
def test_df_input(
    binary_classification_data: tuple[np.ndarray, np.ndarray],
    curve_func: Callable[..., ScatterPlotWidget],
) -> None:
    """Test curve plotting with DataFrame input."""
    targets, probs = binary_classification_data
    df_clf = pd.DataFrame({"target": targets, "prob": probs})

    # no_skill=False for consistent trace counting
    widget = curve_func("target", "prob", df=df_clf, no_skill=False)
    assert widget.series == curve_func(targets, probs, no_skill=False).series


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
@pytest.mark.parametrize("curve_func", [precision_recall_curve, roc_curve])
def test_invalid_inputs(
    invalid_case: tuple[list[Any], list[Any]]
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, Literal["not_an_array"]],
    curve_func: Callable[..., ScatterPlotWidget],
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
    """Single-class targets and degenerate probabilities don't raise."""
    targets, probs = edge_case
    precision_recall_curve(targets, probs, no_skill=False)
    roc_curve(targets, probs, no_skill=False)


def test_custom_styling() -> None:
    """Test custom styling options."""
    targets = np.array([0, 1, 0, 1])
    probs = {
        "Model A": {
            "probs_positive": np.array([0.1, 0.9, 0.2, 0.8]),
            "color": "red",
        },
        "Model B": {
            "probs_positive": np.array([0.2, 0.8, 0.3, 0.7]),
            "line_style": {"dash": "4 4"},
        },
    }

    widget = precision_recall_curve(targets, probs, no_skill=False)
    assert widget.series[0]["color"] == "red"
    assert widget.series[0]["line_style"] == {"width": 2, "dash": ""}
    assert widget.series[1]["line_style"] == {"dash": "4 4"}


@pytest.mark.parametrize("targets", [[0, 1, 0, 1], [0, 0, 0, 1], [0, 1, 1, 1]])
@pytest.mark.parametrize("no_skill", [None, False, {"style": {"color": "red"}}])
def test_no_skill_line(
    targets: list[int], no_skill: dict[str, Any] | Literal[False] | None
) -> None:
    """No-skill baselines are ref lines: a ROC diagonal, a PR line at prevalence."""
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    widget_pr = precision_recall_curve(targets, probs, no_skill=no_skill)
    widget_roc = roc_curve(targets, probs, no_skill=no_skill)
    assert len(widget_pr.series) == len(widget_roc.series) == 1
    if no_skill is False:
        assert widget_pr.ref_lines == widget_roc.ref_lines == []
        return
    (pr_line,) = widget_pr.ref_lines
    (roc_line,) = widget_roc.ref_lines
    assert (pr_line["type"], pr_line["y"]) == ("horizontal", np.mean(targets))
    assert pr_line["annotation"]["text"] == "No skill"
    assert (roc_line["p1"], roc_line["p2"]) == ([0, 0], [1, 1])
    color = "red" if no_skill else "gray"
    assert pr_line["style"]["color"] == roc_line["style"]["color"] == color


@pytest.mark.parametrize(
    ("probs", "expected_baselines"),
    [
        ([np.nan, 0.9, 0.8, 0.2], {"No skill": 1.0}),
        (
            {"A": [0.1, 0.9, 0.8, 0.2], "B": [np.nan, 0.8, 0.3, 0.9]},
            {"No skill (A)": 2 / 3, "No skill (B)": 1.0},
        ),
        (
            {"A": [np.nan, 0.9, 0.8, 0.2], "B": [np.nan, 0.8, 0.3, 0.9]},
            {"No skill": 1.0},
        ),
    ],
)
def test_no_skill_line_missing_values(
    probs: list[float] | dict[str, list[float]], expected_baselines: dict[str, float]
) -> None:
    """Match baselines to each curve's valid rows and collapse equal prevalences."""
    targets = [0, 1, np.nan, 1]
    widget = precision_recall_curve(targets, probs)
    assert {
        line["annotation"]["text"]: line["y"] for line in widget.ref_lines
    } == expected_baselines
    assert {srs["y"][0] for srs in widget.series} == set(expected_baselines.values())


def test_curve_threshold_metadata() -> None:
    """Point metadata holds sklearn's thresholds (inf/missing as None) and F1."""
    targets, probs = [0, 1, 0, 1, 1], [0.1, 0.9, 0.2, 0.8, 0.4]
    roc_meta = roc_curve(targets, probs, no_skill=False).series[0]["metadata"]
    _, _, roc_thresholds = skm.roc_curve(targets, probs)
    assert roc_meta[0]["threshold"] is None  # sklearn's inf first threshold
    assert [meta["threshold"] for meta in roc_meta[1:]] == roc_thresholds[1:].tolist()

    pr_series = precision_recall_curve(targets, probs, no_skill=False).series[0]
    prec, recall, pr_thresholds = skm.precision_recall_curve(targets, probs)
    pr_meta = pr_series["metadata"]
    assert [meta["threshold"] for meta in pr_meta] == [*pr_thresholds.tolist(), None]
    expected_f1 = 2 * prec * recall / (prec + recall)
    np.testing.assert_allclose(
        [meta["f1"] for meta in pr_meta], expected_f1, rtol=0, atol=0
    )


@pytest.mark.parametrize("curve_func", [precision_recall_curve, roc_curve])
def test_dict_with_dataframe_raises_error(
    curve_func: Callable[..., ScatterPlotWidget],
) -> None:
    """Test that passing a dict with a DataFrame raises a clear TypeError."""
    df_clf = pd.DataFrame({"target": [0, 1, 0, 1], "probs": [0.1, 0.9, 0.2, 0.8]})
    probs_dict = {"Model A": np.array([0.1, 0.9, 0.2, 0.8])}

    with pytest.raises(
        TypeError,
        match="when passing a DataFrame, probs_positive must be a column name",
    ):
        curve_func("target", probs_dict, df=df_clf)


@pytest.mark.parametrize("curve_func", [precision_recall_curve, roc_curve])
def test_curve_does_not_mutate_input_dicts(
    binary_classification_data: tuple[np.ndarray, np.ndarray],
    curve_func: Callable[..., ScatterPlotWidget],
) -> None:
    """dict-of-dicts predictions must not have probs_positive popped off the caller."""
    targets, probs = binary_classification_data
    model_kwargs = {"probs_positive": probs, "color": "red"}

    curve_func(targets, {"Model A": model_kwargs}, no_skill=False)

    assert "probs_positive" in model_kwargs  # caller dict untouched
    assert model_kwargs["color"] == "red"
