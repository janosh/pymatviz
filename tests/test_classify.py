"""Tests for pymatviz.classify: confusion matrices, ROC and precision-recall curves."""

# `.values` below is HeatmapMatrixWidget's trait, not a pandas attribute
# ruff: noqa: PD011

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import pytest
import sklearn.metrics as skm

from pymatviz.classify import confusion_matrix, precision_recall_curve, roc_curve
from pymatviz.widgets.heatmap_matrix import HeatmapMatrixWidget


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from numpy.typing import ArrayLike

    from pymatviz.widgets.scatter_plot import ScatterPlotWidget


@pytest.fixture
def sample_conf_mat() -> np.ndarray:
    """2x2 confusion matrix for binary classification."""
    return np.array([[50, 10], [5, 35]])  # TN, FP, FN, TP


@pytest.mark.parametrize(
    ("normalize", "float_fmt", "expected_fmt"),
    [(True, "", ".1%"), (False, "", ".0f"), (True, ".2f", ".2f")],
)
def test_confusion_matrix_basic(
    sample_conf_mat: np.ndarray, normalize: bool, float_fmt: str, expected_fmt: str
) -> None:
    """Rows are true classes, columns predicted, values optionally normalized."""
    widget = confusion_matrix(
        sample_conf_mat,
        x_labels=("Negative", "Positive"),
        normalize=normalize,
        float_fmt=float_fmt,
    )
    assert isinstance(widget, HeatmapMatrixWidget)
    assert [item["label"] for item in widget.x_items] == ["Negative", "Positive"]
    assert widget.y_items == widget.x_items
    expected = sample_conf_mat / (sample_conf_mat.sum() if normalize else 1)
    np.testing.assert_array_equal(widget.values, expected)
    assert widget.show_values == expected_fmt
    assert widget.color_scale == "interpolateBlues"
    assert widget.color_bar is None
    assert widget.y_axis == {"label": "Actual"}
    assert widget.x_axis["label"].startswith("Predicted (Acc=")


@pytest.mark.parametrize(
    ("metrics", "expected_title"),
    [
        (("Acc", "MCC"), "Predicted (Acc=85.0%, MCC=69.8%)"),
        ({"Acc": ".2f", "MCC": None}, "Predicted (Acc=0.85, MCC=69.8%)"),
        (("Prec", "Rec", "F1"), "Predicted (Prec=77.8%, Rec=87.5%, F1=82.4%)"),
        ((), "Predicted"),
    ],
)
def test_confusion_matrix_metrics(
    sample_conf_mat: np.ndarray,
    metrics: Mapping[str, str | None] | Sequence[str],
    expected_title: str,
) -> None:
    """Requested metrics with their formats are appended to the x-axis title."""
    widget = confusion_matrix(sample_conf_mat, x_labels=("N", "P"), metrics=metrics)
    assert widget.x_axis == {"label": expected_title}
    # rows in another class order are aligned for metrics, not for display
    flipped = confusion_matrix(
        sample_conf_mat[::-1], x_labels=("N", "P"), y_labels=("P", "N"), metrics=metrics
    )
    assert flipped.x_axis == {"label": expected_title}
    assert flipped.values == widget.values[::-1]
    # disjoint axis names (no shared class) are aligned by position
    named = confusion_matrix(
        sample_conf_mat,
        x_labels=("Pred N", "Pred P"),
        y_labels=("True N", "True P"),
        metrics=metrics,
    )
    assert named.x_axis == {"label": expected_title}
    assert named.values == widget.values
    # int64 counts whose MCC denominator product overflows (2**64)
    big = confusion_matrix(
        np.eye(2, dtype=np.int64) * 65_536, x_labels=("N", "P"), metrics=("MCC",)
    )
    assert big.x_axis == {"label": "Predicted (MCC=100.0%)"}


def test_confusion_matrix_multi_class() -> None:
    """Only accuracy is defined for multi-class matrices."""
    conf_mat = np.array([[40, 5, 5], [4, 35, 6], [6, 4, 45]])
    labels = ("A", "B", "C")
    widget = confusion_matrix(conf_mat, x_labels=labels, normalize=False)
    assert widget.x_axis == {"label": "Predicted (Acc=80.0%)"}
    with pytest.raises(ValueError, match=r"Unknown metrics \['Prec'\]"):
        confusion_matrix(conf_mat, x_labels=labels, metrics=("Prec",))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({}, "Must provide either conf_mat or both"),
        ({"conf_mat": [[1]], "y_true": [0], "y_pred": [0]}, "Must provide either"),
        ({"conf_mat": [[1]], "y_true": [0]}, "Must provide either conf_mat or both"),
        ({"y_pred": [0]}, "Must provide either conf_mat or both"),
        ({"conf_mat": [[1, 2], [3, 4]]}, "Must provide x_labels"),
        ({"conf_mat": [[1, 2], [3, 4]], "x_labels": ("A",)}, "doesn't match"),
        (
            {"conf_mat": [[1, 2], [3, 4]], "x_labels": ("A", "B"), "metrics": ["x"]},
            r"Unknown metrics \['x'\]",
        ),
        ({"y_true": [0, 1], "y_pred": [1, 1], "x_labels": ("0", "2")}, "must name"),
        (
            {"conf_mat": np.zeros((2, 2)), "x_labels": ("A", "B")},
            "must contain samples",
        ),
        (
            {"y_true": [0, 1], "y_pred": [1, 1], "y_labels": ("1", "0")},
            "y_labels must equal",
        ),
        (
            {"conf_mat": np.eye(2), "x_labels": ("A", "B"), "y_labels": ("A", "C")},
            "metrics need x_labels and y_labels to name the same classes",
        ),
        (
            {"y_true": [0, "a"], "y_pred": ["a", 0]},
            r"mutually sortable \(one type\), got types \['int', 'str'\]",
        ),
    ],
)
def test_confusion_matrix_errors(kwargs: dict, match: str) -> None:
    """Invalid inputs raise ValueError with a clear message."""
    with pytest.raises(ValueError, match=match):
        confusion_matrix(**kwargs)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (["N"] * 3 + ["P"] * 3, ["N"] * 3 + ["P"] * 3, [[0.5, 0], [0, 0.5]]),
        (["N"] * 3 + ["P"] * 3, ["P"] * 3 + ["N"] * 3, [[0, 0.5], [0.5, 0]]),
        (["N"] * 3 + ["P"] * 3, ["N"] * 6, [[0.5, 0], [0.5, 0]]),
        (["N"] * 99 + ["P"], ["N"] * 99 + ["P"], [[0.99, 0], [0, 0.01]]),
    ],
)
def test_confusion_matrix_from_labels(
    y_true: list[str], y_pred: list[str], expected: list[list[float]]
) -> None:
    """Raw labels give a normalized matrix with true rows and predicted columns."""
    widget = confusion_matrix(y_true=y_true, y_pred=y_pred)
    assert widget.values == expected
    assert [item["key"] for item in widget.x_items] == ["N", "P"]


def test_confusion_matrix_label_order() -> None:
    """Explicit x_labels reorder rows and columns of the matrix computed from labels."""
    y_true = ["P", "N", "P", "N", "N"]
    y_pred = ["P", "N", "N", "N", "P"]
    default = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=False)
    reordered = confusion_matrix(
        y_true=y_true, y_pred=y_pred, x_labels=("P", "N"), normalize=False
    )
    assert default.values == [[2, 1], [1, 1]]  # rows/cols: N, P
    assert reordered.values == [[1, 1], [1, 2]]  # rows/cols: P, N
    assert [item["label"] for item in reordered.x_items] == ["P", "N"]
    # integer classes match string labels
    ints = confusion_matrix(y_true=[0, 1, 1], y_pred=[1, 1, 0], x_labels=("1", "0"))
    assert [item["label"] for item in ints.x_items] == ["1", "0"]


def test_confusion_matrix_kwargs(sample_conf_mat: np.ndarray) -> None:
    """Extra kwargs go to the widget, overriding defaults."""
    widget = confusion_matrix(
        sample_conf_mat,
        x_labels=("N", "P"),
        color_scale="interpolateReds",
        tile_size="80px",
    )
    assert (widget.color_scale, widget.tile_size) == ("interpolateReds", "80px")
    labels = np.array(["N", "P"])  # array labels must not hit truthiness checks
    from_arrays = confusion_matrix(sample_conf_mat, x_labels=labels, y_labels=labels)
    assert [item["key"] for item in from_arrays.y_items] == ["N", "P"]


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
    ("targets", "probs", "match"),
    [
        ([], [], "has no targets of class"),
        (np.array([0, 1, 1]), np.array([0.1, 0.2]), r"shape \(2,\), expected one"),
        ([0, 1], [[0.1, 0.2], [0.3, 0.4]], r"shape \(2, 2\), expected one"),
        (np.array([0, 1, 1]), "not_an_array", "could not convert string to float"),
        (np.array([0, 1, 2]), np.array([0.1, 0.2, 0.3]), "targets must be binary"),
    ],
    ids=["empty", "mismatched-lengths", "2d-probs", "invalid-type", "non-binary"],
)
@pytest.mark.parametrize("curve_func", [precision_recall_curve, roc_curve])
def test_invalid_inputs(
    targets: ArrayLike,
    probs: ArrayLike,
    match: str,
    curve_func: Callable[..., ScatterPlotWidget],
) -> None:
    """Invalid targets or probabilities raise errors naming the problem."""
    with pytest.raises(ValueError, match=match):
        curve_func(targets, probs)


@pytest.mark.parametrize(
    ("curve_func", "targets", "missing"),
    [
        (roc_curve, np.ones(4), "0"),
        (roc_curve, [0, 0, np.nan, 0], "1"),
        (roc_curve, [-1, -1, -1, -1], "1"),
        (precision_recall_curve, np.zeros(4), "1"),
    ],
)
def test_single_class_targets_raise(
    curve_func: Callable[..., ScatterPlotWidget], targets: ArrayLike, missing: str
) -> None:
    """Curves undefined for single-class targets raise naming the classifier."""
    probs = {"Model A": [0.1, 0.9, 0.8, 0.2]}
    with pytest.raises(
        ValueError, match=rf"'Model A' has no targets of class {missing} \("
    ):
        curve_func(targets, probs)


def test_pr_curve_all_positive_targets() -> None:
    """All-positive targets give a defined PR curve with precision 1."""
    widget = precision_recall_curve(np.ones(4), [0.1, 0.9, 0.8, 0.2], no_skill=False)
    assert set(widget.series[0]["y"]) == {1.0}


@pytest.mark.parametrize(
    "edge_case",
    [
        (np.array([0, 1, 0, 1]), np.array([0.0, 1.0, 0.0, 1.0])),
        (np.array([0, 1, 0, 1]), np.array([1.0, 0.0, 1.0, 0.0])),
        (np.array([0, 1, 0, 1]), np.full(4, 0.5)),
    ],
)
def test_edge_cases(edge_case: tuple[Sequence[float], Sequence[float]]) -> None:
    """Degenerate probabilities don't raise."""
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
    # a partial axis option merges over the default axis label instead of replacing it
    widget = roc_curve(targets, probs, x_axis={"scale_type": "log"})
    assert widget.x_axis == {"label": "False Positive Rate", "scale_type": "log"}


@pytest.mark.parametrize("targets", [[0, 1, 0, 1], [0, 0, 0, 1], [0, 1, 1, 1]])
@pytest.mark.parametrize(
    "no_skill",
    [
        None,
        False,
        {"style": {"color": "red"}},
        {"style": {"width": 3}, "annotation": {"position": "start"}},
    ],
)
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
    assert (roc_line["p1"], roc_line["p2"]) == ([0, 0], [1, 1])
    # overrides merge one level deep into the default style and annotation
    overrides = no_skill or {}
    for line in (pr_line, roc_line):
        assert line["annotation"]["text"] == "No skill"
        assert line["annotation"]["color"] == "gray"
        expected_style = {"color": "gray", "dash": "6 4"} | overrides.get("style", {})
        assert line["style"] == expected_style
        if "annotation" in overrides:
            assert line["annotation"]["position"] == "start"


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
    baselines = {line["annotation"]["text"]: line["y"] for line in widget.ref_lines}
    assert baselines == expected_baselines
    assert {srs["y"][0] for srs in widget.series} == set(expected_baselines.values())


def test_curve_threshold_metadata() -> None:
    """Point metadata holds sklearn's thresholds (inf/missing as None) and F1."""
    targets, probs = [0, 1, 0, 1, 1], [0.1, 0.9, 0.2, 0.8, 0.4]
    roc_meta = roc_curve(targets, probs, no_skill=False).series[0]["metadata"]
    _, _, roc_thresholds = skm.roc_curve(targets, probs)
    assert roc_meta[0]["threshold"] is None  # sklearn's inf first threshold
    assert [meta["threshold"] for meta in roc_meta[1:]] == roc_thresholds[1:].tolist()

    pr_series = precision_recall_curve(targets, probs, no_skill=False).series[0]
    # collinear points that don't change the plotted curve are dropped
    prec, recall, pr_thresholds = skm.precision_recall_curve(
        targets, probs, drop_intermediate=True
    )
    assert len(pr_thresholds) < len(skm.precision_recall_curve(targets, probs)[2])
    pr_meta = pr_series["metadata"]
    assert [meta["threshold"] for meta in pr_meta] == [*pr_thresholds.tolist(), None]
    expected_f1 = 2 * prec * recall / (prec + recall)
    np.testing.assert_allclose(
        [meta["f1"] for meta in pr_meta], expected_f1, rtol=0, atol=0
    )


@pytest.mark.parametrize("curve_func", [precision_recall_curve, roc_curve])
@pytest.mark.parametrize(
    "probs",
    [{"Model A": np.array([0.1, 0.9, 0.2, 0.8])}, np.array([0.1, 0.9, 0.2, 0.8])],
    ids=["dict", "array"],
)
def test_non_column_probs_with_dataframe_raise(
    curve_func: Callable[..., ScatterPlotWidget], probs: Any
) -> None:
    """With a DataFrame, probs_positive must name a column (arrays could misalign
    with the rows dropped for NaN).
    """
    df_clf = pd.DataFrame({"target": [0, 1, 0, 1], "probs": [0.1, 0.9, 0.2, 0.8]})

    with pytest.raises(
        TypeError,
        match="when passing a DataFrame, probs_positive must be a column name",
    ):
        curve_func("target", probs, df=df_clf)


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


def test_curve_labels_match_sklearn() -> None:
    """ROC AUC and PR F1 (positive for probs >= 0.5) in labels match sklearn."""
    rng = np.random.default_rng(seed=0)
    targets = rng.integers(0, 2, size=200)
    probs = np.clip(0.3 * targets + rng.random(size=200) * 0.7, 0, 1)
    probs[:20] = 0.5  # ties at the threshold count as positive predictions
    roc_label = roc_curve(targets, {"m": probs}).series[0]["label"]
    assert roc_label == f"m · AUC={skm.roc_auc_score(targets, probs):.2f}"
    # -1 is also taken as the negative class (sklearn's f1_score rejects it)
    pm_targets, pm_probs = [-1, 1, -1, 1, np.nan], [0.1, 0.9, 0.6, 0.8, 0.3]
    assert roc_curve(pm_targets, pm_probs).series[0]["label"] == "AUC=1.00"
    pm_pr = precision_recall_curve(pm_targets, pm_probs)
    assert pm_pr.series[0]["label"] == "F1=0.80"
    assert pm_pr.ref_lines[0]["y"] == 0.5  # prevalence of the +1 class
    pr_label = precision_recall_curve(targets, {"m": probs}).series[0]["label"]
    expected_f1 = skm.f1_score(targets, probs >= 0.5)
    assert pr_label == f"m · F1={expected_f1:.2f}"
    # with np.round, 0.5 -> 0 would predict all negative: F1=0.00 instead of 0.80
    half_label = precision_recall_curve([0, 1, 1], np.full(3, 0.5)).series[0]["label"]
    assert half_label == f"F1={skm.f1_score([0, 1, 1], [1, 1, 1]):.2f}" == "F1=0.80"


@pytest.mark.parametrize("curve_func", [precision_recall_curve, roc_curve])
@pytest.mark.parametrize(
    "probs", ["probs", 0, ["probs", 0]], ids=["str", "int", "list"]
)
def test_dataframe_column_labels(
    curve_func: Callable[..., ScatterPlotWidget], probs: Any
) -> None:
    """With a DataFrame, probs_positive names one column (str or int label) or a list
    of them, one curve each.
    """
    df_clf = pd.DataFrame({"target": [0, 1, 0, 1], "probs": [0.1, 0.9, 0.2, 0.8]})
    df_clf[0] = [0.3, 0.7, 0.4, 0.6]
    widget = curve_func("target", probs, df=df_clf, no_skill=False)
    assert len(widget.series) == (2 if isinstance(probs, list) else 1)
