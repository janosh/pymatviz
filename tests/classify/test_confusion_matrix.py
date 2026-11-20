from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pymatviz.classify.confusion_matrix import confusion_matrix
from pymatviz.widgets.heatmap_matrix import HeatmapMatrixWidget


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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
