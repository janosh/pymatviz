from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import pytest

from pymatviz.classify.confusion_matrix import confusion_matrix


if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.fixture
def sample_conf_mat() -> np.ndarray:
    """2x2 confusion matrix for binary classification."""
    return np.array([[50, 10], [5, 35]])  # TN, FP, FN, TP


@pytest.fixture
def multi_class_conf_mat() -> np.ndarray:
    """3x3 confusion matrix for multi-class classification."""
    return np.array([[40, 5, 5], [4, 35, 6], [6, 4, 45]])


def test_confusion_matrix_basic(sample_conf_mat: np.ndarray) -> None:
    """Test basic functionality with pre-computed confusion matrix."""
    fig = confusion_matrix(
        sample_conf_mat,
        x_labels=("Negative", "Positive"),
        y_labels=("Negative", "Positive"),
    )
    assert isinstance(fig, go.Figure)
    assert fig.layout.xaxis.constrain == "domain"
    assert fig.layout.yaxis.constrain == "domain"
    assert fig.layout.yaxis.scaleanchor == "x"
    assert fig.layout.font.size == 18

    # Test with raw labels
    y_true = np.array([0, 0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1])
    fig = confusion_matrix(y_true=y_true, y_pred=y_pred)
    assert isinstance(fig, go.Figure)


@pytest.mark.parametrize(
    ("normalize", "colorscale", "float_fmt"),
    [
        (True, "blues", ".1%"),  # default settings
        (False, "RdBu", ".2f"),  # raw counts, different colorscale
        (True, "Viridis", ".2%"),  # normalized, different colorscale
        (False, "YlOrRd", ".0f"),  # raw counts, integer format
    ],
)
def test_confusion_matrix_display_options(
    sample_conf_mat: np.ndarray,
    normalize: bool,
    colorscale: str,
    float_fmt: str,
) -> None:
    """Test various display options."""
    fig = confusion_matrix(
        sample_conf_mat,
        x_labels=("Negative", "Positive"),
        normalize=normalize,
        colorscale=colorscale,
        float_fmt=float_fmt,
    )
    assert isinstance(fig, go.Figure)

    # Check heatmap trace properties
    heatmap = fig.data[0]
    first_color = heatmap.colorscale[0][1].lower()
    expected_first_colors = {
        "blues": "rgb(247,251,255)".lower(),
        "rdbu": "rgb(103,0,31)".lower(),
        "viridis": "#440154".lower(),
        "ylorrd": "rgb(255,255,204)".lower(),
    }
    assert first_color == expected_first_colors[colorscale.lower()]

    # Check annotations format
    annotations = [anno.text for anno in fig.layout.annotations if anno.text]
    if normalize:
        assert all("%" in anno for anno in annotations if anno != "")
    else:
        assert not any("%" in anno for anno in annotations if anno != "")


@pytest.mark.parametrize(
    ("metrics", "metrics_kwargs", "expected_metrics"),
    [
        (("Acc", "MCC"), None, {"Acc", "MCC"}),  # default metrics
        ({"Acc": ".2%", "MCC": ".3f"}, None, {"Acc", "MCC"}),  # custom formats
        (("Prec", "Rec", "F1"), dict(y=1.1), {"Prec", "Rec", "F1"}),  # multiple metrics
        ([], None, set()),  # no metrics
        ((), None, {}),  # defaults to accuracy only
    ],
)
def test_confusion_matrix_metrics(
    sample_conf_mat: np.ndarray,
    metrics: dict[str, str | None] | Sequence[str],
    metrics_kwargs: dict[str, Any] | None,
    expected_metrics: set[str],
) -> None:
    """Test metrics display options."""
    fig = confusion_matrix(
        sample_conf_mat,
        x_labels=("Negative", "Positive"),
        metrics=metrics,
        metrics_kwargs=metrics_kwargs,
    )

    metric_annotations = [
        anno.text
        for anno in fig.layout.annotations
        if any(
            f"{metric}=" in anno.text for metric in ("Acc", "MCC", "Prec", "Rec", "F1")
        )
    ]

    if not expected_metrics:
        assert metric_annotations == []
    else:
        # There should be exactly one annotation containing all metrics
        assert len(metric_annotations) == 1
        annotation = metric_annotations[0]
        # Check that all expected metrics are present in the annotation
        for metric in expected_metrics:
            assert f"{metric}=" in annotation


def test_confusion_matrix_multi_class(multi_class_conf_mat: np.ndarray) -> None:
    """Test handling of multi-class confusion matrices."""
    labels = ("Class A", "Class B", "Class C")
    fig = confusion_matrix(
        multi_class_conf_mat,
        x_labels=labels,
        y_labels=labels,
        metrics={"Acc"},  # only Acc available for multi-class
    )

    # Check that only accuracy is shown for multi-class
    anno_texts = [anno.text for anno in fig.layout.annotations]
    assert len(anno_texts) == 10
    assert any("Acc=" in anno_text for anno_text in anno_texts), f"{anno_texts=}"

    with pytest.raises(ValueError, match="Unknown metric='Prec'"):
        confusion_matrix(
            multi_class_conf_mat, x_labels=labels, y_labels=labels, metrics={"Prec"}
        )


def test_confusion_matrix_hover_text(sample_conf_mat: np.ndarray) -> None:
    """Test hover text formatting."""
    fig = confusion_matrix(
        sample_conf_mat, x_labels=("Negative", "Positive"), normalize=True
    )

    hover_text = fig.data[0].text.flatten()
    for text in hover_text:
        if not text:  # skip empty hover texts
            continue
        assert "True:" in text
        assert "Pred:" in text
        assert "Count:" in text
        assert "Percent:" in text
        assert "Row %" in text
        assert "Col %" in text


def test_confusion_matrix_error_cases() -> None:
    """Test error handling."""
    # Test missing input
    with pytest.raises(ValueError, match="Must provide either conf_mat or both"):
        confusion_matrix()

    # Test missing labels for pre-computed matrix
    with pytest.raises(ValueError, match="Must provide x_labels"):
        confusion_matrix(np.array([[1, 2], [3, 4]]))

    # Test unknown metric
    with pytest.raises(ValueError, match="Unknown metric='invalid'"):
        confusion_matrix(
            np.array([[1, 2], [3, 4]]),
            x_labels=("A", "B"),
            metrics={"invalid": None},
        )


def test_confusion_matrix_custom_annotations(sample_conf_mat: np.ndarray) -> None:
    """Test custom cell annotations."""
    custom_annotations = [["tile 11", "tile 12"], ["tile 21", "tile 22"]]
    fig = confusion_matrix(
        sample_conf_mat,
        x_labels=("Negative", "Positive"),
        annotations=custom_annotations,
    )

    # Check if custom annotations are used
    anno_texts = {
        anno.text.split("<br>")[0] for anno in fig.layout.annotations if anno.text
    }
    assert anno_texts > set(np.array(custom_annotations).flatten())


def test_confusion_matrix_heatmap_kwargs(sample_conf_mat: np.ndarray) -> None:
    """Test customization via heatmap_kwargs."""
    heatmap_kwargs = {
        "colorscale": "Viridis",
        "showscale": True,
        "xgap": 5,
        "ygap": 5,
    }
    fig = confusion_matrix(
        sample_conf_mat,
        x_labels=("Negative", "Positive"),
        heatmap_kwargs=heatmap_kwargs,
    )

    # Check if heatmap properties are applied
    heatmap = fig.data[0]
    assert heatmap.showscale is True
    assert heatmap.xgap == 5
    assert heatmap.ygap == 5
    assert heatmap.colorscale[0][1].lower() == "#440154"


def test_confusion_matrix_long_labels(sample_conf_mat: np.ndarray) -> None:
    """Test handling of long labels."""
    long_labels = ("Very Long Negative Label", "Very Long Positive Label")
    fig = confusion_matrix(sample_conf_mat, x_labels=long_labels, y_labels=long_labels)

    y_labels = [anno.y for anno in fig.layout.annotations]
    x_labels = [anno.x for anno in fig.layout.annotations]

    for label in long_labels:
        split_label = label.replace("Long ", "Long<br>")
        assert split_label in x_labels, f"{split_label} not in {x_labels=}"
        assert split_label in y_labels, f"{split_label} not in {y_labels=}"
