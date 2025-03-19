from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import pytest

from pymatviz.classify.confusion_matrix import confusion_matrix


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    AnnotationCallable = Callable[[int, int, float, float, float, float], str]


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


def test_confusion_matrix_color_text_alignment(sample_conf_mat: np.ndarray) -> None:
    """Test that colors and text annotations are properly aligned in the confusion
    matrix. This test specifically checks that the cell colors (z values) match their
    corresponding text annotations and hover text."""
    fig = confusion_matrix(
        sample_conf_mat,
        x_labels=("Negative", "Positive"),
        y_labels=("Negative", "Positive"),
        normalize=True,  # use normalization to get percentages
    )

    # Get the heatmap trace
    heatmap = fig.data[0]

    # Get z-values (colors), annotations, and hover text
    z_values = heatmap.z
    # Filter out metric annotations (like Acc=85.0%) and keep only cell annotations
    annotations = [
        anno.text
        for anno in fig.layout.annotations
        if anno.text
        and not any(
            metric in anno.text for metric in ["Acc=", "Prec=", "F1=", "MCC=", "Rec="]
        )
    ]
    hover_text = heatmap.text

    # The largest value in the confusion matrix should correspond to
    # the darkest color and highest percentage in both annotations and hover text
    max_val_idx = np.unravel_index(np.argmax(z_values), z_values.shape)

    # Check that the cell with maximum value has:
    # 1. The highest z-value (darkest color)
    assert z_values[max_val_idx] == np.max(z_values)

    # 2. The highest percentage in its annotation
    max_anno_pct = max(
        float(anno.rstrip("%").split("<br>")[-1]) / 100
        for anno in annotations
        if anno and "%" in anno
    )
    assert abs(z_values[max_val_idx] - max_anno_pct) < 1e-10

    # 3. The highest percentage in its hover text
    max_hover_pct = max(
        float(text.split("Percent: ")[1].split("%")[0]) / 100
        for text in hover_text.flatten()
        if text and "Percent:" in text
    )
    assert abs(z_values[max_val_idx] - max_hover_pct) < 1e-10

    # Test alignment with raw labels
    # Create a test case where we know TN should be highest
    y_true = ["Negative"] * 6 + ["Positive"] * 2
    # 5 TN, 1 FP, 1 FN, 1 TP
    y_pred = ["Negative"] * 5 + ["Positive", "Negative", "Positive"]

    fig = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=True)
    heatmap = fig.data[0]

    # Check that false positives and false negatives have correct relative positions
    z_values = heatmap.z
    # In a 2x2 confusion matrix with our test data:
    # After np.rot90() is applied in the confusion_matrix function:
    # z[1,0] = TN = 5/8 = 0.625 (highest)
    # z[1,1] = FP = 1/8 = 0.125
    # z[0,0] = FN = 1/8 = 0.125
    # z[0,1] = TP = 1/8 = 0.125
    assert z_values[1, 0] > z_values[1, 1]  # TN > FP
    assert z_values[0, 1] == z_values[0, 0]  # TP = FN in this case

    # Verify this matches the hover text
    hover_text = heatmap.text
    for ii, jj in [(1, 0), (1, 1), (0, 0), (0, 1)]:  # indices after rotation
        text = hover_text[ii][jj]
        pct = float(text.split("Percent: ")[1].split("%")[0]) / 100
        assert abs(z_values[ii, jj] - pct) < 1e-10


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


def test_confusion_matrix_edge_cases() -> None:
    """Test edge cases and special situations in confusion matrix visualization."""
    # Test case 1: Perfect predictions (only TN and TP)
    y_true = ["Negative"] * 3 + ["Positive"] * 3
    y_pred = ["Negative"] * 3 + ["Positive"] * 3
    fig = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=True)
    z_values = fig.data[0].z
    # After rotation, diagonal should be 0.5 each, off-diagonal 0
    assert z_values.tolist() == [[0.0, 0.5], [0.5, 0.0]]

    # Test case 2: All wrong predictions (only FP and FN)
    y_true = ["Negative"] * 3 + ["Positive"] * 3
    y_pred = ["Positive"] * 3 + ["Negative"] * 3
    fig = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=True)
    z_values = fig.data[0].z
    # After rotation, off-diagonal should be 0.5 each, diagonal 0
    assert z_values.tolist() == [[0.5, 0], [0, 0.5]]

    # Test case 3: All predictions same class (no variation)
    y_true = ["Negative"] * 3 + ["Positive"] * 3
    y_pred = ["Negative"] * 6  # predict all negative
    fig = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=True)
    z_values = fig.data[0].z
    # After rotation, TN = 3/6, FN = 3/6, others 0
    assert z_values.tolist() == [[0.0, 0.0], [0.5, 0.5]]

    # Test case 4: Single example per class
    y_true = ["Negative", "Positive"]
    y_pred = ["Negative", "Positive"]
    fig = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=True)
    z_values = fig.data[0].z
    # After rotation, diagonal should be 0.5 each
    assert z_values.tolist() == [[0.0, 0.5], [0.5, 0.0]]

    # Test case 5: Highly imbalanced classes
    y_true = ["Negative"] * 99 + ["Positive"]
    y_pred = ["Negative"] * 99 + ["Positive"]
    fig = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize=True)
    z_values = fig.data[0].z
    # After rotation, TN should dominate
    assert z_values.tolist() == [[0.0, 0.01], [0.99, 0.0]]


def test_confusion_matrix_label_order() -> None:
    """Test that confusion matrix maintains correct label order regardless of input
    order."""
    # Test with labels in different orders
    y_true = ["Positive", "Negative", "Positive", "Negative"]
    y_pred = ["Positive", "Negative", "Negative", "Positive"]

    # Case 1: Default label order (alphabetical)
    fig1 = confusion_matrix(y_true=y_true, y_pred=y_pred)
    labels1 = fig1.data[0].x

    # Case 2: Explicit reverse label order
    fig2 = confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        x_labels=("Positive", "Negative"),
        y_labels=("Positive", "Negative"),
    )
    labels2 = fig2.data[0].x

    # Values should be consistent regardless of label order
    z1 = fig1.data[0].z
    z2 = fig2.data[0].z
    # Check that the matrices contain the same values (possibly in different positions)
    assert {*z1.flatten()} == {*z2.flatten()}
    # Check that labels are in the specified order
    assert tuple(reversed(labels1)) == labels2 == ("Positive", "Negative")


def make_annotation(count: int, total: int, row_pct: float, col_pct: float) -> str:
    """Custom annotation function that shows all available metrics."""
    return (
        f"Count: {count}<br>"
        f"Total: {total}<br>"
        f"Global: {count / total:.1%}<br>"
        f"Row: {row_pct:.1%}<br>"
        f"Col: {col_pct:.1%}"
    )


def test_confusion_matrix_callable_annotations(sample_conf_mat: np.ndarray) -> None:
    """Test that callable annotations work correctly."""
    fig = confusion_matrix(
        sample_conf_mat,
        x_labels=("Negative", "Positive"),
        y_labels=("Negative", "Positive"),
        annotations=make_annotation,
        normalize=True,
    )

    # Check that annotations contain all expected parts
    # Filter out metrics annotations (like Acc=85.0%)
    cell_annotations = [
        anno.text
        for anno in fig.layout.annotations
        if anno.text
        and not any(
            metric in anno.text for metric in ["Acc=", "Prec=", "F1=", "MCC=", "Rec="]
        )
    ]

    for anno in cell_annotations:
        assert "Count:" in anno
        assert "Total:" in anno
        assert "Global:" in anno
        assert "Row:" in anno
        assert "Col:" in anno

    # Verify values for a specific cell (e.g. TN = 50)
    tn_anno = next(
        anno.text
        for anno in fig.layout.annotations
        if anno.text and "Count: 50" in anno.text
    )
    assert "Total: 100" in tn_anno  # total should be sum of all cells (50+10+5+35)
    assert "Global: 50.0%" in tn_anno  # 50/100
    assert "Row: 83.3%" in tn_anno  # 50/(50+10)
    assert "Col: 90.9%" in tn_anno  # 50/(50+5)


def test_confusion_matrix_callable_annotations_multiclass(
    multi_class_conf_mat: np.ndarray,
) -> None:
    """Test callable annotations with multi-class confusion matrices."""
    fig = confusion_matrix(
        multi_class_conf_mat,
        x_labels=("A", "B", "C"),
        annotations=make_annotation,
        normalize=True,
    )

    # Get cell annotations (excluding metrics)
    cell_annotations = [
        anno.text
        for anno in fig.layout.annotations
        if anno.text and "Acc=" not in anno.text
    ]

    # Check number of cells and required fields
    assert len(cell_annotations) == 9  # 3x3 matrix
    assert all(
        all(field in anno for field in ["Count:", "Total:", "Global:", "Row:", "Col:"])
        for anno in cell_annotations
    )

    # Verify values for first diagonal element (40)
    diag_anno = next(anno for anno in cell_annotations if "Count: 40" in anno)
    total = multi_class_conf_mat.sum()
    assert all(
        text in diag_anno
        for text in [
            f"Total: {total}",  # 150
            "Global: 26.7%",  # 40/150
            "Row: 80.0%",  # 40/(40+5+5)
            "Col: 80.0%",  # 40/(40+4+6)
        ]
    )
