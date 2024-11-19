from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import matplotlib.pyplot as plt
import pytest

import pymatviz as pmv


if TYPE_CHECKING:
    from collections.abc import Sequence


@pytest.mark.parametrize(
    ("v_offset", "h_offset", "labels", "fontsize", "y_max_headroom", "adjust_test_pos"),
    [
        (10, 0, None, 14, 1.2, False),
        (20, 0, ["label1", "label2", "label3"], 10, 1.5, True),
        (5, 5, [100, 200, 300], 16, 1.0, False),
    ],
)
def test_annotate_bars(
    v_offset: int,
    h_offset: int,
    labels: Sequence[str] | None,
    fontsize: int,
    y_max_headroom: float,
    adjust_test_pos: bool,
) -> None:
    bars = plt.bar(["A", "B", "C"], [1, 3, 2])
    ax = plt.gca()
    pmv.powerups.annotate_bars(
        ax=ax,
        v_offset=v_offset,
        h_offset=h_offset,
        labels=labels,
        fontsize=fontsize,
        y_max_headroom=y_max_headroom,
        adjust_test_pos=adjust_test_pos,
    )

    assert len(ax.texts) == len(bars)

    if labels is None:
        labels = [str(bar.get_height()) for bar in bars]

    # test that labels have expected text and fontsize
    for text, label in zip(ax.texts, labels, strict=True):
        assert text.get_text() == str(label)
        assert text.get_fontsize() == fontsize

    # test that y_max_headroom is respected
    ylim_max = ax.get_ylim()[1]
    assert ylim_max >= max(bar.get_height() for bar in bars) * y_max_headroom

    # test error when passing wrong number of labels
    bad_labels = ("label1", "label2")
    with pytest.raises(
        ValueError,
        match=f"Got {len(bad_labels)} labels but {len(bars)} bars to annotate",
    ):
        pmv.powerups.annotate_bars(ax, labels=bad_labels)

    # test error message if adjustText not installed
    err_msg = (
        "adjustText not installed, falling back to default matplotlib label "
        "placement. Use pip install adjustText."
    )
    with (
        patch.dict("sys.modules", {"adjustText": None}),
        pytest.raises(ImportError, match=err_msg),
    ):
        pmv.powerups.annotate_bars(ax, adjust_test_pos=True)


def test_with_marginal_hist() -> None:
    fig, ax = plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6])
    ax_main = pmv.powerups.with_marginal_hist([1, 2, 3], [4, 5, 6], fig=fig)
    assert isinstance(ax_main, plt.Axes)
    assert len(fig.axes) == 4

    gs = fig.add_gridspec(ncols=2, nrows=2)
    ax_main = pmv.powerups.with_marginal_hist([1, 2, 3], [4, 5, 6], cell=gs[1, 0])
    assert isinstance(ax_main, plt.Axes)
    assert len(fig.axes) == 4
    assert fig.get_axes()[0].get_position().get_points().flat == pytest.approx(
        [0.125, 0.11, 0.9, 0.88]
    )
