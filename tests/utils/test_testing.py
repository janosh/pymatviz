from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pytest

from pymatviz.utils.testing import TEST_FILES, interactive_check


def test_test_files_dir() -> None:
    assert os.path.isdir(TEST_FILES)


class TestInteractiveCheck:
    def test_good_figure(self) -> None:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        # Test Axes
        interactive_check(ax, elem_to_check="line")

        # TODO: after closing the first figure, the following don't show up

        # Test Figure
        interactive_check(fig, elem_to_check="line")

        # Test KeyboardInterrupt capture (should be treated as "y")
        interactive_check(fig, elem_to_check="line")

    def test_bad_figure(self) -> None:
        # TODO: xfail
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [4, 5, 6])

        interactive_check(fig, elem_to_check="bad figure")

    def test_skip_ci(self) -> None:
        pass

    def test_not_supported_type(self) -> None:
        plot = [1, 2, 3]
        with pytest.raises(TypeError, match="plot type list is not supported"):
            interactive_check(plot)
