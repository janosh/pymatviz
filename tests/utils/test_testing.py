from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pytest

from pymatviz.utils.testing import TEST_FILES, interactive_check


def test_test_files_dir() -> None:
    assert os.path.isdir(TEST_FILES)


@pytest.fixture
def create_figure() -> tuple[plt.Figure, plt.Axes]:
    """A matplotlib figure and axes for test."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([1, 2, 3], [4, 5, 6])
    return fig, ax


class TestInteractiveCheck:
    def test_good_figure(self, create_figure: tuple[plt.Figure, plt.Axes]) -> None:
        fig, ax = create_figure

        # Test Axes
        interactive_check(ax, elem_to_check="line")

        # Test Figure
        interactive_check(fig, elem_to_check="line")

        # Test KeyboardInterrupt capture (should be treated as "y")
        interactive_check(fig, elem_to_check="line")

    @pytest.mark.xfail(reason="test bad figure")
    def test_bad_figure(self, create_figure: tuple[plt.Figure, plt.Axes]) -> None:
        fig, _ax = create_figure

        interactive_check(fig, elem_to_check="bad figure")

    def test_skip_ci(
        self,
        create_figure: tuple[plt.Figure, plt.Axes],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        fig, _ax = create_figure

        monkeypatch.setenv("CI", "true")
        assert interactive_check(fig, elem_to_check="figure") is None

    def test_not_supported_type(self) -> None:
        plot = [1, 2, 3]
        with pytest.raises(TypeError, match="plot type list is not supported"):
            interactive_check(plot)
