"""Shared pytest configuration and fixtures for widget tests."""

import os
import subprocess
from typing import Any
from unittest.mock import patch

import pytest
from pymatgen.core import Structure

from pymatviz import PKG_DIR
from pymatviz.widgets.composition import CompositionWidget
from pymatviz.widgets.structure import StructureWidget
from pymatviz.widgets.trajectory import TrajectoryWidget


@pytest.fixture(scope="session", autouse=True)
def build_web_files() -> None:
    """Build web files (if not yet exist) before any tests run."""
    widgets_dir = f"{PKG_DIR}/widgets"
    web_dir = f"{widgets_dir}/web"

    # Only build if the file doesn't exist
    if not os.path.isfile(f"{web_dir}/build/matterviz.mjs"):
        try:
            subprocess.run(
                ["deno", "task", "build"],  # noqa: S607
                check=True,
                cwd=web_dir,
                capture_output=True,
                timeout=20,
            )
        except (subprocess.SubprocessError, subprocess.TimeoutExpired) as exc:
            exc.add_note("Failed to build web files")
            raise


@pytest.fixture
def multi_frame_trajectory(fe3co4_disordered: Structure) -> dict[str, Any]:
    """Multi-frame trajectory for testing."""
    frames = [
        {"structure": fe3co4_disordered.as_dict(), "metadata": {"step": idx}}
        for idx in range(5)
    ]
    return {"frames": frames}


# Shared test helper functions
def assert_widget_build_files(
    widget: CompositionWidget | StructureWidget | TrajectoryWidget,
) -> None:
    """Test that widget loads build files correctly."""
    # Test build files are loaded
    assert hasattr(widget, "_esm"), "Widget missing JavaScript code"
    assert hasattr(widget, "_css"), "Widget missing CSS code"
    assert len(widget._esm) > 1000, "JavaScript code suspiciously short"
    assert len(widget._css) > 100, "CSS code suspiciously short"

    # Test widget has required display attributes
    required_attrs = ["_model_name", "_view_name", "_model_module", "_view_module"]
    for attr in required_attrs:
        assert hasattr(widget, attr), f"Widget missing {attr}"


def assert_widget_notebook_integration(
    widget: CompositionWidget | StructureWidget | TrajectoryWidget,
) -> None:
    """Test that widget integrates with notebook environments."""
    # Test Jupyter display functionality
    with patch("IPython.display.display") as mock_display:
        try:
            from IPython.display import display

            display(widget)
            mock_display.assert_called_once_with(widget)
        except ImportError:
            # IPython not available in test environment
            pass

    # Test Marimo anywidget interface
    widget_interface = {"esm": widget._esm, "css": widget._css, "model": widget}
    assert len(widget_interface["esm"]) > 1000
    assert len(widget_interface["css"]) > 100
    assert widget_interface["model"] == widget


def assert_widget_property_sync(
    widget: CompositionWidget | StructureWidget | TrajectoryWidget,
    property_name: str,
    test_value: Any,
) -> None:
    """Test that widget properties sync correctly."""
    setattr(widget, property_name, test_value)
    assert getattr(widget, property_name) == test_value

    # Test that attribute is tagged for sync
    trait = widget.class_traits()[property_name]
    assert trait.metadata.get("sync") is True, f"Property {property_name} not synced"


def assert_widget_edge_cases(
    widget: CompositionWidget | StructureWidget | TrajectoryWidget,
) -> None:
    """Test widget edge cases and error handling."""
    # Test widget handles missing/corrupted build files gracefully
    assert hasattr(widget, "_esm")
    assert hasattr(widget, "_css")
    assert len(widget._esm) > 1000  # Should contain actual content
    assert len(widget._css) > 100  # Should contain actual content
