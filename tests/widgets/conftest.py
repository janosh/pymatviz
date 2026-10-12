"""Shared pytest configuration and fixtures for widget tests."""

from typing import Any

import pytest
from pymatgen.core import Structure

from pymatviz.widgets.matterviz import MatterVizWidget, fetch_widget_asset


@pytest.fixture(scope="session", autouse=True)
def ensure_widget_assets() -> None:
    """Warm the matterviz-anywidget asset cache before any tests run.

    Widget construction needs the ESM/CSS bundle; fetch it once (from
    MATTERVIZ_ANYWIDGET_DIR, the on-disk cache, or jsDelivr) so per-widget
    construction is fast and works offline after the first fetch.
    """
    for filename in ("matterviz.js", "matterviz.css"):
        fetch_widget_asset(filename)


@pytest.fixture
def multi_frame_trajectory(fe3co4_disordered: Structure) -> dict[str, Any]:
    """Multi-frame trajectory for testing."""
    frames = [
        {
            "structure": fe3co4_disordered.as_dict(),
            "step": idx,
            "metadata": {"step": idx},
        }
        for idx in range(5)
    ]
    return {"frames": frames}


def assert_widget_property_sync(
    widget: MatterVizWidget, property_name: str, test_value: Any
) -> None:
    """Test that widget properties sync correctly."""
    setattr(widget, property_name, test_value)
    assert getattr(widget, property_name) == test_value

    trait = widget.class_traits()[property_name]
    assert trait.metadata.get("sync") is True, f"Property {property_name} not synced"
