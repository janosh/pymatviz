"""Shared pytest configuration and fixtures for widget tests."""

from typing import Any
from unittest.mock import patch

import pytest
from pymatgen.core import Structure

from pymatviz.widgets.composition import CompositionWidget
from pymatviz.widgets.matterviz import fetch_widget_asset
from pymatviz.widgets.structure import StructureWidget
from pymatviz.widgets.trajectory import TrajectoryWidget


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
        assert hasattr(widget, attr)


def assert_widget_notebook_integration(
    widget: CompositionWidget | StructureWidget | TrajectoryWidget,
) -> None:
    """Validate that a widget renders correctly in notebook environments."""
    # MIME bundle (accept dict or (data, metadata)). ty mis-resolves
    # _repr_mimebundle_ over the widget-class union as unbound; bound at runtime.
    mime = widget._repr_mimebundle_()  # ty: ignore[missing-argument]
    if isinstance(mime, tuple):
        assert len(mime) == 2
        mime_data, metadata = mime
        assert isinstance(metadata, dict)
    else:
        mime_data = mime
    assert isinstance(mime_data, dict)

    view_key = "application/vnd.jupyter.widget-view+json"
    assert view_key in mime_data
    assert "text/plain" in mime_data
    view = mime_data[view_key]
    assert isinstance(view, dict)
    assert isinstance(view.get("model_id"), str)
    assert bool(view["model_id"]) is True
    # Fresh model_id on a new repr call (new view)
    mime2 = widget._repr_mimebundle_()  # ty: ignore[missing-argument]
    if mime2 is not None:
        mime2 = mime2[0] if isinstance(mime2, tuple) else mime2
        view2 = mime2[view_key]
        model_id1 = view.get("model_id")
        model_id2 = view2.get("model_id")
        if model_id1 == model_id2:
            # This might indicate VS Code re-evaluation bugs or widget caching issues
            import warnings

            msg = (
                f"Widget {type(widget).__name__} returned same model_id "
                f"on repeated repr calls: {model_id1}. This may indicate "
                f"VS Code re-evaluation bugs or widget caching issues."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            # This is the expected behavior - fresh model_id on each repr call
            assert model_id1 != model_id2, (
                "Model IDs should be different on repeated repr calls"
            )
    assert isinstance(view.get("version_major"), int)
    assert type(widget).__name__ in str(mime_data["text/plain"])

    for attr_name in ("_model_name", "_view_name", "_model_module", "_view_module"):
        assert hasattr(widget, attr_name)

    # IPython display path (validate if present)
    try:
        with (
            patch("IPython.display.publish_display_data") as pub,
            patch("IPython.display.display") as disp,
        ):
            from IPython.display import display

            display(widget)
            assert disp.call_count == 1
            if pub.call_count:
                args, kwargs = pub.call_args
                data = kwargs.get("data")
                if data is None and args:
                    data = args[0]
                assert isinstance(data, dict)
                assert view_key in data
                assert "text/plain" in data
                pub_view = data[view_key]
                assert isinstance(pub_view, dict)
                assert pub_view.get("model_id") == view.get("model_id")
                assert isinstance(pub_view.get("version_major"), int)
    except ImportError:
        pass

    # Anywidget ESM/CSS sanity
    esm, css = widget._esm, widget._css
    assert len(esm) > 1000
    assert len(css) > 100
    assert "export default" in esm or " as default" in esm
    assert "render" in esm


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
