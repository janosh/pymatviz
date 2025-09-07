"""Test notebook integration for automatic pymatgen object rendering."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest
from pymatgen.core import Lattice, Structure


if "IPython" not in sys.modules:
    # Mock IPython module to avoid import errors in CI
    mock_ipython = MagicMock()
    mock_ipython.display.publish_display_data = MagicMock()
    # Set version_info to a high version to satisfy matplotlib's version check
    mock_ipython.version_info = (8, 24, 0)  # >= (8, 24) to pass matplotlib check
    sys.modules["IPython"] = mock_ipython
    sys.modules["IPython.display"] = mock_ipython.display

import pymatviz as pmv
from pymatviz import notebook, structure_3d
from pymatviz.widgets.mime import set_renderer


@pytest.mark.parametrize(
    "renderer_type",
    [pmv.StructureWidget.__name__, lambda: pmv.StructureWidget, lambda: structure_3d],
)
def test_set_renderer_structure(renderer_type: str | Any) -> None:
    """Test set_renderer for Structure with string, class, and function renderers."""
    # Clear registry to start fresh
    from pymatviz.notebook import _RENDERER_REGISTRY

    _RENDERER_REGISTRY.clear()

    prev = set_renderer(Structure, structure_3d)
    assert prev is None
    # Set renderer
    renderer = renderer_type() if callable(renderer_type) else renderer_type
    prev2 = set_renderer(Structure, renderer)
    assert prev2 == structure_3d
    # Switch back to string
    prev3 = set_renderer(Structure, pmv.StructureWidget.__name__)
    assert prev3 == renderer


@pytest.mark.parametrize(
    "renderer_type",
    [pmv.StructureWidget.__name__, lambda: pmv.StructureWidget, lambda: structure_3d],
)
def test_set_renderer_ase_atoms(renderer_type: str | Any) -> None:
    """Test set_renderer for ASE Atoms with string, class, and function renderers."""
    pytest.importorskip("ase")
    from ase.atoms import Atoms

    # Clear registry to start fresh
    from pymatviz.notebook import _RENDERER_REGISTRY

    _RENDERER_REGISTRY.clear()

    prev = set_renderer(Atoms, structure_3d)
    assert prev is None
    renderer = renderer_type() if callable(renderer_type) else renderer_type
    prev2 = set_renderer(Atoms, renderer)
    assert prev2 == structure_3d
    prev3 = set_renderer(Atoms, pmv.StructureWidget.__name__)
    assert prev3 == renderer


def test_set_renderer_registry_isolation() -> None:
    """Test that different classes have isolated renderer registry entries."""
    from pymatviz.notebook import _RENDERER_REGISTRY

    _RENDERER_REGISTRY.clear()
    set_renderer(Structure, structure_3d)

    class CustomStructure:
        pass

    def custom_renderer(obj: Any) -> Any:
        return structure_3d(obj)

    set_renderer(CustomStructure, custom_renderer)
    assert Structure in _RENDERER_REGISTRY
    assert CustomStructure in _RENDERER_REGISTRY
    assert _RENDERER_REGISTRY[Structure] == structure_3d
    assert _RENDERER_REGISTRY[CustomStructure] == custom_renderer


def test_set_renderer_widget_and_string_equivalence() -> None:
    """Test that both string and class renderers for StructureWidget
    produce a widget MIME bundle.
    """
    pmv.notebook_mode(on=True)
    struct = Structure(Lattice.cubic(3), ["Fe", "Fe"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    for renderer in (pmv.StructureWidget.__name__, pmv.StructureWidget):
        set_renderer(Structure, renderer)
        mime = struct._repr_mimebundle_()
        assert isinstance(mime, dict)
        assert "application/vnd.jupyter.widget-view+json" in mime
        assert "text/plain" in mime
    pmv.notebook_mode(on=False)

    # check type error for invalid renderer
    with pytest.raises(TypeError, match="Unknown renderer=None"):
        set_renderer(Structure, None)  # type: ignore[arg-type]


@pytest.fixture
def diffraction_pattern() -> Any:
    """Create a simple diffraction pattern for testing."""
    try:
        from pymatgen.analysis.diffraction.xrd import DiffractionPattern

        return DiffractionPattern(
            x=[10, 20, 30],
            y=[100, 50, 75],
            hkls=[[{"hkl": (1, 0, 0)}], [{"hkl": (1, 1, 0)}], [{"hkl": (1, 1, 1)}]],
            d_hkls=[2.5, 2.0, 1.8],
        )
    except ImportError:
        return None


@pytest.fixture
def test_objects(
    structures: tuple[Structure, Structure], ase_atoms: tuple[Any, Any]
) -> dict[str, Any]:
    """Create test objects for all supported types using existing fixtures."""
    objects = {}

    # Use first structure from conftest.py structures fixture
    objects["structure"] = structures[0]

    # Use first ase_atoms from conftest.py ase_atoms fixture, or create simple one
    if ase_atoms is not None:
        objects["ase_atoms"] = ase_atoms[0]
    else:
        # Fallback for when ASE is not available
        try:
            from ase.build import bulk

            objects["ase_atoms"] = bulk("Si", "diamond", a=5.43)
        except ImportError:
            objects["ase_atoms"] = None

    # DiffractionPattern (optional)
    try:
        from pymatgen.analysis.diffraction.xrd import DiffractionPattern

        objects["diffraction_pattern"] = DiffractionPattern(
            x=[10, 20, 30],
            y=[100, 50, 75],
            hkls=[[{"hkl": (1, 0, 0)}], [{"hkl": (1, 1, 0)}], [{"hkl": (1, 1, 1)}]],
            d_hkls=[2.5, 2.0, 1.8],
        )
    except ImportError:
        objects["diffraction_pattern"] = None

    # Mock phonon objects
    class MockPhononBands:
        def __repr__(self) -> str:
            return "MockPhononBands()"

    class MockPhononDos:
        def __repr__(self) -> str:
            return "MockPhononDos()"

    objects["phonon_bands"] = MockPhononBands()
    objects["phonon_dos"] = MockPhononDos()

    return objects


def test_hide_plotly_toolbar() -> None:
    """Test the _hide_plotly_toolbar function."""
    from unittest.mock import MagicMock

    # Mock the Plotly Figure class
    fig = MagicMock(spec=["update_layout"])
    notebook._hide_plotly_toolbar(fig)

    # Check that update_layout was called with modebar config
    fig.update_layout.assert_called_once()
    call_args = fig.update_layout.call_args[1]  # Get kwargs
    assert "modebar" in call_args
    assert "remove" in call_args["modebar"]

    # Check that expected buttons are removed
    removed_buttons = call_args["modebar"]["remove"]
    expected_buttons = [
        "pan",
        "select",
        "lasso",
        "zoomIn",
        "zoomOut",
        "autoScale",
        "resetScale3d",
    ]
    for button in expected_buttons:
        assert button in removed_buttons


@pytest.mark.parametrize("enable", [True, False])
def test_notebook_mode_toggle(enable: bool) -> None:
    """Test enabling and disabling notebook integration."""
    # Start with opposite state
    pmv.notebook_mode(on=not enable)

    # Toggle to desired state
    pmv.notebook_mode(on=enable)

    # Check Structure (always available)
    assert hasattr(Structure, "_ipython_display_") == enable
    assert hasattr(Structure, "_repr_mimebundle_") == enable

    if enable:
        assert callable(Structure._ipython_display_)
        assert callable(Structure._repr_mimebundle_)

    # Check optional classes
    optional_classes = [
        ("ase.atoms", "Atoms"),
        ("pymatgen.analysis.diffraction.xrd", "DiffractionPattern"),
        ("pymatgen.phonon.bandstructure", "PhononBandStructureSymmLine"),
        ("pymatgen.phonon.dos", "PhononDos"),
        ("phonopy.phonon.dos", "TotalDos"),
    ]

    for module_name, class_name in optional_classes:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            assert hasattr(cls, "_ipython_display_") == enable
            assert hasattr(cls, "_repr_mimebundle_") == enable
        except ImportError:
            pass  # Module not available

    # Clean up
    pmv.notebook_mode(on=False)


@pytest.mark.parametrize(
    ("obj_name", "should_skip"),
    [("structure", False), ("ase_atoms", True), ("diffraction_pattern", True)],
)
def test_object_display_methods(
    test_objects: dict[str, Any],
    obj_name: str,
    should_skip: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test _ipython_display_ method for all supported object types."""
    obj = test_objects[obj_name]

    if should_skip and obj is None:
        pytest.skip(f"{obj_name} not available")

    pmv.notebook_mode(on=True)

    # Mock publish_display_data to capture what gets published
    published_data = []

    def mock_publish_display_data(data: dict[str, str]) -> None:
        published_data.append(data)

    monkeypatch.setattr(
        "IPython.display.publish_display_data", mock_publish_display_data
    )

    try:
        # Call the display method
        obj._ipython_display_()

        # Check that data was published
        assert len(published_data) == 1
        display_data = published_data[0]

        # Check MIME types
        assert "text/plain" in display_data
        # May or may not have plotly data depending on pymatviz availability
        if "application/vnd.plotly.v1+json" in display_data:
            plotly_json = display_data["application/vnd.plotly.v1+json"]
            assert isinstance(plotly_json, dict)
            assert "data" in plotly_json
            assert "layout" in plotly_json

    finally:
        pmv.notebook_mode(on=False)


@pytest.mark.parametrize(
    ("obj_name", "should_skip"),
    [("structure", False), ("ase_atoms", True), ("diffraction_pattern", True)],
)
def test_object_repr_mimebundle(
    test_objects: dict[str, Any], obj_name: str, should_skip: bool
) -> None:
    """Test _repr_mimebundle_ method for all supported object types."""
    obj = test_objects[obj_name]

    if should_skip and obj is None:
        pytest.skip(f"{obj_name} not available")

    pmv.notebook_mode(on=True)

    try:
        mime_bundle = obj._repr_mimebundle_()

        assert isinstance(mime_bundle, dict)
        assert "text/plain" in mime_bundle

        # May or may not have plotly data depending on pymatviz availability
        if "application/vnd.plotly.v1+json" in mime_bundle:
            plotly_json = mime_bundle["application/vnd.plotly.v1+json"]
            assert isinstance(plotly_json, dict)
            assert "data" in plotly_json
            assert "layout" in plotly_json

    finally:
        pmv.notebook_mode(on=False)


def test_structure_toolbar_hiding(structures: tuple[Structure, Structure]) -> None:
    """Test that plotly toolbar is properly hidden in structure display."""
    pmv.notebook_mode(on=True)

    try:
        mime_bundle = structures[0]._repr_mimebundle_()

        # Check that plotly JSON has toolbar configuration (if using plotly renderer)
        if "application/vnd.plotly.v1+json" in mime_bundle:
            plotly_json = mime_bundle["application/vnd.plotly.v1+json"]
            layout = plotly_json["layout"]
            assert "modebar" in layout
            modebar_config = layout["modebar"]
            assert "remove" in modebar_config
            # Check that some common toolbar buttons are removed
            removed_buttons = modebar_config["remove"]
            assert "pan" in removed_buttons
            assert "select" in removed_buttons
            assert "lasso" in removed_buttons

    finally:
        pmv.notebook_mode(on=False)


def test_fallback_without_pymatviz(
    structures: tuple[Structure, Structure], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that methods fall back gracefully when pymatviz is not available."""
    # Mock import to fail for pymatviz
    original_import = __import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pymatviz":
            raise ImportError("No module named 'pymatviz'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    # Mock publish_display_data to capture fallback behavior
    published_data = []

    def mock_publish_display_data(data: dict[str, str]) -> None:
        published_data.append(data)

    monkeypatch.setattr(
        "IPython.display.publish_display_data", mock_publish_display_data
    )

    pmv.notebook_mode(on=True)

    try:  # Test _ipython_display_ fallback
        structures[0]._ipython_display_()
        assert len(published_data) == 1
        display_data = published_data[0]
        assert "text/plain" in display_data
        assert "application/vnd.plotly.v1+json" not in display_data

        # Test _repr_mimebundle_ fallback
        mime_bundle = structures[0]._repr_mimebundle_()
        assert "text/plain" in mime_bundle
        assert "application/vnd.plotly.v1+json" not in mime_bundle

    finally:
        pmv.notebook_mode(on=False)


def test_multiple_enable_disable_cycles() -> None:
    """Test that multiple enable/disable cycles work correctly."""
    pmv.notebook_mode(on=False)  # Start with disabled state
    assert not hasattr(Structure, "_ipython_display_")

    # Enable -> Disable -> Enable -> Disable
    for _ in range(3):
        pmv.notebook_mode(on=True)
        assert hasattr(Structure, "_ipython_display_")
        assert hasattr(Structure, "_repr_mimebundle_")

        pmv.notebook_mode(on=False)
        assert not hasattr(Structure, "_ipython_display_")
        assert not hasattr(Structure, "_repr_mimebundle_")


@pytest.mark.parametrize(
    "import_error_modules",
    [
        ["pymatgen.core"],
        ["ase.atoms"],
        ["pymatgen.analysis.diffraction.xrd"],
        ["pymatgen.phonon.bandstructure"],
        ["pymatgen.phonon.dos"],
        ["phonopy.phonon.dos"],
        ["ase.atoms", "pymatgen.phonon.dos"],
    ],
)
def test_import_error_handling(
    import_error_modules: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test notebook_mode function when various imports fail."""
    original_import = __import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in import_error_modules:
            raise ImportError(f"No module with {name=}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", mock_import)

    # Test enabling and disabling with import failures
    pmv.notebook_mode(on=True)
    pmv.notebook_mode(on=False)


def test_direct_function_calls_coverage() -> None:
    """Test calling display functions directly to increase coverage."""
    # Create test objects
    lattice = Lattice.cubic(4.0)
    structure = Structure(lattice, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])

    result = notebook._structure_repr_mimebundle_(structure)
    assert isinstance(result, dict)
    assert "text/plain" in result

    try:  # Test ASE atoms functions if available
        from ase.build import bulk

        atoms = bulk("Si", "diamond", a=5.43)
        result = notebook._ase_atoms_repr_mimebundle_(atoms)
        assert isinstance(result, dict)
        assert "text/plain" in result
    except ImportError:
        pass  # ASE not available

    try:  # Test diffraction pattern functions if available
        from pymatgen.analysis.diffraction.xrd import DiffractionPattern

        pattern = DiffractionPattern(
            x=[10, 20, 30],
            y=[100, 50, 75],
            hkls=[[{"hkl": (1, 0, 0)}], [{"hkl": (1, 1, 0)}], [{"hkl": (1, 1, 1)}]],
            d_hkls=[2.5, 2.0, 1.8],
        )
        result = notebook._diffraction_pattern_repr_mimebundle_(pattern)
        assert isinstance(result, dict)
        assert "text/plain" in result
    except ImportError:
        pass  # pymatgen.analysis.diffraction not available


def test_phonon_display_functions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test phonon display functions with mock objects."""

    # Create mock phonon objects
    class MockPhononBands:
        def __repr__(self) -> str:
            return "MockPhononBands()"

    class MockPhononDos:
        def __repr__(self) -> str:
            return "MockPhononDos()"

    mock_bands = MockPhononBands()
    mock_dos = MockPhononDos()

    # Mock publish_display_data
    published_data = []

    def mock_publish_display_data(data: dict[str, str]) -> None:
        published_data.append(data)

    monkeypatch.setattr(
        "IPython.display.publish_display_data", mock_publish_display_data
    )

    # Test phonon bands functions
    try:
        notebook._phonon_bands_ipython_display_(mock_bands)
        assert len(published_data) >= 1

        result = notebook._phonon_bands_repr_mimebundle_(mock_bands)
        assert isinstance(result, dict)
        assert "text/plain" in result
    except (TypeError, ImportError):
        # error expected if pymatviz functions fail with mock objects or pymatviz
        pass  # not available

    # Test phonon DOS functions
    try:
        published_data.clear()
        notebook._phonon_dos_ipython_display_(mock_dos)
        assert len(published_data) >= 1

        result = notebook._phonon_dos_repr_mimebundle_(mock_dos)
        assert isinstance(result, dict)
        assert "text/plain" in result
    except (TypeError, ImportError):
        # error expected if pymatviz functions fail with mock objects or pymatviz
        pass  # not available


def test_phonopy_dos_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test phonopy TotalDos notebook integration."""
    pytest.importorskip("phonopy", reason="phonopy not available")

    from pymatviz.utils.testing import load_phonopy_nacl

    # Load phonopy object and calculate DOS
    phonopy_nacl = load_phonopy_nacl()
    phonopy_nacl.run_mesh([10, 10, 10])
    phonopy_nacl.run_total_dos()

    pmv.notebook_mode(on=True)

    published_data: list[dict[str, str]] = []  # capture what gets published

    def mock_publish_display_data(data: dict[str, str]) -> None:
        published_data.append(data)

    monkeypatch.setattr(
        "IPython.display.publish_display_data", mock_publish_display_data
    )

    try:  # Check that display methods were added to the class
        from phonopy.phonon.dos import TotalDos

        assert hasattr(TotalDos, "_ipython_display_")
        assert hasattr(TotalDos, "_repr_mimebundle_")
        assert callable(TotalDos._ipython_display_)
        assert callable(TotalDos._repr_mimebundle_)

        # Test _ipython_display_ method on the instance
        phonopy_nacl.total_dos._ipython_display_()  # type: ignore[attr-defined]

        # Check that data was published
        assert len(published_data) == 1
        display_data = published_data[0]

        # Check MIME types
        assert "text/plain" in display_data
        # Should have plotly data since we have a real DOS object
        if "application/vnd.plotly.v1+json" in display_data:
            plotly_json = display_data["application/vnd.plotly.v1+json"]
            assert isinstance(plotly_json, dict)
            assert "data" in plotly_json
            assert "layout" in plotly_json

        # Test _repr_mimebundle_ method on the instance
        mime_bundle = phonopy_nacl.total_dos._repr_mimebundle_()  # type: ignore[attr-defined]
        assert isinstance(mime_bundle, dict)
        assert "text/plain" in mime_bundle

    finally:  # Clean up
        pmv.notebook_mode(on=False)
