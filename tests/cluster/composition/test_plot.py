"""Unit tests for composition cluster plots."""

from __future__ import annotations

import itertools
import re
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pymatgen.core import Composition
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap

import pymatviz as pmv
from tests.conftest import np_rng


if TYPE_CHECKING:
    from collections.abc import Callable

    from pymatviz.cluster.composition import ProjectionMethod
    from pymatviz.cluster.composition.plot import ShowChemSys


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a sample DataFrame with compositions and properties."""
    df_sample = pd.DataFrame(
        {"composition": ["Fe0.5Co0.5", "Ni0.7Cu0.3", "Zr0.6Ti0.4"]}
    )
    df_sample["property"] = np.array([1, 10, 100])
    return df_sample


@pytest.fixture
def df_comp() -> pd.DataFrame:
    """Base DataFrame with only compositions for testing."""
    return pd.DataFrame({"composition": ["Fe2O3", "Al2O3", "Cu", "Au", "Ag"]})


@pytest.fixture
def df_prop(df_comp: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with compositions and property values for testing."""
    df_prop = df_comp.copy()
    df_prop["property"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    return df_prop


@pytest.fixture
def categorical_property_df(df_comp: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with compositions and categorical property values."""
    df_cat = df_comp.copy()
    df_cat["property"] = ["A", "B", "A", "C", "B"]
    return df_cat


@pytest.fixture
def sort_df(df_comp: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with compositions and property values for sort testing."""
    df_sort = df_comp.copy()
    df_sort["property"] = [3.0, 1.0, 4.0, 2.0, 5.0]
    return df_sort


@pytest.fixture
def float_precision_df(df_comp: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with floating-point values for testing format precision."""
    df_float = df_comp.copy().iloc[:3]
    df_float["property"] = [1.2345, 2.3456, 3.4567]
    return df_float


@pytest.fixture
def hover_alignment_df(df_comp: pd.DataFrame) -> pd.DataFrame:
    """DataFrame for testing hover text alignment."""
    df_hover = df_comp.copy()
    df_hover["property"] = [30.0, 10.0, 50.0, 20.0, 40.0]
    return df_hover


@pytest.fixture
def df_with_embeddings(hover_alignment_df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame with pre-computed embeddings."""
    df_embed = hover_alignment_df.copy()
    embeddings = np_rng.random((len(df_embed), 10))  # Create pre-computed embeddings
    df_embed["my_embeddings"] = list(embeddings)
    df_embed["embeddings"] = list(embeddings)  # Add a second copy with standard name
    return df_embed


@pytest.fixture
def custom_projection_func() -> Callable[[np.ndarray, int, Any], np.ndarray]:
    """Custom projection function for testing."""

    def projector(
        embeddings: np.ndarray, n_components: int, **_kwargs: Any
    ) -> np.ndarray:
        """Custom projection that returns the first n_components features."""
        return embeddings[:, :n_components]

    return projector  # type: ignore[return-value]


@pytest.fixture
def custom_coords_2d(df_prop: pd.DataFrame) -> np.ndarray:
    """Pre-computed 2D coordinates for testing."""
    return np_rng.random((len(df_prop), 2))


@pytest.fixture
def custom_coords_3d(df_prop: pd.DataFrame) -> np.ndarray:
    """Pre-computed 3D coordinates for testing."""
    return np_rng.random((len(df_prop), 3))


@pytest.fixture
def df_color_scale() -> pd.DataFrame:
    """Create a DataFrame with various property values for color scale testing."""
    compositions = ["Fe2O3", "Al2O3", "Cu", "Au", "Ag", "Ti", "Zn", "Si", "Mg", "Ca"]
    # Create property values spanning several orders of magnitude
    properties = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, -0.01, -0.1, -1.0, -10.0]
    return pd.DataFrame({"composition": compositions, "property": properties})


@pytest.mark.parametrize("prop_name", [None, "property"])
def test_basic_functionality(sample_df: pd.DataFrame, prop_name: str | None) -> None:
    """Test cluster_compositions with and without property coloring."""
    fig = pmv.cluster_compositions(
        df_in=sample_df,
        embedding_method="one-hot",
        projection="pca",
        prop_name=prop_name,
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)

    # Check property coloring if prop_name is provided
    if prop_name:
        data_dict = fig.to_dict()["data"][0]
        assert "coloraxis" in data_dict["marker"]
        assert data_dict["marker"]["coloraxis"] == "coloraxis"
        assert hasattr(fig.layout, "coloraxis")
        assert fig.layout.coloraxis.colorbar.title.text == prop_name


@pytest.mark.parametrize(
    ("show_chem_sys", "prop_name", "expected_traces", "check_symbol"),
    [
        (None, None, 1, False),  # No chem system, no property
        (None, "property", 1, False),  # No chem system, with property
        ("shape", None, 1, True),  # Shape mode, no property
        ("shape", "property", 1, True),  # Shape mode, with property
        ("color", None, 3, False),  # Color mode, no property
        # Color mode, with property (overrides color mode)
        ("color", "property", 1, False),
        ("color+shape", None, 1, True),  # Color+shape mode, no property
        ("color+shape", "property", 1, True),  # Color+shape mode, with property
    ],
)
def test_chemical_system_visualization(
    sample_df: pd.DataFrame,
    show_chem_sys: ShowChemSys | None,
    prop_name: str | None,
    expected_traces: int,
    check_symbol: bool,
) -> None:
    """Test different chemical system viz options with and without properties."""
    fig = pmv.cluster_compositions(
        df_in=sample_df,
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys=show_chem_sys,
        prop_name=prop_name,
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces

    # Get data dictionary for the first trace
    data_dict = fig.to_dict()["data"][0]

    # Check if property coloring is applied when prop_name is provided
    if prop_name:
        assert "marker" in data_dict
        assert "coloraxis" in data_dict["marker"]
        assert data_dict["marker"]["coloraxis"] == "coloraxis"
        assert hasattr(fig.layout, "coloraxis")
        assert fig.layout.coloraxis.colorbar.title.text == prop_name

    # Check if symbols are properly applied
    if check_symbol:
        assert "marker" in data_dict
        assert "symbol" in data_dict["marker"]
        assert isinstance(data_dict["marker"]["symbol"], list)
        symbols = data_dict["marker"]["symbol"]
        assert len(set(symbols)) >= min(len(sample_df), 3)

    # Check specific behavior for color mode without properties
    if show_chem_sys == "color" and not prop_name:
        # Each trace should have distinct color
        colors = [trace.marker.color for trace in fig.data]
        assert len(set(colors)) == 3  # Three unique colors

        # Each trace should contain 1 point (one composition per chem system)
        for trace in fig.data:
            assert len(trace.x) == 1
            assert len(trace.y) == 1

        # Ensure colors follow standard plotly palette (case-insensitive)
        standard_colors = ["#636efa", "#ef553b", "#00cc96"]
        for color in colors:
            assert color.lower() in [c.lower() for c in standard_colors]

    # Check for color+shape mode without properties
    if show_chem_sys == "color+shape" and not prop_name:
        # Check that marker has multiple colors (one per chemical system)
        assert isinstance(fig.data[0].marker.color, (list, tuple))
        # Each chemical system should have a different color
        assert len(fig.data[0].marker.color) == len(sample_df)

        # The colors should be standard Plotly colors (case-insensitive)
        standard_colors = ["#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a"]
        for color in fig.data[0].marker.color:
            assert color.lower() in [c.lower() for c in standard_colors]


def test_chemical_system_with_properties(sample_df: pd.DataFrame) -> None:
    """Test detailed behavior of chemical system visualization with properties."""
    # Test shape mode with properties
    fig_shape = pmv.cluster_compositions(
        df_in=sample_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys="shape",
    )

    assert len(fig_shape.data) == 1

    # Verify shape mapping
    data_dict = fig_shape.to_dict()["data"][0]
    assert isinstance(data_dict["marker"]["symbol"], list)

    # Verify properties are used for coloring
    assert "coloraxis" in data_dict["marker"]
    assert data_dict["marker"]["coloraxis"] == "coloraxis"

    # Verify the colorbar exists with the right title
    assert hasattr(fig_shape.layout, "coloraxis")
    assert fig_shape.layout.coloraxis.colorbar.title.text == "property"

    # Test color mode with properties
    fig_color = pmv.cluster_compositions(
        df_in=sample_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys="color",  # Should use properties, not chem systems
    )

    assert len(fig_color.data) == 1  # Should have one trace (all properties)

    # Verify properties are used for coloring
    data_dict = fig_color.to_dict()["data"][0]
    assert "coloraxis" in data_dict["marker"]
    assert data_dict["marker"]["coloraxis"] == "coloraxis"

    # Verify the colorbar exists with the right title
    assert hasattr(fig_color.layout, "coloraxis")
    assert fig_color.layout.coloraxis.colorbar.title.text == "property"

    # Test color+shape mode with properties
    fig_both = pmv.cluster_compositions(
        df_in=sample_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys="color+shape",
    )

    assert len(fig_both.data) == 1  # Should have one trace (all properties)

    # Verify properties are used for coloring
    data_dict = fig_both.to_dict()["data"][0]
    assert "coloraxis" in data_dict["marker"]
    assert data_dict["marker"]["coloraxis"] == "coloraxis"

    # Verify the colorbar exists with the right title
    assert hasattr(fig_both.layout, "coloraxis")
    assert fig_both.layout.coloraxis.colorbar.title.text == "property"

    # Verify shapes are set
    assert isinstance(data_dict["marker"]["symbol"], list)


@pytest.mark.parametrize(
    "sort_value",
    [
        True,  # Ascending order
        False,  # No sorting
        1,  # Ascending order
        0,  # No sorting
        -1,  # Descending order
        pytest.param(
            lambda values: np.argsort(values)[::-1], id="custom_sort_func"
        ),  # Custom sorting function
    ],
)
def test_sorting_options(
    sample_df: pd.DataFrame,
    sort_value: bool | int | Callable[[np.ndarray], np.ndarray],
) -> None:
    """Test different sorting options for property values."""
    fig = pmv.cluster_compositions(
        df_in=sample_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        sort=sort_value,
    )

    # Check that the figure has coloraxis for property coloring
    assert hasattr(fig.layout, "coloraxis")

    # Get the custom data which contains the composition strings in current order
    data_dict = fig.to_dict()["data"][0]
    custom_data = data_dict["customdata"]
    composition_order = [item[0] for item in custom_data]

    # Get the expected sort order based on properties
    if sort_value in (True, 1):
        # Ascending order
        expected_indices = np.argsort(sample_df["property"])
        expected_compositions = sample_df.iloc[expected_indices]["composition"].tolist()
    elif sort_value == -1 or (
        callable(sort_value) and sort_value.__name__ == "<lambda>"
    ):
        # Descending order (or custom function that does the same)
        expected_indices = np.argsort(sample_df["property"])[::-1]
        expected_compositions = sample_df.iloc[expected_indices]["composition"].tolist()
    else:
        # No sorting (original order)
        expected_compositions = sample_df["composition"].tolist()

    # Assert compositions are in the expected order
    # Compare simplified formulas to account for possible normalization differences
    simplified_actual = [comp.split()[0] for comp in composition_order]
    simplified_expected = [comp.split()[0] for comp in expected_compositions]
    assert simplified_actual == simplified_expected


def test_composite_viz_modes(sample_df: pd.DataFrame) -> None:
    """Test combined visualization modes with different options."""
    # Test color+shape with custom color map
    custom_colors = {
        "Co-Fe": "#ff0000",  # Red
        "Cu-Ni": "#00ff00",  # Green
        "Ti-Zr": "#0000ff",  # Blue
    }

    # Run without properties, using custom color map
    fig = pmv.cluster_compositions(
        df_in=sample_df,
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys="color",
        color_discrete_map=custom_colors,  # type: ignore[arg-type]
    )

    # Check that colors match our custom map
    trace_names = [trace.name for trace in fig.data]
    trace_colors = [trace.marker.color for trace in fig.data]

    for name, color in zip(trace_names, trace_colors, strict=False):
        assert color == custom_colors[name]

    # Edge case: Empty property array but with show_chem_sys=None
    # Should use default coloring
    fig_no_chem = pmv.cluster_compositions(
        df_in=sample_df[["composition"]],  # DataFrame without property column
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys=None,
    )

    # Should have a single trace
    assert len(fig_no_chem.data) == 1
    # With default color
    assert fig_no_chem.data[0].marker.color == "#636efa"

    # Edge case: With property values and shape mode, using custom marker size
    large_marker = 20
    fig_large = pmv.cluster_compositions(
        df_in=sample_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys="shape",
        marker_size=large_marker,
    )

    # Check marker size
    assert fig_large.data[0].marker.size == large_marker


@pytest.mark.parametrize(
    ("projection", "projection_kwargs", "expected_texts"),
    [
        (
            "pca",
            {},
            ["PC1", "cumulative"],
        ),
        (
            "tsne",
            {"perplexity": 1.0, "learning_rate": "auto"},
            ["perplexity", "learning_rate"],
        ),
        (
            "isomap",
            {"n_neighbors": 2, "metric": "euclidean"},
            ["n_neighbors", "metric"],
        ),
        (
            "kernel_pca",
            {"kernel": "rbf", "gamma": 0.1},
            ["kernel", "gamma"],
        ),
        pytest.param(
            "umap",
            {"n_neighbors": 15, "min_dist": 0.1},
            ["n_neighbors", "min_dist"],
            marks=pytest.mark.skipif(
                "umap" not in sys.modules, reason="umap not installed"
            ),
        ),
    ],
)
def test_projection_stats(
    request: pytest.FixtureRequest,
    projection: ProjectionMethod,
    projection_kwargs: dict[str, Any],
    expected_texts: list[str],
) -> None:
    """Test projection statistics display for different methods."""
    # Get the dataframe from fixture
    df_prop = request.getfixturevalue("df_prop")

    fig = pmv.cluster_compositions(
        df_in=df_prop,
        embedding_method="one-hot",
        projection=projection,
        projection_kwargs=projection_kwargs,
        show_projection_stats=True,
    )

    # Check that all expected texts are present in at least one annotation
    for expected_text in expected_texts:
        assert any(expected_text in ann.text for ann in fig.layout.annotations)

    # For PCA, perform additional checks on variance percentages
    if projection == "pca":
        # Find the stats annotation
        stats_text = next(
            ann.text for ann in fig.layout.annotations if "PC1" in ann.text
        )

        # Extract all PC percentages
        pc_percentages = []
        for line in stats_text.split("<br>"):
            if "PC" in line:
                # Extract both individual and cumulative percentages
                match = re.search(
                    r"PC\d+: (\d+\.?\d*)% \(cumulative: (\d+\.?\d*)%\)", line
                )
                assert match is not None, f"Could not parse PCA line: {line}"
                pc_percentages.append((float(match[1]), float(match[2])))

        # Check that we have expected number of principal components
        assert len(pc_percentages) == 2

        # Check percentages are between 0 and 100 (since they're in percentage form)
        for pc, cum in pc_percentages:
            assert 0 <= pc <= 100, f"PC percentage {pc} out of range [0,100]"
            assert 0 <= cum <= 100, f"Cumulative percentage {cum} out of range [0,100]"

        # Check that percentages are in descending order
        assert pc_percentages[0][0] >= pc_percentages[1][0]

        # Check that cumulative percentages are in ascending order
        assert pc_percentages[0][1] <= pc_percentages[1][1]

        # Check that cumulative percentages are reasonable
        assert pc_percentages[-1][1] <= 100


@pytest.mark.parametrize("show_projection_stats", [1, (1,), "foo"])
def test_projection_stats_invalid_type(
    sample_df: pd.DataFrame,
    show_projection_stats: Any,
) -> None:
    """Test that invalid show_projection_stats types raise TypeError."""
    with pytest.raises(
        TypeError, match=re.escape(f"{show_projection_stats=} must be bool or dict")
    ):
        pmv.cluster_compositions(
            df_in=sample_df,
            prop_name="property",
            embedding_method="one-hot",
            projection="pca",
            show_projection_stats=show_projection_stats,
        )


@pytest.mark.parametrize(
    ("embedding_method", "projection", "expected_shape", "expected_labels"),
    [
        ("magpie", "pca", (3,), ("Principal Component", "Principal Component")),
        ("one-hot", "tsne", (3,), ("t-SNE Component", "t-SNE Component")),
        ("deml", "pca", (3,), ("Principal Component", "Principal Component")),
        ("matminer", "isomap", (3,), ("Isomap Component", "Isomap Component")),
        (
            "matscholar_el",
            "kernel_pca",
            (3,),
            ("Kernel PCA Component", "Kernel PCA Component"),
        ),
    ],
)
def test_cluster_compositions_methods(
    sample_df: pd.DataFrame,
    embedding_method: str,
    projection: str,
    expected_shape: tuple[int, ...],
    expected_labels: tuple[str, str],
) -> None:
    """Test different combinations of embedding methods with PCA projection."""
    # Create a DataFrame with compositions and properties

    # Test 2D projection
    fig_2d = pmv.cluster_compositions(
        df_in=sample_df,
        composition_col="composition",
        prop_name="property",
        embedding_method=embedding_method,
        projection=projection,
        n_components=2,
    )

    # Check that we got a valid figure
    assert isinstance(fig_2d, go.Figure)
    assert len(fig_2d.data) == 1
    assert fig_2d.data[0].type == "scatter"
    assert fig_2d.data[0].x.shape == expected_shape
    assert fig_2d.data[0].y.shape == expected_shape
    assert not hasattr(fig_2d.data[0], "z")

    # Check axis labels
    assert fig_2d.layout.xaxis.title.text == f"{expected_labels[0]} 1"
    assert fig_2d.layout.yaxis.title.text == f"{expected_labels[1]} 2"

    # Some projection methods may produce NaN values when used with certain embeddings
    # and small number of points, so we'll skip 3D testing for those combinations
    known_problematic_combos = (
        ("one-hot", "tsne"),
        ("matminer", "isomap"),
        ("matscholar_el", "kernel_pca"),
    )
    if (embedding_method, projection) in known_problematic_combos:
        pytest.skip(f"Skip 3D test for {embedding_method}-{projection} combo")

    # Test 3D projection
    fig_3d = pmv.cluster_compositions(
        df_in=sample_df,
        composition_col="composition",
        prop_name="property",
        embedding_method=embedding_method,
        projection=projection,
        n_components=3,
    )

    # Check that we got a valid figure
    assert isinstance(fig_3d, go.Figure)
    assert len(fig_3d.data) == 1
    assert fig_3d.data[0].type == "scatter3d"
    assert fig_3d.data[0].x.shape == expected_shape
    assert fig_3d.data[0].y.shape == expected_shape
    assert fig_3d.data[0].z.shape == expected_shape

    # Verify z values have some variation (not all same value)
    var_z = np.var(fig_3d.data[0].z)
    assert var_z > 0

    # Check 3D axis labels
    assert fig_3d.layout.scene.xaxis.title.text == f"{expected_labels[0]} 1"
    assert fig_3d.layout.scene.yaxis.title.text == f"{expected_labels[1]} 2"
    assert fig_3d.layout.scene.zaxis.title.text == f"{expected_labels[0]} 3"


@pytest.mark.parametrize(
    ("custom_embedding", "custom_projection", "with_kwargs"),
    [
        (True, False, False),  # Only custom embedding
        (False, True, False),  # Only custom projection
        (True, True, False),  # Both custom functions
        (True, True, True),  # Both with custom kwargs
    ],
)
def test_custom_embedding_projection(
    sample_df: pd.DataFrame,
    custom_embedding: bool,
    custom_projection: bool,
    with_kwargs: bool,
) -> None:
    """Test using custom embedding and projection functions with optional kwargs."""

    def custom_embedding_func(
        compositions: list[str], scale: float = 1.0, **_kwargs: Any
    ) -> np.ndarray:
        """Custom embedding that just scales random values."""
        return np_rng.random((len(compositions), 10)) * scale

    def custom_projection_func(
        data: np.ndarray, n_components: int = 2, scale: float = 1.0, **_kwargs: Any
    ) -> np.ndarray:
        """Custom projection that just scales random values."""
        return np_rng.random((data.shape[0], n_components)) * scale

    # Set up the kwargs if requested
    embedding_kwargs = {"scale": 2.0} if with_kwargs else {}
    projection_kwargs = {"scale": 3.0} if with_kwargs else {}

    # Set the embedding and projection methods based on test parameters
    embedding_method = custom_embedding_func if custom_embedding else "one-hot"
    projection = custom_projection_func if custom_projection else "pca"

    # Create the figure
    fig = pmv.cluster_compositions(
        df_in=sample_df,
        prop_name="property",
        embedding_method=embedding_method,  # type: ignore[arg-type]
        projection=projection,
        embedding_kwargs=embedding_kwargs,
        projection_kwargs=projection_kwargs,
    )

    # Check that we got a valid figure
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)


def test_precomputed_embeddings() -> None:
    """Test passing pre-computed embeddings as dataframe column."""
    # Create sample compositions
    comp_formulas = ["Fe0.5Co0.5", "Ni0.7Cu0.3", "Zr0.6Ti0.4"]
    property_values = [1.0, 10.0, 100.0]
    embeddings = {comp: np_rng.random(10) for comp in comp_formulas}

    # Create DataFrame with pre-computed embeddings
    df_embed = pd.DataFrame({"composition": list(embeddings)})
    df_embed["property"] = property_values
    df_embed["embedding"] = list(embeddings.values())

    # Test with pre-computed embeddings
    fig = pmv.cluster_compositions(
        df_in=df_embed,
        composition_col="composition",
        prop_name="property",
        projection="pca",
    )

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].x.shape == (3,)
    assert fig.data[0].y.shape == (3,)


def test_marker_size_adjustment_3d(sample_df: pd.DataFrame) -> None:
    """Test that marker size is automatically adjusted for 3D plots."""
    # Test with 2D and 3D plots - default marker size should be scaled for 3D
    marker_size = 12

    for show_chem_sys, n_components in itertools.product(
        (None, "color", "shape", "color+shape"), (2, 3)
    ):
        fig = pmv.cluster_compositions(
            df_in=sample_df,
            prop_name="property",
            projection="pca",
            n_components=n_components,
            marker_size=marker_size,
            show_chem_sys=show_chem_sys,  # type: ignore[arg-type]
        )
        expected_size = marker_size / (3 if n_components == 3 else 1)
        for trace in fig.data:
            assert trace.marker.size == expected_size


@pytest.mark.parametrize(
    ("fixture_name", "sort_option", "viz_mode", "prop_name", "embedding_method"),
    [
        # Basic cases with different sort options
        ("hover_alignment_df", True, None, "property", "one-hot"),
        ("hover_alignment_df", False, None, "property", "one-hot"),
        ("hover_alignment_df", -1, None, "property", "one-hot"),
        ("hover_alignment_df", None, None, "property", "one-hot"),
        # Custom sort function
        pytest.param(
            "hover_alignment_df",
            lambda values: np.argsort(-np.array(values)),
            None,
            "property",
            "one-hot",
            id="custom_sort",
        ),
        # No property case
        ("hover_alignment_df", None, "color", None, "one-hot"),
        # With pre-computed embeddings
        ("df_with_embeddings", True, None, "property", "my_embeddings"),
        ("df_with_embeddings", False, "shape", "property", "my_embeddings"),
        ("df_with_embeddings", True, "color+shape", "property", "my_embeddings"),
    ],
)
def test_hover_text_alignment(
    request: pytest.FixtureRequest,
    fixture_name: str,
    sort_option: bool | int | None | Callable[[np.ndarray], np.ndarray],
    viz_mode: ShowChemSys | None,
    prop_name: str | None,
    embedding_method: str,
) -> None:
    """Test hover text alignment under various configurations."""
    # Get the dataframe from the fixture
    df_data = request.getfixturevalue(fixture_name)

    # Handle the special case of custom sort function test
    is_custom_sort = callable(sort_option)
    expected_order = None
    if is_custom_sort:
        # For the custom sort test, calculate expected order
        # (highest to lowest property values)
        expected_order = [
            comp
            for _, comp in sorted(  # Sort by negative property (descending)
                zip(df_data["property"], df_data["composition"], strict=True),
                key=lambda x: -x[0],
            )
        ]

    # Create figure with the specified configuration
    fig = pmv.cluster_compositions(
        df_in=df_data,
        composition_col="composition",
        prop_name=prop_name,
        embedding_method=embedding_method,
        projection="pca",
        sort=sort_option,  # type: ignore[arg-type]
        show_chem_sys=viz_mode,
    )

    # If using chemical system coloring with no property, data structure is different
    if viz_mode == "color" and prop_name is None:
        # For this case, we need to check all traces
        for trace in fig.data:
            custom_data = trace.customdata
            compositions = [item[0] for item in custom_data]
            hover_texts = [item[1] for item in custom_data]

            for comp, hover_text in zip(compositions, hover_texts, strict=True):
                # Verify composition in hover text
                comp_match = re.search(r"Composition: (.+?)<br>", hover_text)
                assert comp_match is not None
                hover_comp = comp_match.group(1)
                assert hover_comp == comp

                # Should have chemical system info instead of property
                assert "Chemical System:" in hover_text

        # No need for further checks in this case
        return

    # For all other cases, extract data from the first trace
    custom_data = fig.data[0].customdata
    compositions = [item[0] for item in custom_data]
    hover_texts = [item[1] for item in custom_data]

    # Check if the custom sort produced the expected order
    if is_custom_sort:
        assert compositions == expected_order

    # Verify each hover text's alignment with the original data
    for comp, hover_text in zip(compositions, hover_texts, strict=True):
        # Verify composition in hover text matches the one in custom_data
        comp_match = re.search(r"Composition: (.+?)<br>", hover_text)
        assert comp_match is not None
        hover_comp = comp_match.group(1)
        assert hover_comp == comp

        # If property is provided, verify it matches the original dataframe
        if prop_name:
            prop_match = re.search(r"property: (.+?)$", hover_text)
            assert prop_match is not None
            prop_str = prop_match.group(1)
            prop = float(prop_str)

            # Find the corresponding row in the original dataframe
            orig_row = df_data[df_data["composition"] == comp]
            assert len(orig_row) == 1

            # Verify property value matches
            expected_prop = orig_row["property"].iloc[0]
            assert abs(prop - expected_prop) < 1e-6


def test_hover_format_with_color_scales() -> None:
    """Test hover format interacts correctly with different color scales."""
    # Create data with varied values for testing
    df_test = pd.DataFrame(
        {
            "composition": ["Fe2O3", "Al2O3", "Cu", "SiO2", "TiO2"],
            "property": [0.123, 1.234, 12.345, 123.456, 1234.567],
        }
    )

    # Test different combinations of hover format and color scales
    hover_formats = [".1f", ".2f", ".3f", ".1e"]
    color_scales = ["linear", "log", "arcsinh"]

    for fmt in hover_formats:
        for scale in color_scales:
            # Skip log scale with negative values
            if scale == "log" and any(v <= 0 for v in df_test["property"]):
                continue

            fig = pmv.cluster_compositions(
                df_in=df_test,
                composition_col="composition",
                prop_name="property",
                projection="pca",
                color_scale=scale,  # type: ignore[arg-type]
                hover_format=fmt,  # Pass as string, not dict
            )

            # Extract hover text from customdata
            custom_data = fig.data[0].customdata
            hover_texts = [item[1] for item in custom_data]

            # Verify format string is applied correctly for each value
            for i, orig_val in enumerate(df_test["property"]):
                hover_text = hover_texts[i]

                # Extract formatted value from hover text
                match = re.search(r"property: ([\d.-]+(?:e[+-]\d+)?)", hover_text)
                assert match is not None, (
                    f"Couldn't find property value in hover text: {hover_text}"
                )

                formatted_val = match.group(1)

                # Compare with expected format
                # Just check length and general format matches
                if fmt == ".1f":
                    if orig_val < 10:
                        assert "." in formatted_val
                        assert len(formatted_val.split(".")[1]) <= 1
                    else:
                        assert "." in formatted_val or "e" in formatted_val.lower()
                elif fmt == ".2f":
                    if orig_val < 100:
                        assert "." in formatted_val
                        assert len(formatted_val.split(".")[1]) <= 2
                    else:
                        assert "." in formatted_val or "e" in formatted_val.lower()
                elif fmt == ".3f":
                    if orig_val < 1000:
                        assert "." in formatted_val
                        assert len(formatted_val.split(".")[1]) <= 3
                    else:
                        assert "." in formatted_val or "e" in formatted_val.lower()
                elif fmt == ".1e":
                    # Scientific notation should have e or E
                    assert "e" in formatted_val.lower() or "E" in formatted_val


def test_hover_text_formatting() -> None:
    """Test hover text formatting with various data types and edge cases."""
    # Create data with different types of property values
    df_test = pd.DataFrame(
        {
            "composition": ["Fe2O3", "Al2O3", "Cu", "SiO2", "TiO2", "ZnO"],
            "property_numeric": [
                0.123,
                12.345,
                1234.567,
                float("nan"),
                float("inf"),
                -float("inf"),
            ],
            "property_str": ["low", "medium", "high", "unknown", "extreme", "none"],
            "property_bool": [True, False, True, False, True, False],
        }
    )

    # Test numeric property with different hover formats
    for prop_name in ["property_numeric", "property_str", "property_bool"]:
        for fmt in [".1f", ".2f", ".0f", ".2e", "g"]:
            # Create plot with this property and format
            fig = pmv.cluster_compositions(
                df_in=df_test,
                composition_col="composition",
                prop_name=prop_name,
                projection="pca",
                hover_format=fmt,
            )

            # Extract hover text from customdata
            custom_data = fig.data[0].customdata
            hover_texts = [item[1] for item in custom_data]

            # Verify hover text contains the property name
            for hover_text in hover_texts:
                assert f"{prop_name}:" in hover_text, (
                    f"Hover text missing property name: {hover_text}"
                )

            # For numeric properties with NaN or Inf, ensure they're handled properly
            if prop_name == "property_numeric":
                # Check that NaN and Inf don't break formatting
                nan_idx = 3  # Index where NaN is located
                inf_idx = 4  # Index where Inf is located
                neg_inf_idx = 5  # Index where -Inf is located

                # Extract NaN/Inf hover texts
                nan_text = hover_texts[nan_idx]
                inf_text = hover_texts[inf_idx]
                neg_inf_text = hover_texts[neg_inf_idx]

                # NaN and Inf should still have the property name in the text
                assert f"{prop_name}:" in nan_text, (
                    f"NaN hover text missing property name: {nan_text}"
                )
                assert f"{prop_name}:" in inf_text, (
                    f"Inf hover text missing property name: {inf_text}"
                )
                assert f"{prop_name}:" in neg_inf_text, (
                    f"-Inf hover text missing property name: {neg_inf_text}"
                )

    # Test with pre-computed coordinates to ensure hover text includes coordinates
    coords_2d = np.random.default_rng(0).random((len(df_test), 2))
    df_test["coords"] = list(coords_2d)

    fig = pmv.cluster_compositions(
        df_in=df_test,
        composition_col="composition",
        prop_name="property_numeric",
        projection="coords",  # Use pre-computed coordinates
        hover_format=".2f",
    )

    custom_data = fig.data[0].customdata
    hover_texts = [item[1] for item in custom_data]

    # Verify hover text contains coordinate information
    for hover_text in hover_texts:
        assert "Component 1:" in hover_text, (
            f"Hover text missing coordinate info: {hover_text}"
        )
        assert "Component 2:" in hover_text, (
            f"Hover text missing coordinate info: {hover_text}"
        )

    # Test with custom hover format that has a specific prefix
    fig = pmv.cluster_compositions(
        df_in=df_test,
        composition_col="composition",
        prop_name="property_numeric",
        projection="pca",
        hover_format="$%.2f",  # Currency format
    )

    custom_data = fig.data[0].customdata
    hover_texts = [item[1] for item in custom_data]

    # Check for some formatted values that aren't NaN or Inf
    has_formatted_value = False
    for i, val in enumerate(df_test["property_numeric"]):
        if not pd.isna(val) and not np.isinf(val):
            # Should find the dollar sign in the hover text
            hover_text = hover_texts[i]
            if "$" in hover_text:
                has_formatted_value = True
                break

    assert has_formatted_value, (
        "No properly formatted currency values found in hover text"
    )


@pytest.mark.parametrize(
    ("test_case", "expected_assertions"),
    [
        # Basic hover formatting with different formats and color scales
        (
            {
                "data": [0.123, 1.234, 12.345, 123.456, 1234.567],
                "prop_name": "property",
                "hover_format": ".1f",
                "color_scale": "linear",
                "name": "basic_fmt_linear",
            },
            {"format_check": {"type": ".1f", "show_decimal": True}},
        ),
        (
            {
                "data": [0.123, 1.234, 12.345, 123.456, 1234.567],
                "prop_name": "property",
                "hover_format": ".2f",
                "color_scale": "log",
                "name": "basic_fmt_log",
            },
            {"format_check": {"type": ".2f", "show_decimal": True}},
        ),
        (
            {
                "data": [0.123, 1.234, 12.345, 123.456, 1234.567],
                "prop_name": "property",
                "hover_format": ".1e",
                "color_scale": "arcsinh",
                "name": "basic_fmt_arcsinh",
            },
            {"format_check": {"type": ".1e", "scientific": True}},
        ),
        # Special value handling with NaN and Inf
        (
            {
                "data": [
                    0.123,
                    12.345,
                    1234.567,
                    float("nan"),
                    float("inf"),
                    -float("inf"),
                ],
                "prop_name": "property_special",
                "hover_format": ".2f",
                "color_scale": "linear",
                "name": "special_values",
            },
            {
                "check_special_values": True,
                "nan_idx": 3,
                "inf_idx": 4,
                "neg_inf_idx": 5,
            },
        ),
        # Different property types
        (
            {
                "data": ["low", "medium", "high", "unknown", "extreme"],
                "prop_name": "property_str",
                "hover_format": ".2f",  # Format is ignored for strings
                "color_scale": "linear",
                "name": "string_property",
            },
            {"check_property_name": True},
        ),
        (
            {
                "data": [True, False, True, False, True],
                "prop_name": "property_bool",
                "hover_format": ".2f",  # Format is ignored for booleans
                "color_scale": "linear",
                "name": "boolean_property",
            },
            {"check_property_name": True},
        ),
        # Custom format with prefix
        (
            {
                "data": [0.123, 1.234, 12.345, 123.456, 1234.567],
                "prop_name": "property",
                "hover_format": "$%.2f",
                "color_scale": "linear",
                "name": "custom_prefix",
            },
            {"check_custom_prefix": "$"},
        ),
    ],
)
def test_hover_formatting(
    test_case: dict[str, Any], expected_assertions: dict[str, Any]
) -> None:
    """Parameterized test for hover text formatting scenarios."""
    # Common compositions for all tests
    compositions = ["Fe2O3", "Al2O3", "Cu", "SiO2", "TiO2", "ZnO", "MgO", "CaO"][
        : len(test_case["data"])
    ]

    # Create dataframe
    df_test = pd.DataFrame(
        {"composition": compositions, test_case["prop_name"]: test_case["data"]}
    )

    # Generate the plot
    fig = pmv.cluster_compositions(
        df_in=df_test,
        composition_col="composition",
        prop_name=test_case["prop_name"],
        projection="pca",
        hover_format=test_case["hover_format"],
        color_scale=test_case["color_scale"],
    )

    # Extract hover text from customdata
    custom_data = fig.data[0].customdata
    hover_texts = [item[1] for item in custom_data]

    # Basic check - all hover texts should include the property name
    for hover_text in hover_texts:
        assert f"{test_case['prop_name']}:" in hover_text, (
            f"Hover text missing property name: {hover_text}"
        )

    # Format-specific checks
    if "format_check" in expected_assertions:
        fmt_checks = expected_assertions["format_check"]

        # Only verify numeric formatting for numeric data
        if all(
            isinstance(val, (int, float))
            for val in test_case["data"]
            if not pd.isna(val) and not np.isinf(val)
        ):
            for i, val in enumerate(test_case["data"]):
                if pd.isna(val) or np.isinf(val):
                    continue

                hover_text = hover_texts[i]
                match = re.search(
                    rf"{test_case['prop_name']}: ([\d.-]+(?:e[+-]\d+)?)", hover_text
                )
                assert match is not None, (
                    f"Couldn't find property value in hover text: {hover_text}"
                )

                formatted_val = match.group(1)

                if fmt_checks.get("show_decimal", False):
                    assert "." in formatted_val, (
                        f"Expected decimal point in formatted value: {formatted_val}"
                    )

                if fmt_checks.get("scientific", False):
                    assert "e" in formatted_val.lower() or "E" in formatted_val, (
                        f"Expected scientific notation: {formatted_val}"
                    )

                # Check specific format types
                if fmt_checks.get("type") == ".1f" and val < 10:
                    if "." in formatted_val:
                        assert len(formatted_val.split(".")[1]) <= 1, (
                            f"Expected at most 1 decimal place: {formatted_val}"
                        )

                elif (
                    fmt_checks.get("type") == ".2f"
                    and val < 100
                    and "." in formatted_val
                ):
                    assert len(formatted_val.split(".")[1]) <= 2, (
                        f"Expected at most 2 decimal places: {formatted_val}"
                    )

    # Special values checks
    if expected_assertions.get("check_special_values", False):
        nan_idx = expected_assertions.get("nan_idx")
        inf_idx = expected_assertions.get("inf_idx")
        neg_inf_idx = expected_assertions.get("neg_inf_idx")

        if nan_idx is not None and nan_idx < len(hover_texts):
            assert f"{test_case['prop_name']}:" in hover_texts[nan_idx], (
                f"NaN hover text missing property name: {hover_texts[nan_idx]}"
            )

        if inf_idx is not None and inf_idx < len(hover_texts):
            assert f"{test_case['prop_name']}:" in hover_texts[inf_idx], (
                f"Inf hover text missing property name: {hover_texts[inf_idx]}"
            )

        if neg_inf_idx is not None and neg_inf_idx < len(hover_texts):
            assert f"{test_case['prop_name']}:" in hover_texts[neg_inf_idx], (
                f"Neg Inf hover text missing property name: {hover_texts[neg_inf_idx]}"
            )

    # Custom prefix check
    if "check_custom_prefix" in expected_assertions:
        prefix = expected_assertions["check_custom_prefix"]

        # Look for prefix in at least one hover text
        prefix_found = False
        for hover_text in hover_texts:
            if prefix in hover_text:
                prefix_found = True
                break

        assert prefix_found, f"Custom prefix '{prefix}' not found in any hover text"


# Test with pre-computed coordinates as a separate test to keep it focused
def test_hover_formatting_with_coords() -> None:
    """Test hover formatting with pre-computed coordinates."""
    # Create test data
    df_test = pd.DataFrame(
        {
            "composition": ["Fe2O3", "Al2O3", "Cu", "SiO2", "TiO2"],
            "property": [0.123, 1.234, 12.345, 123.456, 1234.567],
        }
    )

    # Add pre-computed coordinates
    coords_2d = np.random.default_rng(0).random((len(df_test), 2))
    df_test["coords"] = list(coords_2d)

    # Generate plot using the pre-computed coordinates
    fig = pmv.cluster_compositions(
        df_in=df_test,
        composition_col="composition",
        prop_name="property",
        projection="coords",  # Use pre-computed coordinates
        hover_format=".2f",
    )

    # Extract hover text
    custom_data = fig.data[0].customdata
    hover_texts = [item[1] for item in custom_data]

    # Verify hover text contains coordinate information
    for hover_text in hover_texts:
        assert "Component 1:" in hover_text, (
            f"Hover text missing coordinate info: {hover_text}"
        )
        assert "Component 2:" in hover_text, (
            f"Hover text missing coordinate info: {hover_text}"
        )
        assert "property:" in hover_text, (
            f"Hover text missing property info: {hover_text}"
        )


@pytest.mark.parametrize(
    ("projection", "expected_projector_type"),
    [("pca", PCA), ("tsne", TSNE), ("isomap", Isomap), ("kernel_pca", KernelPCA)],
)
def test_attached_projector_and_embeddings(
    sample_df: pd.DataFrame,
    projection: ProjectionMethod,
    expected_projector_type: type,
) -> None:
    """Test that projector and embeddings attributes are attached to the figure."""
    fig = pmv.cluster_compositions(
        df_in=sample_df,
        embedding_method="one-hot",
        projection=projection,
    )

    # Check that _pymatviz attribute exists and has the correct keys
    assert hasattr(fig, "_pymatviz"), "Figure missing _pymatviz attribute"
    assert "projector" in fig._pymatviz, "Missing projector in _pymatviz"
    assert "embeddings" in fig._pymatviz, "Missing embeddings in _pymatviz"

    # Check projector type
    assert isinstance(fig._pymatviz["projector"], expected_projector_type)

    # Check embeddings shape
    assert isinstance(fig._pymatviz["embeddings"], np.ndarray)
    assert fig._pymatviz["embeddings"].shape[0] == len(sample_df)

    # For PCA, also check that projector can transform data back and forth
    if projection == "pca":
        # Check that projector can transform new data correctly
        projector = fig._pymatviz["projector"]
        assert isinstance(projector, PCA)
        transformed = projector.transform(fig._pymatviz["embeddings"])
        assert transformed.shape == (len(sample_df), 2)

        # Try reconstructing the original data from the projection
        reconstructed = projector.inverse_transform(transformed)
        assert reconstructed.shape == fig._pymatviz["embeddings"].shape


def test_custom_projection_function_attributes(sample_df: pd.DataFrame) -> None:
    """Test projector and embeddings are attached when using custom projection func."""

    def custom_projection_func(  # For custom projection funcs, projector will be None
        data: np.ndarray, n_components: int = 2, **_kwargs: Any
    ) -> np.ndarray:
        """Custom projection function that returns only projected data."""
        return np_rng.random((data.shape[0], n_components))

    fig = pmv.cluster_compositions(
        df_in=sample_df,
        embedding_method="one-hot",
        projection=custom_projection_func,  # type: ignore[arg-type]
    )

    # Check that _pymatviz attribute exists and has the correct keys
    assert hasattr(fig, "_pymatviz"), "Figure missing _pymatviz attribute"
    assert "projector" in fig._pymatviz, "Missing projector in _pymatviz"
    assert "embeddings" in fig._pymatviz, "Missing embeddings in _pymatviz"

    # Check embeddings shape
    assert isinstance(fig._pymatviz["embeddings"], np.ndarray)
    assert fig._pymatviz["embeddings"].shape[0] == len(sample_df)

    # For custom projection function, projector should be None
    assert fig._pymatviz["projector"] is None


def test_precomputed_embeddings_attributes(sample_df: pd.DataFrame) -> None:
    """Test embeddings attribute is set correctly when using precomputed embeddings."""
    original_embeddings = np.array([np_rng.random(10) for _ in range(len(sample_df))])

    # Add precomputed embeddings to dataframe
    df_with_embeddings = sample_df.copy()
    df_with_embeddings["embedding"] = list(original_embeddings)

    fig = pmv.cluster_compositions(
        df_in=df_with_embeddings,
        embedding_method="embedding",  # Use the column name containing embeddings
        projection="pca",
    )

    # Check that _pymatviz attribute exists and has the correct keys
    assert hasattr(fig, "_pymatviz"), "Figure missing _pymatviz attribute"
    assert "projector" in fig._pymatviz, "Missing projector in _pymatviz"
    assert "embeddings" in fig._pymatviz, "Missing embeddings in _pymatviz"

    # Check that embeddings match the original embeddings
    assert np.allclose(fig._pymatviz["embeddings"], original_embeddings)

    # Check that projector is the correct type
    assert isinstance(fig._pymatviz["projector"], PCA)


@pytest.mark.parametrize(
    ("n_components", "marker_size", "custom_title"),
    [
        (2, 12, None),  # 2D, default title
        (2, 12, "Custom 2D Plot"),  # 2D, custom title
        (3, 12, None),  # 3D, default title
        (3, 12, "Custom 3D Plot"),  # 3D, custom title
    ],
)
def test_title_and_marker_size(
    sample_df: pd.DataFrame,
    n_components: int,
    marker_size: int,
    custom_title: str | None,
) -> None:
    """Test title, axis labels, marker sizes and coordinates in both 2D and 3D plots."""
    # Create the figure
    fig = pmv.cluster_compositions(
        df_in=sample_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        n_components=n_components,
        marker_size=marker_size,
        title=custom_title,
    )

    # Check figure type
    assert isinstance(fig, go.Figure)
    expected_trace_type = "scatter3d" if n_components == 3 else "scatter"
    assert fig.data[0].type == expected_trace_type

    # Check title
    if custom_title:
        assert fig.layout.title.text == custom_title

    # Check marker size (scaled down for 3D)
    expected_size = marker_size / (3 if n_components == 3 else 1)
    for trace in fig.data:
        assert trace.marker.size == expected_size

    # Check axis labels - different location for 2D vs 3D plots
    if n_components == 2:
        assert fig.layout.xaxis.title.text == "Principal Component 1"
        assert fig.layout.yaxis.title.text == "Principal Component 2"
        # Verify x and y values are present
        assert fig.data[0].x.shape == (len(sample_df),)
        assert fig.data[0].y.shape == (len(sample_df),)
        assert not hasattr(fig.data[0], "z")
    else:  # 3D
        assert fig.layout.scene.xaxis.title.text == "Principal Component 1"
        assert fig.layout.scene.yaxis.title.text == "Principal Component 2"
        assert fig.layout.scene.zaxis.title.text == "Principal Component 3"
        assert fig.data[0].z.shape == (len(sample_df),)
        # Check that z values are valid numbers (not None or NaN)
        n_nans = np.isnan(fig.data[0].z).sum()
        assert n_nans == 0, f"{n_nans=}"


@pytest.mark.parametrize(
    ("composition_data", "property_data", "prop_name", "expected_traces", "test_id"),
    [
        (  # Categorical data test
            ["Fe0.5Co0.5", "Ni0.7Cu0.3", "Zr0.6Ti0.4"],
            ["A", "B", "C"],
            "category",
            3,  # One trace per category
            "categorical",
        ),
        (  # NaN property values test
            ["Fe0.5Co0.5", "Ni0.7Cu0.3", "Zr0.6Ti0.4"],
            [1.0, np.nan, 100.0],
            "property",
            1,  # Single trace with NaN handled
            "nan_values",
        ),
        (  # Mixed composition types test
            ["Fe2O3", Composition("Al2O3"), "Cu"],
            [10.0, 20.0, 30.0],
            "property",
            1,  # Single trace
            "mixed_compositions",
        ),
    ],
)
def test_special_data_cases(
    composition_data: list[str | Composition],
    property_data: list[Any],
    prop_name: str,
    expected_traces: int,
    test_id: str,
) -> None:
    """Test special data cases: categorical properties, NaN values, mixed
    composition types.
    """
    df_data = pd.DataFrame({"composition": composition_data, prop_name: property_data})

    fig = pmv.cluster_compositions(
        df_in=df_data,
        composition_col="composition",
        prop_name=prop_name,
        embedding_method="one-hot",
        projection="pca",
    )

    # Check number of traces
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == expected_traces

    # For categorical data, check that each category has its own trace
    if test_id == "categorical":
        trace_names = {trace.name for trace in fig.data}
        categories = set(property_data)
        assert trace_names == categories

        # Each trace should have a unique color
        trace_colors = [trace.marker.color for trace in fig.data]
        assert len(set(trace_colors)) == len(trace_colors)

    # For NaN values, verify all points are plotted
    if test_id == "nan_values":
        # Check that all points are plotted (including NaN)
        assert len(fig.data[0].x) == len(composition_data)
        assert len(fig.data[0].y) == len(composition_data)

        # Check hover text for NaN value
        custom_data = fig.data[0].customdata
        hover_texts = [item[1] for item in custom_data]

        # At least one hover text should contain "nan" or "NaN"
        assert any("nan" in text.lower() for text in hover_texts)

    # For mixed composition types, check elements in each composition
    if test_id == "mixed_compositions":
        custom_data = fig.data[0].customdata
        compositions = [item[0] for item in custom_data]

        # Check each composition contains the expected elements
        expected_elements = [{"Fe", "O"}, {"Al", "O"}, {"Cu"}]
        for comp, expected in zip(compositions, expected_elements, strict=True):
            # Extract elements from composition formula
            elements = set(re.findall(r"[A-Z][a-z]*", comp))
            assert elements == expected


def test_precomputed_coordinates(sample_df: pd.DataFrame) -> None:
    """Test using pre-computed coordinates with cluster_compositions."""
    # Create some embeddings and get projected coordinates
    one_hot_embeddings = pmv.cluster.composition.embed.one_hot_encode(
        sample_df["composition"]
    )
    # Project using PCA to get coordinates
    pca = PCA(n_components=2, random_state=42)
    precomputed_coords = pca.fit_transform(one_hot_embeddings)

    # Add precomputed coordinates to dataframe
    df_with_coords = sample_df.copy()
    df_with_coords["coords"] = list(precomputed_coords)

    # Use these coordinates directly via column name
    fig = pmv.cluster_compositions(
        df_in=df_with_coords,
        prop_name="property",
        embedding_method="one-hot",  # Still need embeddings for hover info
        projection="coords",  # Use the coordinates from this column
    )

    # Check that the figure was created correctly
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1

    # Extract the x,y coordinates from the figure
    fig_coords = np.vstack((fig.data[0].x, fig.data[0].y)).T

    # Verify the coordinates match what we provided
    assert np.allclose(fig_coords, precomputed_coords)

    # Check metadata: when coordinates provided, only projector should be in metadata
    assert hasattr(fig, "_pymatviz")
    assert "projector" in fig._pymatviz
    assert fig._pymatviz["projector"] is None
    # Embeddings should not be calculated since coordinates were provided
    assert "embeddings" not in fig._pymatviz

    # Test with invalid coordinates shape (wrong number of components)
    df_wrong_shape = sample_df.copy()
    df_wrong_shape["coords"] = list(
        np_rng.random((len(sample_df), 3))
    )  # 3D coords for 2D plot
    with pytest.raises(ValueError, match="must have length 2"):
        pmv.cluster_compositions(
            df_in=df_wrong_shape,
            embedding_method="one-hot",
            projection="coords",
        )

    # Test with wrong type of data in coordinates column
    df_wrong_type = sample_df.copy()
    df_wrong_type["coords"] = ["not_array"] * len(sample_df)
    with pytest.raises(ValueError, match="must contain arrays or lists"):
        pmv.cluster_compositions(
            df_in=df_wrong_type,
            embedding_method="one-hot",
            projection="coords",
        )


@pytest.mark.parametrize("n_components", [2, 3])
def test_coordinates_priority(
    df_prop: pd.DataFrame, df_with_embeddings: pd.DataFrame, n_components: int
) -> None:
    """Test that provided coordinates take priority over embedding calculations."""
    # Create some embeddings and get projected coordinates using PCA using property_df
    one_hot_embeddings = pmv.cluster.composition.embed.one_hot_encode(
        df_prop["composition"]
    )
    pca = PCA(n_components=n_components, random_state=42)
    pca_coords = pca.fit_transform(one_hot_embeddings)
    assert pca_coords.shape == (len(df_prop), n_components)

    # Create different coordinates that match the df_with_embeddings size
    custom_coords = np_rng.random((len(df_with_embeddings), n_components))

    # Add custom coordinates to the DataFrame
    df_with_embeddings_and_coords = df_with_embeddings.copy()
    df_with_embeddings_and_coords["coords"] = list(custom_coords)

    # Use both precomputed embeddings and direct coordinates via column
    fig = pmv.cluster_compositions(
        df_in=df_with_embeddings_and_coords,
        prop_name="property",
        embedding_method="my_embeddings",  # Use column from fixture
        projection="coords",  # Use coordinates from this column
        n_components=n_components,  # Make sure to specify n_components
    )

    if n_components == 2:  # Extract coordinates from the figure
        fig_coords = np.vstack((fig.data[0].x, fig.data[0].y)).T
    else:
        fig_coords = np.vstack((fig.data[0].x, fig.data[0].y, fig.data[0].z)).T

    # coordinates might be in a different order, so need to check that each coordinate
    # in figure appears somewhere in custom_coords
    custom_coords_set = {tuple(coord) for coord in custom_coords}
    fig_coords_set = {tuple(coord) for coord in fig_coords}
    assert custom_coords_set == fig_coords_set

    # Check that metadata is correctly set
    assert hasattr(fig, "_pymatviz")
    assert "projector" in fig._pymatviz
    assert fig._pymatviz["projector"] is None
    # Embeddings should not be calculated or attached since coordinates were provided
    assert "embeddings" not in fig._pymatviz


@pytest.mark.parametrize("categorical", [True, False])
def test_coordinates_with_categorical_property(
    categorical_property_df: pd.DataFrame,
    df_prop: pd.DataFrame,
    custom_coords_2d: np.ndarray,
    categorical: bool,
) -> None:
    """Test that coordinates work correctly with categorical properties."""
    # Use correct fixture based on whether we need categorical or numerical properties
    df_cat = categorical_property_df if categorical else df_prop

    # Add custom coordinates to DataFrame
    df_with_coords = df_cat.copy()
    df_with_coords["coords"] = list(custom_coords_2d)

    # Create figure with custom coordinates
    fig = pmv.cluster_compositions(
        df_in=df_with_coords,
        prop_name="property",
        embedding_method="one-hot",
        projection="coords",  # Use coordinates from this column
    )

    # Check that figure was created correctly
    assert isinstance(fig, go.Figure)

    # For categorical data, we should have one trace per category
    if categorical:
        assert len(fig.data) == 3  # A, B, C categories
        # Check that each trace has the right number of points
        a_count = sum(p == "A" for p in df_cat["property"])
        b_count = sum(p == "B" for p in df_cat["property"])
        c_count = sum(p == "C" for p in df_cat["property"])

        # Find traces by name
        for trace in fig.data:
            if trace.name == "A":
                assert len(trace.x) == a_count
            elif trace.name == "B":
                assert len(trace.x) == b_count
            elif trace.name == "C":
                assert len(trace.x) == c_count
    else:
        # For continuous data, we should have one trace
        assert len(fig.data) == 1
        assert len(fig.data[0].x) == len(df_cat)

    # Verify embeddings are not calculated
    assert "embeddings" not in fig._pymatviz


def test_coordinates_with_show_chem_sys(
    df_comp: pd.DataFrame, custom_coords_2d: np.ndarray
) -> None:
    """Test that coordinates work correctly with show_chem_sys."""
    # Add custom coordinates to DataFrame
    df_with_coords = df_comp.copy()
    df_with_coords["coords"] = list(custom_coords_2d)

    # Test with show_chem_sys='color'
    fig = pmv.cluster_compositions(
        df_in=df_with_coords,
        embedding_method="one-hot",
        projection="coords",
        show_chem_sys="color",
    )

    # Check that chemical systems are included in the plot
    assert "chem_system" in fig.data[0].hovertemplate or "Chemical System" in str(
        fig.data[0].customdata
    )

    # Test with show_chem_sys='shape'
    fig = pmv.cluster_compositions(
        df_in=df_with_coords,
        embedding_method="one-hot",
        projection="coords",
        show_chem_sys="shape",
    )

    # Check that marker symbols are set
    assert hasattr(fig.data[0].marker, "symbol")
    # Symbol can be a list or tuple or array
    assert isinstance(fig.data[0].marker.symbol, (list, tuple)) or hasattr(
        fig.data[0].marker.symbol, "__len__"
    )

    # Test with show_chem_sys='color+shape'
    fig = pmv.cluster_compositions(
        df_in=df_with_coords,
        embedding_method="one-hot",
        projection="coords",
        show_chem_sys="color+shape",
    )

    # Check that both marker symbols and colors are set
    assert hasattr(fig.data[0].marker, "symbol")
    assert isinstance(fig.data[0].marker.symbol, (list, tuple)) or hasattr(
        fig.data[0].marker.symbol, "__len__"
    )
    assert hasattr(fig.data[0].marker, "color")


def test_precomputed_embeddings_in_composition_col() -> None:
    """Test using pre-computed embeddings directly in composition column."""
    # Create sample data with embeddings in the composition column
    embeddings = np_rng.random((5, 10))  # 5 samples, 10 features
    # Put embeddings directly in composition column
    df_emb = pd.DataFrame(
        {"composition": list(embeddings), "property": [1.0, 2.0, 3.0, 4.0, 5.0]}
    )

    # Use direct embeddings with PCA
    fig = pmv.cluster_compositions(
        df_in=df_emb,
        prop_name="property",
        embedding_method="one-hot",  # This should be ignored since we have embeddings
        projection="pca",
    )

    # Check that the figure was created successfully
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == len(df_emb)

    # Verify metadata
    assert "embeddings" in fig._pymatviz
    assert fig._pymatviz["embeddings"].shape == embeddings.shape

    # Verify embeddings are the same, though possibly reordered
    embeddings_set = {tuple(emb) for emb in embeddings}
    fig_embeddings_set = {tuple(emb) for emb in fig._pymatviz["embeddings"]}
    assert embeddings_set == fig_embeddings_set


def test_invalid_composition_type() -> None:
    """Test error with invalid composition types."""

    # Create sample data with invalid composition type
    class InvalidType:
        def __str__(self) -> str:
            return "Invalid"

    df_bad = pd.DataFrame({"composition": [InvalidType(), InvalidType()]})
    # Should raise TypeError when using a column with embeddings
    with pytest.raises(TypeError, match="Expected str or Composition"):
        pmv.cluster_compositions(
            df_in=df_bad,
            embedding_method="magpie",
            projection="pca",
        )


def test_method_labels_constant() -> None:
    """Test the method_labels constant has expected values."""
    from pymatviz.cluster.composition.plot import method_labels

    # Check that method_labels is a Final dictionary
    assert isinstance(method_labels, dict)

    # Check specific values in method_labels
    assert method_labels["pca"] == "Principal Component"
    assert method_labels["tsne"] == "t-SNE Component"
    assert method_labels["umap"] == "UMAP Component"
    assert method_labels["isomap"] == "Isomap Component"
    assert method_labels["kernel_pca"] == "Kernel PCA Component"


def test_invalid_n_components(df_comp: pd.DataFrame) -> None:
    """Test that invalid n_components raises ValueError."""
    with pytest.raises(ValueError, match="n_components=4 must be 2 or 3"):
        pmv.cluster_compositions(
            df_in=df_comp,
            embedding_method="one-hot",
            projection="pca",
            n_components=4,
        )


def test_invalid_show_projection_stats(df_comp: pd.DataFrame) -> None:
    """Test that invalid show_projection_stats raises TypeError."""
    with pytest.raises(TypeError, match="show_projection_stats="):
        pmv.cluster_compositions(
            df_in=df_comp,
            embedding_method="one-hot",
            projection="pca",
            show_projection_stats="invalid",  # type: ignore[arg-type]
        )


def test_invalid_embedding_method(df_comp: pd.DataFrame) -> None:
    """Test that invalid embedding_method raises ValueError."""
    with pytest.raises(ValueError, match="embedding_method="):
        pmv.cluster_compositions(
            df_in=df_comp,
            embedding_method="invalid_method",
            projection="pca",
        )


def test_too_many_chem_systems() -> None:
    """Test warning when there are too many chemical systems for shape markers."""
    # Create data with more chemical systems than available symbols
    compositions = [
        f"{el}2O3"
        for el in "Fe Al Cr Ga In Sc Y La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Cs Ba Rb Sr Fr Ra".split()  # noqa: SIM905, E501
    ]
    df_chem_sys = pd.DataFrame({"composition": compositions})

    # First check how many valid symbols we have
    from plotly.validator_cache import ValidatorCache
    symbolValidator = ValidatorCache().get_validator("scatter.marker", "symbol")

    all_symbols = symbolValidator.values  # noqa: PD011
    valid_symbols = [
        s
        for s in all_symbols
        if isinstance(s, str)
        and not s.isdigit()
        and not s.endswith("-dot")
        and "-open" not in s
    ]

    # Only run the test if we have enough different systems to exceed symbols
    if len(compositions) > len(valid_symbols):
        # Should warn about too many chemical systems
        msg = "Number of unique chemical systems .* exceeds available marker symbols"
        with pytest.warns(UserWarning, match=msg):
            pmv.cluster_compositions(
                df_in=df_chem_sys,
                embedding_method="one-hot",
                projection="pca",
                show_chem_sys="shape",
            )
    else:
        # If we can't trigger the warning, mark the test as skipped
        pytest.skip("Not enough unique chemical systems to exceed available symbols")


def test_custom_sort_function(sort_df: pd.DataFrame) -> None:
    """Test using a custom sort function with cluster_compositions."""

    # Define a custom sort function that sorts by odd/even property values
    def custom_sort(prop_values: list[float]) -> np.ndarray:
        return np.argsort([p % 2 for p in prop_values])

    # Create figure with custom sort function
    fig = pmv.cluster_compositions(
        df_in=sort_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        sort=custom_sort,
    )

    # Extract property values from the customdata
    hover_texts = [item[1] for item in fig.data[0].customdata]

    # Verify the sorting order based on our custom function
    # Even values should come before odd values
    property_pattern = r"property: ([\d.]+)"
    props_in_order = []

    for text in hover_texts:
        match = re.search(property_pattern, text)
        if match:
            props_in_order.append(float(match.group(1)))

    # Check the odd/even ordering - all even values should come before odd values
    even_indices = [idx for idx, prop in enumerate(props_in_order) if prop % 2 == 0]
    odd_indices = [idx for idx, prop in enumerate(props_in_order) if prop % 2 == 1]

    assert all(ei < oi for ei in even_indices for oi in odd_indices)


def test_custom_hover_format(float_precision_df: pd.DataFrame) -> None:
    """Test custom hover_format for hover labels."""
    # Test with default format (.2f)
    fig_default = pmv.cluster_compositions(
        df_in=float_precision_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
    )

    # Default should format to 2 decimal places
    hover_texts = [item[1] for item in fig_default.data[0].customdata]
    assert any("property: 1.23" in text for text in hover_texts)

    # Test with custom format (.4f)
    fig_custom = pmv.cluster_compositions(
        df_in=float_precision_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        hover_format=".4f",
    )

    # Custom should format to 4 decimal places
    hover_texts = [item[1] for item in fig_custom.data[0].customdata]
    assert any("property: 1.2345" in text for text in hover_texts)


def test_color_discrete_map(df_comp: pd.DataFrame) -> None:
    """Test custom color_discrete_map for chemical systems."""
    color_map = {
        "Fe-O": "red",
        "Al-O": "blue",
        "Cu": "green",
        "Au": "gold",
        "Ag": "silver",
    }
    fig = pmv.cluster_compositions(
        df_in=df_comp,
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys="color",
        color_discrete_map=color_map,  # type: ignore[arg-type]
    )

    # For "color" mode, there should be multiple traces with the custom colors
    trace_names = [trace.name for trace in fig.data if hasattr(trace, "name")]

    # Check that chemical systems are represented in the traces
    assert "Fe-O" in trace_names
    assert "Al-O" in trace_names
    assert "Cu" in trace_names
    assert "Au" in trace_names


def test_embeddings_from_column(df_with_embeddings: pd.DataFrame) -> None:
    """Test using embeddings from a DataFrame column."""
    # Get the embeddings array from the DataFrame for later comparison
    embeddings = np.array(df_with_embeddings["my_embeddings"].tolist())

    # Use embeddings from the specified column
    fig = pmv.cluster_compositions(
        df_in=df_with_embeddings,
        prop_name="property",
        embedding_method="my_embeddings",  # Use this column for embeddings
        projection="pca",
    )

    # Check that the figure was created successfully
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == len(df_with_embeddings)

    # Verify metadata
    fig_embeddings = fig._pymatviz["embeddings"]
    assert fig_embeddings.shape == embeddings.shape
    # Convert embeddings to a form we can compare (tuples for hashability)
    embeddings_set = {tuple(row) for row in embeddings}
    fig_embeddings_set = {tuple(row) for row in fig_embeddings}
    assert embeddings_set == fig_embeddings_set


def test_custom_projection_function(
    df_prop: pd.DataFrame,
    custom_projection_func: Callable[[np.ndarray, int, Any], np.ndarray],
) -> None:
    """Test using a custom projection function."""
    # Add some tracking to verify it's called
    custom_projection_func.called = False  # type: ignore[attr-defined]

    original_func = custom_projection_func

    def tracking_wrapper(
        embeddings: np.ndarray, n_components: int, **kwargs: Any
    ) -> np.ndarray:
        """Wrapper to track that the function was called."""
        original_func.called = True  # type: ignore[attr-defined]
        return original_func(embeddings, n_components, **kwargs)  # type: ignore[call-arg]

    # Use the custom projection function
    fig = pmv.cluster_compositions(
        df_in=df_prop,
        prop_name="property",
        embedding_method="one-hot",
        projection=tracking_wrapper,  # type: ignore[arg-type]
    )

    # Check that the figure was created successfully
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == len(df_prop)

    # Verify the custom projector was called
    assert original_func.called  # type: ignore[attr-defined]

    # Verify metadata
    assert "projector" in fig._pymatviz
    assert fig._pymatviz["projector"] is None  # Custom projectors store None


def test_property_colorbar(df_prop: pd.DataFrame) -> None:
    """Test that a colorbar is added for properties."""
    # Create figure with property coloring
    fig = pmv.cluster_compositions(
        df_in=df_prop,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
    )

    # Check that colorbar is configured with property name
    assert hasattr(fig.layout, "coloraxis")
    assert hasattr(fig.layout.coloraxis, "colorbar")
    assert fig.layout.coloraxis.colorbar.title.text == "property"
    assert fig.layout.coloraxis.colorbar.orientation == "h"
    assert fig.layout.coloraxis.colorbar.y == 0

    # Create figure without property coloring but with chemical system coloring
    fig_no_prop = pmv.cluster_compositions(
        df_in=df_prop,
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys="color",
    )

    # Check that colorbar doesn't have property title
    if hasattr(fig_no_prop.layout, "coloraxis") and hasattr(
        fig_no_prop.layout.coloraxis, "colorbar"
    ):
        assert fig_no_prop.layout.coloraxis.colorbar.title.text != "property"


def test_color_plus_shape_with_property(df_prop: pd.DataFrame) -> None:
    """Test color+shape mode with property coloring."""
    # Create figure with color+shape and property
    fig = pmv.cluster_compositions(
        df_in=df_prop,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        show_chem_sys="color+shape",
    )

    # Check that we have one trace with both property colors and chemical system shapes
    assert len(fig.data) == 1
    assert hasattr(fig.data[0].marker, "symbol")
    assert hasattr(fig.data[0].marker, "color")
    assert isinstance(fig.data[0].marker.symbol, (list, tuple)) or hasattr(
        fig.data[0].marker.symbol, "__len__"
    )

    # Verify that a colorbar is present for the property
    assert fig.layout.coloraxis.colorbar.title.text == "property"


@pytest.mark.parametrize(
    ("projection", "component_label"),
    [
        ("pca", "Principal Component"),
        ("tsne", "t-SNE Component"),
        ("isomap", "Isomap Component"),
        ("kernel_pca", "Kernel PCA Component"),
        # Custom function should use the function name or 'Component' if no proper name
        (
            lambda x, n_components, **_: np_rng.random((x.shape[0], n_components)),
            "Component",
        ),
        # Column name of coordinates should use generic "Component"
        ("coords", "Component"),
    ],
)
def test_coordinates_with_labels(
    df_prop: pd.DataFrame,
    custom_coords_2d: np.ndarray,
    custom_coords_3d: np.ndarray,
    projection: str | Callable[[np.ndarray, int, Any], np.ndarray],
    component_label: str,
) -> None:
    """Test that axis labels are set correctly when using precomputed coordinates."""
    # Create copies to add coordinates
    df_2d = df_prop.copy()
    df_3d = df_prop.copy()

    # Add coordinates to DataFrames if testing column of coordinates
    if projection == "coords":
        df_2d["coords"] = list(custom_coords_2d)
        df_3d["coords"] = list(custom_coords_3d)

    # Create figure with custom coordinates for 2D
    fig = pmv.cluster_compositions(
        df_in=df_2d,
        prop_name="property",
        embedding_method="one-hot",
        projection=projection,
        n_components=2,
    )

    # Check axis labels are correctly set based on projection
    assert fig.layout.xaxis.title.text == f"{component_label} 1"
    assert fig.layout.yaxis.title.text == f"{component_label} 2"

    # Skip 3D test for column coordinates if not provided properly
    if projection == "coords":
        # Test with 3D
        fig_3d = pmv.cluster_compositions(
            df_in=df_3d,
            prop_name="property",
            embedding_method="one-hot",
            projection=projection,
            n_components=3,
        )

        # Check 3D axis labels
        assert fig_3d.layout.scene.xaxis.title.text == f"{component_label} 1"
        assert fig_3d.layout.scene.yaxis.title.text == f"{component_label} 2"
        assert fig_3d.layout.scene.zaxis.title.text == f"{component_label} 3"

        # Verify embeddings are not calculated when using column
        assert "embeddings" not in fig_3d._pymatviz


@pytest.mark.parametrize(
    ("color_scale", "expected_title", "verify_ticks"),
    [
        ("linear", "property", False),  # Default scale
        ("log", "log scale", True),  # Log scale
        ("arcsinh", "arcsinh scale", True),  # Arcsinh scale
        # Custom arcsinh config
        ({"type": "arcsinh", "scale_factor": 1.5}, "property (arcsinh scale", True),
    ],
)
def test_color_scale_options(
    sample_df: pd.DataFrame,
    color_scale: str | dict[str, Any],
    expected_title: str,
    verify_ticks: bool,
) -> None:
    """Test different color scale options with appropriate data."""
    # Create data with wider range of property values to better test scaling
    df_test = sample_df.copy()
    df_test["property"] = [0.1, 10, 1000]  # Values spanning several orders of magnitude

    # For log scale, ensure no negative values
    if color_scale == "log" or (
        isinstance(color_scale, dict) and color_scale.get("type") == "log"
    ):
        pass  # Our test data is already positive

    fig = pmv.cluster_compositions(
        df_in=df_test,
        composition_col="composition",
        prop_name="property",
        projection="pca",
        color_scale=color_scale,  # type: ignore[arg-type]
    )

    # Verify colorbar title contains the expected text
    assert hasattr(fig.layout, "coloraxis")
    assert hasattr(fig.layout.coloraxis, "colorbar")
    if expected_title != "property":  # Non-linear scales modify the title
        assert expected_title in fig.layout.coloraxis.colorbar.title.text
    else:
        assert fig.layout.coloraxis.colorbar.title.text == "property"

    # Verify tick labels match original data range if needed
    if (
        verify_ticks
        and getattr(fig.layout.coloraxis.colorbar, "ticktext", None) is not None
    ):
        tick_text_values = [
            float(t)
            for t in fig.layout.coloraxis.colorbar.ticktext
            if (isinstance(t, str) and t.replace(".", "").replace("-", "").isdigit())
            or isinstance(t, (int, float))
        ]

        # Verify that min and max values are covered by the colorbar ticks
        if tick_text_values:
            if color_scale == "log" or (
                isinstance(color_scale, dict) and color_scale.get("type") == "log"
            ):
                # Log scale omits values <= 0, so we only check that maximum is covered
                assert max(tick_text_values) >= 100.0, (
                    f"Max tick value {max(tick_text_values)} doesn't cover "
                    f"data range [0.1, 1000.0]"
                )
            elif color_scale == "arcsinh" or (
                isinstance(color_scale, dict) and color_scale.get("type") == "arcsinh"
            ):
                # arcsinh scales can handle negative values, but our test data is +ve
                # We still check that maximum is covered
                assert max(tick_text_values) >= 100


@pytest.mark.parametrize(
    ("test_case", "expected_error"),
    [
        (
            {"color_scale": "invalid"},
            "color_scale='invalid' must be one of ('linear', 'log', 'arcsinh') or dict",
        ),
        (
            {"color_scale": {"lin_thresh": 0.5}},  # Missing 'type'
            "When color_scale is a dict, 'type' key must be provided",
        ),
        (
            {
                "color_scale": {"type": "arcsinh", "scale_factor": -1.0}
            },  # Negative scale_factor
            "scale_factor=-1.0 must be positive for arcsinh scale",
        ),
    ],
)
def test_color_scale_errors(
    sample_df: pd.DataFrame, test_case: dict[str, Any], expected_error: str
) -> None:
    """Test error handling for various invalid color scale configurations."""
    with pytest.raises(ValueError, match=re.escape(expected_error)):
        pmv.cluster_compositions(
            df_in=sample_df,
            composition_col="composition",
            prop_name="property",
            projection="pca",
            **test_case,
        )


@pytest.mark.parametrize(
    ("color_scale", "min_expected_ticks"),
    [
        ("linear", 2),  # Linear scales might have only 2 ticks in some cases
        ("log", 5),  # We want at least 5 ticks for log scale
        ("arcsinh", 5),  # We want at least 5 ticks for arcsinh scale
        (
            {"type": "arcsinh", "lin_thresh": 0.5, "lin_scale": 0.8},
            5,
        ),  # Custom arcsinh config
        (
            {"type": "arcsinh", "scale_factor": 1.5},
            5,
        ),  # Custom arcsinh config
    ],
)
def test_colorbar_tick_count(
    color_scale: str | dict[str, Any], min_expected_ticks: int
) -> None:
    """Test that colorbars have at least the minimum number of tick labels."""
    # Create data with wide range of values spanning several orders of magnitude
    df_test = pd.DataFrame(
        {
            "composition": ["Fe2O3", "Al2O3", "Cu", "FeSiO3", "CaTiO3", "ZnO"],
            "property": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        }
    )

    # For log scale, ensure no negative values
    if color_scale == "log" or (
        isinstance(color_scale, dict) and color_scale.get("type") == "log"
    ):
        # Our test data already has only positive values
        pass

    fig = pmv.cluster_compositions(
        df_in=df_test,
        composition_col="composition",
        prop_name="property",
        projection="pca",
        color_scale=color_scale,  # type: ignore[arg-type]
    )

    # Verify colorbar has tick properties
    assert hasattr(fig.layout, "coloraxis")
    assert hasattr(fig.layout.coloraxis, "colorbar")

    # Check that we have a sufficient number of tick labels
    if hasattr(fig.layout.coloraxis.colorbar, "ticktext"):
        # Get actual ticktext values
        ticktext = fig.layout.coloraxis.colorbar.ticktext

        # Handle case where ticktext might be None
        if ticktext is None:  # check if there are tickvals instead
            assert hasattr(fig.layout.coloraxis.colorbar, "tickvals")
            tickvals = fig.layout.coloraxis.colorbar.tickvals
            if tickvals is not None:
                tick_count = len(tickvals)
                assert tick_count >= min_expected_ticks
        else:
            tick_count = len(ticktext)
            assert tick_count >= min_expected_ticks


def test_varied_data_range_tick_formatting() -> None:
    """Test colorbar tick formatting with various data ranges."""
    # List of valid compositions to use
    valid_compositions = "Fe2O3 Al2O3 Cu FeSiO3 CaTiO3 ZnO MgO SiO2 TiO2 ZnS".split()  # noqa: SIM905

    # Test different data ranges
    test_ranges: list[list[float]] = [
        # Narrow range
        [0.95, 1.0, 1.05, 1.1, 1.15],
        # Multi-decade range
        [0.001, 0.1, 10, 1000],
        # Negative range
        [-100, -50, -10, -1],
    ]

    for data_range in test_ranges:
        # Create DataFrame with valid compositions
        comps = valid_compositions[: len(data_range)]
        df_prop = pd.DataFrame({"composition": comps, "property": data_range})

        # Calculate the range span as a ratio for use in tests
        range_min = min(data_range)
        range_max = max(data_range)
        range_span_ratio = (
            range_max / abs(range_min) if range_min != 0 else float("inf")
        )

        for scale_type in ("linear", "log", "arcsinh"):
            # Skip log scale for negative data
            if scale_type == "log" and any(v <= 0 for v in data_range):
                continue

            # Skip log tests for very narrow ranges where they don't make sense
            if scale_type == "log" and range_span_ratio < 2.0:
                # Too narrow for log scale to show meaningful ticks
                continue

            fig = pmv.cluster_compositions(
                df_in=df_prop,
                composition_col="composition",
                prop_name="property",
                projection="pca",
                color_scale=scale_type,
            )

            # Verify we have tick labels
            if hasattr(fig.layout.coloraxis.colorbar, "ticktext"):
                ticktext = fig.layout.coloraxis.colorbar.ticktext

                # Handle case where ticktext might be None
                if ticktext is not None:
                    tick_count = len(ticktext)

                    # Adjust expectations based on range span
                    if scale_type == "log":
                        # For log scale, number of ticks depends on decade span
                        decades_spanned = np.log10(range_max) - np.log10(range_min)
                        min_expected = max(2, min(5, int(decades_spanned * 2) + 1))
                    elif scale_type == "arcsinh":
                        # For arcsinh scale, we generally want at least 5 ticks
                        # But for ranges with moderate span, accept fewer
                        min_expected = 3 if range_span_ratio < 20.0 else 5
                    else:
                        # For linear scale, 2 ticks minimum is reasonable
                        min_expected = 2

                    assert tick_count >= min_expected, (
                        f"{min_expected=} ticks for {scale_type=} with {range_min=} - "
                        f"{range_max=} ({range_span_ratio=:.1f})"
                    )


def test_log_scale_nice_tick_values() -> None:
    """Test that log scale colorbar has nice round numbers for tick labels."""
    # Create data with wide range of values
    df_prop = pd.DataFrame(
        {"composition": ["Fe2O3", "Al2O3", "Cu", "FeSiO3", "CaTiO3", "ZnO"]}
    )
    df_prop["property"] = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]  # Nice round numbers

    fig = pmv.cluster_compositions(
        df_in=df_prop,
        composition_col="composition",
        prop_name="property",
        projection="pca",
        color_scale="log",
    )

    # Check that tick values are nicely formatted
    if hasattr(fig.layout.coloraxis.colorbar, "ticktext"):
        ticktext = fig.layout.coloraxis.colorbar.ticktext

        # Handle case where ticktext might be None
        if ticktext is None:
            pytest.skip("No ticktext available to check")

        # Verify some ticks match the nice round values in our data
        nice_values = ["0.01", "0.1", "1", "10", "100", "1000"]
        nice_values_scientific = ["1e-02", "1e-01", "1e+00", "1e+01", "1e+02", "1e+03"]

        # Check that at least 3 of our nice values appear in the tick labels
        matches = sum(
            any(
                nv in tt or sci_nv in tt
                for nv, sci_nv in zip(nice_values, nice_values_scientific, strict=True)
            )
            for tt in ticktext
        )

        assert matches >= 3, (
            f"Expected at least 3 nice round tick values, found {matches}"
        )


def test_arcsinh_scale_tick_formatting() -> None:
    """Test that arcsinh scale colorbar has properly formatted tick labels for
    positive, negative and near-zero values.
    """
    # Create data with positive, negative and near-zero values
    comps = "Fe2O3 Al2O3 FeSiO3 CaTiO3 ZnO MgO SiO2 TiO2 ZnS".split()  # noqa: SIM905
    props = [-100.0, -10.0, -1.0, -0.1, 0, 0.1, 1.0, 10.0, 100.0]
    df_prop = pd.DataFrame({"composition": comps, "property": props})

    fig = pmv.cluster_compositions(
        df_in=df_prop,
        composition_col="composition",
        prop_name="property",
        projection="pca",
        color_scale="arcsinh",
    )

    # Check that tick values cover different regions
    if hasattr(fig.layout.coloraxis.colorbar, "ticktext"):
        ticktext = fig.layout.coloraxis.colorbar.ticktext

        # Handle case where ticktext might be None
        if ticktext is None:
            pytest.skip("No ticktext available to check")

        # Convert tick texts to floats where possible
        tick_values = []
        for tt in ticktext:
            if "e" in tt.lower():
                tick_values.append(float(tt.replace("e", "E")))
            else:
                tick_values.append(float(tt))

        # Verify we have ticks in different regions
        has_negative = any(v < 0 for v in tick_values)
        has_positive = any(v > 0 for v in tick_values)

        assert has_negative, "Expected negative tick values for arcsinh scale"
        assert has_positive, "Expected positive tick values for arcsinh scale"

        # Check for nice round numbers (-100, -10, -1, 0, 1, 10, 100)
        # Allow small differences due to formatting precision
        nice_values = [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]
        has_nice_values = sum(
            any(abs(v - nv) < 0.01 * max(1, abs(nv)) for nv in nice_values)
            for v in tick_values
        )

        assert has_nice_values >= 3


def test_scatter_colors_match_colorbar_values() -> None:
    """Test that scatter point colors match the colorbar values and ticks.

    This test verifies that a point with a certain property value gets the correct
    color according to the colorbar, regardless of color scale type.
    """
    # Create a dataframe with known values
    df_prop = pd.DataFrame(
        {
            "composition": ["Fe2O3", "Al2O3", "Cu", "SiO2", "TiO2"],
            "property": [0.5, 1.0, 2.0, 5.0, 10.0],  # Nice incremental values
        }
    )

    # Test each color scale type
    for scale_type in ["linear", "arcsinh"]:
        # Skip log scale for this test to simplify
        if scale_type == "log":
            continue

        # Create the plot
        fig = pmv.cluster_compositions(
            df_in=df_prop,
            composition_col="composition",
            prop_name="property",
            projection="pca",
            color_scale=scale_type,  # type: ignore[arg-type]
        )

        # Get the scatter trace with the color mapping
        scatter_trace = fig.data[0]

        # Extract the color values (z-values) for each point
        point_values = scatter_trace.marker.color

        # Extract coloraxis information from the layout
        if hasattr(fig.layout, "coloraxis"):
            # Verify that coloraxis exists and has cmin/cmax set
            if (
                getattr(fig.layout.coloraxis, "cmin", None) is not None
                and getattr(fig.layout.coloraxis, "cmax", None) is not None
            ):
                # Simply verify that color values are within the colorbar range
                assert min(point_values) >= fig.layout.coloraxis.cmin - 1e-6
                assert max(point_values) <= fig.layout.coloraxis.cmax + 1e-6

            # Now verify points with different property values have different colors
            # This ensures the color mapping is monotonic
            sorted_indices = np.argsort(df_prop["property"])
            sorted_colors = [point_values[i] for i in sorted_indices]

            # Check that colors increase monotonically with property values
            for idx in range(len(sorted_colors) - 1):
                # Allow for a small tolerance in floating point comparison
                assert sorted_colors[idx] <= sorted_colors[idx + 1] + 1e-6


def test_arcsinh_colorbar_matches_data_values() -> None:
    """Test specifically targeting the arcsinh scale to ensure colors match values.

    This tests that a point with a specific value is correctly colored
    according to the colorbar scale.
    """
    # Create a test dataframe with specific values around the problematic area
    df_prop = pd.DataFrame(
        {
            "composition": ["P4H12", "SiO2", "ZnO", "Fe2O3", "TiO2"],
            "refractive_index": [1.87, 2.0, 5.0, 10.0, 20.0],
        }
    )

    # Create the plot with arcsinh scale
    fig = pmv.cluster_compositions(
        df_in=df_prop,
        composition_col="composition",
        prop_name="refractive_index",
        projection="pca",
        color_scale="arcsinh",
    )

    # Get the scatter trace
    scatter = fig.data[0]

    # Get the color values for each point
    color_values = scatter.marker.color

    # Check that the point with value 1.87 (P4H12) has the correct color
    p4h12_idx = 0
    p4h12_value = df_prop["refractive_index"][p4h12_idx]

    # The value 1.87 should be less than 2.0
    assert p4h12_value < 2.0, f"P4H12 value {p4h12_value} should be less than 2.0"

    # Find the index of the data value 2.0 in our dataframe (should be index 1)
    val_2_idx = 1  # SiO2 has value 2.0

    # Now let's verify that the two points get different colors
    # We just need to check that the color values array correctly preserves the ordering
    p4h12_color = color_values[p4h12_idx]
    val_2_color = color_values[val_2_idx]

    # The color value for 1.87 should be less than color value for 2.0
    # This ensures proper monotonic color mapping
    assert p4h12_color < val_2_color, (
        f"Color value for 1.87 ({p4h12_color}) should be "
        f"less than color value for 2.0 ({val_2_color})"
    )


def test_hover_tooltip_shows_original_values() -> None:
    """Test that hover tooltips show original property values, not transformed
    log/arcsinh values.
    """
    df_prop = pd.DataFrame(
        {
            "composition": ["Fe2O3", "Al2O3", "Cu", "SiO2", "TiO2"],
            # Wide range good for log scale
            "property": [0.5, 1.0, 10.0, 100.0, 1000.0],
        }
    )

    # Test each scale type
    for scale_type in ["log", "arcsinh"]:
        # Create the plot with the specified scale
        fig = pmv.cluster_compositions(
            df_in=df_prop,
            composition_col="composition",
            prop_name="property",
            projection="pca",
            color_scale=scale_type,  # type: ignore[arg-type]
        )

        # Extract hover text from customdata
        custom_data = fig.data[0].customdata
        hover_texts = [item[1] for item in custom_data]

        # For each data point, verify the hover text shows the original property value
        for i, orig_val in enumerate(df_prop["property"]):
            # Extract property value from hover text
            hover_text = hover_texts[i]
            match = re.search(r"property: ([\d.-]+)", hover_text)
            assert match is not None, (
                f"Couldn't find property value in hover text: {hover_text}"
            )

            hover_val = float(match.group(1))

            # Verify the hover value is close to the original, not the transformed value
            assert abs(hover_val - orig_val) < 1e-6

            # Specifically make sure it's NOT showing the transformed value
            if scale_type == "log":
                transformed_val = np.log10(orig_val)
                assert abs(hover_val - transformed_val) > 0.1

        # Also test with a custom configuration
        if scale_type == "arcsinh":
            # Test custom arcsinh configuration
            fig = pmv.cluster_compositions(
                df_in=df_prop,
                composition_col="composition",
                prop_name="property",
                projection="pca",
                color_scale={"type": "arcsinh", "scale_factor": 1.5},
            )

            # Extract hover text again
            custom_data = fig.data[0].customdata
            hover_texts = [item[1] for item in custom_data]

            # Verify values again
            for i, orig_val in enumerate(df_prop["property"]):
                hover_text = hover_texts[i]
                match = re.search(r"property: ([\d.-]+)", hover_text)
                assert match is not None
                hover_val = float(match.group(1))

                # Should match original value
                assert abs(hover_val - orig_val) < 1e-6


def test_log_scale_small_range() -> None:
    """Test log scale with a small range of values to check tight tick generation."""
    # Create data with small range of values
    df_prop = pd.DataFrame(  # Small range within same decade
        {"composition": ["Fe2O3", "Al2O3", "Cu"], "property": [1.2, 2.3, 3.4]}
    )

    fig = pmv.cluster_compositions(
        df_in=df_prop,
        composition_col="composition",
        prop_name="property",
        projection="pca",
        color_scale="log",
    )

    # Check tick formatting and density
    if hasattr(fig.layout.coloraxis.colorbar, "ticktext"):
        ticktext = fig.layout.coloraxis.colorbar.ticktext

        # Handle case where ticktext might be None
        if ticktext is None:
            pytest.skip("No ticktext available to check")

        # Should have at least 3 ticks even for small range (reduced from 5)
        assert len(ticktext) >= 3, f"Expected at least 3 ticks but got {len(ticktext)}"

        # Check for intermediate ticks (not just 1, 2, 3): should see some more precise
        # values like 1.5, 2.5 for small ranges
        tick_values = []
        for tt in ticktext:
            # Remove any spaces in tick text
            t_clean = tt.replace(" ", "")
            if "e" in t_clean.lower():
                tick_values.append(float(t_clean.replace("e", "E")))
            else:
                tick_values.append(float(t_clean))

        # For small ranges, we should have some tick values in our data range
        data_min, data_max = 1.2, 3.4
        has_values_in_range = any(data_min <= v <= data_max for v in tick_values)
        assert has_values_in_range, "Expected tick values within data range"


def test_arcsinh_zero_crossing_data() -> None:
    """Test arcsinh scale with zero-crossing data to check proper tick generation."""
    # Create data with values crossing zero with both small and large magnitudes
    df_prop = pd.DataFrame(
        {
            "composition": ["Fe2O3", "Al2O3", "Cu", "SiO2", "TiO2", "ZnO"],
            # Cross zero, with large range
            "property": [-100.0, -1.0, -0.1, 0.1, 1.0, 100.0],
        }
    )

    fig = pmv.cluster_compositions(
        df_in=df_prop,
        composition_col="composition",
        prop_name="property",
        projection="pca",
        color_scale="arcsinh",
    )

    # Verify colorbar has zero tick and both positive and negative regions
    if hasattr(fig.layout.coloraxis.colorbar, "ticktext"):
        ticktext = fig.layout.coloraxis.colorbar.ticktext

        # Handle case where ticktext might be None
        if ticktext is None:
            pytest.skip("No ticktext available to check")

        # Convert tick texts to floats where possible
        tick_values = []
        for tt in ticktext:
            # Handle scientific notation
            if "e" in tt.lower():
                tick_values.append(float(tt.replace("e", "E")))
            else:
                tick_values.append(float(tt))

        # Verify we have ticks in different regions: some negative and positive values
        has_negative = any(v < 0 for v in tick_values)
        has_positive = any(v > 0 for v in tick_values)

        # Check coverage of different data regions
        assert has_negative, "Missing negative tick values"
        assert has_positive, "Missing positive tick values"

        # Check that we have at least one value close to our data points
        data_points = [-100.0, -1.0, -0.1, 0.1, 1.0, 100.0]
        has_matching_ticks = any(
            any(abs(v - dp) / max(1.0, abs(dp)) < 0.1 for dp in data_points)
            for v in tick_values
        )
        assert has_matching_ticks, "No tick values close to any data points"


def test_arcsinh_custom_parameters() -> None:
    """Test arcsinh scale with custom parameters for lin_thresh and lin_scale."""
    # Create data with varied values for testing
    df_prop = pd.DataFrame(
        {"composition": ["Fe2O3", "Al2O3", "Cu", "SiO2", "TiO2", "ZnO"]}
    )
    df_prop["property"] = [-10.0, -5.0, -0.5, 0.5, 5.0, 10.0]

    # Test three different customizations
    custom_configs = [
        {"type": "arcsinh", "lin_thresh": 0.1, "lin_scale": 0.5},  # Small linear region
        {"type": "arcsinh", "lin_thresh": 2.0, "lin_scale": 1.0},  # Med linear region
        {"type": "arcsinh", "lin_thresh": 6.0, "lin_scale": 2.0},  # Large linear region
    ]

    for config in custom_configs:
        fig = pmv.cluster_compositions(
            df_in=df_prop,
            composition_col="composition",
            prop_name="property",
            projection="pca",
            color_scale=config,
        )
        title_text = fig.layout.coloraxis.colorbar.title.text
        assert "(arcsinh scale)" in title_text

        # Check that tick values are generated (but don't check specific values)
        if hasattr(fig.layout.coloraxis.colorbar, "ticktext"):
            ticktext = fig.layout.coloraxis.colorbar.ticktext
            assert ticktext is not None, "Tick text should be present"
            assert len(ticktext) > 0, "Should have at least some tick values"


@pytest.mark.parametrize(
    "test_case",
    [
        {  # Zero crossing arcsinh case
            "name": "zero_crossing_arcsinh",
            "data_values": [-100.0, -1.0, -0.1, 0.1, 1.0, 100.0],
            "color_scale": "arcsinh",
            "assertions": {
                "has_negative_ticks": True,
                "has_positive_ticks": True,
                "has_matching_ticks": True,
            },
        },
        {  # Modified tests for arcsinh custom params - removing specific boundary check
            "name": "custom_arcsinh_small",
            "data_values": [-10.0, -5.0, -0.5, 0.5, 5.0, 10.0],
            "color_scale": {"type": "arcsinh", "lin_thresh": 0.1, "lin_scale": 0.5},
            "assertions": {"title_contains": "property (arcsinh scale"},
        },
        {
            "name": "custom_arcsinh_medium",
            "data_values": [-10.0, -5.0, -0.5, 0.5, 5.0, 10.0],
            "color_scale": {"type": "arcsinh", "lin_thresh": 2.0, "lin_scale": 1.0},
            "assertions": {"title_contains": "property (arcsinh scale"},
        },
        {
            "name": "custom_arcsinh_large",
            "data_values": [-10.0, -5.0, -0.5, 0.5, 5.0, 10.0],
            "color_scale": {"type": "arcsinh", "lin_thresh": 6.0, "lin_scale": 2.0},
            "assertions": {"title_contains": "property (arcsinh scale"},
        },
        {  # Add more test cases that don't rely on specific tick values
            "name": "wide_range_arcsinh",
            "data_values": [0.0001, 0.01, 1.0, 100.0, 10000.0],
            "color_scale": "arcsinh",
            "assertions": {"min_tick_count": 5, "has_positive_ticks": True},
        },
        {  # very small arcsinh
            "name": "very_small_arcsinh",
            "data_values": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
            "color_scale": "arcsinh",
            "assertions": {"has_positive_ticks": True},
        },
        {  # very large arcsinh
            "name": "very_large_arcsinh",
            "data_values": [1e5, 1e6, 1e7, 1e8, 1e9],
            "color_scale": "arcsinh",
            "assertions": {"has_positive_ticks": True},
        },
        {  # mixed signs arcsinh
            "name": "mixed_signs_arcsinh",
            "data_values": [-1000.0, -10.0, -0.1, 0.0, 0.1, 10.0, 1000.0],
            "color_scale": "arcsinh",
            "assertions": {"has_negative_ticks": True, "has_positive_ticks": True},
        },
        {  # near zero arcsinh
            "name": "near_zero_arcsinh",
            "data_values": [-0.01, -0.001, -0.0001, 0.0, 0.0001, 0.001, 0.01],
            "color_scale": "arcsinh",
            "assertions": {"has_negative_ticks": True, "has_positive_ticks": True},
        },
        {  # negative large arcsinh
            "name": "negative_large_arcsinh",
            "data_values": [-10000.0, -1000.0, -100.0, -10.0, -1.0],
            "color_scale": "arcsinh",
            "assertions": {"has_negative_ticks": True},
        },
        {  # Modified custom arcsinh tests - only check title contains
            "name": "custom_arcsinh_tiny_thresh",
            "data_values": [-10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0],
            "color_scale": {
                "type": "arcsinh",
                "lin_thresh": 0.01,
                "lin_scale": 0.5,
            },
            "assertions": {"title_contains": "property (arcsinh scale"},
        },
        {  # custom_arcsinh_huge_thresh
            "name": "custom_arcsinh_huge_thresh",
            "data_values": [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0],
            "color_scale": {
                "type": "arcsinh",
                "lin_thresh": 50.0,
                "lin_scale": 2.0,
            },
            "assertions": {"title_contains": "property (arcsinh scale"},
        },
        {  # one_decade_arcsinh
            "name": "one_decade_arcsinh",
            "data_values": [1.0, 2.0, 3.0, 5.0, 8.0, 10.0],
            "color_scale": "arcsinh",
            "assertions": {"has_positive_ticks": True},
        },
    ],
)
def test_colorscale_tick_generation(test_case: dict[str, Any]) -> None:
    """Parameterized test for various color scale tick generation scenarios."""
    # Create a DataFrame from the test data
    test_name = test_case["name"]

    if "data_values" in test_case:
        # Use the new property name in updated tests
        data_values = test_case["data_values"]
    elif "data" in test_case:
        # Support the old property name for backwards compatibility
        data_values = test_case["data"]
    else:
        raise ValueError(f"Test case {test_name} must have data_values")

    element_list = "Fe2O3 Al2O3 Cu SiO2 TiO2 ZnO MgO CaO Na2O K2O".split()  # noqa: SIM905
    df_test = pd.DataFrame(
        {"composition": element_list[: len(data_values)], "property": data_values}
    )

    # Create the plot
    color_scale = test_case["color_scale"]
    assertions = test_case.get("assertions", {})

    fig = pmv.cluster_compositions(
        df_in=df_test,
        composition_col="composition",
        prop_name="property",
        projection="pca",
        color_scale=color_scale,
    )

    # Check that the colorbar has the expected title
    if "title_contains" in assertions:
        expected_title_fragment = assertions["title_contains"]
        actual_title = fig.layout.coloraxis.colorbar.title.text
        assert expected_title_fragment in actual_title

    # Verify tick marks exist and match expectations
    if hasattr(fig.layout.coloraxis.colorbar, "ticktext"):
        ticktext = fig.layout.coloraxis.colorbar.ticktext

        # Handle case where ticktext might be None
        if ticktext is None:
            # Skip further checks if ticktext is None
            return

        # Check minimum tick count
        if "min_tick_count" in assertions:
            min_count = assertions["min_tick_count"]
            assert len(ticktext) >= min_count, (
                f"Expected at least {min_count} ticks, got {len(ticktext)}"
            )

        # Convert tick texts to floats where possible for further checks
        tick_values = []
        for tt in ticktext:
            try:
                if "e" in str(tt).lower():
                    # Handle scientific notation
                    tick_values.append(float(str(tt).replace("e", "E")))
                else:
                    tick_values.append(float(tt))
            except (ValueError, AttributeError):
                continue  # Skip non-numeric tick labels

        # Check for positive and negative ticks
        if assertions.get("has_positive_ticks", False):
            assert any(val > 0 for val in tick_values), "No positive tick values found"

        if assertions.get("has_negative_ticks", False):
            assert any(val < 0 for val in tick_values), "No negative tick values found"


def test_log_scale_with_negative_values(sample_df: pd.DataFrame) -> None:
    """Test log scale with -ve values doesn't crash and fall back to linear scale."""
    # Create data with negative values
    df_test = sample_df.copy()
    df_test["property"] = [1.0, -1.0, 2.0]  # Contains negative value

    # Should not raise an exception, even with negative values
    fig = pmv.cluster_compositions(
        df_in=df_test,
        composition_col="composition",
        prop_name="property",
        projection="pca",
        color_scale="log",
    )

    # Verify the figure was created successfully
    assert isinstance(fig, go.Figure)

    # Verify we have a colorbar (should have fallen back to not using log scale)
    assert hasattr(fig.layout.coloraxis, "colorbar")


def test_arcsinh_tick_count() -> None:
    """Test that arcsinh scale generates at least 5 ticks."""
    # Create simple test data
    df_test = pd.DataFrame(
        {
            "composition": ["Fe2O3", "Al2O3", "Cu", "Au", "Ag"],
            "property": [1.0, 10.0, 100.0, -1.0, -10.0],
        }
    )

    # Create figure with arcsinh scale
    fig = pmv.cluster_compositions(
        df_in=df_test,
        composition_col="composition",
        prop_name="property",
        projection="pca",
        color_scale="arcsinh",
    )

    # Check that we have at least 5 ticks
    assert hasattr(fig.layout.coloraxis.colorbar, "ticktext")
    assert len(fig.layout.coloraxis.colorbar.ticktext) >= 5

    # Check that we have both positive and negative ticks
    tick_values = [
        float(text.replace("^", "e"))
        for text in fig.layout.coloraxis.colorbar.ticktext
        if text != "0"  # Skip zero
    ]

    assert any(val < 0 for val in tick_values), "No negative tick values found"
    assert any(val > 0 for val in tick_values), "No positive tick values found"

    # Check for common spacing pattern (1, 2, 5)
    positive_ticks = sorted([val for val in tick_values if val > 0])

    # There should be some ticks in the common pattern
    standard_values = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    matches = [
        val
        for val in positive_ticks
        if any(abs(val / std - 1) < 0.01 for std in standard_values)
    ]

    assert len(matches) > 0, "No ticks following the common spacing pattern found"


def test_log_tick_count() -> None:
    """Test that log scale generates at least 5 ticks."""
    # Create simple test data with positive values only
    df_test = pd.DataFrame({"composition": ["Fe2O3", "Al2O3", "Cu", "Au", "Ag"]})
    df_test["property"] = [0.1, 1.0, 10.0, 100.0, 1000.0]

    fig = pmv.cluster_compositions(  # Create figure with log scale
        df_in=df_test,
        composition_col="composition",
        prop_name="property",
        projection="pca",
        color_scale="log",
    )

    # Check that we have at least 5 ticks
    assert hasattr(fig.layout.coloraxis.colorbar, "ticktext")
    assert len(fig.layout.coloraxis.colorbar.ticktext) >= 5


def test_whole_number_formatting_in_tick_labels() -> None:
    """Test that whole numbers in colorbar tick labels don't have decimal places."""
    # Create data with whole numbers
    df_test = pd.DataFrame(
        {
            "composition": ["Fe2O3", "Al2O3", "Cu", "FeSiO3", "CaTiO3", "ZnO"],
            "property": [1.0, 5.0, 10.0, 50.0, 100.0, 1000.0],  # Whole numbers
        }
    )

    # Test both log and arcsinh scales
    for color_scale in ["log", "arcsinh"]:
        fig = pmv.cluster_compositions(
            df_in=df_test,
            composition_col="composition",
            prop_name="property",
            projection="pca",
            color_scale=color_scale,  # type: ignore[arg-type]
        )

        # Check that tick values are formatted without decimal places
        if hasattr(fig.layout.coloraxis.colorbar, "ticktext"):
            ticktext = fig.layout.coloraxis.colorbar.ticktext

            # Handle case where ticktext might be None
            if ticktext is None:
                pytest.skip(f"No ticktext available to check for {color_scale} scale")

            # Check for whole numbers in the tick labels
            whole_numbers = ["1", "5", "10", "50", "100", "1000"]

            # Look for any values that have incorrect formatting
            incorrect_formats = [
                tick
                for num in whole_numbers
                for tick in ticktext
                if (
                    isinstance(tick, str)
                    and f"{num}." in tick
                    and not any(c in tick for c in "x^")
                )
            ]

            assert len(incorrect_formats) == 0, (
                f"Found {len(incorrect_formats)} incorrectly formatted whole numbers "
                f"with decimal places: {incorrect_formats}"
            )


@pytest.mark.parametrize("n_components", [2, 3])
def test_annotate_points_basic(sample_df: pd.DataFrame, n_components: int) -> None:
    """Test basic annotation functionality."""

    def annotate_func(row: pd.Series) -> str:
        """Simple annotation function returning composition."""
        return f"Comp: {row['composition']}"

    fig = pmv.cluster_compositions(
        df_in=sample_df,
        embedding_method="one-hot",
        projection="pca",
        n_components=n_components,
        annotate_points=annotate_func,
        show_projection_stats=False,  # Disable stats for simpler counting
    )

    expected_annotation_count = len(sample_df)
    if n_components == 2:
        point_annotations = fig.layout.annotations
    else:  # 3D
        point_annotations = fig.layout.scene.annotations

    assert len(point_annotations) == expected_annotation_count

    for idx, annotation in enumerate(point_annotations):
        expected_text = f"Comp: {sample_df['composition'].iloc[idx]}"
        assert annotation.text == expected_text


@pytest.mark.parametrize("n_components", [2, 3])
def test_annotate_points_dict(sample_df: pd.DataFrame, n_components: int) -> None:
    """Test annotation using a dictionary to customize properties."""

    def annotate_func(row: pd.Series) -> dict[str, Any]:
        """Annotation function returning a dictionary."""
        return {
            "text": f"ID: {row.name}",
            "font": {"color": "red", "size": 12},
            "showarrow": True,
            "arrowhead": 2,
        }

    fig = pmv.cluster_compositions(
        df_in=sample_df,
        embedding_method="one-hot",
        projection="pca",
        n_components=n_components,
        annotate_points=annotate_func,
        show_projection_stats=False,  # Disable stats for simpler counting
    )

    expected_annotation_count = len(sample_df)
    if n_components == 2:
        point_annotations = fig.layout.annotations
    else:  # 3D
        point_annotations = fig.layout.scene.annotations

    assert len(point_annotations) == expected_annotation_count

    for idx, annotation in enumerate(point_annotations):
        assert annotation.text == f"ID: {sample_df.index[idx]}"
        assert annotation.font.color == "red"
        assert annotation.font.size == 12
        assert annotation.showarrow is True
        assert annotation.arrowhead == 2


@pytest.mark.parametrize("n_components", [2, 3])
def test_annotate_points_dict_override(
    sample_df: pd.DataFrame, n_components: int
) -> None:
    """Test that returned dict overrides default annotation parameters."""

    def annotate_func(row: pd.Series) -> dict[str, Any]:
        """Return dict to override yshift and showarrow."""
        return {
            "text": f"Override {row.name}",
            "yshift": 25,  # Override default 10
            "showarrow": True,  # Override default False
        }

    fig = pmv.cluster_compositions(
        df_in=sample_df,
        embedding_method="one-hot",
        projection="pca",
        n_components=n_components,
        annotate_points=annotate_func,
        show_projection_stats=False,  # Disable stats for simpler counting
    )

    expected_annotation_count = len(sample_df)
    if n_components == 2:
        point_annotations = fig.layout.annotations
    else:  # 3D
        point_annotations = fig.layout.scene.annotations

    assert len(point_annotations) == expected_annotation_count

    for annotation in point_annotations:
        assert annotation.yshift == 25
        assert annotation.showarrow is True
        assert annotation.font.size == 10  # Check default font size is retained


@pytest.mark.parametrize("n_components", [2, 3])
def test_annotate_points_conditional(
    sample_df: pd.DataFrame, n_components: int
) -> None:
    """Test conditional annotation where some points are annotated."""
    target_composition = sample_df["composition"].iloc[0]

    def annotate_func(row: pd.Series) -> str | None:
        """Annotate only the first composition."""
        if row["composition"] == target_composition:
            return f"Target: {row['composition']}"
        return None

    fig = pmv.cluster_compositions(
        df_in=sample_df,
        prop_name="property",
        embedding_method="one-hot",
        projection="pca",
        n_components=n_components,
        annotate_points=annotate_func,
        show_projection_stats=False,  # Disable stats for simpler counting
    )

    expected_annotation_count = 1
    if n_components == 2:
        point_annotations = fig.layout.annotations
    else:  # 3D
        point_annotations = fig.layout.scene.annotations

    assert len(point_annotations) == expected_annotation_count

    annotation = point_annotations[0]
    assert annotation.text == f"Target: {target_composition}"
    # Check if annotation is placed near the correct point
    point_x = fig.data[0].x[0]
    point_y = fig.data[0].y[0]
    assert abs(annotation.x - point_x) < 1e-6
    assert abs(annotation.y - point_y) < 1e-6


def test_annotate_points_with_coords(sample_df: pd.DataFrame) -> None:
    """Test annotation when using pre-computed coordinates."""
    # Create pre-computed coordinates
    coords = np_rng.random((len(sample_df), 2))
    df_coords = sample_df.copy()
    df_coords["coords"] = list(coords)

    def annotate_func(row: pd.Series) -> str:
        """Simple annotation function."""
        return f"Row {row.name}"

    fig = pmv.cluster_compositions(
        df_in=df_coords,
        embedding_method="one-hot",
        projection="coords",
        annotate_points=annotate_func,
    )

    # For precomputed coords, no stats annotation is added by default
    # so we expect exactly the number of points annotated.
    expected_annotation_count = len(sample_df)
    actual_annotations = fig.layout.annotations
    assert len(actual_annotations) == expected_annotation_count

    for idx, annotation in enumerate(actual_annotations):
        assert annotation.text == f"Row {sample_df.index[idx]}"
        # Check annotation position matches the precomputed coordinate
        assert abs(annotation.x - coords[idx, 0]) < 1e-6
        assert abs(annotation.y - coords[idx, 1]) < 1e-6


def test_annotate_points_access_projected_coords(sample_df: pd.DataFrame) -> None:
    """Test that the annotation function can access projected coordinates."""

    def annotate_func(row: pd.Series) -> str:
        """Annotation function using projected PCA coordinates."""
        pca1 = row.get("pca1", float("nan"))
        pca2 = row.get("pca2", float("nan"))
        return f"PCA: ({pca1:.2f}, {pca2:.2f})"

    fig = pmv.cluster_compositions(
        df_in=sample_df,
        embedding_method="one-hot",
        projection="pca",
        n_components=2,
        annotate_points=annotate_func,
    )

    # Account for the optional projection stats annotation (2D only)
    expected_annotation_count = len(sample_df)
    actual_annotations = fig.layout.annotations

    # Filter out the stats annotation if present
    point_annotations = [
        ann
        for ann in actual_annotations
        if not (ann.x == 0.02 and ann.y == 0.98)  # Heuristic for stats box
    ]
    assert len(point_annotations) == expected_annotation_count

    for idx, annotation in enumerate(point_annotations):
        point_x = fig.data[0].x[idx]
        point_y = fig.data[0].y[idx]
        expected_text = f"PCA: ({point_x:.2f}, {point_y:.2f})"
        assert annotation.text == expected_text


def test_annotate_points_access_original_columns(sample_df: pd.DataFrame) -> None:
    """Test that the annotation function can access original DataFrame columns."""
    # Add an extra column to the sample DataFrame
    df_extra = sample_df.copy()
    df_extra["extra_info"] = [f"info_{i}" for i in range(len(sample_df))]

    def annotate_func(row: pd.Series) -> str:
        """Annotation function using an original column."""
        extra = row.get("extra_info", "N/A")
        return f"Extra: {extra}"

    fig = pmv.cluster_compositions(
        df_in=df_extra,
        embedding_method="one-hot",
        projection="pca",
        annotate_points=annotate_func,
    )

    # Account for the optional projection stats annotation (2D only)
    # This test does not disable stats, so we keep the check
    expected_annotation_count = len(df_extra)
    actual_annotations = fig.layout.annotations

    # Filter out the stats annotation if present
    point_annotations = [
        ann for ann in actual_annotations if not (ann.x == 0.02 and ann.y == 0.98)
    ]
    assert len(point_annotations) == expected_annotation_count

    for idx, annotation in enumerate(point_annotations):
        expected_text = f"Extra: info_{idx}"
        assert annotation.text == expected_text
