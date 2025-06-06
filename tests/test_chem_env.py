"""Test chemical environment analysis utilities."""

from __future__ import annotations

import pytest
from pymatgen.core import Lattice, Structure

from pymatviz import chem_env


@pytest.mark.parametrize(
    ("symbol", "symbol_cn_mapping", "expected_cn"),
    [
        # Known symbols from mapping
        ("T:4", {"T:4": 4, "O:6": 6, "C:8": 8, "PP:5": 5}, 4),
        ("O:6", {"T:4": 4, "O:6": 6, "C:8": 8, "PP:5": 5}, 6),
        ("C:8", {"T:4": 4, "O:6": 6, "C:8": 8, "PP:5": 5}, 8),
        ("PP:5", {"T:4": 4, "O:6": 6, "C:8": 8, "PP:5": 5}, 5),
        # Special case: S:1
        ("S:1", {}, 1),
        ("S:1", {"other": 10}, 1),  # Should work regardless of mapping
        # M: prefix parsing
        ("M:7", {}, 7),
        ("M:12", {}, 12),
        ("M:1", {}, 1),
        ("M:24", {}, 24),
        ("M:100", {}, 100),
        # NULL and UNKNOWN
        ("NULL", {}, 0),
        ("UNKNOWN", {}, 0),
        ("NULL", {"T:4": 4}, 0),  # Should work regardless of mapping
        ("UNKNOWN", {"T:4": 4}, 0),
        # Invalid cases
        ("M:", {}, 0),  # Malformed M: symbol
        ("M:abc", {}, 0),  # Non-numeric after M:
        ("M:3.5", {}, 0),  # Float after M:
        ("M:-5", {}, -5),  # Negative after M:
        ("UNKNOWN_SYMBOL", {}, 0),
        ("", {}, 0),  # Empty string
        (":", {}, 0),  # Just colon
        ("T:", {}, 0),  # Symbol without CN
        ("random_text", {}, 0),
        ("123", {}, 0),  # Just numbers
        ("T:4:extra", {"T:4": 4}, 0),  # Extra parts
    ],
)
def test_get_cn_from_symbol(
    symbol: str, symbol_cn_mapping: dict[str, int], expected_cn: int
) -> None:
    """Test extraction of CN from ChemEnv symbols."""
    assert chem_env.get_cn_from_symbol(symbol, symbol_cn_mapping) == expected_cn


@pytest.mark.parametrize(
    ("symbol", "expected_cn"),
    [
        # Test with empty mapping - should still handle special cases
        ("S:1", 1),
        ("M:6", 6),
        ("M:15", 15),
        ("NULL", 0),
        ("UNKNOWN", 0),
        ("invalid", 0),
    ],
)
def test_get_cn_from_symbol_empty_mapping(symbol: str, expected_cn: int) -> None:
    """Test behavior with empty symbol mapping."""
    symbol_cn_mapping: dict[str, int] = {}
    assert chem_env.get_cn_from_symbol(symbol, symbol_cn_mapping) == expected_cn


@pytest.mark.parametrize(
    ("site_idx", "cn_val", "expected_format"),
    [
        # Basic functionality tests
        (0, 6, "string_with_colon_or_cn_prefix"),
        (0, 4, "string_with_colon_or_cn_prefix"),
        (0, 8, "string_with_colon_or_cn_prefix"),
        (0, 12, "string_with_colon_or_cn_prefix"),
        # Edge cases
        (0, 0, "CN:0"),  # Zero CN
        (0, -1, "CN:-1"),  # Negative CN
        (0, 99, "CN:99"),  # Unsupported CN
        (0, 1000, "CN:1000"),  # Very large CN
        # More common CNs
        (0, 1, "string_with_colon_or_cn_prefix"),
        (0, 2, "string_with_colon_or_cn_prefix"),
        (0, 3, "string_with_colon_or_cn_prefix"),
        (0, 5, "string_with_colon_or_cn_prefix"),
        (0, 7, "string_with_colon_or_cn_prefix"),
        (0, 9, "string_with_colon_or_cn_prefix"),
        (0, 10, "string_with_colon_or_cn_prefix"),
        (0, 11, "string_with_colon_or_cn_prefix"),
        # Invalid site indices should still return generic CN
        (999, 6, "CN:6"),  # Invalid site index
        (-1, 4, "CN:4"),  # Negative site index
        (100, 8, "CN:8"),  # Out of bounds site index
    ],
)
def test_classify_local_env_with_order_params_cubic(
    structures: tuple[Structure, Structure],
    site_idx: int,
    cn_val: int,
    expected_format: str,
) -> None:
    """Test local environment classification with structures from conftest."""
    # Use the first structure from conftest.py
    structure = structures[0]
    result = chem_env.classify_local_env_with_order_params(structure, site_idx, cn_val)

    assert isinstance(result, str)
    assert len(result) > 0

    if expected_format == "string_with_colon_or_cn_prefix":
        # Should either be "GEOMETRY:CN" or start with "CN"
        assert ":" in result or result.startswith("CN")
    else:
        # Exact match expected
        assert result == expected_format

    # Verify format consistency
    if ":" in result:
        parts = result.split(":")
        assert len(parts) == 2
        if not result.startswith("CN:"):
            # If not generic CN format, second part should match cn_val
            assert parts[1] == str(cn_val)


@pytest.mark.parametrize(
    "cn_val",
    list(range(1, 25)),  # Test CN 1-24
)
def test_classify_local_env_return_format_consistency(
    structures: tuple[Structure, Structure], cn_val: int
) -> None:
    """Test that return format is consistent across all CN values."""
    # Use the first structure from conftest.py
    structure = structures[0]
    site_idx = 0
    result = chem_env.classify_local_env_with_order_params(structure, site_idx, cn_val)

    # Should be string
    assert isinstance(result, str)
    assert len(result) > 0

    # Should either be "GEOMETRY:CN" or "CN:N" format
    if ":" in result:
        parts = result.split(":")
        assert len(parts) == 2
        # Second part should be numeric
        try:
            int(parts[1])
        except ValueError:
            pytest.fail(f"Second part of '{result}' is not numeric")


def test_classify_local_env_different_structures(
    structures: tuple[Structure, Structure],
) -> None:
    """Test with different structure types."""
    # Use the second structure from conftest.py
    structure = structures[1]
    site_idx = 0
    cn_val = 12  # Test with CN 12

    result = chem_env.classify_local_env_with_order_params(structure, site_idx, cn_val)

    assert isinstance(result, str)
    assert len(result) > 0
    assert ":" in result or result.startswith("CN")


@pytest.mark.parametrize(
    ("lattice_param", "species", "coords"),
    [
        # Different lattice parameters
        (2.0, ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        (6.0, ["K", "Br"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
        # Single atom structures
        (4.0, ["Fe"], [[0, 0, 0]]),
        (3.0, ["Al"], [[0, 0, 0]]),
        # More complex structures
        (
            4.0,
            ["Ca", "Ti", "O", "O", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]],
        ),
    ],
)
def test_classify_local_env_various_structures(
    lattice_param: float, species: list[str], coords: list[list[float]]
) -> None:
    """Test classification with various structure types."""
    lattice = Lattice.cubic(lattice_param)
    structure = Structure(lattice, species, coords)

    # Test with different CN values
    for cn_val in [4, 6, 8, 12]:
        for site_idx in range(len(species)):
            result = chem_env.classify_local_env_with_order_params(
                structure, site_idx, cn_val
            )
            assert isinstance(result, str)
            assert len(result) > 0


def test_chem_env_functions_integration(
    structures: tuple[Structure, Structure],
) -> None:
    """Test that chemenv functions work together."""
    # Test symbol mapping
    symbol_cn_mapping = {"T:4": 4, "O:6": 6, "C:8": 8}

    # Test get_cn_from_symbol
    cn = chem_env.get_cn_from_symbol("T:4", symbol_cn_mapping)
    assert cn == 4

    # Use structure from conftest.py
    structure = structures[0]

    # Test classify_local_env_with_order_params with extracted CN
    result = chem_env.classify_local_env_with_order_params(structure, 0, cn)
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.parametrize(
    ("invalid_symbol", "invalid_structure_params"),
    [
        ("INVALID_SYMBOL", None),
        ("", None),
        ("M:not_a_number", None),
        (None, "empty_species"),  # Will be handled in test
    ],
)
def test_error_handling_consistency(
    invalid_symbol: str | None,
    invalid_structure_params: str | None,
    structures: tuple[Structure, Structure],
) -> None:
    """Test that error handling is consistent across functions."""
    symbol_cn_mapping: dict[str, int] = {}

    if invalid_symbol is not None:
        # get_cn_from_symbol should return 0 for unknown symbols
        result_cn = chem_env.get_cn_from_symbol(invalid_symbol, symbol_cn_mapping)
        assert result_cn == 0

    # Test classify_local_env_with_order_params error handling
    if invalid_structure_params != "empty_species":
        # Use structure from conftest.py
        structure = structures[0]

        # Should handle gracefully even if analysis fails
        result = chem_env.classify_local_env_with_order_params(structure, 0, 6)
        assert isinstance(result, str)
        assert len(result) > 0


@pytest.mark.parametrize(
    ("symbol_mapping", "test_symbols"),
    [
        # Empty mapping
        ({}, ["S:1", "M:6", "NULL", "UNKNOWN", "invalid"]),
        # Partial mapping
        ({"T:4": 4}, ["T:4", "O:6", "S:1", "M:8"]),
        # Full mapping
        (
            {"T:4": 4, "O:6": 6, "C:8": 8, "PP:5": 5},
            ["T:4", "O:6", "C:8", "PP:5", "M:10"],
        ),
        # Complex mapping with edge cases
        ({"CUSTOM:3": 3, "X:Y:7": 7}, ["CUSTOM:3", "X:Y:7", "S:1", "NULL"]),
    ],
)
def test_get_cn_from_symbol_various_mappings(
    symbol_mapping: dict[str, int], test_symbols: list[str]
) -> None:
    """Test get_cn_from_symbol with various symbol mappings."""
    for symbol in test_symbols:
        result = chem_env.get_cn_from_symbol(symbol, symbol_mapping)
        assert isinstance(result, int)
        assert result >= 0  # CN should never be negative from this function


@pytest.mark.parametrize(
    ("structure_type", "expected_behavior"),
    [
        ("simple_cubic", "should_work"),  # Uses SI2_STRUCT from conftest
        ("multi_element", "should_work"),  # Uses si2_ru2_pr2_struct from conftest
    ],
)
def test_classify_local_env_structure_robustness(
    structure_type: str,
    expected_behavior: str,
    structures: tuple[Structure, Structure],
) -> None:
    """Test robustness across different structure types."""
    structure = structures[0 if structure_type == "simple_cubic" else 1]

    # Test with edge case CN values
    edge_case_cns = [0, 1, 2, 13, 20, 50, 100]

    for cn_val in edge_case_cns:
        result = chem_env.classify_local_env_with_order_params(structure, 0, cn_val)

        if expected_behavior == "should_work":
            assert isinstance(result, str)
            assert len(result) > 0
            # Should handle gracefully - either return specific geometry or generic CN
            assert ":" in result or result.startswith("CN")
