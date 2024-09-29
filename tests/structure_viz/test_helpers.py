import re
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from pymatgen.core import Structure

from pymatviz.enums import ElemColorScheme
from pymatviz.structure_viz.helpers import (
    NO_SYM_MSG,
    UNIT_CELL_EDGES,
    _angles_to_rotation_matrix,
    add_site_to_plot,
    generate_subplot_title,
    get_atomic_radii,
    get_elem_colors,
    get_image_atoms,
    get_structures,
)


@pytest.fixture
def mock_figure() -> Any:
    class MockFigure:
        def add_scatter(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_scatter3d(self, *args: Any, **kwargs: Any) -> None:
            pass

    return MockFigure()


@pytest.mark.parametrize(
    ("angles", "expected_shape"),
    [("10x,8y,3z", (3, 3)), ("30x,45y,60z", (3, 3))],
)
def test_angles_to_rotation_matrix(
    angles: str, expected_shape: tuple[int, int]
) -> None:
    rot_matrix = _angles_to_rotation_matrix(angles)
    assert rot_matrix.shape == expected_shape
    assert np.allclose(np.linalg.det(rot_matrix), 1.0)


def test_angles_to_rotation_matrix_invalid_input() -> None:
    with pytest.raises(ValueError, match="could not convert string to float: "):
        _angles_to_rotation_matrix("invalid_input")


def test_get_structures(structures: list[Structure]) -> None:
    # Test with single structure
    result = get_structures(structures[0])
    assert isinstance(result, dict)
    assert len(result) == 1
    assert isinstance(next(iter(result.values())), Structure)

    # Test with list of structures
    result = get_structures(structures)
    assert isinstance(result, dict)
    assert len(result) == len(structures)
    assert all(isinstance(s, Structure) for s in result.values())

    # Test with dict of structures
    struct_dict = {f"struct{i}": s for i, s in enumerate(structures)}
    result = get_structures(struct_dict)
    assert isinstance(result, dict)
    assert len(result) == len(structures)
    assert all(isinstance(s, Structure) for s in result.values())


@pytest.mark.parametrize(
    ("elem_colors", "expected_result"),
    [
        (ElemColorScheme.jmol, dict),
        (ElemColorScheme.vesta, dict),
        ({"Si": "#FF0000", "O": "#00FF00"}, dict),
    ],
)
def test_get_elem_colors(
    elem_colors: ElemColorScheme | dict[str, str], expected_result: type
) -> None:
    colors = get_elem_colors(elem_colors)
    assert isinstance(colors, expected_result)
    if isinstance(elem_colors, dict):
        assert colors == elem_colors
    else:
        assert "Si" in colors
        assert "O" in colors


def test_get_elem_colors_invalid_input() -> None:
    with pytest.raises(
        ValueError, match=re.escape("colors must be a dict or one of ('jmol, vesta')")
    ):
        get_elem_colors("invalid_input")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("atomic_radii", "expected_type"),
    [(None, dict), (1.5, dict), ({"Si": 0.3, "O": 0.2}, dict), ("invalid_input", str)],
)
def test_get_atomic_radii(
    atomic_radii: float | dict[str, float] | None | str, expected_type: type
) -> None:
    radii = get_atomic_radii(atomic_radii)  # type: ignore[arg-type]
    assert isinstance(radii, expected_type)

    if atomic_radii is None or isinstance(atomic_radii, float):
        assert "Si" in radii
        assert "O" in radii
        if isinstance(atomic_radii, float):
            assert radii["Si"] == pytest.approx(1.11 * atomic_radii)
    elif isinstance(atomic_radii, dict):
        assert radii == atomic_radii
    else:
        assert radii == atomic_radii


def test_get_image_atoms(structures: list[Structure]) -> None:
    structure = structures[0]
    site = structure[0]
    lattice = structure.lattice

    # Test with default tolerance
    image_atoms = get_image_atoms(site, lattice)
    assert isinstance(image_atoms, np.ndarray)
    assert image_atoms.ndim in (1, 2)  # Allow both 1D and 2D arrays
    if image_atoms.size > 0:
        assert image_atoms.shape[1] == 3  # Each image atom should have 3 coordinates

    # Test with custom tolerance
    image_atoms = get_image_atoms(site, lattice, tol=0.1)
    assert isinstance(image_atoms, np.ndarray)
    assert image_atoms.ndim in (1, 2)
    if image_atoms.size > 0:
        assert image_atoms.shape[1] == 3

    # Test with site at lattice origin (should return empty array)
    site.coords = [0, 0, 0]
    image_atoms = get_image_atoms(site, lattice)
    assert len(image_atoms) == 0


def test_add_site_to_plot(structures: list[Structure], mock_figure: Any) -> None:
    structure = structures[0]
    site = structure[0]
    coords = site.coords
    elem_colors = get_elem_colors(ElemColorScheme.jmol)
    atomic_radii = get_atomic_radii(None)

    # Test cases
    test_cases = [
        {"is_3d": False, "is_image": False},
        {"is_3d": True, "is_image": False},
        {"is_3d": False, "is_image": True},
    ]

    for case in test_cases:
        add_site_to_plot(
            mock_figure,
            site,
            coords,
            0,
            "symbol",
            elem_colors,
            atomic_radii,
            atom_size=40,
            scale=1,
            site_kwargs={},
            **case,  # type: ignore[arg-type]
        )

    # Test with custom site labels
    custom_labels = {site.species_string: "Custom"}
    add_site_to_plot(
        mock_figure,
        site,
        coords,
        0,
        custom_labels,
        elem_colors,
        atomic_radii,
        atom_size=40,
        scale=1,
        site_kwargs={},
        is_3d=False,
    )


@pytest.mark.parametrize(
    ("struct_key", "subplot_title", "expected_text"),
    [
        ("key", None, "key"),
        (1, None, "1. Si2 (spg="),  # Partial match due to dynamic spg number
        (
            "key",
            lambda _struct, key: f"Custom title for {key}",
            "Custom title for key",
        ),
        (
            "key",
            lambda _struct, key: {"text": f"Custom for {key}", "font": {"size": 14}},
            "Custom for key",
        ),
    ],
)
def test_generate_subplot_title(
    structures: list[Structure],
    struct_key: Any,
    subplot_title: Callable[[Structure, str | int], str | dict[str, Any]] | None,
    expected_text: str,
) -> None:
    structure = structures[0]
    title = generate_subplot_title(structure, struct_key, 1, subplot_title)
    assert isinstance(title, dict)
    assert "text" in title
    assert expected_text in title["text"]

    if callable(subplot_title) and isinstance(
        subplot_title(structure, struct_key), dict
    ):
        assert "font" in title
        assert title["font"]["size"] == 14


def test_constants() -> None:
    assert isinstance(NO_SYM_MSG, str)
    assert isinstance(UNIT_CELL_EDGES, tuple)
    assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in UNIT_CELL_EDGES)
