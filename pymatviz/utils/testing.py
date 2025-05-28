"""Testing related utils."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pymatviz.utils import ROOT


if TYPE_CHECKING:
    from phonopy import Phonopy


TEST_FILES: str = f"{ROOT}/tests/files"


def load_phonopy_nacl() -> Phonopy:
    """Load phonopy NaCl object from test files.

    Returns a Phonopy class instance of NaCl 2x2x2 without symmetrizing fc2.
    This function can be used in both tests and examples to get a consistent
    phonopy object for demonstrations.

    Returns:
        Phonopy: Loaded phonopy object with force constants.

    Raises:
        ImportError: If phonopy is not installed.

    Example:
        >>> phonopy_nacl = load_phonopy_nacl()
        >>> phonopy_nacl.run_mesh([10, 10, 10])
        >>> phonopy_nacl.run_total_dos()
        >>> dos = phonopy_nacl.total_dos
    """
    import phonopy

    return phonopy.load(
        f"{TEST_FILES}/phonons/NaCl/phonopy_disp.yaml.xz",
        force_sets_filename=f"{TEST_FILES}/phonons/NaCl/force_sets.dat",
        symmetrize_fc=False,
        produce_fc=True,
    )
