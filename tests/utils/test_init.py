from __future__ import annotations

import os

from pymatviz.utils import PKG_DIR, ROOT


def test_dir_globals() -> None:
    assert os.path.isdir(PKG_DIR)
    assert os.path.isdir(ROOT)

    assert os.path.dirname(PKG_DIR) == ROOT
    assert set(os.listdir(ROOT)) >= {"examples", "pymatviz", "tests", "assets"}
