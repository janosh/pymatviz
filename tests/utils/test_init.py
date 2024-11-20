from __future__ import annotations

import os

from pymatviz.utils import PKG_DIR, ROOT


assert os.path.isdir(ROOT)
assert os.path.isdir(PKG_DIR)


assert set(os.listdir(ROOT)).issuperset({"examples", "pymatviz", "tests", "assets"})
