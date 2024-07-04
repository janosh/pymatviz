from __future__ import annotations

import numpy as np
import pandas as pd

from pymatviz._preprocess_data import get_df_nest_level
from pymatviz.enums import Key


def test_get_df_nest_level() -> None:
    # Test nest level 0
    df_level_0 = pd.DataFrame(
        {"Fe": 1, "O": 2}.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert get_df_nest_level(df_level_0, col=Key.heat_val) == 0

    # Test nest level 1
    df_level_1 = pd.DataFrame(
        {"Fe": [1, 2, 3], "O": [4, 5, 6]}.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert get_df_nest_level(df_level_1, col=Key.heat_val) == 1

    df_level_1_arr = pd.DataFrame(
        {"Fe": 1, "O": np.array([4, 5, 6])}.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert get_df_nest_level(df_level_1_arr, col=Key.heat_val) == 1

    # Test nest level 2
    df_level_2 = pd.DataFrame(
        {"Fe": [1, 2, 3], "O": [[4, 5], [6, 7]]}.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert get_df_nest_level(df_level_2, col=Key.heat_val) == 2

    df_level_2_arr = pd.DataFrame(
        {
            "Fe": [1, 2, 3],
            "O": np.array([[4, 5], [6, 7]]),  # get max level
        }.items(),
        columns=[Key.element, Key.heat_val],
    ).set_index(Key.element)

    assert get_df_nest_level(df_level_2_arr, col=Key.heat_val) == 2
