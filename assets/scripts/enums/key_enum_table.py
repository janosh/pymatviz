"""This script tabulates the keys, labels, symbols, units, descriptions, and categories
of the pymatviz.enums.Key enum to visually inspect them.
"""

import pandas as pd

from pymatviz.enums import Key


df_keys = pd.DataFrame(
    [
        {
            "Key": key.name,
            "Label": key.label,
            "Symbol": key.symbol,
            "Unit": key.unit,
            "Description": key.desc,
            "Category": key.category,
        }
        for key in Key
    ]
)
# Sort by category then name
df_keys = df_keys.sort_values(["Category", "Key"], ignore_index=True)
df_keys.index += 1

# MatterViz has no table widget, so print the table instead of rendering a plotly one
with pd.option_context("display.max_rows", None, "display.max_colwidth", 60):
    print(df_keys.to_string())
