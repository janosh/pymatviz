"""This script plots the keys, labels, symbols, units, descriptions, and categories of
pymatviz.enums.Key enum to visually inspect how they render in plotly.
"""

import numpy as np
import plotly.graph_objects as go

from pymatviz.enums import Key


key_data = [
    (key.name, key.label, key.symbol, key.unit, key.desc, key.category) for key in Key
]
# Sort by category then name
key_data.sort(key=lambda x: (x[5], x[0]))
headers: dict[str, float] = {
    "": 0.7,
    "Key": 4,
    "Label": 4,
    "Symbol": 2,
    "Unit": 2,
    "Description": 4,
    "Category": 2.5,
}
table_rows = list(
    zip(
        *[(idx, *tup) for idx, tup in enumerate(key_data, start=1)],
        strict=True,
    )
)

table = go.Table(
    header=dict(
        values=[f"<b>{h}</b>" for h in headers],
        line_color="darkslategray",
        fill_color="#4169E1",
        # align="left",
        font=dict(color="white", size=12),
    ),
    cells=dict(
        values=table_rows,
        line_color="#ddd",
        # alternate row colors for better readability
        fill_color=np.where(
            np.arange(len(key_data))[None, :] % 2 == 0, "white", "#F5F5F5"
        ),
        align="left",
        font=dict(color="#333", size=11),
        height=22,
    ),
    columnwidth=list(headers.values()),
)
fig = go.Figure(data=[table])

fig.layout.title = dict(
    text="pymatviz.enums.Key Attributes", x=0.5, y=0.99, yanchor="top"
)
fig.layout.margin = dict(l=0, r=0, t=25, b=0)
fig.layout.paper_bgcolor = "white"

fig.show()
