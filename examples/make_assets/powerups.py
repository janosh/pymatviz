# %%
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pymatviz.histogram import histogram
from pymatviz.powerups.plotly import (
    add_ecdf_line,
    select_colorscale,
    select_marker_mode,
    toggle_grid,
    toggle_log_linear_x_axis,
    toggle_log_linear_y_axis,
)
from pymatviz.templates import set_plotly_template


set_plotly_template("pymatviz_white")


# %% Configure matplotlib and load test data
# Random regression data
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
gauss1 = np_rng.normal(5, 4, rand_regression_size)
gauss2 = np_rng.normal(10, 2, rand_regression_size)


# %%
fig = histogram({"Gaussian 1": gauss1, "Gaussian 2": gauss2}, bins=200)
for idx in range(len(fig.data)):
    add_ecdf_line(fig, trace_idx=idx)
fig.show()


# %% Configure matplotlib and load test data
# Random regression data
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
gauss1 = np_rng.normal(5, 4, rand_regression_size)
gauss2 = np_rng.normal(10, 2, rand_regression_size)


# %% ECDF line
fig = histogram({"Gaussian 1": gauss1, "Gaussian 2": gauss2}, bins=200)
for idx in range(len(fig.data)):
    add_ecdf_line(fig, trace_idx=idx)
fig.layout.title = "Histogram with ECDF Lines"
fig.show()


# %% toggle log/linear x-axis
x = np.logspace(0, 3, 100)
y = np.exp(x / 100)
fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers"))
fig.layout.title = "Exponential Growth"
fig.layout.updatemenus = [toggle_log_linear_x_axis]
fig.show()


# %% Toggle log/linear y-axis
fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers"))
fig.layout.title = "Exponential Growth"
fig.layout.updatemenus = [toggle_log_linear_y_axis]
fig.show()


# %% Toggle grid
t = np.linspace(0, 10, 100)
x = np.cos(t)
y = np.sin(t)
fig = go.Figure(go.Scatter(x=x, y=y, mode="lines"))
fig.layout.title = "Parametric Curve (cos(t), sin(t))"
fig.layout.updatemenus = [toggle_grid]
fig.show()


# %% Toggle colorscale
z = np.random.default_rng().standard_normal((50, 50))
fig = go.Figure(data=go.Heatmap(z=z))
fig.layout.title = "Random Heatmap with Colorscale Toggle"
fig.layout.updatemenus = [select_colorscale]

fig.show()


# %% Toggle plot type
t = np.linspace(0, 10, 50)
y = np.sin(t)
fig = go.Figure(go.Scatter(x=t, y=y))
fig.layout.title = "Sine Wave with Plot Type Toggle"
fig.layout.updatemenus = [select_marker_mode]

fig.show()


# %% Multiple powerups combined
df_gap = px.data.gapminder().query("year == 2007")
fig = px.scatter(
    df_gap,
    x="gdpPercap",
    y="lifeExp",
    size="pop",
    color="continent",
    hover_name="country",
    size_max=60,
)
fig.layout.title = dict(text="Gapminder 2007: GDP per Capita vs Life Expectancy", x=0.5)
fig.layout.updatemenus = [
    toggle_log_linear_x_axis | dict(x=1.2, y=0.12, xanchor="right", yanchor="bottom"),
    toggle_log_linear_y_axis | dict(x=1.2, y=0.02, xanchor="right", yanchor="bottom"),
    toggle_grid | dict(x=0.02, y=1, xanchor="left", yanchor="top"),
]
fig.show()
