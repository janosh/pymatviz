"""Plotly powerups examples."""

# %%
from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pymatviz as pmv
from pymatviz.powerups.plotly import (
    add_ecdf_line,
    select_colorscale,
    select_marker_mode,
    toggle_grid,
    toggle_log_linear_x_axis,
    toggle_log_linear_y_axis,
)


pmv.set_plotly_template("pymatviz_white")


# %% Generate random regression data
rand_regression_size = 500
np_rng = np.random.default_rng(seed=0)
gauss1 = np_rng.normal(5, 4, rand_regression_size)
gauss2 = np_rng.normal(10, 2, rand_regression_size)


# %% ECDF line
fig = pmv.histogram({"Gaussian 1": gauss1, "Gaussian 2": gauss2}, bins=200)
add_ecdf_line(fig)
fig.layout.title.update(text="Histogram with ECDF Lines", x=0.5)
fig.layout.margin.t = 50
fig.show()


# %% toggle log/linear x-axis
x = np.logspace(0, 3, 100)
y = np.exp(x / 100)
fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers"))
fig.layout.title.update(text="Exponential Growth", x=0.5)
fig.layout.margin.t = 50
fig.layout.updatemenus = [toggle_log_linear_x_axis]
fig.show()


# %% Toggle grid
t = np.linspace(0, 10, 100)
x = np.cos(t)
y = np.sin(t)

fig = go.Figure(go.Scatter(x=x, y=y, mode="lines"))
fig.layout.title.update(text="Parametric Curve (cos(t), sin(t))", x=0.5)
fig.layout.margin.t = 50
fig.layout.updatemenus = [toggle_grid]
fig.show()


# %% Toggle colorscale
z = np.random.default_rng(seed=0).standard_normal((50, 50))
fig = go.Figure(data=go.Heatmap(z=z))
fig.layout.title.update(text="Random Heatmap with Colorscale Toggle", x=0.5)
fig.layout.margin.t = 50
fig.layout.updatemenus = [select_colorscale]
fig.show()


# %% Toggle plot type
t = np.linspace(0, 10, 50)
y = np.sin(t)
fig = go.Figure(go.Scatter(x=t, y=y))
fig.layout.title.update(text="Sine Wave with Plot Type Toggle", x=0.5)
fig.layout.margin.t = 50
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
fig.layout.title.update(text="Gapminder 2007: GDP per Capita vs Life Expectancy", x=0.5)
fig.layout.margin.t = 50
fig.layout.updatemenus = [
    toggle_log_linear_x_axis | dict(x=1.2, y=0.12, xanchor="right", yanchor="bottom"),
    toggle_log_linear_y_axis | dict(x=1.2, y=0.02, xanchor="right", yanchor="bottom"),
    toggle_grid | dict(x=0.02, y=1, xanchor="left", yanchor="top"),
]
fig.show()


# %%
fig = px.scatter(x=gauss1, y=gauss2)
pmv.powerups.enhance_parity_plot(fig)
fig.layout.title.update(x=0.5)
fig.layout.margin.t = 50
fig.show()


# closer agreement between xs and ys to see stats
fig = px.scatter(x=np.arange(100), y=np.arange(100) + 10 * np_rng.random(100))
pmv.powerups.enhance_parity_plot(fig)
fig.layout.title.update(x=0.5)
fig.layout.margin.t = 50
fig.show()


# %% Multi-trace parity plot with per-trace stats
# Create a multi-trace figure with different synthetic datasets
np_rng = np.random.default_rng(seed=42)
x1 = np.arange(0, 100)
y1 = x1 * 0.8 + 5 + np_rng.normal(0, 8, len(x1))  # Linear with positive slope and noise
x2 = np.arange(0, 100)
y2 = (
    100 - x2 * 0.7 + np_rng.normal(0, 10, len(x2))
)  # Linear with negative slope and noise
x3 = np.arange(0, 100)
y3 = x3 + np_rng.normal(0, 20, len(x3))  # Perfect correlation with high noise


# %% Multi-trace parity plot with combined stats
# Use the same data as above but show overall stats
fig = go.Figure()
fig.add_scatter(
    x=x1, y=y1, mode="markers", name="Positive Slope", marker=dict(color="blue")
)
fig.add_scatter(
    x=x2, y=y2, mode="markers", name="Negative Slope", marker=dict(color="red")
)
fig.add_scatter(
    x=x3,
    y=y3,
    mode="markers",
    name="Perfect with Noise",
    marker=dict(color="green"),
)
title = "Multi-trace Parity Plot with Combined Stats"
fig.update_layout(title=title, xaxis_title="X Values", yaxis_title="Y Values")
pmv.powerups.enhance_parity_plot(
    fig,
    traces=slice(None),
    annotation_mode="combined",
    stats=dict(x=0.02, y=0.02, xanchor="left", yanchor="bottom"),
)
fig.layout.title.update(x=0.5)
fig.layout.margin.t = 50
fig.show()


# %% Multi-trace parity plot with custom styling
# Create a new dataset
x4 = np.linspace(0, 100, 80)
y4 = x4 * 1.2 - 10 + np_rng.normal(0, 5, len(x4))


# %% Faceted parity plot with 2x1 subplots
fig = make_subplots(
    rows=2, cols=1, subplot_titles=["Dataset 1", "Dataset 2"], vertical_spacing=0.1
)

# Add traces to each subplot
fig.add_scatter(x=x1, y=y1, mode="markers", name="Dataset 1", row=1, col=1)
fig.add_scatter(x=x4, y=y4, mode="markers", name="Dataset 2", row=2, col=1)

fig.update_layout(title="Parity Plot with Faceted Subplots", showlegend=False)

# Apply identity lines and best-fit lines to each subplot
# Note: These will be applied automatically to each subplot
pmv.powerups.enhance_parity_plot(fig, annotation_mode="per_trace")
fig.layout.title.update(x=0.5)
fig.layout.margin.t = 50
fig.show()


# %% Custom regression metrics example
fig = px.scatter(x=x1, y=y1, title="Parity Plot with Custom Metrics")

# Apply parity plot with custom metrics
pmv.powerups.enhance_parity_plot(
    fig,
    stats=dict(
        prefix="Model Performance:\n",
        suffix=f"N={len(x1)}",
        fmt=".4f",  # More precision
        x=0.02,
        y=0.8,
        xanchor="left",
        yanchor="bottom",
    ),
)
fig.layout.title.update(x=0.5)
fig.layout.margin.t = 50
fig.show()


# %% More complex trace filtering example
# Create multi-trace figure with various marker sizes
fig = go.Figure()
# Add points with different marker sizes
for size, color, name in zip(
    [5, 10, 15], ["blue", "red", "green"], ["Small", "Medium", "Large"], strict=False
):
    # Add different amount of noise based on size
    noise = np_rng.normal(0, size / 2, 100)
    fig.add_scatter(
        x=np.arange(100),
        y=np.arange(100) + noise,
        mode="markers",
        marker=dict(size=size, color=color),
        name=name,
    )

fig.update_layout(title="Filter Traces by Marker Properties")

# Add parity plot enhancements only to traces with marker size > 7
pmv.powerups.enhance_parity_plot(
    fig,
    traces=lambda trace: hasattr(trace, "marker")
    and hasattr(trace.marker, "size")
    and trace.marker.size > 7,
    annotation_mode="per_trace",
)
fig.layout.title.update(x=0.5)
fig.layout.margin.t = 50
fig.show()
