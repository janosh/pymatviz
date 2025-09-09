"""Comprehensive density scatter examples."""

# %%
from __future__ import annotations

import numpy as np
import pandas as pd

import pymatviz as pmv


pmv.set_plotly_template("pymatviz_white")


# %% Basic density scatter with regression data
y_true, y_pred, _y_std = pmv.data.regression()

fig = pmv.density_scatter(y_true, y_pred)
fig.show()
# pmv.io.save_and_compress_svg(fig, "density-scatter")


# %% Density scatter with histograms
fig = pmv.density_scatter_with_hist(y_true, y_pred)
fig.show()
# pmv.io.save_and_compress_svg(fig, "density-scatter-with-hist")


# %% Visualizing ML model predictions with uncertainty
np_rng = np.random.default_rng(seed=123)
n_samples = 20000
true_values = np.linspace(0, 10, n_samples) + np_rng.normal(0, 0.5, n_samples)
predictions = true_values + np_rng.normal(0, 1.0, n_samples)

df_ml = pd.DataFrame(
    {
        "True Values": true_values,
        "Predictions": predictions,
    }
)

fig = pmv.density_scatter(
    df=df_ml,
    x="True Values",
    y="Predictions",
    hover_format=".0f",
    title="ML Model Performance (20k points)",
    identity_line={"line": {"color": "black", "width": 1, "dash": "dash"}},
    best_fit_line={"line": {"color": "red", "width": 2}},
)
fig.layout.update(margin_t=40, title_x=0.5)
fig.show()


# %% Visualizing structure-property relationships
np_rng = np.random.default_rng(seed=456)
n_samples = 30000

lattice_constant = np_rng.uniform(3.5, 6.0, n_samples)
bond_length = lattice_constant * 0.4 + np_rng.normal(0, 0.1, n_samples)
conductivity = np.exp(-bond_length) * 1000 + np_rng.normal(0, 50, n_samples)
df_structure = pd.DataFrame(
    {
        "Lattice Constant (Å)": lattice_constant,
        "Bond Length (Å)": bond_length,
        "Conductivity (S/cm)": conductivity,
    }
)

fig = pmv.density_scatter(
    df=df_structure,
    x="Bond Length (Å)",
    y="Conductivity (S/cm)",
    hover_format=".0f",
    title="Structure-Property Relationship (30k points)",
    log_density=True,
    identity_line=False,
)
fig.layout.update(margin_t=40, title_x=0.5)
fig.show()


# %% Visualizing composition-property relationships
np_rng = np.random.default_rng(seed=101)
n_samples = 40000

comp_a = np_rng.uniform(20, 80, n_samples)
comp_b = np_rng.uniform(10, 50, n_samples)
comp_c = 100 - comp_a - comp_b
valid_idx = comp_c > 0
comp_a, comp_b, comp_c = comp_a[valid_idx], comp_b[valid_idx], comp_c[valid_idx]

hardness = 0.3 * comp_a + 0.5 * comp_b + 0.1 * comp_c + np_rng.normal(0, 5, len(comp_a))

df_alloys = pd.DataFrame(
    {
        "Composition A (%)": comp_a,
        "Composition B (%)": comp_b,
        "Composition C (%)": comp_c,
        "Hardness (HV)": hardness,
    }
)

df_alloys["Dominant Element"] = "Mixed"
df_alloys.loc[comp_a > 50, "Dominant Element"] = "A-rich"
df_alloys.loc[comp_b > 40, "Dominant Element"] = "B-rich"
df_alloys.loc[comp_c > 40, "Dominant Element"] = "C-rich"

fig = pmv.density_scatter(
    df=df_alloys,
    x="Composition A (%)",
    y="Hardness (HV)",
    facet_col="Dominant Element",
    bin_counts_col="Data Density",
    hover_format=".0f",
    n_bins=200,
    title="Composition-Property Relationship (40k points)",
)
fig.layout.update(margin_t=40, title_x=0.5)
fig.show()


# %% Materials science relationship: atomic radius vs melting point
np_rng = np.random.default_rng(seed=555)
n_samples = 75000

# Create a more realistic materials science relationship:
# Atomic radius vs melting point (generally inversely related)
atomic_radius = np_rng.uniform(0.5, 3.0, n_samples)  # Angstroms
# Melting point decreases with increasing atomic radius (with noise)
melting_point = 2500 / (atomic_radius + 0.2) + np_rng.normal(0, 150, n_samples)

# Group by element groups
element_groups = (
    "Alkali Metals,Transition Metals,Noble Gases,Lanthanides,Actinides".split(",")  # noqa: SIM905
)
element_weights = [0.2, 0.4, 0.15, 0.15, 0.1]
element_group = np_rng.choice(element_groups, n_samples, p=element_weights)

# Add some group-specific offsets to make the data more realistic
for group in element_groups:
    mask = element_group == group
    # Add group-specific characteristics
    if group == "Alkali Metals":
        atomic_radius[mask] += 0.5  # Larger atomic radii
        melting_point[mask] -= 300  # Lower melting points
    elif group == "Transition Metals":
        melting_point[mask] += 500  # Higher melting points
    elif group == "Noble Gases":
        melting_point[mask] -= 1000  # Very low melting points
    elif group == "Lanthanides":
        atomic_radius[mask] += 0.2  # Slightly larger radii
    # Actinides remain as baseline

df_publication = pd.DataFrame(
    {
        "Atomic Radius (Å)": atomic_radius,
        "Melting Point (K)": melting_point,
        "Element Group": element_group,
    }
)

# Ensure melting points are physically reasonable
df_publication.loc[df_publication["Melting Point (K)"] < 0, "Melting Point (K)"] = (
    np_rng.uniform(4, 100, len(df_publication[df_publication["Melting Point (K)"] < 0]))
)

fig = pmv.density_scatter(
    df=df_publication,
    x="Atomic Radius (Å)",
    y="Melting Point (K)",
    facet_col="Element Group",
    hover_format=".0f",
    n_bins=250,
    log_density=True,
    identity_line=False,
    title="Atomic Radius vs Melting Point by Element Group (75k points)",
)
fig.layout.update(margin_t=60, title_x=0.5)
fig.show()


# %% Large data scale: 1 million points
np_rng = np.random.default_rng(seed=999)
n_samples = 1_000_000

x1 = np_rng.normal(-5, 1, n_samples // 4)
y1 = np_rng.normal(-5, 1, n_samples // 4)
x2 = np_rng.normal(5, 1, n_samples // 4)
y2 = np_rng.normal(5, 1, n_samples // 4)
x3 = np_rng.normal(0, 5, n_samples // 2)
y3 = x3 + np_rng.normal(0, 2, n_samples // 2)

x = np.concatenate([x1, x2, x3])
y = np.concatenate([y1, y2, y3])

df_massive = pd.DataFrame({"X Coordinate": x, "Y Coordinate": y})

fig = pmv.density_scatter(
    df=df_massive,
    x="X Coordinate",
    y="Y Coordinate",
    n_bins=300,
    hover_format=".0f",
    log_density=True,
    identity_line=False,
    title="Large Data: 1 Million Points",
)
fig.layout.update(margin_t=40, title_x=0.5)
fig.show()
