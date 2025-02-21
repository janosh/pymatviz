"""This script uses simple ML methods to try to predict glass-forming ability (GFA)
from composition.
"""

# %%
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pymatgen.core import Composition
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

import pymatviz as pmv
from examples.dataset_exploration.ward_metallic_glasses import (
    create_canonical_splits,
    formula_features,
    negative_examples,
)
from examples.dataset_exploration.ward_metallic_glasses.formula_features import (
    liu_featurize,
    load_binary_liquidus_data,
)
from pymatviz.enums import Key


# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor  # commented out as not currently used


module_dir = os.path.dirname(__file__)
pmv.set_plotly_template("pymatviz_white")
sim_features_path = f"{module_dir}/ward_df_with_viscosities.csv"
df_sim = pd.read_csv(sim_features_path)


# Configuration
N_L1O_ELEMENTS = 3
N_L1O_SYSTEMS = 0
TEST_SIZE = 0.2  # fraction of data to use for testing in random splits
# number of random binary compositions to add as negative examples
N_RANDOM_NEG_EXAMPLES = 10_000
N_RANDOM_TERNARY_NEG_EXAMPLES = 5_000  # number of random ternary compositions to add
MAX_RADIUS_DIFF_PERCENT = 5.0  # maximum allowed difference in atomic radii (%)

# Define models for each target type
regression_models: dict[str, BaseEstimator] = {
    # "RandomForest": RandomForestRegressor(n_estimators=100, random_state=0),
    # "XGBoost": XGBRegressor(n_estimators=100, random_state=0),
}
fitted_regressors: dict[
    str, dict[str, BaseEstimator]
] = {}  # split_name -> model_name -> model

classification_models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=0),
}
fitted_classifiers: dict[
    str, dict[str, BaseEstimator]
] = {}  # split_name -> model_name -> model


def evaluate_split(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str,
    target: str,
    split_name: str,
    compositions: pd.Series | None = None,
    element_split_counts: dict[str, tuple[int, int]] | None = None,
    n_random_samples: int = 0,
    y_train: pd.Series | np.ndarray | None = None,
) -> tuple[go.Figure | None, go.Figure | None]:
    """Evaluate model predictions and create visualization plots.

    This function handles both classification and regression tasks differently:
    - For classification (target='is_glass'):
        - Generates a confusion matrix with counts and percentages
        - Shows accuracy, precision and F1 score
        - Optionally indicates number of random negative examples added
    - For regression (target='dTx' or 'D_max'):
        - Creates a parity plot of predicted vs actual values
        - Shows R² score and mean absolute error
        - Adds composition hover data if provided

    The function also creates a periodic table heatmap showing the distribution
    of elements between train and test sets, but only once per split type
    (since it's the same for all models in a given split).

    Args:
        y_true: Ground truth target values
        y_pred: Model predictions
        model_name: Name of the model (e.g. 'RandomForest')
        target: Name of target variable ('is_glass', 'dTx', or 'D_max')
        split_name: Description of the split type (e.g. 'Random split',
            'Leave out Fe')
        compositions: Composition strings for hover data in parity plots
        element_split_counts: Element counts in train/test sets for periodic
            table plot. Dict mapping element symbols to (test_count, train_count)
            tuples.
        n_random_samples: Number of random negative examples added (only relevant
            for classification)
        y_train: Training set labels (only needed for classification to show
            class distribution)

    Returns:
        A tuple of (main_plot, ptable_plot) where:
        - main_plot is either a confusion matrix (classification) or parity plot
          (regression)
        - ptable_plot is a periodic table heatmap showing element distributions
          between train and test sets (only generated once per split type)
    """
    figs_dir = f"{module_dir}/figs/{split_name}"
    os.makedirs(figs_dir, exist_ok=True)

    # Convert inputs to pandas Series for easier handling
    y_true, y_pred = map(pd.Series, (y_true, y_pred))
    if y_train is not None:
        y_train = pd.Series(y_train)

    # Drop NaN values
    valid_mask = ~(y_true.isna() | y_pred.isna())
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    if compositions is not None:
        # Ensure compositions has the same index as y_true/y_pred before filtering
        compositions = pd.Series(compositions)
        compositions.index = y_true.index
        compositions = compositions[valid_mask]

    if len(y_true) == 0:
        print("No valid predictions after dropping NaN values")
        return None, None

    # Plot train/test element prevalence once since it's the same for all models
    fig_ptable = None
    ptable_pdf_path = f"{figs_dir}/ptable-train-test-split.svg"
    if element_split_counts is not None and not os.path.isfile(ptable_pdf_path):
        fig_ptable = pmv.ptable_heatmap_splits_plotly(element_split_counts)
        title = f"Element counts in train (top) and test (bottom)<br>{split_name}"
        fig_ptable.layout.title.update(text=title, x=0.4, y=0.9)
        fig_ptable.layout.paper_bgcolor = "white"
        fig_ptable.show()

        # Save ptable plot
        # fig_ptable.write_image(ptable_pdf_path)

    # Handle classification and regression metrics differently
    if target == "is_glass":
        # Create confusion matrix plot
        title = f"{model_name}, {split_name}"
        if n_random_samples > 0:
            title += f"<br>({n_random_samples:,} random negative examples)"

        # Add training set statistics if y_train is provided
        if y_train is not None:
            # Drop NaN values from y_train
            y_train = y_train[~y_train.isna()]
            if len(y_train) > 0:
                n_pos = (y_train == "glass").sum()
                n_neg = (y_train == "crystal").sum()
                total = len(y_train)
                title += (
                    f"<br>Training set: {n_pos:,} glass ({n_pos / total:.1%}), "
                    f"{n_neg:,} crystal ({n_neg / total:.1%})"
                )

        fig = pmv.confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            metrics={"Acc": ".1%", "Prec": ".1%", "F1": ".2f"},
            metrics_kwargs={"y": 1.3, "font_size": 18},
        )
        fig.layout.xaxis.title = "Predicted"
        fig.layout.yaxis.title = "Actual"
        fig.layout.update(
            title=dict(text=title, x=0.5, y=0.95, font_size=18, yanchor="top"),
            paper_bgcolor="white",
            margin=dict(t=200, l=100),  # increase top and left margins
            font=dict(size=18),
        )
        # img_path = f"{figs_dir}/confusion-matrix-{target}-{model_name}.svg"
        # fig.write_image(img_path)
        fig.show()

        return fig, fig_ptable

    # Regression metrics
    print(
        f"{model_name}, {split_name}: "
        f"R² = {r2_score(y_true, y_pred):.3f}, "
        f"MAE = {mean_absolute_error(y_true, y_pred):.3f}"
    )

    # Create DataFrame for plotting with aligned indices
    plot_data = {"Actual": y_true, "Predicted": y_pred}
    if compositions is not None:
        plot_data["Composition"] = compositions

    df_plot = pd.DataFrame(plot_data)

    # Plot parity plot
    hover_data = ["Composition"] if compositions is not None else None
    fig = px.scatter(df_plot, x="Actual", y="Predicted", hover_data=hover_data)
    fig = pmv.powerups.enhance_parity_plot(fig)
    title = f"{model_name}, {split_name}"
    fig.layout.title = dict(text=title, x=0.5, y=0.96)
    fig.layout.margin = dict(l=0, r=0, t=50, b=0)
    fig.layout.paper_bgcolor = "white"
    fig.show()

    # Save parity plot
    # fig.write_image(f"{figs_dir}/parity-plot-{target}-{model_name}.svg")

    return fig, fig_ptable


def map_sim_features_to_ward(
    df_ward: pd.DataFrame, df_sim: pd.DataFrame
) -> pd.DataFrame:
    """Map simulation features (diffusivity, viscosity) from df_sim to df_ward based on
    matching compositions.

    Args:
        df_ward: Ward metallic glasses dataset
        df_sim: Simulation dataset with diffusivity and viscosity data

    Returns:
        DataFrame with ward data plus mapped simulation features
    """
    # Create a copy of ward df to avoid modifying original
    df_ward_with_sim = df_ward.copy()

    # Initialize new columns with NaN
    df_ward_with_sim["diffusivity"] = np.nan
    df_ward_with_sim["viscosity"] = np.nan

    # Create normalized compositions for comparison
    ward_comps = df_ward[Key.composition].map(
        lambda comp: str(Composition(comp).fractional_composition)
    )
    sim_comps = df_sim["source_composition_wt%"].map(
        lambda comp: str(Composition(comp).fractional_composition)
    )

    # Find matching compositions
    for ward_idx, ward_comp in ward_comps.items():
        matches = sim_comps == ward_comp
        if matches.any():
            sim_idx = matches.idxmax()  # Get first match if multiple exist
            df_ward_with_sim.loc[ward_idx, "diffusivity"] = df_sim.loc[
                sim_idx, "diffusivity"
            ]
            df_ward_with_sim.loc[ward_idx, "viscosity"] = df_sim.loc[
                sim_idx, "viscosity"
            ]

    # Print mapping statistics
    n_mapped = (
        (~df_ward_with_sim[["diffusivity", "viscosity"]].isna()).all(axis=1).sum()
    )
    print(
        f"\nMapped {n_mapped} compositions out of {len(df_ward)} "
        f"({n_mapped / len(df_ward) * 100:.1f}%)"
    )

    return df_ward_with_sim


# --- Main benchmark routine ---
ward_csv_path = f"{module_dir}/ward-metallic-glasses.csv.xz"

# Load data and drop duplicates
df_ward = (
    pd.read_csv(ward_csv_path, na_values=["Unknown", "DifferentMeasurement?"])
    .query("comment.isna()")
    .set_index(Key.mat_id)
)
df_ward = create_canonical_splits.drop_dupe_formulas(df_ward)

# drop 1348 duplicates before analysis
if len(df_ward) != 6832:
    raise ValueError(f"{len(df_ward)=}, expected 6832")

target_cols = ccd_col, liquidus_temp_col = ["D_max", "dTx"]
# some rows with strange special characters in target_cols
df_ward[target_cols].isin(["–", "֠None"]).sum()  # noqa: RUF001
# Replace problematic values in dTx column with NaN
for col in target_cols:
    df_ward[col] = pd.to_numeric(df_ward[col], errors="coerce")
    print(f"{col}: {df_ward[col].isna().sum()}")


# Create binary classification target: non-ribbon (NaN) vs. ribbon/BMG
df_ward["is_glass"] = df_ward["gfa_type"].map(
    lambda gfa: "crystal" if pd.isna(gfa) else "glass"
)
print("\nGlass formation type distribution:")
print(df_ward["is_glass"].value_counts())

# Load binary liquidus data for Liu features
if (df_liu := locals().get("df_liu")) is None:
    zip_path = f"{module_dir}/tmp/binary-liquidus-temperatures.zip"
    binary_interpolations = load_binary_liquidus_data(zip_path)

    # Calculate Liu features
    df_liu = liu_featurize(df_ward, binary_liquidus_data=binary_interpolations)

# Add classification target
df_liu["is_glass"] = df_ward["is_glass"]

# List of targets (both regression and classification)
targets = ["dTx", "D_max", "is_glass"]

# Convert 'Unknown' to NaN and drop rows with NaN in target columns
for target in ["dTx", "D_max"]:  # only convert regression targets
    df_liu[target] = pd.to_numeric(df_liu[target], errors="coerce")

# Find most prevalent elements and systems
element_counts: dict[str, int] = {}
system_counts: dict[str, int] = {}
for comp in df_liu[Key.composition]:
    elements = Composition(comp).chemical_system_set
    system = "-".join(sorted(elements))

    for el in elements:
        element_counts[el] = element_counts.get(el, 0) + 1
    system_counts[system] = system_counts.get(system, 0) + 1

most_common_elements = sorted(element_counts.items(), key=lambda x: x[1], reverse=True)[
    :N_L1O_ELEMENTS
]
most_common_systems = sorted(system_counts.items(), key=lambda x: x[1], reverse=True)[
    :N_L1O_SYSTEMS
]

print("\nMost common elements:")
for el, count in most_common_elements:
    print(f"{el}: {count} compositions")

print("\nMost common systems:")
for system, count in most_common_systems:
    print(f"{system}: {count} compositions")

# Build features
df_feat = formula_features.one_hot_encode(df_liu)

# non-feature columns
feature_exclude_cols = [
    Key.composition,
    "D_max",
    "dTx",
    "is_glass",
    "gfa_type",
    "Unnamed: 4",
    "comment",
]
df_feat = df_feat.drop(columns=feature_exclude_cols)  # Keep only feature columns

# Map simulation features to ward dataset
df_ward_with_sim = map_sim_features_to_ward(df_ward, df_sim)

df_ward_with_sim.describe().to_string()

for target in targets:
    y = df_liu[target]
    print(f"\n--- Predicting {target} ---")

    # Select appropriate models based on target type
    models = classification_models if target == "is_glass" else regression_models
    fitted_models = fitted_classifiers if target == "is_glass" else fitted_regressors

    # For classification target, add random negative examples and encode labels
    if target == "is_glass":
        # Make copies to avoid modifying originals since we'll be adding examples
        df_feat_target = df_feat.copy()
        data_df_for_comps = df_liu.copy()
        n_random = 0

        # Drop rows with NaN values in target
        valid_mask = ~data_df_for_comps["is_glass"].isna()
        df_feat_target = df_feat_target[valid_mask]
        data_df_for_comps = data_df_for_comps[valid_mask]

        if N_RANDOM_NEG_EXAMPLES > 0 or N_RANDOM_TERNARY_NEG_EXAMPLES > 0:
            print(
                f"\nAdding {N_RANDOM_NEG_EXAMPLES} random binary compositions and "
                f"{N_RANDOM_TERNARY_NEG_EXAMPLES} random ternary compositions with "
                f"similar atomic radii (≤{MAX_RADIUS_DIFF_PERCENT}% difference) as "
                "negative examples"
            )
            # Add random examples and get updated DataFrames
            data_df_for_comps, df_feat_target = (
                negative_examples.add_random_negative_examples(
                    data_df=data_df_for_comps,  # use filtered data
                    df_feat=df_feat_target,  # use filtered features
                    n_binary_samples=N_RANDOM_NEG_EXAMPLES,
                    n_ternary_samples=N_RANDOM_TERNARY_NEG_EXAMPLES,
                    max_radius_diff_percent=MAX_RADIUS_DIFF_PERCENT,
                )
            )
            n_random = N_RANDOM_NEG_EXAMPLES + N_RANDOM_TERNARY_NEG_EXAMPLES

        y = data_df_for_comps["is_glass"]
    else:
        # For regression targets, filter out rows with NaN values
        non_nan_mask = ~df_liu[target].isna()
        df_feat_target = df_feat[non_nan_mask].copy()
        y = y[non_nan_mask]
        data_df_for_comps = df_liu[non_nan_mask].copy()  # Filter data_df as well
        n_random = 0

    # 1. Random train/test split
    print("\n=== Random train/test split ===")
    df_train, df_test, y_train, y_test = train_test_split(
        df_feat_target, y, test_size=TEST_SIZE, random_state=0
    )
    print(f"Training size: {len(df_train)}, Test size: {len(df_test)}")

    fitted_models["random"] = {}
    for name, model in models.items():
        if "is_glass" in df_train:
            raise ValueError("Target column found in training data")
        model.fit(df_train, y_train)
        fitted_models["random"][name] = model
        preds = model.predict(df_test)
        if target == "is_glass":
            evaluate_split(
                y_test,
                preds,
                name,
                target,
                "Random split",
                data_df_for_comps.loc[y_test.index, Key.composition],
                n_random_samples=n_random,
                y_train=y_train,
            )
        else:
            evaluate_split(
                y_test,
                preds,
                name,
                target,
                "Random split",
                data_df_for_comps.loc[y_test.index, Key.composition],
                y_train=None,
            )

    # 2. Leave-one-element-out testing
    print("\n=== Leave-one-element-out testing ===")
    for holdout_elem, _ in most_common_elements:
        # Create masks for the augmented dataset if using classification
        if target == "is_glass":
            test_mask = data_df_for_comps[Key.composition].map(
                lambda comp, holdout=holdout_elem: holdout
                in Composition(comp).chemical_system_set
            )
            train_mask = ~test_mask

            # Use the masks on the feature matrix and encoded labels
            X_train = df_feat_target[train_mask]
            X_test = df_feat_target[test_mask]
            y_train = y[train_mask]  # y is already encoded
            y_test = y[test_mask]
        else:
            test_mask = df_liu[Key.composition].map(
                lambda comp, holdout=holdout_elem: holdout
                in Composition(comp).chemical_system_set
            )
            train_mask = ~test_mask

            # For regression, use original features and targets
            X_train = df_feat_target[train_mask]
            X_test = df_feat_target[test_mask]
            y_train = y[train_mask]
            y_test = y[test_mask]

        if test_mask.sum() == 0:
            print(f"No entries found containing {holdout_elem}")
            continue

        print(f"Training size: {train_mask.sum()}, Test size: {test_mask.sum()}")

        # Count elements in train and test sets
        train_comps = data_df_for_comps[train_mask][Key.composition]
        test_comps = data_df_for_comps[test_mask][Key.composition]

        train_counts = pmv.count_elements(train_comps)
        test_counts = pmv.count_elements(test_comps)

        element_split_counts = {
            el: (test_counts.get(el, 0), train_counts.get(el, 0))
            for el in set(train_counts.index) | set(test_counts.index)
        }

        split_name = f"Leave out {holdout_elem}"
        fitted_models[split_name] = {}
        for name, model in models.items():
            if "is_glass" in X_train:
                raise ValueError("Target column found in training data")
            model.fit(X_train, y_train)
            fitted_models[split_name][name] = model
            preds = model.predict(X_test)
            if target == "is_glass":
                evaluate_split(
                    y_test,
                    preds,
                    name,
                    target,
                    split_name,
                    data_df_for_comps[test_mask][Key.composition],
                    element_split_counts,
                    n_random_samples=n_random,
                    y_train=y_train,
                )
            else:
                evaluate_split(
                    y_test,
                    preds,
                    name,
                    target,
                    split_name,
                    df_liu[test_mask][Key.composition],
                    element_split_counts,
                    y_train=None,
                )

    # 3. Leave-one-system-out testing
    print("\n=== Leave-one-system-out testing ===")
    for holdout_chem_sys, _ in most_common_systems:
        print(f"\nLeaving out system: {holdout_chem_sys}")

        # Create masks for the augmented dataset if using classification
        if target == "is_glass":
            test_mask = data_df_for_comps[Key.composition].map(
                lambda comp, holdout=holdout_chem_sys: Composition(
                    comp
                ).chemical_system_set
                == set(holdout.split("-"))
            )
            train_mask = ~test_mask
            y_train = y[train_mask]
            y_test = y[test_mask]
        else:
            test_mask = df_liu[Key.composition].map(
                lambda comp, holdout=holdout_chem_sys: Composition(
                    comp
                ).chemical_system_set
                == set(holdout.split("-"))
            )
            train_mask = ~test_mask
            y_train = y[train_mask]
            y_test = y[test_mask]

        if test_mask.sum() == 0:
            print(f"No entries found for system {holdout_chem_sys}")
            continue

        print(f"Training size: {train_mask.sum()}, Test size: {test_mask.sum()}")

        # Count elements in train and test sets
        train_comps = data_df_for_comps[train_mask][Key.composition]
        test_comps = data_df_for_comps[test_mask][Key.composition]

        train_counts = pmv.count_elements(train_comps)
        test_counts = pmv.count_elements(test_comps)

        element_split_counts = {
            el: (test_counts.get(el, 0), train_counts.get(el, 0))
            for el in set(train_counts.index) | set(test_counts.index)
        }

        split_name = f"Leave out {holdout_chem_sys}"
        fitted_models[split_name] = {}
        for name, model in models.items():
            if "is_glass" in df_feat_target:
                raise ValueError("Target column found in training data")
            model.fit(df_feat_target[train_mask], y_train)
            fitted_models[split_name][name] = model
            preds = model.predict(df_feat_target[test_mask])
            if target == "is_glass":
                evaluate_split(
                    y_test,
                    preds,
                    name,
                    target,
                    split_name,
                    data_df_for_comps[test_mask][Key.composition],
                    element_split_counts,
                    n_random_samples=n_random,
                    y_train=y_train,
                )
            else:
                evaluate_split(
                    y_test,
                    preds,
                    name,
                    target,
                    split_name,
                    df_liu[test_mask][Key.composition],
                    element_split_counts,
                    y_train=None,
                )
