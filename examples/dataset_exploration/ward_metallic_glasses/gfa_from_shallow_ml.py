"""This script uses simple ML methods to try to predict glass-forming ability (GFA)
from composition.
"""

# %%
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml  # Add YAML import
from pymatgen.core import Composition
from pymongo import MongoClient
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import pymatviz as pmv
from pymatviz.enums import Key


sys.path.append(os.path.dirname(module_dir := os.path.dirname(__file__)))

from ward_metallic_glasses import (
    create_canonical_splits,
    formula_features,
    synthetic_non_glasses,
)


# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor  # commented out as not currently used
module_dir = os.path.dirname(__file__)
pmv.set_plotly_template("pymatviz_white")
# sim_features_path = f"{module_dir}/tmp/ward_df_with_viscosities.csv"  # no longer used
mongo_uri = "mongodb+srv://ysanspeur:PerovNorm123*@production.fjgmii3.mongodb.net"
db_name = "ray_md_testing"
viscosity_coll_name = "viscosity_and_diffusivity_means"

# Connect to MongoDB and fetch viscosity data
client = MongoClient(mongo_uri)
viscosity_coll = client[db_name][viscosity_coll_name]
df_vd = (  # viscosity and diffusivity
    pd.DataFrame(
        viscosity_coll.find({}, projection=[Key.mat_id, "viscosity", "diffusivity"])
    )
    .set_index(Key.mat_id, drop=False)
    .drop(columns=["_id"])
)


# %% Load A2C comparison metrics
mongo_uri = "mongodb+srv://ysanspeur:PerovNorm123*@production.fjgmii3.mongodb.net"
db_name = "ward-gfa-prod"
model_key = "mace_mpa_0_medium_2025_03_03"
model_key = "FT_BMG_GN_S_OMat_grad_noquad_cutoff6_2025_03_04"
if "mace" in model_key:
    model_type = "MACE"
elif "GN" in model_key:
    model_type = "GemNet"
else:
    raise ValueError(f"Unknown {model_key=}")
coll_name = f"a2c_ward_bmgs_{model_key}"

client = MongoClient(mongo_uri)
a2c_coll = client[db_name][coll_name]
a2c_fields = [
    "result.material_id",
    "result.config.n_relaxed_structures",
    "result.comparison",
    "result.n_relaxed_structures",
    "result.crystalline.formula",
    "result.crystalline.symmetry.number",
    "result.crystalline.energy_per_atom",
    "result.crystalline.forces",
    "result.crystalline.stress",
    "result.amorphous.formula",
    "result.amorphous.energy_per_atom",
    "result.amorphous.pressure",
    "result.amorphous.density",
    "result.amorphous.forces",
    "result.amorphous.stress",
]
n_docs = a2c_coll.count_documents({})
print(f"Found {n_docs:,} documents in {coll_name}")
docs = list(a2c_coll.find({}, projection=a2c_fields))
df_a2c = pd.json_normalize(docs)
df_a2c.columns = df_a2c.columns.str.removeprefix("result.")
df_a2c = df_a2c.set_index(Key.mat_id, drop=False)
df_a2c["amorphous.max_force"] = (
    df_a2c["amorphous.forces"]
    .map(lambda forces: np.linalg.norm(forces, axis=1).max())
    .round(4)
)
n_dupe_ids = df_a2c.index.duplicated().sum()
print(f"Number of duplicate IDs: {n_dupe_ids:,}")
for dup_id in df_a2c.index[df_a2c.index.duplicated()]:
    n_dupe_docs = a2c_coll.count_documents({"result.material_id": dup_id})
    if n_dupe_docs > 1:
        print(f"Found {n_dupe_docs} documents with {dup_id=}")
        res = a2c_coll.delete_one({"result.material_id": dup_id})
        print(f"Deleted {res.deleted_count} document(s) with {dup_id=}")

# Load A2C+FIRE data from the specified collection
fire_model_key = "FT_BMG_GN_S_OMat_grad_noquad_cutoff6_2025_03_05"
fire_coll_name = f"a2c_ward_bmgs_{fire_model_key}"
fire_coll = client[db_name][fire_coll_name]
n_a2c_fire_docs = fire_coll.count_documents({})
print(f"Found {n_a2c_fire_docs:,} documents in {fire_coll_name}")
df_a2c_fire = pd.json_normalize(fire_coll.find({}, projection=a2c_fields))
df_a2c_fire.columns = df_a2c_fire.columns.str.removeprefix("result.")
df_a2c_fire = df_a2c_fire.set_index(Key.mat_id, drop=False)
df_a2c_fire["amorphous.max_force"] = (
    df_a2c_fire["amorphous.forces"]
    .map(lambda forces: np.linalg.norm(forces, axis=1).max())
    .round(4)
)
n_fire_dupe_ids = df_a2c_fire.index.duplicated().sum()
print(f"Number of duplicate IDs in FIRE collection: {n_fire_dupe_ids:,}")
df_a2c_fire = df_a2c_fire.loc[~df_a2c_fire.index.duplicated(keep="first")]

# leave-one-out element and chemical system analysis
N_L1O_ELEMENTS = 3
N_L1O_SYSTEMS = 3
TEST_SIZE = 0.2  # fraction of data to use for testing in random splits
# number of random binary compositions to add as negative examples
N_RANDOM_NEG_EXAMPLES = 0  # Temporarily set to 0 to disable random negative examples
N_RANDOM_TERNARY_NEG_EXAMPLES = (
    0  # Temporarily set to 0 to disable random negative examples
)
MAX_RADIUS_DIFF_PERCENT = 5.0  # maximum allowed difference in atomic radii (%)

# Define models for each target type
regression_models: dict[str, BaseEstimator] = {
    # RandomForestRegressor.__name__: RandomForestRegressor(
    #     n_estimators=100, random_state=0
    # ),
    # XGBRegressor.__name__: XGBRegressor(n_estimators=100, random_state=0),
}
fitted_regressors: dict[
    str, dict[str, BaseEstimator]
] = {}  # split_name -> model_name -> model

classification_models = {
    RandomForestClassifier.__name__: RandomForestClassifier(
        n_estimators=100, random_state=0
    ),
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
    feature_set: str = "default",
) -> tuple[go.Figure | None, dict[str, float]]:
    """Evaluate a model on a train/test split.

    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
        target: Name of the target variable
        split_name: Name of the split
        n_random_samples: Number of random samples to plot
        y_train: Training set values (for periodic table plot)
        feature_set: Name of the feature set used

    Returns:
        fig: Figure object for the confusion matrix or parity plot
        metrics: Dictionary of metrics for this evaluation
    """
    # Create a metrics dictionary to store all evaluation metrics
    metrics = {}

    if target == "is_glass":
        # Classification metrics
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            # For string labels, we need to specify the pos_label
            pos_label = "glass"  # Assuming 'glass' is the positive class

            metrics["accuracy"] = accuracy_score(y_true, y_pred)
            metrics["precision"] = precision_score(y_true, y_pred, pos_label=pos_label)
            metrics["recall"] = recall_score(y_true, y_pred, pos_label=pos_label)
            metrics["f1"] = f1_score(y_true, y_pred, pos_label=pos_label)
            # For ROC AUC, we need to convert labels to binary
            y_true_binary = np.array([1 if y == pos_label else 0 for y in y_true])
            y_pred_binary = np.array([1 if y == pos_label else 0 for y in y_pred])
            metrics["roc_auc"] = roc_auc_score(y_true_binary, y_pred_binary)
            metrics["prc_auc"] = average_precision_score(y_true_binary, y_pred_binary)
        else:
            metrics |= dict.fromkeys(
                ("accuracy", "precision", "recall", "f1", "roc_auc", "prc_auc"), np.nan
            )

        # Create confusion matrix
        fig = pmv.confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            metrics_kwargs=dict(
                y=1.5,  # Increased from 1.3 to avoid overlap with title
                x=0.5,
                showarrow=False,
                font=dict(size=16),
                align="center",
            ),
        )

        # Update the layout with title and other properties
        fig.update_layout(
            title=dict(text=f"{model_name} - {split_name} - {feature_set}", x=0.5),
            margin=dict(t=250, l=50, r=50, b=50),  # Increased top margin
            width=800,
            height=700,
            paper_bgcolor="white",
        )
    else:
        # Regression metrics
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["r2"] = (
            r2_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan
        )

        # Create parity plot
        df_plot = pd.DataFrame({"actual": y_true, "predicted": y_pred})

        fig = px.scatter(
            df_plot,
            x="actual",
            y="predicted",
            title=f"{model_name} - {split_name} - {target} - {feature_set}",
        )
        fig.update_layout(
            margin=dict(t=100, l=50, r=50, b=50),
            width=800,
            height=700,
            paper_bgcolor="white",
        )

    return fig, metrics


def clean_features_and_target(
    features_df: pd.DataFrame,
    target_series: pd.Series | np.ndarray,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series | np.ndarray]:
    """Clean features dataframe and target series by removing NaN, Inf, and values
    too large for float32.
    """
    # Make a copy to avoid modifying the original
    features = features_df.copy()

    # Determine columns to drop
    cols_to_drop = [target_col]

    # Drop specified columns
    features = features.drop(columns=cols_to_drop, errors="ignore")

    # Create mask for invalid values
    invalid_mask = (
        features.isna().any(axis=1)
        | np.isinf(features).any(axis=1)
        | (features.abs() > np.finfo(np.float32).max).any(axis=1)
    )

    # Filter out invalid rows
    cleaned_features = features[~invalid_mask].copy()
    # Convert to float32 to match model expectations
    cleaned_features = cleaned_features.astype(np.float32)
    # Also filter the target using the same mask
    cleaned_target = target_series[~invalid_mask].copy()

    return cleaned_features, cleaned_target


def find_common_valid_subset(
    data_df: pd.DataFrame,
    feature_sets: dict[str, pd.DataFrame],
    target_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Find the subset of data where all feature sets and targets have valid values.

    Args:
        data_df: DataFrame containing the data
        feature_sets: Dictionary of feature set name -> feature DataFrame
        target_cols: List of target column names

    Returns:
        Tuple of (filtered_data_df, filtered_feature_sets)
    """
    # Start with all rows
    valid_indices = set(data_df.index)

    # Filter for valid target values
    for col in target_cols:
        valid_indices &= set(data_df[~data_df[col].isna()].index)

    # Filter for valid feature values in each feature set
    for df in feature_sets.values():
        valid_indices &= set(df.dropna().index)

    # Apply the filter to data_df and all feature sets
    filtered_data_df = data_df.loc[list(valid_indices)]
    filtered_feature_sets = {
        name: df.loc[list(valid_indices)] for name, df in feature_sets.items()
    }

    return filtered_data_df, filtered_feature_sets


def evaluate_model_on_split(
    df_feat_target: pd.DataFrame,
    y: pd.Series | np.ndarray,
    train_mask: pd.Series,
    test_mask: pd.Series,
    target: str,
    split_name: str,
    feature_set_name: str,
    models: dict[str, BaseEstimator],
    all_plots: dict[str, dict[str, dict[str, go.Figure]]],
    all_metrics: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]],
) -> None:
    """Evaluate models on a specific train/test split.

    Args:
        df_feat_target: DataFrame containing features
        y: Target values
        train_mask: Boolean mask for training data
        test_mask: Boolean mask for test data
        target: Target column name
        split_name: Name of the split
        feature_set_name: Name of the feature set
        models: Dictionary of model name -> model
        all_plots: Dictionary to store plots
        all_metrics: Dictionary to store metrics
    """
    y_train = y[train_mask]
    y_test = y[test_mask]

    print(f"Training size: {train_mask.sum()}, Test size: {test_mask.sum()}")

    if split_name not in all_plots:
        all_plots[split_name] = {}

    for model_name, model in models.items():
        if model_name not in all_plots[split_name]:
            all_plots[model_name][split_name] = {}

        # Clean and prepare training data
        train_features, y_train_clean = clean_features_and_target(
            df_feat_target.loc[train_mask], y_train, target
        )

        # Fit the model
        model.fit(train_features, y_train_clean)

        # Clean and prepare test data
        test_features, y_test_clean = clean_features_and_target(
            df_feat_target.loc[test_mask], y_test, target
        )

        # Make predictions
        preds = model.predict(test_features)

        # Evaluate the model
        fig, metrics = evaluate_split(
            y_test_clean,
            preds,
            model_name,
            target,
            split_name,
            feature_set=feature_set_name,
        )

        # Store results
        all_plots[model_name][split_name][feature_set_name] = fig

        # Store metrics
        all_metrics[target][model_name][split_name][feature_set_name] = metrics | {
            "train_size": len(train_features),
            "test_size": len(test_features),
            "n_features": train_features.shape[1],
        }


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
    binary_interpolations = formula_features.load_binary_liquidus_data(zip_path)

    # Calculate Liu features
    df_liu = formula_features.liu_featurize(
        df_ward, binary_liquidus_data=binary_interpolations
    )

# Add classification target
df_liu["is_glass"] = df_ward["is_glass"]

# List of targets (both regression and classification)
all_targets = ["dTx", "D_max", "is_glass"]

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


# Define columns to exclude from feature sets (target columns and non-feature columns)
feature_exclude_cols = [
    Key.composition,
    "D_max",
    "dTx",
    "is_glass",
    "gfa_type",
    "Unnamed: 4",
    "comment",
]
# Map simulation features to ward dataset
df_ward_with_vd_fea = df_ward.copy()
df_ward_with_vd_fea[df_vd.columns] = df_vd


# Dictionary to store all plots, organized by split_name -> model_name -> feature_set
all_plots: dict[str, dict[str, dict[str, go.Figure]]] = defaultdict(dict)

# Get A2C comparison metrics
comparison_cols = df_a2c.filter(like="comparison.").columns
print(f"\nFound {len(comparison_cols)} A2C comparison metrics as features:")
for col in comparison_cols:
    print(f"- {col}")

# Get A2C+FIRE comparison metrics
fire_comparison_cols = df_a2c_fire.filter(like="comparison.").columns
# Create A2C features DataFrame
df_a2c_features = df_a2c[comparison_cols].copy()

# Create A2C+FIRE features DataFrame
df_a2c_fire_features = df_a2c_fire[fire_comparison_cols].copy()

# Keep the first occurrence of each duplicate index
df_a2c_fire_features = df_a2c_fire_features[
    ~df_a2c_fire_features.index.duplicated(keep="first")
]

# Ensure all feature dataframes have the same index structure
# First, get the index from df_liu which will be our reference
liu_index = df_liu.index

# Reindex all feature dataframes to match df_liu's index
# For dataframes that don't have all indices, NaN values will be introduced
df_ward_with_vd_reindexed = df_ward_with_vd_fea.reindex(liu_index)
df_a2c_features_reindexed = df_a2c_features.reindex(liu_index)
df_a2c_fire_features_reindexed = df_a2c_fire_features.reindex(liu_index)

# Create combined feature sets with aligned indices
# Liu + VD features
df_liu_vd = pd.concat([df_ward_with_vd_reindexed, df_liu], axis=1)
df_liu_vd = df_liu_vd.loc[:, ~df_liu_vd.columns.duplicated()]

# A2C + Liu features
df_a2c_liu = pd.concat([df_a2c_features_reindexed, df_liu], axis=1)
df_a2c_liu = df_a2c_liu.loc[:, ~df_a2c_liu.columns.duplicated()]

# A2C + Liu + VD features
df_a2c_liu_vd = pd.concat(
    [df_a2c_features_reindexed, df_liu, df_ward_with_vd_reindexed],
    axis=1,
)
df_a2c_liu_vd = df_a2c_liu_vd.loc[:, ~df_a2c_liu_vd.columns.duplicated()]

# A2C+FIRE features
df_a2c_fire_only = df_a2c_fire_features_reindexed.copy()

# Create one-hot encoding features
print("\nCreating one-hot encoding features...")
# Use formula_features.one_hot_encode but extract only the element columns
df_onehot = formula_features.one_hot_encode(df_liu[["composition"]])
# Get only the element columns (exclude any non-numeric columns)
print(f"One-hot encoding features: {df_onehot.shape[1]} features")


# Define feature sets
feature_sets = {
    "Liu": df_liu,
    "VD": df_ward_with_vd_reindexed,
    "Liu+VD": df_liu_vd,
    "OneHot": df_onehot,  # Add one-hot encoding as a separate feature set
    "A2C": df_a2c_features_reindexed,
    "A2C+Liu": pd.concat([df_a2c_features_reindexed, df_liu], axis=1),
    "A2C+Liu+VD": df_a2c_liu_vd,
    "A2C+OneHot": pd.concat([df_a2c_features_reindexed, df_onehot], axis=1),
    "A2C+FIRE": df_a2c_fire_only,
}

# Create a dictionary to store all metrics
all_metrics: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = defaultdict(
    lambda: defaultdict(lambda: defaultdict(dict))
)

# Print feature set sizes
print("\nFeature set sizes:")
for model_name, df in feature_sets.items():
    print(f"- {model_name}: {df.shape[1]} features")

for target in all_targets:
    y = df_liu[target]
    print(f"\n--- Predicting {target} ---")

    # Select appropriate models based on target type
    models = classification_models if target == "is_glass" else regression_models

    for feature_set_name, df_raw in feature_sets.items():
        print(f"\n=== Using feature set: {feature_set_name} ===")

        # Double-check that target column is not in features
        df_features = df_raw.drop(
            columns=[target, "_id", Key.mat_id, Key.composition, *feature_exclude_cols],
            errors="ignore",
        )

        # For classification target, add random negative examples and encode labels
        if target == "is_glass":
            # Make copies to avoid modifying originals since we'll be adding examples
            df_feat_target = df_features.copy()
            data_df_for_comps = df_liu.copy()
            n_random = 0

            # Drop rows with NaN values in target
            valid_mask = ~data_df_for_comps["is_glass"].isna()
            df_feat_target = df_feat_target.loc[valid_mask]
            data_df_for_comps = data_df_for_comps.loc[valid_mask]

            if N_RANDOM_NEG_EXAMPLES > 0 or N_RANDOM_TERNARY_NEG_EXAMPLES > 0:
                print(
                    f"\nAdding {N_RANDOM_NEG_EXAMPLES} random binary compositions and "
                    f"{N_RANDOM_TERNARY_NEG_EXAMPLES} random ternary compositions with "
                    f"similar atomic radii (≤{MAX_RADIUS_DIFF_PERCENT}% difference) as "
                    "negative examples"
                )
                # Add random examples and get updated DataFrames
                data_df_for_comps, df_feat_target = (
                    synthetic_non_glasses.add_random_negative_examples(
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
            # Use .loc to avoid the IndexingError
            df_feat_target = df_features.loc[non_nan_mask].copy()
            y = y[non_nan_mask]
            data_df_for_comps = df_liu.loc[
                non_nan_mask
            ].copy()  # Filter data_df as well
            n_random = 0

        # 1. Random train/test split
        print("\n=== Random train/test split ===")
        df_train, df_test, y_train, y_test = train_test_split(
            df_feat_target, y, test_size=TEST_SIZE, random_state=0
        )

        # Create masks for the random split
        train_indices = df_train.index
        test_indices = df_test.index
        train_mask = df_feat_target.index.isin(train_indices)
        test_mask = df_feat_target.index.isin(test_indices)

        # Evaluate models on random split
        evaluate_model_on_split(
            df_feat_target=df_feat_target,
            y=y,
            train_mask=train_mask,
            test_mask=test_mask,
            target=target,
            split_name="Random split",
            feature_set_name=feature_set_name,
            models=models,
            all_plots=all_plots,
            all_metrics=all_metrics,
        )

        # 2. Leave-one-element-out testing
        print("\n=== Leave-one-element-out testing ===")
        for holdout_elem, _ in most_common_elements[
            :3
        ]:  # Limit to top 3 elements for efficiency
            # Create masks for the leave-one-element-out split
            test_mask = data_df_for_comps[Key.composition].map(
                lambda comp, holdout=holdout_elem: holdout
                in Composition(comp).chemical_system_set
            )
            train_mask = ~test_mask

            if test_mask.sum() == 0:
                print(f"No entries found containing {holdout_elem}")
                continue

            # Evaluate models on leave-one-element-out split
            evaluate_model_on_split(
                df_feat_target=df_feat_target,
                y=y,
                train_mask=train_mask,
                test_mask=test_mask,
                target=target,
                split_name=f"Hold out {holdout_elem}",
                feature_set_name=feature_set_name,
                models=models,
                all_plots=all_plots,
                all_metrics=all_metrics,
            )

        # 3. Leave-one-system-out testing
        print("\n=== Leave-one-system-out testing ===")
        for idx, (holdout_chem_sys, _) in enumerate(most_common_systems[:3]):
            print(f"Testing system {idx + 1}/3: {holdout_chem_sys}")

            # Create masks for the leave-one-system-out split
            test_mask = data_df_for_comps[Key.composition].map(
                lambda comp, holdout=holdout_chem_sys: Composition(
                    comp
                ).chemical_system_set
                == set(holdout.split("-"))
            )
            train_mask = ~test_mask

            if test_mask.sum() == 0:
                print(f"No entries found for system {holdout_chem_sys}")
                continue

            # Evaluate models on leave-one-system-out split
            evaluate_model_on_split(
                df_feat_target=df_feat_target,
                y=y,
                train_mask=train_mask,
                test_mask=test_mask,
                target=target,
                split_name=f"Hold out {holdout_chem_sys}",
                feature_set_name=feature_set_name,
                models=models,
                all_plots=all_plots,
                all_metrics=all_metrics,
            )


# Display plots grouped by split and model
print("\n=== Displaying plots grouped by split and model ===")
for split_name, split_plots in all_plots.items():
    print(f"\n--- {split_name} ---")
    for model_name, model_plots in split_plots.items():
        print(f"\n{model_name}:")
        for feature_set_name, fig in model_plots.items():
            if fig is None:
                continue
            print(f"Feature set: {feature_set_name}")
            # fig.show()  # Removing this line to avoid showing too many figures


metrics_file = f"{module_dir}/gfa_metrics.yaml"
serialized_metrics = json.loads(json.dumps(all_metrics))

with open(metrics_file, mode="w") as yaml_file:
    yaml.dump(serialized_metrics, yaml_file, default_flow_style=False)

print(f"\nMetrics saved to {metrics_file}")
