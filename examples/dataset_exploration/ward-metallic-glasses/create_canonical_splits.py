"""Create canonical train/validation/test splits for the Ward metallic glasses dataset.

The script creates three types of splits:
1. Random split (80/10/10)
2. Element-based split (leave out most common elements)
3. System-based split (leave out most common chemical systems)

For element and system splits, the held-out data is split 50/50 into validation and test
sets to maximize compositional diversity between the sets.
"""

import os
from collections import Counter
from typing import Any, Literal, get_args

import pandas as pd
from pymatgen.core import Composition
from sklearn.model_selection import train_test_split

import pymatviz as pmv
from pymatviz.enums import ElemCountMode, Key


DataSplit = Literal["train", "val", "test"]
data_splits = get_args(DataSplit)
module_dir = os.path.dirname(__file__)


def get_most_common_elements(df_in: pd.DataFrame, n_elements: int = 3) -> list[str]:
    """Get the n most common elements in the dataset."""
    element_counts: Counter[str] = Counter()
    for comp in df_in[Key.composition]:
        element_counts.update(Composition(comp).chemical_system_set)
    return [el for el, _ in element_counts.most_common(n_elements)]


def get_most_common_systems(df_in: pd.DataFrame, n_systems: int = 3) -> list[str]:
    """Get the N most common chemical systems in the dataset."""
    system_counts: Counter[str] = Counter()
    for comp in df_in[Key.composition]:
        system_counts[Composition(comp).chemical_system] += 1
    return [sys for sys, _ in system_counts.most_common(n_systems)]


def create_random_split(
    df_in: pd.DataFrame, train_size: float = 0.8, **kwargs: Any
) -> pd.Series:
    """Create random train/val/test split.

    Args:
        df_in: Input DataFrame
        train_size: Fraction of data to use for training. Default is 0.8.
        kwargs: Passed to sklearn.model_selection.train_test_split

    Returns:
        Series with values "train"/"val"/"test" for each row
    """
    # First split into train and temp (val + test)
    train_idx, temp_idx = train_test_split(df_in.index, train_size=train_size, **kwargs)

    # Split temp into val and test (50/50)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, **kwargs)

    split = pd.Series("train", index=df_in.index)
    split[val_idx] = "val"
    split[test_idx] = "test"

    return split


def create_element_split(
    df_in: pd.DataFrame, elements: list[str], random_state: int = 0
) -> pd.Series:
    """Create element-based train/val/test split.

    Args:
        df_in: Input DataFrame
        elements: List of elements to hold out
        random_state: Random seed for reproducibility

    Returns:
        Series with values "train"/"val"/"test" for each row
    """
    # Initialize all as train
    split = pd.Series("train", index=df_in.index)

    # Find compositions containing any of the held-out elements
    holdout_mask = df_in[Key.composition].apply(
        lambda comp: any(el in Composition(comp).chemical_system_set for el in elements)
    )
    holdout_idx = df_in.index[holdout_mask]

    # Split holdout indices into val and test
    if len(holdout_idx) > 0:
        val_idx, test_idx = train_test_split(
            holdout_idx, test_size=0.5, random_state=random_state
        )
        split[val_idx] = "val"
        split[test_idx] = "test"

    return split


def create_system_split(
    df_in: pd.DataFrame, systems: list[str], random_state: int = 0
) -> pd.Series:
    """Create system-based train/val/test split.

    Args:
        df_in: Input DataFrame
        systems: List of chemical systems to hold out
        random_state: Random seed for reproducibility

    Returns:
        Series with values "train"/"val"/"test" for each row
    """
    # Initialize all as train
    split = pd.Series("train", index=df_in.index)

    # Find compositions matching any of the held-out systems
    holdout_mask = df_in[Key.composition].apply(
        lambda x: "-".join(sorted(Composition(x).chemical_system_set)) in systems
    )
    holdout_idx = df_in.index[holdout_mask]

    # Split holdout indices into val and test
    if len(holdout_idx) > 0:
        val_idx, test_idx = train_test_split(
            holdout_idx, test_size=0.5, random_state=random_state
        )
        split[val_idx] = "val"
        split[test_idx] = "test"

    return split


def print_split_stats(df_in: pd.DataFrame, split: pd.Series, split_name: str) -> None:
    """Print statistics about a data split and create visualization."""
    print(f"\n=== {split_name} split ===")
    print("Split sizes:")
    split_counts = split.value_counts()
    for name, count in split_counts.items():
        print(f"{name}: {count:,} ({count / len(df_in):.1%})")

    # Count elements in each split
    split_elements: dict[DataSplit, set[str]] = {name: set() for name in data_splits}
    split_comps: dict[DataSplit, list[str]] = {name: [] for name in data_splits}

    for idx, comp in df_in[Key.composition].items():
        split_comps[split[idx]].append(comp)
        split_elements[split[idx]].update(Composition(comp).chemical_system_set)

    # Count element occurrences using pmv.count_elements
    split_counts = {
        name: pmv.count_elements(comps, count_mode=ElemCountMode.occurrence)
        for name, comps in split_comps.items()
    }

    print("\nUnique elements in each split:")
    for name, elements in split_elements.items():
        print(f"{name}: {len(elements)} elements")

    # Count unique systems in each split
    split_systems: dict[DataSplit, set[str]] = {name: set() for name in data_splits}
    for idx, comp in df_in[Key.composition].items():
        system = "-".join(sorted(Composition(comp).chemical_system_set))
        split_systems[split[idx]].add(system)

    print("\nUnique systems in each split:")
    for name, systems in split_systems.items():
        print(f"{name}: {len(systems)} systems")

    # Create element counts dictionary for visualization
    # Format: {element: (train_count, val_count, test_count)}
    element_split_counts = {}
    all_elements = set().union(*split_elements.values())

    for el in all_elements:
        train_count = split_counts["train"].get(el, 0)
        val_count = split_counts["val"].get(el, 0)
        test_count = split_counts["test"].get(el, 0)
        element_split_counts[el] = (train_count, val_count, test_count)

    common_kwargs = dict(y=0.75, len=0.17, tickangle=0, orientation="h")
    # Create periodic table heatmap
    fig = pmv.ptable_heatmap_splits_plotly(
        element_split_counts,
        colorbar=(
            dict(title="Train count<br>", x=0.2, tickformat="~s", **common_kwargs),
            dict(title="Val count<br>", x=0.38, tickformat="~s", **common_kwargs),
            dict(title="Test count<br>", x=0.56, tickformat="~s", **common_kwargs),
        ),
        nan_color="rgba(0,0,0,0)",  # Make zero-value tiles transparent
        symbol_kwargs=dict(font_color="gray"),
        hover_fmt=lambda x: f"{x:,.0f}",  # Format hover values with thousands separator
    )
    title = f"Element counts in train/val/test splits for {split_name}"
    fig.layout.margin.update(t=10, b=10, l=10, r=10)
    fig.layout.title.update(text=title, x=0.5, y=0.95)
    fig.layout.paper_bgcolor = "white"
    fig.show()
    os.makedirs(f"{module_dir}/figs/splits", exist_ok=True)
    pmv.save_fig(
        fig, f"{module_dir}/figs/splits/{split_name}-element-counts.png", scale=2
    )


# Load dataset
ward_csv_path = f"{module_dir}/ward-metallic-glasses.csv.xz"
df_ward = pd.read_csv(
    ward_csv_path, na_values=["Unknown", "DifferentMeasurement?"]
).query("comment.isna()")

# 1. Random split (80/10/10)
random_split = create_random_split(df_ward, train_size=0.8, random_state=0)
print_split_stats(df_ward, random_split, "holdout Random")
df_ward["split_random"] = random_split

# 2. Element-based splits (one for each of the 3 most common elements)
n_elements = 3
most_common_elements = get_most_common_elements(df_ward, n_elements)
print(f"\nMost common elements: {', '.join(most_common_elements)}")

for element in most_common_elements:
    split = create_element_split(df_ward, [element], random_state=0)
    split_name = f"split_holdout_element_{element}"
    print_split_stats(df_ward, split, f"holdout {element}")
    df_ward[split_name] = split

# 3. System-based splits (one for each of the 3 most common systems)
n_systems = 3
most_common_systems = get_most_common_systems(df_ward, n_systems)
print(f"\nMost common systems: {', '.join(most_common_systems)}")

for system in most_common_systems:
    split = create_system_split(df_ward, [system], random_state=0)
    split_name = f"split_holdout_chem_sys_{system}"
    print_split_stats(df_ward, split, f"holdout {system}")
    df_ward[split_name] = split

# Save augmented dataset
out_path = f"{module_dir}/ward-metallic-glasses-with-splits.csv.xz"
df_ward.to_csv(out_path, index=False)
print(f"\nSaved augmented dataset to {out_path}")

# Print column names for verification
split_cols = [col for col in df_ward.columns if col.startswith("split_")]
print("\nSplit columns in output dataset:")
for col in split_cols:
    print(f"- {col}")
