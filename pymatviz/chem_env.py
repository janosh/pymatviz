"""Chemical environment analysis utilities.

Functions for analyzing coordination numbers and chemical environments
using pymatgen's ChemEnv and local environment modules.
"""

from __future__ import annotations

import textwrap
import warnings
from typing import TYPE_CHECKING, Any, Final, Literal, get_args

import pandas as pd
import plotly.graph_objects as go

from pymatviz.process_data import normalize_structures
from pymatviz.typing import ShowCounts


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pymatgen.core import IStructure, Structure

    from pymatviz.typing import AnyStructure

ChemEnvSettings = dict[str, Any] | Literal["chemenv", "crystal_nn"]

# Cell/slice text templates shared by chem_env_sunburst and chem_env_treemap
CHEM_ENV_TEXT_TEMPLATES: Final[dict[str | bool, str]] = {
    "value": "%{label}: %{value:.2f}",
    "percent": "%{label}: %{percentParent:.1%}",
    "value+percent": "%{label}: %{value:.2f} (%{percentParent:.1%})",
    False: "%{label}",
}


def get_cn_from_symbol(ce_symbol: str, symbol_cn_mapping: dict[str, int]) -> int:
    """Extract coordination number from ChemEnv symbol.

    Args:
        ce_symbol (str): ChemEnv symbol (e.g., 'T:4', 'O:6', 'M:8')
        symbol_cn_mapping (dict[str, int]): Mapping from symbols to coordination numbers

    Returns:
        int: Coordination number
    """
    if ce_symbol in symbol_cn_mapping:
        return symbol_cn_mapping[ce_symbol]

    if ce_symbol == "S:1":
        return 1

    if ce_symbol.startswith("M:"):
        try:
            cn_val = int(ce_symbol.split(":")[1])
        except ValueError:
            return 0
        return max(0, cn_val)

    return 0


def classify_local_env_with_order_params(
    structure: Structure | IStructure, site_idx: int, cn_val: int
) -> str:
    """Classify local coordination environment using LocalStructOrderParams.

    Based on Crystal Toolkit's approach using pymatgen's LocalStructOrderParams
    to calculate order parameters for different coordination geometries.

    Args:
        structure (Structure | IStructure): The crystal structure
        site_idx (int): Index of the site to analyze
        cn_val (int): Coordination number of the site

    Returns:
        str: String describing the coordination environment (e.g. "T:4", "O:6", "CN:8")
    """
    from pymatgen.analysis import local_env
    from pymatgen.core import Structure

    if not 0 <= site_idx < len(structure):
        raise IndexError(
            f"{site_idx=} out of range for structure with {len(structure)} sites"
        )

    try:
        analysis_structure = Structure.from_sites(structure)

        # Check if we have order parameters for this coordination number
        cn_params = local_env.CN_OPT_PARAMS.get(cn_val)
        if cn_params is None:
            return f"CN:{cn_val}"

        # Get the parameter names and settings for this CN
        names = list(cn_params)
        order_params = list(cn_params.values())
        types = [str(order_param[0]) for order_param in order_params]
        params = [
            (
                order_param[1]
                if len(order_param) > 1 and isinstance(order_param[1], dict)
                else None
            )
            for order_param in order_params
        ]

        # Create LocalStructOrderParams instance
        local_ops = local_env.LocalStructOrderParams(types, parameters=params)

        # Get neighboring sites using CrystalNN
        nn_info = local_env.CrystalNN().get_nn_info(
            structure=analysis_structure, n=site_idx
        )

        # Create local structure: central site + neighbors
        sites = [analysis_structure[site_idx]] + [info["site"] for info in nn_info]
        local_structure = Structure.from_sites(sites)

        # Calculate order parameters
        local_order_params = local_ops.get_order_parameters(
            structure=local_structure,
            n=0,
            indices_neighs=list(range(1, len(sites))),
        )

        # Find the geometry with the highest order parameter
        best_match_idx, best_value = max(
            (
                (idx, val)
                for idx, val in enumerate(local_order_params)
                if val is not None
            ),
            key=lambda item: item[1],
            default=(0, 0),
        )
        if best_value > 0.5:  # Only use if reasonably good match
            return f"{names[best_match_idx]}:{cn_val}"

    except (RuntimeError, ValueError, IndexError):
        pass  # order params unavailable for this site, use generic CN label
    return f"CN:{cn_val}"  # no confident geometry match, use generic CN label


def collect_coord_envs_chemenv(
    structures: Iterable[Any],
    *,
    chem_env_settings: dict[str, Any] | None = None,
    normalize: bool = False,
) -> list[dict[str, Any]]:
    """Collect coordination environment counts using pymatgen's ChemEnv module.

    Args:
        structures (Iterable[Structure]): Structures to analyze.
        chem_env_settings (dict | None): Passed to LocalGeometryFinder
            .setup_parameters. Defaults to None.
        normalize (bool): Normalize counts per structure. Defaults to False.

    Returns:
        list[dict]: One dict per (coord_num, chem_env_symbol) pair per structure
            with keys coord_num, chem_env_symbol, count.
    """
    import pymatgen.analysis.chemenv.coordination_environments.coordination_geometries as coord_geoms  # noqa: E501
    import pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder as coord_finder  # noqa: E501
    import pymatgen.analysis.chemenv.coordination_environments.structure_environments as struct_envs  # noqa: E501
    from pymatgen.analysis.chemenv.coordination_environments import chemenv_strategies

    # setup failures (e.g. invalid chem_env_settings) propagate to the caller
    lgf = coord_finder.LocalGeometryFinder()
    lgf.setup_parameters(**(chem_env_settings or {}))
    strategy = chemenv_strategies.SimplestChemenvStrategy()
    symbol_cn_mapping = coord_geoms.AllCoordinationGeometries().get_symbol_cn_mapping()

    chem_env_data: list[dict[str, Any]] = []
    for struct_idx, structure in enumerate(structures):
        try:
            lgf.setup_structure(structure=structure)
            structure_environments = lgf.compute_structure_environments()
            lse = struct_envs.LightStructureEnvironments.from_structure_environments(
                strategy, structure_environments
            )
        except (RuntimeError, KeyError) as exc:
            warnings.warn(
                f"ChemEnv analysis failed for structure {struct_idx}, skipping it: "
                f"{exc!r}",
                UserWarning,
                stacklevel=2,
            )
            continue

        coord_envs_dict: dict[tuple[int, str], float] = {}
        for env_list in lse.coordination_environments or []:
            for coord_env in env_list or []:
                ce_symbol = coord_env["ce_symbol"]
                cn_val = get_cn_from_symbol(ce_symbol, symbol_cn_mapping)
                key = (cn_val, ce_symbol)
                coord_envs_dict[key] = coord_envs_dict.get(key, 0) + 1

        total = sum(coord_envs_dict.values())
        chem_env_data += [
            dict(
                coord_num=cn_val,
                chem_env_symbol=ce_symbol,
                count=env_count / total if normalize else env_count,
            )
            for (cn_val, ce_symbol), env_count in coord_envs_dict.items()
        ]

    return chem_env_data


def collect_coord_envs_crystal_nn(
    structures: Iterable[Any],
    *,
    normalize: bool = False,
) -> list[dict[str, Any]]:
    """Collect coordination environment counts using CrystalNN + order params
    (faster than ChemEnv but less detailed geometric classification).

    Args:
        structures (Iterable[Structure]): Structures to analyze.
        normalize (bool): Normalize counts per structure. Defaults to False.

    Returns:
        list[dict]: One dict per site per structure with keys coord_num,
            chem_env_symbol, count.
    """
    from pymatgen.analysis.local_env import CrystalNN

    chem_env_data: list[dict[str, Any]] = []
    crystal_nn = CrystalNN()

    for struct_idx, structure in enumerate(structures):
        site_count = 1.0 / len(structure) if normalize else 1.0
        # collect per structure so a failure midway doesn't leave partial rows
        struct_rows: list[dict[str, Any]] = []
        try:
            for site_idx in range(len(structure)):
                cn_val = len(crystal_nn.get_nn_info(structure, site_idx))
                # Get best matching coordination environment using order parameters
                ce_symbol = classify_local_env_with_order_params(
                    structure, site_idx, cn_val
                )
                struct_rows.append(
                    dict(coord_num=cn_val, chem_env_symbol=ce_symbol, count=site_count)
                )
        except (RuntimeError, ValueError) as exc:
            warnings.warn(
                f"CrystalNN analysis failed for structure {struct_idx}, skipping it: "
                f"{exc!r}",
                UserWarning,
                stacklevel=2,
            )
            continue
        chem_env_data += struct_rows

    return chem_env_data


def collect_chem_env_data(
    structures: AnyStructure | Sequence[AnyStructure],
    *,
    chem_env_settings: ChemEnvSettings,
    normalize: bool,
    show_counts: ShowCounts,
) -> list[dict[str, Any]]:
    """Validate shared chem_env plot options and collect CN/CE counts.

    Shared setup of chem_env_sunburst and chem_env_treemap.

    Args:
        structures (AnyStructure | Sequence[AnyStructure]): Structures to analyze.
        chem_env_settings (dict[str, Any] | "chemenv" | "crystal_nn"): "crystal_nn"
            uses CrystalNN + order params, "chemenv" uses ChemEnv with default
            settings and a dict uses ChemEnv with custom LocalGeometryFinder settings.
        normalize (bool): Whether to normalize counts per structure.
        show_counts ("value" | "percent" | "value+percent" | False): Text display
            mode, only validated here.

    Returns:
        list[dict[str, Any]]: Rows with keys coord_num, chem_env_symbol, count.

    Raises:
        ValueError: For invalid show_counts or chem_env_settings.
    """
    if show_counts not in get_args(ShowCounts):
        raise ValueError(
            f"Invalid {show_counts=}, must be one of {get_args(ShowCounts)}"
        )
    valid_methods = ("chemenv", "crystal_nn")
    if not isinstance(chem_env_settings, dict) and (
        chem_env_settings not in valid_methods
    ):
        raise ValueError(
            f"Invalid {chem_env_settings=}, must be one of {valid_methods} or a dict "
            "of ChemEnv settings"
        )

    structs = normalize_structures(structures).values()
    if chem_env_settings == "crystal_nn":
        return collect_coord_envs_crystal_nn(structs, normalize=normalize)
    settings = {} if chem_env_settings == "chemenv" else chem_env_settings
    return collect_coord_envs_chemenv(
        structs, chem_env_settings=settings, normalize=normalize
    )


def empty_chem_env_figure() -> go.Figure:
    """Return the placeholder figure shown when there is no CN/CE data."""
    fig = go.Figure()
    fig.layout.title = dict(text="No CN/CE data to display", x=0.5, font_size=16)
    return fig


def aggregate_chem_env_data(chem_env_data: list[dict[str, Any]]) -> pd.DataFrame:
    """Sum counts per (coord_num, chem_env_symbol) pair.

    Args:
        chem_env_data (list[dict[str, Any]]): Rows with keys coord_num,
            chem_env_symbol, count.

    Returns:
        pd.DataFrame: with columns coord_num, chem_env_symbol, count.
    """
    df_chem_env = pd.DataFrame(chem_env_data)
    return df_chem_env.groupby(["coord_num", "chem_env_symbol"], as_index=False)[
        "count"
    ].sum()


def wrap_chem_env_label(text: str) -> str:
    """Wrap long CE labels with <br> so plots can use larger fonts in small cells."""
    return "<br>".join(
        textwrap.wrap(text, width=15, break_long_words=True, break_on_hyphens=True)
    )
