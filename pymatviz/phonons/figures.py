"""Plot element- or site-projected phonon densities of states."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from matscikit.interop import to_phonon_dos

from pymatviz.structures import site_symbols
from pymatviz.widgets.dos import DosWidget


if TYPE_CHECKING:
    import matscikit as mk
    from pymatgen.phonon.dos import CompletePhononDos

    from pymatviz.typing import AnyDos

type DosInput = AnyDos | CompletePhononDos | Mapping[str, AnyDos | CompletePhononDos]


def _unwrap_single[T](labeled: dict[str, T]) -> T | dict[str, T]:
    """Pass an unlabeled single input as itself so MatterViz shows no legend entry."""
    return labeled[""] if list(labeled) == [""] else labeled


def _phonon_dos(label: str, dos: Any) -> mk.PhononDos:
    """PhononDos (matscikit) of a matscikit, phonopy or pymatgen phonon DOS.

    Raises:
        TypeError: For other objects, naming the label.
    """
    try:
        return to_phonon_dos(dos)
    except TypeError as exc:
        raise TypeError(f"{exc} (key {label!r})") from exc


def _dos_dict(frequencies: Any, densities: Any) -> dict[str, Any]:
    """One DOS curve as a pymatgen PhononDos dict (MatterViz's format)."""
    return {
        "@module": "pymatgen.phonon.dos",
        "@class": "PhononDos",
        "frequencies": np.asarray(frequencies, dtype=float).tolist(),
        "densities": np.asarray(densities, dtype=float).tolist(),
    }


def _labeled_doses(
    doses: DosInput,
    project: Literal["element", "site"] | None,
    *,
    show_total: bool,
) -> dict[str, dict[str, Any]]:
    """Map labels to total or (with project) element/site-projected phonon DOS."""
    if project not in (None, "element", "site"):
        raise ValueError(f"Invalid {project=}, must be 'element' or 'site'")
    raw_doses = dict(doses) if isinstance(doses, Mapping) else {"": doses}
    if not raw_doses:
        raise ValueError("Empty DOS dict")

    dos_dict: dict[str, dict[str, Any]] = {}
    for label, raw_dos in raw_doses.items():
        ph_dos = _phonon_dos(label, raw_dos)
        freqs = ph_dos.frequency_points
        total = _dos_dict(freqs, ph_dos.total_dos)
        if project is None:
            dos_dict[label] = total
            continue
        if ph_dos.projected_dos is None or ph_dos.structure is None:
            raise TypeError(
                f"project={project!r} needs a DOS with site projections (pymatgen "
                "CompletePhononDos, matscikit Phonons.projected_dos, phonopy after "
                f"run_projected_dos), got {type(raw_dos).__name__} for key {label!r}"
            )
        symbols = site_symbols(ph_dos.structure)
        projected = (
            ph_dos.element_dos()
            if project == "element"
            else {
                f"{symbol}{site_idx}": densities
                for site_idx, (symbol, densities) in enumerate(
                    zip(symbols, ph_dos.projected_dos, strict=True)
                )
            }
        )
        prefix = f"{label} - " if label else ""
        dos_dict |= {
            f"{prefix}{key}": _dos_dict(freqs, densities)
            for key, densities in projected.items()
        }
        if show_total:
            dos_dict[f"{prefix}Total"] = total
    return dos_dict


def phonon_dos(
    doses: DosInput,
    *,
    stack: bool = False,
    sigma: float = 0,
    normalize: Literal["max", "sum", "integral"] | None = None,
    project: Literal["element", "site"] | None = None,
    show_total: bool = True,
    **kwargs: Any,
) -> DosWidget:
    """Plot one or several (overlaid) phonon densities of states.

    Args:
        doses (DosInput): matscikit PhononDos, phonopy TotalDos or Phonopy (after
            run_total_dos), pymatgen PhononDos or CompletePhononDos, or a dict
            mapping legend labels to them.
        stack (bool): Whether to plot the DOS as a stacked area graph. Defaults to
            False.
        sigma (float): Standard deviation in THz of the Gaussian smearing MatterViz
            applies. Defaults to 0 (no smearing).
        normalize ("max" | "sum" | "integral" | None): Normalization mode MatterViz
            applies to each DOS. Defaults to None.
        project ("element" | "site" | None): Decompose each site-projected DOS
            (pymatgen CompletePhononDos, matscikit PhononDos with projections,
            phonopy after run_projected_dos) into per-element or per-site partial
            DOS. Defaults to None (total DOS only).
        show_total (bool): When projecting, also plot each total DOS (labeled
            ``Total``). Defaults to True.
        **kwargs: Passed to DosWidget, e.g. ``orientation``, ``show_legend`` or
            ``style``.

    Returns:
        DosWidget: MatterViz density of states plot.

    Raises:
        ValueError: If doses is empty, normalize or project is invalid, or stacking a
            projection would also stack its total or several DOS onto each other,
            or stacked DOS have different frequency grids.
        TypeError: If a DOS is unsupported or project is set for a DOS without site
            projections.
    """
    valid_normalize = (None, "max", "sum", "integral")
    if normalize not in valid_normalize:
        raise ValueError(f"Invalid {normalize=}, must be one of {valid_normalize}.")
    n_labels = len(doses) if isinstance(doses, Mapping) else 1
    if stack and project and (show_total or n_labels > 1):
        # MatterViz stacks every curve onto one running sum
        raise ValueError(
            f"stack=True with {project=} needs show_total=False and a single DOS, "
            f"got {show_total=} and {n_labels} DOS"
        )
    dos_dict = _labeled_doses(doses, project, show_total=show_total)
    return DosWidget(
        doses=_unwrap_single(dos_dict),
        stack=stack,
        sigma=sigma,
        normalize=normalize,
        **kwargs,
    )
