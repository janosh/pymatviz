"""Plot element- or site-projected phonon densities of states."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pymatviz.structures import is_instance_of
from pymatviz.widgets._normalize import phonon_dos_dict
from pymatviz.widgets.dos import DosWidget


if TYPE_CHECKING:
    from pymatgen.phonon.dos import CompletePhononDos

    from pymatviz.typing import AnyDos

type DosInput = AnyDos | CompletePhononDos | Mapping[str, AnyDos | CompletePhononDos]


def _unwrap_single[T](labeled: dict[str, T]) -> T | dict[str, T]:
    """Pass an unlabeled single input as itself so MatterViz shows no legend entry."""
    return labeled[""] if list(labeled) == [""] else labeled


def _to_phonon_dos(label: str, dos: Any) -> dict[str, Any]:
    """Total phonon DOS dict of a matscikit PhononDos, phonopy TotalDos or pymatgen
    (Complete)PhononDos.
    """
    if (dos_dict := phonon_dos_dict(dos)) is None:
        raise TypeError(
            "Only matscikit PhononDos, phonopy TotalDos, pymatgen PhononDos or "
            "CompletePhononDos, or a dict of these supported, got "
            f"{type(dos).__name__} for key {label!r}"
        )
    return dos_dict


def _dos_dict(
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
        if project is None:
            dos_dict[label] = _to_phonon_dos(label, raw_dos)
            continue
        if not is_instance_of(raw_dos, "pymatgen.phonon.dos.CompletePhononDos"):
            raise TypeError(
                f"project={project!r} requires CompletePhononDos, "
                f"got {type(raw_dos).__name__} for key {label!r}"
            )
        complete_dos: Any = raw_dos  # a pymatgen CompletePhononDos
        projected = (
            {str(elem): dos for elem, dos in complete_dos.get_element_dos().items()}
            if project == "element"
            else {
                f"{site.specie}{site_idx}": complete_dos.get_site_dos(site)
                for site_idx, site in enumerate(complete_dos.structure)
            }
        )
        prefix = f"{label} - " if label else ""
        dos_dict |= {
            f"{prefix}{key}": _to_phonon_dos(key, dos) for key, dos in projected.items()
        }
        if show_total:
            dos_dict[f"{prefix}Total"] = _to_phonon_dos(label, raw_dos)
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
        doses (DosInput): matscikit PhononDos, phonopy TotalDos, pymatgen PhononDos
            or CompletePhononDos, or a dict mapping legend labels to them.
        stack (bool): Whether to plot the DOS as a stacked area graph. Defaults to
            False.
        sigma (float): Standard deviation in THz of the Gaussian smearing MatterViz
            applies. Defaults to 0 (no smearing).
        normalize ("max" | "sum" | "integral" | None): Normalization mode MatterViz
            applies to each DOS. Defaults to None.
        project ("element" | "site" | None): Decompose each CompletePhononDos into
            per-element or per-site partial DOS. Defaults to None (total DOS only).
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
        TypeError: If a DOS is unsupported or project is set for a DOS that isn't a
            CompletePhononDos.
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
    dos_dict = _dos_dict(doses, project, show_total=show_total)
    return DosWidget(
        doses=_unwrap_single(dos_dict),
        stack=stack,
        sigma=sigma,
        normalize=normalize,
        **kwargs,
    )
