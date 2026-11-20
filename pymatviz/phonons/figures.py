"""Plot phonon band structures and densities of states with MatterViz widgets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from pymatgen.phonon.dos import CompletePhononDos, PhononDos

from pymatviz.process_data import normalize_phonon_bands
from pymatviz.widgets.band_structure import BandStructureWidget
from pymatviz.widgets.bands_and_dos import BandsAndDosWidget
from pymatviz.widgets.dos import DosWidget


if TYPE_CHECKING:
    from phonopy.phonon.band_structure import BandStructure as PhonopyBandStructure
    from pymatgen.phonon.bandstructure import PhononBandStructureSymmLine as PhononBands

    from pymatviz.typing import AnyDos

type BandsInput = (
    PhononBands
    | PhonopyBandStructure
    | Mapping[str, PhononBands | PhonopyBandStructure]
)
type DosInput = AnyDos | CompletePhononDos | Mapping[str, AnyDos | CompletePhononDos]


def _bands_docs(band_structs: BandsInput) -> dict[str, dict[str, Any]]:
    """Band structure dicts without eigendisplacements, which the band plot doesn't
    use but which can grow the widget state to hundreds of MB.
    """
    docs = {}
    for label, bands in normalize_phonon_bands(band_structs).items():
        docs[label] = bands.as_dict()
        docs[label].pop("eigendisplacements", None)
    return docs


def _unwrap_single[T](labeled: dict[str, T]) -> T | dict[str, T]:
    """Pass an unlabeled single input as itself so MatterViz shows no legend entry."""
    return labeled[""] if list(labeled) == [""] else labeled


def _to_phonon_dos(label: str, dos: Any) -> PhononDos:
    """Convert a phonopy TotalDos or CompletePhononDos to a total PhononDos."""
    cls_name = f"{type(dos).__module__}.{type(dos).__qualname__}"
    if cls_name == "phonopy.phonon.dos.TotalDos":
        return PhononDos(frequencies=dos.frequency_points, densities=dos.dos)
    if isinstance(dos, CompletePhononDos):
        return PhononDos(dos.frequencies, dos.densities)
    if isinstance(dos, PhononDos):
        return dos
    raise TypeError(
        f"Only {PhononDos.__name__}, {CompletePhononDos.__name__}, phonopy TotalDos, "
        f"or dict of these supported, got {type(dos).__name__} for key {label!r}"
    )


def _dos_dict(
    doses: DosInput,
    project: Literal["element", "site"] | None,
    *,
    show_total: bool,
) -> dict[str, PhononDos]:
    """Map labels to total or (with project) element/site-projected phonon DOS."""
    if project not in (None, "element", "site"):
        raise ValueError(f"Invalid {project=}, must be 'element' or 'site'")
    raw_doses = dict(doses) if isinstance(doses, Mapping) else {"": doses}
    if not raw_doses:
        raise ValueError("Empty DOS dict")

    dos_dict: dict[str, PhononDos] = {}
    for label, raw_dos in raw_doses.items():
        if project is None:
            dos_dict[label] = _to_phonon_dos(label, raw_dos)
            continue
        if not isinstance(raw_dos, CompletePhononDos):
            raise TypeError(
                f"project={project!r} requires CompletePhononDos, "
                f"got {type(raw_dos).__name__} for key {label!r}"
            )
        projected = (
            {str(elem): dos for elem, dos in raw_dos.get_element_dos().items()}
            if project == "element"
            else {
                f"{site.specie}{site_idx}": raw_dos.get_site_dos(site)
                for site_idx, site in enumerate(raw_dos.structure)
            }
        )
        prefix = f"{label} - " if label else ""
        dos_dict |= {f"{prefix}{key}": dos for key, dos in projected.items()}
        if show_total:
            dos_dict[f"{prefix}Total"] = _to_phonon_dos(label, raw_dos)
    return dos_dict


def phonon_bands(band_structs: BandsInput, **kwargs: Any) -> BandStructureWidget:
    """Plot one or several (overlaid) pymatgen or phonopy phonon band structures.

    Args:
        band_structs (BandsInput): A PhononBandStructureSymmLine or phonopy
            BandStructure, or a dict mapping legend labels to them. Band structures
            must share their q-point path.
        **kwargs: Passed to BandStructureWidget, e.g. ``reference_frequency``,
            ``show_legend`` or ``style``.

    Returns:
        BandStructureWidget: MatterViz band structure plot.

    Raises:
        TypeError: If band_structs is not a PhononBandStructureSymmLine, phonopy
            BandStructure or dict of these.
        ValueError: If band_structs is an empty mapping.
    """
    bands = _unwrap_single(_bands_docs(band_structs))
    return BandStructureWidget(band_structs=bands, **kwargs)


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
        doses (DosInput): pymatgen PhononDos, CompletePhononDos, phonopy TotalDos, or
            a dict mapping legend labels to them.
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
            projection would also stack its total or several DOS onto each other.
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


def phonon_bands_and_dos(
    band_structs: BandsInput,
    doses: DosInput,
    **kwargs: Any,
) -> BandsAndDosWidget:
    """Plot phonon band structures next to their densities of states, sharing the
    frequency axis.

    Args:
        band_structs (BandsInput): A PhononBandStructureSymmLine or phonopy
            BandStructure, or a dict mapping legend labels to them.
        doses (DosInput): A PhononDos, CompletePhononDos or phonopy TotalDos, or a
            dict with the same labels as band_structs.
        **kwargs: Passed to BandsAndDosWidget, e.g. ``structure`` to add its
            Brillouin zone, or ``stack``, ``sigma`` (THz) and ``normalize`` for the
            DOS panel.

    Returns:
        BandsAndDosWidget: MatterViz combined band structure and DOS plot.

    Raises:
        ValueError: If band_structs and doses keys don't match.
    """
    bands = _bands_docs(band_structs)
    dos_dict = _dos_dict(doses, None, show_total=False)
    if (band_keys := set(bands)) != (dos_keys := set(dos_dict)):
        raise ValueError(f"{band_keys=} and {dos_keys=} must be identical")
    return BandsAndDosWidget(
        band_structs=_unwrap_single(bands), doses=_unwrap_single(dos_dict), **kwargs
    )
