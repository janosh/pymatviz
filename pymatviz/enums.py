"""Enums for element counting, color schemes and downloadable data files."""

from __future__ import annotations

import os
from enum import EnumType, StrEnum, _EnumDict, unique
from typing import TYPE_CHECKING
from urllib.request import urlopen


if TYPE_CHECKING:
    from typing import Any, Self


class LabelEnum(StrEnum):
    """StrEnum with label and optional description attributes, given as a tuple
    starting with the member's value. Members pickle as plain strings.
    """

    def __new__(cls, val: str, label: str, desc: str = "") -> Self:
        """Create a member from a value, label and optional description."""
        member = str.__new__(cls, val)
        member._value_ = val
        member.__dict__ |= dict(label=label, desc=desc)
        return member

    def __reduce_ex__(self, proto: object) -> tuple[type, tuple[str]]:
        """Pickle as a plain string, so unpickling survives renamed or moved enums."""
        return str, (self.value,)

    def __repr__(self) -> str:
        """Return label if available, else type name and value."""
        return self.label or f"{type(self).__name__}.{self.name}"

    @property
    def label(self) -> str:
        """Make label read-only."""
        return self.__dict__["label"]

    @property
    def description(self) -> str:
        """Make description read-only."""
        return self.__dict__["desc"]


@unique
class ElemCountMode(LabelEnum):
    """Mode of counting elements in a chemical formula."""

    composition = (
        "composition",
        "Composition",
        "Count elements by their amount in the composition",
    )
    fractional_composition = (
        "fractional_composition",
        "Fractional Composition",
        "Count elements by their fraction of the composition",
    )
    reduced_composition = (
        "reduced_composition",
        "Reduced Composition",
        "Count elements by their amount in the reduced composition",
    )
    occurrence = (
        "occurrence",
        "Occurrence",
        "Count elements by occurrence in composition, ignoring their amount",
    )


class MetaFiles(EnumType):
    """Metaclass of Files enum that adds base_dir class kwarg."""

    _base_dir: str
    _auto_download: bool

    def __new__(
        cls,
        name: str,
        bases: tuple[type, ...],
        namespace: _EnumDict,
        *,
        base_dir: str = "",
        auto_download: bool = False,
        **kwargs: Any,
    ) -> MetaFiles:
        """Create new Files enum with given base directory."""
        obj = super().__new__(cls, name, bases, namespace, **kwargs)
        obj._base_dir = base_dir
        obj._auto_download = auto_download
        return obj


class Files(StrEnum, metaclass=MetaFiles):
    """Enum of data files (paths relative to base_dir) with the URLs to download
    them from.
    """

    def __new__(cls, file_path: str, url: str = "") -> Self:
        """Create a member from a file path and an optional download URL."""
        obj = str.__new__(cls)
        obj._value_ = file_path
        obj.__dict__["url"] = url
        return obj

    def __repr__(self) -> str:
        """String representation of the file."""
        return f"{type(self).__name__}.{self.name}"

    def __str__(self) -> str:
        """String representation of the file."""
        return self.name

    @property
    def url(self) -> str:
        """URL associated with the file."""
        return self.__dict__["url"]

    @property
    def file_path(self) -> str:
        """File path associated with the file."""
        path = f"{type(self)._base_dir}/{self.value}"
        if type(self)._auto_download and not os.path.isfile(path) and self.url:
            print(f"Downloading {self.url} to {path}")  # noqa: T201
            # read fully before opening path: a failed read must not leave a partial
            # file that later calls would take as downloaded
            with urlopen(self.url) as response:  # noqa: S310
                content = response.read()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, mode="wb") as file:
                file.write(content)
        return path
