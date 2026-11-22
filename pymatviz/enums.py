"""Enums for element counting, color schemes and downloadable data files."""

from __future__ import annotations

import os
from enum import EnumType, StrEnum, _EnumDict, unique
from typing import TYPE_CHECKING
from urllib.request import urlopen


if TYPE_CHECKING:
    from typing import Any, Self


class _PicklableStrEnum(StrEnum):
    """StrEnum that pickles its members as plain strings."""

    def __reduce_ex__(self, proto: object) -> tuple[type, tuple[str]]:
        """Return as a string when pickling. Overrides Enum.__reduce_ex__ which returns
        the tuple self.__class__, (self._value_,). self.__class can cause pickle
        failures if the corresponding Enum was moved or renamed in the meantime.
        """
        return str, (self.value,)


class LabelEnum(_PicklableStrEnum):
    """StrEnum with optional label and description attributes plus dict() methods.

    Simply add label and description as a tuple starting with the key's value.
    """

    def __new__(cls, val: str, label: str, desc: str = "") -> Self:
        """Create a new class from a value, label, and description. Label and
        description are optional.
        """
        member = str.__new__(cls, val)
        member._value_ = val
        member.__dict__ |= dict(label=label, desc=desc)
        return member

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

    # key, label, color
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
    """Enum of data files with associated file directories and URLs."""

    def __new__(
        cls, file_path: str, url: str = "", label: str = "", desc: str = ""
    ) -> Self:
        """Create a new member of the FileUrls enum with a given URL where to load the
        file from and directory where to save it to.
        """
        obj = str.__new__(cls)
        obj._value_ = file_path
        obj.__dict__ |= dict(file_path=file_path, url=url, label=label, desc=desc)
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
        path = f"{type(self)._base_dir}/{self.__dict__['file_path']}"
        if type(self)._auto_download and not os.path.isfile(path) and self.url:
            print(f"Downloading {self.url} to {path}")  # noqa: T201
            with urlopen(self.url) as response, open(path, mode="wb") as file:  # noqa: S310
                file.write(response.read())
        return path

    @property
    def label(self) -> str:
        """Label associated with the file."""
        return self.__dict__["label"]

    @property
    def description(self) -> str:
        """Description associated with the file."""
        return self.__dict__["desc"]

    @property
    def rel_path(self) -> str:
        """Path of the file relative to the repo's ROOT directory."""
        return self.__dict__["file_path"]

    @classmethod
    def from_label(cls, label: str) -> Self:
        """Get enum member from pretty label."""
        file = next((attr for attr in cls if attr.label == label), None)
        if file is None:
            import difflib

            similar_labels = difflib.get_close_matches(label, [k.label for k in cls])
            raise ValueError(
                f"{label=} not found in {cls.__name__}. Did you mean one "
                f"of {similar_labels}?"
            )
        return file
