from __future__ import annotations

import pickle
import sys

import pytest

from pymatviz.enums import Key, LabelEnum, Model, StrEnum


class DummyEnum(LabelEnum):
    TEST1 = "test1", "Test Label 1", "Test Description 1"
    TEST2 = "test2"


def test_str_enum() -> None:
    # ensure all pymatviz Enums classes are subclasses of StrEnum
    # either the standard library StrEnum if 3.11+ or our own StrEnum
    assert issubclass(StrEnum, str)
    if sys.version_info >= (3, 11):
        from enum import StrEnum as StdLibStrEnum

        assert StrEnum is StdLibStrEnum
    else:
        assert issubclass(StrEnum, str)
        assert StrEnum.__name__ == "StrEnum"


def test_model_enum() -> None:
    assert Model.mace_mp == "mace-mp-0-medium"
    assert Model.mace_mp.label == "MACE-MP"
    assert Model.mace_mp.description == "green"


def test_key_enum() -> None:
    # access any attribute to trigger @unique decorator check
    assert Key.energy_per_atom == "energy_per_atom"
    assert Key.volume == "volume"


def test_pickle_enum() -> None:
    key = Key.energy_per_atom
    pickled_key = pickle.dumps(key)
    unpickled_key = pickle.loads(pickled_key)  # noqa: S301

    # ensure key unpickles to str, not Key (don't use isinstance check as
    # isinstance(StrEnum, str) is True)
    assert type(unpickled_key) is str
    assert unpickled_key == "energy_per_atom"
    assert unpickled_key == Key.energy_per_atom
    assert type(key) is Key

    assert Key.energy.__reduce_ex__(1) == (str, ("energy",))


def test_label_enum_new() -> None:
    assert DummyEnum.TEST1 == "test1"
    assert DummyEnum.TEST1.label == "Test Label 1"
    assert DummyEnum.TEST1.description == "Test Description 1"

    # check label and description are not settable
    assert DummyEnum.TEST1.label == "Test Label 1"
    with pytest.raises(AttributeError):
        DummyEnum.TEST1.label = "New Label"

    assert DummyEnum.TEST1.description == "Test Description 1"
    with pytest.raises(AttributeError):
        DummyEnum.TEST1.description = "New Description"


def test_label_enum_repr() -> None:
    assert repr(DummyEnum.TEST1) == DummyEnum.TEST1.label == "Test Label 1"
    assert repr(DummyEnum.TEST2) == "DummyEnum.TEST2"
    assert DummyEnum.TEST2.label is None


def test_label_enum_key_val_dict() -> None:
    expected = {"TEST1": "test1", "TEST2": "test2"}
    assert DummyEnum.key_val_dict() == expected


def test_label_enum_val_label_dict() -> None:
    expected = {"test1": "Test Label 1", "test2": None}
    assert DummyEnum.val_label_dict() == expected


def test_label_enum_val_desc_dict() -> None:
    expected = {"test1": "Test Description 1", "test2": None}
    assert DummyEnum.val_desc_dict() == expected


def test_label_enum_label_desc_dict() -> None:
    expected = {"Test Label 1": "Test Description 1"}
    assert DummyEnum.label_desc_dict() == expected
