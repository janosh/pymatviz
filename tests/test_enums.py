from __future__ import annotations

import pickle
import sys

from pymatviz.enums import Key, Model, StrEnum


def test_str_enum() -> None:
    # ensure all pymatviz Enums classes are subclasses of StrEnum
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
