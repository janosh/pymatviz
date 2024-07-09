from __future__ import annotations

import pickle
import sys

from pymatviz.enums import Key, Model, StrEnum


def test_str_enum() -> None:
    assert issubclass(StrEnum, str)
    if sys.version_info >= (3, 11):
        from enum import StrEnum as StdStrEnum

        assert StrEnum is StdStrEnum
    else:
        assert issubclass(StrEnum, str)
        assert StrEnum.__name__ == "StrEnum"


def test_model_enum() -> None:
    assert Model.mace_mp == "mace-mp-0-medium"
    assert Model.mace_mp.label == "MACE-MP"
    assert Model.mace_mp.description == "green"


def test_key_enum() -> None:
    # access any attributes to trigger @unique decorator check
    assert Key.energy_per_atom == "energy_per_atom"
    assert Key.volume == "volume"


def test_pickle_enum() -> None:
    key = Key.energy_per_atom
    pickled_key = pickle.dumps(key)
    unpickled_key = pickle.loads(pickled_key)

    assert type(unpickled_key) == str
    assert unpickled_key == "energy_per_atom"
    assert unpickled_key == Key.energy_per_atom
    assert type(key) == Key
