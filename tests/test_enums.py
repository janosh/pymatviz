from __future__ import annotations

from pymatviz.enums import Key, Model


def test_model_enum() -> None:
    assert Model.mace_mp.value == "mace-mp-0-medium"
    assert Model.mace_mp.label == "MACE-MP"
    assert Model.mace_mp.description == "green"


def test_key_enum() -> None:
    assert Key.energy_per_atom.value == "energy_per_atom"
    assert Key.volume.value == "volume"
