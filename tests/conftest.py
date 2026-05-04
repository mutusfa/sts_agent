"""Shared pytest fixtures for sts_agent tests."""

import pytest

from sts_env.combat.encounters import cultist, jaw_worm
from sts_env.combat.player_state import PlayerState


@pytest.fixture()
def ironclad_vs_cultist():
    return cultist(0, PlayerState.ironclad_starter())


@pytest.fixture()
def ironclad_vs_jaw_worm():
    return jaw_worm(0, PlayerState.ironclad_starter())
