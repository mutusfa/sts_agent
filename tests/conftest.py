"""Shared pytest fixtures for sts_agent tests."""

import pytest

from sts_env.combat.encounters import cultist, jaw_worm


@pytest.fixture()
def ironclad_vs_cultist():
    return cultist(seed=0)


@pytest.fixture()
def ironclad_vs_jaw_worm():
    return jaw_worm(seed=0)
