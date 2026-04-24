"""Shared pytest fixtures for sts_agent tests."""

import pytest

from sts_env.combat import Combat


@pytest.fixture()
def ironclad_vs_cultist() -> Combat:
    return Combat.ironclad_starter(enemy="Cultist", seed=0)


@pytest.fixture()
def ironclad_vs_jaw_worm() -> Combat:
    return Combat.ironclad_starter(enemy="JawWorm", seed=0)
