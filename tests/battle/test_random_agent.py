"""Tests for RandomAgent — written before implementation (TDD)."""

from __future__ import annotations

import pytest

from sts_env.combat import Action, Combat

from sts_agent.battle import RandomAgent, run_agent
from sts_agent.battle.base import BattleAgent


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------

def test_random_agent_satisfies_battle_agent_protocol():
    """RandomAgent must structurally satisfy BattleAgent without inheriting it."""
    agent = RandomAgent(seed=0)
    # Runtime isinstance check via runtime_checkable would require it;
    # instead just verify the method signature is callable with expected args.
    assert callable(getattr(agent, "act", None))


# ---------------------------------------------------------------------------
# Action membership — every returned action must be in valid_actions
# ---------------------------------------------------------------------------

def test_random_agent_always_returns_valid_action(ironclad_vs_cultist: Combat):
    """Over many calls, RandomAgent never picks an action outside valid_actions."""
    agent = RandomAgent(seed=42)
    combat = ironclad_vs_cultist
    obs = combat.reset()
    for _ in range(30):
        if obs.done:
            break
        valid = combat.valid_actions()
        chosen = agent.act(obs, valid)
        assert chosen in valid, f"Chose {chosen} which is not in {valid}"
        obs, _, _ = combat.step(chosen)


# ---------------------------------------------------------------------------
# Reproducibility — same seed → identical action sequence
# ---------------------------------------------------------------------------

def test_random_agent_same_seed_reproducible():
    """Two agents with the same seed must produce the same action sequence."""
    combat_a = Combat.ironclad_starter(enemy="Cultist", seed=7)
    combat_b = Combat.ironclad_starter(enemy="Cultist", seed=7)

    agent_a = RandomAgent(seed=99)
    agent_b = RandomAgent(seed=99)

    obs_a = combat_a.reset()
    obs_b = combat_b.reset()

    actions_a: list[Action] = []
    actions_b: list[Action] = []

    for _ in range(20):
        if obs_a.done or obs_b.done:
            break
        valid_a = combat_a.valid_actions()
        valid_b = combat_b.valid_actions()
        act_a = agent_a.act(obs_a, valid_a)
        act_b = agent_b.act(obs_b, valid_b)
        actions_a.append(act_a)
        actions_b.append(act_b)
        obs_a, _, _ = combat_a.step(act_a)
        obs_b, _, _ = combat_b.step(act_b)

    assert actions_a == actions_b


# ---------------------------------------------------------------------------
# Different seeds → different action sequences (probabilistic)
# ---------------------------------------------------------------------------

def test_random_agent_different_seeds_differ():
    """Two agents with different seeds must diverge at some point."""
    combat_a = Combat.ironclad_starter(enemy="Cultist", seed=7)
    combat_b = Combat.ironclad_starter(enemy="Cultist", seed=7)

    agent_a = RandomAgent(seed=1)
    agent_b = RandomAgent(seed=2)

    obs_a = combat_a.reset()
    obs_b = combat_b.reset()

    actions_a: list[Action] = []
    actions_b: list[Action] = []

    for _ in range(20):
        if obs_a.done or obs_b.done:
            break
        valid_a = combat_a.valid_actions()
        valid_b = combat_b.valid_actions()
        act_a = agent_a.act(obs_a, valid_a)
        act_b = agent_b.act(obs_b, valid_b)
        actions_a.append(act_a)
        actions_b.append(act_b)
        obs_a, _, _ = combat_a.step(act_a)
        obs_b, _, _ = combat_b.step(act_b)

    assert actions_a != actions_b, (
        "Seeds 1 and 2 produced the same action sequence — very unlikely, check RNG"
    )


# ---------------------------------------------------------------------------
# Termination — run_agent completes and obs.done is True
# ---------------------------------------------------------------------------

def test_run_agent_terminates(ironclad_vs_cultist: Combat):
    """run_agent must always terminate (not loop forever) and return damage_taken."""
    agent = RandomAgent(seed=0)
    damage = run_agent(agent, ironclad_vs_cultist)
    assert isinstance(damage, int)
    assert damage >= 0


def test_run_agent_terminates_vs_jaw_worm(ironclad_vs_jaw_worm: Combat):
    agent = RandomAgent(seed=0)
    damage = run_agent(agent, ironclad_vs_jaw_worm)
    assert damage >= 0


# ---------------------------------------------------------------------------
# Stateless re-use — same agent can run multiple combats sequentially
# ---------------------------------------------------------------------------

def test_random_agent_reusable_across_combats():
    """RandomAgent should work correctly when used for multiple separate runs."""
    agent = RandomAgent(seed=5)
    for seed in range(3):
        combat = Combat.ironclad_starter(enemy="Cultist", seed=seed)
        damage = run_agent(agent, combat)
        assert damage >= 0
