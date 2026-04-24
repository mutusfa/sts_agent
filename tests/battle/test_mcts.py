"""Tests for MCTSPlanner — written before implementation (TDD).

All tests use a minimal Cultist encounter so simulations finish quickly.
"""

from __future__ import annotations

import math

import pytest

from sts_env.combat import Combat

from sts_agent.battle import MCTSPlanner
from sts_agent.battle.mcts import _mcts_state_key


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_combat(seed: int = 0, enemy_hp: int = 12, player_hp: int = 80) -> Combat:
    """Starter deck vs Cultist with adjustable HP values.

    Default enemy_hp=12: two Strikes kill in T0 before the enemy attacks.
    Cultist T0 intent is Incantation (no damage), so optimal score is 0.
    """
    combat = Combat(
        deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash"],
        enemies=["Cultist"],
        seed=seed,
        player_hp=player_hp,
    )
    combat.reset()
    combat._state.enemies[0].hp = enemy_hp
    combat._state.enemies[0].max_hp = enemy_hp
    return combat


# ---------------------------------------------------------------------------
# 1. Basic sanity: act() returns a legal action
# ---------------------------------------------------------------------------


def test_act_returns_valid_action():
    """MCTSPlanner(simulations=1) must return a member of valid_actions()."""
    combat = _make_combat()
    planner = MCTSPlanner(simulations=1)
    action = planner.act(combat)
    assert action in combat.valid_actions()


# ---------------------------------------------------------------------------
# 2. State key ignores RNG
# ---------------------------------------------------------------------------


def test_mcts_state_key_ignores_rng():
    """Two clones with advanced-but-identical game state produce the same key."""
    base = _make_combat()

    clone_a = base.clone()
    clone_b = base.clone()

    # Advance the RNG of clone_b without changing game state visibly.
    # We do this by calling getstate/setstate gymnastics: just make its internal
    # rng differ from clone_a's.
    import random as _random
    rng_obj = clone_b._state.rng._rng
    state = rng_obj.getstate()
    # Bump the version counter in the state tuple (safe no-op on state identity)
    rng_obj.seed(99999)

    # Both should still agree on game-visible components (hp, piles, etc.)
    # because we never called combat.step().  The keys must match.
    key_a = _mcts_state_key(clone_a)
    key_b = _mcts_state_key(clone_b)
    assert key_a == key_b, "MCTS state key must not include RNG state"


# ---------------------------------------------------------------------------
# 3. Convergence: finds the obvious kill
# ---------------------------------------------------------------------------


def test_converges_to_killing_move():
    """With enough simulations, MCTS converges toward the 0-damage kill line.

    Cultist has 1 HP — any Strike/Bash kills instantly in T0 before the enemy
    can attack.  Cultist T0 intent is Incantation (no damage), so the optimal
    expected score is 0 damage.

    We call act() directly on the already-reset combat (not via run_planner,
    which would call reset() again and lose our HP patch).
    """
    combat = _make_combat(seed=0, enemy_hp=1)
    planner = MCTSPlanner(simulations=200, seed=0)
    planner.act(combat)
    # All rollouts through an attack action kill instantly → mean == 0.
    # We verify convergence: mean must be strictly below one Cultist hit (6 dmg).
    assert planner.last_stats["mean"] < 1.0
    assert planner.last_stats["max"] < 1.0


# ---------------------------------------------------------------------------
# 4. last_stats populated with correct keys
# ---------------------------------------------------------------------------


def test_last_stats_populated():
    """After act(), last_stats must contain the required keys with sane values."""
    combat = _make_combat()
    planner = MCTSPlanner(simulations=20, seed=0)
    planner.act(combat)

    stats = planner.last_stats
    required = {"mean", "std", "max", "n", "simulations", "nodes"}
    assert required == set(stats.keys()), f"Missing keys: {required - set(stats.keys())}"

    assert stats["n"] > 0
    assert stats["simulations"] <= 20
    assert stats["nodes"] >= 0
    assert stats["mean"] >= 0
    assert stats["std"] >= 0
    assert stats["max"] >= stats["mean"] - 1e-9  # max >= mean


# ---------------------------------------------------------------------------
# 5a. Simulations budget: exactly N simulations run
# ---------------------------------------------------------------------------


def test_respects_simulation_budget():
    """simulations=5 with no node cap → exactly 5 simulations run."""
    combat = _make_combat()
    planner = MCTSPlanner(simulations=5)
    planner.act(combat)
    assert planner.last_stats["simulations"] == 5


# ---------------------------------------------------------------------------
# 5b. Node budget: node expansions capped at max_nodes
# ---------------------------------------------------------------------------


def test_respects_node_budget():
    """max_nodes=10 with large simulations cap → node expansions do not exceed 10."""
    combat = _make_combat()
    planner = MCTSPlanner(simulations=100_000, max_nodes=10)
    planner.act(combat)
    assert planner.last_stats["nodes"] <= 10


# ---------------------------------------------------------------------------
# 6. Std / tail are finite floats
# ---------------------------------------------------------------------------


def test_std_and_max_are_finite():
    """std and max must be finite (not NaN or inf) after a multi-sim run."""
    combat = _make_combat()
    planner = MCTSPlanner(simulations=50, seed=42)
    planner.act(combat)
    assert math.isfinite(planner.last_stats["std"])
    assert math.isfinite(planner.last_stats["max"])
