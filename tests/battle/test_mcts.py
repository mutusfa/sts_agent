"""Tests for MCTSPlanner — written before implementation (TDD).

All tests use a minimal Cultist encounter so simulations finish quickly.
"""

from __future__ import annotations

import math

import pytest

from sts_env.combat import Combat
from sts_env.combat.encounters import IRONCLAD_STARTER

from sts_agent.battle import MCTSPlanner, run_planner
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
    required = {"mean", "std", "max", "n", "deaths", "simulations", "nodes"}
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


# ---------------------------------------------------------------------------
# 7. Tree reuse across deterministic turn boundary
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_tree_reused_across_deterministic_turn_boundary():
    """Nodes explored during act() must be reused as priors on subsequent calls.

    Starter deck is 10 cards: T0 draws 5, T1 draws the remaining 5 — the draw
    pile is empty after T1 so no shuffle occurs until T2's first draw.  Cultist
    follows a scripted intent sequence (Incantation T0, then a repeating pattern)
    so enemy transitions add no entropy.  This means the T0→T1 trajectory is
    fully deterministic: the realized successor state-key after stepping an
    action *must* already be present in _node_store if MCTS explored that path.

    Protocol:
    1. Use high enemy HP so T0 never ends the fight.
    2. Walk the real combat one step at a time with planner.act() → combat.step().
    3. After each act() (except the very first), assert that the *current*
       root key was already in _node_store before that call ran — verified by
       capturing the store's key-set before the call.
    4. Require that at least one cross-turn hit is observed (i.e. a cache hit
       after an END_TURN), proving reuse works across turn boundaries, not just
       within a turn.
    """
    combat = _make_combat(seed=0, enemy_hp=80)
    planner = MCTSPlanner(simulations=500, seed=0)

    cross_turn_hit = False
    prev_turn = combat.observe().turn

    for step_idx in range(12):  # enough to comfortably span T0 and T1
        if combat.observe().done:
            break

        # Capture the store contents and turn *before* this act() call.
        keys_before = set(planner._node_store.keys())
        current_key = _mcts_state_key(combat)
        current_turn = combat.observe().turn

        action = planner.act(combat)

        if step_idx > 0:
            # The root state must have been in the store before this call.
            assert current_key in keys_before, (
                f"Step {step_idx} (turn {current_turn}): root key not found in "
                f"_node_store before act() — tree was not reused"
            )
            if current_turn > prev_turn:
                cross_turn_hit = True

        prev_turn = current_turn
        combat.step(action)

    assert cross_turn_hit, (
        "No cross-turn cache hit observed — reuse did not survive an END_TURN"
    )


# ---------------------------------------------------------------------------
# 8. BlockPotion reduces damage on AcidSlimeL
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_block_potion_reduces_acid_slime_l_damage():
    """Agent given a BlockPotion should clear AcidSlimeL with ≤26 damage.

    Without the potion, the baseline is 34 damage.  A BlockPotion grants 12
    block (enough to absorb one of the slime's corrosive-spit hits), so the
    optimal planner should hold off playing the potion until the right moment
    and finish with at most 26 damage taken.
    """
    baseline = Combat(IRONCLAD_STARTER, ["AcidSlimeL", "Empty"], seed=0, player_hp=80)
    with_potion = Combat(
        IRONCLAD_STARTER,
        ["AcidSlimeL", "Empty"],
        seed=0,
        player_hp=80,
        potions=["BlockPotion"],
    )

    base_dmg = run_planner(MCTSPlanner(simulations=2000, seed=0), baseline)
    pot_dmg = run_planner(MCTSPlanner(simulations=2000, seed=0), with_potion)

    assert base_dmg == 34, f"Baseline changed: expected 34, got {base_dmg}"
    assert pot_dmg <= 26, f"BlockPotion didn't help enough: {pot_dmg} > 26"
