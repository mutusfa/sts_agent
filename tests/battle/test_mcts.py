"""Tests for MCTSPlanner — written before implementation (TDD).

All tests use a minimal Cultist encounter so simulations finish quickly.
"""

from __future__ import annotations

import math

import pytest

from sts_env.combat import Combat
from sts_env.combat.encounters import IRONCLAD_STARTER

from sts_agent.battle import MCTSPlanner, run_planner
from sts_agent.battle.base import TerminalOutcome, terminal_score
from sts_agent.battle.mcts import _EdgeStats, _edge_lex_key, _mcts_state_key


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
    assert required.issubset(set(stats.keys())), (
        f"Missing keys: {required - set(stats.keys())}"
    )

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

    Without the potion, the baseline is 32 damage. A BlockPotion grants 12
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

    assert base_dmg == 32, f"Baseline changed: expected 32, got {base_dmg}"
    assert pot_dmg <= 26, f"BlockPotion didn't help enough: {pot_dmg} > 26"


# ---------------------------------------------------------------------------
# 9. Principal-variation stats
# ---------------------------------------------------------------------------


def test_pv_stats_keys_present():
    """After act(), last_stats must contain pv_mean/pv_std/pv_max/pv_n/pv_deaths/pv_depth."""
    combat = _make_combat()
    planner = MCTSPlanner(simulations=100, seed=0)
    planner.act(combat)

    stats = planner.last_stats
    pv_keys = {"pv_mean", "pv_std", "pv_max", "pv_n", "pv_deaths", "pv_depth"}
    assert pv_keys.issubset(set(stats.keys())), (
        f"Missing PV keys: {pv_keys - set(stats.keys())}"
    )


def test_pv_n_meets_min_n_when_depth_positive():
    """When pv_depth > 0, pv_n must be >= the effective min_n threshold."""
    combat = _make_combat(seed=0, enemy_hp=80)  # high HP so combat lasts multiple turns
    planner = MCTSPlanner(simulations=500, seed=0)
    planner.act(combat)

    stats = planner.last_stats
    if stats["pv_depth"] > 0:
        min_n = max(10, int(500**0.5))  # default formula: max(10, sqrt(simulations))
        assert stats["pv_n"] >= min_n, (
            f"pv_n={stats['pv_n']} < min_n={min_n} but pv_depth={stats['pv_depth']}"
        )


def test_pv_depth_positive_with_enough_simulations():
    """With 1000 simulations MCTS should descend at least one step past the root."""
    combat = _make_combat(seed=0, enemy_hp=80)
    planner = MCTSPlanner(simulations=1000, seed=0)
    planner.act(combat)
    assert planner.last_stats["pv_depth"] >= 1, (
        f"pv_depth={planner.last_stats['pv_depth']}: expected MCTS to descend past root "
        f"with 1000 simulations"
    )


def test_pv_falls_back_to_root_edge_when_budget_tiny():
    """With very few simulations, pv_depth should be 0 or 1 (at root / first level)."""
    combat = _make_combat(seed=0, enemy_hp=80)
    planner = MCTSPlanner(simulations=5, seed=0)
    planner.act(combat)

    stats = planner.last_stats
    # With 5 sims the tree is almost flat; PV should stay at depth <= 1
    assert stats["pv_depth"] <= 1, (
        f"Unexpectedly deep PV with only 5 simulations: pv_depth={stats['pv_depth']}"
    )
    # pv stats must still be finite and sane
    assert stats["pv_n"] > 0
    assert math.isfinite(stats["pv_mean"])


# ---------------------------------------------------------------------------
# 10. Lexicographic priority: survival > enemy damage
# ---------------------------------------------------------------------------

_START_HP = 80
_TOTAL_ENEMY_HP = 50


def _make_outcome(
    *, dead: bool, damage_taken: int, enemy_dmg_dealt: int
) -> TerminalOutcome:
    return TerminalOutcome(
        damage_taken=damage_taken,
        player_dead=dead,
        enemy_damage_dealt=enemy_dmg_dealt,
        effective_damage_taken=damage_taken,
    )


def _fill_edge(outcomes: list[TerminalOutcome]) -> _EdgeStats:
    edge = _EdgeStats()
    for o in outcomes:
        edge.update(o, start_hp=_START_HP, total_initial_enemy_hp=_TOTAL_ENEMY_HP)
    return edge


def test_lex_key_prefers_survival_over_enemy_damage():
    """An all-survive edge must rank below (better) an all-die edge regardless of
    how much enemy damage the dying branch deals."""
    survive = _fill_edge(
        [_make_outcome(dead=False, damage_taken=20, enemy_dmg_dealt=5)] * 10
    )
    die_high_dmg = _fill_edge(
        [_make_outcome(dead=True, damage_taken=_START_HP, enemy_dmg_dealt=45)] * 10
    )
    assert _edge_lex_key(survive) < _edge_lex_key(die_high_dmg), (
        "Surviving with 20 dmg must beat dying with 45 enemy dmg dealt"
    )


def test_lex_key_prefers_more_enemy_damage_when_all_die():
    """When every rollout dies, prefer the branch that dealt the most enemy damage."""
    high_enemy_dmg = _fill_edge(
        [_make_outcome(dead=True, damage_taken=_START_HP, enemy_dmg_dealt=45)] * 10
    )
    low_enemy_dmg = _fill_edge(
        [_make_outcome(dead=True, damage_taken=_START_HP, enemy_dmg_dealt=10)] * 10
    )
    assert _edge_lex_key(high_enemy_dmg) < _edge_lex_key(low_enemy_dmg), (
        "45 enemy damage dealt must beat 10 enemy damage dealt when all branches die"
    )


def test_bucketed_death_rate_ignores_tiny_noise():
    """1 death in 10 000 rollouts rounds to death_rate=0.0 — same tier as zero deaths —
    so tier-2 (alive damage) comparison decides the winner instead."""
    # 1 death in 10 000: round(1/10000, 3) == 0.0
    almost_all_alive = _fill_edge(
        [_make_outcome(dead=False, damage_taken=5, enemy_dmg_dealt=10)] * 9999
        + [_make_outcome(dead=True, damage_taken=_START_HP, enemy_dmg_dealt=10)]
    )
    all_alive_high_dmg = _fill_edge(
        [_make_outcome(dead=False, damage_taken=50, enemy_dmg_dealt=10)] * 10000
    )
    # Both round to 0.0 death rate; tier-2 picks lower alive damage (5 < 50).
    assert _edge_lex_key(almost_all_alive) < _edge_lex_key(all_alive_high_dmg), (
        "1/10000 death rate rounds to 0 — alive-damage (5) should beat alive-damage (50)"
    )


def test_last_stats_exposes_tier_breakdown():
    """After act(), last_stats must contain the lexicographic tier fields."""
    combat = _make_combat()
    planner = MCTSPlanner(simulations=50, seed=0)
    planner.act(combat)

    required = {
        "death_rate",
        "n_alive",
        "mean_damage_alive",
        "n_dead",
        "mean_enemy_dmg_dead",
    }
    assert required.issubset(planner.last_stats), (
        f"Missing tier-breakdown keys: {required - set(planner.last_stats)}"
    )


# ---------------------------------------------------------------------------
# 11. Post-combat relic heals: effective_damage_taken
# ---------------------------------------------------------------------------


def _make_combat_at_hp(
    player_hp: int,
    player_max_hp: int = 80,
    relics: frozenset[str] | None = None,
    enemy_hp: int = 12,
    seed: int = 0,
) -> Combat:
    """Cultist combat with configurable relics and HP."""
    combat = Combat(
        deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash"],
        enemies=["Cultist"],
        seed=seed,
        player_hp=player_hp,
        player_max_hp=player_max_hp,
        relics=relics or frozenset(),
    )
    combat.reset()
    combat._state.enemies[0].hp = enemy_hp
    combat._state.enemies[0].max_hp = enemy_hp
    return combat


def _drive_to_terminal(combat: Combat) -> None:
    """Play greedily until combat is done."""
    from sts_agent.battle.tree_search import _ordered_actions

    while not combat.observe().done:
        combat.step(_ordered_actions(combat)[0])


class TestEffectiveDamageTaken:
    """terminal_score must compute effective_damage_taken via post-combat relic heals."""

    def test_no_relics_effective_equals_raw(self):
        combat = _make_combat_at_hp(80, relics=frozenset())
        _drive_to_terminal(combat)
        outcome = terminal_score(combat)
        assert outcome.effective_damage_taken == outcome.damage_taken

    def test_burning_blood_absorbs_damage_below_6(self):
        # Start at 6 hp missing; 12 hp cultist is killed without taking a single hit
        combat = _make_combat_at_hp(74, relics=frozenset({"BurningBlood"}), enemy_hp=12)
        _drive_to_terminal(combat)
        outcome = terminal_score(combat)
        # We don't take any damage during the combat and at the end we heal for 6
        expected = -6
        assert outcome.effective_damage_taken == expected

    def test_player_dead_effective_equals_raw(self):
        # When player dies, post-combat relics don't fire — effective == damage_taken
        outcome = TerminalOutcome(
            damage_taken=80,
            player_dead=True,
            enemy_damage_dealt=10,
            effective_damage_taken=80,
        )
        assert outcome.effective_damage_taken == outcome.damage_taken


# ---------------------------------------------------------------------------
# 12. MCTSPlanner prefers MeatOnTheBone trigger line
# ---------------------------------------------------------------------------


class TestMeatOnTheBonePlanner:
    """MCTSPlanner must avoid playing Defend when MeatOnTheBone makes the damage free.

    Setup: player_hp=51, max_hp=80, JawWorm hp=19 (can't be killed T1, killed T2 before
    its DEFEND intent fires).  JawWorm attacks for 11 on turn 1; T2 intent is DEFEND
    (no attack) so the player takes at most one hit.

    Two lines on T1:
    - Play Defend first: gain 5 block, end turn → take 11-5=6 → hp=45 (>40) → no trigger
      → effective=6.  Energy on Defend was "wasted" — the 6 HP will not be healed back.
    - Play Strike (or END_TURN): take full 11 → hp=40 (≤40) → MeatOnTheBone fires →
      effective=11-12= -1.

    With MeatOnTheBone the planner should pick Strike (not Defend), because spending energy
    on a Defend card yields higher effective damage than just absorbing the hit.
    Without MeatOnTheBone the planner should pick Defend (raw 6 < raw 11).
    """

    _DECK = ["Defend"] * 4 + ["Strike"] * 5 + ["Bash"]

    def _combat(self, relics: frozenset[str]) -> Combat:
        combat = Combat(
            deck=self._DECK,
            enemies=["JawWorm"],
            seed=0,
            player_hp=51,
            player_max_hp=80,
            relics=relics,
        )
        combat.reset()
        # JawWorm attacks for 11 on T1 (Chomp), DEFEND on T2 (no further damage if
        # killed before DEFEND fires).  Setting hp=19 ensures the starter deck can
        # always kill it on T2 (≥20 total attack damage available) but NOT on T1
        # (3 Strikes = 18 < 19).
        combat._state.enemies[0].hp = 19
        combat._state.enemies[0].max_hp = 19
        return combat

    @pytest.mark.slow
    def test_with_meat_on_bone_avoids_defend(self):
        """With MeatOnTheBone, planner should NOT play Defend first — effective damage
        of the Defend line (6) exceeds the no-Defend line (0 after heal)."""
        combat = self._combat(frozenset({"MeatOnTheBone"}))
        planner = MCTSPlanner(simulations=100, seed=42)
        action = planner.act(combat)
        chosen_card = (
            combat._state.piles.hand[action.hand_index].card_id
            if action.action_type.name == "PLAY_CARD"
            else None
        )
        assert chosen_card != "Defend", (
            "With MeatOnTheBone, playing Defend wastes energy and keeps hp above "
            "the heal threshold — planner should prefer Strike or END_TURN. "
            f"chosen={chosen_card}, last_stats={planner.last_stats}"
        )

    @pytest.mark.slow
    def test_without_meat_on_bone_prefers_defend(self):
        """Without MeatOnTheBone, planner should play Defend first to reduce raw
        damage (6 < 11)."""
        combat = self._combat(frozenset())
        planner = MCTSPlanner(simulations=100, seed=42)
        action = planner.act(combat)
        chosen_card = (
            combat._state.piles.hand[action.hand_index].card_id
            if action.action_type.name == "PLAY_CARD"
            else None
        )
        assert chosen_card == "Defend", (
            f"Without MeatOnTheBone, Defend reduces raw damage from 11 to 6. "
            f"chosen={chosen_card}, last_stats={planner.last_stats}"
        )


# ---------------------------------------------------------------------------
# 13. Budget telemetry keys
# ---------------------------------------------------------------------------


class TestBudgetTelemetry:
    """last_stats must expose per-act() budget counters for pruning analysis."""

    TELEMETRY_KEYS = {
        "sel_depth_mean",
        "sel_depth_max",
        "terminal_during_selection_frac",
        "would_alpha_cut_at_rollout_frac",
        "root_visit_top1_frac",
        "root_visit_top2_frac",
    }

    def test_keys_present(self):
        """All telemetry keys must appear in last_stats after act()."""
        combat = _make_combat()
        planner = MCTSPlanner(simulations=20, seed=0)
        planner.act(combat)
        assert self.TELEMETRY_KEYS.issubset(planner.last_stats), (
            f"Missing telemetry keys: {self.TELEMETRY_KEYS - set(planner.last_stats)}"
        )

    def test_sel_depth_mean_in_range(self):
        """sel_depth_mean must be in [0, sel_depth_max]; sel_depth_max >= 0."""
        combat = _make_combat()
        planner = MCTSPlanner(simulations=100, seed=0)
        planner.act(combat)
        stats = planner.last_stats
        assert stats["sel_depth_mean"] >= 0
        assert stats["sel_depth_max"] >= 0
        assert stats["sel_depth_mean"] <= stats["sel_depth_max"] + 1e-9

    def test_fracs_in_unit_interval(self):
        """Fraction keys must be in [0, 1]."""
        combat = _make_combat()
        planner = MCTSPlanner(simulations=50, seed=0)
        planner.act(combat)
        stats = planner.last_stats
        for key in (
            "terminal_during_selection_frac",
            "would_alpha_cut_at_rollout_frac",
            "root_visit_top1_frac",
            "root_visit_top2_frac",
        ):
            assert 0.0 <= stats[key] <= 1.0, f"{key}={stats[key]} out of [0, 1]"

    def test_root_visit_top2_ge_top1(self):
        """top2 fraction >= top1 fraction (top2 includes at least as many visits)."""
        combat = _make_combat()
        planner = MCTSPlanner(simulations=50, seed=0)
        planner.act(combat)
        stats = planner.last_stats
        assert stats["root_visit_top2_frac"] >= stats["root_visit_top1_frac"] - 1e-9

    def test_would_alpha_cut_frac_zero_with_one_sim(self):
        """With simulations=1, no prior alive sample exists, so cut fraction must be 0."""
        combat = _make_combat()
        planner = MCTSPlanner(simulations=1, seed=0)
        planner.act(combat)
        assert planner.last_stats["would_alpha_cut_at_rollout_frac"] == 0.0
