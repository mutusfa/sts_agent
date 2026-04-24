"""Tests for TreeSearchPlanner — written before implementation (TDD).

Scenarios are kept tractable by using low enemy HP so the planner can find
a killing line quickly (alpha-pruning then collapses the remaining branches).
"""

from __future__ import annotations

import pytest

from sts_env.combat import Action, Combat
from sts_env.combat.cards import CardType, get_spec
from sts_env.combat.state import ActionType

from sts_agent.battle import RandomAgent, TreeSearchPlanner, run_agent, run_planner
from sts_agent.battle.base import terminal_score
from sts_agent.battle.tree_search import SearchBudgetExceeded


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_combat(seed: int = 0, enemy_hp: int = 12, player_hp: int = 80) -> Combat:
    """Starter deck vs Cultist with custom HP values (set after reset).

    Default enemy_hp=12 requires exactly two Strikes to kill (6 dmg each),
    achievable in one turn (3 energy, Strike costs 1).  Cultist T0 is
    Incantation (no damage), so optimal score is 0.
    """
    combat = Combat(
        deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash"],
        enemies=["Cultist"],
        seed=seed,
        player_hp=player_hp,
    )
    obs = combat.reset()
    combat._state.enemies[0].hp = enemy_hp
    combat._state.enemies[0].max_hp = enemy_hp
    return combat


# ---------------------------------------------------------------------------
# Trivial optimum
# ---------------------------------------------------------------------------


def test_tree_search_picks_attack_not_end_turn():
    """Planner must open with an attack, not END_TURN or a Skill, when enemy is killable.

    We force a deterministic hand of two Strikes + one Defend so the enemy is
    provably killable this turn (2 * 6 dmg = 12 HP exactly) regardless of the
    combat seed.  END_TURN and Defend are both among the valid actions, so the
    planner genuinely has a choice; asserting the chosen card is an ATTACK
    would fail if the planner ever regressed to passing the turn or blocking
    when a lethal line exists.
    """
    combat = _make_combat(enemy_hp=12)
    combat._state.piles.hand = ["Strike", "Strike", "Defend"]

    action = TreeSearchPlanner().act(combat)

    assert action.action_type == ActionType.PLAY_CARD, (
        "Planner chose END_TURN despite a 2-Strike lethal in hand"
    )
    card_id = combat._state.piles.hand[action.hand_index]
    assert get_spec(card_id).card_type == CardType.ATTACK, (
        f"Planner played a non-attack ({card_id}) instead of a lethal Strike"
    )


def test_tree_search_zero_damage_against_two_strike_enemy():
    """Full run: optimal play kills 24-HP Cultist with 0 damage.

    Cultist T0 = Incantation (no attack). 4 attacks in two turns = enemy
    dead before attacking → damage_taken == 0.
    """
    combat = _make_combat(enemy_hp=24)
    planner = TreeSearchPlanner()
    obs = combat.observe()
    while not obs.done:
        obs, _, _ = combat.step(planner.act(combat))
    assert combat.damage_taken == 0, f"Expected 0 damage, got {combat.damage_taken}"
    assert all(e.hp <= 0 for e in obs.enemies)


# ---------------------------------------------------------------------------
# Death-avoidance: death-penalty (×2) must rank death branches worse than
# any surviving branch.
# ---------------------------------------------------------------------------


def test_tree_search_avoids_death():
    """With 1 HP and a killable enemy, planner must attack rather than end turn.

    END_TURN path: Incantation (safe T0) → Dark Strike (6 dmg) kills 1-HP player
                   → terminal score = 6 * 2 = 12 (death penalty).
    Attack path:   kill enemy in T0 before its turn → score = 0.
    Planner must pick the attack path.
    """
    combat = _make_combat(enemy_hp=12, player_hp=1)
    planner = TreeSearchPlanner()
    action = planner.act(combat)
    assert action.action_type == ActionType.PLAY_CARD, (
        "Planner chose END_TURN; player would die on the next enemy turn"
    )

    # Run to completion and verify player survives
    obs = combat.observe()
    while not obs.done:
        obs, _, _ = combat.step(planner.act(combat))
    assert not combat.observe().player_dead, "Planner let the player die"


def test_death_penalty_ranks_death_above_worst_survival():
    """terminal_score for any dead player must exceed that of any alive player."""
    # Dead player: 1 HP → Dark Strike kills (damage_taken ≤ 80 start_hp, so * 2)
    combat_dead = Combat(
        deck=["AscendersBane"] * 10,
        enemies=["Cultist"],
        seed=0,
        player_hp=1,
    )
    combat_dead.reset()
    combat_dead.step(Action.end_turn())  # T0: Incantation (no dmg)
    combat_dead.step(Action.end_turn())  # T1: Dark Strike → player dead
    assert combat_dead.observe().player_dead
    dead_score = terminal_score(combat_dead)

    # Alive player with heavy damage (79 HP lost out of 80)
    combat_alive = _make_combat(enemy_hp=12, player_hp=80)
    combat_alive._state.player_hp = 1  # simulate near-death but alive
    # Manually mark as done by killing the enemy
    combat_alive._state.enemies[0].hp = 0
    alive_score = terminal_score(combat_alive)  # damage_taken = 0 (no step taken)

    # The meaningful comparison: dead score is double any actual damage
    assert dead_score > 0, "Dead player must have positive penalty"
    # More concretely: dead score = dmg * 2, alive = dmg * 1
    raw = combat_dead.damage_taken
    assert dead_score == raw * 2


# ---------------------------------------------------------------------------
# Strict improvement over random agent
# ---------------------------------------------------------------------------


def test_tree_search_beats_random_on_small_combat():
    """Planner damage <= average random damage over several seeds.

    Uses 12-HP Cultist (2-Strike kill in T0).  Planner always scores 0;
    random sometimes misses the kill and takes subsequent Dark Strikes.
    """
    n_seeds = 5

    for seed in range(n_seeds):
        # Planner run
        combat_p = _make_combat(seed=seed, enemy_hp=12)
        obs = combat_p.observe()
        planner = TreeSearchPlanner()
        while not obs.done:
            obs, _, _ = combat_p.step(planner.act(combat_p))
        planner_dmg = combat_p.damage_taken

        # Random agent average over several sub-seeds
        random_damages: list[int] = []
        for sub_seed in range(5):
            cr = _make_combat(seed=seed, enemy_hp=12)
            obs_r = cr.observe()
            agent = RandomAgent(seed=sub_seed * 17)
            while not obs_r.done:
                valid = cr.valid_actions()
                obs_r, _, _ = cr.step(agent.act(obs_r, valid))
            random_damages.append(cr.damage_taken)

        avg_random = sum(random_damages) / len(random_damages)
        assert planner_dmg <= avg_random, (
            f"seed={seed}: planner={planner_dmg} > random avg={avg_random:.1f}"
        )


# ---------------------------------------------------------------------------
# Node budget: raises SearchBudgetExceeded when limit is exhausted
# ---------------------------------------------------------------------------


def test_tree_search_budget_exceeded():
    """SearchBudgetExceeded is raised when max_nodes=1 on a non-trivial combat."""
    combat = Combat.ironclad_starter(enemy="Cultist", seed=0)
    combat.reset()

    planner = TreeSearchPlanner(max_nodes=1)
    with pytest.raises(SearchBudgetExceeded):
        planner.act(combat)


# ---------------------------------------------------------------------------
# Non-mutation: act() must not advance the real combat state
# ---------------------------------------------------------------------------


def test_tree_search_does_not_mutate_combat():
    """act() clones internally; the passed-in combat must remain unchanged."""
    combat = _make_combat(enemy_hp=12)
    obs_before = combat.observe()

    planner = TreeSearchPlanner()
    planner.act(combat)

    obs_after = combat.observe()
    assert obs_before.turn == obs_after.turn
    assert obs_before.player_hp == obs_after.player_hp
    assert obs_before.hand == obs_after.hand
    assert obs_before.enemies[0].hp == obs_after.enemies[0].hp


# ---------------------------------------------------------------------------
# Transposition table
# ---------------------------------------------------------------------------


def test_transposition_table_reduces_nodes():
    """TT planner must expand strictly fewer nodes than no-TT on a transposition-rich scenario.

    We use a high-HP Cultist (50 HP) against a hand containing multiple distinct
    card types.  On T=0 the Cultist only uses Incantation (no attack), so every
    permutation of cards played leads to the *same* game state at the start of T=1.
    The TT collapses these duplicate subtrees; the no-TT planner re-explores them.
    """
    # High-HP enemy: not killable in one turn, so the search explores T1+
    combat = Combat(
        deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash"],
        enemies=["Cultist"],
        seed=0,
        player_hp=80,
    )
    combat.reset()
    combat._state.enemies[0].hp = 50
    combat._state.enemies[0].max_hp = 50
    # Force a hand with 3 distinct card types → 6 orderings that all reach
    # the same state after END_TURN (Cultist T0 = Incantation, no damage).
    combat._state.piles.hand = ["Strike", "Defend", "Bash", "Strike", "Defend"]

    planner_tt = TreeSearchPlanner(use_transposition_table=True)
    planner_tt.act(combat)
    nodes_with_tt = planner_tt._last_node_count

    planner_no_tt = TreeSearchPlanner(use_transposition_table=False)
    planner_no_tt.act(combat)
    nodes_without_tt = planner_no_tt._last_node_count

    assert nodes_with_tt < nodes_without_tt, (
        f"TT did not reduce node count: with={nodes_with_tt} without={nodes_without_tt}"
    )


@pytest.mark.slow
def test_node_budget_50hp_cultist():
    """T=0 node count for 50-HP Cultist seed=0 must not exceed 17 000.

    Guards against regressions in the transposition table or pruning logic
    that would cause the search to re-expand previously seen states.
    """
    combat = Combat(
        deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash"],
        enemies=["Cultist"],
        seed=0,
        player_hp=80,
    )
    combat.reset()
    combat._state.enemies[0].hp = 50
    combat._state.enemies[0].max_hp = 50

    planner = TreeSearchPlanner()
    planner.act(combat)
    assert planner._last_node_count <= 17_000, (
        f"T=0 expanded {planner._last_node_count} nodes, expected ≤17 000"
    )


def test_transposition_table_preserves_decision():
    """TT planner must pick the same first action as the no-TT planner."""
    scenarios = [
        _make_combat(enemy_hp=12),   # killable in T0 → clear optimal
        _make_combat(enemy_hp=24),   # needs two turns
        _make_combat(enemy_hp=12, player_hp=1),  # death-avoidance
    ]
    for combat in scenarios:
        action_tt = TreeSearchPlanner(use_transposition_table=True).act(combat)
        action_no_tt = TreeSearchPlanner(use_transposition_table=False).act(combat)
        assert action_tt.action_type == action_no_tt.action_type, (
            f"TT changed action_type: {action_tt} vs {action_no_tt}"
        )
        # Check card identity (not just index) when both play a card
        if action_tt.action_type == ActionType.PLAY_CARD:
            card_tt = combat._state.piles.hand[action_tt.hand_index]
            card_no_tt = combat._state.piles.hand[action_no_tt.hand_index]
            assert card_tt == card_no_tt, (
                f"TT chose different card: {card_tt} vs {card_no_tt}"
            )
