"""Tests for TreeSearchPlanner — written before implementation (TDD).

Scenarios are kept tractable by using low enemy HP so the planner can find
a killing line quickly (alpha-pruning then collapses the remaining branches).
"""

from __future__ import annotations

import pytest

from sts_env.combat import Action, Combat
from sts_env.combat.card import Card
from sts_env.combat.cards import CardType, get_spec
from sts_env.combat.state import ActionType

from sts_agent.battle import RandomAgent, TreeSearchPlanner, run_agent, run_planner
from sts_agent.battle.base import terminal_score
from sts_agent.battle.tree_search import SearchBudgetExceeded, _ordered_actions


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
    combat._state.piles.hand = [Card("Strike"), Card("Strike"), Card("Defend")]

    action = TreeSearchPlanner().act(combat)

    assert action.action_type == ActionType.PLAY_CARD, (
        "Planner chose END_TURN despite a 2-Strike lethal in hand"
    )
    card = combat._state.piles.hand[action.hand_index]
    assert get_spec(card.card_id).card_type == CardType.ATTACK, (
        f"Planner played a non-attack ({card.card_id}) instead of a lethal Strike"
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
    """terminal_score for a dead player must set player_dead=True and report damage."""
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
    dead_outcome = terminal_score(combat_dead)

    # Alive player with heavy damage (survive but badly hurt)
    combat_alive = _make_combat(enemy_hp=12, player_hp=80)
    combat_alive._state.player_hp = 1  # near-death but alive
    combat_alive._state.enemies[0].hp = 0  # enemy killed
    alive_outcome = terminal_score(combat_alive)

    assert dead_outcome.player_dead, "Dead combat must report player_dead=True"
    assert not alive_outcome.player_dead, "Alive combat must report player_dead=False"
    assert dead_outcome.damage_taken > 0, "Dead player must have taken positive damage"
    # terminal_score_scalar encodes that any death outranks any survival
    from sts_agent.battle.base import terminal_score_scalar
    start_hp = 80.0
    total_enemy_hp = float(sum(e.max_hp for e in combat_dead._state.enemies))
    dead_scalar = terminal_score_scalar(combat_dead, start_hp=start_hp, total_initial_enemy_hp=total_enemy_hp)
    alive_scalar = terminal_score_scalar(combat_alive, start_hp=start_hp, total_initial_enemy_hp=total_enemy_hp)
    assert dead_scalar > alive_scalar, (
        f"Dead scalar ({dead_scalar}) must exceed alive scalar ({alive_scalar})"
    )


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
    from sts_env.combat.encounters import cultist

    combat = cultist(seed=0)
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
    combat._state.piles.hand = [Card("Strike"), Card("Defend"), Card("Bash"), Card("Strike"), Card("Defend")]

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


# ---------------------------------------------------------------------------
# Move ordering heuristic
# ---------------------------------------------------------------------------


def test_move_ordering_tries_bash_before_strike():
    """_ordered_actions must return Bash before Strike before Defend before END_TURN.

    Bash (tier 20) < Strike (tier 44) < Defend (tier 52) < END_TURN (tier 60).
    """
    combat = _make_combat(enemy_hp=50)
    combat._state.piles.hand = [Card("Strike"), Card("Bash"), Card("Defend")]

    actions = _ordered_actions(combat)
    cards = [
        combat._state.piles.hand[a.hand_index].card_id
        if a.action_type == ActionType.PLAY_CARD
        else "END_TURN"
        for a in actions
    ]
    bash_idx = cards.index("Bash")
    strike_idx = cards.index("Strike")
    defend_idx = cards.index("Defend")
    end_idx = cards.index("END_TURN")

    assert bash_idx < strike_idx, f"Bash should come before Strike: {cards}"
    assert strike_idx < defend_idx, f"Strike should come before Defend: {cards}"
    assert defend_idx < end_idx, f"Defend should come before END_TURN: {cards}"


def test_move_ordering_targets_lowest_hp_first():
    """Within attack actions, the lowest-HP target must come first.

    Uses two Cultists (both MadGremlins via enemies list) with different HP.
    """
    combat = Combat(
        deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash"],
        enemies=["MadGremlin", "MadGremlin"],
        seed=0,
        player_hp=80,
    )
    combat.reset()
    combat._state.enemies[0].hp = 20
    combat._state.enemies[0].max_hp = 20
    combat._state.enemies[1].hp = 5
    combat._state.enemies[1].max_hp = 5
    combat._state.piles.hand = [Card("Strike"), Card("Defend")]

    actions = _ordered_actions(combat)
    strike_actions = [
        a
        for a in actions
        if a.action_type == ActionType.PLAY_CARD
        and combat._state.piles.hand[a.hand_index].card_id == "Strike"
    ]
    assert len(strike_actions) >= 2, "Expected Strike against each enemy"
    first_target_hp = combat._state.enemies[strike_actions[0].target_index].hp
    assert first_target_hp == 5, (
        f"Expected first Strike target to have 5 HP, got {first_target_hp}"
    )


def test_move_ordering_reduces_nodes_on_multi_enemy():
    """Move ordering must reduce nodes on a 2-enemy fight where target choice matters.

    enemy[0] has 14 HP (needs 3 Strikes to kill), enemy[1] has 8 HP (needs 2).
    Unordered tries enemy[0] first (natural valid_actions() index order), so
    the first completed path is suboptimal.  Ordered tries the lower-HP enemy[1]
    first, killing it faster and establishing a tighter cutoff that prunes the
    remaining branches.  TT is disabled to isolate the ordering effect.
    """

    def _run(use_ordering: bool) -> int:
        planner = TreeSearchPlanner(
            use_transposition_table=False,
            use_move_ordering=use_ordering,
        )
        combat = Combat(
            deck=["Strike"] * 5 + ["Defend"] * 4 + ["Bash"],
            enemies=["MadGremlin", "MadGremlin"],
            seed=0,
            player_hp=80,
        )
        combat.reset()
        combat._state.enemies[0].hp = combat._state.enemies[0].max_hp = 14
        combat._state.enemies[1].hp = combat._state.enemies[1].max_hp = 8
        combat._state.piles.hand = [Card("Strike"), Card("Strike"), Card("Strike"), Card("Defend"), Card("Defend")]
        planner.act(combat)
        return planner._last_node_count

    nodes_ordered = _run(True)
    nodes_unordered = _run(False)

    assert nodes_ordered < nodes_unordered, (
        f"Move ordering did not reduce nodes: "
        f"ordered={nodes_ordered} unordered={nodes_unordered}"
    )


@pytest.mark.slow
def test_move_ordering_preserves_decision():
    """Move ordering must not degrade outcome quality.

    Ordering may legitimately choose a different first action when multiple
    moves are tied at the optimal score (e.g. both Strike and Defend lead to
    0 damage when the enemy is killable).  What must be preserved is the
    *terminal damage_taken*, not the specific first action chosen.
    """
    scenarios = [
        _make_combat(enemy_hp=12),
        _make_combat(enemy_hp=24),
        _make_combat(enemy_hp=12, player_hp=1),
    ]
    for combat in scenarios:

        def _run_damage(use_ordering: bool) -> int:
            c = combat.clone()
            c.reset()
            p = TreeSearchPlanner(use_move_ordering=use_ordering)
            obs = c.observe()
            while not obs.done:
                obs, _, _ = c.step(p.act(c))
            return c.damage_taken

        dmg_ord = _run_damage(True)
        dmg_unord = _run_damage(False)
        assert dmg_ord == dmg_unord, (
            f"Move ordering changed terminal damage: ordered={dmg_ord} unordered={dmg_unord}"
        )


def test_transposition_table_preserves_decision():
    """TT planner must pick the same first action as the no-TT planner."""
    scenarios = [
        _make_combat(enemy_hp=12),  # killable in T0 → clear optimal
        _make_combat(enemy_hp=24),  # needs two turns
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
            assert card_tt.card_id == card_no_tt.card_id, (
                f"TT chose different card: {card_tt.card_id} vs {card_no_tt.card_id}"
            )


# ---------------------------------------------------------------------------
# MeatOnTheBone: pruning must not use raw damage_taken against effective cutoff
# ---------------------------------------------------------------------------


class TestMeatOnTheBoneTreeSearch:
    """TreeSearchPlanner must not prune branches whose raw damage exceeds the
    current cutoff when a post-combat relic heal can bring the effective score
    below that cutoff.

    Setup
    -----
    player_hp=44, max_hp=80.  MeatOnTheBone heals 12 when ending combat at
    ≤ 50 % of max_hp (≤ 40 HP).  The combat is advanced past T0 (Cultist
    Incantation, no damage) with a single END_TURN so that T1 starts with
    Cultist intent = Dark Strike (6 dmg).

    The T1 hand is then forced to [Defend]*5 and the draw pile to [Strike]*5
    so that move ordering tries Defend before END_TURN — exactly the ordering
    that triggers the pruning bug:

      1. Defend path: 3 Defends absorb all 6 dmg → raw=0, hp=44>40, no heal.
         effective=0. Sets cutoff=0.
      2. END_TURN path: take 6 → hp=38≤40 → MeatOnTheBone heals 12.
         effective=-6.
         Old prune: raw 6 >= cutoff 0  → pruned (wrong).
         Fixed prune: max(6-12, min_eff=-8) = -6 >= 0 → not pruned (correct).

    T2 draws Strikes and kills the Cultist before its next attack.
    """

    _PLAYER_HP = 44
    _MAX_HP = 80
    _CULTIST_HP = 12  # 2 Strikes kill it in T2

    def _combat(self, relics: frozenset[str]) -> Combat:
        from sts_env.combat import Action as _Action

        combat = Combat(
            deck=["Defend"] * 4 + ["Strike"] * 5 + ["Bash"],
            enemies=["Cultist"],
            seed=0,
            player_hp=self._PLAYER_HP,
            player_max_hp=self._MAX_HP,
            relics=relics,
        )
        combat.reset()
        combat._state.enemies[0].hp = self._CULTIST_HP
        combat._state.enemies[0].max_hp = self._CULTIST_HP
        # Consume T0 (Incantation, no damage) so T1 intent = Dark Strike (6 dmg).
        combat.step(_Action.end_turn())
        # Force T1 hand to all-Defend so Defend (tier 52) is tried before
        # END_TURN (tier 60), reproducing the move-ordering that triggers the bug.
        # Strikes in the draw pile let T2 finish the Cultist before it attacks.
        combat._state.piles.hand = [Card("Defend")] * 5
        combat._state.piles.draw = [Card("Strike")] * 5
        combat._state.piles.discard = []
        return combat

    def test_with_meat_on_bone_picks_end_turn(self):
        """With MeatOnTheBone, END_TURN (effective=-6) must beat Defend (effective=0).

        Without the fix, raw 6 >= cutoff 0 (set by the 3-Defend path) prunes
        the END_TURN branch entirely, causing the planner to return a Defend.
        With the fix the prune is bypassed and END_TURN is correctly chosen.
        """
        combat = self._combat(frozenset({"MeatOnTheBone"}))
        action = TreeSearchPlanner().act(combat)
        assert action.action_type == ActionType.END_TURN, (
            "With MeatOnTheBone, END_TURN (effective=-6) must beat the best "
            "Defend line (effective=0), but planner returned "
            f"action_type={action.action_type}"
        )

    def test_without_meat_on_bone_prefers_defend(self):
        """Without MeatOnTheBone, 3 Defends (raw=0) beats END_TURN (raw=6)."""
        combat = self._combat(frozenset())
        action = TreeSearchPlanner().act(combat)
        assert action.action_type == ActionType.PLAY_CARD, (
            "Without MeatOnTheBone, blocking reduces raw damage from 6 to 0; "
            f"planner should play Defend, got action_type={action.action_type}"
        )
        assert combat._state.piles.hand[action.hand_index].card_id == "Defend", (
            "Without MeatOnTheBone, planner should play Defend to reduce raw damage"
        )
