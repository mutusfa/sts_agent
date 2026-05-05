"""TreeSearchPlanner — exhaustive minimax DFS over cloned combat states.

The search is deterministic: because the Combat's RNG state is captured in
clone(), each branch unrolls the exact same sequence of draws and enemy
intents.  There are no chance nodes — the tree is a pure AND-tree of player
choices.

Pruning (alpha / monotone lower-bound)
---------------------------------------
damage_taken (= start_hp - current_hp) is *mostly* monotonically
non-decreasing through a combat — enemies deal damage each turn, and
you can only block so much.  The Feed card breaks strict monotonicity
(its heal can reduce damage_taken), but the effect is small (+3 HP per
kill) and the pruning still produces correct results in the vast
majority of cases.  The max_hp gain from Feed is exposed separately
via combat.max_hp_gained for strategic-layer valuation.

  If combat.damage_taken >= best_so_far, this branch can score no better than
  best_so_far, so it can be pruned immediately.

In practice this is decisive: as soon as the planner finds a 0-damage
solution (e.g. "kill enemy before end-of-turn attack") it sets best=0 and
prunes every remaining branch at their first node, making the search fast
even on multi-turn combats where the defensive branches would otherwise
explode exponentially.

Deduplication
--------------
valid_actions() emits one entry per hand-slot, but two Strikes in hand
produce identical successor states.  Actions are deduplicated by
(card_id, target_index) at every node, reducing the branching factor
to the number of *distinct* card types in hand + END_TURN.

Transposition table
--------------------
Different orderings of card plays within a turn produce the same resulting
game state (hand/discard/draw piles are order-invariant for the discard,
exhaust, and hand sub-piles).  A per-act() transposition table maps a
normalised state key → exact minimum score, so duplicate subtrees are
resolved immediately on first re-encounter.

Key normalisation:
- draw pile:    ordered tuple (draw order matters — drawn from index 0)
- hand:         sorted tuple (multiset; player picks which card to play)
- discard pile: sorted tuple (multiset; reshuffled before next draw)
- exhaust pile: sorted tuple (multiset)
- RNG state:    included verbatim (ensures correctness for branches that
                diverge because of differing shuffle/intent outcomes)

Only exact scores (returned_score < incoming_cutoff) are cached; pruned
returns (== cutoff, a lower bound) are not stored.
"""

from __future__ import annotations

import logging
import math
from dataclasses import astuple

from sts_env.combat import Action, Combat
from sts_env.combat.state import ActionType
from sts_env.combat.card import Card

from .base import _fmt_action, max_relic_heal, min_effective_score, terminal_score_scalar

log = logging.getLogger(__name__)


class SearchBudgetExceeded(Exception):
    """Raised when the node-expansion budget is exhausted."""


def _powers_key(powers) -> tuple:  # type: ignore[no-untyped-def]
    raw = astuple(powers)
    # bomb_fuses is a list[tuple] — must be converted to tuple for hashability
    return tuple(
        tuple(v) if isinstance(v, list) else v
        for v in raw
    )


def _state_key_base(combat: Combat) -> tuple:
    """Return a normalised, hashable key for the current combat state, **excluding** RNG.

    Pile ordering is normalised where order is semantically irrelevant:
    hand/discard/exhaust are sorted; draw pile retains its order.

    Shared by both the exact tree-search (which appends RNG state) and the
    open-loop MCTS (which omits it to collapse stochastic outcomes).
    """
    s = combat._state  # type: ignore[union-attr]
    enemies_key = tuple(
        (e.name, e.hp, e.max_hp, e.block, _powers_key(e.powers),
         tuple(e.move_history), e.misc, e.pending_split, e.pending_mode_shift, e.is_escaping)
        for e in s.enemies
    )
    return (
        s.player_hp,
        s.player_max_hp,
        s.player_block,
        s.energy,
        s.turn,
        _powers_key(s.player_powers),
        tuple(c.to_key() for c in s.piles.draw),
        tuple(sorted(c.to_key() for c in s.piles.hand)),
        tuple(sorted(c.to_key() for c in s.piles.discard)),
        tuple(sorted(c.to_key() for c in s.piles.exhaust)),
        enemies_key,
        combat.damage_taken,
        tuple(s.potions),
        s.energy_loss_next_turn,
        s.rampage_extra,
        tuple(
            (type(f).__name__, tuple(c.to_key() for c in f.choices))
            if hasattr(f, 'choices') else (type(f).__name__, f.label)
            for f in s.pending_stack
        ),
    )


def _state_key(combat: Combat) -> tuple:
    """Return a normalised, hashable key including RNG state (exact solver).

    RNG state is appended so that branches diverging due to shuffle or intent
    outcomes are treated as distinct nodes.
    """
    s = combat._state  # type: ignore[union-attr]
    return _state_key_base(combat) + (s.rng._rng.getstate(),)


# ---------------------------------------------------------------------------
# Move ordering
# ---------------------------------------------------------------------------

# Per-action priority — lower = tried first.  Spaced by 10 so new entries can
# slot between tiers without renumbering.
#
#  10   (reserved for POWER cards — none implemented yet)
#  20   cards that apply lasting effects this turn (Bash → Vulnerable)
#  25   USE_POTION — defensive/utility potions best used before taking damage
#  30   free (cost-0) cards — always play, they cost nothing
#  40-49 attack-dominant cards, ranked by expected damage/energy
#  50-59 block/defensive skills, ranked by expected block/energy
#  59   DISCARD_POTION — almost never useful; try last before END_TURN
#  60   END_TURN (always last)
_CARD_TIER: dict[str, int] = {
    "Bash": 20,
    "Anger": 30,
    "Cleave": 40,
    "PommelStrike": 42,
    "Strike": 44,
    "IronWave": 46,
    "ShrugItOff": 50,
    "Defend": 52,
}
_DEFAULT_TIER = 44  # unknown card → Strike-equivalent (safe middle)
_USE_POTION_TIER = 25
_DISCARD_POTION_TIER = 59
_END_TURN_TIER = 60


def _order_key(action: Action, combat: Combat) -> tuple:
    """Sort key for move ordering: (tier, effective_cost, target_hp, label)."""
    s = combat._state  # type: ignore[union-attr]
    if action.action_type == ActionType.PLAY_CARD:
        card: Card = s.piles.hand[action.hand_index]
        tier = _CARD_TIER.get(card.card_id, _DEFAULT_TIER)
        # Factor in effective cost so free cards sort first
        from sts_env.combat.cards import get_spec
        spec = get_spec(card.card_id)
        effective_cost = card.cost_override if card.cost_override is not None else spec.cost
        target_hp = 0
        ti = action.target_index
        if 0 <= ti < len(s.enemies):
            target_hp = s.enemies[ti].hp
        return (tier, effective_cost, target_hp, card.card_id)
    if action.action_type == ActionType.USE_POTION:
        potion_id = s.potions[action.potion_index] if action.potion_index < len(s.potions) else ""
        return (_USE_POTION_TIER, 0, 0, potion_id)
    if action.action_type == ActionType.DISCARD_POTION:
        potion_id = s.potions[action.potion_index] if action.potion_index < len(s.potions) else ""
        return (_DISCARD_POTION_TIER, 0, 0, potion_id)
    if action.action_type == ActionType.CHOOSE_CARD:
        return (_USE_POTION_TIER, 0, action.choice_index, "CHOOSE")
    if action.action_type == ActionType.SKIP_CHOICE:
        return (58, 0, 0, "SKIP")
    return (_END_TURN_TIER, 0, 0, "")


def _ordered_actions(combat: Combat) -> list[Action]:
    """Return deduplicated actions sorted by move-ordering heuristic."""
    return sorted(_dedupe_actions(combat), key=lambda a: _order_key(a, combat))


def _dedupe_actions(combat: Combat) -> list[Action]:
    """Return valid actions with duplicates collapsed by a type-tagged key.

    - PLAY_CARD: keyed by (card_id, cost_override, target_index) — hand slot doesn't matter.
    - USE_POTION: keyed by (potion_id, target_index) — slot index doesn't matter.
    - DISCARD_POTION: keyed by potion_id — order-independent.
    - END_TURN: unique singleton.
    """
    s = combat._state  # type: ignore[union-attr]
    seen: set[tuple] = set()
    result: list[Action] = []
    for action in combat.valid_actions():
        if action.action_type == ActionType.PLAY_CARD:
            card: Card = s.piles.hand[action.hand_index]
            key: tuple = ("CARD", card.card_id, card.cost_override, action.target_index)
        elif action.action_type == ActionType.USE_POTION:
            potion_id = s.potions[action.potion_index]
            key = ("USE", potion_id, action.target_index)
        elif action.action_type == ActionType.DISCARD_POTION:
            potion_id = s.potions[action.potion_index]
            key = ("DISCARD", potion_id)
        elif action.action_type == ActionType.CHOOSE_CARD:
            key = ("CHOOSE", action.choice_index)
        elif action.action_type == ActionType.SKIP_CHOICE:
            key = ("SKIP",)
        else:
            key = ("END",)
        if key not in seen:
            seen.add(key)
            result.append(action)
    return result


class TreeSearchPlanner:
    """Exhaustive depth-first search with alpha-pruning that minimises
    terminal_score.

    Parameters
    ----------
    max_nodes:
        Maximum number of node expansions.  ``None`` = unlimited.
        Raises ``SearchBudgetExceeded`` if the budget is exceeded.
    use_transposition_table:
        Enable per-act() memoisation of exact subtree scores.  Defaults to
        True.  Set False to disable (useful for benchmarking or debugging).
    """

    def __init__(
        self,
        max_nodes: int | None = None,
        use_transposition_table: bool = True,
        use_move_ordering: bool = True,
    ) -> None:
        self.max_nodes = max_nodes
        self.use_transposition_table = use_transposition_table
        self.use_move_ordering = use_move_ordering
        self._last_node_count: int = 0

    def act(self, combat: Combat) -> Action:
        """Return the root action whose full subtree achieves the lowest score.

        The original ``combat`` is *never* mutated — all rollouts use clones.
        """
        obs = combat.observe()
        start_hp: float = obs.player_max_hp
        total_initial_enemy_hp: float = sum(
            e.max_hp for e in combat._state.enemies  # type: ignore[union-attr]
        )
        state = combat._state  # type: ignore[union-attr]
        mrl = max_relic_heal(state.relics, int(start_hp))
        # player_hp_initial: HP at combat.reset() = current HP + damage already taken.
        player_hp_initial = obs.player_hp + combat.damage_taken
        min_eff = min_effective_score(state.relics, player_hp_initial, int(start_hp))

        counter = _Counter(self.max_nodes)
        tt: dict[tuple, int | float] | None = (
            {} if self.use_transposition_table else None
        )
        _actions = _ordered_actions if self.use_move_ordering else _dedupe_actions
        actions = _actions(combat)

        best_action = actions[0]
        best_score: int | float = math.inf

        for action in actions:
            clone = combat.clone()
            clone.step(action)
            score = _min_score(
                clone, counter, best_score, tt, self.use_move_ordering,
                start_hp=start_hp, total_initial_enemy_hp=total_initial_enemy_hp,
                max_relic_heal_amount=mrl,
                min_eff_global=min_eff,
            )
            if score < best_score:
                best_score = score
                best_action = action
            if best_score <= min_eff:
                # Optimal; effective score can't go below the global minimum.
                break

        self._last_node_count = counter._count
        obs = combat.observe()
        log.debug(
            "T=%d search done: action=%s score=%s nodes=%d",
            obs.turn,
            _fmt_action(best_action, obs.hand),
            best_score,
            counter._count,
        )
        return best_action


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _Counter:
    """Mutable node-expansion counter; raises SearchBudgetExceeded at limit."""

    def __init__(self, max_nodes: int | None) -> None:
        self._max = max_nodes
        self._count = 0

    def tick(self) -> None:
        self._count += 1
        if self._max is not None and self._count >= self._max:
            raise SearchBudgetExceeded(f"Search budget of {self._max} nodes exceeded.")


def _min_score(
    combat: Combat,
    counter: _Counter,
    cutoff: int | float = math.inf,
    tt: dict[tuple, int | float] | None = None,
    use_move_ordering: bool = True,
    *,
    start_hp: float = math.inf,
    total_initial_enemy_hp: float = 0.0,
    max_relic_heal_amount: int = 0,
    min_eff_global: int | float = 0,
) -> int | float:
    """Recursive DFS with alpha-pruning, optional transposition table, and
    optional move ordering.

    Parameters
    ----------
    combat:
        The current state (a clone; will not be mutated further by the caller).
    counter:
        Shared node-expansion counter.
    cutoff:
        Best score found so far at the parent level.  ``cutoff`` is in
        *effective_damage_taken* units (raw damage minus post-combat relic heals).
        A branch is pruned when its effective-score lower bound cannot beat cutoff.
    tt:
        Transposition table mapping state key → exact minimum score.  ``None``
        disables memoisation.  Only exact results (score < cutoff) are stored;
        pruned returns (score == cutoff, a lower bound) are not cached.
    use_move_ordering:
        When True, use _ordered_actions (card-tier + low-HP-target ordering).
        When False, use plain _dedupe_actions order.
    start_hp:
        Player's max HP at the start of combat — used for tier encoding.
    total_initial_enemy_hp:
        Sum of all enemies' max_hp at the start of combat — used for tier encoding.
    max_relic_heal_amount:
        Upper bound on post-combat relic heals (from ``max_relic_heal()``).
    min_eff_global:
        Global lower bound on effective_damage_taken (from ``min_effective_score()``).
        Equals 0 when no relics can heal.  The state-specific effective lower
        bound is ``max(raw - max_relic_heal_amount, min_eff_global)``; once this
        equals ``min_eff_global`` the entire remaining tree is at best ``min_eff_global``.
    """
    counter.tick()

    # Alpha-prune: this branch's effective lower bound cannot beat cutoff.
    #   effective_lb = max(raw - mrl, min_eff_global)
    # Prune when effective_lb >= cutoff, i.e. the best achievable here can't
    # improve on what we've already found.
    raw = combat.damage_taken
    if max(raw - max_relic_heal_amount, min_eff_global) >= cutoff:
        return cutoff

    if combat.observe().done:
        return terminal_score_scalar(
            combat,
            start_hp=start_hp,
            total_initial_enemy_hp=total_initial_enemy_hp,
        )

    # Transposition table lookup — only valid for exact scores stored earlier.
    key: tuple | None = None
    if tt is not None:
        key = _state_key(combat)
        if key in tt:
            return tt[key]

    _actions = _ordered_actions if use_move_ordering else _dedupe_actions
    best: int | float = cutoff
    for action in _actions(combat):
        clone = combat.clone()
        clone.step(action)
        score = _min_score(
            clone, counter, best, tt, use_move_ordering,
            start_hp=start_hp, total_initial_enemy_hp=total_initial_enemy_hp,
            max_relic_heal_amount=max_relic_heal_amount,
            min_eff_global=min_eff_global,
        )
        if score < best:
            best = score
        if best <= min_eff_global:
            break  # optimal; can't do better than the global minimum

    # Cache only exact results (not pruned lower-bounds).
    if tt is not None and key is not None and best < cutoff:
        tt[key] = best

    return best
