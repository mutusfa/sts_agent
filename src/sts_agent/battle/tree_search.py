"""TreeSearchPlanner — exhaustive minimax DFS over cloned combat states.

The search is deterministic: because the Combat's RNG state is captured in
clone(), each branch unrolls the exact same sequence of draws and enemy
intents.  There are no chance nodes — the tree is a pure AND-tree of player
choices.

Pruning (alpha / monotone lower-bound)
---------------------------------------
damage_taken is monotonically non-decreasing through a combat — you can only
accumulate damage, never heal.  Therefore:

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

from .base import _fmt_action, terminal_score

log = logging.getLogger(__name__)


class SearchBudgetExceeded(Exception):
    """Raised when the node-expansion budget is exhausted."""


def _powers_key(powers) -> tuple:  # type: ignore[no-untyped-def]
    return astuple(powers)


def _state_key_base(combat: Combat) -> tuple:
    """Return a normalised, hashable key for the current combat state, **excluding** RNG.

    Pile ordering is normalised where order is semantically irrelevant:
    hand/discard/exhaust are sorted; draw pile retains its order.

    Shared by both the exact tree-search (which appends RNG state) and the
    open-loop MCTS (which omits it to collapse stochastic outcomes).
    """
    s = combat._state  # type: ignore[union-attr]
    enemies_key = tuple(
        (e.name, e.hp, e.block, _powers_key(e.powers), tuple(e.move_history))
        for e in s.enemies
    )
    return (
        s.player_hp,
        s.player_block,
        s.energy,
        s.turn,
        _powers_key(s.player_powers),
        tuple(s.piles.draw),
        tuple(sorted(s.piles.hand)),
        tuple(sorted(s.piles.discard)),
        tuple(sorted(s.piles.exhaust)),
        enemies_key,
        combat.damage_taken,
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

# Per-card priority — lower = tried first.  Spaced by 10 so new cards can
# slot between tiers without renumbering.
#
#  10   (reserved for POWER cards — none implemented yet)
#  20   cards that apply lasting effects this turn (Bash → Vulnerable)
#  30   free (cost-0) cards — always play, they cost nothing
#  40-49 attack-dominant cards, ranked by expected damage/energy
#  50-59 block/defensive skills, ranked by expected block/energy
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
_END_TURN_TIER = 60  # always last


def _order_key(action: Action, combat: Combat) -> tuple:
    """Sort key for move ordering: (tier, target_hp, card_id)."""
    if action.action_type != ActionType.PLAY_CARD:
        return (_END_TURN_TIER, 0, "")
    card_id = combat._state.piles.hand[action.hand_index]  # type: ignore[union-attr]
    tier = _CARD_TIER.get(card_id, _DEFAULT_TIER)
    target_hp = 0
    ti = action.target_index
    enemies = combat._state.enemies  # type: ignore[union-attr]
    if 0 <= ti < len(enemies):
        target_hp = enemies[ti].hp
    return (tier, target_hp, card_id)


def _ordered_actions(combat: Combat) -> list[Action]:
    """Return deduplicated actions sorted by move-ordering heuristic."""
    return sorted(_dedupe_actions(combat), key=lambda a: _order_key(a, combat))


def _dedupe_actions(combat: Combat) -> list[Action]:
    """Return valid actions with duplicates collapsed by (card_id, target_index).

    For PLAY_CARD actions: two actions with the same card_id and target_index
    lead to identical game states regardless of which hand slot was used.
    END_TURN is unique.
    """
    seen: set[tuple] = set()
    result: list[Action] = []
    for action in combat.valid_actions():
        if action.action_type == ActionType.PLAY_CARD:
            card_id = combat._state.piles.hand[action.hand_index]  # type: ignore[union-attr]
            key = (card_id, action.target_index)
        else:
            key = (ActionType.END_TURN,)
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
            score = _min_score(clone, counter, best_score, tt, self.use_move_ordering)
            if score < best_score:
                best_score = score
                best_action = action
            if best_score == 0:
                # Score can't go below 0 — no point exploring further.
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
        Best score found so far at the parent level.  Since damage_taken is
        monotonically non-decreasing, any node whose damage_taken >= cutoff
        can only produce terminal scores >= cutoff and is pruned immediately.
    tt:
        Transposition table mapping state key → exact minimum score.  ``None``
        disables memoisation.  Only exact results (score < cutoff) are stored;
        pruned returns (score == cutoff, a lower bound) are not cached.
    use_move_ordering:
        When True, use _ordered_actions (card-tier + low-HP-target ordering).
        When False, use plain _dedupe_actions order.
    """
    counter.tick()

    # Alpha-prune: this branch is already at least as bad as the best we know.
    if combat.damage_taken >= cutoff:
        return cutoff

    if combat.observe().done:
        return terminal_score(combat)

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
        score = _min_score(clone, counter, best, tt, use_move_ordering)
        if score < best:
            best = score
        if best == 0:
            break  # optimal; prune remaining siblings

    # Cache only exact results (not pruned lower-bounds).
    if tt is not None and key is not None and best < cutoff:
        tt[key] = best

    return best
