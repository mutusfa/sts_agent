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
"""

from __future__ import annotations

import math

from sts_env.combat import Action, Combat
from sts_env.combat.state import ActionType

from .base import terminal_score


class SearchBudgetExceeded(Exception):
    """Raised when the node-expansion budget is exhausted."""


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
    """

    def __init__(self, max_nodes: int | None = None) -> None:
        self.max_nodes = max_nodes

    def act(self, combat: Combat) -> Action:
        """Return the root action whose full subtree achieves the lowest score.

        The original ``combat`` is *never* mutated — all rollouts use clones.
        """
        counter = _Counter(self.max_nodes)
        actions = _dedupe_actions(combat)

        best_action = actions[0]
        best_score: int | float = math.inf

        for action in actions:
            clone = combat.clone()
            clone.step(action)
            score = _min_score(clone, counter, best_score)
            if score < best_score:
                best_score = score
                best_action = action
            if best_score == 0:
                # Score can't go below 0 — no point exploring further.
                break

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
            raise SearchBudgetExceeded(
                f"Search budget of {self._max} nodes exceeded."
            )


def _min_score(
    combat: Combat,
    counter: _Counter,
    cutoff: int | float = math.inf,
) -> int | float:
    """Recursive DFS with alpha-pruning.

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
    """
    counter.tick()

    # Alpha-prune: this branch is already at least as bad as the best we know.
    if combat.damage_taken >= cutoff:
        return cutoff

    if combat.observe().done:
        return terminal_score(combat)

    best: int | float = cutoff
    for action in _dedupe_actions(combat):
        clone = combat.clone()
        clone.step(action)
        score = _min_score(clone, counter, best)
        if score < best:
            best = score
        if best == 0:
            break  # optimal; prune remaining siblings

    return best
