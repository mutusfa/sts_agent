"""MCTSPlanner — open-loop UCT that collapses stochastic RNG outcomes.

Design overview
---------------
Unlike the exact ``TreeSearchPlanner`` (which fixes RNG per branch), this
solver builds a single tree where each node is keyed by the *RNG-agnostic*
combat state.  Stochastic transitions (card draws, shuffle order, enemy
intents) are not branched over; instead they are absorbed into the running
value estimates at each (node, action) edge.

This is the *open-loop* MCTS variant studied for Slay the Spire and similar
card games (see Dailey, "Slay the Spire AI", and related competitive write-ups).

RNG sampling
------------
Each simulation reseeds the cloned combat's RNG from the planner's own PRNG
before stepping any actions.  This ensures that two simulations taking the
same action sequence experience *different* shuffles and enemy intents, so
edge statistics accumulate genuine RNG variance rather than replaying one
fixed trajectory.  The planner PRNG is seeded from the constructor ``seed``
argument, keeping the whole thing reproducible.

Value model
-----------
We minimise ``terminal_score`` (damage taken, doubled on death).  Each edge
stores running statistics sufficient to compute mean, std, and worst-case:

    n       visit count
    _sum    Σ terminal_score
    _sum_sq Σ terminal_score²
    _max    max terminal_score seen

UCB selection (lower-is-better variant)
----------------------------------------
At a fully-expanded node, choose the action that minimises:

    mean(a) − c · sqrt(ln N_parent / n_a)

where c controls exploration width.  Default: ``start_hp`` (the range of
terminal_score for a surviving player).  Unvisited actions are tried first
(in ``_ordered_actions`` order so move ordering still helps).

Budget
------
``act()`` stops when ``simulations_run == simulations`` OR
``nodes_expanded >= max_nodes``, whichever comes first.  No exception is
raised on budget exhaustion — budget exhaustion is normal termination for an
anytime algorithm.

After each ``act()`` call the attribute ``last_stats`` is populated with::

    mean        expected terminal_score for the chosen root action
    std         standard deviation of terminal_scores for that action
    max         worst-case (maximum) terminal_score for that action
    n           visit count for the chosen root action
    simulations actual simulations run this act()
    nodes       total node expansions this act()

mean/std/max come directly from the chosen root-edge statistics accumulated
during the search.  Because each sim uses a freshly reseeded RNG, these
statistics reflect genuine stochastic variance under the current policy.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

from sts_env.combat import Action, Combat
from sts_env.combat.card import Card
from sts_env.combat.state import ActionType

from .base import _fmt_action, terminal_score
from .tree_search import _ordered_actions, _state_key_base

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public state-key helper (imported by tests)
# ---------------------------------------------------------------------------


def _mcts_state_key(combat: Combat) -> tuple:
    """Hashable state key with RNG excluded — the MCTS node identity."""
    return _state_key_base(combat)


# ---------------------------------------------------------------------------
# Internal data structures
# ---------------------------------------------------------------------------


@dataclass
class _EdgeStats:
    """Per-(parent-node, action) running statistics."""

    n: int = 0
    _sum: float = 0.0
    _sum_sq: float = 0.0
    _max: float = -math.inf
    deaths: int = 0  # rollouts that ended in player death (score >= start_hp)

    def update(self, score: float, *, start_hp: float = math.inf) -> None:
        self.n += 1
        self._sum += score
        self._sum_sq += score * score
        if score > self._max:
            self._max = score
        if score >= start_hp:
            self.deaths += 1

    @property
    def mean(self) -> float:
        return self._sum / self.n if self.n > 0 else math.inf

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        variance = self._sum_sq / self.n - (self._sum / self.n) ** 2
        return math.sqrt(max(0.0, variance))

    @property
    def max(self) -> float:
        return self._max if self.n > 0 else math.inf


@dataclass
class _Node:
    """A node in the MCTS tree."""

    state_key: tuple
    # Actions not yet expanded (in priority order, pop from front).
    # Stored as conceptual keys; see _action_concept_key for the key schema.
    untried_action_keys: list[tuple] = field(default_factory=list)
    # Edge stats keyed by conceptual action key.
    edges: dict[tuple, _EdgeStats] = field(default_factory=dict)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_action_keys) == 0

    def visit_count(self) -> int:
        return sum(e.n for e in self.edges.values())


def _action_concept_key(action: Action, combat: Combat) -> tuple:
    """Stable key for an action, invariant to slot positions.

    - PLAY_CARD:     ("CARD", card_id, cost_override, target_index)
    - USE_POTION:    ("USE",  potion_id, target_index)
    - DISCARD_POTION:("DISCARD", potion_id)
    - END_TURN:      ("END",)

    Mirrors the key shape used by _dedupe_actions so the two helpers stay in sync.
    """
    s = combat._state  # type: ignore[union-attr]
    if action.action_type == ActionType.END_TURN:
        return ("END",)
    if action.action_type == ActionType.PLAY_CARD:
        card: Card = s.piles.hand[action.hand_index]
        return ("CARD", card.card_id, card.cost_override, action.target_index)
    if action.action_type == ActionType.USE_POTION:
        potion_id = s.potions[action.potion_index]
        return ("USE", potion_id, action.target_index)
    if action.action_type == ActionType.DISCARD_POTION:
        potion_id = s.potions[action.potion_index]
        return ("DISCARD", potion_id)
    if action.action_type == ActionType.CHOOSE_CARD:
        return ("CHOOSE", action.choice_index)
    if action.action_type == ActionType.SKIP_CHOICE:
        return ("SKIP",)
    # Fallback for unknown types
    return ("END",)


def _resolve_action(concept_key: tuple, combat: Combat) -> Action | None:
    """Find a valid Action matching a conceptual key in the current combat state.

    Returns None if the action is no longer legal (e.g. card costs more energy
    than available, or potion has already been used).
    """
    tag = concept_key[0]
    if tag == "END":
        return Action.end_turn()
    for action in combat.valid_actions():
        if action.action_type == ActionType.END_TURN:
            continue
        if _action_concept_key(action, combat) == concept_key:
            return action
    return None


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def _heuristic_rollout(combat: Combat) -> int:
    """Play greedily (lowest _order_key first) until combat is done.

    Returns terminal_score at the end state.
    """
    while not combat.observe().done:
        actions = _ordered_actions(combat)
        combat.step(actions[0])
    return terminal_score(combat)


# ---------------------------------------------------------------------------
# MCTSPlanner
# ---------------------------------------------------------------------------


class MCTSPlanner:
    """Open-loop UCT planner that collapses RNG outcomes into value averages.

    Each simulation reseeds the cloned combat's RNG so that edge statistics
    accumulate genuine RNG variance — two simulations taking the same action
    sequence will experience different card draws and enemy intents.

    Parameters
    ----------
    simulations:
        Number of full select→expand→rollout→backup passes per ``act()``.
        Default: 1000.
    max_nodes:
        Maximum total node expansions per ``act()``.  ``None`` = unlimited.
        Simulation loop stops when this is reached, whichever comes first.
    exploration_c:
        UCB exploration constant.  Defaults to ``start_hp``.  Larger values
        explore more; smaller values exploit more.
    seed:
        Seed for the internal PRNG used to reseed each simulation's combat RNG
        and to break UCB ties.  ``None`` = non-deterministic.
    """

    def __init__(
        self,
        simulations: int = 1000,
        max_nodes: int | None = None,
        exploration_c: float | None = None,
        seed: int | None = None,
        pv_min_n: int | None = None,
    ) -> None:
        self.simulations = simulations
        self.max_nodes = max_nodes
        self._exploration_c = exploration_c
        self._rng = random.Random(seed)
        self._pv_min_n_override = pv_min_n
        self.last_stats: dict[str, float] = {}
        self._node_store: dict[tuple, _Node] = {}

    def act(self, combat: Combat) -> Action:
        """Return the action with the lowest estimated expected damage.

        The original ``combat`` is never mutated.
        """
        obs = combat.observe()
        start_hp: float = obs.player_max_hp

        c = self._exploration_c if self._exploration_c is not None else start_hp

        root_key = _mcts_state_key(combat)
        if root_key in self._node_store:
            root_node = self._node_store[root_key]
        else:
            root_node = _Node(
                state_key=root_key,
                untried_action_keys=[
                    _action_concept_key(a, combat) for a in _ordered_actions(combat)
                ],
            )
            self._node_store = {root_key: root_node}
        node_store = self._node_store

        sims_run = 0
        nodes_expanded = 0

        for _ in range(self.simulations):
            if self.max_nodes is not None and nodes_expanded >= self.max_nodes:
                break

            # path stores (node, concept_key) tuples for backup
            path: list[tuple[_Node, tuple]] = []
            current_node = root_node
            current_combat = combat.clone()
            # Reseed so each simulation samples different shuffles / intents,
            # giving edge stats genuine RNG variance.
            current_combat._state.rng._rng.seed(  # type: ignore[union-attr]
                self._rng.randint(0, 2**31 - 1)
            )

            # --- Selection ---
            while not current_combat.observe().done and current_node.is_fully_expanded():
                concept_key = self._select_concept_key(current_node, c)
                action = _resolve_action(concept_key, current_combat)
                if action is None:
                    # Action no longer valid in this sim path — fall back to END_TURN
                    action = Action.end_turn()
                    concept_key = ("END",)
                path.append((current_node, concept_key))
                current_combat.step(action)
                child_key = _mcts_state_key(current_combat)
                if child_key not in node_store:
                    child_node = _Node(
                        state_key=child_key,
                        untried_action_keys=[
                            _action_concept_key(a, current_combat)
                            for a in _ordered_actions(current_combat)
                        ],
                    )
                    node_store[child_key] = child_node
                    nodes_expanded += 1
                    current_node = child_node
                    break
                else:
                    current_node = node_store[child_key]

            # --- Expansion ---
            if not current_combat.observe().done and not current_node.is_fully_expanded():
                concept_key = current_node.untried_action_keys.pop(0)
                action = _resolve_action(concept_key, current_combat)
                if action is None:
                    action = Action.end_turn()
                    concept_key = ("END",)
                path.append((current_node, concept_key))
                current_combat.step(action)
                child_key = _mcts_state_key(current_combat)
                if child_key not in node_store:
                    child_node = _Node(
                        state_key=child_key,
                        untried_action_keys=[
                            _action_concept_key(a, current_combat)
                            for a in _ordered_actions(current_combat)
                        ],
                    )
                    node_store[child_key] = child_node
                    nodes_expanded += 1
                else:
                    current_node = node_store[child_key]

            # --- Rollout ---
            rollout_combat = current_combat.clone()
            score = _heuristic_rollout(rollout_combat)

            # --- Backup ---
            for node, concept_key in path:
                if concept_key not in node.edges:
                    node.edges[concept_key] = _EdgeStats()
                node.edges[concept_key].update(score, start_hp=start_hp)

            sims_run += 1

        # Choose best action at root: lowest mean
        best_concept_key, best_edge = self._best_root_concept(root_node)
        best_action = _resolve_action(best_concept_key, combat) or Action.end_turn()

        pv_edge, pv_depth = self._pv_well_visited_edge(root_node, combat)
        pv = pv_edge if pv_edge is not None else best_edge

        self.last_stats = {
            "mean": best_edge.mean if best_edge.n > 0 else math.nan,
            "std": best_edge.std,
            "max": best_edge.max if best_edge.n > 0 else math.nan,
            "n": float(best_edge.n),
            "deaths": float(best_edge.deaths),
            "simulations": float(sims_run),
            "nodes": float(nodes_expanded),
            "pv_mean": pv.mean if pv.n > 0 else math.nan,
            "pv_std": pv.std,
            "pv_max": pv.max if pv.n > 0 else math.nan,
            "pv_n": float(pv.n),
            "pv_deaths": float(pv.deaths),
            "pv_depth": float(pv_depth),
        }

        log.debug(
            "T=%d MCTS done: action=%s mean=%.1f std=%.1f max=%.1f "
            "n=%d sims=%d nodes=%d",
            obs.turn,
            _fmt_action(best_action, obs.hand),
            self.last_stats["mean"],
            self.last_stats["std"],
            self.last_stats["max"],
            int(self.last_stats["n"]),
            sims_run,
            nodes_expanded,
        )

        return best_action

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_concept_key(self, node: _Node, c: float) -> tuple:
        """UCB selection: argmin mean(a) - c * sqrt(ln N / n_a).

        Returns the conceptual action key with the best UCB value.
        Unvisited actions are never selected here (they stay in
        ``untried_action_keys`` until expansion).
        """
        N = node.visit_count()
        ln_N = math.log(N) if N > 1 else 0.0

        best_score = math.inf
        best_keys: list[tuple] = []

        for concept_key, stats in node.edges.items():
            ucb = stats.mean - c * math.sqrt(ln_N / stats.n)
            if ucb < best_score:
                best_score = ucb
                best_keys = []
            if ucb <= best_score + 1e-12:
                best_keys.append(concept_key)

        if not best_keys:
            raise RuntimeError("No selectable actions at fully-expanded node.")
        return self._rng.choice(best_keys)

    def _effective_pv_min_n(self) -> int:
        """Minimum visit count required to descend one step along the PV."""
        if self._pv_min_n_override is not None:
            return self._pv_min_n_override
        return max(10, int(math.sqrt(self.simulations)))

    def _pv_well_visited_edge(
        self, root: _Node, combat: Combat
    ) -> tuple[_EdgeStats | None, int]:
        """Walk root → argmin-mean child while edge.n >= _effective_pv_min_n.

        Returns (deepest_edge_satisfying_min_n, depth). Falls back to None
        (caller uses chosen root edge) when no edge meets the threshold.
        """
        min_n = self._effective_pv_min_n()
        node = root
        sim_combat = combat.clone()
        last_edge: _EdgeStats | None = None
        last_depth = 0
        depth = 0
        while node.edges:
            key = min(node.edges, key=lambda k: node.edges[k].mean)
            edge = node.edges[key]
            if edge.n < min_n:
                break
            last_edge = edge
            last_depth = depth + 1
            action = _resolve_action(key, sim_combat)
            if action is None:
                break
            sim_combat.step(action)
            if sim_combat.observe().done:
                break
            child = self._node_store.get(_mcts_state_key(sim_combat))
            if child is None or not child.edges:
                break
            node = child
            depth += 1
        return last_edge, last_depth

    def _best_root_concept(self, root: _Node) -> tuple[tuple, _EdgeStats]:
        """Pick the root concept key with the lowest mean score.

        Falls back to the first untried concept key if no edges were visited.
        """
        if not root.edges:
            dummy = _EdgeStats()
            dummy.update(math.inf)
            fallback = root.untried_action_keys[0] if root.untried_action_keys else ("END",)
            return fallback, dummy

        best_key = min(root.edges, key=lambda k: root.edges[k].mean)
        return best_key, root.edges[best_key]
