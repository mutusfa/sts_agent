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

Value model — lexicographic two-tier objective
----------------------------------------------
Priority 1: minimise effective_damage_taken while surviving (death_rate = 0).
Priority 2: if survival is unreachable, maximise enemy damage dealt.

effective_damage_taken accounts for post-combat relic heals (BurningBlood, MeatOnTheBone,
etc.): it is damage_taken minus whatever would be healed back at the end of combat.
Negative values are valid — they represent a net-gain combat (more HP healed than lost).
This lets the planner correctly price in "free" damage that will be fully recovered.

Each edge tracks two separate pools:

  alive pool  → running stats of effective_damage_taken for rollouts where player survived.
  dead pool   → running stats of enemy_damage_dealt for rollouts where player died.
  ucb_score   → tier-encoded scalar used only for UCB in-tree exploration:
                  alive rollout → effective_damage_taken
                  dead  rollout → start_hp + 1 + (total_enemy_hp − enemy_dmg_dealt)
                Both ranges never overlap, preserving ordinal correctness for UCB.

Final root selection and PV descent use a lexicographic key (lower = better):

    (round(death_rate, 3), mean_effective_damage_alive, -mean_enemy_dmg_dead)

Rounding to 3 decimal places buckets noise below 0.1 %, so a single unlucky
rollout in 10 000 sims (death_rate=0.0001 → 0.0) does not flip the alive tier.

UCB selection (lower-is-better variant)
----------------------------------------
At a fully-expanded node, choose the action that minimises:

    ucb_mean(a) − c · sqrt(ln N_parent / n_a)

where c controls exploration width.  Default: ``start_hp``.  Unvisited actions
are tried first (in ``_ordered_actions`` order so move ordering still helps).

Budget
------
``act()`` stops when ``simulations_run == simulations`` OR
``nodes_expanded >= max_nodes``, whichever comes first.  No exception is
raised on budget exhaustion — budget exhaustion is normal termination for an
anytime algorithm.

After each ``act()`` call the attribute ``last_stats`` is populated with::

    mean              tier-encoded scalar mean for the chosen root action
    std               std of the tier-encoded scalar
    max               worst-case tier-encoded scalar
    n                 total visit count for the chosen root action
    death_rate        fraction of rollouts ending in player death
    n_alive           rollouts where player survived
    mean_damage_alive mean damage taken in surviving rollouts (inf if none)
    n_dead            rollouts where player died
    mean_enemy_dmg_dead mean enemy damage dealt in dying rollouts (nan if none)
    simulations       actual simulations run this act()
    nodes             total node expansions this act()
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field

from sts_env.combat import Action, Combat
from sts_env.combat.card import Card
from sts_env.combat.state import ActionType

from .base import TerminalOutcome, _fmt_action, terminal_score
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
    """Per-(parent-node, action) running statistics.

    Two separate pools:
      alive pool  — effective_damage_taken in rollouts where the player survived.
      dead pool   — enemy damage dealt in rollouts where the player died.
      ucb scalar  — tier-encoded scalar for UCB exploration.
    """

    # Alive pool
    n_alive: int = 0
    _sum_alive: float = 0.0
    _sum_sq_alive: float = 0.0
    _max_alive: float = -math.inf

    # Dead pool
    n_dead: int = 0
    _sum_dead: float = 0.0
    _sum_sq_dead: float = 0.0
    _max_dead: float = -math.inf

    # Tier-encoded UCB scalar (shared)
    _sum_ucb: float = 0.0
    _sum_sq_ucb: float = 0.0
    _max_ucb: float = -math.inf

    @property
    def n(self) -> int:
        return self.n_alive + self.n_dead

    @property
    def deaths(self) -> int:
        return self.n_dead

    def update(
        self,
        outcome: TerminalOutcome,
        *,
        start_hp: float,
        total_initial_enemy_hp: float,
        potion_cost: float = 0.0,
    ) -> None:
        if not outcome.player_dead:
            s = float(outcome.effective_damage_taken) + potion_cost
            self.n_alive += 1
            self._sum_alive += s
            self._sum_sq_alive += s * s
            if s > self._max_alive:
                self._max_alive = s
        else:
            s = float(outcome.enemy_damage_dealt)
            self.n_dead += 1
            self._sum_dead += s
            self._sum_sq_dead += s * s
            if s > self._max_dead:
                self._max_dead = s

        # Tier-encoded scalar: alive → damage_taken + potion_cost; dead → start_hp + 1 + remaining_hp
        if not outcome.player_dead:
            ucb = float(outcome.effective_damage_taken) + potion_cost
        else:
            remaining = total_initial_enemy_hp - outcome.enemy_damage_dealt
            ucb = start_hp + 1.0 + remaining
        self._sum_ucb += ucb
        self._sum_sq_ucb += ucb * ucb
        if ucb > self._max_ucb:
            self._max_ucb = ucb

    # UCB scalar properties (used for in-tree exploration)

    @property
    def mean(self) -> float:
        """Tier-encoded scalar mean — used for UCB exploration."""
        return self._sum_ucb / self.n if self.n > 0 else math.inf

    @property
    def std(self) -> float:
        """Std of the tier-encoded scalar."""
        if self.n < 2:
            return 0.0
        variance = self._sum_sq_ucb / self.n - (self._sum_ucb / self.n) ** 2
        return math.sqrt(max(0.0, variance))

    @property
    def max(self) -> float:
        """Worst-case tier-encoded scalar."""
        return self._max_ucb if self.n > 0 else math.inf

    # Alive-pool properties (effective_damage_taken — relic heals subtracted)

    @property
    def mean_damage_alive(self) -> float:
        return self._sum_alive / self.n_alive if self.n_alive > 0 else math.inf

    # Dead-pool properties

    @property
    def mean_enemy_dmg_dead(self) -> float:
        return self._sum_dead / self.n_dead if self.n_dead > 0 else math.nan

    @property
    def death_rate(self) -> float:
        return self.n_dead / self.n if self.n > 0 else 1.0


def _edge_lex_key(stats: _EdgeStats) -> tuple:
    """Lexicographic key for final root-selection and PV descent.

    Lower is better:
      (bucketed_death_rate, mean_damage_alive, -mean_enemy_dmg_dead)

    Bucketing at 3 decimal places (0.1 % granularity) ensures that a single
    unlucky rollout in > 1000 sims doesn't flip the alive-damage tier.
    When all rollouts died (n_alive == 0), mean_damage_alive == inf — correctly
    ranking any surviving edge above any all-dead edge.
    When no rollouts died, neg_mean_enemy_dmg == inf (tie-break irrelevant).
    """
    bucketed_death_rate = round(stats.death_rate, 3)
    neg_mean_enemy_dmg = -stats.mean_enemy_dmg_dead if stats.n_dead > 0 else math.inf
    return (bucketed_death_rate, stats.mean_damage_alive, neg_mean_enemy_dmg)


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


def _heuristic_rollout(combat: Combat) -> TerminalOutcome:
    """Play greedily (lowest _order_key first) until combat is done.

    Returns a TerminalOutcome describing the terminal state.
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
    potion_costs:
        Mapping from potion_id to virtual HP cost.  When a potion is used
        during a rollout, its cost is added to the reward so MCTS treats
        using a potion as "spending" HP.  Default: empty dict (potions free).
    """

    def __init__(
        self,
        simulations: int = 1000,
        max_nodes: int | None = None,
        exploration_c: float | None = None,
        seed: int | None = None,
        pv_min_n: int | None = None,
        potion_costs: dict[str, float] | None = None,
    ) -> None:
        self.simulations = simulations
        self.max_nodes = max_nodes
        self._exploration_c = exploration_c
        self._rng = random.Random(seed)
        self._pv_min_n_override = pv_min_n
        self.potion_costs: dict[str, float] = potion_costs or {}
        self.last_stats: dict[str, float] = {}
        self._node_store: dict[tuple, _Node] = {}

    def act(self, combat: Combat) -> Action:
        """Return the action with the lowest lexicographic objective.

        The original ``combat`` is never mutated.
        """
        obs = combat.observe()
        start_hp: float = obs.player_max_hp
        total_initial_enemy_hp: float = sum(
            e.max_hp
            for e in combat._state.enemies  # type: ignore[union-attr]
        )

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

            # Capture initial potions to compute virtual cost of potions used.
            _initial_potions = list(current_combat._state.potions)  # type: ignore[union-attr]

            # --- Selection ---
            while (
                not current_combat.observe().done and current_node.is_fully_expanded()
            ):
                concept_key = self._select_concept_key(current_node, c)
                action = _resolve_action(concept_key, current_combat)
                if action is None:
                    # Action no longer valid in this sim path — fall back to END_TURN
                    action = Action.end_turn()
                    concept_key = ("END",)
                path.append((current_node, concept_key))
                current_combat.step(action)
                # Some actions are not fully deterministic without tracking rng.
                # Say, we could have tried to draw cards from a state that is identical
                # in all but rng to the current state. Due to rng differences, we might
                # have ended up in a different state.
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
            if (
                not current_combat.observe().done
                and not current_node.is_fully_expanded()
            ):
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
            outcome = _heuristic_rollout(rollout_combat)

            # --- Compute potion virtual cost for this rollout ---
            # Diff initial potions vs final state to find used ones.
            _final_potions = rollout_combat._state.potions  # type: ignore[union-attr]
            _used_counts: dict[str, int] = {}
            for p in _initial_potions:
                _used_counts[p] = _used_counts.get(p, 0) + 1
            for p in _final_potions:
                _used_counts[p] = _used_counts.get(p, 0) - 1
            _potion_cost = sum(
                self.potion_costs.get(p, 0.0) * cnt
                for p, cnt in _used_counts.items()
                if cnt > 0
            )

            # --- Backup ---
            for node, concept_key in path:
                if concept_key not in node.edges:
                    node.edges[concept_key] = _EdgeStats()
                node.edges[concept_key].update(
                    outcome,
                    start_hp=start_hp,
                    total_initial_enemy_hp=total_initial_enemy_hp,
                    potion_cost=_potion_cost,
                )

            sims_run += 1

        # Choose best action at root via lexicographic key
        best_concept_key, best_edge = self._best_root_concept(root_node)
        best_action = _resolve_action(best_concept_key, combat) or Action.end_turn()

        pv_edge, pv_depth = self._pv_well_visited_edge(root_node, combat)
        pv = pv_edge if pv_edge is not None else best_edge

        self.last_stats = {
            # Tier-encoded scalar stats (for log continuity)
            "mean": best_edge.mean if best_edge.n > 0 else math.nan,
            "std": best_edge.std,
            "max": best_edge.max if best_edge.n > 0 else math.nan,
            "n": float(best_edge.n),
            "deaths": float(best_edge.deaths),
            # Lexicographic tier breakdown
            "death_rate": best_edge.death_rate if best_edge.n > 0 else math.nan,
            "n_alive": float(best_edge.n_alive),
            "mean_damage_alive": best_edge.mean_damage_alive,
            "n_dead": float(best_edge.n_dead),
            "mean_enemy_dmg_dead": best_edge.mean_enemy_dmg_dead,
            # Budget
            "simulations": float(sims_run),
            "nodes": float(nodes_expanded),
            # PV
            "pv_mean": pv.mean if pv.n > 0 else math.nan,
            "pv_std": pv.std,
            "pv_max": pv.max if pv.n > 0 else math.nan,
            "pv_n": float(pv.n),
            "pv_deaths": float(pv.deaths),
            "pv_depth": float(pv_depth),
        }

        log.debug(
            "T=%d MCTS done: action=%s mean=%.1f std=%.1f max=%.1f "
            "n=%d death_rate=%.1f%% sims=%d nodes=%d",
            obs.turn,
            _fmt_action(best_action, obs.hand),
            self.last_stats["mean"],
            self.last_stats["std"],
            self.last_stats["max"],
            int(self.last_stats["n"]),
            self.last_stats["death_rate"] * 100
            if math.isfinite(self.last_stats["death_rate"])
            else float("nan"),
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
        """Walk root → lex-best child while edge.n >= _effective_pv_min_n.

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
            key = min(node.edges, key=lambda k: _edge_lex_key(node.edges[k]))
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
        """Pick the root concept key with the best lexicographic objective.

        Falls back to the first untried concept key if no edges were visited.
        """
        if not root.edges:
            dummy = _EdgeStats()
            fallback = (
                root.untried_action_keys[0] if root.untried_action_keys else ("END",)
            )
            return fallback, dummy

        best_key = min(root.edges, key=lambda k: _edge_lex_key(root.edges[k]))
        return best_key, root.edges[best_key]
