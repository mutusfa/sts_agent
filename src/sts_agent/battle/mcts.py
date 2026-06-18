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
    sel_depth_mean    mean len(path) at end of selection+expansion across sims
    sel_depth_max     max len(path) across sims
    terminal_during_selection_frac  fraction of sims that reached a done state
                      before the rollout phase (bounds MCTS-Solver payoff)
    would_alpha_cut_at_rollout_frac fraction of sims where damage_taken - mrl
                      >= best_alive_so_far at rollout start (per-sim alpha-abort savings)
    root_visit_top1_frac  fraction of root visits going to the single most-visited edge
    root_visit_top2_frac  fraction of root visits going to the top-2 most-visited edges

Known limitations (accepted for now)
------------------------------------
We deliberately keep high-sim-count open-loop MCTS as the combat pilot while
it doubles as the data-collection and deployment agent.  The oracle and
strategy probes therefore measure value *for this pilot*, not universal deck
strength under expert play.

**Open-loop / no draw conditioning.**  Stochastic transitions (shuffles, enemy
intents) are collapsed into edge averages — the tree does not branch on what
you drew.  More simulations buy breadth across RNG samples, not closed-loop
planning: they cannot fix the inability to conditionally sequence on the cards
currently in hand.  Infinite and high-draw decks suffer most; Ironclad rarely
builds them (they need rare card/relic combos), so the gap affects a small
slice of finished decks but still shows up upstream as **pivot-avoidance
bias** — the agent undervalues rare enablers (Corruption, Dead Branch, draw
engines) that lead to those decks, not just misplaying the finished build.

**High action-count / infinite decks (deferred).**  Best current idea when
revisited: horizon in *actions ahead*, not turns ahead.  Depth alone is only
half the problem — true infinites also need within-turn branching control (the
per-turn action tree explodes), loop/transposition detection, and closed-loop
reaction to mid-turn draws.  Expect closed-loop branching plus loop detection,
not just deeper search.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Literal

from sts_env.combat import Action, Combat
from sts_env.combat.card import Card
from sts_env.combat.state import ActionType

from .base import TerminalOutcome, _fmt_action, max_relic_heal, terminal_score
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

    # Max-HP gained (all rollouts; dead rollouts contribute 0)
    _sum_max_hp: float = 0.0
    _sum_sq_max_hp: float = 0.0

    # Turns to kill (alive rollouts only)
    _sum_turns_alive: float = 0.0
    _n_turns_alive: int = 0

    # Death timeline (dead rollouts only)
    _sum_turns_dead: float = 0.0
    _n_turns_dead: int = 0
    _sum_damage_taken_dead: float = 0.0

    # Tree linkage — set on first expansion; used for pointer-walk PV descent.
    child_state_key: tuple | None = None
    leads_to_terminal: bool = False

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
            self._sum_damage_taken_dead += float(outcome.damage_taken)
            if outcome.turns_elapsed is not None:
                self._sum_turns_dead += float(outcome.turns_elapsed)
                self._n_turns_dead += 1

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

        # Max-HP gained (dead rollouts contribute 0)
        hp_gained = float(outcome.max_hp_gained) if not outcome.player_dead else 0.0
        self._sum_max_hp += hp_gained
        self._sum_sq_max_hp += hp_gained * hp_gained

        # Turns to kill (alive rollouts only)
        if not outcome.player_dead:
            self._sum_turns_alive += outcome.turns_elapsed
            self._n_turns_alive += 1

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
    def mean_turns_dead(self) -> float:
        return (
            self._sum_turns_dead / self._n_turns_dead
            if self._n_turns_dead > 0
            else math.nan
        )

    @property
    def mean_damage_taken_dead(self) -> float:
        return (
            self._sum_damage_taken_dead / self.n_dead if self.n_dead > 0 else math.nan
        )

    @property
    def mean_max_hp_gained(self) -> float:
        return self._sum_max_hp / self.n if self.n > 0 else 0.0

    @property
    def std_max_hp_gained(self) -> float:
        if self.n < 2:
            return 0.0
        variance = self._sum_sq_max_hp / self.n - (self._sum_max_hp / self.n) ** 2
        return math.sqrt(max(0.0, variance))

    @property
    def mean_turns_alive(self) -> float:
        return (
            self._sum_turns_alive / self._n_turns_alive
            if self._n_turns_alive > 0
            else math.nan
        )

    @property
    def death_rate(self) -> float:
        return self.n_dead / self.n if self.n > 0 else 1.0


def _edge_lex_key(stats: _EdgeStats) -> tuple:
    """Lexicographic key for final root-selection and PV descent.

    Lower is better:
      (bucketed_death_rate, mean_damage_alive, -mean_enemy_dmg_dead)

    Death rates below 5% are treated as 0 so open-loop rollout noise does not
    force potion use when the fight is reliably survivable without one.
    Above that threshold, round to 3 decimal places (0.1% granularity).
    When all rollouts died (n_alive == 0), mean_damage_alive == inf — correctly
    ranking any surviving edge above any all-dead edge.
    When no rollouts died, neg_mean_enemy_dmg == inf (tie-break irrelevant).
    """
    dr = stats.death_rate
    if dr < 0.05:
        bucketed_death_rate = 0.0
    else:
        bucketed_death_rate = round(dr, 3)
    neg_mean_enemy_dmg = -stats.mean_enemy_dmg_dead if stats.n_dead > 0 else math.inf
    return (bucketed_death_rate, stats.mean_damage_alive, neg_mean_enemy_dmg)


def _link_edge(
    node: _Node,
    concept_key: tuple,
    child_state_key: tuple,
    *,
    leads_to_terminal: bool,
) -> None:
    """Record child linkage for *concept_key* on first expansion."""
    if concept_key not in node.edges:
        node.edges[concept_key] = _EdgeStats()
    edge = node.edges[concept_key]
    if edge.child_state_key is None:
        edge.child_state_key = child_state_key
        edge.leads_to_terminal = leads_to_terminal


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


def _filter_potion_actions_for_rollout(
    combat: Combat,
    actions: list[Action],
    potion_costs: dict[str, float],
) -> list[Action]:
    """Remove potion actions with positive virtual cost from rollout policy.

    Greedy rollouts must not auto-consume potions — that makes every branch
    pay the virtual cost and rewards early use on lower real damage alone.
    Tree expansion still explores USE_POTION explicitly; survival-critical use
    is handled via death_rate on edges that omit the potion.
    """
    s = combat._state  # type: ignore[union-attr]
    filtered: list[Action] = []
    for action in actions:
        if action.action_type in (ActionType.USE_POTION, ActionType.DISCARD_POTION):
            potion_id = s.potions[action.potion_index]
            if potion_costs.get(potion_id, 0.0) > 0:
                continue
        filtered.append(action)
    return filtered


def _heuristic_rollout(
    combat: Combat,
    *,
    potion_costs: dict[str, float] | None = None,
) -> TerminalOutcome:
    """Play greedily (lowest _order_key first) until combat is done.

    When *potion_costs* assigns positive virtual HP to a potion, rollouts skip
    USE/DISCARD for that potion so counterfactual branches reflect saving it.
    Returns a TerminalOutcome describing the terminal state.
    """
    costs = potion_costs or {}
    _turn = 0
    while not combat._is_done():
        if _turn >= 300:
            break
        actions = _ordered_actions(combat)
        if costs:
            filtered = _filter_potion_actions_for_rollout(combat, actions, costs)
            if filtered:
                actions = filtered
        combat._step_quiet(actions[0])
        _turn += 1
    return terminal_score(combat)


# ---------------------------------------------------------------------------
# MCTSPlanner
# ---------------------------------------------------------------------------

_ROOT_CONVERGE_MAX_DEPTH = 16
_ROOT_STABLE_CI_K = 2.0


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
    pv_early_stop:
        When True (default), stop the simulation loop once the principal
        variation reaches a terminal state with each edge visited at least
        ``_effective_pv_min_n()`` times.  PV is checked every
        ``_pv_check_period()`` simulations (``max(1, floor(sqrt(budget)))``).
    root_stable_early_stop:
        When True (default), stop once the root is fully expanded, every root
        edge has ``n >= _effective_pv_min_n()``, and the lex-best two root
        children either share a combat state key along their lex-best lines
        (within-turn permutation) or have separated tier-encoded means.
        Checked on the same throttled schedule as PV early stop.
    rollout_mode:
        ``"heuristic"`` (default) — shallow select/expand then greedy rollout.
        ``"in_tree"`` — select/expand until combat is done (no rollout tail).
    """

    def __init__(
        self,
        simulations: int = 1000,
        max_nodes: int | None = None,
        exploration_c: float | None = None,
        seed: int | None = None,
        pv_min_n: int | None = None,
        pv_early_stop: bool = True,
        root_stable_early_stop: bool = True,
        rollout_mode: Literal["heuristic", "in_tree"] = "heuristic",
        potion_costs: dict[str, float] | None = None,
    ) -> None:
        self.simulations = simulations
        self.max_nodes = max_nodes
        self._exploration_c = exploration_c
        self._rng = random.Random(seed)
        self._pv_min_n_override = pv_min_n
        self.pv_early_stop = pv_early_stop
        self.root_stable_early_stop = root_stable_early_stop
        self.rollout_mode = rollout_mode
        self.potion_costs: dict[str, float] = potion_costs or {}
        # When set, skip dynamic evaluate_potions() and use these static costs.
        self.static_potion_costs: dict[str, float] | None = None
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

        if self.potion_costs:
            log.debug("MCTSPlanner potion_costs=%s", self.potion_costs)

        actions = _ordered_actions(combat)
        if len(actions) == 1:
            return self._act_single_choice(actions[0], obs=obs, start_hp=start_hp)

        c = self._exploration_c if self._exploration_c is not None else start_hp

        root_key = _mcts_state_key(combat)
        root_cache_hit = root_key in self._node_store
        if root_cache_hit:
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

        state = combat._state  # type: ignore[union-attr]
        mrl = max_relic_heal(state.relics, int(start_hp))

        sims_run = 0
        nodes_expanded = 0
        _potion_cost_total = 0.0
        _potion_cost_n = 0

        # Telemetry counters (no algorithmic effect)
        sel_depth_sum = 0
        sel_depth_max_val = 0
        terminal_in_sel = 0
        in_tree_terminal_count = 0
        would_cut = 0
        best_alive_so_far: float = math.inf
        early_stop_reason = ""

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

            if self.rollout_mode == "in_tree":
                while not current_combat._is_done():
                    if self.max_nodes is not None and nodes_expanded >= self.max_nodes:
                        break
                    if (
                        current_node.is_fully_expanded()
                        and not current_node.untried_action_keys
                        and not current_node.edges
                    ):
                        break
                    path_len_before = len(path)
                    nodes_before = nodes_expanded
                    current_node, nodes_expanded = self._sim_select_expand_once(
                        path=path,
                        current_node=current_node,
                        current_combat=current_combat,
                        node_store=node_store,
                        c=c,
                        nodes_expanded=nodes_expanded,
                    )
                    if (
                        len(path) == path_len_before
                        and nodes_expanded == nodes_before
                    ):
                        break
            else:
                current_node, nodes_expanded = self._sim_select_expand_once(
                    path=path,
                    current_node=current_node,
                    current_combat=current_combat,
                    node_store=node_store,
                    c=c,
                    nodes_expanded=nodes_expanded,
                )

            # --- Telemetry: depth, terminal-in-sel, would-alpha-cut ---
            depth = len(path)
            sel_depth_sum += depth
            if depth > sel_depth_max_val:
                sel_depth_max_val = depth
            _done_before_rollout = current_combat._is_done()
            if _done_before_rollout:
                terminal_in_sel += 1
                if self.rollout_mode == "in_tree":
                    in_tree_terminal_count += 1
            elif (
                self.rollout_mode == "heuristic"
                and current_combat.damage_taken - mrl >= best_alive_so_far
            ):
                would_cut += 1

            # --- Outcome (rollout or in-tree terminal) ---
            if self.rollout_mode == "in_tree":
                outcome = terminal_score(current_combat)
                _score_combat = current_combat
            else:
                rollout_combat = current_combat.clone()
                outcome = _heuristic_rollout(
                    rollout_combat, potion_costs=self.potion_costs
                )
                _score_combat = rollout_combat

            # --- Compute potion virtual cost ---
            _final_potions = _score_combat._state.potions  # type: ignore[union-attr]
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
            if _potion_cost > 0:
                _potion_cost_total += _potion_cost
                _potion_cost_n += 1

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

            # --- Telemetry: update best alive so far ---
            if not outcome.player_dead:
                eff = float(outcome.effective_damage_taken)
                if eff < best_alive_so_far:
                    best_alive_so_far = eff

            sims_run += 1

            if sims_run % self._pv_check_period() == 0:
                if self.pv_early_stop:
                    _, _, pv_terminal_now = self._compute_pv(root_node)
                    if pv_terminal_now:
                        early_stop_reason = "pv_terminal"
                        break
                if self.root_stable_early_stop and self._root_is_stable(root_node):
                    early_stop_reason = "root_stable"
                    break

        # Root visit concentration
        _root_visits = sorted((e.n for e in root_node.edges.values()), reverse=True)
        _total_root = sum(_root_visits) or 1
        _top1_frac = _root_visits[0] / _total_root if _root_visits else 0.0
        _top2_frac = sum(_root_visits[:2]) / _total_root if _root_visits else 0.0

        # Choose best action at root via lexicographic key
        best_concept_key, best_edge = self._best_root_concept(root_node)
        best_action = _resolve_action(best_concept_key, combat) or Action.end_turn()

        pv_edge, pv_depth, pv_terminal = self._compute_pv(root_node)
        pv = pv_edge if pv_edge is not None else best_edge
        pv_death_rate = pv.deaths / pv.n if pv.n > 0 else math.nan
        pv_zero_damage_confirmed = (
            pv_terminal
            and pv.n > 0
            and math.isfinite(pv_death_rate)
            and pv_death_rate < 0.05
            and pv.mean < 1.0
        )
        root_stable = self._root_is_stable(root_node)

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
            "mean_turns_dead": best_edge.mean_turns_dead,
            "mean_damage_taken_dead": best_edge.mean_damage_taken_dead,
            # Budget
            "simulations": float(sims_run),
            "nodes": float(nodes_expanded),
            # Budget telemetry
            "sel_depth_mean": sel_depth_sum / sims_run if sims_run > 0 else 0.0,
            "sel_depth_max": float(sel_depth_max_val),
            "terminal_during_selection_frac": terminal_in_sel / sims_run
            if sims_run > 0
            else 0.0,
            "would_alpha_cut_at_rollout_frac": would_cut / sims_run
            if sims_run > 0
            else 0.0,
            "root_visit_top1_frac": _top1_frac,
            "root_visit_top2_frac": _top2_frac,
            # PV
            "pv_mean": pv.mean if pv.n > 0 else math.nan,
            "pv_std": pv.std,
            "pv_max": pv.max if pv.n > 0 else math.nan,
            "pv_n": float(pv.n),
            "pv_deaths": float(pv.deaths),
            "pv_depth": float(pv_depth),
            "pv_kill_turn_mean": pv.mean_turns_alive,
            "pv_max_hp_gained_mean": pv.mean_max_hp_gained,
            "pv_max_hp_gained_std": pv.std_max_hp_gained,
            "pv_terminal": float(pv_terminal),
            "pv_early_stop_sim": float(sims_run),
            "pv_zero_damage_confirmed": float(pv_zero_damage_confirmed),
            "root_stable": float(root_stable),
            "early_stop_reason": early_stop_reason,
            # Potion penalty
            "potion_cost_n": float(_potion_cost_n),
            "potion_cost_mean": (
                _potion_cost_total / _potion_cost_n if _potion_cost_n else 0.0
            ),
            # Rollout mode / cache
            "rollout_mode_in_tree": float(self.rollout_mode == "in_tree"),
            "root_cache_hit": float(root_cache_hit),
            "in_tree_terminal_frac": in_tree_terminal_count / sims_run
            if sims_run > 0
            else 0.0,
        }

        if self.potion_costs:
            log.debug(
                "T=%d potion penalty: %d/%d rollouts used potions  avg_cost=%.1f",
                obs.turn,
                _potion_cost_n,
                sims_run,
                _potion_cost_total / _potion_cost_n if _potion_cost_n else 0.0,
            )

        pv_expected_dmg = min(pv.mean, float(start_hp)) if pv.n > 0 else math.nan
        pv_kill_t = pv.mean_turns_alive
        log.debug(
            "T=%d → %-16s  dmg=%s±%.0f (%s die)  kill_t=%s  pv_depth=%d  sims=%d",
            obs.turn,
            _fmt_action(best_action, obs.hand),
            f"{pv_expected_dmg:.0f}" if math.isfinite(pv_expected_dmg) else "?",
            pv.std if pv.n > 0 else 0.0,
            f"{pv_death_rate:.0%}" if math.isfinite(pv_death_rate) else "?",
            f"{pv_kill_t:.1f}" if math.isfinite(pv_kill_t) else "?",
            pv_depth,
            sims_run,
        )

        return best_action

    def _act_single_choice(
        self, action: Action, *, obs, start_hp: float
    ) -> Action:
        """Skip search when only one deduplicated root action exists."""
        self.last_stats = {
            "mean": math.nan,
            "std": 0.0,
            "max": math.nan,
            "n": 0.0,
            "deaths": 0.0,
            "death_rate": math.nan,
            "n_alive": 0.0,
            "mean_damage_alive": math.nan,
            "n_dead": 0.0,
            "mean_enemy_dmg_dead": math.nan,
            "mean_turns_dead": math.nan,
            "mean_damage_taken_dead": math.nan,
            "simulations": 0.0,
            "nodes": 0.0,
            "sel_depth_mean": 0.0,
            "sel_depth_max": 0.0,
            "terminal_during_selection_frac": 0.0,
            "would_alpha_cut_at_rollout_frac": 0.0,
            "root_visit_top1_frac": 0.0,
            "root_visit_top2_frac": 0.0,
            "pv_mean": math.nan,
            "pv_std": 0.0,
            "pv_max": math.nan,
            "pv_n": 0.0,
            "pv_deaths": 0.0,
            "pv_depth": 0.0,
            "pv_kill_turn_mean": math.nan,
            "pv_max_hp_gained_mean": 0.0,
            "pv_max_hp_gained_std": 0.0,
            "pv_terminal": 0.0,
            "pv_early_stop_sim": 0.0,
            "pv_zero_damage_confirmed": 0.0,
            "root_stable": 0.0,
            "early_stop_reason": "single_action",
            "potion_cost_n": 0.0,
            "potion_cost_mean": 0.0,
            "rollout_mode_in_tree": float(self.rollout_mode == "in_tree"),
            "root_cache_hit": 0.0,
            "in_tree_terminal_frac": 0.0,
        }
        log.debug(
            "T=%d → %-16s  single action — search skipped",
            obs.turn,
            _fmt_action(action, obs.hand),
        )
        return action

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sim_select_expand_once(
        self,
        *,
        path: list[tuple[_Node, tuple]],
        current_node: _Node,
        current_combat: Combat,
        node_store: dict[tuple, _Node],
        c: float,
        nodes_expanded: int,
    ) -> tuple[_Node, int]:
        """One selection+expansion cycle within a simulation."""
        while (
            not current_combat._is_done()
            and current_node.is_fully_expanded()
            and current_node.edges
        ):
            concept_key = self._select_concept_key(current_node, c)
            if concept_key is None:
                break
            action = _resolve_action(concept_key, current_combat)
            if action is None:
                action = Action.end_turn()
                concept_key = ("END",)
            path.append((current_node, concept_key))
            current_combat._step_quiet(action)
            child_key = _mcts_state_key(current_combat)
            _link_edge(
                current_node,
                concept_key,
                child_key,
                leads_to_terminal=current_combat.observe().done,
            )
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
            current_node = node_store[child_key]

        if (
            not current_combat._is_done()
            and not current_node.is_fully_expanded()
        ):
            concept_key = current_node.untried_action_keys.pop(0)
            action = _resolve_action(concept_key, current_combat)
            if action is None:
                action = Action.end_turn()
                concept_key = ("END",)
            path.append((current_node, concept_key))
            current_combat._step_quiet(action)
            child_key = _mcts_state_key(current_combat)
            _link_edge(
                current_node,
                concept_key,
                child_key,
                leads_to_terminal=current_combat.observe().done,
            )
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

        return current_node, nodes_expanded

    def _select_concept_key(self, node: _Node, c: float) -> tuple | None:
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
            if stats.n <= 0:
                continue
            ucb = stats.mean - c * math.sqrt(ln_N / stats.n)
            if ucb < best_score:
                best_score = ucb
                best_keys = []
            if ucb <= best_score + 1e-12:
                best_keys.append(concept_key)

        if not best_keys:
            return None
        return self._rng.choice(best_keys)

    def _effective_pv_min_n(self) -> int:
        """Minimum visit count required to descend one step along the PV."""
        if self._pv_min_n_override is not None:
            return self._pv_min_n_override
        return max(10, int(math.sqrt(self.simulations)))

    def _pv_check_period(self) -> int:
        """Simulations between PV early-stop checks."""
        return max(1, int(math.sqrt(self.simulations)))

    def _compute_pv(
        self, root: _Node
    ) -> tuple[_EdgeStats | None, int, bool]:
        """Walk root → lex-best child while edge.n >= _effective_pv_min_n.

        Follows cached ``child_state_key`` links — no combat replay.
        Returns (deepest_edge_satisfying_min_n, depth, pv_terminal).
        ``pv_terminal`` is True when the walk reaches a done combat state.
        Falls back to None (caller uses chosen root edge) when no edge meets
        the threshold.
        """
        min_n = self._effective_pv_min_n()
        node = root
        last_edge: _EdgeStats | None = None
        last_depth = 0
        depth = 0
        pv_terminal = False
        while node.edges:
            key = min(node.edges, key=lambda k: _edge_lex_key(node.edges[k]))
            edge = node.edges[key]
            if edge.n < min_n:
                break
            last_edge = edge
            last_depth = depth + 1
            if edge.leads_to_terminal:
                pv_terminal = True
                break
            if edge.child_state_key is None:
                break
            child = self._node_store.get(edge.child_state_key)
            if child is None or not child.edges:
                break
            node = child
            depth += 1
        return last_edge, last_depth, pv_terminal

    def _root_is_stable(self, root: _Node) -> bool:
        """True when root is fully explored and top-two are tied or separated."""
        if not root.is_fully_expanded():
            return False
        if len(root.edges) < 2:
            return False

        min_n = self._effective_pv_min_n()
        if any(e.n < min_n for e in root.edges.values()):
            return False

        ordered = sorted(root.edges.values(), key=_edge_lex_key)
        best, runner = ordered[0], ordered[1]
        return self._children_converge(best, runner) or self._edges_statistically_separable(
            best, runner
        )

    def _edges_statistically_separable(
        self, best: _EdgeStats, runner: _EdgeStats
    ) -> bool:
        """True when tier-encoded CIs for lex-best vs runner-up do not overlap."""
        if best.n < 2 or runner.n < 2:
            return False
        k = _ROOT_STABLE_CI_K
        se_best = best.std / math.sqrt(best.n)
        se_runner = runner.std / math.sqrt(runner.n)
        return best.mean + k * se_best < runner.mean - k * se_runner

    def _children_converge(self, best: _EdgeStats, runner: _EdgeStats) -> bool:
        """True when lex-best lines from two root edges share a state key."""
        return bool(
            self._best_line_keys(best) & self._best_line_keys(runner)
        )

    def _best_line_keys(self, edge: _EdgeStats) -> set[tuple]:
        """State keys along the lex-best line from *edge*'s child (pointer walk)."""
        keys: set[tuple] = set()
        ck = edge.child_state_key
        depth = 0
        while ck is not None and depth < _ROOT_CONVERGE_MAX_DEPTH:
            keys.add(ck)
            node = self._node_store.get(ck)
            if node is None or not node.edges:
                break
            nkey = min(node.edges, key=lambda k: _edge_lex_key(node.edges[k]))
            nedge = node.edges[nkey]
            if nedge.leads_to_terminal:
                if nedge.child_state_key is not None:
                    keys.add(nedge.child_state_key)
                break
            ck = nedge.child_state_key
            depth += 1
        return keys

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
