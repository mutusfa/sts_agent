"""Run-level adapter — wires sts_agent planners/agents into the sts_env orchestrator.

The actual run loop lives in :mod:`sts_env.run.orchestrator`.  This module
provides:

* :class:`_RunAgentAdapter` — combines a ``BattlePlanner | BattleAgent`` with
  a :class:`~sts_agent.strategy.BaseStrategyAgent` into the single
  :class:`~sts_env.run.RunAgentProtocol` the orchestrator expects.
* :class:`_MlflowObserver` — emits a per-floor MLflow child span so the run
  retains the same observability as before.
* :func:`run_act1` — public entry point (unchanged signature, MLflow-traced).

Usage::

    from sts_agent.run import run_act1
    from sts_agent.battle.mcts import MCTSPlanner
    from sts_agent.strategy import StrategyAgent

    result = run_act1(MCTSPlanner(), seed=42, strategy_agent=StrategyAgent())
    print(result)
"""

from __future__ import annotations

import inspect
import logging
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import mlflow

from sts_env.combat import Combat
from sts_env.run import run_act1 as _env_run_act1
from sts_env.run.orchestrator import PotionRecord, RunResult  # noqa: F401 — re-exported
from sts_env.run.character import Character

from .battle.base import BattleAgent, BattlePlanner, run_agent, run_planner
from .strategy.base import BaseStrategyAgent

log = logging.getLogger(__name__)

__all__ = ["run_act1", "RunResult", "PotionRecord", "format_potion_log"]


# ---------------------------------------------------------------------------
# Battle dispatch helper
# ---------------------------------------------------------------------------


def _run_battle(
    planner_or_agent: BattlePlanner | BattleAgent,
    combat: Combat,
) -> int:
    """Dispatch to run_planner or run_agent based on the act() signature."""
    try:
        sig = inspect.signature(planner_or_agent.act)
        params = list(sig.parameters.keys())
        is_planner = len(params) == 1 and params[0] not in ("obs", "observation")
    except (ValueError, TypeError):
        is_planner = True
    if is_planner:
        return run_planner(planner_or_agent, combat)  # type: ignore[arg-type]
    else:
        return run_agent(planner_or_agent, combat)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Adapter: (planner_or_agent, strategy_agent) → RunAgentProtocol
# ---------------------------------------------------------------------------


def _deck_from_combat(combat: Combat) -> list[str]:
    """Reconstruct the deck card-id list from combat piles."""
    piles = combat._state.piles  # type: ignore[union-attr]
    return [
        card.card_id
        for card in piles.draw + piles.hand + piles.discard + piles.exhaust
    ]


# ---------------------------------------------------------------------------
# Potion lifecycle tracking
# ---------------------------------------------------------------------------


@dataclass
class _PotionTracker:
    """Track per-potion gained/spent floors across a run."""

    _current_floor: int = 0
    _entries: list[PotionRecord] | None = None
    _spent_this_floor: Counter[str] | None = None
    _floor_start_potions: Counter[str] | None = None

    def __post_init__(self) -> None:
        self._entries = []
        self._spent_this_floor = Counter()
        self._floor_start_potions = Counter()

    @property
    def entries(self) -> list[PotionRecord]:
        assert self._entries is not None
        return self._entries

    @property
    def current_floor(self) -> int:
        return self._current_floor

    def begin_floor(self, floor: int, potions: list[str]) -> None:
        self._current_floor = floor
        self._spent_this_floor = Counter()
        self._floor_start_potions = Counter(potions)

    def record_spent(self, potion_ids: list[str]) -> None:
        assert self._spent_this_floor is not None
        assert self._entries is not None
        for potion_id in potion_ids:
            self._spent_this_floor[potion_id] += 1
            for entry in self._entries:
                if entry.potion_id == potion_id and entry.spent_floor is None:
                    entry.spent_floor = self._current_floor
                    break
            else:
                self._entries.append(
                    PotionRecord(
                        potion_id=potion_id,
                        gained_floor=0,
                        spent_floor=self._current_floor,
                    )
                )

    def end_floor(self, potions: list[str]) -> None:
        assert self._floor_start_potions is not None
        assert self._spent_this_floor is not None
        assert self._entries is not None
        end_counts = Counter(potions)
        for potion_id, end_count in end_counts.items():
            start_count = self._floor_start_potions.get(potion_id, 0)
            spent_count = self._spent_this_floor.get(potion_id, 0)
            gained_count = end_count - (start_count - spent_count)
            for _ in range(max(0, gained_count)):
                self._entries.append(
                    PotionRecord(
                        potion_id=potion_id,
                        gained_floor=self._current_floor,
                    )
                )


def _log_potion_summary(result: RunResult) -> None:
    if not result.potion_log:
        return
    log.info("Potion log: %s", _format_potion_log(result.potion_log))


def _format_potion_log(entries: list[PotionRecord]) -> str:
    if not entries:
        return "none"
    parts: list[str] = []
    for entry in entries:
        if entry.spent_floor is None:
            parts.append(
                f"{entry.potion_id}: gained floor {entry.gained_floor}, unused"
            )
        else:
            parts.append(
                f"{entry.potion_id}: gained floor {entry.gained_floor}, "
                f"spent floor {entry.spent_floor}"
            )
    return "; ".join(parts)


def format_potion_log(result: RunResult) -> str:
    """Format potion lifecycle entries for run summaries."""
    return _format_potion_log(result.potion_log)


class _RunAgentAdapter:
    """Combines a battle planner/agent with a strategy agent into one object
    satisfying RunAgentProtocol.

    Automatically evaluates potion virtual costs before each combat (when using
    MCTSPlanner) and injects them into the planner.
    """

    def __init__(
        self,
        planner_or_agent: BattlePlanner | BattleAgent,
        strategy_agent: BaseStrategyAgent,
        *,
        potion_tracker: _PotionTracker | None = None,
    ) -> None:
        self._planner = planner_or_agent
        self._strategy = strategy_agent
        self._potion_tracker = potion_tracker or _PotionTracker()
        self._path: list[tuple[int, int]] = []
        self._combat_count: int = 0
        self._sts_map: object = None
        self._run_seed: int = 0
        self._walk_started: bool = False

    def begin_map_run(self, sts_map, seed: int) -> None:
        self._sts_map = sts_map
        self._run_seed = seed
        self._path = []
        self._combat_count = 0
        self._walk_started = False
        self._strategy.begin_map_run(sts_map, seed)

    def on_map_step(self, coord: tuple[int, int]) -> None:
        self._path.append(coord)
        self._walk_started = True
        self._strategy.on_map_step(coord)

    def provisional_remaining_path(
        self,
        sts_map,
        character: Character,
        current_position: tuple[int, int],
    ) -> list[tuple[int, int]]:
        from .strategy.map_routing import (
            count_shops_visited,
            provisional_remaining_path,
        )

        shops = count_shops_visited(sts_map, self._path)
        return provisional_remaining_path(
            sts_map, current_position, shops_visited=shops,
        )

    def _upcoming_encounters(self) -> list[tuple[str, str]]:
        """Derive upcoming (room_type, enc_id) from committed + provisional path."""
        if self._sts_map is None or not self._path:
            return []
        from .strategy.map_routing import (
            count_shops_visited,
            path_to_upcoming,
            provisional_remaining_path,
        )

        if not self._walk_started:
            return path_to_upcoming(self._sts_map, self._path)

        current = self._path[-1]
        shops = count_shops_visited(self._sts_map, self._path)
        suffix = provisional_remaining_path(
            self._sts_map, current, shops_visited=shops,
        )
        return path_to_upcoming(self._sts_map, suffix)

    def run_battle(self, combat: Combat) -> int:
        potions_before = (
            list(combat._state.potions)  # type: ignore[union-attr]
            if hasattr(combat, "_state")
            else []
        )
        # Auto-evaluate potion costs for MCTSPlanner
        if hasattr(self._planner, "potion_costs") and hasattr(combat, "_state"):
            potions = combat._state.potions  # type: ignore[union-attr]
            if potions:
                from .strategy.evaluate_potions import evaluate_potions
                from sts_env.run.character import Character

                # Build a minimal character for evaluate_potions.
                char = Character(
                    player_hp=combat._state.player_hp,  # type: ignore[union-attr]
                    player_max_hp=combat._state.player_max_hp,  # type: ignore[union-attr]
                    deck=_deck_from_combat(combat),
                    potions=list(potions),
                )
                upcoming = self._upcoming_encounters()
                self._planner.potion_costs = evaluate_potions(
                    char,
                    upcoming,
                    self._run_seed,
                    possible_encounters=(
                        self._strategy.get_possible_encounters()
                        if hasattr(self._strategy, "get_possible_encounters")
                        else None
                    ),
                )
                log.debug("Potion costs: %s", self._planner.potion_costs)
            else:
                self._planner.potion_costs = {}
        self._combat_count += 1
        damage = _run_battle(self._planner, combat)
        if hasattr(combat, "_state"):
            potions_after = list(combat._state.potions)  # type: ignore[union-attr]
            spent = list((Counter(potions_before) - Counter(potions_after)).elements())
            if spent:
                self._potion_tracker.record_spent(spent)
                for potion_id in spent:
                    log.info(
                        "  Potion spent: %s on floor %d",
                        potion_id,
                        self._potion_tracker.current_floor,
                    )
        return damage

    # Delegate all strategy decisions to the strategy agent.

    def pick_neow(self, options):
        return self._strategy.pick_neow(options)

    def pick_branch(self, sts_map, character, current, seed):
        return self._strategy.pick_branch(sts_map, character, current, seed)

    def pick_card(self, character, card_choices, upcoming_encounters, seed, **kwargs):
        return self._strategy.pick_card(
            character, card_choices, upcoming_encounters, seed, **kwargs
        )

    def pick_rest_choice(self, character, **kwargs):
        return self._strategy.pick_rest_choice(character, **kwargs)

    def pick_event_choice(self, event, character, **kwargs):
        return self._strategy.pick_event_choice(event, character, **kwargs)

    def pick_card_to_remove(self, character, **kwargs):
        return self._strategy.pick_card_to_remove(character, **kwargs)

    def pick_card_to_transform(self, character, **kwargs):
        return self._strategy.pick_card_to_transform(character, **kwargs)

    def pick_card_to_upgrade(self, character, **kwargs):
        return self._strategy.pick_card_to_upgrade(character, **kwargs)

    def shop(self, inventory, character):
        return self._strategy.shop(inventory, character)

    def pick_boss_relic(self, character, choices):
        return self._strategy.pick_boss_relic(character, choices)

    def pick_potion_to_discard(self, character, new_potion, **kwargs):
        return self._strategy.pick_potion_to_discard(character, new_potion, **kwargs)

    def set_encounter_tracking(self, encounter_queue, hallway_seen, elites_seen):
        self._strategy.set_encounter_tracking(
            encounter_queue, hallway_seen, elites_seen
        )


# ---------------------------------------------------------------------------
# MLflow observer
# ---------------------------------------------------------------------------


class _CombinedObserver:
    """FloorObserver that tracks potion lifecycle and delegates to MLflow."""

    def __init__(
        self,
        potion_tracker: _PotionTracker,
        inner: _MlflowObserver,
    ) -> None:
        self._potion_tracker = potion_tracker
        self._inner = inner

    @contextmanager
    def floor_scope(
        self,
        floor: int,
        room_type: str,
        character: Character,
    ) -> Iterator[dict[str, Any]]:
        self._potion_tracker.begin_floor(floor, list(character.potions))
        with self._inner.floor_scope(floor, room_type, character) as attrs:
            try:
                yield attrs
            finally:
                self._potion_tracker.end_floor(list(character.potions))


class _MlflowObserver:
    """FloorObserver that wraps each floor in an MLflow child span.

    Exceptions inside a floor are recorded as span attributes but do NOT
    propagate through the MLflow span context — this prevents one bad floor
    from killing the entire trace.  The exception is re-raised *after* the
    span closes cleanly so the orchestrator can decide what to do.
    """

    @contextmanager
    def floor_scope(
        self,
        floor: int,
        room_type: str,
        character: Character,
    ) -> Iterator[dict[str, Any]]:
        exc_to_reraise: BaseException | None = None
        with mlflow.start_span(name=f"floor_{floor}_{room_type}") as span:
            span.set_attributes(
                {
                    "floor": floor,
                    "room_type": room_type,
                    "hp_before": character.player_hp,
                    "max_hp_before": character.player_max_hp,
                    "gold_before": character.gold,
                    "deck_size_before": len(character.deck),
                }
            )
            attrs: dict[str, Any] = {}
            try:
                yield attrs
            except BaseException as exc:
                # Record the error in the span but don't let MLflow see it
                # — the span closes cleanly, keeping the trace alive.
                span.set_attributes({"error": f"{type(exc).__name__}: {exc}"})
                exc_to_reraise = exc
            finally:
                span.set_attributes(
                    {
                        "hp_after": character.player_hp,
                        "max_hp_after": character.player_max_hp,
                        "gold_after": character.gold,
                        "deck_size_after": len(character.deck),
                        **attrs,
                    }
                )
        # Re-raise outside the span context so the orchestrator sees it,
        # but the MLflow span already closed cleanly (status OK).
        if exc_to_reraise is not None:
            raise exc_to_reraise


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@mlflow.trace(name="run_act1")
def run_act1(
    planner_or_agent: BattlePlanner | BattleAgent,
    seed: int,
    *,
    strategy_agent: BaseStrategyAgent | None = None,
    use_map: bool = True,
) -> RunResult:
    """Run a full Act 1 scenario.

    Parameters
    ----------
    planner_or_agent:
        The battle AI for combat.
    seed:
        Master seed for the run.
    strategy_agent:
        A :class:`~sts_agent.strategy.BaseStrategyAgent` (or subclass) that
        owns every per-floor strategic decision.  When ``None``, a fresh
        ``BaseStrategyAgent`` seeded from ``seed`` is used.
    use_map:
        If True (default), generate a branching map and walk it (15 floors).
        If False, use the fixed 8-floor linear encounter list.  A single-path
        map is still registered for scored map views.

    Returns
    -------
    RunResult with full run statistics.
    """
    if strategy_agent is None:
        strategy_agent = BaseStrategyAgent(seed=seed)

    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes(
            {
                "seed": seed,
                "use_map": use_map,
                "strategy_agent": type(strategy_agent).__name__,
            }
        )

    agent = _RunAgentAdapter(planner_or_agent, strategy_agent)
    if not use_map:
        from sts_env.run.scenarios import act1_encounters

        from .strategy.map_routing import linear_scenario_map

        agent.begin_map_run(linear_scenario_map(act1_encounters(seed), seed), seed)
    potion_tracker = agent._potion_tracker
    observer = _CombinedObserver(potion_tracker, _MlflowObserver())
    result = _env_run_act1(seed, agent, use_map=use_map, observer=observer)
    result.potion_log = list(potion_tracker.entries)
    _log_potion_summary(result)
    return result
