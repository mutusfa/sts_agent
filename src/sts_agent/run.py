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
from contextlib import contextmanager
from typing import Any, Iterator

import mlflow

from sts_env.combat import Combat
from sts_env.run import run_act1 as _env_run_act1
from sts_env.run.orchestrator import RunResult  # noqa: F401 — re-exported for callers
from sts_env.run.character import Character

from .battle.base import BattleAgent, BattlePlanner, run_agent, run_planner
from .strategy.base import BaseStrategyAgent

log = logging.getLogger(__name__)

__all__ = ["run_act1", "RunResult"]


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
    ) -> None:
        self._planner = planner_or_agent
        self._strategy = strategy_agent
        self._path: list[tuple[int, int]] = []
        self._combat_count: int = 0
        self._sts_map: object = None
        self._run_seed: int = 0

    def _upcoming_encounters(self) -> list[tuple[str, str]]:
        """Derive upcoming (room_type, enc_id) from remaining path.

        Approximate: uses combat count to estimate position in the path.
        Non-combat rooms (rest, shop, event, treasure) are included since
        the discount only needs rough distance estimates.
        """
        upcoming: list[tuple[str, str]] = []
        if self._sts_map is not None:
            # Start from a rough offset based on combat count
            start = min(self._combat_count, len(self._path))
            for i in range(start, len(self._path)):
                floor_num, x_pos = self._path[i]
                node = self._sts_map.get_node(floor_num, x_pos)
                if node is not None:
                    upcoming.append((node.room_type.name.lower(), ""))
        return upcoming

    def run_battle(self, combat: Combat) -> int:
        # Auto-evaluate potion costs for MCTSPlanner
        if hasattr(self._planner, 'potion_costs') and hasattr(combat, '_state'):
            potions = combat._state.potions  # type: ignore[union-attr]
            if potions and not self._planner.potion_costs:
                from .strategy.evaluate_potions import evaluate_potions
                from sts_env.run.character import Character

                # Build a minimal character for evaluate_potions.
                char = Character(
                    player_hp=combat._state.player_hp,  # type: ignore[union-attr]
                    player_max_hp=combat._state.player_max_hp,  # type: ignore[union-attr]
                    deck=[],  # deck not needed for cost computation
                    potions=list(potions),
                )
                upcoming = self._upcoming_encounters()
                self._planner.potion_costs = evaluate_potions(
                    char, upcoming, self._run_seed,
                    possible_encounters=(
                        self._strategy.get_possible_encounters()
                        if hasattr(self._strategy, 'get_possible_encounters')
                        else None
                    ),
                )
        self._combat_count += 1
        return _run_battle(self._planner, combat)

    # Delegate all strategy decisions to the strategy agent.

    def pick_neow(self, options):
        return self._strategy.pick_neow(options)

    def plan_route(self, sts_map, character, seed):
        path = self._strategy.plan_route(sts_map, character, seed)
        self._path = path
        self._path_idx = 0
        self._sts_map = sts_map
        self._run_seed = seed
        return path

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

    def shop(self, inventory, character):
        return self._strategy.shop(inventory, character)

    def pick_boss_relic(self, character, choices):
        return self._strategy.pick_boss_relic(character, choices)

    def pick_potion_to_discard(self, character, new_potion, **kwargs):
        return self._strategy.pick_potion_to_discard(character, new_potion, **kwargs)

    def set_encounter_tracking(self, encounter_queue, hallway_seen, elites_seen):
        self._strategy.set_encounter_tracking(encounter_queue, hallway_seen, elites_seen)


# ---------------------------------------------------------------------------
# MLflow observer
# ---------------------------------------------------------------------------

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
            span.set_attributes({
                "floor": floor,
                "room_type": room_type,
                "hp_before": character.player_hp,
                "max_hp_before": character.player_max_hp,
                "gold_before": character.gold,
                "deck_size_before": len(character.deck),
            })
            attrs: dict[str, Any] = {}
            try:
                yield attrs
            except BaseException as exc:
                # Record the error in the span but don't let MLflow see it
                # — the span closes cleanly, keeping the trace alive.
                span.set_attributes({"error": f"{type(exc).__name__}: {exc}"})
                exc_to_reraise = exc
            finally:
                span.set_attributes({
                    "hp_after": character.player_hp,
                    "max_hp_after": character.player_max_hp,
                    "gold_after": character.gold,
                    "deck_size_after": len(character.deck),
                    **attrs,
                })
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
        If False, use the old linear encounter list (8 floors, backwards compat).

    Returns
    -------
    RunResult with full run statistics.
    """
    if strategy_agent is None:
        strategy_agent = BaseStrategyAgent(seed=seed)

    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "seed": seed,
            "use_map": use_map,
            "strategy_agent": type(strategy_agent).__name__,
        })

    agent = _RunAgentAdapter(planner_or_agent, strategy_agent)
    observer = _MlflowObserver()
    return _env_run_act1(seed, agent, use_map=use_map, observer=observer)
