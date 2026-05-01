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
    satisfying RunAgentProtocol."""

    def __init__(
        self,
        planner_or_agent: BattlePlanner | BattleAgent,
        strategy_agent: BaseStrategyAgent,
    ) -> None:
        self._planner = planner_or_agent
        self._strategy = strategy_agent

    def run_battle(self, combat: Combat) -> int:
        return _run_battle(self._planner, combat)

    # Delegate all strategy decisions to the strategy agent.

    def pick_neow(self, options):
        return self._strategy.pick_neow(options)

    def plan_route(self, sts_map, character, seed):
        return self._strategy.plan_route(sts_map, character, seed)

    def pick_card(self, character, card_choices, upcoming_encounters, seed, **kwargs):
        return self._strategy.pick_card(
            character, card_choices, upcoming_encounters, seed, **kwargs
        )

    def pick_rest_choice(self, character):
        return self._strategy.pick_rest_choice(character)

    def pick_event_choice(self, event, character):
        return self._strategy.pick_event_choice(event, character)

    def shop(self, inventory, character):
        return self._strategy.shop(inventory, character)

    def pick_boss_relic(self, character, choices):
        return self._strategy.pick_boss_relic(character, choices)


# ---------------------------------------------------------------------------
# MLflow observer
# ---------------------------------------------------------------------------

class _MlflowObserver:
    """FloorObserver that wraps each floor in an MLflow child span."""

    @contextmanager
    def floor_scope(
        self,
        floor: int,
        room_type: str,
        character: Character,
    ) -> Iterator[dict[str, Any]]:
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
            finally:
                span.set_attributes({
                    "hp_after": character.player_hp,
                    "max_hp_after": character.player_max_hp,
                    "gold_after": character.gold,
                    "deck_size_after": len(character.deck),
                    **attrs,
                })


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
