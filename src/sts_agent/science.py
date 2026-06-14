"""Counterfactual ("for science") run support.

Replays an Act 1 run from the same master seed, overriding a single strategic
decision at a target floor.  Downstream RNG stays aligned because ``sts_env``
uses position-independent :class:`~sts_env.run.rng_streams.RunRNG` streams.

Usage::

    from sts_agent.science import DecisionOverride, run_for_science
    from sts_agent.battle.mcts import MCTSPlanner
    from sts_agent.strategy import SimStrategyAgent

    agent = SimStrategyAgent(max_encounters_to_sim=3, sim_nodes=1000, sim_sims=1000)
    override = DecisionOverride(
        floor=5,
        decision_type="pick_card",
        counterfactual="Inflame",
        baseline="Anger",
    )
    result = run_for_science(
        MCTSPlanner(),
        seed=42,
        override=override,
        strategy_agent=agent,
        parent_run_id="abc123",
    )

Path counterfactuals use the same ``run_seed`` as the parent run and::

    DecisionOverride(
        floor=5,
        decision_type="pick_branch",
        counterfactual=(3, 1),
        baseline=(3, 0),
    )

The child's JSONL is the authoritative counterfactual trajectory; compare run
metrics or aligned floors only where prefixes still match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlflow

from sts_env.run.character import Character
from sts_env.run.orchestrator import RunResult

from .battle.base import BattleAgent, BattlePlanner
from .run import run_act1
from .strategy.base import BaseStrategyAgent

log = logging.getLogger(__name__)

SCIENCE_TAG = "for_science"


@dataclass(frozen=True)
class DecisionOverride:
    """Single decision counterfactual applied at a target floor."""

    floor: int
    decision_type: str
    counterfactual: Any
    baseline: Any | None = None


@dataclass
class RunCheckpoint:
    """Snapshot of run state at a decision point (for inspection / future resume)."""

    seed: int
    floor: int
    map_x: int | None
    character: dict[str, Any]
    path: list[tuple[int, int]]

    @classmethod
    def from_character(
        cls,
        *,
        seed: int,
        character: Character,
        path: list[tuple[int, int]],
    ) -> RunCheckpoint:
        return cls(
            seed=seed,
            floor=character.floor,
            map_x=character.map_x,
            character=character.snapshot(),
            path=list(path),
        )


class _OverrideStrategyAgent(BaseStrategyAgent):
    """Wraps a strategy agent, substituting one decision at ``override.floor``."""

    def __init__(
        self,
        inner: BaseStrategyAgent,
        override: DecisionOverride,
    ) -> None:
        super().__init__(seed=getattr(inner.rng, "seed", 0))
        self._inner = inner
        self._override = override

    def _matches(self, decision_type: str, floor: int) -> bool:
        return (
            decision_type == self._override.decision_type
            and floor == self._override.floor
        )

    def set_encounter_tracking(self, encounter_queue, hallway_seen, elites_seen):
        return self._inner.set_encounter_tracking(
            encounter_queue, hallway_seen, elites_seen
        )

    def begin_map_run(self, sts_map, seed: int) -> None:
        return self._inner.begin_map_run(sts_map, seed)

    def on_map_step(self, coord: tuple[int, int]) -> None:
        return self._inner.on_map_step(coord)

    def get_possible_encounters(self):
        return self._inner.get_possible_encounters()

    def pick_neow(self, options):
        return self._inner.pick_neow(options)

    def pick_branch(self, sts_map, character, current, seed):
        if self._matches("pick_branch", character.floor):
            log.info(
                "Science override pick_branch floor=%d counterfactual=%s",
                character.floor,
                self._override.counterfactual,
            )
            return self._override.counterfactual
        return self._inner.pick_branch(sts_map, character, current, seed)

    def pick_card(
        self,
        character,
        card_choices,
        upcoming_encounters,
        seed,
        **kwargs,
    ):
        if self._matches("pick_card", character.floor):
            log.info(
                "Science override pick_card floor=%d counterfactual=%s",
                character.floor,
                self._override.counterfactual,
            )
            return self._override.counterfactual
        return self._inner.pick_card(
            character, card_choices, upcoming_encounters, seed, **kwargs
        )

    def pick_rest_choice(self, character, **kwargs):
        if self._matches("pick_rest_choice", character.floor):
            return self._override.counterfactual
        return self._inner.pick_rest_choice(character, **kwargs)

    def pick_event_choice(self, event, character, **kwargs):
        if self._matches("pick_event_choice", character.floor):
            return self._override.counterfactual
        return self._inner.pick_event_choice(event, character, **kwargs)

    def pick_card_to_remove(self, character, **kwargs):
        return self._inner.pick_card_to_remove(character, **kwargs)

    def pick_card_to_transform(self, character, **kwargs):
        return self._inner.pick_card_to_transform(character, **kwargs)

    def pick_card_to_upgrade(self, character, **kwargs):
        return self._inner.pick_card_to_upgrade(character, **kwargs)

    def shop(self, inventory, character):
        return self._inner.shop(inventory, character)

    def pick_boss_relic(self, character, choices):
        if self._matches("pick_boss_relic", character.floor):
            return self._override.counterfactual
        return self._inner.pick_boss_relic(character, choices)

    def pick_potion_to_discard(self, character, new_potion, **kwargs):
        return self._inner.pick_potion_to_discard(character, new_potion, **kwargs)


def _science_tags(override: DecisionOverride, parent_run_id: str | None) -> dict[str, str]:
    tags = {
        SCIENCE_TAG: "true",
        "fork_floor": str(override.floor),
        "decision_type": override.decision_type,
        "counterfactual_choice": str(override.counterfactual),
    }
    if parent_run_id:
        tags["parent_run_id"] = parent_run_id
    if override.baseline is not None:
        tags["baseline_choice"] = str(override.baseline)
    return tags


def run_for_science(
    planner_or_agent: BattlePlanner | BattleAgent,
    seed: int,
    override: DecisionOverride,
    *,
    strategy_agent: BaseStrategyAgent | None = None,
    parent_run_id: str | None = None,
    use_map: bool = True,
    run_name: str | None = None,
) -> RunResult:
    """Run Act 1 with a single counterfactual decision override.

    Parameters
    ----------
    planner_or_agent:
        Battle pilot for combat floors.
    seed:
        Master run seed (same as the parent run being compared).
    override:
        Which floor/decision to override and the counterfactual choice.
    strategy_agent:
        Base strategy agent to wrap.  Defaults to ``BaseStrategyAgent(seed=seed)``.
    parent_run_id:
        MLflow run id of the parent run for lineage tags.
    use_map:
        Whether to use the branching map (default True).
    run_name:
        Optional MLflow run name; defaults to a science-prefixed name.
    """
    base_agent = strategy_agent or BaseStrategyAgent(seed=seed)
    wrapped = _OverrideStrategyAgent(base_agent, override)

    tags = _science_tags(override, parent_run_id)
    default_name = (
        f"science seed={seed} floor={override.floor} "
        f"{override.decision_type}={override.counterfactual}"
    )

    with mlflow.start_run(run_name=run_name or default_name):
        mlflow.set_tags(tags)
        result = run_act1(
            planner_or_agent,
            seed,
            strategy_agent=wrapped,
            use_map=use_map,
        )
        mlflow.log_metrics(
            {
                "victory": float(result.victory),
                "floors_cleared": float(result.floors_cleared),
                "total_damage": float(result.damage_taken_total),
            }
        )
        return result