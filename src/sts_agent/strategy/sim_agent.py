"""Simulation-driven strategy agent — no external LLM needed.

Uses MCTS first-action probes to evaluate each card option against upcoming
encounters and picks the one that maximises survival + minimises damage.

Optimisation
------------
Instead of running full combats for evaluation (which replays MCTS for every
action in every encounter for every card option), this agent calls
``probe_encounter`` / ``probe_with_card`` which only invoke MCTS.act() *once*
per evaluation.  The single act() call already runs ``simulations`` full-battle
rollouts internally — we just extract the root-edge distribution rather than
replaying the deterministic combat to completion.

This is ~5-10× faster than the naive approach and provides richer information
(a distribution of outcomes rather than a single sample path).

Strategy hierarchy
------------------
1. Survive — any card that improves survival rate wins
2. Minimize expected damage across upcoming encounters
3. Tiebreak: lowest worst-case (max_score), then card rarity
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal

import mlflow

from sts_env.run.character import Character
from sts_env.combat.card_pools import pool
from sts_env.combat.cards import CardColor, Rarity

from sts_agent.tracing import log_probe_artifact, set_span_attributes

from .base import BaseStrategyAgent, _normalize_map_edge
from .probe_data import (
    ProbeCache,
    ProbeCollector,
    ProbeContext,
    ProbeDecisionRecord,
    card_evaluation_to_dict,
    character_state_snapshot,
    decision_budget,
    probe_context,
)
from .shop_eval import (
    encounters_for_shop_probes,
    evaluate_shop_baseline,
    evaluate_shop_option,
    execute_shop_action,
    format_shop_inventory,
    list_shop_candidates,
    shop_score_to_dict,
)
from .simulate import SimDistribution, probe_encounter, probe_with_card

if TYPE_CHECKING:
    from sts_env.run.encounter_queue import EncounterQueue
    from sts_env.run.map import MapNode, StSMap
    from sts_env.run.shop import ShopInventory

log = logging.getLogger(__name__)


@dataclass
class CardEvaluation:
    """Score for one card option across all upcoming encounters."""
    card_id: str | None  # None = skip
    total_expected_damage: float
    total_survival_rate: float
    worst_max_score: float  # worst-case across encounters (lower = better)
    total_encounters: int
    total_deaths: int
    total_rollouts: int
    per_encounter: list[SimDistribution] = field(default_factory=list)

    @property
    def score(self) -> tuple[bool, float, float, int]:
        """Comparison key: (high_survival, low_damage, low_worst_case, rarity).

        Higher is better.
        """
        rarity = _rarity_rank(self.card_id)
        return (
            self.total_survival_rate >= 1.0,       # survived all on average
            -self.total_expected_damage,            # less damage = better
            -self.worst_max_score,                  # lower worst-case = better
            rarity,
        )

    @property
    def death_pct(self) -> str:
        """Human-readable death rate across all encounters."""
        if self.total_rollouts == 0:
            return "?%"
        return f"{self.total_deaths / self.total_rollouts:.0%}"


_RARE_SET = set(pool(CardColor.RED, Rarity.RARE))
_UNCOMMON_SET = set(pool(CardColor.RED, Rarity.UNCOMMON))
_PROBEABLE_ENCOUNTER_TYPES = frozenset(
    {"easy", "hard", "elite", "boss", "monster"}
)


def _combat_encounters_only(
    encounters: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Keep only room types that :func:`probe_encounter` can simulate."""
    return [
        (enc_type, enc_id)
        for enc_type, enc_id in encounters
        if enc_type in _PROBEABLE_ENCOUNTER_TYPES
    ]


def _rarity_rank(card_id: str | None) -> int:
    """2=rare, 1=uncommon, 0=common, -1=skip."""
    if card_id is None:
        return -1
    if card_id in _RARE_SET:
        return 2
    if card_id in _UNCOMMON_SET:
        return 1
    return 0


class SimStrategyAgent(BaseStrategyAgent):
    """Simulation-driven card-pick + route-planning agent using fast MCTS probes.

    Overrides :meth:`pick_card`; inherits probe-based map routing and random
    defaults from :class:`BaseStrategyAgent` for everything else.

    Parameters
    ----------
    max_encounters_to_sim:
        Only probe the first N upcoming encounters (default 5).
        Later encounters are less relevant to the current pick.
    sim_nodes:
        MCTS node budget per probe (default 5000).
    sim_sims:
        MCTS simulation budget per probe (default 5000).
    timeout_seconds:
        Wall-clock budget per strategy decision — pick_card, shop, pick_branch,
        and potion evaluation (default 600 s).  On timeout the agent falls back
        to a cheap heuristic instead of probing further.
    seed:
        Forwarded to :class:`BaseStrategyAgent` for the inherited random
        decisions (Neow, rest, events, boss relic).
    shop_max_remove_candidates:
        When set, cap how many ``remove:`` actions are probe-evaluated per
        shop visit (default ``None`` = all unique deck cards).
    """

    def __init__(
        self,
        max_encounters_to_sim: int = 5,
        sim_nodes: int = 5_000,
        sim_sims: int = 5_000,
        timeout_seconds: int = 600,
        seed: int | None = None,
        *,
        shop_max_remove_candidates: int | None = None,
        probe_collector: ProbeCollector | None = None,
        probe_jsonl: Path | str | None = None,
        rollout_mode: Literal["heuristic", "in_tree"] = "heuristic",
    ) -> None:
        super().__init__(seed=seed)
        self.max_encounters = max_encounters_to_sim
        self.sim_nodes = sim_nodes
        self.sim_sims = sim_sims
        self.timeout_seconds = timeout_seconds
        self.rollout_mode = rollout_mode
        self.shop_max_remove_candidates = shop_max_remove_candidates
        self.probe_cache = ProbeCache()
        if probe_collector is not None:
            self.probe_collector = probe_collector
        elif probe_jsonl is not None:
            self.probe_collector = ProbeCollector(probe_jsonl)
        else:
            self.probe_collector = ProbeCollector()

    @property
    def probe_stats(self) -> dict[str, object]:
        """Summary of MCTS probe calls vs cache hits for this run."""
        return self.probe_cache.stats.summary()

    def begin_map_run(self, sts_map, seed: int) -> None:
        super().begin_map_run(sts_map, seed)
        self.probe_collector.run_seed = seed
        self.probe_cache.clear()

    @contextmanager
    def decision_budget(self, decision_type: str) -> Iterator[None]:
        """Wall-clock budget for one strategy decision (see :func:`decision_budget`)."""
        with decision_budget(self.timeout_seconds):
            yield

    def _pick_branch_fallback(
        self,
        sts_map: "StSMap",
        current: tuple[int, int] | None,
    ) -> tuple[int, int]:
        """Random valid fork when probe budget is exhausted."""
        if current is None:
            options = [
                (0, n.x) for n in sts_map.nodes.get(0, []) if n.edges
            ]
            if not options:
                return (0, 0)
        else:
            floor, x_pos = current
            node = sts_map.get_node(floor, x_pos)
            if node is None or not node.edges:
                return current
            options = [_normalize_map_edge(floor, edge) for edge in node.edges]
        if len(options) == 1:
            return options[0]
        return self.rng.choice(options)

    # ------------------------------------------------------------------
    # Card-pick (specialised)
    # ------------------------------------------------------------------

    def pick_card(
        self,
        character: Character,
        card_choices: list[str],
        upcoming_encounters: list[tuple[str, str]],
        seed: int,
        *,
        sts_map: "StSMap | None" = None,
        current_position: tuple[int, int] | None = None,
    ) -> str | None:
        """Evaluate all card options and pick the best one."""
        map_view = self._build_map_view(character, sts_map, current_position)
        return self._pick_card_traced(
            character,
            card_choices,
            upcoming_encounters,
            seed,
            map_view,
        )

    @mlflow.trace(name="pick_card")
    def _pick_card_traced(
        self,
        character: Character,
        card_choices: list[str],
        upcoming_encounters: list[tuple[str, str]],
        seed: int,
        map_view: str,
    ) -> str | None:
        """Traced pick_card body with persisted probe evaluations."""
        try:
            with self.decision_budget("pick_card"):
                return self._pick_card_probed(
                    character,
                    card_choices,
                    upcoming_encounters,
                    seed,
                    map_view,
                )
        except TimeoutError:
            pick = self._pick_by_rarity(card_choices)
            log.warning(
                "pick_card timed out after %ss on floor %d — rarity fallback %s",
                self.timeout_seconds,
                character.floor,
                pick or "skip",
            )
            set_span_attributes(
                {
                    "pick": pick or "skip",
                    "fallback": True,
                    "timeout": True,
                    "evaluation_count": 0,
                }
            )
            return pick

    def _pick_card_probed(
        self,
        character: Character,
        card_choices: list[str],
        upcoming_encounters: list[tuple[str, str]],
        seed: int,
        map_view: str,
    ) -> str | None:
        """Traced pick_card body with persisted probe evaluations."""
        if not card_choices:
            set_span_attributes({"pick": "skip", "card_choices": ""})
            return None

        to_sim = _combat_encounters_only(upcoming_encounters[: self.max_encounters])
        if not to_sim:
            pick = self._pick_by_rarity(card_choices)
            set_span_attributes(
                {
                    "character_state": character.summary(),
                    "card_choices": ", ".join(card_choices),
                    "seed": seed,
                    "map_view": map_view,
                    "pick": pick or "skip",
                    "fallback": True,
                    "evaluation_count": 0,
                }
            )
            return pick

        set_span_attributes(
            {
                "character_state": character.summary(),
                "card_choices": ", ".join(card_choices),
                "seed": seed,
                "map_view": map_view,
                "upcoming_encounters": ", ".join(
                    f"{t}/{i or '*'}" for t, i in to_sim
                ),
            }
        )

        run_seed = self._run_seed if self._run_seed is not None else seed

        options: list[CardEvaluation] = []
        candidates = [None] + list(card_choices)  # None = skip (baseline)

        for card_id in candidates:
            option_ctx = ProbeContext(
                collector=self.probe_collector,
                decision_type="pick_card",
                floor=character.floor,
                run_seed=run_seed,
                option=card_id or "SKIP",
            )
            with probe_context(option_ctx):
                evals = self._evaluate_option(character, card_id, to_sim, seed)
            options.append(evals)
            label = card_id or "SKIP"
            spreads = " | ".join(
                f"[{enc_type[:4]}] {d.damage_spread}"
                for (enc_type, _), d in zip(to_sim, evals.per_encounter)
            )
            log.info(
                "  %-18s  %s",
                label,
                spreads,
            )

        best = max(options, key=lambda e: e.score)
        pick = best.card_id
        log.info(
            "  → Best: %s (death=%s)",
            best.card_id or "SKIP",
            best.death_pct,
        )

        serialized = [
            card_evaluation_to_dict(e, encounters=to_sim) for e in options
        ]
        record = ProbeDecisionRecord(
            decision_type="pick_card",
            floor=character.floor,
            seed=seed,
            run_seed=run_seed,
            character_state=character_state_snapshot(character),
            pick=pick,
            card_choices=list(card_choices),
            upcoming_encounters=list(to_sim),
            evaluations=serialized,
            map_view=map_view,
        )
        self.probe_collector.record(record)

        artifact_name = f"floor_{character.floor}_pick_card"
        log_probe_artifact(artifact_name, record.to_dict())

        set_span_attributes(
            {
                "pick": pick or "skip",
                "fallback": False,
                "evaluation_count": len(options),
                "best_card": pick or "skip",
                "best_death_pct": best.death_pct,
            }
        )
        return pick

    def _build_map_view(
        self,
        character: Character,
        sts_map: "StSMap | None",
        current_position: tuple[int, int] | None,
    ) -> str:
        if sts_map is None or current_position is None:
            return character.summary()
        from .map_routing import build_scored_map_view

        return build_scored_map_view(
            sts_map,
            character,
            current_position,
            committed_path=self._committed_path(),
        )

    def _evaluate_option(
        self,
        character: Character,
        card_id: str | None,
        encounters: list[tuple[str, str]],
        seed: int,
    ) -> CardEvaluation:
        """Probe all encounters with (or without) a card."""
        total_expected_damage = 0.0
        total_survival_rate = 0.0
        worst_max_score = 0.0
        total_deaths = 0
        total_rollouts = 0
        per_encounter: list[SimDistribution] = []

        for idx, (enc_type, enc_id) in enumerate(encounters):
            enc_seed = seed * 1000 + idx
            if card_id is None:
                dist = probe_encounter(
                    character, enc_type, enc_id, enc_seed,
                    max_nodes=self.sim_nodes, simulations=self.sim_sims,
                    probe_cache=self.probe_cache,
                    rollout_mode=self.rollout_mode,
                )
            else:
                dist = probe_with_card(
                    character, card_id, enc_type, enc_id, enc_seed,
                    max_nodes=self.sim_nodes, simulations=self.sim_sims,
                    probe_cache=self.probe_cache,
                    rollout_mode=self.rollout_mode,
                )

            total_expected_damage += dist.expected_damage
            total_survival_rate += dist.survival_rate
            total_deaths += dist.deaths
            total_rollouts += dist.n
            if dist.max_score > worst_max_score:
                worst_max_score = dist.max_score
            per_encounter.append(dist)

        avg_survival = total_survival_rate / len(encounters) if encounters else 1.0

        return CardEvaluation(
            card_id=card_id,
            total_expected_damage=total_expected_damage,
            total_survival_rate=avg_survival,
            worst_max_score=worst_max_score,
            total_encounters=len(encounters),
            total_deaths=total_deaths,
            total_rollouts=total_rollouts,
            per_encounter=per_encounter,
        )

    @staticmethod
    def _pick_by_rarity(choices: list[str]) -> str | None:
        """Fallback: pick rarest card."""
        for c in choices:
            if c in _RARE_SET:
                return c
        for c in choices:
            if c in _UNCOMMON_SET:
                return c
        return choices[0] if choices else None

    # ------------------------------------------------------------------
    # Shop (specialised)
    # ------------------------------------------------------------------

    def shop(self, inventory: "ShopInventory", character: Character) -> None:
        """Probe-based greedy shop visit."""
        self._shop_traced(inventory, character)

    @mlflow.trace(name="shop")
    def _shop_traced(self, inventory: "ShopInventory", character: Character) -> None:
        try:
            with self.decision_budget("shop"):
                self._shop_probed(inventory, character)
        except TimeoutError:
            log.warning(
                "shop timed out after %ss on floor %d — leaving",
                self.timeout_seconds,
                character.floor,
            )
            set_span_attributes(
                {"fallback": True, "timeout": True, "shop_actions": "leave"}
            )

    def _shop_probed(self, inventory: "ShopInventory", character: Character) -> None:
        possible = self.get_possible_encounters()
        if possible is None:
            super().shop(inventory, character)
            set_span_attributes({"fallback": True, "shop_actions": "random"})
            return

        encounters = encounters_for_shop_probes(possible)
        seed = self._probe_seed(character)
        actions_taken: list[str] = []
        shop_iterations: list[dict[str, object]] = []
        run_seed = self._run_seed if self._run_seed is not None else seed
        entry_snapshot = character_state_snapshot(character)

        set_span_attributes(
            {
                "character_state": character.summary(),
                "floor": character.floor,
                "seed": seed,
                "fallback": False,
            }
        )

        for iteration in range(10):
            candidates = list_shop_candidates(
                inventory,
                character,
                max_remove_candidates=self.shop_max_remove_candidates,
            )
            if len(candidates) <= 1:
                break

            iteration_evals: list[dict[str, object]] = []
            baseline_dists: list[SimDistribution] = []
            shop_ctx = ProbeContext(
                collector=self.probe_collector,
                decision_type="shop",
                floor=character.floor,
                run_seed=run_seed,
                option="leave",
            )
            with probe_context(shop_ctx):
                baseline = evaluate_shop_baseline(
                    character,
                    encounters,
                    seed + iteration,
                    max_nodes=self.sim_nodes,
                    simulations=self.sim_sims,
                    out_dists=baseline_dists,
                    probe_cache=self.probe_cache,
                    rollout_mode=self.rollout_mode,
                )
            iteration_evals.append(
                shop_score_to_dict(
                    baseline,
                    encounters=encounters,
                    per_encounter=baseline_dists,
                )
            )
            best_action = "leave"
            best_score = baseline.score

            for action in candidates:
                if action == "leave":
                    continue
                action_dists: list[SimDistribution] = []
                action_ctx = ProbeContext(
                    collector=self.probe_collector,
                    decision_type="shop",
                    floor=character.floor,
                    run_seed=run_seed,
                    option=action,
                )
                with probe_context(action_ctx):
                    option = evaluate_shop_option(
                        action,
                        character,
                        inventory,
                        encounters,
                        seed + iteration,
                        max_nodes=self.sim_nodes,
                        simulations=self.sim_sims,
                        possible_encounters=possible,
                        out_dists=action_dists,
                        probe_cache=self.probe_cache,
                        baseline=baseline,
                        rollout_mode=self.rollout_mode,
                    )
                iteration_evals.append(
                    shop_score_to_dict(
                        option,
                        encounters=encounters,
                        per_encounter=action_dists,
                    )
                )
                if option.score > best_score:
                    best_score = option.score
                    best_action = action

            shop_iterations.append(
                {
                    "iteration": iteration,
                    "candidates": candidates,
                    "evaluations": iteration_evals,
                    "pick": best_action,
                }
            )

            if best_action == "leave":
                break

            log.info("  Shop: %s", best_action)
            execute_shop_action(best_action, inventory, character)
            actions_taken.append(best_action)

        self.probe_collector.record(
            ProbeDecisionRecord(
                decision_type="shop",
                floor=character.floor,
                seed=seed,
                run_seed=run_seed,
                character_state=entry_snapshot,
                pick=", ".join(actions_taken) or "leave",
                extra={
                    "shop_inventory": format_shop_inventory(inventory),
                    "iterations": shop_iterations,
                    "actions_taken": actions_taken,
                },
            )
        )

        set_span_attributes({"shop_actions": ", ".join(actions_taken) or "leave"})

    # ------------------------------------------------------------------
    # Map route (specialised)
    # ------------------------------------------------------------------

    def pick_branch(
        self,
        sts_map: "StSMap",
        character: Character,
        current: tuple[int, int] | None,
        seed: int,
    ) -> tuple[int, int]:
        """Pick the next map step using run-scoped probe cache."""
        try:
            with self.decision_budget("pick_branch"):
                return self._pick_branch_probed(sts_map, character, current, seed)
        except TimeoutError:
            coord = self._pick_branch_fallback(sts_map, current)
            log.warning(
                "pick_branch timed out after %ss on floor %d — random fallback %s",
                self.timeout_seconds,
                character.floor,
                coord,
            )
            return coord

    def _pick_branch_probed(
        self,
        sts_map: "StSMap",
        character: Character,
        current: tuple[int, int] | None,
        seed: int,
    ) -> tuple[int, int]:
        """Pick the next map step using run-scoped probe cache."""
        from .map_routing import count_shops_visited, pick_fork_coord

        shops = (
            0
            if current is None
            else count_shops_visited(sts_map, self._committed_path())
        )
        return pick_fork_coord(
            sts_map,
            character,
            current,
            seed,
            self.rng,
            self._encounter_queue,
            shops_visited=shops,
            probe_cache=self.probe_cache,
            config=self._branch_probe_config(),
        )

    def _branch_probe_config(self):
        from .map_routing import ProbeConfig
        return ProbeConfig(
            max_nodes=self.sim_nodes,
            simulations=self.sim_sims,
            rollout_mode=self.rollout_mode,
        )
