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
from dataclasses import dataclass, field

from sts_env.run.character import Character
from sts_env.run.rewards import IRONCLAD_RARE_CARDS, IRONCLAD_UNCOMMON_CARDS
from .simulate import SimDistribution, probe_encounter, probe_with_card

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


_RARE_SET = set(IRONCLAD_RARE_CARDS)
_UNCOMMON_SET = set(IRONCLAD_UNCOMMON_CARDS)


def _rarity_rank(card_id: str | None) -> int:
    """2=rare, 1=uncommon, 0=common, -1=skip."""
    if card_id is None:
        return -1
    if card_id in _RARE_SET:
        return 2
    if card_id in _UNCOMMON_SET:
        return 1
    return 0


class SimStrategyAgent:
    """Simulation-driven card-pick agent using fast MCTS probes.

    For each card reward, evaluates every option (including skip) by probing
    all remaining encounters with and without each card.  Picks the best.

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
        Ignored (kept for API compat with StrategyAgent).
    """

    def __init__(
        self,
        max_encounters_to_sim: int = 5,
        sim_nodes: int = 5_000,
        sim_sims: int = 5_000,
        timeout_seconds: int = 300,
    ) -> None:
        self.max_encounters = max_encounters_to_sim
        self.sim_nodes = sim_nodes
        self.sim_sims = sim_sims

    def pick_card(
        self,
        character: Character,
        card_choices: list[str],
        upcoming_encounters: list[tuple[str, str]],
        seed: int,
    ) -> str | None:
        """Evaluate all card options and pick the best one."""
        if not card_choices:
            return None

        # Limit encounters to simulate
        to_sim = upcoming_encounters[:self.max_encounters]
        if not to_sim:
            # Last floor — no upcoming encounters, just pick rarest
            return self._pick_by_rarity(card_choices)

        # Evaluate each option: skip + each card
        options: list[CardEvaluation] = []
        candidates = [None] + list(card_choices)  # None = skip (baseline)

        for card_id in candidates:
            evals = self._evaluate_option(character, card_id, to_sim, seed)
            options.append(evals)
            label = card_id or "SKIP"
            # Log per-encounter spreads
            spreads = " | ".join(
                f"[{enc_type[:4]}] {d.damage_spread}"
                for (enc_type, _), d in zip(to_sim, evals.per_encounter)
            )
            log.info(
                "  %-18s  %s",
                label, spreads,
            )

        # Pick the best
        best = max(options, key=lambda e: e.score)
        log.info(
            "  → Best: %s (death=%s)",
            best.card_id or "SKIP", best.death_pct,
        )
        return best.card_id

    def _evaluate_option(
        self,
        character: Character,
        card_id: str | None,
        encounters: list[tuple[str, str]],
        seed: int,
    ) -> CardEvaluation:
        """Probe all encounters with (or without) a card.

        Uses the fast probe API — single MCTS act() per encounter per option,
        extracting the rollout distribution from root edge statistics.
        """
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
                )
            else:
                dist = probe_with_card(
                    character, card_id, enc_type, enc_id, enc_seed,
                    max_nodes=self.sim_nodes, simulations=self.sim_sims,
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
