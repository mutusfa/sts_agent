"""Combat simulation helpers for the strategy agent.

Public API
----------
simulate_encounter    — run one encounter with MCTS and return structured results.
simulate_with_card    — convenience wrapper that adds a hypothetical card first.
probe_encounter       — fast single-act MCTS probe returning distribution stats.
probe_with_card       — same but with a hypothetical card added.

The ``probe_*`` functions are an optimisation over ``simulate_*``: they only
call ``MCTSPlanner.act()`` once (which internally runs ``simulations``
rollouts) and return the distribution of outcomes from the root edge rather
than replaying the full combat to completion.  This is ~5-10× faster and
provides *better* information (a distribution over N rollouts vs a single
deterministic play-through).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

from sts_env.run.builder import build_combat
from sts_env.run.character import Character

from ..battle.base import run_planner
from ..battle.mcts import MCTSPlanner

log = logging.getLogger(__name__)


@dataclass
class SimResult:
    """Structured outcome of a simulated encounter.

    Attributes
    ----------
    survived:
        True if the player is alive when combat ends.
    damage_taken:
        HP lost during combat.
    max_hp_gained:
        Maximum HP gained during combat (e.g. from Feed).
    final_hp:
        Player HP at the end of combat.
    final_max_hp:
        Player maximum HP at the end of combat.
    turns:
        Number of turns the combat lasted.
    """

    survived: bool
    damage_taken: int
    max_hp_gained: int
    final_hp: int
    final_max_hp: int
    turns: int


@dataclass
class SimDistribution:
    """Statistical summary of MCTS rollout outcomes from a single act() call.

    This is what the MCTS planner *already knows* after its first decision:
    across ``n`` simulated play-throughs of the full combat, here are the
    distribution of terminal scores.

    Attributes
    ----------
    mean_score:
        Mean terminal score across rollouts (damage, doubled on death).
    std_score:
        Standard deviation of terminal scores.
    max_score:
        Worst-case (maximum) terminal score seen.
    n:
        Number of rollouts sampled.
    deaths:
        Number of rollouts where the player died.
    start_hp:
        Player max HP at the start of combat (used to interpret scores).
    """

    mean_score: float
    std_score: float
    max_score: float
    n: int
    deaths: int
    start_hp: int

    @property
    def survival_rate(self) -> float:
        """Fraction of rollouts where the player survived."""
        if self.n == 0:
            return 0.0
        return 1.0 - (self.deaths / self.n)

    @property
    def death_rate(self) -> float:
        """Fraction of rollouts where the player died."""
        if self.n == 0:
            return 1.0
        return self.deaths / self.n

    @property
    def expected_damage(self) -> float:
        """Expected damage taken.

        For surviving rollouts, score == damage_taken.
        For dying rollouts, score == damage_taken * 2.
        We estimate expected damage as min(mean_score, start_hp) which is
        reasonable when most rollouts survive.  For high death rates the
        interpretation is less precise, but the ordering is still correct
        for comparison purposes.
        """
        return min(self.mean_score, float(self.start_hp))

    @property
    def damage_spread(self) -> str:
        """Human-readable spread, e.g. '40±20 (12% death)'."""
        pct = f"{self.death_rate:.0%}" if self.n > 0 else "?"
        return f"{self.expected_damage:.0f}±{self.std_score:.0f} ({pct} death)"


def simulate_encounter(
    character: Character,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    *,
    max_nodes: int = 10_000,
    simulations: int = 10_000,
) -> SimResult:
    """Simulate a combat encounter and return the outcome.

    Parameters
    ----------
    character:
        A :class:`Character` whose deck / HP / potions define the player
        state.  A deep-copy is made so the original is never mutated.
    encounter_type:
        ``"easy"``, ``"hard"``, ``"elite"``, or ``"boss"``.
    encounter_id:
        Encounter identifier string (e.g. ``"cultist"``, ``"Lagavulin"``).
    seed:
        Combat seed for deterministic results.
    max_nodes:
        MCTS node-expansion budget per action.
    simulations:
        MCTS simulation budget per action.

    Returns
    -------
    SimResult with survival, damage, HP, and turn information.
    """
    char_copy = copy.deepcopy(character)
    combat = build_combat(
        encounter_type, encounter_id, seed, character=char_copy
    )
    planner = MCTSPlanner(simulations=simulations, max_nodes=max_nodes)
    damage_taken = run_planner(planner, combat)

    obs = combat.observe()
    return SimResult(
        survived=not obs.player_dead,
        damage_taken=damage_taken,
        max_hp_gained=combat.max_hp_gained,
        final_hp=obs.player_hp,
        final_max_hp=obs.player_max_hp,
        turns=obs.turn,
    )


def simulate_with_card(
    character: Character,
    card_id: str,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    **kwargs: object,
) -> SimResult:
    """Simulate an encounter with a hypothetical card added to the deck.

    The original *character* is not mutated.
    """
    char_copy = copy.deepcopy(character)
    char_copy.add_card(card_id)
    return simulate_encounter(
        char_copy, encounter_type, encounter_id, seed, **kwargs
    )


# ---------------------------------------------------------------------------
# Fast probe API — single act() call, returns distribution
# ---------------------------------------------------------------------------


def probe_encounter(
    character: Character,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    *,
    max_nodes: int = 10_000,
    simulations: int = 10_000,
) -> SimDistribution:
    """Probe an encounter with a single MCTS act() call.

    Instead of playing the full combat to completion, this runs one round of
    MCTS (which internally simulates the full combat ``simulations`` times)
    and returns the distribution of rollout outcomes from the root edge.

    This is ~5-10× faster than ``simulate_encounter`` and provides richer
    information (a distribution over N rollouts vs a single play-through).

    Parameters
    ----------
    character:
        A :class:`Character`.  Deep-copied internally.
    encounter_type, encounter_id, seed:
        Encounter specification.
    max_nodes:
        MCTS node budget.
    simulations:
        MCTS simulation budget.

    Returns
    -------
    SimDistribution with mean/std/max of terminal scores.
    """
    char_copy = copy.deepcopy(character)
    combat = build_combat(
        encounter_type, encounter_id, seed, character=char_copy
    )
    planner = MCTSPlanner(simulations=simulations, max_nodes=max_nodes)
    combat.reset()

    # Single act() — MCTS does `simulations` full-battle rollouts internally
    planner.act(combat)

    stats = planner.last_stats
    obs = combat.observe()

    return SimDistribution(
        mean_score=stats.get("mean", float("nan")),
        std_score=stats.get("std", 0.0),
        max_score=stats.get("max", float("nan")),
        n=int(stats.get("n", 0)),
        deaths=int(stats.get("deaths", 0)),
        start_hp=obs.player_max_hp,
    )


def probe_with_card(
    character: Character,
    card_id: str,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    **kwargs: object,
) -> SimDistribution:
    """Probe an encounter with a hypothetical card added to the deck.

    The original *character* is not mutated.
    """
    char_copy = copy.deepcopy(character)
    char_copy.add_card(card_id)
    return probe_encounter(
        char_copy, encounter_type, encounter_id, seed, **kwargs
    )
