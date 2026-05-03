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
import random
from dataclasses import dataclass

from sts_env.run.builder import build_combat
from sts_env.run.character import Character

from ..battle.base import run_planner
from ..battle.mcts import MCTSPlanner

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encounter pool resolution
# ---------------------------------------------------------------------------

# Each entry is (enc_type_for_build_combat, enc_id).
# "monster" covers the full Act 1 hallway pool (easy + hard) for use when
# only the room type is known.  Future: sts_env could restrict the pool based
# on how many easy/hard fights have already been seen.
_ACT1_POOLS: dict[str, list[tuple[str, str]]] = {
    "easy": [
        ("easy", "cultist"),
        ("easy", "jaw_worm"),
        ("easy", "two_louses"),
        ("easy", "small_slimes"),
    ],
    "hard": [
        ("hard", "gremlin_gang"),
        ("hard", "red_slaver"),
        ("hard", "blue_slaver"),
        ("hard", "looter"),
        ("hard", "exordium_thugs"),
        ("hard", "exordium_wildlife"),
        ("hard", "large_slime"),
        ("hard", "three_louse"),
        ("hard", "two_fungi_beasts"),
    ],
    "elite": [
        ("elite", "Gremlin Nob"),
        ("elite", "Lagavulin"),
        ("elite", "Three Sentries"),
    ],
    "boss": [
        ("boss", "slime_boss"),
        ("boss", "guardian"),
        ("boss", "hexaghost"),
    ],
}
# "monster" = unknown hallway difficulty → combined easy + hard pool
_ACT1_POOLS["monster"] = _ACT1_POOLS["easy"] + _ACT1_POOLS["hard"]


def _resolve_enc(enc_type: str, enc_id: str, seed: int) -> tuple[str, str]:
    """Resolve a ``(enc_type, enc_id)`` pair, sampling from the pool when enc_id is empty.

    Parameters
    ----------
    enc_type:
        Encounter type as received (``"easy"``, ``"hard"``, ``"elite"``,
        ``"boss"``, or ``"monster"`` for an unspecified hallway room).
    enc_id:
        Specific encounter identifier.  When empty the pool for *enc_type*
        is sampled deterministically using *seed*.
    seed:
        Used to seed the RNG so repeated calls with the same arguments
        always return the same encounter.

    Returns
    -------
    ``(resolved_enc_type, resolved_enc_id)`` safe to pass to ``build_combat``.

    Raises
    ------
    ValueError
        If *enc_type* is not recognised and *enc_id* is also empty.
    """
    if enc_id:
        return enc_type, enc_id

    pool = _ACT1_POOLS.get(enc_type)
    if pool is None:
        raise ValueError(
            f"Unknown encounter type {enc_type!r} and no enc_id provided. "
            f"Known types: {list(_ACT1_POOLS)}"
        )
    return random.Random(seed).choice(pool)


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
    enemy_hp_remaining:
        Total enemy HP remaining at end of combat.  0 when the player won;
        positive when the player died, indicating how close the fight was.
    """

    survived: bool
    damage_taken: int
    max_hp_gained: int
    final_hp: int
    final_max_hp: int
    turns: int
    enemy_hp_remaining: int = 0


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


# ---------------------------------------------------------------------------
# Simulation as player skill
# ---------------------------------------------------------------------------
# These functions are a stand-in for player skill / game knowledge.
#
# An experienced StS player can look at a deck and estimate outcomes:
#   - "Against Gremlin Nob this deck takes ~20 damage, probably dies to
#     Lagavulin, Sentries are free"
#   - "I want AoE for Sentries, Lagavulin lets me set up, for Gremlin Nob
#     I need ways to mitigate damage that aren't skill-based"
#
# Our MCTS simulations give us the same thing in numbers — slightly more
# exact than a human could estimate, but the same concept: evaluate deck
# strength against specific upcoming encounters.  The `enemy_hp_remaining`
# field on SimResult tells the LLM *how close* a loss was, turning a
# binary "DIED" into a graded signal like "died but left the boss at 5 HP".
# ---------------------------------------------------------------------------


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
        ``"easy"``, ``"hard"``, ``"elite"``, ``"boss"``, ``"event"``, or ``"monster"``
        (unknown hallway difficulty — sampled from the combined easy+hard pool).
        ``"event"`` uses event-specific encounter variants (e.g. awake Lagavulin).
    encounter_id:
        Encounter identifier string (e.g. ``"cultist"``, ``"Lagavulin"``).
        Pass an empty string to sample a representative encounter from the pool.
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
    encounter_type, encounter_id = _resolve_enc(encounter_type, encounter_id, seed)
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
        enemy_hp_remaining=sum(e.hp for e in obs.enemies),
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


def simulate_with_upgrade(
    character: Character,
    card_id: str,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    **kwargs: object,
) -> SimResult:
    """Simulate an encounter with a card upgraded in the deck.

    Finds the first un-upgraded copy of *card_id* and replaces it with the
    upgraded version (``card_id + "+"``).  The original *character* is not
    mutated.
    """
    char_copy = copy.deepcopy(character)
    for i, card in enumerate(char_copy.deck):
        if card == card_id:
            char_copy.deck[i] = card_id + "+"
            break
    return simulate_encounter(
        char_copy, encounter_type, encounter_id, seed, **kwargs
    )


def simulate_without_card(
    character: Character,
    card_id: str,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    **kwargs: object,
) -> SimResult:
    """Simulate an encounter with a card removed from the deck.

    Removes the first occurrence of *card_id* from the copy's deck.
    The original *character* is not mutated.
    """
    char_copy = copy.deepcopy(character)
    for i, card in enumerate(char_copy.deck):
        if card == card_id:
            del char_copy.deck[i]
            break
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
        Encounter specification.  ``encounter_type`` may be ``"monster"``
        and ``encounter_id`` may be empty — both are resolved via
        :func:`_resolve_enc` before building the combat.
    max_nodes:
        MCTS node budget.
    simulations:
        MCTS simulation budget.

    Returns
    -------
    SimDistribution with mean/std/max of terminal scores.
    """
    encounter_type, encounter_id = _resolve_enc(encounter_type, encounter_id, seed)
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
