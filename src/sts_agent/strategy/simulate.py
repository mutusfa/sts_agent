"""Combat simulation helpers for the strategy agent.

Public API
----------
probe_encounter       — single-act MCTS probe returning a SimDistribution.
probe_with_card       — same but with a hypothetical card added.
probe_with_upgrade    — same but with a card upgraded.
probe_after_rest      — same but with the character healed first.
probe_without_card    — same but with a card removed.

Each probe function calls ``MCTSPlanner.act()`` exactly once, which
internally runs ``simulations`` full-combat rollouts and captures the
distribution of terminal scores from the root edge.  This is the most
informative and most efficient way to evaluate a deck against an encounter.
"""

from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import dataclass
from time import perf_counter
from typing import TYPE_CHECKING, Literal

from sts_env.run.builder import build_combat
from sts_env.run.character import Character
from sts_env.run.rooms import rest_heal

from ..battle.mcts import MCTSPlanner

if TYPE_CHECKING:
    from .probe_data import ProbeCache

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


def get_encounter_pool(
    room_type: str, encounter_id: str
) -> list[tuple[str, str]] | None:
    """Return the pool for this encounter, or ``None`` if it is a specific encounter.

    A reference is considered a *pool* when *encounter_id* is empty (use
    *room_type* as the pool key) or when *encounter_id* is itself a pool
    key (e.g. the LLM passed ``"monster"`` or ``"elite"`` literally).

    Parameters
    ----------
    room_type:
        Room type from the map (``"monster"``, ``"elite"``, ``"boss"``, …).
    encounter_id:
        The encounter identifier supplied by the caller.  May be empty,
        a pool key, or a specific encounter ID.

    Returns
    -------
    The pool list when the encounter is a pool reference, ``None`` when it
    names a specific enemy.
    """
    if not encounter_id:
        return _ACT1_POOLS.get(room_type)
    return _ACT1_POOLS.get(encounter_id)  # None when encounter_id is specific


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
        Player current HP at the start of combat (used to interpret scores).
    max_hp:
        Player max HP at the start of combat (shown as the final_hp denominator).
    mean_damage_alive:
        Mean effective damage in surviving rollouts (from MCTS stats).  When set,
        :attr:`expected_damage` uses this instead of the tier-encoded scalar.
    mean_enemy_dmg_dead:
        Mean enemy damage dealt in rollouts where the player died.
    mean_turns_dead:
        Mean turns elapsed at death in dying rollouts.
    mean_damage_taken_dead:
        Mean raw player damage taken in dying rollouts.
    """

    mean_score: float
    std_score: float
    max_score: float
    n: int
    deaths: int
    start_hp: int
    max_hp: int
    mean_damage_alive: float = float("nan")
    max_hp_gained_mean: float = 0.0
    max_hp_gained_std: float = 0.0
    mean_enemy_dmg_dead: float = float("nan")
    mean_turns_dead: float = float("nan")
    mean_damage_taken_dead: float = float("nan")

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
        """Expected damage taken in surviving rollouts.

        Uses ``mean_damage_alive`` from MCTS when available.  When every rollout
        died, returns ``start_hp`` (all current HP is lost).  Otherwise falls
        back to the tier-encoded scalar capped at current HP.
        """
        if math.isfinite(self.mean_damage_alive):
            return min(self.mean_damage_alive, float(self.start_hp))
        if self.n > 0 and self.deaths == self.n:
            return float(self.start_hp)
        return min(self.mean_score, float(self.start_hp))

    @property
    def damage_spread(self) -> str:
        """Human-readable spread, e.g. '40±20 (12% death)'."""
        pct = f"{self.death_rate:.0%}" if self.n > 0 else "?"
        return f"{self.expected_damage:.0f}±{self.std_score:.0f} ({pct} death)"

    @property
    def max_hp_gained_spread(self) -> str:
        """Human-readable max-HP-gained spread, e.g. '1.2±0.4'. Only shown when mean > 0."""
        return f"{self.max_hp_gained_mean:.1f}±{self.max_hp_gained_std:.1f}"


# Threshold for applying death-progress heuristic (matches high-death probes).
_DEATH_PROGRESS_DEATH_RATE = 0.95


def _death_progress_benefit(without: SimDistribution, with_: SimDistribution) -> float:
    """Value extra enemy damage on dying timelines in player-HP equivalents."""
    if not math.isfinite(without.mean_enemy_dmg_dead) or not math.isfinite(
        with_.mean_enemy_dmg_dead
    ):
        return 0.0
    extra_enemy = with_.mean_enemy_dmg_dead - without.mean_enemy_dmg_dead
    if extra_enemy <= 0:
        return 0.0
    if (
        not math.isfinite(without.mean_turns_dead)
        or without.mean_turns_dead <= 0
        or without.mean_enemy_dmg_dead <= 0
        or not math.isfinite(without.mean_damage_taken_dead)
    ):
        return 0.0

    avg_enemy_per_turn = without.mean_enemy_dmg_dead / without.mean_turns_dead
    avg_player_per_turn = without.mean_damage_taken_dead / without.mean_turns_dead
    turns_worth = extra_enemy / avg_enemy_per_turn
    return turns_worth * avg_player_per_turn


def marginal_probe_benefit(
    without: SimDistribution,
    with_: SimDistribution,
) -> float:
    """Estimate HP saved by the potion from a WITH vs WITHOUT probe pair."""
    hp = without.expected_damage - with_.expected_damage
    survival = (without.death_rate - with_.death_rate) * without.start_hp
    hp = max(hp, survival)

    if (
        without.death_rate >= _DEATH_PROGRESS_DEATH_RATE
        and with_.death_rate >= _DEATH_PROGRESS_DEATH_RATE
    ):
        hp = max(hp, _death_progress_benefit(without, with_))

    return max(0.0, hp)


# ---------------------------------------------------------------------------
# Probe API — single act() call, returns distribution
# ---------------------------------------------------------------------------


def probe_encounter(
    character: Character,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    *,
    max_nodes: int = 10_000,
    simulations: int = 10_000,
    probe_cache: ProbeCache | None = None,
    rollout_mode: Literal["heuristic", "in_tree"] = "heuristic",
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
    probe_cache:
        Optional run-scoped cache; identical keys skip MCTS.

    Returns
    -------
    SimDistribution with mean/std/max of terminal scores.
    """
    from .probe_data import (
        get_probe_context,
        probe_cache_key,
    )

    encounter_type, encounter_id = _resolve_enc(encounter_type, encounter_id, seed)
    cache_key = probe_cache_key(
        character,
        encounter_type,
        encounter_id,
        seed,
        max_nodes=max_nodes,
        simulations=simulations,
        rollout_mode=rollout_mode,
    )

    ctx = get_probe_context()
    decision_type = ctx.decision_type if ctx is not None else None

    if probe_cache is not None:
        cached = probe_cache.get(cache_key)
        if cached is not None:
            probe_cache.record_hit(decision_type)
            return cached

    t0 = perf_counter()
    dist = _probe_encounter_uncached(
        character,
        encounter_type,
        encounter_id,
        seed,
        max_nodes=max_nodes,
        simulations=simulations,
        rollout_mode=rollout_mode,
    )
    elapsed = perf_counter() - t0

    if probe_cache is not None:
        probe_cache.put(cache_key, dist)
        probe_cache.record_miss(decision_type, elapsed)

    return dist


def _probe_encounter_uncached(
    character: Character,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    *,
    max_nodes: int,
    simulations: int,
    rollout_mode: Literal["heuristic", "in_tree"] = "heuristic",
) -> SimDistribution:
    """Run MCTS for one probe (no cache). *encounter_* must be pool-resolved."""
    char_copy = copy.deepcopy(character)
    combat = build_combat(
        encounter_type, encounter_id, seed, character=char_copy
    )
    planner = MCTSPlanner(
        simulations=simulations,
        max_nodes=max_nodes,
        rollout_mode=rollout_mode,
    )

    planner.act(combat)

    stats = planner.last_stats
    obs = combat.observe()

    dist = SimDistribution(
        mean_score=stats.get("mean", float("nan")),
        std_score=stats.get("std", 0.0),
        max_score=stats.get("max", float("nan")),
        n=int(stats.get("n", 0)),
        deaths=int(stats.get("deaths", 0)),
        start_hp=obs.player_hp,
        max_hp=obs.player_max_hp,
        mean_damage_alive=float(stats.get("mean_damage_alive", float("nan"))),
        max_hp_gained_mean=float(stats.get("pv_max_hp_gained_mean", 0.0)),
        max_hp_gained_std=float(stats.get("pv_max_hp_gained_std", 0.0)),
        mean_enemy_dmg_dead=float(stats.get("mean_enemy_dmg_dead", float("nan"))),
        mean_turns_dead=float(stats.get("mean_turns_dead", float("nan"))),
        mean_damage_taken_dead=float(
            stats.get("mean_damage_taken_dead", float("nan"))
        ),
    )

    from .probe_data import get_probe_context

    ctx = get_probe_context()
    if ctx is not None:
        ctx.collector.log_probe(
            dist,
            encounter_type=encounter_type,
            encounter_id=encounter_id,
            enc_seed=seed,
            character=character,
            ctx=ctx,
        )

    return dist


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


def probe_with_upgrade(
    character: Character,
    card_id: str,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    **kwargs: object,
) -> SimDistribution:
    """Probe an encounter with a card upgraded in the deck.

    Replaces the first un-upgraded copy of *card_id* with ``card_id + "+"``.
    The original *character* is not mutated.
    """
    char_copy = copy.deepcopy(character)
    for i, card in enumerate(char_copy.deck):
        if card == card_id:
            char_copy.deck[i] = card_id + "+"
            break
    return probe_encounter(
        char_copy, encounter_type, encounter_id, seed, **kwargs
    )


def probe_after_rest(
    character: Character,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    **kwargs: object,
) -> SimDistribution:
    """Probe an encounter after resting (healing 30% of max HP).

    Applies the same heal logic as a real rest site visit.
    The original *character* is not mutated.
    """
    char_copy = copy.deepcopy(character)
    rest_heal(char_copy)
    return probe_encounter(char_copy, encounter_type, encounter_id, seed, **kwargs)


def probe_without_card(
    character: Character,
    card_id: str,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    **kwargs: object,
) -> SimDistribution:
    """Probe an encounter with a card removed from the deck.

    Removes the first occurrence of *card_id* from the copy's deck.
    The original *character* is not mutated.
    """
    char_copy = copy.deepcopy(character)
    for i, card in enumerate(char_copy.deck):
        if card == card_id:
            del char_copy.deck[i]
            break
    return probe_encounter(
        char_copy, encounter_type, encounter_id, seed, **kwargs
    )


def probe_with_relic(
    character: Character,
    relic_id: str,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    **kwargs: object,
) -> SimDistribution:
    """Probe an encounter with a hypothetical relic added.

    The original *character* is not mutated.
    """
    char_copy = copy.deepcopy(character)
    char_copy.add_relic(relic_id)
    return probe_encounter(
        char_copy, encounter_type, encounter_id, seed, **kwargs
    )
