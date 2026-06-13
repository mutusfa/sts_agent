"""Evaluate potion virtual costs via MCTS mini-simulations.

Public API
----------
evaluate_potions    — compute virtual HP cost for each potion the character holds.

The strategy layer calls this before combat to populate ``MCTSPlanner.potion_costs``.
MCTS then treats using a potion as "spending" that much HP, so it only uses potions
when they save more than their cost or prevent death.

Design
------
For each non-fairy potion, we compare MCTS outcomes WITH an isolated bag
containing only that potion vs an empty bag against the two most important
upcoming encounters (boss + nearest reachable elite).

Virtual cost = max(single_savings, pair_savings), discounted by floors remaining.

Constants from the design spec:
    FLOOR_DISCOUNT       = 0.9   (compounds: 0.9^10 ≈ 0.35)
    FULL_BAG_DISCOUNT    = 0.8   (~1 potion drop per 3 fights)
    FAIRY_VIRTUAL_COST   = 0.6 * max_hp
    PAIR_CREDIT_FACTOR   = 0.6   (1.2× total, keeps combos together)
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

from sts_env.run.character import Character

from .probe_data import sim_distribution_to_dict
from .simulate import (
    SimDistribution,
    _resolve_enc,
    marginal_probe_benefit,
    probe_encounter,
)

if TYPE_CHECKING:
    from .probe_data import ProbeCache

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# V1 Constants
# ---------------------------------------------------------------------------

FLOOR_DISCOUNT = 0.9
FULL_BAG_DISCOUNT = 0.8
FAIRY_VIRTUAL_COST_FACTOR = 0.6
PAIR_CREDIT_FACTOR = 0.6

# Sim budget for each probe — fast, approximate.
_PROBE_SIMULATIONS = 300
_PROBE_MAX_NODES = 5_000

TargetKey = tuple[str, str, int]


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


def _select_targets(
    upcoming: list[tuple[str, str]],
    possible_encounters: dict | None,
) -> list[tuple[str, str, int]]:
    """Select encounter targets for potion evaluation.

    Returns list of (enc_type, enc_id, floors_ahead) for the boss and
    nearest elite on the remaining path.

    Falls back to generic pools if no specific encounters are found.
    """
    targets: list[tuple[str, str, int]] = []
    boss_found = False
    elite_found = False

    for i, (room_type, enc_id) in enumerate(upcoming):
        if not boss_found and room_type == "boss":
            targets.append(("boss", enc_id or "", i + 1))
            boss_found = True
        if not elite_found and room_type == "elite":
            targets.append(("elite", enc_id or "", i + 1))
            elite_found = True
        if boss_found and elite_found:
            break

    # Fallbacks if path doesn't include these room types
    if not boss_found:
        targets.append(("boss", "", 10))  # assume far away
    if not elite_found:
        targets.append(("elite", "", 5))  # assume mid-range

    return targets


# ---------------------------------------------------------------------------
# Probe helpers
# ---------------------------------------------------------------------------


def _resolved_encounter(
    enc_type: str,
    enc_id: str,
    seed: int,
) -> tuple[str, str]:
    """Resolve pool targets once so WITH/WITHOUT probes the same encounter."""
    return _resolve_enc(enc_type, enc_id, seed)


def _probe_without_baseline(
    character: Character,
    enc_type: str,
    enc_id: str,
    seed: int,
    *,
    probe_cache: ProbeCache | None = None,
) -> SimDistribution:
    """Probe with an empty potion bag."""
    resolved_type, resolved_id = _resolved_encounter(enc_type, enc_id, seed)
    char = copy.deepcopy(character)
    char.potions = []
    return probe_encounter(
        char,
        resolved_type,
        resolved_id,
        seed,
        simulations=_PROBE_SIMULATIONS,
        max_nodes=_PROBE_MAX_NODES,
        probe_cache=probe_cache,
    )


def _probe_with_potion(
    character: Character,
    potion_id: str,
    enc_type: str,
    enc_id: str,
    seed: int,
    *,
    probe_cache: ProbeCache | None = None,
) -> SimDistribution:
    """Probe with an isolated bag containing only *potion_id*."""
    resolved_type, resolved_id = _resolved_encounter(enc_type, enc_id, seed)
    char = copy.deepcopy(character)
    char.potions = [potion_id]
    return probe_encounter(
        char,
        resolved_type,
        resolved_id,
        seed,
        simulations=_PROBE_SIMULATIONS,
        max_nodes=_PROBE_MAX_NODES,
        probe_cache=probe_cache,
    )


def _build_without_cache(
    character: Character,
    targets: list[TargetKey],
    seed: int,
    *,
    probe_cache: ProbeCache | None = None,
) -> dict[TargetKey, SimDistribution]:
    """Probe empty-bag baseline once per target."""
    cache: dict[TargetKey, SimDistribution] = {}
    for enc_type, enc_id, floors_ahead in targets:
        key = (enc_type, enc_id, floors_ahead)
        if key not in cache:
            cache[key] = _probe_without_baseline(
                character, enc_type, enc_id, seed, probe_cache=probe_cache,
            )
    return cache


def _marginal_saving_for_target(
    without: SimDistribution,
    with_: SimDistribution,
    floors_ahead: int,
) -> float:
    """Discounted HP benefit for one target."""
    hp_saved = marginal_probe_benefit(without, with_)
    return hp_saved * (FLOOR_DISCOUNT**floors_ahead)


# ---------------------------------------------------------------------------
# Single potion evaluation
# ---------------------------------------------------------------------------


def _eval_single(
    character: Character,
    potion_id: str,
    targets: list[TargetKey],
    seed: int,
    *,
    without_cache: dict[TargetKey, SimDistribution] | None = None,
    detail_out: dict[str, object] | None = None,
    probe_cache: ProbeCache | None = None,
) -> float:
    """Estimate standalone HP saved by having this potion across all targets.

    Probes with an isolated bag: WITH ``[potion_id]`` vs WITHOUT ``[]``.
    The potion need not already be in the character's bag (shop/discard flows).
    """
    cache = without_cache or _build_without_cache(
        character, targets, seed, probe_cache=probe_cache,
    )
    best_saving = 0.0
    probe_pairs: list[dict[str, object]] = []

    for enc_type, enc_id, floors_ahead in targets:
        key = (enc_type, enc_id, floors_ahead)
        without_result = cache[key]
        with_result = _probe_with_potion(
            character,
            potion_id,
            enc_type,
            enc_id,
            seed,
            probe_cache=probe_cache,
        )

        discounted = _marginal_saving_for_target(
            without_result, with_result, floors_ahead
        )

        probe_pairs.append(
            {
                "potion_id": potion_id,
                "encounter_type": enc_type,
                "encounter_id": enc_id,
                "floors_ahead": floors_ahead,
                "without": sim_distribution_to_dict(without_result),
                "with": sim_distribution_to_dict(with_result),
                "discounted_benefit": discounted,
            }
        )

        log.debug(
            "  %s potion=%s enc=%s/%s floors=%d  "
            "death_without=%.0f%% death_with=%.0f%%  benefit=%.1f disc=%.2f -> %.1f",
            "(new best)" if discounted > best_saving else "       ",
            potion_id,
            enc_type,
            enc_id,
            floors_ahead,
            without_result.death_rate * 100,
            with_result.death_rate * 100,
            marginal_probe_benefit(without_result, with_result),
            FLOOR_DISCOUNT**floors_ahead,
            discounted,
        )
        if discounted > best_saving:
            best_saving = discounted

    if detail_out is not None:
        detail_out.setdefault("probe_pairs", []).extend(probe_pairs)
        detail_out.setdefault("singles", {})[potion_id] = best_saving

    return best_saving


# ---------------------------------------------------------------------------
# Pair evaluation
# ---------------------------------------------------------------------------


def _eval_pairs(
    character: Character,
    potion_costs_singles: dict[str, float],
    targets: list[TargetKey],
    seed: int,
    *,
    without_cache: dict[TargetKey, SimDistribution] | None = None,
    probe_cache: ProbeCache | None = None,
) -> dict[str, float]:
    """Evaluate potion pairs and return updated costs.

    For each pair of non-fairy potions, run a probe with BOTH vs NEITHER.
    Each potion in the pair gets credited PAIR_CREDIT_FACTOR of the pair's HP savings.
    """
    potions = [p for p in character.potions if p != "FairyInABottle"]
    pair_savings: dict[str, float] = {}

    if len(potions) < 2:
        return pair_savings

    for i in range(len(potions)):
        for j in range(i + 1, len(potions)):
            p1, p2 = potions[i], potions[j]

            best_pair = 0.0
            for enc_type, enc_id, floors_ahead in targets:
                pair_seed = seed + i * 10 + j
                resolved_type, resolved_id = _resolved_encounter(
                    enc_type, enc_id, pair_seed
                )

                char_with = copy.deepcopy(character)
                with_result = probe_encounter(
                    char_with,
                    resolved_type,
                    resolved_id,
                    pair_seed,
                    simulations=_PROBE_SIMULATIONS,
                    max_nodes=_PROBE_MAX_NODES,
                    probe_cache=probe_cache,
                )

                char_without = copy.deepcopy(character)
                char_without.potions = [
                    p for p in char_without.potions if p != p1 and p != p2
                ]
                without_result = probe_encounter(
                    char_without,
                    resolved_type,
                    resolved_id,
                    pair_seed,
                    simulations=_PROBE_SIMULATIONS,
                    max_nodes=_PROBE_MAX_NODES,
                    probe_cache=probe_cache,
                )

                pair_hp_saved = _marginal_saving_for_target(
                    without_result, with_result, floors_ahead
                )

                log.debug(
                    "  pair (%s, %s) enc=%s/%s  pair_hp_saved=%.1f (best=%.1f)",
                    p1,
                    p2,
                    enc_type,
                    enc_id,
                    pair_hp_saved,
                    best_pair,
                )
                if pair_hp_saved > best_pair:
                    best_pair = pair_hp_saved

            credit = PAIR_CREDIT_FACTOR * best_pair
            pair_savings[p1] = max(pair_savings.get(p1, 0.0), credit)
            pair_savings[p2] = max(pair_savings.get(p2, 0.0), credit)

    return pair_savings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_potions(
    character: Character,
    upcoming: list[tuple[str, str]],
    seed: int,
    *,
    possible_encounters: dict | None = None,
    detail_out: dict[str, object] | None = None,
    probe_cache: ProbeCache | None = None,
) -> dict[str, float]:
    """Compute virtual HP costs for all potions the character currently holds.

    Parameters
    ----------
    character:
        The current Character state (potions, deck, HP, etc.).
    upcoming:
        List of (room_type, enc_id) for remaining floors on the path.
    seed:
        Base seed for deterministic simulations.
    possible_encounters:
        Optional result from ``get_possible_encounters()`` for pool-aware
        target selection (currently unused — deferred to later phase).

    Returns
    -------
    dict mapping potion_id → virtual HP cost. Empty dict if character has
    no potions.
    """
    if not character.potions:
        return {}

    targets = _select_targets(upcoming, possible_encounters)
    log.debug("Targets for potion eval: %s", targets)
    if detail_out is not None:
        detail_out.clear()
        detail_out["targets"] = [
            {"encounter_type": t, "encounter_id": e, "floors_ahead": f}
            for t, e, f in targets
        ]
        detail_out["potions_held"] = list(character.potions)
        detail_out["probe_pairs"] = []
        detail_out["singles"] = {}
        detail_out["pair_credits"] = {}
    costs: dict[str, float] = {}
    bag_full = len(character.potions) >= character.max_potion_slots

    for p in character.potions:
        if p == "FairyInABottle":
            costs[p] = FAIRY_VIRTUAL_COST_FACTOR * character.player_max_hp

    non_fairy = [p for p in character.potions if p != "FairyInABottle"]
    if not non_fairy:
        log.info(
            "Potion costs: %s (bag_full=%s)",
            {p: f"{c:.1f}" for p, c in costs.items()},
            bag_full,
        )
        return costs

    without_cache = _build_without_cache(
        character, targets, seed, probe_cache=probe_cache,
    )

    singles: dict[str, float] = {}
    for p in non_fairy:
        singles[p] = _eval_single(
            character, p, targets, seed,
            without_cache=without_cache,
            detail_out=detail_out,
            probe_cache=probe_cache,
        )

    pair_savings = _eval_pairs(
        character,
        singles,
        targets,
        seed,
        without_cache=without_cache,
        probe_cache=probe_cache,
    )
    if detail_out is not None:
        detail_out["pair_credits"] = dict(pair_savings)

    for p in non_fairy:
        single_val = singles.get(p, 0.0)
        pair_val = pair_savings.get(p, 0.0)
        cost = max(single_val, pair_val)
        if bag_full:
            cost *= FULL_BAG_DISCOUNT
        costs[p] = cost

    if detail_out is not None:
        detail_out["assigned_costs"] = dict(costs)
        detail_out["bag_full"] = bag_full

    log.info(
        "Potion costs: %s (bag_full=%s)",
        {p: f"{c:.1f}" for p, c in costs.items()},
        bag_full,
    )
    return costs
