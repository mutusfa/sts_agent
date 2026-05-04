"""Evaluate potion virtual costs via MCTS mini-simulations.

Public API
----------
evaluate_potions    — compute virtual HP cost for each potion the character holds.

The strategy layer calls this before combat to populate ``MCTSPlanner.potion_costs``.
MCTS then treats using a potion as "spending" that much HP, so it only uses potions
when they save more than their cost or prevent death.

Design
------
For each non-fairy potion, we compare MCTS outcomes WITH vs WITHOUT the potion
against the two most important upcoming encounters (boss + nearest reachable elite).

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
from collections import Counter

from sts_env.run.character import Character

from .simulate import probe_encounter

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
# Single potion evaluation
# ---------------------------------------------------------------------------

def _eval_single(
    character: Character,
    potion_id: str,
    targets: list[tuple[str, str, int]],
    seed: int,
) -> float:
    """Estimate HP saved by having this potion across all targets.

    For each target, run a probe WITH the potion and WITHOUT, then
    compute the damage difference discounted by floors_ahead.
    """
    best_saving = 0.0

    for enc_type, enc_id, floors_ahead in targets:
        # Run WITH potion (character has it)
        char_with = copy.deepcopy(character)
        if potion_id not in char_with.potions:
            # Potion isn't in bag; skip
            continue

        with_result = probe_encounter(
            char_with, enc_type, enc_id, seed,
            simulations=_PROBE_SIMULATIONS, max_nodes=_PROBE_MAX_NODES,
        )

        # Run WITHOUT potion (remove it)
        char_without = copy.deepcopy(character)
        if potion_id in char_without.potions:
            char_without.potions.remove(potion_id)

        without_result = probe_encounter(
            char_without, enc_type, enc_id, seed + 1,
            simulations=_PROBE_SIMULATIONS, max_nodes=_PROBE_MAX_NODES,
        )

        # HP saved = damage without - damage with
        # Use expected_damage (capped at start_hp) for robustness
        damage_without = without_result.expected_damage
        damage_with = with_result.expected_damage
        hp_saved = damage_without - damage_with

        # Discount by distance
        discount = FLOOR_DISCOUNT ** floors_ahead
        discounted = hp_saved * discount

        if discounted > best_saving:
            best_saving = discounted

    return best_saving


# ---------------------------------------------------------------------------
# Pair evaluation
# ---------------------------------------------------------------------------

def _eval_pairs(
    character: Character,
    potion_costs_singles: dict[str, float],
    targets: list[tuple[str, str, int]],
    seed: int,
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

            # Find best pair saving across targets
            best_pair = 0.0
            for enc_type, enc_id, floors_ahead in targets:
                # WITH both
                char_with = copy.deepcopy(character)
                with_result = probe_encounter(
                    char_with, enc_type, enc_id, seed + i * 10 + j,
                    simulations=_PROBE_SIMULATIONS, max_nodes=_PROBE_MAX_NODES,
                )

                # WITHOUT both
                char_without = copy.deepcopy(character)
                char_without.potions = [
                    p for p in char_without.potions if p != p1 and p != p2
                ]
                without_result = probe_encounter(
                    char_without, enc_type, enc_id, seed + i * 10 + j + 1,
                    simulations=_PROBE_SIMULATIONS, max_nodes=_PROBE_MAX_NODES,
                )

                discount = FLOOR_DISCOUNT ** floors_ahead
                pair_hp_saved = (without_result.expected_damage - with_result.expected_damage) * discount

                if pair_hp_saved > best_pair:
                    best_pair = pair_hp_saved

            # Credit each potion in the pair
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
    costs: dict[str, float] = {}
    bag_full = len(character.potions) >= character.max_potion_slots

    # FairyInABottle: fixed cost, no simulation needed
    for p in character.potions:
        if p == "FairyInABottle":
            costs[p] = FAIRY_VIRTUAL_COST_FACTOR * character.player_max_hp

    # Single-potion evaluation
    non_fairy = [p for p in character.potions if p != "FairyInABottle"]
    singles: dict[str, float] = {}
    for p in non_fairy:
        singles[p] = _eval_single(character, p, targets, seed)

    # Pair evaluation
    pair_savings = _eval_pairs(character, singles, targets, seed)

    # Combine: virtual_cost = max(single, pair)
    for p in non_fairy:
        single_val = singles.get(p, 0.0)
        pair_val = pair_savings.get(p, 0.0)
        cost = max(single_val, pair_val)
        if bag_full:
            cost *= FULL_BAG_DISCOUNT
        costs[p] = cost

    log.info(
        "Potion costs: %s (bag_full=%s)",
        {p: f"{c:.1f}" for p, c in costs.items()},
        bag_full,
    )
    return costs
