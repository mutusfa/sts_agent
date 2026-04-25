"""Combat simulation helpers for the LLM strategy agent.

Public API
----------
simulate_encounter  — run one encounter with MCTS and return structured results.
simulate_with_card  — convenience wrapper that adds a hypothetical card first.
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
