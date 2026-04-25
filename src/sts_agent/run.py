"""Run-level orchestrator — chains multiple combats into a full run.

Drives a BattlePlanner (or BattleAgent) through a sequence of encounters,
applying relic effects, card rewards, and potion rewards between combats.

Two run modes:

* ``run_scenario3`` — legacy 5-floor scenario using ``RunState``.
* ``run_act1`` — full Act 1 (8 floors) using ``Character`` and an optional
  LLM strategy agent for card picks.

Usage::

    # Legacy
    from sts_agent.run import run_scenario3
    from sts_agent.battle.mcts import MCTSPlanner

    result = run_scenario3(MCTSPlanner(), seed=42)
    print(result)

    # Act 1 with LLM strategy
    from sts_agent.run import run_act1
    from sts_agent.strategy import StrategyAgent

    result = run_act1(MCTSPlanner(), seed=42, strategy_agent=StrategyAgent())
    print(result)
"""

from __future__ import annotations

import logging
import inspect
from dataclasses import dataclass, field

from sts_env.combat import Combat
from sts_env.run.state import RunState
from sts_env.run import relics, rewards, scenarios, builder
from sts_env.run.character import Character
from sts_env.combat.rng import RNG

from .battle.base import BattleAgent, BattlePlanner, run_agent, run_planner

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Result of a completed (or failed) run."""

    victory: bool               # True if all combats survived
    floors_cleared: int         # Number of combats won
    total_floors: int           # Total combats attempted
    final_hp: int               # Player HP at end of run
    max_hp: int                 # Player max HP at end of run
    damage_taken_total: int     # Total damage taken across all combats (can be negative with Feed)
    max_hp_gained_total: int    # Total max HP gained across all combats (e.g. Feed kills)
    damage_per_floor: list[int] = field(default_factory=list)
    encounter_types: list[str] = field(default_factory=list)
    cards_added: list[str] = field(default_factory=list)
    potions_gained: list[str] = field(default_factory=list)


# =====================================================================
# Act 1 run (Character + optional LLM strategy agent)
# =====================================================================


def run_act1(
    planner_or_agent: BattlePlanner | BattleAgent,
    seed: int,
    *,
    strategy_agent: object | None = None,
    card_pick_strategy: str = "random",
    potion_use_strategy: str = "immediate",
) -> RunResult:
    """Run a full Act 1 scenario (8 floors) using :class:`Character`.

    Parameters
    ----------
    planner_or_agent:
        The battle AI for combat.
    seed:
        Master seed for the run.
    strategy_agent:
        A :class:`~sts_agent.strategy.StrategyAgent` for card-pick decisions.
        If ``None``, falls back to *card_pick_strategy*.
    card_pick_strategy:
        Fallback when no strategy agent: ``"random"`` or ``"skip"``.
    potion_use_strategy:
        How to use potions: ``"immediate"`` or ``"never"``.

    Returns
    -------
    RunResult with full run statistics.
    """
    encounter_list = scenarios.act1_encounters(seed)
    character = Character.ironclad()

    # Separate RNG for inter-combat events (rewards, etc.)
    reward_rng = RNG(seed ^ 0xBEEF)

    result = RunResult(
        victory=False,
        floors_cleared=0,
        total_floors=len(encounter_list),
        final_hp=character.player_hp,
        max_hp=character.player_max_hp,
        damage_taken_total=0,
        max_hp_gained_total=0,
    )

    for floor_idx, (encounter_type, encounter_id) in enumerate(encounter_list):
        character.floor = floor_idx + 1
        result.encounter_types.append(encounter_type)

        # Build combat with current character state
        combat_seed = seed * 1000 + floor_idx
        combat = builder.build_combat(
            encounter_type,
            encounter_id,
            combat_seed,
            character=character,
        )

        # Fight the combat
        damage = _run_battle(planner_or_agent, combat)

        result.damage_per_floor.append(damage)
        result.damage_taken_total += damage
        result.max_hp_gained_total += combat.max_hp_gained

        obs = combat.observe()
        if obs.player_dead:
            log.info(
                "FLOOR %d (%s/%s): DIED (damage=%d)",
                floor_idx + 1, encounter_type, encounter_id, damage,
            )
            result.final_hp = 0
            return result

        # Won — update character state
        character.player_hp = obs.player_hp
        character.player_max_hp = obs.player_max_hp

        # Apply relic effects (Burning Blood: heal 6)
        _apply_relic_effects(character)

        result.final_hp = character.player_hp
        result.max_hp = character.player_max_hp

        log.info(
            "FLOOR %d (%s/%s): WON (damage=%d, hp=%d/%d)",
            floor_idx + 1, encounter_type, encounter_id, damage,
            character.player_hp, character.player_max_hp,
        )

        # Card reward
        is_elite = encounter_type == "elite"
        card_choices = rewards.roll_card_rewards(reward_rng, is_elite=is_elite)

        # Remaining encounters (for LLM context)
        remaining = encounter_list[floor_idx + 1:]

        picked = _pick_card_act1(
            character=character,
            card_choices=card_choices,
            remaining_encounters=remaining,
            seed=combat_seed,
            strategy_agent=strategy_agent,
            fallback_strategy=card_pick_strategy,
            rng=reward_rng,
        )
        if picked is not None:
            character.add_card(picked)
            result.cards_added.append(picked)
            log.info("  Card reward: picked %s from %s", picked, card_choices)
        else:
            log.info("  Card reward: skipped %s", card_choices)

        # Potion reward
        potion = rewards.roll_potion_reward(reward_rng)
        if potion is not None:
            if len(character.potions) < character.max_potion_slots:
                character.add_potion(potion)
                result.potions_gained.append(potion)
                log.info("  Potion reward: %s (slots: %s)", potion, character.potions)
            else:
                log.info("  Potion reward: %s discarded (no slot)", potion)

        # Gold reward
        gold_reward = 30 if is_elite else 20 if encounter_type == "boss" else 10
        character.gold += gold_reward

        result.floors_cleared += 1

    result.victory = True
    result.final_hp = character.player_hp
    result.max_hp = character.player_max_hp
    return result


# =====================================================================
# Legacy Scenario 3 run (RunState-based)
# =====================================================================


def run_scenario3(
    planner_or_agent: BattlePlanner | BattleAgent,
    seed: int,
    *,
    card_pick_strategy: str = "random",
    potion_use_strategy: str = "immediate",
) -> RunResult:
    """Run Scenario 3: 3 easy hallways + 1 hard + 1 elite.

    Parameters
    ----------
    planner_or_agent:
        The battle AI — either a BattlePlanner (full combat clone access)
        or a BattleAgent (observation-only).
    seed:
        Master seed for the run.
    card_pick_strategy:
        How to pick card rewards: "random" (default) or "skip".
    potion_use_strategy:
        How to use potions: "immediate" (default) or "never".

    Returns
    -------
    RunResult with full run statistics.
    """
    encounter_list = scenarios.scenario3_encounters(seed)
    run_state = RunState()

    # Separate RNG for inter-combat events (rewards, etc.)
    reward_rng = RNG(seed ^ 0xBEEF)

    result = RunResult(
        victory=False,
        floors_cleared=0,
        total_floors=len(encounter_list),
        final_hp=80,
        max_hp=80,
        damage_taken_total=0,
        max_hp_gained_total=0,
    )

    for floor_idx, (encounter_type, encounter_id) in enumerate(encounter_list):
        run_state.floor = floor_idx + 1
        result.encounter_types.append(encounter_type)

        # Build combat with current run state
        combat_seed = seed * 1000 + floor_idx  # deterministic per-floor seed
        combat = builder.build_combat(
            encounter_type,
            encounter_id,
            combat_seed,
            deck=run_state.deck,
            player_hp=run_state.player_hp,
            player_max_hp=run_state.player_max_hp,
            potions=run_state.potions,
        )

        # Fight the combat
        damage = _run_battle(planner_or_agent, combat)

        result.damage_per_floor.append(damage)
        result.damage_taken_total += damage
        result.max_hp_gained_total += combat.max_hp_gained

        obs = combat.observe()
        if obs.player_dead:
            log.info(
                "FLOOR %d (%s/%s): DIED (damage=%d)",
                floor_idx + 1, encounter_type, encounter_id, damage,
            )
            result.final_hp = 0
            return result

        # Won the combat — apply rewards
        run_state.player_hp = obs.player_hp
        run_state.player_max_hp = obs.player_max_hp
        result.final_hp = run_state.player_hp

        # Apply relic effects (Burning Blood: heal 6)
        relics.on_combat_end(run_state)
        result.final_hp = run_state.player_hp

        log.info(
            "FLOOR %d (%s/%s): WON (damage=%d, hp=%d/%d)",
            floor_idx + 1, encounter_type, encounter_id, damage,
            run_state.player_hp, run_state.player_max_hp,
        )

        # Card reward: pick one of 3 cards (or skip)
        is_elite = encounter_type == "elite"
        card_choices = rewards.roll_card_rewards(reward_rng, is_elite=is_elite)
        picked = _pick_card(card_choices, card_pick_strategy, reward_rng)
        if picked is not None:
            run_state.add_card(picked)
            result.cards_added.append(picked)
            log.info("  Card reward: picked %s from %s", picked, card_choices)
        else:
            log.info("  Card reward: skipped %s", card_choices)

        # Potion reward
        potion = rewards.roll_potion_reward(reward_rng)
        if potion is not None:
            if len(run_state.potions) < 3:
                run_state.add_potion(potion)
                result.potions_gained.append(potion)
                log.info("  Potion reward: %s (slots: %s)", potion, run_state.potions)
            else:
                log.info("  Potion reward: %s discarded (no slot)", potion)

        # Gold reward (not strategic, just tracking)
        gold_reward = 20 if is_elite else 10
        run_state.gold += gold_reward

        result.floors_cleared += 1

    # All combats survived
    result.victory = True
    result.final_hp = run_state.player_hp
    result.max_hp = run_state.player_max_hp
    return result


# =====================================================================
# Internal helpers
# =====================================================================


def _run_battle(
    planner_or_agent: BattlePlanner | BattleAgent,
    combat: Combat,
) -> int:
    """Run a combat and return damage taken."""
    try:
        sig = inspect.signature(planner_or_agent.act)
        params = list(sig.parameters.keys())
        is_planner = len(params) == 1 and params[0] not in ('obs', 'observation')
    except (ValueError, TypeError):
        is_planner = True
    if is_planner:
        return run_planner(planner_or_agent, combat)  # type: ignore[arg-type]
    else:
        return run_agent(planner_or_agent, combat)  # type: ignore[arg-type]


def _apply_relic_effects(character: Character) -> None:
    """Apply end-of-combat relic effects (Burning Blood: heal 6)."""
    if character.has_relic("BurningBlood"):
        character.heal(6)


def _pick_card_act1(
    *,
    character: Character,
    card_choices: list[str],
    remaining_encounters: list[tuple[str, str]],
    seed: int,
    strategy_agent: object | None,
    fallback_strategy: str,
    rng: RNG,
) -> str | None:
    """Pick a card using the strategy agent if available, else fallback."""
    if strategy_agent is not None and hasattr(strategy_agent, "pick_card"):
        return strategy_agent.pick_card(
            character, card_choices, remaining_encounters, seed,
        )
    return _pick_card(card_choices, fallback_strategy, rng)


def _pick_card(
    choices: list[str],
    strategy: str,
    rng: RNG,
) -> str | None:
    """Pick a card from the reward choices.

    Parameters
    ----------
    choices:
        3 card IDs to choose from.
    strategy:
        "random" — pick a random card from choices.
        "skip" — don't pick any card (return None).

    Returns
    -------
    The picked card ID, or None if skipped.
    """
    if strategy == "skip" or not choices:
        return None
    # "random" — any card is generally better than starter cards early in the run
    return rng.choice(choices)
