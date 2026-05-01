"""Run-level orchestrator — chains multiple combats into a full run.

Drives a BattlePlanner (or BattleAgent) through a sequence of encounters,
applying relic effects, card rewards, and potion rewards between combats.

All per-floor strategic decisions (Neow blessing, route through the map,
card rewards, rest sites, events, shops, boss relic) are delegated to a
:class:`~sts_agent.strategy.BaseStrategyAgent`.  When ``strategy_agent``
is ``None`` the run defaults to a fresh :class:`BaseStrategyAgent` seeded
from the run seed — i.e. a uniformly random valid choice for every
question.

Two run modes:

* ``run_scenario3`` — legacy 5-floor scenario using ``RunState``.
* ``run_act1`` — full Act 1 (15 floors via map, 8 floors via legacy linear)
  using ``Character``.

Usage::

    from sts_agent.run import run_act1
    from sts_agent.battle.mcts import MCTSPlanner
    from sts_agent.strategy import StrategyAgent

    result = run_act1(MCTSPlanner(), seed=42, strategy_agent=StrategyAgent())
    print(result)
"""

from __future__ import annotations

import logging
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field

import mlflow

from sts_env.combat import Combat
from sts_env.run.state import RunState
from sts_env.run import relics, rewards, scenarios, builder
from sts_env.run.rewards import Room as _RewardRoom
from sts_env.run.character import Character
from sts_env.run.map import generate_act1_map, get_encounter_for_room, RoomType
from sts_env.run.rooms import (
    RestChoice,
    rest_heal,
    rest_upgrade,
    _best_upgrade_target,
)
from sts_env.run.events import random_act1_event, resolve_event
from sts_env.run.shop import generate_shop
from sts_env.run.treasure import open_treasure
from sts_env.run.neow import roll_neow_options, apply_neow
from sts_env.combat.rng import RNG

from .battle.base import BattleAgent, BattlePlanner, run_agent, run_planner
from .strategy.base import BaseStrategyAgent

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
# Act 1 run (Character + strategy agent)
# =====================================================================


@mlflow.trace(name="run_act1")
def run_act1(
    planner_or_agent: BattlePlanner | BattleAgent,
    seed: int,
    *,
    strategy_agent: BaseStrategyAgent | None = None,
    use_map: bool = True,
) -> RunResult:
    """Run a full Act 1 scenario using :class:`Character` and generated map.

    Parameters
    ----------
    planner_or_agent:
        The battle AI for combat.
    seed:
        Master seed for the run.
    strategy_agent:
        A :class:`~sts_agent.strategy.BaseStrategyAgent` (or subclass) that
        owns every per-floor decision.  When ``None``, a fresh
        ``BaseStrategyAgent`` seeded from ``seed`` is used — i.e. random
        valid choices.
    use_map:
        If True (default), generate a branching map and walk it (15 floors).
        If False, use the old linear encounter list (8 floors, backwards compat).

    Returns
    -------
    RunResult with full run statistics.
    """
    if strategy_agent is None:
        strategy_agent = BaseStrategyAgent(seed=seed)

    character = Character.ironclad()
    reward_rng = RNG(seed ^ 0xBEEF)
    neow_rng = RNG(seed ^ 0xCA7)

    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "seed": seed,
            "use_map": use_map,
            "strategy_agent": type(strategy_agent).__name__,
        })

    # --- Neow's blessing (before any floor) ---
    neow_options = roll_neow_options(neow_rng)
    neow_pick = strategy_agent.pick_neow(neow_options)
    neow_desc = apply_neow(neow_pick, character, neow_rng)
    log.info("NEOW: %s", neow_desc)

    if use_map:
        return _run_act1_map(
            planner_or_agent, seed, character, reward_rng,
            strategy_agent=strategy_agent,
        )
    else:
        return _run_act1_linear(
            planner_or_agent, seed, character, reward_rng,
            strategy_agent=strategy_agent,
        )


# =====================================================================
# Legacy Scenario 3 run (RunState-based)
# =====================================================================


def run_scenario3(
    planner_or_agent: BattlePlanner | BattleAgent,
    seed: int,
    *,
    strategy_agent: BaseStrategyAgent | None = None,
) -> RunResult:
    """Run Scenario 3: 3 easy hallways + 1 hard + 1 elite.

    Parameters
    ----------
    planner_or_agent:
        The battle AI — either a BattlePlanner (full combat clone access)
        or a BattleAgent (observation-only).
    seed:
        Master seed for the run.
    strategy_agent:
        Strategy agent for card-pick decisions.  Defaults to a random
        :class:`BaseStrategyAgent` seeded from ``seed``.

    Returns
    -------
    RunResult with full run statistics.
    """
    if strategy_agent is None:
        strategy_agent = BaseStrategyAgent(seed=seed)

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

        combat_seed = seed * 1000 + floor_idx
        combat = builder.build_combat(
            encounter_type,
            encounter_id,
            combat_seed,
            deck=run_state.deck,
            player_hp=run_state.player_hp,
            player_max_hp=run_state.player_max_hp,
            potions=run_state.potions,
        )

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

        # Won the combat — sync state back
        run_state.player_hp = obs.player_hp
        run_state.player_max_hp = obs.player_max_hp
        run_state.potions = list(combat._state.potions)
        result.final_hp = run_state.player_hp

        relics.on_combat_end(run_state)
        result.final_hp = run_state.player_hp

        log.info(
            "FLOOR %d (%s/%s): WON (damage=%d, hp=%d/%d)",
            floor_idx + 1, encounter_type, encounter_id, damage,
            run_state.player_hp, run_state.player_max_hp,
        )

        room = _RewardRoom.ELITE if encounter_type == "elite" else _RewardRoom.BOSS if encounter_type == "boss" else _RewardRoom.MONSTER
        offer, _ = rewards.roll_combat_reward_offer(
            reward_rng, room, event_bus=run_state.event_bus,
        )
        upcoming = [(t, e) for t, e in encounter_list[floor_idx + 1:]]
        # Scenario 3 has no Character; fake one is overkill — pass run_state
        # via a thin shim only if the agent needs it.  Most agents only inspect
        # ``character.deck``/``character.player_hp`` — both available on RunState.
        picked = strategy_agent.pick_card(
            run_state, list(offer.card_choices), upcoming, combat_seed,
        )
        if picked is not None:
            run_state.add_card(picked)
            result.cards_added.append(picked)
            log.info("  Card reward: picked %s from %s", picked, offer.card_choices)
        else:
            log.info("  Card reward: skipped %s", offer.card_choices)

        if offer.potion is not None:
            if len(run_state.potions) < 3:
                run_state.add_potion(offer.potion)
                result.potions_gained.append(offer.potion)
                log.info("  Potion reward: %s (slots: %s)", offer.potion, run_state.potions)
            else:
                log.info("  Potion reward: %s discarded (no slot)", offer.potion)

        run_state.gold += offer.gold

        result.floors_cleared += 1

    result.victory = True
    result.final_hp = run_state.player_hp
    result.max_hp = run_state.player_max_hp
    return result


# =====================================================================
# Act 1 linear run (old fixed encounter list, for backwards compat)
# =====================================================================


def _run_act1_linear(
    planner_or_agent: BattlePlanner | BattleAgent,
    seed: int,
    character: Character,
    reward_rng: RNG,
    *,
    strategy_agent: BaseStrategyAgent,
) -> RunResult:
    """Run Act 1 with the old fixed linear encounter list (8 floors)."""
    encounter_list = scenarios.act1_encounters(seed)

    result = RunResult(
        victory=False,
        floors_cleared=0,
        total_floors=len(encounter_list),
        final_hp=character.player_hp,
        max_hp=character.player_max_hp,
        damage_taken_total=0,
        max_hp_gained_total=0,
    )

    died = False
    for floor_idx, (encounter_type, encounter_id) in enumerate(encounter_list):
        character.floor = floor_idx + 1
        result.encounter_types.append(encounter_type)

        combat_seed = seed * 1000 + floor_idx

        with _floor_span(floor_idx + 1, encounter_type, character) as span:
            combat = builder.build_combat(
                encounter_type, encounter_id, combat_seed, character=character,
            )

            damage = _run_battle(planner_or_agent, combat)
            result.damage_per_floor.append(damage)
            result.damage_taken_total += damage
            result.max_hp_gained_total += combat.max_hp_gained

            obs = combat.observe()
            span.set_attributes({
                "encounter_id": encounter_id,
                "damage_taken": damage,
                "max_hp_gained": combat.max_hp_gained,
                "survived": not obs.player_dead,
                "turns": obs.turn,
            })

            if obs.player_dead:
                log.info("FLOOR %d (%s/%s): DIED (damage=%d)",
                         floor_idx + 1, encounter_type, encounter_id, damage)
                character.player_hp = 0
                result.final_hp = 0
                died = True
            else:
                character.player_hp = obs.player_hp
                character.player_max_hp = obs.player_max_hp
                character.potions = list(combat._state.potions)
                _apply_relic_effects(character)
                result.final_hp = character.player_hp
                result.max_hp = character.player_max_hp

                log.info("FLOOR %d (%s/%s): WON (damage=%d, hp=%d/%d)",
                         floor_idx + 1, encounter_type, encounter_id, damage,
                         character.player_hp, character.player_max_hp)

                _apply_combat_rewards(
                    character, result, encounter_type, encounter_list, floor_idx,
                    combat_seed, reward_rng, strategy_agent,
                )

                result.floors_cleared += 1

        if died:
            return result

    result.victory = True
    result.final_hp = character.player_hp
    result.max_hp = character.player_max_hp
    return result


# =====================================================================
# Act 1 map-based run (new branching map)
# =====================================================================


def _run_act1_map(
    planner_or_agent: BattlePlanner | BattleAgent,
    seed: int,
    character: Character,
    reward_rng: RNG,
    *,
    strategy_agent: BaseStrategyAgent,
) -> RunResult:
    """Run Act 1 with a generated branching map (15 floors)."""
    sts_map = generate_act1_map(seed)
    encounter_rng = RNG(seed ^ 0xCAFE)

    from sts_env.run.encounter_queue import EncounterQueue
    encounter_queue = EncounterQueue(encounter_rng)

    path = strategy_agent.plan_route(sts_map, character, seed)
    total_floors = len(path)

    result = RunResult(
        victory=False,
        floors_cleared=0,
        total_floors=total_floors,
        final_hp=character.player_hp,
        max_hp=character.player_max_hp,
        damage_taken_total=0,
        max_hp_gained_total=0,
    )

    for step_idx, (floor_num, x_pos) in enumerate(path):
        node = sts_map.get_node(floor_num, x_pos)
        if node is None:
            log.warning("Path step %d: no node at (%d, %d), skipping", step_idx, floor_num, x_pos)
            continue

        character.floor = floor_num + 1
        room_type = node.room_type
        room_type_str = room_type.name.lower()

        died = False
        with _floor_span(floor_num + 1, room_type_str, character) as span:

            # --- REST rooms ---
            if room_type == RoomType.REST:
                rest_result = _execute_rest_choice(strategy_agent, character)
                if rest_result.choice == RestChoice.REST:
                    log.info("FLOOR %d REST: healed %d HP (hp=%d/%d)",
                             floor_num + 1, rest_result.hp_healed,
                             character.player_hp, character.player_max_hp)
                    span.set_attributes({"rest_choice": "rest", "hp_healed": rest_result.hp_healed})
                else:
                    log.info("FLOOR %d REST: upgraded %s (hp=%d/%d)",
                             floor_num + 1, rest_result.card_upgraded,
                             character.player_hp, character.player_max_hp)
                    span.set_attributes({"rest_choice": "upgrade", "card_upgraded": str(rest_result.card_upgraded)})
                result.encounter_types.append("rest")
                result.damage_per_floor.append(0)
                result.floors_cleared += 1
                result.final_hp = character.player_hp
                result.max_hp = character.player_max_hp

            # --- EVENT rooms ---
            elif room_type == RoomType.EVENT:
                event = random_act1_event(encounter_rng, character.seen_events)
                character.seen_events.append(event.event_id)
                log.info("FLOOR %d EVENT: %s", floor_num + 1, event.event_id)
                choice_idx = strategy_agent.pick_event_choice(event, character)
                desc = resolve_event(event.event_id, choice_idx, character, encounter_rng)
                log.info("  Event result: %s", desc)
                span.set_attributes({
                    "event_id": event.event_id,
                    "choice_idx": choice_idx,
                    "event_result": str(desc),
                })
                result.encounter_types.append("event")
                result.damage_per_floor.append(0)
                result.floors_cleared += 1

            # --- SHOP rooms ---
            elif room_type == RoomType.SHOP:
                log.info("FLOOR %d SHOP", floor_num + 1)
                shop_inv = generate_shop(encounter_rng, character)
                strategy_agent.shop(shop_inv, character)
                result.encounter_types.append("shop")
                result.damage_per_floor.append(0)
                result.floors_cleared += 1

            # --- TREASURE rooms ---
            elif room_type == RoomType.TREASURE:
                log.info("FLOOR %d TREASURE", floor_num + 1)
                tres = open_treasure(character, encounter_rng)
                log.info("  Found %d gold%s", tres.gold_found,
                         f" and {tres.relic_found}" if tres.relic_found else "")
                span.set_attributes({
                    "gold_found": tres.gold_found,
                    "relic_found": str(tres.relic_found) if tres.relic_found else "",
                })
                result.encounter_types.append("treasure")
                result.damage_per_floor.append(0)
                result.floors_cleared += 1

            else:
                # --- Combat rooms (MONSTER / ELITE / BOSS) ---
                encounter_id = get_encounter_for_room(room_type, encounter_queue)
                if encounter_id is None:
                    log.warning("FLOOR %d %s: no encounter assigned, skipping",
                                floor_num + 1, room_type.name)
                else:
                    encounter_type = room_type_str  # "monster", "elite", "boss"
                    result.encounter_types.append(encounter_type)

                    combat_seed = seed * 1000 + floor_num
                    combat = builder.build_combat(
                        encounter_type, encounter_id, combat_seed, character=character,
                    )

                    damage = _run_battle(planner_or_agent, combat)
                    result.damage_per_floor.append(damage)
                    result.damage_taken_total += damage
                    result.max_hp_gained_total += combat.max_hp_gained

                    obs = combat.observe()
                    span.set_attributes({
                        "encounter_id": encounter_id,
                        "damage_taken": damage,
                        "max_hp_gained": combat.max_hp_gained,
                        "survived": not obs.player_dead,
                        "turns": obs.turn,
                    })

                    if obs.player_dead:
                        log.info("FLOOR %d (%s/%s): DIED (damage=%d)",
                                 floor_num + 1, encounter_type, encounter_id, damage)
                        character.player_hp = 0
                        result.final_hp = 0
                        died = True
                    else:
                        character.player_hp = obs.player_hp
                        character.player_max_hp = obs.player_max_hp
                        character.potions = list(combat._state.potions)
                        builder.sync_combat_counters(character, combat)
                        _apply_relic_effects(character)
                        result.final_hp = character.player_hp
                        result.max_hp = character.player_max_hp

                        log.info("FLOOR %d (%s/%s): WON (damage=%d, hp=%d/%d)",
                                 floor_num + 1, encounter_type, encounter_id, damage,
                                 character.player_hp, character.player_max_hp)

                        if room_type == RoomType.BOSS:
                            _apply_boss_relic_reward(character, strategy_agent, reward_rng)

                        _apply_combat_rewards_simple(
                            character, result, encounter_type, combat_seed, reward_rng,
                            strategy_agent,
                            sts_map=sts_map,
                            current_position=(floor_num, x_pos),
                            remaining_path=path[step_idx + 1:],
                        )

                        result.floors_cleared += 1

        if died:
            return result

    result.victory = True
    result.final_hp = character.player_hp
    result.max_hp = character.player_max_hp
    return result


@contextmanager
def _floor_span(floor: int, room_type: str, character: Character):
    """Context manager that wraps a floor iteration in an MLflow child span."""
    with mlflow.start_span(name=f"floor_{floor}_{room_type}") as span:
        span.set_attributes({
            "floor": floor,
            "room_type": room_type,
            "hp_before": character.player_hp,
            "max_hp_before": character.player_max_hp,
            "gold_before": character.gold,
            "deck_size_before": len(character.deck),
        })
        try:
            yield span
        finally:
            span.set_attributes({
                "hp_after": character.player_hp,
                "max_hp_after": character.player_max_hp,
                "gold_after": character.gold,
                "deck_size_after": len(character.deck),
            })


# ---------------------------------------------------------------------
# Rest site execution
# ---------------------------------------------------------------------


@dataclass
class _RestExecResult:
    choice: RestChoice
    hp_healed: int = 0
    card_upgraded: str | None = None


def _execute_rest_choice(
    strategy_agent: BaseStrategyAgent,
    character: Character,
) -> _RestExecResult:
    """Ask the agent for a RestChoice, then execute it.

    Falls back to the alternative action when the chosen action has no
    valid effect (e.g. UPGRADE chosen but no unupgraded cards).
    """
    choice = strategy_agent.pick_rest_choice(character)

    if choice == RestChoice.UPGRADE:
        target = _best_upgrade_target(character)
        if target is not None:
            rest_upgrade(character, target)
            return _RestExecResult(RestChoice.UPGRADE, card_upgraded=target)
        # Nothing to upgrade — fall back to heal.
        healed = rest_heal(character)
        return _RestExecResult(RestChoice.REST, hp_healed=healed)

    healed = rest_heal(character)
    return _RestExecResult(RestChoice.REST, hp_healed=healed)


# ---------------------------------------------------------------------
# Reward application
# ---------------------------------------------------------------------


def _apply_combat_rewards(
    character: Character,
    result: RunResult,
    encounter_type: str,
    encounter_list: list,
    floor_idx: int,
    combat_seed: int,
    reward_rng: RNG,
    strategy_agent: BaseStrategyAgent,
) -> None:
    """Apply post-combat rewards (linear mode with remaining encounters)."""
    room = _RewardRoom.ELITE if encounter_type == "elite" else _RewardRoom.BOSS if encounter_type == "boss" else _RewardRoom.MONSTER
    offer, new_factor = rewards.roll_combat_reward_offer(
        reward_rng, room,
        card_rarity_factor=character.card_rarity_factor,
        event_bus=character.event_bus,
    )
    character.card_rarity_factor = new_factor
    remaining = encounter_list[floor_idx + 1:]

    picked = strategy_agent.pick_card(
        character, list(offer.card_choices), list(remaining), combat_seed,
    )
    if picked is not None:
        character.add_card(picked)
        result.cards_added.append(picked)
        log.info("  Card reward: picked %s from %s", picked, offer.card_choices)
    else:
        log.info("  Card reward: skipped %s", offer.card_choices)

    if offer.potion is not None:
        if len(character.potions) < character.max_potion_slots:
            character.add_potion(offer.potion)
            result.potions_gained.append(offer.potion)
            log.info("  Potion reward: %s (slots: %s)", offer.potion, character.potions)
        else:
            log.info("  Potion reward: %s discarded (no slot)", offer.potion)

    character.gold += offer.gold


def _upcoming_from_path(
    remaining_path: list[tuple[int, int]],
    sts_map,
) -> list[tuple[str, str]]:
    """Derive upcoming (type, id) encounter hints from the remaining path."""
    upcoming: list[tuple[str, str]] = []
    for floor_num, x_pos in remaining_path:
        node = sts_map.get_node(floor_num, x_pos) if sts_map is not None else None
        if node is None:
            continue
        room_name = node.room_type.name.lower()
        upcoming.append((room_name, ""))
    return upcoming


def _apply_combat_rewards_simple(
    character: Character,
    result: RunResult,
    encounter_type: str,
    combat_seed: int,
    reward_rng: RNG,
    strategy_agent: BaseStrategyAgent,
    *,
    sts_map=None,
    current_position: tuple[int, int] | None = None,
    remaining_path: list[tuple[int, int]] | None = None,
) -> None:
    """Apply post-combat rewards (map mode)."""
    room = _RewardRoom.ELITE if encounter_type == "elite" else _RewardRoom.BOSS if encounter_type == "boss" else _RewardRoom.MONSTER
    offer, new_factor = rewards.roll_combat_reward_offer(
        reward_rng, room,
        card_rarity_factor=character.card_rarity_factor,
        event_bus=character.event_bus,
    )
    character.card_rarity_factor = new_factor

    upcoming = _upcoming_from_path(remaining_path or [], sts_map)
    picked = strategy_agent.pick_card(
        character, list(offer.card_choices), upcoming, combat_seed,
        sts_map=sts_map,
        current_position=current_position,
    )

    if picked is not None:
        character.add_card(picked)
        result.cards_added.append(picked)
        log.info("  Card reward: picked %s from %s", picked, offer.card_choices)
    else:
        log.info("  Card reward: skipped %s", offer.card_choices)

    if offer.potion is not None:
        if len(character.potions) < character.max_potion_slots:
            character.add_potion(offer.potion)
            result.potions_gained.append(offer.potion)
            log.info("  Potion reward: %s (slots: %s)", offer.potion, character.potions)
        else:
            log.info("  Potion reward: %s discarded (no slot)", offer.potion)

    character.gold += offer.gold


def _apply_boss_relic_reward(
    character: Character,
    strategy_agent: BaseStrategyAgent,
    rng: RNG,
) -> None:
    """Offer the boss relic reward and let the strategy agent pick."""
    available = rewards.roll_boss_relic_choices(rng, owned=character.relics)
    if not available:
        return
    relic = strategy_agent.pick_boss_relic(character, available)
    if relic is None:
        log.info("  Boss relic reward: skipped %s", available)
        return
    character.add_relic(relic)
    log.info("  Boss relic reward: %s", relic)


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
    """Apply end-of-combat relic effects."""
    if character.has_relic("BurningBlood"):
        character.heal(6)
    # Orichalcum simplification — applied after combat instead of per-turn.
    if character.has_relic("Orichalcum"):
        character.heal(4)
