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
from sts_env.run.map import generate_act1_map, get_encounter_for_room, RoomType
from sts_env.run.rooms import pick_rest_choice, RestChoice
from sts_env.run.events import random_act1_event, resolve_event
from sts_env.run.shop import generate_shop, buy_card, buy_potion, buy_relic, remove_worst_card
from sts_env.run.treasure import open_treasure
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
    rest_strategy: str = "heal_if_hurt",
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
        A :class:`~sts_agent.strategy.StrategyAgent` for card-pick and path decisions.
        If ``None``, falls back to *card_pick_strategy*.
    card_pick_strategy:
        Fallback when no strategy agent: ``"random"`` or ``"skip"``.
    potion_use_strategy:
        How to use potions: ``"immediate"`` or ``"never"``.
    rest_strategy:
        Rest site behaviour: ``"heal_if_hurt"``, ``"always_heal"``, or ``"always_upgrade"``.
    use_map:
        If True (default), generate a branching map and walk it.
        If False, use the old linear encounter list (backwards compat).

    Returns
    -------
    RunResult with full run statistics.
    """
    character = Character.ironclad()
    reward_rng = RNG(seed ^ 0xBEEF)

    if use_map:
        return _run_act1_map(
            planner_or_agent, seed, character, reward_rng,
            strategy_agent=strategy_agent,
            card_pick_strategy=card_pick_strategy,
            rest_strategy=rest_strategy,
        )
    else:
        return _run_act1_linear(
            planner_or_agent, seed, character, reward_rng,
            strategy_agent=strategy_agent,
            card_pick_strategy=card_pick_strategy,
        )


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

        # Won the combat — sync state back
        run_state.player_hp = obs.player_hp
        run_state.player_max_hp = obs.player_max_hp
        run_state.potions = list(combat._state.potions)
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
# Act 1 linear run (old fixed encounter list, for backwards compat)
# =====================================================================


def _run_act1_linear(
    planner_or_agent: BattlePlanner | BattleAgent,
    seed: int,
    character: Character,
    reward_rng: RNG,
    *,
    strategy_agent: object | None,
    card_pick_strategy: str,
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

    for floor_idx, (encounter_type, encounter_id) in enumerate(encounter_list):
        character.floor = floor_idx + 1
        result.encounter_types.append(encounter_type)

        combat_seed = seed * 1000 + floor_idx
        combat = builder.build_combat(
            encounter_type, encounter_id, combat_seed, character=character,
        )

        damage = _run_battle(planner_or_agent, combat)
        result.damage_per_floor.append(damage)
        result.damage_taken_total += damage
        result.max_hp_gained_total += combat.max_hp_gained

        obs = combat.observe()
        if obs.player_dead:
            log.info("FLOOR %d (%s/%s): DIED (damage=%d)",
                     floor_idx + 1, encounter_type, encounter_id, damage)
            result.final_hp = 0
            return result

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
            combat_seed, reward_rng, strategy_agent, card_pick_strategy,
        )

        result.floors_cleared += 1

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
    strategy_agent: object | None,
    card_pick_strategy: str,
    rest_strategy: str,
) -> RunResult:
    """Run Act 1 with a generated branching map (15 floors)."""
    sts_map = generate_act1_map(seed)
    encounter_rng = RNG(seed ^ 0xCAFE)

    # Choose a path through the map
    path = _pick_path(sts_map, character, strategy_agent, seed)
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

        # --- REST rooms ---
        if room_type == RoomType.REST:
            rest_result = pick_rest_choice(character, strategy=rest_strategy)
            if rest_result.choice == RestChoice.REST:
                log.info("FLOOR %d REST: healed %d HP (hp=%d/%d)",
                         floor_num + 1, rest_result.hp_healed,
                         character.player_hp, character.player_max_hp)
            else:
                log.info("FLOOR %d REST: upgraded %s (hp=%d/%d)",
                         floor_num + 1, rest_result.card_upgraded,
                         character.player_hp, character.player_max_hp)
            result.encounter_types.append("rest")
            result.damage_per_floor.append(0)
            result.floors_cleared += 1
            result.final_hp = character.player_hp
            result.max_hp = character.player_max_hp
            continue

        # --- EVENT rooms ---
        if room_type == RoomType.EVENT:
            event = random_act1_event(encounter_rng)
            log.info("FLOOR %d EVENT: %s", floor_num + 1, event.event_id)
            # Default strategy: pick first choice (simplified)
            choice_idx = _pick_event_choice(event, character, strategy_agent)
            desc = resolve_event(event.event_id, choice_idx, character, encounter_rng)
            log.info("  Event result: %s", desc)
            result.encounter_types.append("event")
            result.damage_per_floor.append(0)
            result.floors_cleared += 1
            continue

        # --- SHOP rooms ---
        if room_type == RoomType.SHOP:
            log.info("FLOOR %d SHOP", floor_num + 1)
            shop_inv = generate_shop(encounter_rng, character)
            _auto_shop(shop_inv, character, strategy_agent)
            result.encounter_types.append("shop")
            result.damage_per_floor.append(0)
            result.floors_cleared += 1
            continue

        # --- TREASURE rooms ---
        if room_type == RoomType.TREASURE:
            log.info("FLOOR %d TREASURE", floor_num + 1)
            tres = open_treasure(character, encounter_rng)
            log.info("  Found %d gold%s", tres.gold_found,
                     f" and {tres.relic_found}" if tres.relic_found else "")
            result.encounter_types.append("treasure")
            result.damage_per_floor.append(0)
            result.floors_cleared += 1
            continue

        # --- Combat rooms (MONSTER / ELITE / BOSS) ---
        encounter_id = get_encounter_for_room(room_type, encounter_rng)
        if encounter_id is None:
            log.warning("FLOOR %d %s: no encounter assigned, skipping", floor_num + 1, room_type.name)
            continue

        encounter_type = room_type.name.lower()  # "monster", "elite", "boss"
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
        if obs.player_dead:
            log.info("FLOOR %d (%s/%s): DIED (damage=%d)",
                     floor_num + 1, encounter_type, encounter_id, damage)
            result.final_hp = 0
            return result

        character.player_hp = obs.player_hp
        character.player_max_hp = obs.player_max_hp
        character.potions = list(combat._state.potions)
        _apply_relic_effects(character)
        result.final_hp = character.player_hp
        result.max_hp = character.player_max_hp

        log.info("FLOOR %d (%s/%s): WON (damage=%d, hp=%d/%d)",
                 floor_num + 1, encounter_type, encounter_id, damage,
                 character.player_hp, character.player_max_hp)

        # Boss relic reward (first boss kill)
        if room_type == RoomType.BOSS:
            _apply_boss_relic_reward(character, result, reward_rng)

        # Card / potion / gold rewards for combat rooms
        _apply_combat_rewards_simple(
            character, result, encounter_type, combat_seed, reward_rng,
            strategy_agent, card_pick_strategy,
        )

        result.floors_cleared += 1

    # Survived all floors
    result.victory = True
    result.final_hp = character.player_hp
    result.max_hp = character.player_max_hp
    return result


def _pick_path(
    sts_map,
    character: Character,
    strategy_agent: object | None,
    seed: int,
) -> list[tuple[int, int]]:
    """Choose a path through the map.

    If strategy_agent has a ``pick_path`` method, delegate to it.
    Otherwise, use a greedy heuristic: prefer Monster > Elite > Rest,
    and pick the leftmost available branch.
    """
    if strategy_agent is not None and hasattr(strategy_agent, "pick_path"):
        return strategy_agent.pick_path(sts_map, character, seed)

    # Greedy: walk from floor 0 to 14, at each step pick the "best" next node
    path: list[tuple[int, int]] = []

    # Start from the first reachable node on floor 0
    floor0_nodes = sts_map.nodes.get(0, [])
    start_node = next((n for n in floor0_nodes if n.edges), None)
    if not start_node:
        return path
    current = (0, start_node.x)
    path.append(current)

    # Priority: we want fights early (for card rewards), rest when hurt,
    # but for v1 just pick a random valid edge at each step.
    path_rng = RNG(seed ^ 0xDEAD)

    while True:
        f, x = current
        node = sts_map.get_node(f, x)
        if node is None or not node.edges:
            break
        # Pick a random edge (v1: no strategy)
        next_coord = path_rng.choice(node.edges)
        path.append(next_coord)
        if next_coord[0] == 14:
            break
        current = next_coord

    return path


def _apply_combat_rewards(
    character: Character,
    result: RunResult,
    encounter_type: str,
    encounter_list: list,
    floor_idx: int,
    combat_seed: int,
    reward_rng: RNG,
    strategy_agent: object | None,
    card_pick_strategy: str,
) -> None:
    """Apply post-combat rewards (linear mode with remaining encounters)."""
    is_elite = encounter_type == "elite"
    card_choices = rewards.roll_card_rewards(reward_rng, is_elite=is_elite)
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

    potion = rewards.roll_potion_reward(reward_rng)
    if potion is not None:
        if len(character.potions) < character.max_potion_slots:
            character.add_potion(potion)
            result.potions_gained.append(potion)
            log.info("  Potion reward: %s (slots: %s)", potion, character.potions)
        else:
            log.info("  Potion reward: %s discarded (no slot)", potion)

    gold_reward = 30 if is_elite else 20 if encounter_type == "boss" else 10
    character.gold += gold_reward


def _apply_combat_rewards_simple(
    character: Character,
    result: RunResult,
    encounter_type: str,
    combat_seed: int,
    reward_rng: RNG,
    strategy_agent: object | None,
    card_pick_strategy: str,
) -> None:
    """Apply post-combat rewards (map mode — no remaining encounters list)."""
    is_elite = encounter_type == "elite"
    card_choices = rewards.roll_card_rewards(reward_rng, is_elite=is_elite)

    # Use strategy agent if available, else fallback
    if strategy_agent is not None and hasattr(strategy_agent, "pick_card"):
        picked = strategy_agent.pick_card(character, card_choices, [], combat_seed)
    else:
        picked = _pick_card(card_choices, card_pick_strategy, reward_rng)

    if picked is not None:
        character.add_card(picked)
        result.cards_added.append(picked)
        log.info("  Card reward: picked %s from %s", picked, card_choices)
    else:
        log.info("  Card reward: skipped %s", card_choices)

    potion = rewards.roll_potion_reward(reward_rng)
    if potion is not None:
        if len(character.potions) < character.max_potion_slots:
            character.add_potion(potion)
            result.potions_gained.append(potion)
            log.info("  Potion reward: %s (slots: %s)", potion, character.potions)
        else:
            log.info("  Potion reward: %s discarded (no slot)", potion)

    gold_reward = 30 if is_elite else 20 if encounter_type == "boss" else 10
    character.gold += gold_reward


def _pick_event_choice(
    event: "EventSpec",
    character: Character,
    strategy_agent: object | None,
) -> int:
    """Pick an event choice. Delegates to strategy agent if available."""
    if strategy_agent is not None and hasattr(strategy_agent, "pick_event_choice"):
        return strategy_agent.pick_event_choice(event, character)
    # Default: pick the first (usually safest) choice
    return 0


def _auto_shop(
    inventory: "ShopInventory",
    character: Character,
    strategy_agent: object | None,
) -> None:
    """Auto-spend gold at a shop using simple heuristics.

    Priority: remove worst card > buy best affordable card > buy potion if slot free
    """
    if strategy_agent is not None and hasattr(strategy_agent, "shop"):
        strategy_agent.shop(inventory, character)
        return

    # Simple heuristic shopping:
    # 1. Remove worst card if affordable and deck > 10 cards
    if character.gold >= inventory.remove_cost and len(character.deck) > 10:
        remove_worst_card(character)
        log.info("  Shop: removed worst card for %d gold", inventory.remove_cost)

    # 2. Buy best affordable card (prefer uncommon/rare)
    for idx in range(len(inventory.cards) - 1, -1, -1):
        item = inventory.cards[idx]
        if item is None:
            continue
        card_id, price = item
        if character.gold >= price:
            bought = buy_card(inventory, idx, character)
            if bought:
                log.info("  Shop: bought %s for %d gold", bought, price)
                break

    # 3. Buy a potion if we have a free slot
    if len(character.potions) < character.max_potion_slots:
        for idx in range(len(inventory.potions)):
            item = inventory.potions[idx]
            if item is None:
                continue
            potion_id, price = item
            if character.gold >= price:
                bought = buy_potion(inventory, idx, character)
                if bought:
                    log.info("  Shop: bought potion %s for %d gold", bought, price)
                    break


def _apply_boss_relic_reward(
    character: Character,
    result: RunResult,
    rng: RNG,
) -> None:
    """Give a boss relic reward after defeating the boss."""
    _BOSS_RELICS = [
        "BurningBlood",  # already have it, but listed for completeness
        "RedSkull",
        "CentennialPuzzle",
        "JuzuBracelet",
        "Orichalcum",
        "CeramicFish",
    ]
    # Pick a relic the character doesn't already have
    available = [r for r in _BOSS_RELICS if not character.has_relic(r)]
    if available:
        relic = rng.choice(available)
        character.relics.append(relic)
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
    # Orichalcum: gain 4 block at start of each combat (modeled as +4 HP after combat)
    # Note: this is a simplification — real Orichalcum applies block at combat start
    if character.has_relic("Orichalcum"):
        character.heal(4)


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
