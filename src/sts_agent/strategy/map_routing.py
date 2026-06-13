"""Map path scoring, branch evaluation, and LLM map view formatting.

Design notes
------------
**Heuristic room values** (``room_heuristic_value``) are gold-equivalent rewards
normalized to monster fights.  Elite ends up slightly above rest (~2–2.5 vs 2),
which matches rough player intuition: elites are worth a bit more than healing,
but not so much that the router blindly chains them.

**Fork decisions** (``pick_fork_coord``) multiply that path score by a *local*
survival estimate: chained MCTS probes over the next few rooms, with HP
carryover and post-combat relic heals.  Pass ``current=None`` after Neow to
choose the first map column; pass a map coordinate at later forks.  Low HP
steers away from imminent elites; distant elites on a high-reward path are not
heavily penalized by the static score alone.

**Known limitation — greedy local risk vs path shape.**  Consecutive elites score
high on reward, but the prefix survival product may reject the branch even when
a *different* full path (slightly lower total reward, much safer offramps) would
have been preferable.  At the current fork that safer path is often already
inaccessible, so the router cannot “plan ahead” for elite density + escape
options the way a human does.

**Deferred: whole-path risk.**  A separate estimator over the entire route to
boss (elite clusters, rest/shop offramps, cumulative HP budget) would address
this, but needs run data — full-act MCTS sims are too slow and explode if we
try to model deck power growth room-by-room.  Not implemented; revisit when we
have lighter survival models or logged act outcomes.

**Deferred: gold ↔ HP conversion** — see todo ``gold-hp-conversion`` in the
map-path plan; crossroad survival currently substitutes for fine HP budgeting.

**Linear scenarios** use ``linear_scenario_map()`` — a single-column chain with
one scored path.  There is no separate “unscored linear” mode; fixed encounter
lists are still valid map input for ``build_scored_map_view()``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sts_env.run import relics as relics_mod
from sts_env.run.map import MapNode, RoomType, StSMap, get_encounter_for_room
from sts_env.run.rewards import (
    COMMON_POTIONS,
    UNCOMMON_POTIONS,
    _POTION_DROP_RATE,
    _RARE_CHANCE,
    _UNCOMMON_CHANCE,
    COMBAT_GOLD,
    Room,
)
from sts_env.run.rooms import rest_heal
from sts_env.run.shop import CARD_PRICES

from .simulate import SimDistribution, probe_encounter

if TYPE_CHECKING:
    from sts_env.combat.rng import RNG
    from sts_env.run.character import Character
    from sts_env.run.encounter_queue import EncounterQueue

    from .probe_data import ProbeCache

# Tunable heuristic constants
MONSTER_VALUE = 1.0
ELITE_PENALTY = 1.0
REST_VALUE = 2.0
FIRST_SHOP_VALUE = 2.0
EVENT_VALUE = 1.0
TREASURE_VALUE = 0.0
PREFIX_ROOM_COUNT = 3


class MapViewError(RuntimeError):
    """Raised when scored map context is required but cannot be built.

    Applies to branching and linear runs alike — a linear encounter list still
    yields a single-path map that must be scored like any other.
    """


_LINEAR_ENCOUNTER_ROOM: dict[str, RoomType] = {
    "easy": RoomType.MONSTER,
    "hard": RoomType.MONSTER,
    "elite": RoomType.ELITE,
    "boss": RoomType.BOSS,
}


def linear_scenario_map(
    encounters: list[tuple[str, str]],
    seed: int,
) -> StSMap:
    """Build a single-column map from a fixed encounter sequence."""
    nodes: dict[int, list[MapNode]] = {}
    for floor_idx, (enc_type, _enc_id) in enumerate(encounters):
        room_type = _LINEAR_ENCOUNTER_ROOM.get(enc_type, RoomType.MONSTER)
        edges: list[tuple[int, int]] = []
        if floor_idx + 1 < len(encounters):
            edges = [(floor_idx + 1, 0)]
        nodes[floor_idx] = [
            MapNode(floor=floor_idx, x=0, room_type=room_type, edges=edges),
        ]
    return StSMap(nodes=nodes, seed=seed)


_ROOM_SYMBOLS = {
    RoomType.MONSTER: "M",
    RoomType.ELITE: "E",
    RoomType.REST: "R",
    RoomType.BOSS: "B",
    RoomType.EVENT: "?",
    RoomType.SHOP: "$",
    RoomType.TREASURE: "T",
}

_COUNT_TYPES = {RoomType.ELITE, RoomType.REST, RoomType.EVENT, RoomType.SHOP}
_COMBAT_TYPES = {RoomType.MONSTER, RoomType.ELITE, RoomType.BOSS}

_RELIC_BASE_PRICES = {
    "common": 150,
    "uncommon": 250,
    "rare": 300,
}


@dataclass(frozen=True)
class ScoredPath:
    coords: list[tuple[int, int]]
    score: float
    symbols: list[str]
    counts: dict[RoomType, int]


@dataclass(frozen=True)
class ProbeConfig:
    max_nodes: int = 1000
    simulations: int = 300
    rollout_mode: str = "heuristic"


def _card_slot_ev(room: Room) -> float:
    rare_chance = _RARE_CHANCE[room]
    uncommon_chance = _UNCOMMON_CHANCE[room]
    common_chance = max(0, 100 - rare_chance - uncommon_chance)
    return (
        (rare_chance / 100.0) * CARD_PRICES["rare"]
        + (uncommon_chance / 100.0) * CARD_PRICES["uncommon"]
        + (common_chance / 100.0) * CARD_PRICES["common"]
    )


def _avg_potion_price() -> float:
    common = COMMON_POTIONS
    uncommon = UNCOMMON_POTIONS
    total = len(common) + len(uncommon)
    if total == 0:
        return 55.0
    return (
        len(common) * 50 + len(uncommon) * 75
    ) / total


def _elite_relic_ev() -> float:
    tier_probs = {
        "common": 0.50,
        "uncommon": 0.33,
        "rare": 0.17,
    }
    return sum(
        prob * _RELIC_BASE_PRICES[tier]
        for tier, prob in tier_probs.items()
    )


def room_reward_gold_ev(room_type: RoomType) -> float:
    """Expected gold-equivalent reward for one room visit."""
    if room_type == RoomType.MONSTER:
        reward_room = Room.MONSTER
    elif room_type == RoomType.ELITE:
        reward_room = Room.ELITE
    elif room_type == RoomType.BOSS:
        reward_room = Room.BOSS
    else:
        return 0.0

    gold = float(COMBAT_GOLD.get(reward_room, COMBAT_GOLD[Room.MONSTER]))
    cards = _card_slot_ev(reward_room)
    potion = _POTION_DROP_RATE * _avg_potion_price()
    relic = _elite_relic_ev() if room_type == RoomType.ELITE else 0.0
    return gold + cards + potion + relic


def room_heuristic_value(room_type: RoomType, *, shops_visited: int) -> float:
    if room_type == RoomType.MONSTER:
        return MONSTER_VALUE
    if room_type == RoomType.ELITE:
        monster_ev = room_reward_gold_ev(RoomType.MONSTER)
        elite_ev = room_reward_gold_ev(RoomType.ELITE)
        if monster_ev <= 0:
            return 0.0
        return (elite_ev / monster_ev) - ELITE_PENALTY
    if room_type == RoomType.REST:
        return REST_VALUE
    if room_type == RoomType.EVENT:
        return EVENT_VALUE
    if room_type == RoomType.SHOP:
        return FIRST_SHOP_VALUE if shops_visited == 0 else 0.0
    if room_type == RoomType.TREASURE:
        return TREASURE_VALUE
    return 0.0


def count_shops_visited(
    sts_map: StSMap,
    committed_path: list[tuple[int, int]],
) -> int:
    count = 0
    for floor_num, x_pos in committed_path:
        node = sts_map.get_node(floor_num, x_pos)
        if node is not None and node.room_type == RoomType.SHOP:
            count += 1
    return count


def _normalize_edge(floor: int, edge: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(edge, tuple):
        return edge
    return floor + 1, edge


def enumerate_forward_paths(
    sts_map: StSMap,
    start: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    """Enumerate all paths from *start* to the boss."""
    sf, sx = start
    paths: list[list[tuple[int, int]]] = []
    stack: list[tuple[int, int, list[tuple[int, int]]]] = [(sf, sx, [])]
    while stack:
        f, x, path = stack.pop()
        node = sts_map.get_node(f, x)
        if node is None:
            continue
        path = path + [(f, x)]
        if node.room_type == RoomType.BOSS or not node.edges:
            paths.append(path)
            continue
        for edge in node.edges:
            nf, nx = _normalize_edge(f, edge)
            stack.append((nf, nx, path))
    return paths


def _path_room_counts(
    sts_map: StSMap,
    path: list[tuple[int, int]],
    *,
    mark_at: tuple[int, int] | None = None,
) -> tuple[list[str], dict[RoomType, int]]:
    symbols: list[str] = []
    counts: dict[RoomType, int] = {}
    for floor_num, x_pos in path:
        node = sts_map.get_node(floor_num, x_pos)
        if node is None:
            continue
        if mark_at is not None and (floor_num, x_pos) == mark_at:
            sym = "@"
        else:
            sym = _ROOM_SYMBOLS.get(node.room_type, "?")
        symbols.append(sym)
        if node.room_type in _COUNT_TYPES:
            counts[node.room_type] = counts.get(node.room_type, 0) + 1
    return symbols, counts


def score_path(
    sts_map: StSMap,
    path: list[tuple[int, int]],
    *,
    shops_visited: int,
) -> float:
    total = 0.0
    shops_seen = shops_visited
    for floor_num, x_pos in path:
        node = sts_map.get_node(floor_num, x_pos)
        if node is None:
            continue
        total += room_heuristic_value(node.room_type, shops_visited=shops_seen)
        if node.room_type == RoomType.SHOP:
            shops_seen += 1
    return total


def top_paths(
    sts_map: StSMap,
    current_pos: tuple[int, int],
    *,
    shops_visited: int,
    k: int = 3,
) -> list[ScoredPath]:
    paths = enumerate_forward_paths(sts_map, current_pos)
    seen: set[tuple[tuple[int, int], ...]] = set()
    scored: list[ScoredPath] = []
    for path in paths:
        key = tuple(path)
        if key in seen:
            continue
        seen.add(key)
        symbols, counts = _path_room_counts(sts_map, path, mark_at=current_pos)
        scored.append(
            ScoredPath(
                coords=path,
                score=score_path(sts_map, path, shops_visited=shops_visited),
                symbols=symbols,
                counts=counts,
            )
        )
    scored.sort(key=lambda p: p.score, reverse=True)
    return scored[:k]


def require_top_paths(
    sts_map: StSMap,
    current_pos: tuple[int, int],
    *,
    shops_visited: int,
    k: int = 3,
) -> list[ScoredPath]:
    """Return top-*k* scored paths or raise if the map cannot be scored."""
    scored = top_paths(sts_map, current_pos, shops_visited=shops_visited, k=k)
    if not scored:
        raise MapViewError(
            f"no scored forward paths from {current_pos} "
            f"(map seed={sts_map.seed})"
        )
    return scored


def build_scored_map_view(
    sts_map: StSMap,
    character: Character,
    current_position: tuple[int, int],
    *,
    committed_path: list[tuple[int, int]] | None = None,
    k: int = 3,
) -> str:
    """Build fenced LLM map view with top-*k* scored paths."""
    shops = count_shops_visited(sts_map, committed_path or [])
    scored = require_top_paths(
        sts_map, current_position, shops_visited=shops, k=k,
    )
    view = format_map_view(sts_map, character, current_position, scored)
    return f"```text\n{view}\n```"


def provisional_remaining_path(
    sts_map: StSMap,
    current_pos: tuple[int, int],
    *,
    shops_visited: int,
) -> list[tuple[int, int]]:
    """Return suffix coords after *current_pos* on the highest-scored full path."""
    scored = require_top_paths(
        sts_map, current_pos, shops_visited=shops_visited, k=1,
    )
    full = scored[0].coords
    if not full or full[0] != current_pos:
        return full
    return full[1:]


def path_to_upcoming(
    sts_map: StSMap,
    path: list[tuple[int, int]],
) -> list[tuple[str, str]]:
    upcoming: list[tuple[str, str]] = []
    for floor_num, x_pos in path:
        node = sts_map.get_node(floor_num, x_pos)
        if node is None:
            continue
        upcoming.append((node.room_type.name.lower(), ""))
    return upcoming


def format_map_view(
    sts_map: StSMap,
    character: Character,
    current_pos: tuple[int, int],
    top: list[ScoredPath],
) -> str:
    if not top:
        raise MapViewError(
            f"format_map_view requires scored paths at {current_pos}"
        )
    floor_num, x_pos = current_pos
    node = sts_map.get_node(floor_num, x_pos)
    sym = _ROOM_SYMBOLS.get(node.room_type, "?") if node else "?"
    player_floor = character.floor if character.floor > 0 else floor_num + 1

    potion_text = ", ".join(character.potions) if character.potions else "none"
    lines = [
        f"Current position: floor {player_floor}, col {x_pos} [{sym}]",
        (
            f"HP: {character.player_hp}/{character.player_max_hp} | "
            f"Gold: {character.gold} | Potions: {potion_text}"
        ),
        "",
        "Top 3 paths to boss:",
        "",
    ]

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for idx, scored in enumerate(top):
        label = labels[idx] if idx < len(labels) else str(idx)
        seq = " → ".join(scored.symbols)
        stat_names = {
            RoomType.ELITE: "elites",
            RoomType.REST: "rests",
            RoomType.EVENT: "events",
            RoomType.SHOP: "shops",
        }
        stat_parts = [
            f"{stat_names[rt]}: {scored.counts[rt]}"
            for rt in (RoomType.ELITE, RoomType.REST, RoomType.EVENT, RoomType.SHOP)
            if scored.counts.get(rt, 0)
        ]
        stats = ", ".join(stat_parts) if stat_parts else "elites: 0, rests: 0, events: 0, shops: 0"
        lines.append(f"Path {label} ({len(scored.coords)} rooms): {seq}")
        lines.append(f"  {stats}  (score: {scored.score:.1f})")
        lines.append("")

    return "\n".join(lines).rstrip()


def _clone_character(character: Character) -> Character:
    return copy.deepcopy(character)


def _cached_probe(
    character: Character,
    encounter_type: str,
    encounter_id: str,
    seed: int,
    *,
    probe_cache: ProbeCache | None,
    config: ProbeConfig,
) -> SimDistribution:
    return probe_encounter(
        character,
        encounter_type,
        encounter_id,
        seed,
        max_nodes=config.max_nodes,
        simulations=config.simulations,
        probe_cache=probe_cache,
        rollout_mode=config.rollout_mode,  # type: ignore[arg-type]
    )


def _apply_combat_probe_outcome(
    clone: Character,
    dist: SimDistribution,
) -> None:
    clone.player_hp = max(0, clone.player_hp - int(dist.expected_damage))
    if clone.player_hp > 0:
        relics_mod.on_combat_end(clone)


def _apply_non_combat_room(clone: Character, room_type: RoomType) -> None:
    if room_type == RoomType.REST:
        rest_heal(clone)


def probe_prefix_survival(
    character: Character,
    sts_map: StSMap,
    prefix_coords: list[tuple[int, int]],
    seed: int,
    encounter_queue: EncounterQueue | None,
    *,
    probe_cache: ProbeCache | None = None,
    config: ProbeConfig | None = None,
) -> float:
    """Chained survival estimate for up to *prefix_coords* rooms with HP carryover."""
    if not prefix_coords:
        return 1.0

    cfg = config or ProbeConfig()
    clone = _clone_character(character)
    survival = 1.0

    for idx, (floor_num, x_pos) in enumerate(prefix_coords):
        node = sts_map.get_node(floor_num, x_pos)
        if node is None:
            continue

        if node.room_type in _COMBAT_TYPES:
            if encounter_queue is None:
                survival *= 0.5
                continue
            encounter_id = get_encounter_for_room(node.room_type, encounter_queue)
            if encounter_id is None:
                continue
            enc_seed = seed * 1000 + idx
            dist = _cached_probe(
                clone,
                node.room_type.name.lower(),
                encounter_id,
                enc_seed,
                probe_cache=probe_cache,
                config=cfg,
            )
            survival *= dist.survival_rate
            if survival <= 0.0:
                return 0.0
            _apply_combat_probe_outcome(clone, dist)
        else:
            _apply_non_combat_room(clone, node.room_type)

    return survival


def prefix_coords_for_edge(
    sts_map: StSMap,
    current: tuple[int, int],
    edge: tuple[int, int],
    *,
    shops_visited: int,
    prefix_len: int = PREFIX_ROOM_COUNT,
) -> list[tuple[int, int]]:
    """Next *prefix_len* rooms starting with *edge* on the best full path."""
    paths = enumerate_forward_paths(sts_map, current)
    matching = [p for p in paths if len(p) > 1 and p[1] == edge]
    if not matching:
        nf, nx = edge
        node = sts_map.get_node(nf, nx)
        return [edge] if node is not None else []
    best = max(
        matching,
        key=lambda p: score_path(sts_map, p, shops_visited=shops_visited),
    )
    start_idx = 1
    return best[start_idx:start_idx + prefix_len]


def best_path_score_from_edge(
    sts_map: StSMap,
    current: tuple[int, int],
    edge: tuple[int, int],
    *,
    shops_visited: int,
) -> float:
    paths = enumerate_forward_paths(sts_map, current)
    matching = [p for p in paths if len(p) > 1 and p[1] == edge]
    if not matching:
        return 0.0
    return max(
        score_path(sts_map, p, shops_visited=shops_visited)
        for p in matching
    )


def weighted_choice(rng: RNG, items: list[tuple[int, int]], weights: list[float]):
    total = sum(weights)
    if total <= 0:
        return rng.choice(items)
    roll = rng.random() * total
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if roll <= cumulative:
            return item
    return items[-1]


def _fork_heuristic_and_prefix(
    sts_map: StSMap,
    current: tuple[int, int] | None,
    option: tuple[int, int],
    *,
    shops_visited: int,
) -> tuple[float, list[tuple[int, int]]]:
    if current is None:
        scored = top_paths(sts_map, option, shops_visited=0, k=1)
        if not scored:
            return 0.0, [option]
        return scored[0].score, scored[0].coords[:PREFIX_ROOM_COUNT]
    heuristic = best_path_score_from_edge(
        sts_map, current, option, shops_visited=shops_visited,
    )
    prefix = prefix_coords_for_edge(
        sts_map, current, option, shops_visited=shops_visited,
    )
    return heuristic, prefix


def pick_fork_coord(
    sts_map: StSMap,
    character: Character,
    current: tuple[int, int] | None,
    seed: int,
    rng: RNG,
    encounter_queue: EncounterQueue | None,
    *,
    shops_visited: int = 0,
    probe_cache: ProbeCache | None = None,
    config: ProbeConfig | None = None,
) -> tuple[int, int]:
    """Pick the next map step at a fork (or first column when *current* is None)."""
    if current is None:
        floor0_nodes = sts_map.nodes.get(0, [])
        options = [(0, n.x) for n in floor0_nodes if n.edges]
        if not options:
            return (0, 0)
        effective_shops = 0
    else:
        f, x = current
        node = sts_map.get_node(f, x)
        if node is None or not node.edges:
            return current
        options = [_normalize_edge(f, e) for e in node.edges]
        effective_shops = shops_visited

    if len(options) == 1:
        return options[0]

    weights: list[float] = []
    for option in options:
        heuristic, prefix = _fork_heuristic_and_prefix(
            sts_map, current, option, shops_visited=effective_shops,
        )
        survival = probe_prefix_survival(
            character,
            sts_map,
            prefix,
            seed,
            encounter_queue,
            probe_cache=probe_cache,
            config=config,
        )
        weights.append(max(heuristic * survival, 0.0))

    return weighted_choice(rng, options, weights)
