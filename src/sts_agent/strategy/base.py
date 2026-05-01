"""Base strategy agent — random valid choice for every per-floor decision.

``BaseStrategyAgent`` owns one decision method per question the run loop asks
of strategy: which Neow blessing, which path through the map, which card to
take, what to do at a rest site, which event branch to walk, what to buy at
a shop, and which boss relic to keep.

The defaults are deliberately dumb: pick a uniformly random valid option.
Concrete subclasses (``SimStrategyAgent``, ``StrategyAgent``) override only
the methods they specialise.

Each agent owns a single ``RNG`` seeded at construction so that two
``BaseStrategyAgent(seed=42)`` instances produce identical sequences of
decisions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sts_env.combat.rng import RNG
from sts_env.run import relics as relics_mod
from sts_env.run.rooms import RestChoice

if TYPE_CHECKING:
    from sts_env.run.character import Character
    from sts_env.run.events import EventSpec
    from sts_env.run.map import StSMap
    from sts_env.run.neow import NeowChoice, NeowOption
    from sts_env.run.shop import ShopInventory


class BaseStrategyAgent:
    """Random-valid-option agent. Subclass and override to specialise."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = RNG(seed if seed is not None else 0)

    # ------------------------------------------------------------------
    # Neow's blessing
    # ------------------------------------------------------------------

    def pick_neow(self, options: list[NeowOption]) -> NeowChoice:
        """Pick a Neow blessing uniformly at random."""
        return self.rng.choice(options).choice

    # ------------------------------------------------------------------
    # Map route
    # ------------------------------------------------------------------

    def plan_route(
        self,
        sts_map: StSMap,
        character: Character,
        seed: int,
    ) -> list[tuple[int, int]]:
        """Walk a random valid path from a floor-0 start node to the boss.

        ``seed`` is accepted so concrete subclasses can derive per-encounter
        probe seeds from it; the base implementation ignores it and uses
        ``self.rng`` instead.
        """
        path: list[tuple[int, int]] = []
        floor0_nodes = sts_map.nodes.get(0, [])
        candidates = [n for n in floor0_nodes if n.edges]
        if not candidates:
            return path
        start = self.rng.choice(candidates)
        current: tuple[int, int] = (0, start.x)
        path.append(current)

        while True:
            f, x = current
            node = sts_map.get_node(f, x)
            if node is None or not node.edges:
                break
            next_coord = self.rng.choice(node.edges)
            path.append(next_coord)
            if next_coord[0] == 14:
                break
            current = next_coord
        return path

    # ------------------------------------------------------------------
    # Card rewards
    # ------------------------------------------------------------------

    def pick_card(
        self,
        character: Character,
        card_choices: list[str],
        upcoming_encounters: list[tuple[str, str]],
        seed: int,
        *,
        sts_map: StSMap | None = None,
        current_position: tuple[int, int] | None = None,
    ) -> str | None:
        """Pick a random card from the offered choices (no skip).

        ``seed`` is accepted for compatibility with subclasses that drive
        deterministic per-encounter simulations; the base ignores it.
        """
        if not card_choices:
            return None
        return self.rng.choice(card_choices)

    # ------------------------------------------------------------------
    # Rest sites
    # ------------------------------------------------------------------

    def pick_rest_choice(self, character: Character) -> RestChoice:
        """Pick REST or UPGRADE uniformly at random, filtered by relics."""
        allowed: list[RestChoice] = []
        if relics_mod.can_rest(character.relics):
            allowed.append(RestChoice.REST)
        if relics_mod.can_upgrade(character.relics):
            allowed.append(RestChoice.UPGRADE)
        if not allowed:
            return RestChoice.REST
        return self.rng.choice(allowed)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def pick_event_choice(self, event: EventSpec, character: Character) -> int:
        """Pick a random event branch index."""
        if not event.choices:
            return 0
        return self.rng.randint(0, len(event.choices) - 1)

    # ------------------------------------------------------------------
    # Shops
    # ------------------------------------------------------------------

    def shop(self, inventory: ShopInventory, character: Character) -> None:
        """Default: skip everything. Concrete agents override to spend gold."""
        return None

    # ------------------------------------------------------------------
    # Boss relic reward
    # ------------------------------------------------------------------

    def pick_boss_relic(
        self,
        character: Character,
        choices: list[str],
    ) -> str | None:
        """Pick a random boss relic from the offered list."""
        if not choices:
            return None
        return self.rng.choice(choices)
