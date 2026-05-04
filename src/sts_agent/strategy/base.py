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

import copy
from typing import TYPE_CHECKING

from sts_env.combat.rng import RNG
from sts_env.run import relics as relics_mod
from sts_env.run.rooms import RestChoice, RestResult, _best_upgrade_target

if TYPE_CHECKING:
    from sts_env.run.character import Character
    from sts_env.run.encounter_queue import EncounterQueue
    from sts_env.run.events import EventSpec
    from sts_env.run.map import StSMap
    from sts_env.run.neow import NeowChoice, NeowOption
    from sts_env.run.shop import ShopInventory


class BaseStrategyAgent:
    """Random-valid-option agent. Subclass and override to specialise."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = RNG(seed if seed is not None else 0)
        self._encounter_queue: EncounterQueue | None = None
        self._hallway_seen: list[str] = []
        self._elites_seen: list[str] = []

    # ------------------------------------------------------------------
    # Encounter tracking (open-knowledge query support)
    # ------------------------------------------------------------------

    def set_encounter_tracking(
        self,
        encounter_queue: EncounterQueue,
        hallway_seen: list[str],
        elites_seen: list[str],
    ) -> None:
        """Store references to the orchestrator's encounter state.

        The orchestrator calls this once at run start.  Because *hallway_seen*
        and *elites_seen* are the **same list objects** the orchestrator appends
        to in-place, :meth:`get_possible_encounters` always reflects the latest
        state without the orchestrator needing to push updates.
        """
        self._encounter_queue = encounter_queue
        self._hallway_seen = hallway_seen
        self._elites_seen = elites_seen

    def get_possible_encounters(self) -> dict | None:
        """Compute possible remaining encounters from open knowledge.

        Returns ``None`` when encounter tracking has not been initialised
        (e.g. in unit tests that bypass the orchestrator).
        """
        if self._encounter_queue is None:
            return None
        return self._encounter_queue.possible_encounters(
            self._hallway_seen, self._elites_seen,
        )

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

    def pick_rest_choice(
        self,
        character: Character,
        **kwargs: object,
    ) -> RestResult:
        """Pick REST or UPGRADE uniformly at random, filtered by relics.

        If UPGRADE is chosen, selects target via heuristic priority order.
        """
        allowed: list[RestChoice] = []
        if relics_mod.can_rest(character.relics):
            allowed.append(RestChoice.REST)
        if relics_mod.can_upgrade(character.relics):
            allowed.append(RestChoice.UPGRADE)
        if not allowed:
            return RestResult(choice=RestChoice.REST)
        choice: RestChoice = self.rng.choice(allowed)
        if choice == RestChoice.UPGRADE:
            target = _best_upgrade_target(character)
            if target is not None:
                return RestResult(choice=RestChoice.UPGRADE, card_upgraded=target)
        return RestResult(choice=RestChoice.REST)

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def pick_event_choice(
        self,
        event: EventSpec,
        character: Character,
        *,
        extra_context: str = "",
        reset_budget: bool = True,
        **kwargs: object,
    ) -> int:
        """Pick a random event branch index.

        ``extra_context`` carries open-knowledge context for the current
        decision (e.g. Match and Keep pool composition or grid state).
        ``reset_budget`` is accepted for protocol compatibility and ignored
        by the base implementation.
        """
        if not event.choices:
            return 0
        return self.rng.randint(0, len(event.choices) - 1)

    # ------------------------------------------------------------------
    # Card removal
    # ------------------------------------------------------------------

    def pick_card_to_remove(
        self, character: Character, **kwargs: object,
    ) -> str | None:
        """Pick a random card to remove from the deck. None = skip."""
        if not character.deck:
            return None
        return self.rng.choice(character.deck)

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

    def pick_potion_to_discard(
        self,
        character: Character,
        new_potion: str,
        **kwargs: object,
    ) -> str:
        """Choose which potion to discard when the bag is full.

        Uses evaluate_potions to estimate the value of each potion, then
        drops the least valuable one.  Falls back to declining the new potion
        if evaluation fails or returns empty costs.
        """
        all_candidates = list(character.potions) + [new_potion]
        if len(all_candidates) <= character.max_potion_slots:
            # Shouldn't happen — bag not actually full. Accept the new one.
            return new_potion

        # Try to evaluate potion costs
        try:
            from .evaluate_potions import evaluate_potions
            # No upcoming info available here — use empty list (heavy discounting)
            costs = evaluate_potions(character, [], 0)
            # Add cost for the new potion too (not in character.potions yet)
            if new_potion not in costs:
                # Temporarily add and evaluate
                char_copy = copy.deepcopy(character)
                char_copy.potions.append(new_potion)
                new_costs = evaluate_potions(char_copy, [], 0)
                costs[new_potion] = new_costs.get(new_potion, 0.0)
        except Exception:
            costs = {}

        # Find the least valuable candidate
        worst = min(all_candidates, key=lambda p: costs.get(p, 0.0))
        return worst
