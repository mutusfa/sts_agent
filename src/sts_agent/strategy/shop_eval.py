"""Shop evaluation helpers — shared by Sim and LLM strategy agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sts_env.run.shop import (
    ShopInventory,
    buy_card,
    buy_potion,
    buy_relic,
    remove_card,
)

from .evaluate_potions import _eval_single, _select_targets
from .simulate import (
    SimDistribution,
    probe_encounter,
    probe_with_card,
    probe_with_relic,
    probe_without_card,
)

if TYPE_CHECKING:
    from sts_env.run.character import Character


@dataclass
class ShopScore:
    """Aggregated probe score for one shop action."""

    action: str
    total_expected_damage: float
    total_survival_rate: float
    worst_max_score: float
    price: int = 0
    potion_value: float = 0.0

    @property
    def score(self) -> tuple[bool, float, float, float, int]:
        """Comparison key — higher is better."""
        return (
            self.total_survival_rate >= 1.0,
            -self.total_expected_damage,
            -self.worst_max_score,
            self.potion_value,
            -self.price,
        )


def format_shop_inventory(inventory: ShopInventory) -> str:
    """Human-readable shop inventory for LLM context."""
    lines = [f"Card removal: {inventory.remove_cost}g"]
    for i, (card_id, price) in enumerate(inventory.cards):
        if card_id:
            lines.append(f"  card[{i}]: {card_id} ({price}g)")
    for i, (potion_id, price) in enumerate(inventory.potions):
        if potion_id:
            lines.append(f"  potion[{i}]: {potion_id} ({price}g)")
    for i, entry in enumerate(inventory.relics):
        if entry is not None:
            relic_id, price = entry
            lines.append(f"  relic[{i}]: {relic_id} ({price}g)")
    return "\n".join(lines)


def encounters_for_shop_probes(
    possible_encounters: dict | None,
    *,
    max_n: int = 5,
) -> list[tuple[str, str]]:
    """Build encounter targets from open-knowledge encounter dict."""
    if possible_encounters is None:
        return [("elite", ""), ("boss", "")]

    targets: list[tuple[str, str]] = []

    boss = possible_encounters.get("boss", "")
    if boss:
        targets.append(("boss", boss if isinstance(boss, str) else str(boss)))

    elites = possible_encounters.get("elite", [])
    if elites:
        targets.append(("elite", elites[0]))

    strong = possible_encounters.get("monster_strong", [])
    if strong:
        targets.append(("monster", strong[0]))

    weak = possible_encounters.get("monster_weak", [])
    if weak and len(targets) < max_n:
        targets.append(("monster", weak[0]))

    if not targets:
        targets = [("elite", ""), ("boss", "")]

    return targets[:max_n]


def _aggregate_probes(
    dists: list[SimDistribution],
) -> tuple[float, float, float]:
    if not dists:
        return (0.0, 1.0, 0.0)
    total_damage = sum(d.expected_damage for d in dists)
    avg_survival = sum(d.survival_rate for d in dists) / len(dists)
    worst = max(d.max_score for d in dists)
    return (total_damage, avg_survival, worst)


def _probe_action(
    probe_fn,
    character: Character,
    encounters: list[tuple[str, str]],
    seed: int,
    *,
    max_nodes: int,
    simulations: int,
) -> tuple[float, float, float]:
    dists: list[SimDistribution] = []
    for idx, (enc_type, enc_id) in enumerate(encounters):
        enc_seed = seed * 1000 + idx
        dists.append(
            probe_fn(
                character, enc_type, enc_id, enc_seed,
                max_nodes=max_nodes, simulations=simulations,
            )
        )
    return _aggregate_probes(dists)


def evaluate_shop_baseline(
    character: Character,
    encounters: list[tuple[str, str]],
    seed: int,
    *,
    max_nodes: int = 5000,
    simulations: int = 5000,
) -> ShopScore:
    """Score the current deck against upcoming encounters (leave shop)."""
    damage, survival, worst = _probe_action(
        probe_encounter,
        character,
        encounters,
        seed,
        max_nodes=max_nodes,
        simulations=simulations,
    )
    return ShopScore(
        action="leave",
        total_expected_damage=damage,
        total_survival_rate=survival,
        worst_max_score=worst,
    )


def evaluate_shop_option(
    action: str,
    character: Character,
    inventory: ShopInventory,
    encounters: list[tuple[str, str]],
    seed: int,
    *,
    max_nodes: int = 5000,
    simulations: int = 5000,
    possible_encounters: dict | None = None,
) -> ShopScore:
    """Score a single shop action without mutating *character*."""
    price = _action_price(action, inventory)

    if action == "leave":
        return evaluate_shop_baseline(
            character, encounters, seed,
            max_nodes=max_nodes, simulations=simulations,
        )

    if action.startswith("remove:"):
        card_id = action.split(":", 1)[1]
        damage, survival, worst = _probe_action(
            lambda ch, et, ei, s, **kw: probe_without_card(
                ch, card_id, et, ei, s, **kw,
            ),
            character,
            encounters,
            seed,
            max_nodes=max_nodes,
            simulations=simulations,
        )
        return ShopScore(
            action=action,
            total_expected_damage=damage,
            total_survival_rate=survival,
            worst_max_score=worst,
            price=inventory.remove_cost,
        )

    if action.startswith("buy_card:"):
        idx = int(action.split(":", 1)[1])
        card_id = inventory.cards[idx][0]
        assert card_id is not None
        damage, survival, worst = _probe_action(
            lambda ch, et, ei, s, **kw: probe_with_card(
                ch, card_id, et, ei, s, **kw,
            ),
            character,
            encounters,
            seed,
            max_nodes=max_nodes,
            simulations=simulations,
        )
        return ShopScore(
            action=action,
            total_expected_damage=damage,
            total_survival_rate=survival,
            worst_max_score=worst,
            price=price,
        )

    if action.startswith("buy_relic:"):
        idx = int(action.split(":", 1)[1])
        entry = inventory.relics[idx]
        assert entry is not None
        relic_id = entry[0]
        damage, survival, worst = _probe_action(
            lambda ch, et, ei, s, **kw: probe_with_relic(
                ch, relic_id, et, ei, s, **kw,
            ),
            character,
            encounters,
            seed,
            max_nodes=max_nodes,
            simulations=simulations,
        )
        return ShopScore(
            action=action,
            total_expected_damage=damage,
            total_survival_rate=survival,
            worst_max_score=worst,
            price=price,
        )

    if action.startswith("buy_potion:"):
        idx = int(action.split(":", 1)[1])
        potion_id = inventory.potions[idx][0]
        assert potion_id is not None
        targets = _select_targets(encounters, possible_encounters)
        potion_value = _eval_single(character, potion_id, targets, seed)
        baseline = evaluate_shop_baseline(
            character, encounters, seed,
            max_nodes=max_nodes, simulations=simulations,
        )
        return ShopScore(
            action=action,
            total_expected_damage=baseline.total_expected_damage,
            total_survival_rate=baseline.total_survival_rate,
            worst_max_score=baseline.worst_max_score,
            price=price,
            potion_value=potion_value,
        )

    raise ValueError(f"Unknown shop action: {action!r}")


def list_shop_candidates(
    inventory: ShopInventory,
    character: Character,
) -> list[str]:
    """Return affordable shop actions plus ``leave``."""
    candidates = ["leave"]

    if character.gold >= inventory.remove_cost and character.deck:
        for card_id in sorted(set(character.deck)):
            candidates.append(f"remove:{card_id}")

    for i, (card_id, price) in enumerate(inventory.cards):
        if card_id and character.gold >= price:
            candidates.append(f"buy_card:{i}")

    for i, entry in enumerate(inventory.relics):
        if entry is not None and character.gold >= entry[1]:
            candidates.append(f"buy_relic:{i}")

    for i, (potion_id, price) in enumerate(inventory.potions):
        if (
            potion_id
            and character.gold >= price
            and len(character.potions) < character.max_potion_slots
        ):
            candidates.append(f"buy_potion:{i}")

    return candidates


def _action_price(action: str, inventory: ShopInventory) -> int:
    if action.startswith("remove:"):
        return inventory.remove_cost
    if action.startswith("buy_card:"):
        idx = int(action.split(":", 1)[1])
        return inventory.cards[idx][1]
    if action.startswith("buy_relic:"):
        idx = int(action.split(":", 1)[1])
        entry = inventory.relics[idx]
        assert entry is not None
        return entry[1]
    if action.startswith("buy_potion:"):
        idx = int(action.split(":", 1)[1])
        return inventory.potions[idx][1]
    return 0


def parse_shop_actions(actions: str) -> list[str]:
    """Parse comma-separated shop action plan from LLM output."""
    return [
        part.strip()
        for part in actions.split(",")
        if part.strip() and part.strip().lower() != "leave"
    ]


def execute_shop_action(
    action: str,
    inventory: ShopInventory,
    character: Character,
) -> bool:
    """Execute one shop action. Returns True on success."""
    if action == "leave":
        return False

    if action.startswith("remove:"):
        card_id = action.split(":", 1)[1]
        return remove_card(character, card_id)

    if action.startswith("buy_card:"):
        idx = int(action.split(":", 1)[1])
        return buy_card(inventory, idx, character) is not None

    if action.startswith("buy_relic:"):
        idx = int(action.split(":", 1)[1])
        return buy_relic(inventory, idx, character) is not None

    if action.startswith("buy_potion:"):
        idx = int(action.split(":", 1)[1])
        return buy_potion(inventory, idx, character) is not None

    raise ValueError(f"Unknown shop action: {action!r}")
