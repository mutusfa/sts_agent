"""Tests for shop evaluation helpers."""

from __future__ import annotations

from sts_env.combat.rng import RNG
from sts_env.run.character import Character
from sts_env.run.shop import ShopInventory, buy_card, generate_shop

from sts_agent.strategy.shop_eval import (
    encounters_for_shop_probes,
    evaluate_shop_baseline,
    execute_shop_action,
    format_shop_inventory,
    list_shop_candidates,
    parse_shop_actions,
)


class TestFormatShopInventory:
    def test_includes_slots_and_prices(self):
        inv = ShopInventory(
            cards=[("Inflame", 75), (None, 75)],
            potions=[("FirePotion", 50)],
            relics=[("MealTicket", 150)],
            remove_cost=75,
        )
        text = format_shop_inventory(inv)
        assert "Card removal: 75g" in text
        assert "card[0]: Inflame (75g)" in text
        assert "potion[0]: FirePotion (50g)" in text
        assert "relic[0]: MealTicket (150g)" in text


class TestEncountersForShopProbes:
    def test_builds_from_possible_encounters(self):
        possible = {
            "monster_weak": ["cultist"],
            "monster_strong": ["gremlin_gang"],
            "elite": ["Lagavulin"],
            "boss": "hexaghost",
        }
        targets = encounters_for_shop_probes(possible)
        assert ("boss", "hexaghost") in targets
        assert ("elite", "Lagavulin") in targets

    def test_fallback_when_none(self):
        assert encounters_for_shop_probes(None) == [("elite", ""), ("boss", "")]


class TestParseShopActions:
    def test_parses_comma_separated_plan(self):
        assert parse_shop_actions("remove:Strike,buy_card:2,leave") == [
            "remove:Strike",
            "buy_card:2",
        ]

    def test_empty_plan(self):
        assert parse_shop_actions("leave") == []


class TestExecuteShopAction:
    def test_buy_card(self):
        ch = Character.ironclad()
        ch.gold = 200
        inv = ShopInventory(
            cards=[("Inflame", 75)],
            potions=[],
            relics=[None, None, None],
            remove_cost=75,
        )
        assert execute_shop_action("buy_card:0", inv, ch) is True
        assert "Inflame" in ch.deck
        assert ch.gold == 125
        assert inv.cards[0][0] is None

    def test_remove_card(self):
        ch = Character.ironclad()
        ch.gold = 100
        strike_count = ch.deck.count("Strike")
        inv = ShopInventory([], [], [None, None, None], 75)
        assert execute_shop_action("remove:Strike", inv, ch)
        assert ch.deck.count("Strike") == strike_count - 1
        assert ch.gold == 25


class TestListShopCandidates:
    def test_includes_affordable_actions(self):
        ch = Character.ironclad()
        ch.gold = 200
        inv = generate_shop(RNG(0), ch)
        candidates = list_shop_candidates(inv, ch)
        assert "leave" in candidates
        assert any(c.startswith("buy_card:") for c in candidates)


class TestEvaluateShopBaseline:
    def test_returns_shop_score(self):
        ch = Character.ironclad()
        score = evaluate_shop_baseline(
            ch,
            [("monster", "cultist")],
            seed=0,
            max_nodes=100,
            simulations=100,
        )
        assert score.action == "leave"
        assert score.total_survival_rate >= 0.0
