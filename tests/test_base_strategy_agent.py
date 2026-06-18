"""Tests for BaseStrategyAgent — random valid choice for every decision."""

from __future__ import annotations

import pytest

from sts_env.run.character import Character
from sts_env.run.events import EventChoice, EventSpec
from sts_env.run.map import generate_act1_map
from sts_env.run.neow import NeowChoice, roll_neow_options
from sts_env.combat.rng import RNG
from sts_env.run.rng_streams import RunRNG
from sts_env.run.rooms import RestChoice, RestResult
from sts_env.run.shop import generate_shop

from sts_agent.strategy import BaseStrategyAgent

from tests.map_helpers import walk_map


# ---------------------------------------------------------------------------
# pick_neow
# ---------------------------------------------------------------------------


class TestPickNeow:
    def test_returns_valid_choice(self):
        agent = BaseStrategyAgent(seed=0)
        options = roll_neow_options(RNG(0))
        pick = agent.pick_neow(options)
        assert pick in {opt.choice for opt in options}

    def test_is_deterministic_with_same_seed(self):
        options = roll_neow_options(RNG(0))
        a = BaseStrategyAgent(seed=42)
        b = BaseStrategyAgent(seed=42)
        assert a.pick_neow(options) == b.pick_neow(options)

    def test_different_seeds_can_diverge(self):
        """Across many trials, two different seeds should not always agree."""
        options = roll_neow_options(RNG(0))
        any_diff = False
        for s in range(50):
            a = BaseStrategyAgent(seed=s)
            b = BaseStrategyAgent(seed=s + 100)
            if a.pick_neow(options) != b.pick_neow(options):
                any_diff = True
                break
        assert any_diff


# ---------------------------------------------------------------------------
# Map fork routing
# ---------------------------------------------------------------------------


class TestWalkMap:
    def test_walks_to_boss(self):
        agent = BaseStrategyAgent(seed=1)
        sts_map = generate_act1_map(seed=1)
        path = walk_map(agent, sts_map, Character.ironclad(), seed=1)
        assert path, "walk_map returned empty path"
        # Boss row is floor 15 (MAP_HEIGHT - 1), after the 15 regular floors (0-14)
        assert path[-1][0] == 15

    def test_each_step_is_valid_edge(self):
        agent = BaseStrategyAgent(seed=7)
        sts_map = generate_act1_map(seed=7)
        path = walk_map(agent, sts_map, Character.ironclad(), seed=7)
        for prev, nxt in zip(path, path[1:]):
            prev_node = sts_map.get_node(*prev)
            assert prev_node is not None
            assert nxt in prev_node.edges, f"{nxt} not a valid edge from {prev}"

    def test_deterministic_with_same_agent_seed(self):
        sts_map = generate_act1_map(seed=3)
        a = BaseStrategyAgent(seed=99)
        b = BaseStrategyAgent(seed=99)
        ch = Character.ironclad()
        assert walk_map(a, sts_map, ch, seed=3) == walk_map(b, sts_map, ch, seed=3)


# ---------------------------------------------------------------------------
# pick_card
# ---------------------------------------------------------------------------


class TestPickCard:
    def test_returns_one_of_choices(self):
        agent = BaseStrategyAgent(seed=0)
        choices = ["Anger", "Inflame", "Feed"]
        pick = agent.pick_card(Character.ironclad(), choices, [], seed=0)
        assert pick in choices

    def test_returns_none_for_empty_choices(self):
        agent = BaseStrategyAgent(seed=0)
        assert agent.pick_card(Character.ironclad(), [], [], seed=0) is None

    def test_distribution_across_choices(self):
        """Many calls with the same choices should hit every option."""
        agent = BaseStrategyAgent(seed=0)
        choices = ["A", "B", "C"]
        seen: set[str] = set()
        for _ in range(60):
            seen.add(agent.pick_card(Character.ironclad(), choices, [], seed=0))
        assert seen == set(choices)


# ---------------------------------------------------------------------------
# pick_rest_choice
# ---------------------------------------------------------------------------


class TestPickRestChoice:
    def test_returns_valid_choice(self):
        agent = BaseStrategyAgent(seed=0)
        ch = Character.ironclad()
        for _ in range(20):
            result = agent.pick_rest_choice(ch)
            assert isinstance(result, RestResult)
            assert result.choice in {RestChoice.REST, RestChoice.UPGRADE}

    def test_respects_no_rest_relic(self):
        """CoffeeDripper forbids rest — agent must always upgrade."""
        agent = BaseStrategyAgent(seed=0)
        ch = Character.ironclad()
        ch.add_relic("CoffeeDripper")
        for _ in range(20):
            assert agent.pick_rest_choice(ch).choice == RestChoice.UPGRADE

    def test_respects_no_upgrade_relic(self):
        """FusionHammer forbids upgrade — agent must always rest."""
        agent = BaseStrategyAgent(seed=0)
        ch = Character.ironclad()
        ch.add_relic("FusionHammer")
        for _ in range(20):
            assert agent.pick_rest_choice(ch).choice == RestChoice.REST


# ---------------------------------------------------------------------------
# pick_event_choice
# ---------------------------------------------------------------------------


class TestPickEventChoice:
    def _fake_event(self, n: int) -> EventSpec:
        choices = [
            EventChoice(label=f"opt{i}", effect=lambda c, r, _i=i: f"chose {_i}")
            for i in range(n)
        ]
        return EventSpec(event_id="fake", description="x", choices=choices)

    def test_returns_in_range(self):
        agent = BaseStrategyAgent(seed=0)
        event = self._fake_event(3)
        for _ in range(30):
            idx = agent.pick_event_choice(event, Character.ironclad())
            assert 0 <= idx < 3

    def test_handles_single_choice(self):
        agent = BaseStrategyAgent(seed=0)
        event = self._fake_event(1)
        assert agent.pick_event_choice(event, Character.ironclad()) == 0

    def test_distribution_covers_all_choices(self):
        agent = BaseStrategyAgent(seed=0)
        event = self._fake_event(4)
        seen: set[int] = set()
        for _ in range(80):
            seen.add(agent.pick_event_choice(event, Character.ironclad()))
        assert seen == {0, 1, 2, 3}

    def test_accepts_extra_context_kwarg(self):
        """extra_context is accepted and silently ignored by the base impl."""
        agent = BaseStrategyAgent(seed=0)
        event = self._fake_event(3)
        # Must not raise — base agent ignores the context string
        idx = agent.pick_event_choice(event, Character.ironclad(), extra_context="pool: Strike x3")
        assert 0 <= idx < 3

    def test_accepts_reset_budget_kwarg(self):
        """reset_budget kwarg is accepted and silently ignored by the base impl."""
        agent = BaseStrategyAgent(seed=0)
        event = self._fake_event(2)
        idx = agent.pick_event_choice(event, Character.ironclad(), reset_budget=False)
        assert 0 <= idx < 2

    def test_no_pick_match_and_keep_pair_method(self):
        """pick_match_and_keep_pair must not exist on BaseStrategyAgent."""
        agent = BaseStrategyAgent(seed=0)
        assert not hasattr(agent, "pick_match_and_keep_pair")


# ---------------------------------------------------------------------------
# shop
# ---------------------------------------------------------------------------


class TestShop:
    def test_heuristic_spends_gold_when_affordable(self):
        """Base agent removes cards and buys affordable items."""
        agent = BaseStrategyAgent(seed=0)
        ch = Character.ironclad()
        ch.gold = 500
        ch.floor = 3
        gold_before = ch.gold
        deck_size_before = len(ch.deck)
        shop_inv = generate_shop(RunRNG(0), ch.floor, ch)

        agent.shop(shop_inv, ch)

        assert ch.gold < gold_before or len(ch.deck) != deck_size_before


# ---------------------------------------------------------------------------
# pick_boss_relic
# ---------------------------------------------------------------------------


class TestPickBossRelic:
    def test_returns_one_of_choices(self):
        agent = BaseStrategyAgent(seed=0)
        choices = ["RedSkull", "CentennialPuzzle", "Orichalcum"]
        pick = agent.pick_boss_relic(Character.ironclad(), choices)
        assert pick in choices

    def test_returns_none_for_empty(self):
        agent = BaseStrategyAgent(seed=0)
        assert agent.pick_boss_relic(Character.ironclad(), []) is None


# ---------------------------------------------------------------------------
# pick_card_to_transform
# ---------------------------------------------------------------------------


class TestPickCardToTransform:
    def test_returns_deck_card(self):
        agent = BaseStrategyAgent(seed=0)
        ch = Character.ironclad()
        assert ch.deck, "ironclad starts with cards"
        result = agent.pick_card_to_transform(ch)
        assert result in ch.deck

    def test_returns_none_for_empty_deck(self):
        agent = BaseStrategyAgent(seed=0)
        ch = Character.ironclad()
        ch.deck.clear()
        assert agent.pick_card_to_transform(ch) is None

    def test_accepts_sts_map_kwarg(self):
        """Orchestrator passes sts_map and current_position as kwargs."""
        agent = BaseStrategyAgent(seed=0)
        ch = Character.ironclad()
        result = agent.pick_card_to_transform(
            ch, sts_map=None, current_position=(5, 2)
        )
        assert result in ch.deck

    def test_deterministic_with_same_seed(self):
        ch = Character.ironclad()
        a = BaseStrategyAgent(seed=7)
        b = BaseStrategyAgent(seed=7)
        assert a.pick_card_to_transform(ch) == b.pick_card_to_transform(ch)


# ---------------------------------------------------------------------------
# pick_card_to_upgrade
# ---------------------------------------------------------------------------


class TestPickCardToUpgrade:
    def test_returns_upgradeable_card(self):
        agent = BaseStrategyAgent(seed=0)
        ch = Character.ironclad()
        upgradeable = [c for c in ch.deck if not c.endswith("+")]
        assert upgradeable, "ironclad starts with upgradeable cards"
        result = agent.pick_card_to_upgrade(ch)
        assert result in upgradeable

    def test_returns_none_for_fully_upgraded_deck(self):
        agent = BaseStrategyAgent(seed=0)
        ch = Character.ironclad()
        ch.deck = [c.rstrip("+") + "+" for c in ch.deck]
        assert agent.pick_card_to_upgrade(ch) is None

    def test_accepts_kwargs(self):
        agent = BaseStrategyAgent(seed=0)
        ch = Character.ironclad()
        result = agent.pick_card_to_upgrade(
            ch, sts_map=None, current_position=(3, 1)
        )
        assert result is None or result in ch.deck
