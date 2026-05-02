"""Tests for the LLM strategy agent.

Covers:
- StrategyAgent._forced_pick (deterministic fallback)
- Tool functions (simulate_upcoming, try_card) with budget checker
- StrategyAgent.pick_card with mocked LLM
- run_act1 with mock strategy agent
- Timeout behavior
- Map-aware pick_card (StSMap + EncounterQueue integration)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, call, patch

from sts_env.run.character import Character
from sts_env.run.map import StSMap, MapNode, RoomType, generate_act1_map
from sts_agent.run import run_act1, RunResult
from sts_agent.battle.mcts import MCTSPlanner
from sts_agent.strategy.llm_agent import (
    CardPickSignature,
    StrategyAgent,
    _format_result,
    _make_tools,
)
from sts_agent.strategy.simulate import SimResult


# ---------------------------------------------------------------------------
# Minimal fake StSMap fixture
# ---------------------------------------------------------------------------


def _make_minimal_map() -> StSMap:
    """Two-floor map: floor 0 node at x=3 (MONSTER) → floor 1 node at x=3 (ELITE)."""
    node0 = MapNode(floor=0, x=3, room_type=RoomType.MONSTER, edges=[(1, 3)], parents=[])
    node1 = MapNode(floor=1, x=3, room_type=RoomType.ELITE, edges=[], parents=[3])
    return StSMap(nodes={0: [node0], 1: [node1]}, seed=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ironclad() -> Character:
    return Character.ironclad()


def _upcoming() -> list[tuple[str, str]]:
    """Typical remaining encounters after floor 1."""
    return [
        ("easy", "jaw_worm"),
        ("hard", "byrds"),
        ("elite", "Gremlin Nob"),
        ("easy", "slaver_blue"),
        ("hard", "spheric_guardian"),
        ("elite", "Lagavulin"),
        ("boss", "Slime Boss"),
    ]


# ---------------------------------------------------------------------------
# _forced_pick tests
# ---------------------------------------------------------------------------


class TestForcedPick:
    """Deterministic fallback: rare → uncommon → first."""

    def test_prefers_rare(self):
        choices = ["Anger", "Inflame", "Feed"]
        pick = StrategyAgent._forced_pick(_ironclad(), choices, _upcoming())
        assert pick == "Feed"  # rare

    def test_prefers_uncommon_when_no_rare(self):
        choices = ["Anger", "Carnage", "Flex"]
        pick = StrategyAgent._forced_pick(_ironclad(), choices, _upcoming())
        assert pick == "Carnage"  # uncommon

    def test_picks_first_when_all_common(self):
        choices = ["Anger", "Flex", "Cleave"]
        pick = StrategyAgent._forced_pick(_ironclad(), choices, _upcoming())
        assert pick == "Anger"  # first common

    def test_empty_choices(self):
        pick = StrategyAgent._forced_pick(_ironclad(), [], _upcoming())
        assert pick is None

    def test_single_choice(self):
        pick = StrategyAgent._forced_pick(_ironclad(), ["Bludgeon"], _upcoming())
        assert pick == "Bludgeon"  # rare

    def test_multiple_rares_picks_first(self):
        choices = ["Feed", "Bludgeon", "Anger"]
        pick = StrategyAgent._forced_pick(_ironclad(), choices, _upcoming())
        assert pick == "Feed"  # first rare


# ---------------------------------------------------------------------------
# _format_result tests
# ---------------------------------------------------------------------------


class TestFormatResult:
    def test_survived(self):
        r = SimResult(True, 10, 0, 70, 80, 5)
        s = _format_result("Test", r)
        assert "SURVIVED" in s
        assert "damage_taken=10" in s
        assert "final_hp=70/80" in s
        assert "turns=5" in s

    def test_died(self):
        r = SimResult(False, 80, 0, 0, 80, 3)
        s = _format_result("Test", r)
        assert "DIED" in s

    def test_healing(self):
        r = SimResult(True, -5, 3, 88, 83, 8)
        s = _format_result("Test", r)
        assert "damage_taken=-5" in s
        assert "max_hp_gained=3" in s


# ---------------------------------------------------------------------------
# Tool function tests (mocked simulations — fast)
# ---------------------------------------------------------------------------


class TestTools:
    """Test tool functions with mocked simulate_encounter/simulate_with_card."""

    def test_simulate_upcoming_valid(self):
        mock_result = SimResult(True, 10, 0, 70, 80, 5)
        with patch("sts_agent.strategy.llm_agent.simulate_encounter",
                   return_value=mock_result):
            tools = _make_tools(_ironclad(), [("easy", "jaw_worm")],
                                seed=42, budget_checker=lambda: None)
            result = tools[0]("0")
            assert "SURVIVED" in result
            assert "Baseline" in result
            assert "damage_taken=10" in result

    def test_simulate_upcoming_out_of_range(self):
        tools = _make_tools(_ironclad(), [("easy", "jaw_worm")], seed=42,
                            budget_checker=lambda: None)
        result = tools[0]("5")
        assert "Error" in result

    def test_try_card_valid(self):
        mock_result = SimResult(True, 8, 0, 72, 80, 4)
        with patch("sts_agent.strategy.llm_agent.simulate_with_card",
                   return_value=mock_result):
            tools = _make_tools(_ironclad(), [("easy", "jaw_worm")],
                                seed=42, budget_checker=lambda: None)
            result = tools[1]("Anger", "0")
            assert "Anger" in result
            assert "damage_taken=8" in result

    def test_budget_checker_called(self):
        calls = []
        checker = lambda: calls.append(True)
        mock_result = SimResult(True, 5, 0, 75, 80, 3)
        with patch("sts_agent.strategy.llm_agent.simulate_encounter",
                   return_value=mock_result):
            tools = _make_tools(_ironclad(), [("easy", "jaw_worm")], seed=42,
                                budget_checker=checker)
            tools[0]("0")
        assert len(calls) == 1

    def test_budget_timeout_propagates(self):
        def raise_timeout():
            raise TimeoutError("budget exceeded")

        tools = _make_tools(_ironclad(), [("easy", "jaw_worm")], seed=42,
                            budget_checker=raise_timeout)
        with pytest.raises(TimeoutError, match="budget exceeded"):
            tools[0]("0")

    def test_simulate_upcoming_negative_index(self):
        tools = _make_tools(_ironclad(), [("easy", "jaw_worm")], seed=42,
                            budget_checker=lambda: None)
        result = tools[0]("-1")
        assert "Error" in result

    def test_try_card_out_of_range(self):
        tools = _make_tools(_ironclad(), [("easy", "jaw_worm")], seed=42,
                            budget_checker=lambda: None)
        result = tools[1]("Anger", "99")
        assert "Error" in result


# ---------------------------------------------------------------------------
# StrategyAgent unit tests (mocked LLM — no network calls)
# ---------------------------------------------------------------------------


class TestStrategyAgent:
    """Test StrategyAgent behavior with mocked LLM and ensure_lm."""

    @pytest.fixture(autouse=True)
    def _mock_lm(self):
        """Patch ensure_lm so no real API calls happen."""
        with patch("sts_agent.strategy.llm_agent.ensure_lm"):
            yield

    def _mock_react(self, pick_value: str):
        """Create a mock ReAct that returns the given pick."""
        mock = MagicMock()
        mock_result = MagicMock()
        mock_result.pick = pick_value
        mock.return_value.return_value = mock_result
        return mock

    def test_timeout_triggers_forced_pick(self):
        """Budget of 0s forces immediate fallback."""
        agent = StrategyAgent(timeout_seconds=0)
        pick = agent.pick_card(_ironclad(), ["Anger", "Feed", "Inflame"],
                               _upcoming(), seed=42)
        assert pick == "Feed"

    def test_exception_triggers_forced_pick(self):
        """LM error falls back to forced pick."""
        agent = StrategyAgent()
        with patch("sts_agent.strategy.llm_agent.dspy.ReAct",
                   side_effect=RuntimeError("LM connection failed")):
            pick = agent.pick_card(
                _ironclad(), ["Anger", "Carnage", "Inflame"],
                _upcoming(), seed=42,
            )
            assert pick == "Carnage"  # uncommon forced pick

    def test_invalid_pick_falls_back(self):
        """LLM returning an invalid pick falls back to forced pick."""
        agent = StrategyAgent()
        with patch("sts_agent.strategy.llm_agent.dspy.ReAct",
                   self._mock_react("NonExistentCard")):
            pick = agent.pick_card(
                _ironclad(), ["Anger", "Inflame", "Flex"],
                _upcoming(), seed=42,
            )
            # Falls back to forced pick — Inflame is uncommon, picked first
            assert pick == "Inflame"

    def test_skip_returned_as_none(self):
        """LLM returning 'skip' yields None."""
        agent = StrategyAgent()
        with patch("sts_agent.strategy.llm_agent.dspy.ReAct",
                   self._mock_react("skip")):
            pick = agent.pick_card(
                _ironclad(), ["Anger", "Inflame", "Flex"],
                _upcoming(), seed=42,
            )
            assert pick is None

    def test_exact_match(self):
        """LLM returning exact card ID (case-insensitive) works."""
        agent = StrategyAgent()
        with patch("sts_agent.strategy.llm_agent.dspy.ReAct",
                   self._mock_react("anger")):
            pick = agent.pick_card(
                _ironclad(), ["Anger", "Inflame", "Flex"],
                _upcoming(), seed=42,
            )
            assert pick == "Anger"

    def test_fuzzy_match(self):
        """LLM returning a partial match still resolves."""
        agent = StrategyAgent()
        with patch("sts_agent.strategy.llm_agent.dspy.ReAct",
                   self._mock_react("I'll pick ShrugItOff")):
            pick = agent.pick_card(
                _ironclad(), ["Anger", "ShrugItOff", "Flex"],
                _upcoming(), seed=42,
            )
            assert pick == "ShrugItOff"

    def test_none_pick_treated_as_skip(self):
        """LLM returning empty string is treated as skip."""
        agent = StrategyAgent()
        with patch("sts_agent.strategy.llm_agent.dspy.ReAct",
                   self._mock_react("")):
            pick = agent.pick_card(
                _ironclad(), ["Anger", "Inflame", "Flex"],
                _upcoming(), seed=42,
            )
            assert pick is None


# ---------------------------------------------------------------------------
# run_act1 integration tests (slow — uses real MCTS)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRunAct1:
    """Integration tests for run_act1 (no LLM calls)."""

    def test_random_fallback_completes(self):
        """run_act1 with the default random agent completes 15 floors."""
        result = run_act1(MCTSPlanner(), seed=42)
        assert isinstance(result, RunResult)
        assert result.total_floors == 15  # map mode by default

    def test_with_mock_strategy_agent(self):
        """run_act1 with a mock strategy agent delegates card picks."""
        mock_agent = MagicMock()
        mock_agent.pick_card.return_value = "Anger"
        # plan_route still needs to return a real path; let the default base
        # walker do it via the mock by delegating to a real BaseStrategyAgent.
        from sts_agent.strategy import BaseStrategyAgent
        real_router = BaseStrategyAgent(seed=42)
        mock_agent.plan_route.side_effect = real_router.plan_route
        mock_agent.pick_neow.side_effect = real_router.pick_neow
        mock_agent.pick_rest_choice.side_effect = real_router.pick_rest_choice
        mock_agent.pick_event_choice.side_effect = real_router.pick_event_choice
        mock_agent.shop.side_effect = real_router.shop
        mock_agent.pick_boss_relic.side_effect = real_router.pick_boss_relic

        result = run_act1(MCTSPlanner(), seed=42, strategy_agent=mock_agent)

        if result.floors_cleared > 0 and mock_agent.pick_card.call_count > 0:
            assert all(c == "Anger" for c in result.cards_added)

    def test_deterministic_with_same_seed(self):
        """Same seed + same default agent produces same result."""
        from sts_agent.strategy import BaseStrategyAgent
        r1 = run_act1(MCTSPlanner(), seed=123, strategy_agent=BaseStrategyAgent(seed=123))
        r2 = run_act1(MCTSPlanner(), seed=123, strategy_agent=BaseStrategyAgent(seed=123))
        assert r1 == r2

    def test_result_fields_populated(self):
        """RunResult fields are properly populated (linear mode, 8 floors)."""
        result = run_act1(MCTSPlanner(), seed=42, use_map=False)
        assert len(result.damage_per_floor) == result.total_floors
        assert len(result.encounter_types) == result.total_floors
        assert result.total_floors == 8
        assert result.encounter_types[0] == "easy"
        assert result.encounter_types[-1] == "boss"


# ---------------------------------------------------------------------------
# CardPickSignature pool hints
# ---------------------------------------------------------------------------


class TestSignatureEncounterPools:
    """CardPickSignature docstring must list canonical encounter names."""

    def test_mentions_weak_pool_entries(self):
        doc = CardPickSignature.__doc__ or ""
        for enc in ("cultist", "jaw_worm"):
            assert enc in doc, f"Expected '{enc}' in CardPickSignature docstring"

    def test_mentions_elite_pool_entries(self):
        doc = CardPickSignature.__doc__ or ""
        assert "Gremlin Nob" in doc

    def test_mentions_boss_pool_entries(self):
        doc = CardPickSignature.__doc__ or ""
        assert "hexaghost" in doc


# ---------------------------------------------------------------------------
# Map-aware pick_card tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestPickCardMapAware:
    """pick_card should render a map_view and forward it to dspy.ReAct."""

    @pytest.fixture(autouse=True)
    def _mock_lm(self):
        with patch("sts_agent.strategy.llm_agent.ensure_lm"):
            yield

    def _mock_react(self, pick_value: str):
        mock = MagicMock()
        mock_result = MagicMock()
        mock_result.pick = pick_value
        mock.return_value.return_value = mock_result
        return mock

    def test_pick_card_passes_map_view_when_sts_map_provided(self):
        """When sts_map + current_position supplied, map_view comes from render_ascii."""
        agent = StrategyAgent()
        fake_map = _make_minimal_map()
        fake_map.render_ascii = MagicMock(return_value="FORWARD_ASCII_MAP")

        captured_kwargs: dict = {}

        def fake_react_factory(sig, tools, max_iters):
            inst = MagicMock()
            def call_side_effect(**kwargs):
                captured_kwargs.update(kwargs)
                r = MagicMock()
                r.pick = "Anger"
                return r
            inst.side_effect = call_side_effect
            return inst

        with patch("sts_agent.strategy.llm_agent.dspy.ReAct", side_effect=fake_react_factory):
            agent.pick_card(
                _ironclad(), ["Anger", "Inflame", "Flex"], [], seed=1,
                sts_map=fake_map, current_position=(0, 3),
            )

        assert "map_view" in captured_kwargs
        assert captured_kwargs["map_view"] == "FORWARD_ASCII_MAP"
        fake_map.render_ascii.assert_called_once_with(
            current_floor=0,
            current_x=3,
            reachable_only=True,
        )

    def test_pick_card_works_without_map(self):
        """Without sts_map the call still works and map_view is a stub string."""
        agent = StrategyAgent()

        captured_kwargs: dict = {}

        def fake_react_factory(sig, tools, max_iters):
            inst = MagicMock()
            def call_side_effect(**kwargs):
                captured_kwargs.update(kwargs)
                r = MagicMock()
                r.pick = "Anger"
                return r
            inst.side_effect = call_side_effect
            return inst

        with patch("sts_agent.strategy.llm_agent.dspy.ReAct", side_effect=fake_react_factory):
            agent.pick_card(_ironclad(), ["Anger", "Inflame", "Flex"], [], seed=1)

        assert "map_view" in captured_kwargs
        mv = captured_kwargs["map_view"]
        assert isinstance(mv, str)
        assert mv  # not empty — should be the "no map" stub message
