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
    CardInfo,
    CardPickSignature,
    CardRemoveSignature,
    EventPickSignature,
    RestPickSignature,
    ShopPickSignature,
    StandardContext,
    StrategyAgent,
    _card_info,
    _format_dist,
    _make_tools,
)
from sts_agent.strategy import llm_agent
from sts_agent.strategy.simulate import SimDistribution, _resolve_enc, _ACT1_POOLS


def _choices(*card_ids: str) -> list[CardInfo]:
    """Build a list[CardInfo] from card IDs for _forced_pick tests."""
    return [_card_info(c) for c in card_ids]


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
        pick = StrategyAgent._forced_pick(_choices("Anger", "Inflame", "Feed"))
        assert pick == "Feed"  # rare

    def test_prefers_uncommon_when_no_rare(self):
        pick = StrategyAgent._forced_pick(_choices("Anger", "Carnage", "Flex"))
        assert pick == "Carnage"  # uncommon

    def test_picks_first_when_all_common(self):
        pick = StrategyAgent._forced_pick(_choices("Anger", "Flex", "Cleave"))
        assert pick == "Anger"  # first common

    def test_empty_choices(self):
        pick = StrategyAgent._forced_pick([])
        assert pick is None

    def test_single_choice(self):
        pick = StrategyAgent._forced_pick(_choices("Bludgeon"))
        assert pick == "Bludgeon"  # rare

    def test_multiple_rares_picks_first(self):
        pick = StrategyAgent._forced_pick(_choices("Feed", "Bludgeon", "Anger"))
        assert pick == "Feed"  # first rare


# ---------------------------------------------------------------------------
# _format_dist tests
# ---------------------------------------------------------------------------


def _make_dist(**kwargs) -> SimDistribution:
    defaults = dict(mean_score=10.0, std_score=2.0, max_score=15.0, n=100, deaths=0, start_hp=80)
    defaults.update(kwargs)
    return SimDistribution(**defaults)


class TestFormatDist:
    def test_no_status_tokens(self):
        """_format_dist never emits SURVIVED/DIED."""
        s = _format_dist("Test", _make_dist())
        assert "SURVIVED" not in s
        assert "DIED" not in s

    def test_damage_and_hp_present(self):
        d = _make_dist(mean_score=10.0, std_score=2.0, start_hp=80)
        s = _format_dist("Test", d)
        assert "damage_taken=10±2" in s
        assert "final_hp=70±2/80" in s

    def test_max_hp_gained_shown_when_nonzero(self):
        d = _make_dist(max_hp_gained_mean=1.5, max_hp_gained_std=0.3)
        s = _format_dist("Test", d)
        assert "max_hp_gained=1.5±0.3" in s

    def test_max_hp_gained_omitted_when_zero(self):
        s = _format_dist("Test", _make_dist())
        assert "max_hp_gained" not in s


# ---------------------------------------------------------------------------
# _card_info / CardInfo tests
# ---------------------------------------------------------------------------


class TestCardInfo:
    def test_strike_fields(self):
        info = _card_info("Strike")
        assert info.card_id == "Strike"
        assert info.cost == 1
        assert info.card_type == "attack"
        assert info.rarity == "basic"
        assert info.effects == {"attack": 6}
        assert info.upgrade == {"attack": 3}
        assert info.custom_code is None

    def test_feed_has_custom_code(self):
        info = _card_info("Feed")
        assert info.custom_code is not None
        assert "gain" in info.custom_code

    def test_custom_code_presence(self):
        assert _card_info("Anger").custom_code is not None
        assert _card_info("IronWave").custom_code is None

    def test_iron_wave_effects(self):
        info = _card_info("IronWave")
        assert info.effects == {"attack": 5, "block": 5}
        assert info.upgrade == {"attack": 2, "block": 2}

    def test_multi_hit_card_shows_hits(self):
        info = _card_info("TwinStrike")
        assert info.effects.get("hits") == 2

    def test_strike_no_hits_field(self):
        # hits=1 is the default — should not appear in effects
        assert "hits" not in _card_info("Strike").effects


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _mock_dist(mean=10.0, std=2.0, start_hp=80, deaths=0, n=100) -> SimDistribution:
    return SimDistribution(
        mean_score=mean, std_score=std, max_score=mean + std,
        n=n, deaths=deaths, start_hp=start_hp,
    )


class TestTools:
    """Test tool functions with mocked probe_encounter/probe_with_card."""

    def test_simulate_upcoming_valid(self):
        with patch("sts_agent.strategy.llm_agent.probe_encounter",
                   return_value=_mock_dist(mean=10.0)):
            tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
            result = tools[0]("monster")
            assert "Baseline" in result
            assert "damage_taken=10±2" in result

    def test_simulate_upcoming_label_contains_room_type(self):
        with patch("sts_agent.strategy.llm_agent.probe_encounter",
                   return_value=_mock_dist()):
            tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
            result = tools[0]("elite")
            assert "elite" in result

    def test_try_card_valid(self):
        with patch("sts_agent.strategy.llm_agent.probe_with_card",
                   return_value=_mock_dist(mean=8.0)):
            tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
            result = tools[1]("Anger", "monster")
            assert "Anger" in result
            assert "damage_taken=8±2" in result

    def test_budget_checker_called(self):
        calls = []
        checker = lambda: calls.append(True)
        with patch("sts_agent.strategy.llm_agent.probe_encounter",
                   return_value=_mock_dist()):
            tools = _make_tools(_ironclad(), seed=42, budget_checker=checker)
            tools[0]("monster")
        assert len(calls) == 1

    def test_budget_timeout_returns_message(self):
        """When budget is exhausted, tools return a message instead of raising."""
        def raise_timeout():
            raise TimeoutError("budget exceeded")

        tools = _make_tools(_ironclad(), seed=42, budget_checker=raise_timeout)
        result = tools[0]("monster")
        assert "SIMULATION BUDGET EXHAUSTED" in result

    def test_sim_log_accumulates(self):
        """sim_log collects formatted results from tool calls."""
        log: list[str] = []
        with patch("sts_agent.strategy.llm_agent.probe_encounter",
                   return_value=_mock_dist()):
            tools = _make_tools(
                _ironclad(), seed=42, budget_checker=lambda: None, sim_log=log,
            )
            tools[0]("monster")
        with patch("sts_agent.strategy.llm_agent.probe_with_card",
                   return_value=_mock_dist()):
            tools = _make_tools(
                _ironclad(), seed=42, budget_checker=lambda: None, sim_log=log,
            )
            tools[1]("Anger", "elite")
        assert len(log) == 2
        assert "Baseline" in log[0]
        assert "Anger" in log[1]

    def test_sim_log_not_written_on_timeout(self):
        """When budget is exhausted, no entry is added to sim_log."""
        log: list[str] = []
        def raise_timeout():
            raise TimeoutError("budget exceeded")
        tools = _make_tools(
            _ironclad(), seed=42, budget_checker=raise_timeout, sim_log=log,
        )
        tools[0]("monster")
        assert len(log) == 0


# ---------------------------------------------------------------------------
# _resolve_enc tests
# ---------------------------------------------------------------------------


class TestResolveEnc:
    """_resolve_enc resolves empty enc_ids and maps room-type names to sim types."""

    def test_passthrough_when_enc_id_provided(self):
        """Non-empty enc_id is passed through unchanged."""
        assert _resolve_enc("easy", "cultist", 42) == ("easy", "cultist")
        assert _resolve_enc("elite", "Gremlin Nob", 0) == ("elite", "Gremlin Nob")

    def test_monster_samples_from_combined_pool(self):
        """'monster' with empty id returns a valid (easy|hard, enc_id) pair."""
        enc_type, enc_id = _resolve_enc("monster", "", 42)
        valid_pairs = _ACT1_POOLS["monster"]
        assert (enc_type, enc_id) in valid_pairs

    def test_elite_samples_from_elite_pool(self):
        enc_type, enc_id = _resolve_enc("elite", "", 42)
        assert (enc_type, enc_id) in _ACT1_POOLS["elite"]

    def test_boss_samples_from_boss_pool(self):
        enc_type, enc_id = _resolve_enc("boss", "", 42)
        assert (enc_type, enc_id) in _ACT1_POOLS["boss"]

    def test_easy_samples_from_easy_pool(self):
        enc_type, enc_id = _resolve_enc("easy", "", 42)
        assert (enc_type, enc_id) in _ACT1_POOLS["easy"]

    def test_hard_samples_from_hard_pool(self):
        enc_type, enc_id = _resolve_enc("hard", "", 42)
        assert (enc_type, enc_id) in _ACT1_POOLS["hard"]

    def test_deterministic_same_seed(self):
        """Same seed always returns the same encounter."""
        assert _resolve_enc("monster", "", 99) == _resolve_enc("monster", "", 99)

    def test_different_seeds_may_differ(self):
        """Different seeds can return different encounters (probabilistically)."""
        results = {_resolve_enc("monster", "", s) for s in range(20)}
        assert len(results) > 1, "Expected variety across seeds"

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown encounter type"):
            _resolve_enc("dragon", "", 42)


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

    def test_exception_triggers_forced_pick(self):
        """LM error falls back to forced pick."""
        agent = StrategyAgent()
        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2",
                   side_effect=RuntimeError("LM connection failed")):
            pick = agent.pick_card(
                _ironclad(), ["Anger", "Carnage", "Inflame"],
                _upcoming(), seed=42,
            )
            assert pick == "Carnage"  # uncommon forced pick

    def test_timeout_exhausted_then_llm_pick(self):
        """When tools exhaust budget, LLM still gets to make a final pick."""
        agent = StrategyAgent(timeout_seconds=0)
        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2",
                   self._mock_react("Feed")):
            pick = agent.pick_card(
                _ironclad(), ["Anger", "Feed", "Inflame"],
                _upcoming(), seed=42,
            )
            # LLM picks Feed despite budget being exhausted at tool level
            assert pick == "Feed"

    def test_timeout_exhausted_llm_fails_forced_pick(self):
        """When budget is exhausted AND LLM returns invalid pick → forced pick."""
        agent = StrategyAgent(timeout_seconds=0)
        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2",
                   self._mock_react("NonExistentCard")):
            pick = agent.pick_card(
                _ironclad(), ["Anger", "Inflame", "Flex"],
                _upcoming(), seed=42,
            )
            # Falls back to forced pick — Inflame is uncommon
            assert pick == "Inflame"

    def test_invalid_pick_falls_back(self):
        """LLM returning an invalid pick falls back to forced pick."""
        agent = StrategyAgent()
        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2",
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
        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2",
                   self._mock_react("skip")):
            pick = agent.pick_card(
                _ironclad(), ["Anger", "Inflame", "Flex"],
                _upcoming(), seed=42,
            )
            assert pick is None

    def test_exact_match(self):
        """LLM returning exact card ID (case-insensitive) works."""
        agent = StrategyAgent()
        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2",
                   self._mock_react("anger")):
            pick = agent.pick_card(
                _ironclad(), ["Anger", "Inflame", "Flex"],
                _upcoming(), seed=42,
            )
            assert pick == "Anger"

    def test_fuzzy_match(self):
        """LLM returning a partial match still resolves."""
        agent = StrategyAgent()
        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2",
                   self._mock_react("I'll pick ShrugItOff")):
            pick = agent.pick_card(
                _ironclad(), ["Anger", "ShrugItOff", "Flex"],
                _upcoming(), seed=42,
            )
            assert pick == "ShrugItOff"

    def test_none_pick_treated_as_skip(self):
        """LLM returning empty string is treated as skip."""
        agent = StrategyAgent()
        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2",
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
# _RunAgentAdapter delegation tests
# ---------------------------------------------------------------------------


class TestRunAgentAdapterDelegation:
    """Unit tests for _RunAgentAdapter kwargs forwarding."""

    def test_pick_rest_choice_forwards_kwargs(self):
        """`pick_rest_choice` forwards sts_map and current_position to the strategy."""
        from sts_agent.run import _RunAgentAdapter

        mock_planner = MagicMock()
        mock_strategy = MagicMock()
        adapter = _RunAgentAdapter(mock_planner, mock_strategy)

        character = MagicMock(spec=Character)
        fake_map = MagicMock()

        adapter.pick_rest_choice(character, sts_map=fake_map, current_position=(2, 1))

        mock_strategy.pick_rest_choice.assert_called_once_with(
            character, sts_map=fake_map, current_position=(2, 1)
        )


# ---------------------------------------------------------------------------
# CardPickSignature pool hints
# ---------------------------------------------------------------------------


class TestSignatureEncounterPools:
    """CardPickSignature must include the possible_encounters input field
    and reference it in the docstring so the LLM knows to use it."""

    def test_has_possible_encounters_field(self):
        assert "possible_encounters" in CardPickSignature.model_fields

    def test_docstring_mentions_encounter_id(self):
        doc = CardPickSignature.__doc__ or ""
        assert "encounter_id" in doc, "Expected 'encounter_id' in docstring (tools accept it)"

    def test_docstring_mentions_possible_encounters(self):
        doc = CardPickSignature.__doc__ or ""
        assert "possible_encounters" in doc, "Expected 'possible_encounters' in docstring"


class TestStandardContext:
    """Shared StandardContext base for all strategy dspy.Signatures."""

    _SHARED = ("character_state", "map_view", "possible_encounters", "deck_cards")

    def test_base_has_shared_input_fields(self):
        for name in self._SHARED:
            assert name in StandardContext.model_fields
        assert set(StandardContext.input_fields) == set(self._SHARED)
        assert StandardContext.output_fields == {}

    def test_concrete_signatures_subclass_standard_context(self):
        for sig in (
            CardPickSignature,
            EventPickSignature,
            CardRemoveSignature,
            RestPickSignature,
            ShopPickSignature,
        ):
            assert issubclass(sig, StandardContext)
            assert "possible_encounters" in sig.model_fields


# ---------------------------------------------------------------------------
# Map-aware pick_card tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestPickCardMapAware:
    """pick_card should render a map_view and forward it to dspy.ReActV2."""

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

    def test_pick_card_passes_card_infos_to_react(self):
        """card_choices passed to dspy.ReActV2 should be list[CardInfo] with correct ids."""
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

        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2", side_effect=fake_react_factory):
            agent.pick_card(_ironclad(), ["Anger", "Inflame", "Flex"], [], seed=1)

        choices = captured_kwargs["card_choices"]
        assert isinstance(choices, list)
        assert all(isinstance(c, CardInfo) for c in choices)
        assert [c.card_id for c in choices] == ["Anger", "Inflame", "Flex"]

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

        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2", side_effect=fake_react_factory):
            agent.pick_card(
                _ironclad(), ["Anger", "Inflame", "Flex"], [], seed=1,
                sts_map=fake_map, current_position=(0, 3),
            )

        assert "map_view" in captured_kwargs
        assert "FORWARD_ASCII_MAP" in captured_kwargs["map_view"]
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

        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2", side_effect=fake_react_factory):
            agent.pick_card(_ironclad(), ["Anger", "Inflame", "Flex"], [], seed=1)

        assert "map_view" in captured_kwargs
        mv = captured_kwargs["map_view"]
        assert isinstance(mv, str)
        assert mv  # not empty — should be the "no map" stub message


# ---------------------------------------------------------------------------
# Pool-aware probe detection tests
# ---------------------------------------------------------------------------


class TestToolsPoolDetection:
    """Tools run multi-enemy pool probes or single specific-encounter probes."""

    def _call_counts_and_sims(self, tool_fn, *args):
        """Call tool_fn(*args), return (result, list of simulations kwargs)."""
        calls = []

        def fake_probe(character, enc_type, enc_id, seed, *, max_nodes, simulations):
            calls.append(simulations)
            return _mock_dist()

        with patch("sts_agent.strategy.llm_agent.probe_encounter",
                   side_effect=fake_probe):
            result = tool_fn(*args)
        return result, calls

    def test_empty_encounter_id_triggers_pool(self):
        """No encounter_id → pool path: probe_encounter called multiple times."""
        tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
        _, calls = self._call_counts_and_sims(tools[0], "monster")
        assert len(calls) > 1

    def test_pool_encounter_uses_100_sims(self):
        """Pool path uses 100 sims (base_simulations) per enemy."""
        tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None,
                            max_simulations=100)
        _, calls = self._call_counts_and_sims(tools[0], "monster")
        assert all(n == 100 for n in calls)

    def test_pool_encounter_at_most_5_enemies(self):
        """Pool path samples at most 5 enemies even for the large monster pool."""
        tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
        _, calls = self._call_counts_and_sims(tools[0], "monster")
        assert len(calls) <= 5

    def test_elite_pool_runs_all_three_elites(self):
        """Elite pool has exactly 3 entries; all three should be run."""
        tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
        _, calls = self._call_counts_and_sims(tools[0], "elite")
        assert len(calls) == 3

    def test_specific_encounter_calls_probe_once(self):
        """Specific encounter_id → single probe_encounter call."""
        tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
        _, calls = self._call_counts_and_sims(tools[0], "easy", "cultist")
        assert len(calls) == 1

    def test_specific_encounter_uses_300_sims(self):
        """Specific encounter uses 300 sims for accuracy."""
        tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
        _, calls = self._call_counts_and_sims(tools[0], "easy", "cultist")
        assert calls[0] == 300

    def test_encounter_id_monster_treated_as_pool(self):
        """Passing 'monster' as encounter_id triggers pool path, not specific."""
        tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
        _, calls = self._call_counts_and_sims(tools[0], "monster", "monster")
        assert len(calls) > 1

    def test_encounter_id_elite_treated_as_pool(self):
        """Passing 'elite' as encounter_id triggers pool path."""
        tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
        _, calls = self._call_counts_and_sims(tools[0], "elite", "elite")
        assert len(calls) == 3

    def test_pool_result_contains_enemy_names(self):
        """Pool probe output contains individual enemy IDs from the sampled pool."""
        tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
        result, _ = self._call_counts_and_sims(tools[0], "elite")
        assert any(name in result for name in ("Gremlin Nob", "Lagavulin", "Three Sentries"))

    def test_pool_result_header_mentions_pool_type(self):
        """Pool probe output header contains the pool name."""
        tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
        result, _ = self._call_counts_and_sims(tools[0], "elite")
        assert "elite" in result

    def test_try_card_pool_path(self):
        """try_card also dispatches through pool path when no specific encounter."""
        tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
        calls = []

        def fake_probe_card(character, card_id, enc_type, enc_id, seed, *, max_nodes, simulations):
            calls.append(simulations)
            return _mock_dist()

        with patch("sts_agent.strategy.llm_agent.probe_with_card",
                   side_effect=fake_probe_card):
            tools[1]("Anger", "monster")

        assert len(calls) > 1
        assert all(n == 100 for n in calls)

    def test_try_card_specific_uses_300_sims(self):
        """try_card with specific encounter uses 300 sims."""
        tools = _make_tools(_ironclad(), seed=42, budget_checker=lambda: None)
        calls = []

        def fake_probe_card(character, card_id, enc_type, enc_id, seed, *, max_nodes, simulations):
            calls.append(simulations)
            return _mock_dist()

        with patch("sts_agent.strategy.llm_agent.probe_with_card",
                   side_effect=fake_probe_card):
            tools[1]("Anger", "easy", "cultist")

        assert len(calls) == 1
        assert calls[0] == 300


# ---------------------------------------------------------------------------
# Unified tool factory tests (TDD — these fail before the refactor)
# ---------------------------------------------------------------------------


class TestUnifiedToolFactory:
    """_make_tools is the sole tool factory; old separate factories are gone."""

    def test_make_tools_returns_five_tools(self):
        tools = _make_tools(_ironclad(), budget_checker=lambda: None)
        assert len(tools) == 5

    def test_make_tools_tool_names(self):
        tools = _make_tools(_ironclad(), budget_checker=lambda: None)
        names = [t.__name__ for t in tools]
        assert names == [
            "simulate_upcoming",
            "try_card",
            "try_upgrade",
            "try_remove_card",
            "try_rest",
        ]

    def test_make_rest_tools_does_not_exist(self):
        assert not hasattr(llm_agent, "_make_rest_tools")

    def test_make_card_removal_tools_does_not_exist(self):
        assert not hasattr(llm_agent, "_make_card_removal_tools")

    def test_try_upgrade_tool_returns_string(self):
        """try_upgrade (index 2) runs and returns a formatted result string."""
        tools = _make_tools(_ironclad(), budget_checker=lambda: None)
        with patch("sts_agent.strategy.llm_agent.probe_with_upgrade",
                   return_value=_mock_dist(mean=5.0)):
            result = tools[2]("Strike", "monster", "cultist")
        assert "Strike" in result
        assert "damage_taken=5±2" in result

    def test_try_remove_card_is_last_tool(self):
        """try_remove_card is at index 3."""
        tools = _make_tools(_ironclad(), budget_checker=lambda: None)
        assert tools[3].__name__ == "try_remove_card"


# ---------------------------------------------------------------------------
# pick_event_choice ReAct tests (TDD — fails before the refactor)
# ---------------------------------------------------------------------------


class TestPickEventChoiceUsesReAct:
    """pick_event_choice must use dspy.ReActV2 (not dspy.Predict)."""

    @pytest.fixture(autouse=True)
    def _mock_lm(self):
        with patch("sts_agent.strategy.llm_agent.ensure_lm"):
            yield

    def _make_event(self, n_choices: int = 2):
        event = MagicMock()
        event.choices = [MagicMock(label=f"Option {i}") for i in range(n_choices)]
        event.event_id = "test_event"
        event.description = "A test event."
        event.event_encounters = []
        return event

    def test_pick_event_choice_uses_react_not_predict(self):
        agent = StrategyAgent()
        event = self._make_event()

        react_result = MagicMock()
        react_result.choice_index = "0"

        react_instance = MagicMock(return_value=react_result)
        react_class = MagicMock(return_value=react_instance)

        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2", react_class):
            with patch("sts_agent.strategy.llm_agent.dspy.Predict") as mock_predict:
                agent.pick_event_choice(event, _ironclad())
                react_class.assert_called()
                mock_predict.assert_not_called()

    def test_pick_event_choice_returns_valid_index(self):
        agent = StrategyAgent()
        event = self._make_event(n_choices=3)

        react_result = MagicMock()
        react_result.choice_index = "1"
        react_instance = MagicMock(return_value=react_result)
        react_class = MagicMock(return_value=react_instance)

        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2", react_class):
            idx = agent.pick_event_choice(event, _ironclad())

        assert idx == 1


# ---------------------------------------------------------------------------
# pick_event_choice passes event_encounters kwarg to ReAct
# ---------------------------------------------------------------------------


class TestPickEventChoicePassesEventEncounters:
    """pick_event_choice must forward possible_encounters as event_encounters to ReActV2."""

    @pytest.fixture(autouse=True)
    def _mock_lm(self):
        with patch("sts_agent.strategy.llm_agent.ensure_lm"):
            yield

    def _make_event(self, possible_encounters=()):
        event = MagicMock()
        event.choices = [MagicMock(label="Option 0"), MagicMock(label="Option 1")]
        event.event_id = "test_event"
        event.description = "A test event."
        event.possible_encounters = possible_encounters
        return event

    def _capture_kwargs(self):
        captured: dict = {}

        def fake_react_factory(sig, tools, max_iters):
            inst = MagicMock()

            def call_side_effect(**kwargs):
                captured.update(kwargs)
                r = MagicMock()
                r.choice_index = "0"
                return r

            inst.side_effect = call_side_effect
            return inst

        return captured, fake_react_factory

    def test_combat_event_passes_encounter_ids(self):
        """Dead Adventurer-style event: encounter IDs forwarded as flat list."""
        agent = StrategyAgent()
        event = self._make_event(
            possible_encounters=("Three Sentries", "Gremlin Nob", "lagavulin_event")
        )
        captured, factory = self._capture_kwargs()

        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2", side_effect=factory):
            agent.pick_event_choice(event, _ironclad())

        assert captured["event_encounters"] == [
            "Three Sentries",
            "Gremlin Nob",
            "lagavulin_event",
        ]

    def test_non_combat_event_passes_empty_list(self):
        """Events with no combat encounters forward an empty list."""
        agent = StrategyAgent()
        event = self._make_event(possible_encounters=())
        captured, factory = self._capture_kwargs()

        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2", side_effect=factory):
            agent.pick_event_choice(event, _ironclad())

        assert captured["event_encounters"] == []


# ---------------------------------------------------------------------------
# pick_event_choice extra_context and reset_budget
# ---------------------------------------------------------------------------


class TestPickEventChoiceExtraContext:
    """extra_context must appear in event_description forwarded to ReActV2."""

    @pytest.fixture(autouse=True)
    def _mock_lm(self):
        with patch("sts_agent.strategy.llm_agent.ensure_lm"):
            yield

    def _make_event(self):
        event = MagicMock()
        event.choices = [MagicMock(label="Option 0"), MagicMock(label="Option 1")]
        event.event_id = "Match and Keep - Pick First"
        event.description = "Pick a slot."
        event.possible_encounters = ()
        return event

    def _capture_kwargs(self):
        captured: dict = {}

        def fake_react_factory(sig, tools, max_iters):
            inst = MagicMock()

            def call_side_effect(**kwargs):
                captured.update(kwargs)
                r = MagicMock()
                r.choice_index = "0"
                return r

            inst.side_effect = call_side_effect
            return inst

        return captured, fake_react_factory

    def test_extra_context_appended_to_event_description(self):
        """When extra_context is provided it must appear in event_description."""
        agent = StrategyAgent()
        event = self._make_event()
        captured, factory = self._capture_kwargs()

        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2", side_effect=factory):
            agent.pick_event_choice(event, _ironclad(), extra_context="pool: Strike x3")

        assert "pool: Strike x3" in captured["event_description"]

    def test_empty_extra_context_not_appended(self):
        """When extra_context is empty, event_description should not have trailing Context."""
        agent = StrategyAgent()
        event = self._make_event()
        captured, factory = self._capture_kwargs()

        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2", side_effect=factory):
            agent.pick_event_choice(event, _ironclad(), extra_context="")

        assert "Context:" not in captured["event_description"]

    def test_reset_budget_false_does_not_reset_start_time(self):
        """reset_budget=False must not overwrite _start_time."""
        agent = StrategyAgent()
        event = self._make_event()
        sentinel = 12345.0
        agent._start_time = sentinel
        agent._timed_out = False

        react_result = MagicMock()
        react_result.choice_index = "0"
        react_instance = MagicMock(return_value=react_result)
        react_class = MagicMock(return_value=react_instance)

        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2", react_class):
            agent.pick_event_choice(event, _ironclad(), reset_budget=False)

        assert agent._start_time == sentinel


class TestShop:
    """StrategyAgent.shop executes LLM action plan."""

    @pytest.fixture(autouse=True)
    def _mock_lm(self):
        with patch("sts_agent.strategy.llm_agent.ensure_lm"):
            yield

    def test_shop_executes_parsed_actions(self):
        from sts_env.combat.rng import RNG
        from sts_env.run.shop import generate_shop

        agent = StrategyAgent()
        ch = Character.ironclad()
        ch.gold = 300
        inv = generate_shop(RNG(0), ch)

        card_idx = next(
            i for i, (cid, price) in enumerate(inv.cards)
            if cid and price <= ch.gold
        )
        card_id = inv.cards[card_idx][0]

        def fake_react_factory(sig, tools, max_iters):
            inst = MagicMock()

            def call_side_effect(**kwargs):
                r = MagicMock()
                r.actions = f"buy_card:{card_idx},leave"
                return r

            inst.side_effect = call_side_effect
            return inst

        with patch("sts_agent.strategy.llm_agent.dspy.ReActV2", side_effect=fake_react_factory):
            agent.shop(inv, ch)

        assert card_id in ch.deck
        assert ch.gold < 300

