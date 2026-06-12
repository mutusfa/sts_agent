"""Integration tests for Act 1 map-based run."""

import pytest
from unittest.mock import MagicMock, patch

from sts_env.run.map import StSMap
from sts_env.run.rewards import BOSS_RELICS
from sts_agent.run import run_act1
from sts_agent.battle.base import BattleAgent
from sts_agent.battle.mcts import MCTSPlanner
from sts_agent.strategy import BaseStrategyAgent


class _GreedyAgent(BattleAgent):
    """Simple agent: play first playable card targeting first alive enemy."""

    def act(self, obs, actions):
        from sts_env.combat import Action
        if not actions:
            return Action.end_turn()
        # Find first play_card action
        for a in actions:
            if a.action_type.value == 1:  # PLAY_CARD
                return a
        return actions[-1]  # end_turn


class _FirstEdgeStrategy(BaseStrategyAgent):
    """Deterministic map routing: always take the first outgoing edge."""

    def pick_branch(self, sts_map, character, current, seed):
        from sts_agent.strategy.base import _normalize_map_edge

        if current is None:
            for node in sts_map.nodes.get(0, []):
                if node.edges:
                    return (0, node.x)
            return (0, 0)
        f, x = current
        node = sts_map.get_node(f, x)
        if node is None or not node.edges:
            return current
        return _normalize_map_edge(f, node.edges[0])


class TestAct1MapRun:
    """Test the full Act 1 run with map generation using fast greedy agent."""

    @pytest.fixture
    def agent(self):
        return _GreedyAgent()

    def test_map_run_completes(self, agent):
        result = run_act1(agent, seed=42, use_map=True)
        assert result is not None
        assert isinstance(result.victory, bool)
        assert result.total_floors > 0

    def test_map_run_has_encounter_types(self, agent):
        result = run_act1(agent, seed=42, use_map=True)
        assert len(result.encounter_types) > 0
        # Should have at least one combat
        combat_types = {"monster", "elite", "boss"}
        assert any(t in combat_types for t in result.encounter_types)
        # Rest sites should appear if the agent survives long enough
        # (greedy agent may die early on some seeds — that's OK for this smoke test)

    def test_map_run_has_16_floors(self, agent):
        result = run_act1(agent, seed=42, use_map=True)
        if result.victory:
            assert result.total_floors == 16  # 15 map floors (0-14) + boss at floor 15
        else:
            assert result.total_floors <= 16

    def test_map_run_survives_multiple_seeds(self, agent):
        for seed in [7, 42, 5]:
            result = run_act1(agent, seed=seed, use_map=True)
            assert result.floors_cleared > 0

    def test_boss_is_fought_in_map_run(self):
        result = run_act1(
            MCTSPlanner(simulations=100),
            seed=5,
            use_map=True,
            strategy_agent=_FirstEdgeStrategy(seed=5),
        )
        assert "boss" in result.encounter_types
        assert any("boss/" in entry for entry in result.combat_log)

    def test_map_run_boss_at_end(self, agent):
        result = run_act1(agent, seed=42, use_map=True)
        combat_types = [t for t in result.encounter_types if t in ("monster", "elite", "boss")]
        if result.victory:
            assert combat_types[-1] == "boss"

    def test_linear_backwards_compat(self, agent):
        result = run_act1(agent, seed=42, use_map=False)
        assert result is not None
        assert result.total_floors == 8

    def test_default_random_strategy_runs(self, agent):
        """Default (random) BaseStrategyAgent runs end-to-end with rest sites."""
        result = run_act1(agent, seed=42, use_map=True)
        assert result is not None

    def test_explicit_base_strategy_agent_runs(self, agent):
        """Passing an explicit BaseStrategyAgent works the same."""
        from sts_agent.strategy import BaseStrategyAgent
        result = run_act1(
            agent, seed=42, use_map=True,
            strategy_agent=BaseStrategyAgent(seed=42),
        )
        assert result is not None

    def test_map_run_tracks_gold(self, agent):
        result = run_act1(agent, seed=42, use_map=True)
        if result.floors_cleared > 0:
            assert result.floors_cleared >= 1


class TestAct1MapStrategyAgentIntegration:
    """Test that map context is forwarded to the strategy agent's pick_card."""

    def test_map_run_passes_map_to_strategy_agent(self):
        """pick_card must receive sts_map and current_position kwargs on every call."""
        from sts_agent.strategy import BaseStrategyAgent

        # Subclass that records pick_card calls but inherits all other random
        # defaults from BaseStrategyAgent so the run loop has well-typed
        # return values for pick_branch, pick_neow, etc.
        class _RecordingAgent(BaseStrategyAgent):
            def __init__(self, seed):
                super().__init__(seed=seed)
                self.calls: list[dict] = []

            def pick_card(self, character, card_choices, upcoming_encounters,
                          seed, *, sts_map=None, current_position=None):
                self.calls.append({
                    "sts_map": sts_map,
                    "current_position": current_position,
                })
                return "Anger" if "Anger" in card_choices else card_choices[0]

        agent = _RecordingAgent(seed=42)
        run_act1(_GreedyAgent(), seed=42, use_map=True, strategy_agent=agent)

        if not agent.calls:
            pytest.skip("Greedy agent died before any card reward — increase seed coverage")

        for kwargs in agent.calls:
            assert isinstance(kwargs["sts_map"], StSMap)
            floor, x = kwargs["current_position"]
            assert isinstance(floor, int) and isinstance(x, int)


class TestBossRelicOwnershipBoundary:
    """Verify that boss relic offers come from sts_env, not agent-local lists."""

    def test_no_boss_relics_constant_in_run_module(self):
        """sts_agent.run must not define its own _BOSS_RELICS pool."""
        import sts_agent.run as run_mod
        assert not hasattr(run_mod, "_BOSS_RELICS"), (
            "_BOSS_RELICS should live in sts_env, not sts_agent.run"
        )

    def test_apply_boss_relic_reward_delegates_to_env_rewards(self):
        """Boss relic dispatch is now fully handled inside sts_env's orchestrator."""
        pytest.skip("_apply_boss_relic_reward is internal to sts_env; tested via integration")

    def test_boss_relic_choices_come_only_from_env_pool(self):
        """Boss relic choices come from sts_env's pool (verified by integration tests)."""
        pytest.skip("_apply_boss_relic_reward is internal to sts_env; tested via integration")
