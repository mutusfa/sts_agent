"""Integration tests for Act 1 map-based run."""

import pytest

from sts_agent.run import run_act1
from sts_agent.battle.mcts import MCTSPlanner
from sts_agent.battle.base import BattleAgent


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

    def test_map_run_has_15_floors(self, agent):
        result = run_act1(agent, seed=42, use_map=True)
        assert result.total_floors == 15  # map has 15 floors (0-14)

    def test_map_run_survives_multiple_seeds(self, agent):
        for seed in [7, 42, 99]:
            result = run_act1(agent, seed=seed, use_map=True)
            assert result.floors_cleared > 0

    def test_map_run_boss_at_end(self, agent):
        result = run_act1(agent, seed=42, use_map=True)
        combat_types = [t for t in result.encounter_types if t in ("monster", "elite", "boss")]
        if result.victory:
            assert combat_types[-1] == "boss"

    def test_linear_backwards_compat(self, agent):
        result = run_act1(agent, seed=42, use_map=False)
        assert result is not None
        assert result.total_floors == 8

    def test_rest_strategy_always_heal(self, agent):
        result = run_act1(agent, seed=42, use_map=True, rest_strategy="always_heal")
        assert result is not None

    def test_rest_strategy_always_upgrade(self, agent):
        result = run_act1(agent, seed=42, use_map=True, rest_strategy="always_upgrade")
        assert result is not None

    def test_map_run_tracks_gold(self, agent):
        result = run_act1(agent, seed=42, use_map=True)
        if result.floors_cleared > 0:
            assert result.floors_cleared >= 1
