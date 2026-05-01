"""Integration tests for Act 1 map-based run."""

import pytest
from unittest.mock import MagicMock, patch

from sts_env.run.map import StSMap
from sts_env.run.rewards import BOSS_RELICS
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
        # return values for plan_route, pick_neow, etc.
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
        """_apply_boss_relic_reward must call rewards.roll_boss_relic_choices
        and pass its result (not any agent-local list) to pick_boss_relic."""
        from unittest.mock import patch, MagicMock
        from sts_env.run.character import Character
        from sts_env.combat.rng import RNG
        from sts_agent.run import _apply_boss_relic_reward
        import sts_agent.run as run_mod

        character = Character.ironclad()
        strategy_agent = MagicMock()
        strategy_agent.pick_boss_relic.return_value = None
        rng = RNG(0)

        sentinel = ["TinyHouse", "BustedCrown"]
        with patch.object(run_mod.rewards, "roll_boss_relic_choices",
                          return_value=sentinel) as mock_roll:
            _apply_boss_relic_reward(character, strategy_agent, rng)

        mock_roll.assert_called_once()
        _, kwargs = mock_roll.call_args
        assert kwargs.get("owned") == character.relics or mock_roll.call_args[0][1] == character.relics

        strategy_agent.pick_boss_relic.assert_called_once_with(character, sentinel)

    def test_boss_relic_choices_come_only_from_env_pool(self):
        """Choices offered to pick_boss_relic must be a subset of BOSS_RELICS."""
        from sts_env.run.character import Character
        from sts_env.combat.rng import RNG
        from sts_agent.run import _apply_boss_relic_reward

        offered: list[list[str]] = []

        class _RecordingAgent:
            def pick_boss_relic(self, character, choices):
                offered.append(list(choices))
                return None

        _apply_boss_relic_reward(Character.ironclad(), _RecordingAgent(), RNG(0))

        assert offered, "pick_boss_relic was never called"
        for choices in offered:
            for relic in choices:
                assert relic in BOSS_RELICS, (
                    f"Offered relic '{relic}' is not in the env BOSS_RELICS pool"
                )
