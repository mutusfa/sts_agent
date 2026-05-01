"""Tests for SimStrategyAgent.plan_route — probe-based map path selection."""

import pytest

from sts_env.run.map import RoomType, StSMap, MapNode, generate_act1_map
from sts_env.run.character import Character
from sts_agent.strategy.sim_agent import SimStrategyAgent


def _make_simple_map() -> StSMap:
    """Build a tiny 3-floor map for fast tests.

    Floor 0: M(x=0) → [M(x=0), R(x=1)]
    Floor 1: M(x=0) → [M(x=0)]
              R(x=1) → [M(x=0)]
    Floor 2: M(x=0)
    """
    nodes = {
        0: [MapNode(floor=0, x=0, room_type=RoomType.MONSTER,
                     edges=[(1, 0), (1, 1)])],
        1: [
            MapNode(floor=1, x=0, room_type=RoomType.MONSTER,
                     edges=[(2, 0)]),
            MapNode(floor=1, x=1, room_type=RoomType.REST,
                     edges=[(2, 0)]),
        ],
        2: [MapNode(floor=2, x=0, room_type=RoomType.MONSTER, edges=[])],
    }
    return StSMap(nodes=nodes, seed=42)


class TestPlanRoute:
    """Test plan_route routing logic."""

    def test_plan_route_returns_valid_path(self):
        """plan_route returns a non-empty path through the map."""
        sts_map = _make_simple_map()
        char = Character(player_hp=80, player_max_hp=80)
        agent = SimStrategyAgent(sim_nodes=100, sim_sims=100)
        path = agent.plan_route(sts_map, char, seed=42)
        assert len(path) >= 2
        assert path[0][0] == 0
        for i in range(1, len(path)):
            assert path[i][0] == path[i-1][0] + 1

    def test_plan_route_prefers_rest_when_low_hp(self):
        """When HP is low (< 40%), agent prefers REST branches."""
        sts_map = _make_simple_map()
        char = Character(player_hp=10, player_max_hp=80)
        agent = SimStrategyAgent(sim_nodes=100, sim_sims=100)
        path = agent.plan_route(sts_map, char, seed=42)
        assert path[1] == (1, 1), f"Expected REST at (1,1), got {path[1]}"

    def test_plan_route_on_full_act1_map(self):
        """plan_route works on a full 15-floor map."""
        sts_map = generate_act1_map(seed=123)
        char = Character(player_hp=80, player_max_hp=80)
        agent = SimStrategyAgent(sim_nodes=100, sim_sims=100)
        path = agent.plan_route(sts_map, char, seed=123)
        assert path[-1][0] == 14
        for i in range(1, len(path)):
            assert path[i][0] == path[i-1][0] + 1
        for f, x in path:
            node = sts_map.get_node(f, x)
            assert node is not None, f"No node at ({f},{x})"

    def test_plan_route_produces_valid_path_to_boss(self):
        """plan_route produces a valid, contiguous path reaching the boss."""
        sts_map = generate_act1_map(seed=99)
        char = Character(player_hp=80, player_max_hp=80)
        agent = SimStrategyAgent(sim_nodes=100, sim_sims=100)
        path = agent.plan_route(sts_map, char, seed=99)
        assert path[-1][0] == 14
        for i in range(1, len(path)):
            assert path[i][0] == path[i-1][0] + 1
        for f, x in path:
            node = sts_map.get_node(f, x)
            assert node is not None, f"No node at ({f},{x})"

    @pytest.mark.skip(reason="Slow integration test — run with -k run_act1 explicitly")
    def test_run_act1_uses_strategy_agent_plan_route(self):
        """run_act1 with SimStrategyAgent uses probe-based path selection."""
        from sts_agent.run import run_act1
        from sts_agent.battle.mcts import MCTSPlanner

        agent = SimStrategyAgent(sim_nodes=100, sim_sims=100)
        result = run_act1(
            MCTSPlanner(), seed=77, strategy_agent=agent, use_map=True,
        )
        assert result is not None
        assert result.total_floors > 0
        assert result.floors_cleared > 0
