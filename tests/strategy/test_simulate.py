"""Tests for strategy.simulate — combat simulation tools."""

from __future__ import annotations

from sts_env.run.character import Character

from sts_agent.strategy.simulate import (
    SimDistribution,
    _ACT1_POOLS,
    get_encounter_pool,
    probe_after_rest,
    probe_encounter,
    probe_with_card,
)


# ---------------------------------------------------------------------------
# 1. Result shape
# ---------------------------------------------------------------------------


def test_probe_encounter_returns_distribution():
    """probe_encounter returns a SimDistribution with all expected fields."""
    char = Character.ironclad()
    dist = probe_encounter(
        char, "easy", "cultist", seed=42, max_nodes=50, simulations=50
    )
    assert isinstance(dist, SimDistribution)
    assert dist.n > 0
    assert dist.std_score >= 0
    assert dist.start_hp == char.player_hp
    assert dist.max_hp == char.player_max_hp


def test_probe_encounter_uses_current_hp():
    """SimDistribution.start_hp reflects the character's current HP, not max HP."""
    char = Character.ironclad()
    char.player_hp = char.player_max_hp // 2

    dist = probe_encounter(
        char, "easy", "cultist", seed=0, max_nodes=50, simulations=50
    )

    assert dist.start_hp == char.player_hp
    assert dist.max_hp == char.player_max_hp
    assert dist.start_hp < dist.max_hp


# ---------------------------------------------------------------------------
# 2. Survival signal on easy encounter
# ---------------------------------------------------------------------------


def test_probe_encounter_low_death_rate_easy():
    """Ironclad starter deck should have a low death rate against cultist."""
    char = Character.ironclad()
    dist = probe_encounter(
        char, "easy", "cultist", seed=0, max_nodes=50, simulations=50
    )
    assert dist.death_rate < 0.5, (
        f"Expected low death rate vs cultist but got {dist.death_rate:.0%}"
    )
    assert dist.expected_damage <= char.player_max_hp


# ---------------------------------------------------------------------------
# 3. probe_with_card — adds a card
# ---------------------------------------------------------------------------


def test_probe_with_card():
    """probe_with_card simulates with a hypothetical extra card."""
    char = Character.ironclad()
    original_deck_size = len(char.deck)

    dist = probe_with_card(
        char, "Strike", "easy", "cultist", seed=42, max_nodes=50, simulations=50
    )
    assert isinstance(dist, SimDistribution)
    # Original character should not be mutated
    assert len(char.deck) == original_deck_size


# ---------------------------------------------------------------------------
# 4. Character is not mutated
# ---------------------------------------------------------------------------


def test_character_not_mutated():
    """probe_encounter deep-copies the character — no side effects."""
    char = Character.ironclad()
    snapshot = (
        list(char.deck),
        char.player_hp,
        char.player_max_hp,
        list(char.potions),
        char.gold,
        char.floor,
    )

    probe_encounter(char, "easy", "cultist", seed=0, max_nodes=50, simulations=50)

    post = (
        list(char.deck),
        char.player_hp,
        char.player_max_hp,
        list(char.potions),
        char.gold,
        char.floor,
    )
    assert snapshot == post, "Character was mutated by probe_encounter"


# ---------------------------------------------------------------------------
# 5. Distribution fields are sane
# ---------------------------------------------------------------------------


def test_probe_encounter_distribution_fields():
    """probe_encounter returns a distribution with sane n and std."""
    char = Character.ironclad()
    dist = probe_encounter(
        char, "easy", "cultist", seed=42, max_nodes=200, simulations=200
    )
    assert dist.n > 0, "distribution.n must be positive"
    assert dist.std_score >= 0, "std must be non-negative"
    assert 0.0 <= dist.death_rate <= 1.0


def test_probe_with_card_distribution_fields():
    """probe_with_card also returns a valid distribution."""
    char = Character.ironclad()
    dist = probe_with_card(
        char, "Strike", "easy", "cultist", seed=42, max_nodes=200, simulations=200
    )
    assert dist.n > 0


# ---------------------------------------------------------------------------
# 6. get_encounter_pool
# ---------------------------------------------------------------------------


class TestGetEncounterPool:
    """get_encounter_pool distinguishes pool references from specific encounters."""

    def test_empty_encounter_id_returns_room_type_pool(self):
        """Empty encounter_id → pool keyed by room_type."""
        pool = get_encounter_pool("monster", "")
        assert pool is not None
        assert pool == _ACT1_POOLS["monster"]

    def test_empty_encounter_id_elite(self):
        """Empty encounter_id + elite room_type → elite pool."""
        pool = get_encounter_pool("elite", "")
        assert pool is not None
        assert pool == _ACT1_POOLS["elite"]

    def test_pool_name_as_encounter_id_returns_pool(self):
        """Passing 'elite' as encounter_id is detected as a pool reference."""
        pool = get_encounter_pool("elite", "elite")
        assert pool is not None
        assert pool == _ACT1_POOLS["elite"]

    def test_monster_as_encounter_id_returns_pool(self):
        """Passing 'monster' as encounter_id is detected as a pool reference."""
        pool = get_encounter_pool("monster", "monster")
        assert pool is not None
        assert pool == _ACT1_POOLS["monster"]

    def test_specific_encounter_id_returns_none(self):
        """Specific enemy name → None (not a pool reference)."""
        assert get_encounter_pool("easy", "cultist") is None

    def test_specific_elite_returns_none(self):
        """Specific elite enemy name → None."""
        assert get_encounter_pool("elite", "Lagavulin") is None

    def test_specific_boss_returns_none(self):
        """Specific boss name → None."""
        assert get_encounter_pool("boss", "hexaghost") is None

    def test_pool_entries_are_tuples(self):
        """Returned pool contains (enc_type, enc_id) tuples."""
        pool = get_encounter_pool("elite", "")
        assert all(isinstance(entry, tuple) and len(entry) == 2 for entry in pool)


# ---------------------------------------------------------------------------
# 7. probe_after_rest
# ---------------------------------------------------------------------------


class TestProbeAfterRest:
    """probe_after_rest heals the character before probing."""

    def test_returns_distribution(self):
        """probe_after_rest returns a SimDistribution."""
        char = Character.ironclad()
        dist = probe_after_rest(
            char, "easy", "cultist", seed=42, max_nodes=50, simulations=50
        )
        assert isinstance(dist, SimDistribution)

    def test_does_not_mutate_character(self):
        """probe_after_rest must not change the caller's character."""
        char = Character.ironclad()
        hp_before = char.player_hp
        probe_after_rest(char, "easy", "cultist", seed=0, max_nodes=50, simulations=50)
        assert char.player_hp == hp_before

    def test_healed_character_passed_to_combat(self):
        """probe_after_rest must pass the healed character to build_combat.

        We damage the character first so the 30% heal actually moves the needle,
        then verify the HP given to build_combat is strictly higher than the
        wounded starting HP.
        """
        from unittest.mock import patch
        import sts_agent.strategy.simulate as sim_mod

        char = Character.ironclad()
        char.player_hp = char.player_max_hp // 2
        wounded_hp = char.player_hp

        captured: list[int] = []
        real_build = sim_mod.build_combat

        def capturing_build(enc_type, enc_id, seed, *, character):
            captured.append(character.player_hp)
            return real_build(enc_type, enc_id, seed, character=character)

        with patch.object(sim_mod, "build_combat", side_effect=capturing_build):
            probe_after_rest(char, "easy", "cultist", seed=42, max_nodes=50, simulations=10)

        assert len(captured) == 1
        assert captured[0] > wounded_hp, (
            f"probe_after_rest should pass healed HP ({captured[0]}) "
            f"> wounded HP ({wounded_hp}) to build_combat"
        )

    def test_full_hp_character_start_hp_matches_current(self):
        """A character at full HP reports start_hp == max_hp."""
        char = Character.ironclad()
        assert char.player_hp == char.player_max_hp

        dist = probe_after_rest(
            char, "easy", "cultist", seed=0, max_nodes=50, simulations=50
        )
        assert dist.start_hp == char.player_hp
        assert dist.max_hp == char.player_max_hp


# ---------------------------------------------------------------------------
# 8. try_rest tool in _make_tools
# ---------------------------------------------------------------------------


def test_try_rest_in_make_tools():
    """_make_tools must include a callable named 'try_rest'."""
    from sts_agent.strategy.llm_agent import _make_tools

    char = Character.ironclad()
    tools = _make_tools(char, lambda: None, max_simulations=10, max_nodes=50)
    tool_names = [fn.__name__ for fn in tools]
    assert "try_rest" in tool_names, (
        f"Expected 'try_rest' in tool names, got: {tool_names}"
    )


def test_try_rest_tool_runs(tmp_path):
    """try_rest tool is callable and returns a non-empty string."""
    from sts_agent.strategy.llm_agent import _make_tools

    char = Character.ironclad()
    char.player_hp = char.player_max_hp // 2  # wound so heal is visible

    tools = _make_tools(char, lambda: None, max_simulations=10, max_nodes=50)
    try_rest_fn = next(fn for fn in tools if fn.__name__ == "try_rest")

    result = try_rest_fn("easy", "cultist")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "After rest" in result


# ---------------------------------------------------------------------------
# 9. Probe-only: each tool call invokes MCTSPlanner.act() exactly once
# ---------------------------------------------------------------------------


def test_tool_calls_mcts_act_exactly_once():
    """Each tool call must invoke MCTSPlanner.act() exactly once (probe-only).

    The old simulate_* functions drove MCTSPlanner through the entire combat,
    calling act() at every action step. The new probe_* API calls act() once,
    then reads the rollout distribution from planner.last_stats. This test
    enforces that contract for a specific-encounter tool invocation.
    """
    from unittest.mock import patch
    from sts_agent.strategy.llm_agent import _make_tools
    from sts_agent.battle.mcts import MCTSPlanner

    char = Character.ironclad()
    act_calls: list[int] = []
    original_act = MCTSPlanner.act

    def tracking_act(self, combat):
        act_calls.append(1)
        return original_act(self, combat)

    with patch.object(MCTSPlanner, "act", tracking_act):
        tools = _make_tools(char, lambda: None, max_simulations=10, max_nodes=50)
        simulate_upcoming = next(fn for fn in tools if fn.__name__ == "simulate_upcoming")
        simulate_upcoming("easy", "cultist")

    assert len(act_calls) == 1, (
        f"Expected MCTSPlanner.act() called exactly once, got {len(act_calls)}"
    )
