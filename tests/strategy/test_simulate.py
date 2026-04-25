"""Tests for strategy.simulate — combat simulation tools."""

from __future__ import annotations

from sts_env.run.character import Character

from sts_agent.strategy import simulate_encounter, simulate_with_card, SimResult


# ---------------------------------------------------------------------------
# 1. Result shape
# ---------------------------------------------------------------------------


def test_simulate_encounter_returns_result():
    """simulate_encounter returns a SimResult with all expected fields."""
    char = Character.ironclad()
    result = simulate_encounter(
        char, "easy", "cultist", seed=42, max_nodes=50, simulations=50
    )
    assert isinstance(result, SimResult)
    assert isinstance(result.survived, bool)
    assert isinstance(result.damage_taken, int)
    assert isinstance(result.max_hp_gained, int)
    assert isinstance(result.final_hp, int)
    assert isinstance(result.final_max_hp, int)
    assert isinstance(result.turns, int)
    assert result.damage_taken >= 0
    assert result.turns >= 0


# ---------------------------------------------------------------------------
# 2. Survival on easy encounter
# ---------------------------------------------------------------------------


def test_simulate_encounter_survives_easy():
    """Ironclad starter deck should survive a cultist fight."""
    char = Character.ironclad()
    result = simulate_encounter(
        char, "easy", "cultist", seed=0, max_nodes=50, simulations=50
    )
    assert result.survived, (
        f"Expected to survive cultist but died (damage_taken={result.damage_taken})"
    )
    assert result.final_hp > 0
    assert result.final_hp <= char.player_max_hp


# ---------------------------------------------------------------------------
# 3. simulate_with_card adds a card
# ---------------------------------------------------------------------------


def test_simulate_with_card():
    """simulate_with_card simulates with a hypothetical extra card."""
    char = Character.ironclad()
    original_deck_size = len(char.deck)

    result = simulate_with_card(
        char, "Strike", "easy", "cultist", seed=42, max_nodes=50, simulations=50
    )
    assert isinstance(result, SimResult)
    # Original character should not be mutated
    assert len(char.deck) == original_deck_size


# ---------------------------------------------------------------------------
# 4. Character is not mutated
# ---------------------------------------------------------------------------


def test_character_not_mutated():
    """simulate_encounter deep-copies the character — no side effects."""
    char = Character.ironclad()
    snapshot = (
        list(char.deck),
        char.player_hp,
        char.player_max_hp,
        list(char.potions),
        char.gold,
        char.floor,
    )

    simulate_encounter(char, "easy", "cultist", seed=0, max_nodes=50, simulations=50)

    post = (
        list(char.deck),
        char.player_hp,
        char.player_max_hp,
        list(char.potions),
        char.gold,
        char.floor,
    )
    assert snapshot == post, "Character was mutated by simulate_encounter"
