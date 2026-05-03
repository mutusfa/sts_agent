"""LLM-based strategy agent for card reward decisions.

Uses dspy.ReAct to give the LLM simulation tools for evaluating card choices,
with a configurable timeout and deterministic forced-pick fallback.

Architecture
------------
::

    Character state + card choices + upcoming encounters + map view
        │
        ▼
    dspy.ReAct (thought → tool call → observation loop)
        │  tools: simulate_upcoming, try_card
        │  budget: 5 min (tools raise TimeoutError when exceeded)
        ▼
    Extracted pick (card_id or "skip")
        │
        ▼
    Validate against choices → forced-pick fallback on any failure

Public API
----------
StrategyAgent   — main agent class
configure_lm    — set up the z.ai / OpenAI-compatible LM
ensure_lm       — lazy init helper
"""

from __future__ import annotations

import inspect
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import dspy
import mlflow
from pydantic import BaseModel
from dotenv import load_dotenv

from sts_env.run.character import Character
from sts_env.run.rooms import RestChoice, RestResult

from .base import BaseStrategyAgent
from .simulate import SimResult, simulate_encounter, simulate_with_card, simulate_with_upgrade, simulate_without_card

if TYPE_CHECKING:
    from sts_env.run.map import StSMap
    from sts_env.run.encounter_queue import EncounterQueue
    from sts_env.run.events import EventSpec

import dataclasses

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Card spec model
# ---------------------------------------------------------------------------

# Derive numeric effect field names directly from CardSpec: int fields whose
# default is exactly 0 (excludes hits=1, booleans False==0, cost with no default).
def _derive_effect_fields() -> tuple[str, ...]:
    from sts_env.combat.cards import CardSpec
    return tuple(
        f.name for f in dataclasses.fields(CardSpec)
        if f.default == 0 and not isinstance(f.default, bool)
    )

_EFFECT_FIELDS = _derive_effect_fields()


class CardInfo(BaseModel):
    """Structured card spec for LLM consumption."""

    card_id: str
    cost: int
    card_type: str           # "attack" | "skill" | "power" | "curse" | "status"
    rarity: str              # "basic" | "common" | "uncommon" | "rare" | "special"
    target: str              # "single_enemy" | "all_enemies" | "none"
    effects: dict[str, int]  # non-zero base declarative fields; hits shown only when >1
    upgrade: dict[str, int]  # upgrade deltas (empty = not upgradeable)
    exhausts: bool
    # TODO: Replace raw Python source with English docstrings for custom handlers.
    #   Need to figure out how to represent scaling effects (e.g. Juggernaut/Juggernaut+
    #   damage, Rampage accumulator) concisely. Raw code always works as a fallback
    #   since the LLM can reason over it, but it's token-heavy.
    custom_code: str | None  # source of custom handler if present, else None


def _card_info(card_id: str) -> CardInfo:
    """Build a :class:`CardInfo` from the sts_env card registry."""
    from sts_env.combat.cards import get_spec

    spec = get_spec(card_id)

    effects: dict[str, int] = {}
    for field_name in _EFFECT_FIELDS:
        val = getattr(spec, field_name, 0)
        if val:
            effects[field_name] = int(val)
    if spec.hits > 1:
        effects["hits"] = spec.hits
    if spec.innate:
        effects["innate"] = 1

    custom_code: str | None = None
    if spec.custom is not None:
        custom_code = inspect.getsource(spec.custom)

    return CardInfo(
        card_id=card_id,
        cost=spec.cost,
        card_type=spec.card_type.name.lower(),
        rarity=spec.rarity.name.lower(),
        target=spec.target.name.lower(),
        effects=effects,
        upgrade=dict(spec.upgrade),
        exhausts=spec.exhausts,
        custom_code=custom_code,
    )


# ---------------------------------------------------------------------------
# Environment / API key
# ---------------------------------------------------------------------------

_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"


load_dotenv(dotenv_path=_ENV_PATH)


# ---------------------------------------------------------------------------
# LM configuration
# ---------------------------------------------------------------------------

DEEPSEEK_API_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_TIMEOUT = 300  # seconds (5 min)

_lm_configured = False


def configure_lm(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    api_base: str = DEEPSEEK_API_BASE,
) -> dspy.LM:
    """Configure the dspy language model.

    Parameters
    ----------
    model:
        Model name forwarded to the provider (default ``"deepseek-v4-flash"``).
    api_key:
        API key.  Falls back to ``DEEPSEEK_API_KEY`` env var.
    api_base:
        OpenAI-compatible base URL.

    Returns
    -------
    The configured :class:`dspy.LM` instance.
    """
    global _lm_configured
    key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not found. Set it in sts_agent/.env or as an env var."
        )
    lm = dspy.LM(f"openai/{model}", api_key=key, api_base=api_base)
    dspy.configure(lm=lm)
    _lm_configured = True
    return lm


def ensure_lm() -> None:
    """Configure the LM with defaults on first use."""
    if not _lm_configured:
        configure_lm()


# ---------------------------------------------------------------------------
# Simulation tools (LLM-callable)
# ---------------------------------------------------------------------------


def _format_result(label: str, result: SimResult) -> str:
    """Render a :class:`SimResult` as a compact string for the LLM."""
    status = "SURVIVED" if result.survived else "DIED"
    parts = [
        f"{label}: {status}",
        f"damage_taken={result.damage_taken}",
        f"max_hp_gained={result.max_hp_gained}",
        f"final_hp={result.final_hp}/{result.final_max_hp}",
        f"turns={result.turns}",
    ]
    if result.enemy_hp_remaining > 0:
        parts.append(f"enemy_hp_remaining={result.enemy_hp_remaining}")
    return " | ".join(parts)


def _format_possible_encounters(data: dict | None) -> str:
    """Format possible_encounters dict as a human-readable string for the LLM."""
    if data is None:
        return "(encounter tracking not available)"
    lines = []
    if data.get("monster_weak"):
        lines.append(f"weak: {data['monster_weak']}")
    lines.append(f"strong: {data.get('monster_strong', [])}")
    lines.append(f"elite: {data.get('elite', [])}")
    lines.append(f"boss: {data.get('boss', '?')}")
    return "\n".join(lines)


def _make_tools(
    character: Character,
    seed: int,
    budget_checker: callable,
    max_simulations: int = 100,
    max_nodes: int = 1000,
    sim_log: list[str] | None = None,
) -> list:
    """Create tool callables bound to the current decision context.

    Each tool checks the remaining time budget before running a simulation.
    When budget is exhausted, the tool returns a message instead of raising,
    so the LLM can make a final decision using results collected so far.

    Parameters
    ----------
    character:
        Character state (deep-copied by simulate helpers).
    seed:
        Master seed for deterministic simulations.
    budget_checker:
        Callable that returns None if budget remains, or raises TimeoutError.
    max_simulations:
        MCTS rollouts per tool call.
    max_nodes:
        MCTS node-expansion budget per tool call.
    sim_log:
        Optional list that accumulates formatted simulation results, so
        the final-decision prompt can include all prior results.
    """

    def simulate_upcoming(room_type: str, encounter_id: str = "") -> str:
        """Simulate an upcoming encounter of the given room type with the current deck.

        Args:
            room_type: Room type read from the map view — one of 'monster', 'elite', 'boss'.
            encounter_id: Optional specific encounter ID from possible_encounters
                (e.g. 'Lagavulin', 'cultist', 'hexaghost'). If empty, samples from the pool.
        Returns:
            Human-readable simulation result (survived, damage, HP, turns).
        """
        try:
            budget_checker()
        except TimeoutError:
            return (
                "SIMULATION BUDGET EXHAUSTED — no more simulations available. "
                "Make your pick now based on the results you already have."
            )
        label = f"Baseline ({encounter_id})" if encounter_id else f"Baseline ({room_type})"
        result = simulate_encounter(
            character, room_type, encounter_id, seed * 1000,
            max_nodes=max_nodes, simulations=max_simulations,
        )
        formatted = _format_result(label, result)
        if sim_log is not None:
            sim_log.append(formatted)
        return formatted

    def try_card(card_id: str, room_type: str, encounter_id: str = "") -> str:
        """Simulate an encounter with a hypothetical card added to the deck.

        Args:
            card_id: The card ID to evaluate (one of the reward choices).
            room_type: Room type read from the map view — one of 'monster', 'elite', 'boss'.
            encounter_id: Optional specific encounter ID from possible_encounters
                (e.g. 'Lagavulin', 'cultist', 'hexaghost'). If empty, samples from the pool.
        Returns:
            Human-readable simulation result for the card + encounter.
        """
        try:
            budget_checker()
        except TimeoutError:
            return (
                "SIMULATION BUDGET EXHAUSTED — no more simulations available. "
                "Make your pick now based on the results you already have."
            )
        result = simulate_with_card(
            character, card_id, room_type, encounter_id, seed * 1000,
            max_nodes=max_nodes, simulations=max_simulations,
        )
        label = f"With {card_id} vs {encounter_id}" if encounter_id else f"With {card_id} vs {room_type}"
        formatted = _format_result(label, result)
        if sim_log is not None:
            sim_log.append(formatted)
        return formatted

    def try_remove_card(card_id: str, room_type: str, encounter_id: str = "") -> str:
        """Simulate an encounter with a card removed from the deck.

        Args:
            card_id: The card ID to remove from the deck.
            room_type: Room type read from the map view — one of 'monster', 'elite', 'boss'.
            encounter_id: Optional specific encounter ID from possible_encounters
                (e.g. 'Lagavulin', 'cultist', 'hexaghost'). If empty, samples from the pool.
        Returns:
            Human-readable simulation result for the deck without that card.
        """
        try:
            budget_checker()
        except TimeoutError:
            return (
                "SIMULATION BUDGET EXHAUSTED — no more simulations available. "
                "Make your pick now based on the results you already have."
            )
        result = simulate_without_card(
            character, card_id, room_type, encounter_id, seed * 1000,
            max_nodes=max_nodes, simulations=max_simulations,
        )
        label = f"Without {card_id} vs {encounter_id}" if encounter_id else f"Without {card_id} vs {room_type}"
        formatted = _format_result(label, result)
        if sim_log is not None:
            sim_log.append(formatted)
        return formatted

    return [simulate_upcoming, try_card, try_remove_card]


# ---------------------------------------------------------------------------
# dspy signature
# ---------------------------------------------------------------------------


class CardPickSignature(dspy.Signature):
    """You are a Slay the Spire strategy expert deciding which card to pick from combat rewards.

    Analyze the card choices using simulation tools to evaluate how each card
    performs in upcoming encounters. Pick the card that best improves survivability.

    If no card meaningfully helps, or the deck is already efficient, skip.

    Card evaluation priorities for Ironclad:
    1. Survivability — survive the encounter, minimize damage taken.
    2. Max HP gains — Feed is extremely valuable.
    3. Deck synergy — attack density, block consistency.
    4. Upcoming difficulty — elites and boss need stronger cards.

    Simulation tools accept an optional encounter_id to target a specific enemy.
    Use the possible_encounters field to see which encounters are still possible.

    Map symbols: M=monster, E=elite, B=boss, R=rest, ?=event, $=shop, T=treasure.
    Current position is marked with @ in the map view. Upcoming reachable rooms
    are shown with their type symbol (M/E/B/R/?/$/ T). Use M→'monster',
    E→'elite', B→'boss' as the room_type argument to simulation tools.
    """

    character_state: str = dspy.InputField(
        desc="Current character: HP, deck size, potions, relics, floor"
    )
    # TODO: Consider including full deck composition here for deck synergy reasoning.
    #   Current design relies on simulation tools to evaluate synergy empirically.
    #   Will validate with ablation studies (with/without deck info) before committing.
    card_choices: list[CardInfo] = dspy.InputField(
        desc="Cards offered as reward — each carries full spec, upgrade deltas, and custom handler code if any"
    )
    map_view: str = dspy.InputField(
        desc=(
            "ASCII map of the act. @ = current position; "
            "M/E/B/R/?/$/ T = upcoming reachable rooms. "
            "Pass 'monster', 'elite', or 'boss' to the simulation tools based on what you see ahead."
        )
    )
    possible_encounters: str = dspy.InputField(
        desc=(
            "Encounters still possible based on what you've seen so far. "
            "Format: 'weak: [ids], strong: [ids], elite: [ids], boss: id'. "
            "Use these IDs as the encounter_id parameter in simulation tools "
            "to target specific enemies."
        )
    )
    reasoning: str = dspy.OutputField(
        desc="Brief analysis of each card option based on simulation results"
    )
    pick: str = dspy.OutputField(
        desc="Exact card ID to pick from the choices, or 'skip' to skip all rewards"
    )


# ---------------------------------------------------------------------------
# Map view rendering
# ---------------------------------------------------------------------------

_MAP_NO_MAP_STUB = "(no map available — linear scenario)"


def _format_map_view(
    sts_map: StSMap | None,
    current_position: tuple[int, int] | None,
) -> str:
    """Render map context using sts_env's forward-looking ASCII view."""
    if sts_map is None:
        return _MAP_NO_MAP_STUB

    current_floor = current_position[0] if current_position else None
    current_x = current_position[1] if current_position else None
    return (
        "```text\n"
        f"{sts_map.render_ascii(
            current_floor=current_floor,
            current_x=current_x,
            reachable_only=True,
        )}\n"
        "```"
    )


def _make_rest_tools(
    character: Character,
    check_budget: Callable[[], None],
    *,
    max_simulations: int = 100,
    max_nodes: int = 1000,
    sim_log: list[str] | None = None,
) -> list:
    """Create simulation tools for the rest-site decision.

    Tools are the same ``simulate_upcoming`` and ``try_card`` as card-pick,
    plus a ``try_upgrade`` tool that simulates with one card upgraded.
    """
    import random as _random

    seed = _random.randint(0, 2**31)

    def simulate_upcoming(room_type: str, encounter_id: str = "") -> str:
        """Simulate an upcoming encounter with the current deck.

        Args:
            room_type: Room type: 'monster', 'elite', 'boss', or 'event'.
            encounter_id: Optional specific encounter from possible_encounters.
        """
        try:
            check_budget()
        except TimeoutError:
            return (
                "SIMULATION BUDGET EXHAUSTED — Make your pick now based "
                "on the results you already have."
            )
        result = simulate_encounter(
            character, room_type, encounter_id, seed,
            max_nodes=max_nodes, simulations=max_simulations,
        )
        formatted = _format_result(f"Baseline vs {encounter_id or room_type}", result)
        if sim_log is not None:
            sim_log.append(formatted)
        return formatted

    def try_upgrade(card_id: str, room_type: str, encounter_id: str = "") -> str:
        """Simulate an encounter with a card upgraded in the deck.

        Replaces the first unupgraded copy of card_id with its upgraded
        version (card_id + '+') and runs a simulation.

        Args:
            card_id: Card to upgrade (must be in upgradeable_cards list).
            room_type: Room type: 'monster', 'elite', 'boss', or 'event'.
            encounter_id: Optional specific encounter from possible_encounters.
        """
        try:
            check_budget()
        except TimeoutError:
            return (
                "SIMULATION BUDGET EXHAUSTED — Make your pick now based "
                "on the results you already have."
            )
        result = simulate_with_upgrade(
            character, card_id, room_type, encounter_id, seed,
            max_nodes=max_nodes, simulations=max_simulations,
        )
        label = f"With {card_id}+ vs {encounter_id or room_type}"
        formatted = _format_result(label, result)
        if sim_log is not None:
            sim_log.append(formatted)
        return formatted

    def try_remove_card(card_id: str, room_type: str, encounter_id: str = "") -> str:
        """Simulate an encounter with a card removed from the deck.

        Removes the first copy of card_id from the deck and runs a simulation.

        Args:
            card_id: Card to remove (must be in deck).
            room_type: Room type: 'monster', 'elite', 'boss', or 'event'.
            encounter_id: Optional specific encounter from possible_encounters.
        """
        try:
            check_budget()
        except TimeoutError:
            return (
                "SIMULATION BUDGET EXHAUSTED — Make your pick now based "
                "on the results you already have."
            )
        result = simulate_without_card(
            character, card_id, room_type, encounter_id, seed,
            max_nodes=max_nodes, simulations=max_simulations,
        )
        label = f"Without {card_id} vs {encounter_id or room_type}"
        formatted = _format_result(label, result)
        if sim_log is not None:
            sim_log.append(formatted)
        return formatted

    return [simulate_upcoming, try_upgrade, try_remove_card]


def _make_card_removal_tools(
    character: Character,
    check_budget: Callable[[], None],
    *,
    max_simulations: int = 100,
    max_nodes: int = 1000,
    sim_log: list[str] | None = None,
) -> list:
    """Create simulation tools for the card-removal decision.

    Provides ``simulate_upcoming`` and ``try_remove_card`` so the agent can
    compare deck performance with and without each candidate card.
    """
    import random as _random

    seed = _random.randint(0, 2**31)

    def simulate_upcoming(room_type: str, encounter_id: str = "") -> str:
        """Simulate an upcoming encounter with the current deck.

        Args:
            room_type: Room type: 'monster', 'elite', 'boss', or 'event'.
            encounter_id: Optional specific encounter from possible_encounters.
        """
        try:
            check_budget()
        except TimeoutError:
            return (
                "SIMULATION BUDGET EXHAUSTED — Make your pick now based "
                "on the results you already have."
            )
        result = simulate_encounter(
            character, room_type, encounter_id, seed,
            max_nodes=max_nodes, simulations=max_simulations,
        )
        formatted = _format_result(f"Baseline vs {encounter_id or room_type}", result)
        if sim_log is not None:
            sim_log.append(formatted)
        return formatted

    def try_remove_card(card_id: str, room_type: str, encounter_id: str = "") -> str:
        """Simulate an encounter with a card removed from the deck.

        Args:
            card_id: Card to remove (must be in deck_cards list).
            room_type: Room type: 'monster', 'elite', 'boss', or 'event'.
            encounter_id: Optional specific encounter from possible_encounters.
        """
        try:
            check_budget()
        except TimeoutError:
            return (
                "SIMULATION BUDGET EXHAUSTED — Make your pick now based "
                "on the results you already have."
            )
        result = simulate_without_card(
            character, card_id, room_type, encounter_id, seed,
            max_nodes=max_nodes, simulations=max_simulations,
        )
        formatted = _format_result(f"Without {card_id} vs {encounter_id or room_type}", result)
        if sim_log is not None:
            sim_log.append(formatted)
        return formatted

    return [simulate_upcoming, try_remove_card]


# ---------------------------------------------------------------------------
# Standard context — formalised across all strategy decisions
# ---------------------------------------------------------------------------
# Every LLM strategy decision receives this baseline context:
#   - character_state: HP, max HP, gold, floor, deck size, potions, relics
#   - map_view: ASCII map with current position and upcoming rooms
#   - possible_encounters: remaining encounters by pool (weak/strong/elite/boss)
#   - simulation tools: simulate_upcoming, try_card, try_upgrade (where relevant)
#
# This is the "standard context".  If a human StS expert would have additional
# information that the agent lacks (e.g. exact deck list, specific event outcome
# probabilities, card pool composition), the LLM should flag it in the
# `missing_context` output field so we can iterate on the context design.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Event decision signature
# ---------------------------------------------------------------------------


class EventPickSignature(dspy.Signature):
    """You are a Slay the Spire strategy expert making an event choice.

    You are at an event and must choose one of the presented options.  Each
    choice describes its effect.  Evaluate based on:

    1. HP ratio — low HP means avoid risky choices; high HP means you can
       afford max-HP costs or damage.
    2. Gold — some choices cost gold; consider if the reward is worth it.
    3. Deck state — does your deck benefit from card removal / upgrade / add?
    4. Upcoming difficulty — how much HP do you need for remaining floors?
    5. Relics — some relics synergise with certain outcomes.

    Use simulation tools to evaluate your deck's strength against upcoming
    encounters if that helps weigh the trade-off.

    Map symbols: M=monster, E=elite, B=boss, R=rest, ?=event, $=shop, T=treasure.
    Current position is marked with @ in the map view.
    """

    character_state: str = dspy.InputField(
        desc="Current HP, max HP, gold, floor, deck list (card IDs), potions, relics."
    )
    event_description: str = dspy.InputField(
        desc="Narrative description of the event."
    )
    event_choices: list[str] = dspy.InputField(
        desc="Ordered list of choice labels. Pick by index (0-based)."
    )
    map_view: str = dspy.InputField(
        desc="ASCII map of remaining floors with room types and current position."
    )
    possible_encounters: str = dspy.InputField(
        desc="Possible remaining encounters by pool (weak/strong/elite/boss)."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief analysis of each choice given current state."
    )
    missing_context: str = dspy.OutputField(
        desc=(
            "What additional information would a human StS expert have that "
            "you lack for this decision? Be specific (e.g. 'exact deck list', "
            "'card pool for Scrap Ooze', 'which card Bonfire would upgrade'). "
            "Say 'none' if you have everything you need."
        )
    )
    choice_index: str = dspy.OutputField(
        desc="The 0-based index of the chosen option."
    )


# ---------------------------------------------------------------------------
# Card removal signature
# ---------------------------------------------------------------------------


class CardRemoveSignature(dspy.Signature):
    """You are a Slay the Spire strategy expert choosing a card to remove from your deck.

    Removing a card thins your deck, making it more likely to draw your best cards.
    Basic cards (Strike, Defend) are usually good removal targets.  Curse cards
    should almost always be removed.  Rare/powerful cards should almost never be removed.

    Use the ``try_remove_card`` tool to simulate how your deck performs without
    each candidate card against upcoming encounters.  Pick the card whose removal
    gives the best survival outcome.
    """

    character_state: str = dspy.InputField(
        desc="Current HP, max HP, gold, floor, deck list, potions, relics."
    )
    deck_cards: list[str] = dspy.InputField(
        desc="Full list of card IDs in your deck (may contain duplicates)."
    )
    map_view: str = dspy.InputField(
        desc="ASCII map of remaining floors with room types."
    )
    possible_encounters: str = dspy.InputField(
        desc="Possible remaining encounters by pool."
    )
    reasoning: str = dspy.OutputField(
        desc="Brief analysis of which card to remove and why."
    )
    card_id: str = dspy.OutputField(
        desc="The exact card_id string to remove from deck, or 'skip' to cancel."
    )


# ---------------------------------------------------------------------------
# Rest site signature
# ---------------------------------------------------------------------------


class RestPickSignature(dspy.Signature):
    """You are a Slay the Spire strategy expert deciding what to do at a Rest Site.

    You can either REST (heal 30% of max HP) or UPGRADE one card in your deck.

    Use the simulation tools to evaluate whether upgrading a specific card
    improves your survival in upcoming encounters more than healing would.
    Consider your current HP ratio, upcoming difficulty, and which upgrade
    gives the biggest power spike.

    Card upgrade effects are shown in each card's 'upgrade' field.
    The try_upgrade tool simulates with one card upgraded (replacing the
    unupgraded copy in deck). Compare results against simulate_upcoming
    (no upgrade) to see the delta.

    Simulation tools accept an optional encounter_id to target a specific
    enemy from the possible_encounters list.

    Output format: "REST" to heal, or "UPGRADE <card_id>" to upgrade a card.
    """


    character_state: str = dspy.InputField(
        desc="Current HP, max HP, deck composition, relics, potions, and gold."
    )
    upgradeable_cards: list[str] = dspy.InputField(
        desc="List of card IDs in deck that can still be upgraded (no + suffix)."
    )
    map_view: str = dspy.InputField(
        desc="ASCII map of remaining floors with room types and current position."
    )
    possible_encounters: str = dspy.InputField(
        desc="Possible remaining encounters by pool (weak/strong/elite/boss)."
    )
    pick: str = dspy.OutputField(
        desc='"REST" or "UPGRADE <card_id>".'
    )


# ---------------------------------------------------------------------------
# Strategy Agent
# ---------------------------------------------------------------------------


class StrategyAgent(BaseStrategyAgent):
    """LLM-based card-pick agent with simulation tools and timeout.

    Inherits random-valid-option defaults from :class:`BaseStrategyAgent`
    for every decision except :meth:`pick_card`, which is overridden to
    consult an LLM via dspy.ReAct.

    Parameters
    ----------
    timeout_seconds:
        Wall-clock budget per card pick decision (default 300 s = 5 min).
    model:
        LM model name (default ``"deepseek-v4-flash"``).
    seed:
        Forwarded to :class:`BaseStrategyAgent` for the inherited random
        decisions (Neow, route, rest, events, shop, boss relic).
    max_simulations:
        MCTS simulation budget per tool call (default 100).
    max_nodes:
        MCTS node-expansion budget per tool call (default 1000).

    Usage
    -----
    ::

        agent = StrategyAgent(timeout_seconds=300)
        pick = agent.pick_card(character, ["Anger", "Inflame", "Feed"],
                               upcoming, seed=42)
    """

    def __init__(
        self,
        timeout_seconds: int = DEFAULT_TIMEOUT,
        model: str = DEFAULT_MODEL,
        seed: int | None = None,
        max_simulations: int = 100,
        max_nodes: int = 1000,
    ) -> None:
        super().__init__(seed=seed)
        self.timeout = timeout_seconds
        self.model = model
        self.max_simulations = max_simulations
        self.max_nodes = max_nodes
        self._start_time: float = 0.0
        self._timed_out: bool = False
        self._sim_log: list[str] = []

    # -- budget enforcement --------------------------------------------------

    def _check_budget(self) -> None:
        """Raise :class:`TimeoutError` if the decision budget is exhausted."""
        if self._timed_out:
            raise TimeoutError("Strategy budget already exceeded")
        elapsed = time.time() - self._start_time
        if elapsed > self.timeout:
            self._timed_out = True
            raise TimeoutError(
                f"Strategy budget of {self.timeout}s exceeded "
                f"(elapsed: {elapsed:.1f}s)"
            )

    # -- main entry point ----------------------------------------------------

    @mlflow.trace(name="pick_card")
    def pick_card(
        self,
        character: Character,
        card_choices: list[str],
        upcoming_encounters: list[tuple[str, str]],
        seed: int,
        *,
        sts_map: StSMap | None = None,
        current_position: tuple[int, int] | None = None,
    ) -> str | None:
        """Pick a card from reward choices using LLM + simulation.

        Parameters
        ----------
        character:
            Current character state (deep-copied by simulate tools).
        card_choices:
            3 card IDs to choose from.
        upcoming_encounters:
            Accepted for protocol compatibility; not used — the LLM reads
            upcoming room types from the map view instead.
        seed:
            Master seed for deterministic simulations.
        sts_map:
            The full Act 1 map.  When provided (map mode), the LLM receives
            a rendered grid with room types and the current position marked.
        current_position:
            ``(floor, x)`` of the node just completed, used to mark the map.

        The orchestrator sets ``self.possible_encounters`` before calling
        this method.  It contains possible remaining encounters derived from
        open knowledge (pools + rules + encounters seen), with keys
        ``monster_weak``, ``monster_strong``, ``elite``, ``boss``.

        Returns
        -------
        The picked card ID, or ``None`` if skipped.
        """
        ensure_lm()
        self._start_time = time.time()
        self._timed_out = False
        self._sim_log = []

        card_infos = [_card_info(c) for c in card_choices]
        map_view = _format_map_view(sts_map, current_position)
        encounters_view = _format_possible_encounters(
            self.get_possible_encounters()
        )

        span = mlflow.get_current_active_span()
        if span:
            span.set_attributes({
                "character_state": character.summary(),
                "card_choices": ", ".join(c.card_id for c in card_infos),
                "seed": seed,
                "map_view": map_view[:500],  # truncate for span storage
            })

        # Create context-bound tools with configurable sim budget.
        # Tools catch TimeoutError and return a "budget exhausted" message
        # instead of propagating the exception, so the LLM gets one final
        # ReAct iteration to make a decision from collected results.
        tools = _make_tools(
            character, seed, self._check_budget,
            max_simulations=self.max_simulations,
            max_nodes=self.max_nodes,
            sim_log=self._sim_log,
        )

        try:
            react = dspy.ReAct(CardPickSignature, tools=tools, max_iters=8)
            result = react(
                character_state=character.summary(),
                card_choices=card_infos,
                map_view=map_view,
                possible_encounters=encounters_view,
            )
            raw_pick: str = getattr(result, "pick", "").strip()

            if raw_pick.lower() in ("skip", "none", ""):
                log.info("LLM strategy: skipped (%s)", card_choices)
                self._set_pick_attrs(None, fallback=False)
                return None

            # Exact match (case-insensitive)
            for info in card_infos:
                if info.card_id.lower() == raw_pick.lower():
                    log.info("LLM strategy: picked %s", info.card_id)
                    self._set_pick_attrs(info.card_id, fallback=False)
                    return info.card_id

            # Substring match (LLM sometimes adds spaces/words)
            for info in card_infos:
                if (
                    raw_pick.lower() in info.card_id.lower()
                    or info.card_id.lower() in raw_pick.lower()
                ):
                    log.info(
                        "LLM strategy: fuzzy-matched '%s' → %s",
                        raw_pick, info.card_id,
                    )
                    self._set_pick_attrs(info.card_id, fallback=False)
                    return info.card_id

            log.warning(
                "LLM returned '%s' not matching choices %s; forced pick",
                raw_pick, card_choices,
            )
            pick = self._forced_pick(card_infos)
            self._set_pick_attrs(pick, fallback=True)
            return pick

        except Exception as exc:
            elapsed = time.time() - self._start_time
            log.warning(
                "Strategy agent error after %.1fs (%d sims collected): %s; forced pick",
                elapsed, len(self._sim_log), exc,
            )
            pick = self._forced_pick(card_infos)
            self._set_pick_attrs(pick, fallback=True)
            return pick

    def _set_pick_attrs(self, pick: str | None, *, fallback: bool) -> None:
        """Set result attributes on the active MLflow span."""
        span = mlflow.get_current_active_span()
        if span is None:
            return
        elapsed = time.time() - self._start_time
        span.set_attributes({
            "pick": pick or "skip",
            "fallback": fallback,
            "elapsed_seconds": round(elapsed, 2),
            "timed_out": self._timed_out,
            "sim_count": len(self._sim_log),
        })

    # -- deterministic fallback ----------------------------------------------

    @staticmethod
    def _forced_pick(card_choices: list[CardInfo]) -> str | None:
        """Deterministic fallback when the LLM fails or times out.

        Priority: rare → uncommon → first card.
        """
        for info in card_choices:
            if info.rarity == "rare":
                return info.card_id
        for info in card_choices:
            if info.rarity == "uncommon":
                return info.card_id
        return card_choices[0].card_id if card_choices else None

    # -- event entry point ----------------------------------------------------

    @mlflow.trace(name="pick_event_choice")
    def pick_event_choice(
        self,
        event: "EventSpec",
        character: Character,
        **kwargs: object,
    ) -> int:
        """Choose an event branch using the LLM.

        Returns the 0-based index into ``event.choices``.
        Falls back to the first choice on error.
        """
        ensure_lm()

        if not event.choices:
            return 0

        # Build context --------------------------------------------------
        sts_map = kwargs.get("sts_map")
        current_position = kwargs.get("current_position")
        map_view = _format_map_view(sts_map, current_position)
        encounters_view = _format_possible_encounters(
            self.get_possible_encounters()
        )

        # Format choices with their labels
        choice_labels = [f"[{i}] {ch.label}" for i, ch in enumerate(event.choices)]

        # LLM call -------------------------------------------------------
        try:
            predictor = dspy.Predict(EventPickSignature)
            result = predictor(
                character_state=character.summary(),
                event_description=f"Event: {event.event_id}\n{event.description}",
                event_choices=choice_labels,
                map_view=map_view,
                possible_encounters=encounters_view,
            )

            raw_choice = getattr(result, "choice_index", "0").strip()
            reasoning = getattr(result, "reasoning", "")
            missing = getattr(result, "missing_context", "")

            # Parse choice index
            try:
                idx = int(raw_choice)
            except ValueError:
                # Try extracting first integer from the string
                import re
                m = re.search(r"\d+", raw_choice)
                idx = int(m.group()) if m else 0

            # Clamp to valid range
            idx = max(0, min(idx, len(event.choices) - 1))

            log.info(
                "LLM event %s: choice=%d (%s) | reasoning=%s | missing=%s",
                event.event_id, idx, event.choices[idx].label,
                reasoning[:120], missing[:120],
            )

            # Log missing context as MLflow attributes for analysis
            span = mlflow.get_current_active_span()
            if span:
                span.set_attributes({
                    "event_id": event.event_id,
                    "choice_idx": idx,
                    "choice_label": event.choices[idx].label,
                    "reasoning": reasoning[:500],
                    "missing_context": missing[:500],
                    "num_choices": len(event.choices),
                })

            return idx

        except Exception:
            log.exception(
                "LLM event %s: error, falling back to choice 0", event.event_id,
            )
            return 0

    # -- rest site entry point ------------------------------------------------

    @mlflow.trace(name="pick_rest_choice")
    def pick_rest_choice(
        self,
        character: Character,
        **kwargs: object,
    ) -> RestResult:
        """Decide REST vs UPGRADE at a rest site using LLM + simulation.

        Uses ``self.get_possible_encounters()`` to get fresh encounter info.
        """
        ensure_lm()
        self._start_time = time.time()
        self._timed_out = False
        self._sim_log = []

        # Gather upgradeable cards (unupgraded, with an upgrade delta)
        from sts_env.combat.cards import get_spec
        upgradeable = sorted({
            c for c in character.deck
            if not c.endswith("+")
            and get_spec(c).upgrade
        })

        sts_map = kwargs.get("sts_map")
        current_position = kwargs.get("current_position")
        map_view = _format_map_view(sts_map, current_position)
        encounters_view = _format_possible_encounters(
            self.get_possible_encounters()
        )

        tools = _make_rest_tools(
            character, self._check_budget,
            max_simulations=self.max_simulations,
            max_nodes=self.max_nodes,
            sim_log=self._sim_log,
        )

        try:
            react = dspy.ReAct(RestPickSignature, tools=tools, max_iters=8)
            result = react(
                character_state=character.summary(),
                upgradeable_cards=upgradeable,
                map_view=map_view,
                possible_encounters=encounters_view,
            )
            raw_pick: str = getattr(result, "pick", "").strip()

            if raw_pick.upper().startswith("UPGRADE"):
                parts = raw_pick.split(None, 1)
                if len(parts) == 2:
                    card_id = parts[1].strip()
                    # Verify the card is actually upgradeable
                    if card_id in upgradeable:
                        log.info("LLM strategy rest: upgrade %s", card_id)
                        return RestResult(
                            choice=RestChoice.UPGRADE,
                            card_upgraded=card_id,
                        )
                # Invalid upgrade target — fall through to REST
                log.warning("LLM strategy rest: invalid upgrade target %r, healing", raw_pick)

            # REST (default or explicit)
            log.info("LLM strategy rest: healing")
            return RestResult(choice=RestChoice.REST)

        except Exception:
            log.exception("LLM strategy rest: error, falling back to heal")
            return RestResult(choice=RestChoice.REST)

    # -- card removal entry point -----------------------------------------------

    @mlflow.trace(name="pick_card_to_remove")
    def pick_card_to_remove(
        self,
        character: Character,
        **kwargs: object,
    ) -> str | None:
        """Choose a card to remove from the deck.

        Uses ``try_remove_card`` simulation tool to evaluate the impact of
        removing each candidate.  Returns the card_id to remove, or None.
        """
        ensure_lm()
        self._start_time = time.time()
        self._timed_out = False
        self._sim_log = []

        # Candidates: all cards in deck (agent can simulate to decide)
        candidates = sorted(set(character.deck))

        sts_map = kwargs.get("sts_map")
        current_position = kwargs.get("current_position")
        map_view = _format_map_view(sts_map, current_position)
        encounters_view = _format_possible_encounters(
            self.get_possible_encounters()
        )

        tools = _make_card_removal_tools(
            character, self._check_budget,
            max_simulations=self.max_simulations,
            max_nodes=self.max_nodes,
            sim_log=self._sim_log,
        )

        try:
            react = dspy.ReAct(CardRemoveSignature, tools=tools, max_iters=8)
            result = react(
                character_state=character.summary(),
                deck_cards=list(character.deck),
                map_view=map_view,
                possible_encounters=encounters_view,
            )
            raw_pick: str = getattr(result, "card_id", "").strip()

            if raw_pick and raw_pick in character.deck:
                log.info("LLM card removal: removing %s", raw_pick)
                return raw_pick

            log.warning("LLM card removal: invalid pick %r, skipping", raw_pick)
            return None

        except Exception:
            log.exception("LLM card removal: error, skipping removal")
            return None
