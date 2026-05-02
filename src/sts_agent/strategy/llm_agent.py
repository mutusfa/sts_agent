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
from typing import TYPE_CHECKING

import dspy
import mlflow
from pydantic import BaseModel
from dotenv import load_dotenv

from sts_env.run.character import Character

from .base import BaseStrategyAgent
from .simulate import SimResult, simulate_encounter, simulate_with_card

if TYPE_CHECKING:
    from sts_env.run.map import StSMap
    from sts_env.run.encounter_queue import EncounterQueue

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
    return (
        f"{label}: {status} | "
        f"damage_taken={result.damage_taken} | "
        f"max_hp_gained={result.max_hp_gained} | "
        f"final_hp={result.final_hp}/{result.final_max_hp} | "
        f"turns={result.turns}"
    )


def _make_tools(
    character: Character,
    upcoming: list[tuple[str, str]],
    seed: int,
    budget_checker: callable,
) -> list:
    """Create tool callables bound to the current decision context.

    Each tool checks the remaining time budget before running a simulation.
    """

    def simulate_upcoming(encounter_index: str) -> str:
        """Simulate an upcoming encounter with the current deck (baseline).

        Args:
            encounter_index: 0-based index into the upcoming encounters list.
        Returns:
            Human-readable simulation result (survived, damage, HP, turns).
        """
        budget_checker()
        idx = int(encounter_index)
        if idx < 0 or idx >= len(upcoming):
            return f"Error: index {idx} out of range [0, {len(upcoming) - 1}]"

        enc_type, enc_id = upcoming[idx]
        encounter_seed = seed * 1000 + idx
        result = simulate_encounter(
            character, enc_type, enc_id, encounter_seed,
            max_nodes=5000, simulations=5000,
        )
        return _format_result(f"Baseline ({enc_type}/{enc_id})", result)

    def try_card(card_id: str, encounter_index: str) -> str:
        """Simulate an encounter with a hypothetical card added to the deck.

        Args:
            card_id: The card ID to evaluate (one of the reward choices).
            encounter_index: 0-based index into the upcoming encounters list.
        Returns:
            Human-readable simulation result for the card + encounter.
        """
        budget_checker()
        idx = int(encounter_index)
        if idx < 0 or idx >= len(upcoming):
            return f"Error: index {idx} out of range [0, {len(upcoming) - 1}]"

        enc_type, enc_id = upcoming[idx]
        encounter_seed = seed * 1000 + idx
        result = simulate_with_card(
            character, card_id, enc_type, enc_id, encounter_seed,
            max_nodes=5000, simulations=5000,
        )
        return _format_result(f"With {card_id} vs {enc_type}/{enc_id}", result)

    return [simulate_upcoming, try_card]


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

    Act 1 encounter pool (canonical encounter ids):
    - Hallway weak: cultist, jaw_worm, two_louses, small_slimes
    - Hallway strong: gremlin_gang, lots_of_slimes, red_slaver, exordium_thugs,
      exordium_wildlife, blue_slaver, looter, large_slime, three_louse, two_fungi_beasts
    - Elites: Gremlin Nob, Lagavulin, Three Sentries
    - Bosses: slime_boss, guardian, hexaghost

    Map symbols: M=monster, E=elite, B=boss, R=rest, ?=event, $=shop, T=treasure.
    Current position is marked with @ in the map view.
    """

    character_state: str = dspy.InputField(
        desc="Current character: HP, deck size, potions, relics, floor"
    )
    card_choices: list[CardInfo] = dspy.InputField(
        desc="Cards offered as reward — each carries full spec, upgrade deltas, and custom handler code if any"
    )
    map_view: str = dspy.InputField(
        desc=(
            "Map nodes reachable from the current position, marked with @"
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
        LM model name (default ``"glm-5.1"``).
    seed:
        Forwarded to :class:`BaseStrategyAgent` for the inherited random
        decisions (Neow, route, rest, events, shop, boss relic).

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
    ) -> None:
        super().__init__(seed=seed)
        self.timeout = timeout_seconds
        self.model = model
        self._start_time: float = 0.0
        self._timed_out: bool = False

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
            Remaining ``(type, id)`` encounters in the act.
        seed:
            Master seed for deterministic simulations.
        sts_map:
            The full Act 1 map.  When provided (map mode), the LLM receives
            a rendered grid with room types and the current position marked.
        current_position:
            ``(floor, x)`` of the node just completed, used to mark the map.

        Returns
        -------
        The picked card ID, or ``None`` if skipped.
        """
        ensure_lm()
        self._start_time = time.time()
        self._timed_out = False

        card_infos = [_card_info(c) for c in card_choices]
        upcoming_str = "; ".join(
            f"{i}: {t}/{e}" for i, (t, e) in enumerate(upcoming_encounters)
        )
        map_view = _format_map_view(sts_map, current_position)

        span = mlflow.get_current_active_span()
        if span:
            span.set_attributes({
                "character_state": character.summary(),
                "card_choices": ", ".join(c.card_id for c in card_infos),
                "upcoming_encounters": upcoming_str,
                "seed": seed,
                "map_view": map_view[:500],  # truncate for span storage
            })

        # Create context-bound tools
        tools = _make_tools(
            character, upcoming_encounters, seed, self._check_budget
        )

        try:
            react = dspy.ReAct(CardPickSignature, tools=tools, max_iters=8)
            result = react(
                character_state=character.summary(),
                card_choices=card_infos,
                map_view=map_view,
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
            pick = self._forced_pick(character, card_infos, upcoming_encounters)
            self._set_pick_attrs(pick, fallback=True)
            return pick

        except TimeoutError:
            elapsed = time.time() - self._start_time
            log.warning(
                "Strategy timeout after %.1fs; forced pick", elapsed,
            )
            pick = self._forced_pick(character, card_infos, upcoming_encounters)
            self._set_pick_attrs(pick, fallback=True)
            return pick

        except Exception as exc:
            log.warning("Strategy agent error: %s; forced pick", exc)
            pick = self._forced_pick(character, card_infos, upcoming_encounters)
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
        })

    # -- deterministic fallback ----------------------------------------------

    @staticmethod
    def _forced_pick(
        character: Character,
        card_choices: list[CardInfo],
        upcoming_encounters: list[tuple[str, str]],
    ) -> str | None:
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
