"""Structured probe/evaluation records for oracle data collection.

Records are appended to JSONL with a ``record_type`` discriminator so downstream
analysis can filter without re-running MCTS.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter, time
from typing import TYPE_CHECKING, Any, Iterator

from sts_agent.tracing import current_mlflow_linkage
from sts_env.run.changelog import CharacterChange, RoomRecord
from sts_env.run.orchestrator import PotionRecord

if TYPE_CHECKING:
    from sts_env.run.character import Character

    from .sim_agent import CardEvaluation
    from .simulate import SimDistribution

log = logging.getLogger(__name__)

_decision_deadline: ContextVar[float | None] = ContextVar(
    "decision_deadline", default=None
)

_probe_context: ContextVar[ProbeContext | None] = ContextVar(
    "probe_context", default=None
)


def check_decision_budget() -> None:
    """Raise :class:`TimeoutError` when the active decision budget is exhausted."""
    deadline = _decision_deadline.get()
    if deadline is not None and time() > deadline:
        raise TimeoutError("Decision budget exceeded")


@contextmanager
def decision_budget(timeout_seconds: float) -> Iterator[None]:
    """Wall-clock budget for one strategy decision (checked before each MCTS probe)."""
    if timeout_seconds <= 0:
        token = _decision_deadline.set(time())
        try:
            check_decision_budget()
            yield
        finally:
            _decision_deadline.reset(token)
        return

    token = _decision_deadline.set(time() + timeout_seconds)
    try:
        yield
    finally:
        _decision_deadline.reset(token)


def _json_safe(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sim_distribution_to_dict(dist: SimDistribution) -> dict[str, Any]:
    """Serialize a :class:`SimDistribution` for logging / JSONL."""
    return {
        "mean_score": _json_safe(dist.mean_score),
        "std_score": dist.std_score,
        "max_score": _json_safe(dist.max_score),
        "n": dist.n,
        "deaths": dist.deaths,
        "start_hp": dist.start_hp,
        "max_hp": dist.max_hp,
        "survival_rate": dist.survival_rate,
        "death_rate": dist.death_rate,
        "expected_damage": _json_safe(dist.expected_damage),
        "mean_damage_alive": _json_safe(dist.mean_damage_alive),
        "max_hp_gained_mean": dist.max_hp_gained_mean,
        "max_hp_gained_std": dist.max_hp_gained_std,
        "mean_enemy_dmg_dead": _json_safe(dist.mean_enemy_dmg_dead),
        "mean_turns_dead": _json_safe(dist.mean_turns_dead),
        "mean_damage_taken_dead": _json_safe(dist.mean_damage_taken_dead),
    }


def card_evaluation_to_dict(
    evaluation: CardEvaluation,
    *,
    encounters: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Serialize a :class:`CardEvaluation` with per-encounter distributions."""
    per_encounter: list[dict[str, Any]] = []
    for idx, dist in enumerate(evaluation.per_encounter):
        entry: dict[str, Any] = sim_distribution_to_dict(dist)
        if encounters and idx < len(encounters):
            enc_type, enc_id = encounters[idx]
            entry["encounter_type"] = enc_type
            entry["encounter_id"] = enc_id
        per_encounter.append(entry)
    return {
        "card_id": evaluation.card_id,
        "total_expected_damage": evaluation.total_expected_damage,
        "total_survival_rate": evaluation.total_survival_rate,
        "worst_max_score": evaluation.worst_max_score,
        "total_encounters": evaluation.total_encounters,
        "total_deaths": evaluation.total_deaths,
        "total_rollouts": evaluation.total_rollouts,
        "per_encounter": per_encounter,
    }


def character_state_snapshot(character: Character) -> dict[str, Any]:
    """Combat-entry-relevant state at a strategy decision point."""
    snap = character.snapshot()
    snap["deck_multiset"] = dict(Counter(character.deck))
    return snap


def character_probe_fingerprint(character: Character) -> tuple[Any, ...]:
    """Hashable combat-entry state for probe cache keys."""
    return (
        frozenset(Counter(character.deck).items()),
        tuple(sorted(character.relics)),
        tuple(character.potions),
        character.player_hp,
        character.player_max_hp,
    )


def probe_cache_key(
    character: Character,
    encounter_type: str,
    encounter_id: str,
    enc_seed: int,
    *,
    max_nodes: int,
    simulations: int,
    rollout_mode: str = "heuristic",
) -> tuple[Any, ...]:
    """Full cache key for one MCTS probe (encounter must be pool-resolved)."""
    return (
        *character_probe_fingerprint(character),
        encounter_type,
        encounter_id,
        enc_seed,
        max_nodes,
        simulations,
        rollout_mode,
    )


@dataclass
class ProbeStats:
    """Counters for probe cache effectiveness."""

    mcts_calls: int = 0
    cache_hits: int = 0
    mcts_seconds: float = 0.0
    mcts_by_decision_type: Counter[str] = field(default_factory=Counter)
    hits_by_decision_type: Counter[str] = field(default_factory=Counter)

    def summary(self) -> dict[str, Any]:
        return {
            "mcts_calls": self.mcts_calls,
            "cache_hits": self.cache_hits,
            "mcts_seconds": round(self.mcts_seconds, 3),
            "mcts_by_decision_type": dict(self.mcts_by_decision_type),
            "hits_by_decision_type": dict(self.hits_by_decision_type),
        }


class ProbeCache:
    """Run-scoped dedup cache for :func:`probe_encounter` results."""

    def __init__(self) -> None:
        self._store: dict[tuple[Any, ...], SimDistribution] = {}
        self.stats = ProbeStats()

    def clear(self) -> None:
        self._store.clear()
        self.stats = ProbeStats()

    def get(self, key: tuple[Any, ...]) -> SimDistribution | None:
        return self._store.get(key)

    def put(self, key: tuple[Any, ...], dist: SimDistribution) -> None:
        self._store[key] = dist

    def record_hit(self, decision_type: str | None) -> None:
        self.stats.cache_hits += 1
        if decision_type:
            self.stats.hits_by_decision_type[decision_type] += 1

    def record_miss(self, decision_type: str | None, elapsed_seconds: float) -> None:
        self.stats.mcts_calls += 1
        self.stats.mcts_seconds += elapsed_seconds
        if decision_type:
            self.stats.mcts_by_decision_type[decision_type] += 1


def character_change_to_dict(change: CharacterChange) -> dict[str, Any]:
    return {
        "field": change.field,
        "delta": change.delta,
        "value": change.value,
    }


def room_record_to_dict(record: RoomRecord) -> dict[str, Any]:
    return {
        "floor": record.floor,
        "room_type": record.room_type,
        "changes": [character_change_to_dict(c) for c in record.changes],
    }


def potion_record_to_dict(record: PotionRecord) -> dict[str, Any]:
    return {
        "potion_id": record.potion_id,
        "gained_floor": record.gained_floor,
        "spent_floor": record.spent_floor,
    }


@dataclass(frozen=True)
class ProbeContext:
    """Active decision context for atomic probe logging."""

    collector: ProbeCollector
    decision_type: str
    floor: int
    run_seed: int
    option: str | None = None


def get_probe_context() -> ProbeContext | None:
    return _probe_context.get()


@contextmanager
def probe_context(ctx: ProbeContext) -> Iterator[None]:
    """Set probe context for nested probe_encounter calls."""
    token = _probe_context.set(ctx)
    try:
        yield
    finally:
        _probe_context.reset(token)


@dataclass
class ProbeDecisionRecord:
    """One strategy decision with full probe evaluations."""

    decision_type: str
    floor: int
    seed: int
    character_state: dict[str, Any]
    pick: str | None = None
    card_choices: list[str] | None = None
    upcoming_encounters: list[tuple[str, str]] | None = None
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    map_view: str = ""
    run_seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["record_type"] = "decision"
        return payload


class ProbeCollector:
    """In-run probe store with optional JSONL persistence."""

    def __init__(
        self,
        jsonl_path: Path | str | None = None,
        *,
        run_seed: int | None = None,
    ) -> None:
        self.records: list[ProbeDecisionRecord] = []
        self.raw_records: list[dict[str, Any]] = []
        self._jsonl_path = Path(jsonl_path) if jsonl_path else None
        self._seen_keys: set[str] = set()
        self.run_seed = run_seed
        self.mlflow_run_id: str | None = None

    @staticmethod
    def _record_key(record: ProbeDecisionRecord) -> str:
        choices = ",".join(record.card_choices or [])
        enc = ",".join(f"{t}:{i}" for t, i in (record.upcoming_encounters or []))
        deck = json.dumps(
            record.character_state.get("deck_multiset", {}),
            sort_keys=True,
        )
        return f"{record.decision_type}|{record.floor}|{record.seed}|{choices}|{enc}|{deck}"

    def _enrich_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(payload)
        if self.run_seed is not None and "run_seed" not in enriched:
            enriched["run_seed"] = self.run_seed
        if self.mlflow_run_id is not None and "mlflow_run_id" not in enriched:
            enriched["mlflow_run_id"] = self.mlflow_run_id
        for key, value in current_mlflow_linkage().items():
            if key != "mlflow_run_id" and key not in enriched:
                enriched[key] = value
        return enriched

    def _append_jsonl_dict(self, payload: dict[str, Any]) -> None:
        if self._jsonl_path is None:
            return
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self._jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")

    def log_raw(self, payload: dict[str, Any]) -> None:
        """Append any record dict to memory and optional JSONL."""
        enriched = self._enrich_payload(payload)
        self.raw_records.append(enriched)
        self._append_jsonl_dict(enriched)

    def log_probe(
        self,
        dist: SimDistribution,
        *,
        encounter_type: str,
        encounter_id: str,
        enc_seed: int,
        character: Character | None = None,
        ctx: ProbeContext | None = None,
    ) -> None:
        """Log one atomic MCTS probe result (no dedup)."""
        active = ctx or get_probe_context()
        if active is None:
            return
        payload: dict[str, Any] = {
            "record_type": "probe",
            "decision_type": active.decision_type,
            "floor": active.floor,
            "run_seed": active.run_seed,
            "option": active.option,
            "encounter_type": encounter_type,
            "encounter_id": encounter_id,
            "enc_seed": enc_seed,
            "distribution": sim_distribution_to_dict(dist),
        }
        if character is not None:
            payload["character_state"] = character_state_snapshot(character)
        self.log_raw(payload)

    def log_path_step(
        self,
        *,
        floor: int,
        coord: tuple[int, int],
        room_type: str | None = None,
        committed_path: list[tuple[int, int]] | None = None,
    ) -> None:
        self.log_raw(
            {
                "record_type": "path_step",
                "floor": floor,
                "coord": list(coord),
                "room_type": room_type,
                "committed_path": [list(c) for c in (committed_path or [])],
            }
        )

    def log_room_delta(self, record: RoomRecord) -> None:
        payload = room_record_to_dict(record)
        payload["record_type"] = "room_delta"
        self.log_raw(payload)

    def log_outcome(self, payload: dict[str, Any]) -> None:
        self.log_raw({"record_type": "outcome", **payload})

    def log_potion_spend(self, payload: dict[str, Any]) -> None:
        self.log_raw({"record_type": "potion_spend", **payload})

    def log_potion_lifecycle(self, record: PotionRecord) -> None:
        payload = potion_record_to_dict(record)
        payload["record_type"] = "potion_lifecycle"
        self.log_raw(payload)

    def record(self, record: ProbeDecisionRecord) -> None:
        """Append a decision record once (deduped by decision context)."""
        if record.run_seed is None and self.run_seed is not None:
            record.run_seed = self.run_seed
        key = self._record_key(record)
        if key in self._seen_keys:
            return
        self._seen_keys.add(key)
        self.records.append(record)
        self.log_raw(record.to_dict())

    def flush_room_log(self, room_log: list[RoomRecord]) -> None:
        for entry in room_log:
            self.log_room_delta(entry)

    def flush_potion_log(self, potion_log: list[PotionRecord]) -> None:
        for entry in potion_log:
            self.log_potion_lifecycle(entry)

    def flush(self) -> None:
        """No-op when using append-on-record JSONL; kept for API symmetry."""
        return

    def count_by_record_type(self) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for row in self.raw_records:
            counts[row.get("record_type", "unknown")] += 1
        return dict(counts)
