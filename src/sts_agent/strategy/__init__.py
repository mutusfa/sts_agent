"""Strategy layer — simulation tools and agents for run-level decisions."""

from .base import BaseStrategyAgent
from .llm_agent import StrategyAgent, configure_lm, ensure_lm
from .sim_agent import SimStrategyAgent
from .simulate import (
    SimDistribution,
    SimResult,
    probe_encounter,
    probe_with_card,
    simulate_encounter,
    simulate_with_card,
)

__all__ = [
    "BaseStrategyAgent",
    "SimStrategyAgent",
    "StrategyAgent",
    "SimDistribution",
    "SimResult",
    "configure_lm",
    "ensure_lm",
    "probe_encounter",
    "probe_with_card",
    "simulate_encounter",
    "simulate_with_card",
]
