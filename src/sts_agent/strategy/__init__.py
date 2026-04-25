"""Strategy layer — simulation tools and agents for card reward decisions."""

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
