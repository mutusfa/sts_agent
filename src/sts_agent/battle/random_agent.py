"""RandomAgent — selects a uniformly random action from valid_actions().

Primary purpose: verify that the environment plumbing (reset / step /
valid_actions / done) works end-to-end before training any learned policy.
"""

from __future__ import annotations

import random

from sts_env.combat import Action, Observation


class RandomAgent:
    """Uniformly samples from the legal action set each turn.

    Parameters
    ----------
    seed:
        Seed for the internal RNG.  Same seed → same action sequence given the
        same sequence of valid_actions lists.
    """

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def act(self, obs: Observation, valid_actions: list[Action]) -> Action:  # noqa: ARG002
        return self._rng.choice(valid_actions)
