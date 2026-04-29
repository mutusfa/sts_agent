"""MLflow tracing setup for sts-agent.

Enables automatic DSPy LLM call tracing via ``mlflow.dspy.autolog()``
and configures the MLflow experiment / tracking URI.

Usage
-----
::

    from sts_agent.tracing import setup_tracing

    setup_tracing()                # defaults: experiment="sts-agent", sqlite in project root
    setup_tracing(experiment_name="my-experiment", tracking_uri="sqlite:///custom.db")
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow

log = logging.getLogger(__name__)

DEFAULT_EXPERIMENT = "sts-agent"
DEFAULT_TRACKING_URI = f"sqlite:///{Path.cwd() / 'traces.db'}"


def setup_tracing(
    *,
    experiment_name: str = DEFAULT_EXPERIMENT,
    tracking_uri: str | None = None,
) -> None:
    """Configure MLflow tracing with DSPy autolog.

    Parameters
    ----------
    experiment_name:
        MLflow experiment name (default ``"sts-agent"``).
    tracking_uri:
        MLflow tracking URI.  Defaults to ``sqlite:///traces.db``
        in the current working directory.
    """
    uri = tracking_uri or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    mlflow.dspy.autolog()

    log.info("MLflow tracing enabled: experiment=%s uri=%s", experiment_name, uri)
