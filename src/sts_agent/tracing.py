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
import os
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
        MLflow tracking URI.  Falls back to ``MLFLOW_TRACKING_URI``
        env var, then ``sqlite:///traces.db`` in the current working
        directory.
    """
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)

    mlflow.dspy.autolog()

    log.info("MLflow tracing enabled: experiment=%s uri=%s", experiment_name, uri)


def set_span_attributes(attrs: dict[str, object]) -> None:
    """Set attributes on the active MLflow span, if any."""
    span = mlflow.get_current_active_span()
    if span is not None:
        span.set_attributes(attrs)


def current_mlflow_linkage() -> dict[str, str]:
    """Run / trace / span IDs for correlating JSONL records with MLflow."""
    linkage: dict[str, str] = {}
    run = mlflow.active_run()
    if run is not None:
        linkage["mlflow_run_id"] = run.info.run_id
    trace_id = mlflow.get_active_trace_id()
    if trace_id:
        linkage["mlflow_trace_id"] = trace_id
    span = mlflow.get_current_active_span()
    if span is not None:
        linkage["mlflow_span_id"] = span.span_id
        if span.name:
            linkage["mlflow_span_name"] = span.name
    return linkage


def log_probe_artifact(name: str, payload: dict[str, object]) -> None:
    """Log structured probe data as an MLflow artifact when a run is active."""
    if not mlflow.active_run():
        return
    try:
        mlflow.log_dict(payload, f"probe_data/{name}.json")
    except Exception:
        log.debug("Failed to log probe artifact %s", name, exc_info=True)
