"""
uv run --isolated --extra dev --extra skyrl-train pytest -s tests/train/test_tracking.py
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from skyrl.train.utils.tracking import Tracking


def _mlflow_log_table_call(now):
    """Run log_samples_to_table against a mocked mlflow module (clock pinned to
    ``now``) and return the mlflow mock so callers can inspect log_table."""
    mlflow = MagicMock()
    # Truthy active_run so construction skips start_run.
    mlflow.active_run.return_value = SimpleNamespace(info=SimpleNamespace(run_id="run123"))
    with patch.dict("sys.modules", {"mlflow": mlflow}), patch("time.time", return_value=now):
        # Non-empty config: _compute_mlflow_params_from_objects runs pandas
        # json_normalize at construction, which IndexErrors on an empty dict.
        tracker = Tracking(project_name="proj", experiment_name="exp", backend="mlflow", config={"lr": 0.1})
        tracker.log_samples_to_table(
            key="trajectories_eval",
            columns=["step", "reward"],
            samples=[(250, 1.0)],
            step=250,
        )
    return mlflow


def test_mlflow_log_table_artifact_is_step_and_timestamp_named():
    """The artifact file zero-pads the step and carries an epoch-seconds suffix."""
    mlflow = _mlflow_log_table_call(now=1735689600.9)
    mlflow.log_table.assert_called_once()
    assert mlflow.log_table.call_args.kwargs["artifact_file"] == "trajectories_eval_step_0000250_1735689600.json"


def test_mlflow_log_table_name_changes_across_resumed_attempts():
    """A resumed attempt re-logs the same step at a later wall-clock time, so it
    writes a distinct artifact instead of colliding with the prior (possibly
    truncated) one -- which is what triggered the read-append JSON-parse crash."""
    first = _mlflow_log_table_call(now=1735689600.0)
    second = _mlflow_log_table_call(now=1735693200.0)
    assert (
        first.log_table.call_args.kwargs["artifact_file"]
        != second.log_table.call_args.kwargs["artifact_file"]
    )


def test_wandb_init_receives_tags():
    """Tags passed to Tracking are forwarded to wandb.init."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        Tracking(
            project_name="proj",
            experiment_name="exp",
            backend="wandb",
            config={},
            tags=["foo", "bar"],
        )

        wandb_mock.init.assert_called_once()
        kwargs = wandb_mock.init.call_args.kwargs
        assert kwargs["tags"] == ["foo", "bar"]
        assert kwargs["project"] == "proj"
        assert kwargs["name"] == "exp"


def test_wandb_init_tags_default_none():
    """When tags are not provided, wandb.init receives tags=None."""
    with patch.dict("sys.modules", {"wandb": MagicMock()}) as mocked:
        wandb_mock = mocked["wandb"]
        Tracking(
            project_name="proj",
            experiment_name="exp",
            backend="wandb",
            config={},
        )

        wandb_mock.init.assert_called_once()
        assert wandb_mock.init.call_args.kwargs["tags"] is None
