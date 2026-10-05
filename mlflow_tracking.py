"""Local MLflow tracking for GridWorld CRM experiments."""

import json
import logging
import os
from pathlib import Path
from failure_diagnostics import failure_reason


DEFAULT_EXPERIMENT = "CRM GridWorld"


def _quiet_mlflow_logging():
    """Keep MLflow's asynchronous-tracing status messages out of the CLI."""
    logging.getLogger("mlflow").setLevel(logging.ERROR)


def configure_tracking():
    """Configure and return MLflow, or ``None`` when it is unavailable."""
    try:
        import mlflow
    except ImportError:
        return None
    _quiet_mlflow_logging()
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is not None and experiment.lifecycle_stage == "deleted":
        mlflow.MlflowClient().restore_experiment(experiment.experiment_id)
    mlflow.set_experiment(experiment_name)
    return mlflow


def next_task_id(mlflow):
    """Return the next numeric task label, ignoring legacy non-numeric IDs."""
    experiment = mlflow.get_experiment_by_name(
        os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT)
    )
    if experiment is None:
        return "1"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], output_format="list"
    )
    numeric_ids = [
        int(run.data.params["task_id"])
        for run in runs
        if run.data.params.get("task_id", "").isdigit()
    ]
    return str(max(numeric_ids, default=0) + 1)


def _difficulty_metrics(specifications):
    environment = specifications["difficulty"]["environment"]
    crm = specifications["difficulty"].get("crm") or {}
    return {
        "environment_required_object_count": environment["required_object_count"],
        "environment_hazard_count": environment["hazard_count"],
        "environment_wall_count": environment["wall_count"],
        "environment_minimum_steps_to_complete_trace": environment.get("minimum_steps_to_complete_trace") or 0,
        "crm_state_count": crm.get("states", 0),
        "crm_transition_count": crm.get("transitions", 0),
        "crm_counter_count": crm.get("counter_count", 0),
    }


def log_generation_failure(record):
    """Record an interrupted task, including difficulty when a grid is available."""
    mlflow = configure_tracking()
    if mlflow is None:
        raise RuntimeError("MLflow is not installed; run `python -m pip install mlflow`.")
    with mlflow.start_run(run_name=f"Task {record['task_id']}") as active_run:
        mlflow.log_params({
            "task_id": record["task_id"],
            "task_description": record["task_description"],
            "success_status": False,
            "generator_model": record["generator_model"],
            "judge_model": record["judge_model"],
            "judge_models": json.dumps(record.get("judge_models", [])),
            "iterative": record["iterative"],
            "long": record["long"],
            "failure_reason": failure_reason({**record, "success_status": False}),
            "failure_category": record.get("failure_category", "other"),
            "failure_stage": record.get("failure_stage", "unknown"),
            "generator_call_count": record["generator_call_count"],
            "judge_call_count": record.get("judge_call_count", 0),
            "iterations": record.get("iterations", 0),
        })
        mlflow.log_metrics({
            "total_system_time_seconds": record["total_system_time"],
            "average_generation_time_seconds": record["average_generation_time"],
            "crm_parse_fail_percentage": record.get("crm_parse_fail_percentage", 0.0),
            **({"overall_difficulty_score": record["overall_difficulty_score"]}
               if record.get("overall_difficulty_score") is not None else {}),
        })
        if record.get("specifications"):
            mlflow.log_metrics(_difficulty_metrics(record["specifications"]))
            mlflow.log_dict(record["specifications"], "specifications.json")
        if record.get("desired_trace"):
            mlflow.log_dict({
                "desired_trace": record["desired_trace"],
                "final_trace": record.get("final_trace", []),
            }, "traces.json")
        if record.get("judge_reflection"):
            mlflow.log_text(record["judge_reflection"], "judge_reflection.txt")
        mlflow.log_dict(record.get("generation_history", []), "generations.json")
        mlflow.log_dict(record.get("iteration_history", []), "iterations.json")
        return active_run.info.run_id


def log_experiment_run(record, animation_path=None):
    """Log a completed task run and return its MLflow run ID."""
    mlflow = configure_tracking()
    if mlflow is None:
        raise RuntimeError("MLflow is not installed; run `python -m pip install mlflow`.")

    reason = failure_reason(record)
    with mlflow.start_run(run_name=f"Task {record['task_id']}") as active_run:
        mlflow.log_params({
            "task_id": record["task_id"],
            "task_description": record["task_description"],
            "success_status": record["success_status"],
            **({"failure_reason": reason} if reason else {}),
            "failure_category": record.get("failure_category", "other") if reason else "none",
            "generator_model": record["generator_model"],
            "judge_model": record["judge_model"],
            "judge_models": json.dumps(record.get("judge_models", [])),
            "iterative": record["iterative"],
            "long": record["long"],
            "training_episodes": record["training_episodes"],
            "training_learning_rate": record["training_learning_rate"],
            "training_discount_factor": record["training_discount_factor"],
            "training_initial_epsilon": record["training_initial_epsilon"],
            "training_min_epsilon": record["training_min_epsilon"],
            "training_epsilon_decay": record["training_epsilon_decay"],
            "training_seed": record["training_seed"],
            "training_reward_shaping": record.get("training_reward_shaping", False),
            **{key: record[key] for key in (
                "performance_criterion", "training_step_efficiency_threshold",
                "training_reward_efficiency_threshold",
                "training_evaluation_interval",
            ) if key in record},
            "iterations": record["iterations"],
            "generator_call_count": record["generator_call_count"],
            "judge_call_count": record["judge_call_count"],
            "timing_version": record.get("timing_version", "legacy-v1"),
        })
        mlflow.log_metrics({
            "average_training_time_seconds": record["average_training_time"],
            "average_grid_world_generation_time_seconds": record["average_grid_world_generation_time"],
            "average_generation_time_seconds": record["average_generation_time"],
            "average_judge_time_seconds": record["average_judge_time"],
            "average_iteration_time_seconds": record["average_iteration_time"],
            "total_system_time_seconds": record["total_system_time"],
            "overall_difficulty_score": record["overall_difficulty_score"],
            "episode_reward": record["episode_reward"],
            "maximum_reward": record["maximum_reward"],
            "minimum_reward": record["minimum_reward"],
            "steps_taken": record["steps_taken"],
            **{key: record[key] for key in ("minimum_steps", "step_efficiency") if key in record},
            "reward_efficiency": record["reward_efficiency"],
            "crm_parse_fail_percentage": record["crm_parse_fail_percentage"],
        })
        if record.get("timing_breakdown"):
            mlflow.log_metrics(record["timing_breakdown"])
        mlflow.log_metrics(_difficulty_metrics(record["specifications"]))
        mlflow.log_dict(record["specifications"], "specifications.json")
        mlflow.log_dict(
            {"desired_trace": record["desired_trace"], "final_trace": record["final_trace"]},
            "traces.json",
        )
        mlflow.log_dict(record.get("iteration_history", []), "iterations.json")
        mlflow.log_dict(record.get("generation_history", []), "generations.json")
        if record["judge_reflection"]:
            mlflow.log_text(record["judge_reflection"], "judge_reflection.txt")
        if animation_path and Path(animation_path).is_file():
            mlflow.log_artifact(str(animation_path), artifact_path="animation")
        return active_run.info.run_id
