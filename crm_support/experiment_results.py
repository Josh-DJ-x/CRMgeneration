"""Shared result loading and summaries for the dashboard and research reports."""

import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient
from sqlalchemy.engine import make_url
from crm_support.failure_diagnostics import failure_reason

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = [
    "iterations", "generator_call_count", "judge_call_count",
    "total_system_time_seconds", "overall_difficulty_score", "reward_efficiency",
    "minimum_reward", "environment_minimum_steps_to_complete_trace",
    "crm_parse_fail_percentage", "episode_reward", "maximum_reward", "steps_taken",
    "minimum_steps", "step_efficiency", "training_step_efficiency_threshold",
    "training_reward_efficiency_threshold",
    "average_training_time_seconds", "average_grid_world_generation_time_seconds",
    "average_generation_time_seconds", "average_judge_time_seconds",
    "average_iteration_time_seconds",
    "misc_time_seconds", "total_generation_time_seconds", "total_judge_time_seconds",
    "total_training_time_seconds", "total_grid_world_generation_time_seconds",
]


def database_path(uri=None):
    """Resolve only a file-backed SQLite tracking database."""
    url = make_url(uri or tracking_uri())
    if url.get_backend_name() != "sqlite" or not url.database or url.database == ":memory:":
        raise ValueError("Database downloads require a local SQLite tracking database.")
    return Path(url.database).resolve()


def database_snapshot(uri=None):
    """Export a consistent SQLite snapshot, including committed WAL contents."""
    source_path = database_path(uri)
    with tempfile.TemporaryDirectory() as directory:
        target_path = Path(directory) / "mlflow.db"
        source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True)
        target = sqlite3.connect(target_path)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        return target_path.read_bytes()


def runtime_breakdown(row):
    """Present activity totals, miscellaneous work, and the overall runtime."""
    values = []
    for total, average, count, label in [
        ("total_generation_time_seconds", "average_generation_time_seconds", "generator_call_count", "CRM generation (all calls)"),
        ("total_judge_time_seconds", "average_judge_time_seconds", "judge_call_count", "Judge (all calls)"),
        ("total_training_time_seconds", "average_training_time_seconds", "iterations", "Training (all iterations)"),
        ("total_grid_world_generation_time_seconds", "average_grid_world_generation_time_seconds", "iterations", "Grid generation (all iterations)"),
    ]:
        seconds = row.get(total)
        if pd.isna(seconds):
            mean, calls = row.get(average), row.get(count)
            seconds = mean * calls if pd.notna(mean) and pd.notna(calls) else float("nan")
        values.append((label, seconds))
    total = row.get("total_system_time_seconds")
    misc = row.get("misc_time_seconds")
    if pd.isna(misc) and pd.notna(total) and all(pd.notna(value) for _, value in values):
        misc = max(0.0, total - sum(value for _, value in values))
    values.extend([("Misc time", misc), ("Total system time", total)])
    return pd.DataFrame(values, columns=["Activity", "Seconds"])


def judge_label(requested, resolved, calls=None, backend="openrouter"):
    """Keep requested routing separate from models reported by the responses."""
    if isinstance(resolved, str):
        try:
            resolved = json.loads(resolved)
        except (TypeError, ValueError):
            resolved = None
    if isinstance(resolved, list):
        provider = {"openai": "OpenAI", "openai+openrouter": "OpenAI + OpenRouter"}.get(backend, "OpenRouter")
        if resolved:
            return provider + " (" + ", ".join(dict.fromkeys(resolved)) + ")"
        if calls == 0:
            return provider + " (not called)"
        return provider + " (actual model not recorded)"
    if str(requested).startswith("openrouter/"):
        return "OpenRouter (actual model not recorded)"
    return requested


def overview_figure(frame, all_runs):
    """Pair dark successes with light failures for each generator, across filters."""
    import colorsys
    import plotly.express as px

    models = all_runs.sort_values(["started_at", "run_id"]).generator_model.astype(str).unique()
    pairs = [
        ("#2457A7", "#A9CDF5"), ("#24703C", "#ABE0B5"),
        ("#6941A5", "#D3B8EF"), ("#A85B10", "#F5CEA0"),
        ("#147575", "#A2DEDE"), ("#A33C79", "#EDB6D7"),
    ]
    colours, categories = {}, []
    for index, model in enumerate(models):
        if index < len(pairs):
            dark, light = pairs[index]
        else:
            hue = (0.61 + index * 0.61803398875) % 1
            def shade(lightness):
                rgb = colorsys.hls_to_rgb(hue, lightness, 0.55)
                return "#" + "".join(f"{round(channel * 255):02X}" for channel in rgb)
            dark, light = shade(0.32), shade(0.80)
        for outcome, colour in [("Success", dark), ("Failure", light), ("Unknown", "#808080")]:
            label = f"{model} · {outcome}"
            categories.append(label)
            colours[label] = colour
    points = frame.copy()
    points["Generator / outcome"] = points.generator_model.astype(str) + " · " + points.outcome
    figure = px.scatter(
        points, x="overall_difficulty_score", y="total_system_time_seconds",
        color="Generator / outcome", symbol="generator_model",
        color_discrete_map=colours,
        category_orders={"Generator / outcome": categories, "generator_model": list(models)},
        hover_data=["task_description", "run_id", "outcome"],
        labels={"overall_difficulty_score": "Difficulty score",
                "total_system_time_seconds": "Total runtime (seconds)"},
    )
    # Plotly appends the symbol category to the colour category. The colour
    # label already includes the generator, so remove that duplicate suffix.
    for trace in figure.data:
        trace.name = trace.name.rsplit(", ", 1)[0]
    return figure


def comparison_figure(frame, metric):
    """Keep every filtered model on the axis even if its metric is missing."""
    import plotly.express as px
    models = sorted(frame.generator_model.astype(str).unique())
    points = frame.dropna(subset=[metric])
    figure = px.strip(points, x="generator_model", y=metric, color="generator_model",
                      category_orders={"generator_model": models},
                      hover_data=["task_description", "run_id", "outcome"])
    for trace in figure.data:
        if len(trace.name) > 48:
            trace.name = trace.name[:45] + "..."
    figure.update_layout(height=640, autosize=True,
                         margin={"l": 70, "r": 30, "t": 150, "b": 130},
                         legend={"orientation": "h", "x": 0, "xanchor": "left",
                                 "y": 1.02, "yanchor": "bottom", "maxheight": 120,
                                 "entrywidth": 0.5, "entrywidthmode": "fraction",
                                 "title": {"text": "Generator LLM"}},
                         xaxis={"type": "category", "tickangle": -35,
                                "categoryorder": "array", "categoryarray": models,
                                "tickmode": "array", "tickvals": models, "ticktext": models,
                                "range": [-0.5, len(models) - 0.5]})
    return figure


def difficulty_summary(frame, metric=None, groups=("generator_model",)):
    """Aggregate runs in [0.0,0.1), …, [0.9,1.0] difficulty bins.

    Metric means omit missing values; success is a percentage of known outcomes.
    Empty bins remain explicit so plotted lines do not bridge missing samples.
    """
    groups = list(groups)
    columns = groups + ["difficulty", "interval", "value", "samples", "runs"]
    data = frame[frame.overall_difficulty_score.between(0, 1)].copy()
    if data.empty:
        return pd.DataFrame(columns=columns)
    edges = [i / 10 for i in range(11)]
    edges[-1] += 1e-10  # Include an exact difficulty of 1.0 in the final bin.
    data["bin"] = pd.cut(data.overall_difficulty_score, edges, right=False, labels=False)
    by_bin = data.groupby(groups + ["bin"], observed=True, dropna=False)
    if metric is None:
        values = by_bin.size().rename("value").reset_index()
        values["samples"] = values["runs"] = values["value"]
    else:
        data["measure"] = pd.to_numeric(data[metric], errors="coerce")
        if metric == "success":
            data["measure"] *= 100
        values = data.groupby(groups + ["bin"], observed=True, dropna=False).agg(
            value=("measure", "mean"), samples=("measure", "count"), runs=("measure", "size"),
        ).reset_index()
    bins = pd.DataFrame({
        "bin": range(10), "difficulty": [(i + 0.5) / 10 for i in range(10)],
        "interval": [f"{i / 10:.1f}–{(i + 1) / 10:.1f}" for i in range(10)],
    })
    grid = data[groups].drop_duplicates().merge(bins, how="cross")
    result = grid.merge(values, on=groups + ["bin"], how="left")
    result[["samples", "runs"]] = result[["samples", "runs"]].fillna(0).astype(int)
    if metric is None:
        result["value"] = result["value"].fillna(0).astype(int)
    return result[columns]


def balanced_success_summary(frame):
    """Give every populated difficulty interval equal weight per model and setting."""
    groups = ["generator_model", "specification_length", "iterative_mode"]
    bins = difficulty_summary(frame, "success", groups)
    populated = bins[bins.samples.gt(0)]
    return populated.groupby(groups, observed=True, dropna=False).agg(
        success_percentage=("value", "mean"),
        difficulty_intervals=("value", "count"),
        scored_runs=("samples", "sum"),
    ).reset_index()


def difficulty_figure(frame, metric, label, split_modes=False):
    """Plot bin averages by generator, optionally splitting mode and prompt length."""
    import plotly.express as px
    import plotly.graph_objects as go
    groups = ["generator_model"]
    options = {}
    if split_modes:
        groups += ["specification_length", "iterative_mode"]
        options = {"line_dash": "specification_length", "facet_col": "iterative_mode"}
    data = difficulty_summary(frame, metric, groups)
    if data.empty or data.value.notna().sum() == 0:
        figure = go.Figure()
        figure.add_annotation(text="No recorded values for these filters.", showarrow=False)
    else:
        figure = px.line(
            data, x="difficulty", y="value", color="generator_model", markers=True,
            hover_data={"interval": True, "samples": True, "runs": True, "difficulty": False},
            labels={"difficulty": "Difficulty", "value": label, "generator_model": "LLM",
                    "specification_length": "Specifications", "iterative_mode": "Mode",
                    "interval": "Difficulty interval", "samples": "Recorded values", "runs": "Runs"},
            **options,
        )
        figure.update_traces(connectgaps=False)
    figure.update_xaxes(tick0=0, dtick=0.1, range=[0, 1], fixedrange=True,
                        minallowed=0, maxallowed=1, autorange=False,
                        tickformat=".1f", automargin=True, title_text="Difficulty")
    figure.update_yaxes(title_text=label, autorange=True)
    if metric in {"success", "crm_parse_fail_percentage"}:
        figure.update_yaxes(range=[0, 100], autorange=False)
    return figure


def difficulty_distribution_figure(frame):
    import plotly.express as px
    data = difficulty_summary(frame)
    figure = px.bar(data, x="difficulty", y="value", color="generator_model",
                    barmode="group", hover_data=["interval"],
                    labels={"difficulty": "Difficulty", "value": "Experiment count",
                            "generator_model": "LLM", "interval": "Difficulty interval"})
    figure.update_xaxes(tick0=0, dtick=0.1, range=[0, 1], fixedrange=True,
                        tickformat=".1f", automargin=True)
    return figure


def tracking_uri():
    return os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")


def client(uri=None):
    return MlflowClient(tracking_uri=uri or tracking_uri())


def experiments(uri=None):
    api = client(uri)
    result, token = [], None
    while True:
        page = api.search_experiments(max_results=1000, page_token=token)
        result.extend(
            (e.experiment_id, e.name)
            for e in page
            if e.name not in {"Default", "MLflow Demo"}
        )
        token = page.token
        if not token:
            return result


def delete_run(run_id, uri=None):
    """Soft-delete only the selected task run through the tracking store."""
    client(uri).delete_run(run_id)


def set_manual_failure(run_id, reason="", uri=None, clear=False):
    """Mark any run as failed, or restore its original automatic outcome."""
    api = client(uri)
    run = api.get_run(run_id)
    if clear:
        for key in ("manual_outcome", "manual_failure_reason"):
            if key in run.data.tags:
                api.delete_tag(run_id, key)
    else:
        api.set_tag(run_id, "manual_failure_reason", reason.strip() or "Performance rejected during manual review.")
        api.set_tag(run_id, "manual_outcome", "Failure")


def set_manual_success(run_id, uri=None):
    """Override any run as successful without changing recorded parameters."""
    api = client(uri)
    api.get_run(run_id)
    api.set_tag(run_id, "manual_outcome", "Success")


def backfill_failure_reasons(experiment_id, uri=None):
    """Fill only missing reasons on unsuccessful runs using their saved evidence."""
    api = client(uri)
    repaired, token = [], None
    while True:
        page = api.search_runs([experiment_id], max_results=1000, page_token=token)
        for run in page:
            params = run.data.params
            unsuccessful = str(params.get("success_status", "")).lower() in {"false", "0"}
            if not unsuccessful and run.info.status != "FAILED":
                continue
            if params.get("failure_reason", "").strip():
                continue
            artifacts = load_artifacts(run.info.run_id, uri)
            record = {**params, **run.data.metrics, "success_status": False,
                      "iteration_history": artifacts.get("iterations.json", [])}
            traces = artifacts.get("traces.json", {})
            if isinstance(traces, dict):
                record.update({key: traces[key] for key in ("desired_trace", "final_trace") if key in traces})
            reason = failure_reason(record)
            api.log_param(run.info.run_id, "failure_reason", reason)
            repaired.append({"run_id": run.info.run_id, "task_id": params.get("task_id"), "failure_reason": reason})
        token = page.token
        if not token:
            return repaired


def load_runs(experiment_id, uri=None):
    api = client(uri)
    rows, token = [], None
    while True:
        page = api.search_runs(
            [experiment_id], max_results=1000, page_token=token,
            order_by=["attributes.start_time DESC"],
        )
        for run in page:
            rows.append({
                **run.data.params, **run.data.metrics,
                **{key: value for key, value in getattr(run.data, "tags", {}).items()
                   if key in {"manual_outcome", "manual_failure_reason"}},
                "run_id": run.info.run_id, "run_status": run.info.status,
                "started_at": run.info.start_time,
            })
        token = page.token
        if not token:
            break
    frame = pd.DataFrame(rows)
    for field in ["task_id", "task_description", "generator_model", "judge_model",
                  "success_status", "failure_reason", "failure_category", "iterative", "long", "run_id",
                  "run_status", "started_at", "manual_outcome", "manual_failure_reason"]:
        if field not in frame:
            frame[field] = pd.Series(index=frame.index, dtype="object")
    for field in NUMERIC:
        frame[field] = pd.to_numeric(frame.get(field, pd.Series(dtype=float)), errors="coerce")
    # Task 123 and similar capped runs can borrow the newest older training
    # metrics for their task description when no training occurred in the run.
    for index, row in frame.iterrows():
        capped = str(row.get("failure_reason", "")).startswith("Generator call limit reached")
        task_123 = str(row.get("task_id", "")) == "123"
        if not (capped or task_123):
            continue
        missing = row[["reward_efficiency", "step_efficiency"]].isna()
        if not missing.any():
            continue
        same_task = frame["task_description"].eq(row["task_description"])
        if task_123:
            # Task IDs are assigned at session start, so session start order
            # uses this task's prior training run as the latest candidate.
            same_task = frame.index.to_series(index=frame.index).notna()
        previous = frame.loc[
            (frame.index != index)
            & same_task
            & (frame["started_at"] < row["started_at"])
            & frame[["reward_efficiency", "step_efficiency"]].notna().any(axis=1)
        ].sort_values(["started_at", "run_id"], ascending=[False, False])
        if previous.empty:
            continue
        for metric in ("reward_efficiency", "step_efficiency"):
            if pd.isna(row[metric]):
                frame.at[index, metric] = previous.iloc[0][metric]
    frame["minimum_steps"] = frame["minimum_steps"].fillna(
        frame["environment_minimum_steps_to_complete_trace"].where(
            frame["environment_minimum_steps_to_complete_trace"].gt(0)
        )
    )
    frame["step_efficiency"] = frame["step_efficiency"].fillna(
        frame["minimum_steps"] / frame["steps_taken"].where(frame["steps_taken"].gt(0))
    )
    # Historical values remain in the database, but retired fields are
    # excluded from the active dashboard and new CSV exports.
    frame = frame.drop(columns=[
        "trace_verification_failures", "difficulty_version", "setup_time_seconds",
        "total_evaluation_time_seconds", "postprocessing_time_seconds", "orchestration_time_seconds",
    ], errors="ignore")
    missing_misc = frame["misc_time_seconds"].isna()
    if missing_misc.any():
        frame.loc[missing_misc, "misc_time_seconds"] = frame.loc[missing_misc].apply(
            lambda row: runtime_breakdown(row).iloc[-2]["Seconds"], axis=1
        )
    frame["started_at"] = pd.to_datetime(frame["started_at"], unit="ms", utc=True)
    frame["task_number"] = pd.to_numeric(frame["task_id"], errors="coerce").astype("Int64")
    frame["automatic_success_status"] = frame["success_status"]
    manual_failure = frame["manual_outcome"].eq("Failure")
    frame.loc[manual_failure, "success_status"] = "False"
    frame.loc[manual_failure, "failure_reason"] = frame.loc[manual_failure, "manual_failure_reason"].fillna("Performance rejected during manual review.")
    frame.loc[manual_failure, "failure_category"] = "manual_review"
    manual_success = frame["manual_outcome"].eq("Success")
    frame.loc[manual_success, "success_status"] = "True"
    frame.loc[manual_success, "failure_reason"] = None
    frame.loc[manual_success, "failure_category"] = "none"
    frame["success"] = frame["success_status"].astype("string").str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).astype("boolean")
    frame["outcome"] = frame["success"].map({True: "Success", False: "Failure"}).fillna("Unknown")
    frame["iterative_mode"] = frame["iterative"].astype("string").str.lower().map(
        {"true": "Iterative", "false": "One-shot", "1": "Iterative", "0": "One-shot"}
    ).fillna("Iterative")  # Runs predating this parameter used the iterative loop.
    frame["specification_length"] = frame["long"].astype("string").str.lower().map(
        {"true": "Long", "false": "Short", "1": "Long", "0": "Short"}
    ).fillna("Long")  # Runs predating this parameter used the original specifications.
    for field in ["generator_model", "judge_model", "task_description"]:
        frame[field] = frame[field].fillna("Unknown")
    # NVIDIA Build and OpenRouter's free endpoint serve the same generator.
    # Group their experiments together while retaining the recorded ID.
    frame["generator_model_requested"] = frame["generator_model"]
    frame["generator_model"] = frame["generator_model"].replace({
        "nvidia/nemotron-3-super-120b-a12b:free": "nvidia/nemotron-3-super-120b-a12b",
    })
    frame["judge_model_requested"] = frame["judge_model"]
    frame["judge_model"] = frame.apply(
        lambda row: judge_label(row["judge_model_requested"], row.get("judge_models"),
                                row["judge_call_count"], row.get("judge_backend", "openrouter")), axis=1
    ) if not frame.empty else frame["judge_model"]
    return frame


def summarize(frame, groups):
    """Unknown outcomes are excluded from the success-rate denominator."""
    return frame.groupby(groups, dropna=False).agg(
        runs=("run_id", "size"), runs_with_recorded_outcome=("success", "count"),
        success_rate=("success", "mean"),
        mean_runtime_seconds=("total_system_time_seconds", "mean"),
        mean_reward_efficiency=("reward_efficiency", "mean"),
        mean_parse_fail_percentage=("crm_parse_fail_percentage", "mean"),
    ).reset_index()


def load_artifacts(run_id, uri=None):
    """Read artifacts independently so a missing legacy file cannot hide a run."""
    api = client(uri)
    result = {"warnings": [], "animations": []}
    try:
        available = {item.path for item in api.list_artifacts(run_id)}
    except Exception as error:
        result["warnings"].append(f"Cannot list artifacts: {error}")
        return result
    for name in ["specifications.json", "traces.json", "iterations.json", "generations.json", "judge_reflection.txt"]:
        if name not in available:
            continue
        try:
            path = Path(api.download_artifacts(run_id, name))
            value = path.read_text(encoding="utf-8")
            result[name] = json.loads(value) if name.endswith(".json") else value
        except Exception as error:
            result["warnings"].append(f"Cannot read {name}: {error}")
    if "animation" in available:
        try:
            for item in api.list_artifacts(run_id, "animation"):
                if not item.is_dir and Path(item.path).suffix.lower() in {".gif", ".png"}:
                    path = Path(api.download_artifacts(run_id, item.path))
                    result["animations"].append((path.name, path.read_bytes()))
        except Exception as error:
            result["warnings"].append(f"Cannot read animation: {error}")
    return result


def trace_table(desired, actual):
    length = max(len(desired), len(actual))
    return pd.DataFrame([
        {"Step": i + 1, "Desired": str(desired[i]) if i < len(desired) else "—",
         "Actual": str(actual[i]) if i < len(actual) else "—",
         "Match": i < len(desired) and i < len(actual) and desired[i] == actual[i]}
        for i in range(length)
    ], columns=["Step", "Desired", "Actual", "Match"])
