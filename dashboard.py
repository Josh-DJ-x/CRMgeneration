"""Run with: python -m streamlit run dashboard.py"""

import json

import pandas as pd
import plotly.express as px
import streamlit as st
from failure_diagnostics import effective_failure_category

from experiment_results import (comparison_figure, experiments, load_artifacts, load_runs,
                                summarize, trace_table, tracking_uri, database_path,
                                database_snapshot, runtime_breakdown, delete_run,
                                difficulty_figure, difficulty_distribution_figure)

st.set_page_config(page_title="CRM experiment results", page_icon="🧭", layout="wide")
st.title("CRM experiment results")
st.caption("Explore GridWorld tasks, inspect agent behaviour, and compare recorded outcomes.")

cached_runs = st.cache_data(ttl=60)(load_runs)
cached_artifacts = st.cache_data(ttl=60)(load_artifacts)
uri = tracking_uri()
with st.sidebar:
    st.header("Experiments")
    if "deleted_task_run" in st.session_state:
        st.success(f"Deleted task run: {st.session_state.pop('deleted_task_run')}")
    try:
        downloadable_database = database_path(uri)
    except (ValueError, TypeError):
        downloadable_database = None
    if downloadable_database is not None and downloadable_database.is_file():
        st.download_button("Download database (.db)", data=lambda: database_snapshot(uri),
                           file_name="mlflow.db", mime="application/octet-stream", on_click="ignore")
        st.caption("All experiments and metrics. Animations and JSON artifacts are stored separately in mlruns.")
    if st.button("Refresh results"):
        st.cache_data.clear()
    try:
        available = experiments(uri)
    except Exception as error:
        st.error(f"Cannot open the results store: {error}")
        st.stop()
    if not available:
        st.info("No experiments yet. Run system.py to record a task.")
        st.stop()
    preferred = next((i for i, (_, name) in enumerate(available) if name == "CRM GridWorld"), 0)
    selected = st.selectbox("Experiment", available, index=preferred, format_func=lambda e: e[1],
                            key="selected_experiment")
    try:
        runs = cached_runs(selected[0], uri).copy()
        runs["started_at"] = pd.to_datetime(runs["started_at"], utc=True).dt.tz_convert(
            "Africa/Johannesburg"
        ).dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception as error:
        st.error(f"Cannot load runs: {error}")
        st.stop()
    if runs.empty:
        st.info("This experiment has no runs yet.")
        st.stop()
    query = st.text_input("Task ID", help="Exact match in the task_id column. Enter 1 to show only task 1.").strip()
    generators = st.multiselect("Generator model", sorted(runs.generator_model.unique()),
                                help="Select multiple models to include runs from any selected model.")
    judges = st.multiselect("Judge model", sorted(runs.judge_model.unique()))
    outcomes = st.multiselect("Outcome", ["Success", "Failure", "Unknown"])
    hide_erroneous_errors = st.toggle(
        "Hide erroneous errors",
        help="Exclude runs stopped by rate limits, provider capacity errors, invalid task or environment specifications, or failure to generate a valid environment.",
    )
    modes = st.multiselect("Run mode", ["Iterative", "One-shot"])
    lengths = st.multiselect("Specification length", ["Long", "Short"])
    bounds = st.slider("Difficulty score", 0.0, 1.0, (0.0, 1.0), step=0.01)

st.caption("Task timestamps are shown in South African time (SAST, UTC+02:00).")

filtered = runs.copy()
if hide_erroneous_errors:
    excluded = filtered.apply(effective_failure_category, axis=1).isin({
        "rate_limit", "provider_capacity", "invalid_task", "invalid_environment",
        "environment_generation_limit", "no_valid_environment", "route_search_limit",
    })
    filtered = filtered[~excluded]
if query:
    filtered = filtered[filtered.task_id.astype("string").eq(query).fillna(False)]
for field, values in [("generator_model", generators), ("judge_model", judges),
                      ("outcome", outcomes), ("iterative_mode", modes),
                      ("specification_length", lengths)]:
    if values:
        filtered = filtered[filtered[field].isin(values)]
if bounds:
    mask = filtered.overall_difficulty_score.between(*bounds)
    mask |= filtered.overall_difficulty_score.isna()
    filtered = filtered[mask]
def show_generations(artifacts):
    generations = artifacts.get("generations.json", [])
    if not generations:
        st.info("No raw generator response was saved for this run. Responses are saved for new runs.")
    for generation in generations:
        with st.expander(f"Generator call {generation['call']} ({generation['kind']})",
                         expanded=generation["call"] == 1):
            st.code(generation["response"], language=None)


with st.expander("Failed initial generations"):
    st.caption("Runs that stopped before an agent evaluation. This list uses all runs in the selected "
               "experiment, including failures hidden by the main results filters.")
    early_failures = runs[runs.success.eq(False).fillna(False)
                          & runs.iterations.fillna(0).eq(0)]
    if early_failures.empty:
        st.info("No early generation failures were recorded in this experiment.")
    else:
        failure_lookup = early_failures.set_index("run_id")
        failure_id = st.selectbox(
            "Choose a failed generation", early_failures.run_id.tolist(),
            format_func=lambda rid: f"Task {failure_lookup.loc[rid, 'task_id']}: "
                                   f"{failure_lookup.loc[rid, 'task_description']} ({rid[:8]})",
            key="selected_failed_generation",
        )
        failed_row = failure_lookup.loc[failure_id]
        st.caption(f"Recorded: {failed_row.started_at} | Generator: {failed_row.generator_model} "
                   f"| Run: {failure_id}")
        st.error(failed_row.failure_reason if pd.notna(failed_row.failure_reason)
                 else "No failure reason was recorded.")
        failure_artifacts = cached_artifacts(failure_id, uri)
        for warning in failure_artifacts["warnings"]:
            st.warning(warning)
        show_generations(failure_artifacts)

if filtered.empty:
    st.info("No runs match these filters. Clear a filter to see results.")
    st.stop()

def metric_value(value, suffix="", scale=1):
    return "Unavailable" if pd.isna(value) else f"{value * scale:,.2f}{suffix}"

overview, detail, plots, counts, comparison = st.tabs(
    ["Overview", "Experiment detail", "Plots", "Experiment counts", "Comparison & export"]
)
with overview:
    columns = st.columns(4)
    columns[0].metric("Recorded runs", len(filtered))
    columns[1].metric("Success rate", metric_value(filtered.success.mean(), "%", 100))
    columns[2].metric("Mean runtime", metric_value(filtered.total_system_time_seconds.mean(), " s"))
    columns[3].metric("Average reward efficiency", metric_value(filtered.reward_efficiency.mean()))
    st.caption(
        f"Success rate uses {filtered.success.count()} runs with a recorded true/false outcome. "
        "Runs without one are excluded, as are missing values in metric means."
    )
    overview_table = filtered[["task_number", "started_at", "task_description", "iterative_mode",
                           "specification_length", "outcome", "iterations",
                           "overall_difficulty_score", "total_system_time_seconds",
                           "failure_reason", "generator_model", "judge_model",
                           "reward_efficiency", "environment_minimum_steps_to_complete_trace",
                           "crm_parse_fail_percentage",
                           "run_id"]].rename(
                               columns={"task_number": "task_id"}
                           )
    st.dataframe(overview_table, hide_index=True, width="stretch")
    chart_data = filtered.dropna(subset=["overall_difficulty_score", "total_system_time_seconds"])
    if not chart_data.empty:
        figure = px.scatter(chart_data, x="overall_difficulty_score", y="total_system_time_seconds",
                                   color="outcome", symbol="generator_model",
                                   hover_data=["task_description", "run_id"],
                                   labels={"overall_difficulty_score": "Difficulty score",
                                           "total_system_time_seconds": "Total runtime (seconds)"})
        figure.update_xaxes(tick0=0, dtick=0.1, range=[0, 1], fixedrange=True,
                            tickformat=".1f", automargin=True)
        st.plotly_chart(figure, width="stretch")
    st.caption("CRM failures include parsing and trace acceptance failures; percentages are averaged per run.")
    st.metric("Mean CRM failure percentage", metric_value(filtered.crm_parse_fail_percentage.mean(), "%"))

with detail:
    lookup = filtered.set_index("run_id")
    run_id = st.selectbox("Choose a run", filtered.run_id.tolist(),
                         format_func=lambda rid: f"{lookup.loc[rid, 'task_description']} · {lookup.loc[rid, 'started_at']} · {rid[:8]}",
                         key="selected_task_run")
    row = lookup.loc[run_id]
    st.subheader(row.task_description)
    st.caption(f"Task {row.task_id} · {row.iterative_mode} · {row.specification_length} specifications · {row.outcome} · Generator: {row.generator_model} · Judge: {row.judge_model} · Run: {run_id}")
    if st.button("Delete this task run", key="delete_task_run",
                 help=f"Delete only run {run_id} for task {row.task_id}."):
        try:
            delete_run(run_id, uri)
        except Exception as error:
            st.error(f"Cannot delete task run: {error}")
        else:
            st.cache_data.clear()
            st.session_state.pop("selected_task_run", None)
            st.session_state["deleted_task_run"] = f"Task {row.task_id} · {run_id}"
            st.rerun()
    if pd.notna(row.failure_reason):
        st.error(f"Run stopped: {row.failure_reason}")
    with st.expander("Runtime breakdown"):
        st.dataframe(runtime_breakdown(row), hide_index=True, width="stretch")
        st.caption("Activity times add up to total system time. Misc time includes setup, evaluation, "
                   "difficulty calculation during execution, and processing overhead. "
                   "Total system time starts when task input is recorded and ends when the task passes "
                   "or its attempts stop, before result preparation and the final MLflow result write.")
    artifacts = cached_artifacts(run_id, uri)
    for warning in artifacts["warnings"]:
        st.warning(warning)
    with st.expander("Generator responses"):
        show_generations(artifacts)
    history = artifacts.get("iterations.json", [])
    st.markdown("#### Iteration history")
    if history:
        selected_iteration = st.selectbox(
            "Choose an iteration", range(len(history)),
            format_func=lambda index: f"Iteration {history[index]['iteration']}",
        )
        attempt = history[selected_iteration]
        measures = attempt["run_information"]
        if "time_seconds" in attempt:
            st.caption(f"Iteration duration: {attempt['time_seconds']:.3f} seconds (includes its CRM generation).")
        st.caption(
            f"Trace correct: {attempt['trace_correct']} · "
            f"Performance passed: {attempt['performance_passed']} · "
            f"Steps: {measures.get('steps', 'Unavailable')} · "
            f"Minimum steps: {measures.get('minimum_steps', 'Unavailable')} · "
            f"Step efficiency: {metric_value(measures.get('step_efficiency'))} · "
            f"Reward: {metric_value(measures.get('episode_reward'))} · "
            f"Efficiency: {metric_value(measures.get('reward_efficiency'))}"
        )
        iteration_left, iteration_right = st.columns(2)
        with iteration_left:
            st.markdown("##### Desired and agent traces")
            st.dataframe(trace_table(attempt["desired_trace"], attempt["agent_trace"]),
                         hide_index=True, width="stretch")
            st.markdown("##### Feedback")
            if attempt.get("judge_model"):
                st.caption(f"OpenRouter ({attempt['judge_model']})")
            st.text(attempt.get("judge_feedback") or "No judge feedback for this iteration.")
        with iteration_right:
            st.markdown("##### Environment grid")
            grid = attempt.get("grid", [])
            if grid:
                st.code("\n".join(" ".join(map(str, grid_row)) for grid_row in grid),
                        language=None)
            else:
                st.caption("No grid was recorded for this iteration.")
            with st.expander("Environment specification"):
                st.json(attempt.get("environment_spec", {}))
            with st.expander("Full evaluation data"):
                st.json(measures)
            with st.expander("CRM used in this iteration"):
                st.json(attempt.get("crm_specification", {}))
    else:
        st.info("This run has no saved iteration history. Earlier runs contain only final artifacts.")
    with st.expander("Final result and GridWorld replay"):
        left, right = st.columns(2)
        with left:
            st.markdown("#### Final event sequence")
            traces = artifacts.get("traces.json", {})
            if isinstance(traces, dict) and "desired_trace" in traces and "final_trace" in traces:
                table = trace_table(traces["desired_trace"], traces["final_trace"])
                st.dataframe(table, hide_index=True, width="stretch")
                st.caption("Events are compared by position; an extra event shifts subsequent comparisons.")
            else:
                st.info("No event traces were recorded for this run.")
            if not history:
                st.markdown("#### Latest judge feedback")
                st.text(artifacts.get("judge_reflection.txt", "No judge feedback was recorded for this run."))
        with right:
            st.markdown("#### GridWorld replay")
            if not artifacts["animations"]:
                st.info("No animation was recorded for this run.")
            for name, data in artifacts["animations"]:
                st.image(data, caption=name)
                st.download_button("Download " + name, data, file_name=name, key=f"animation-{run_id}-{name}")
    with st.expander("Environment and CRM specifications"):
        if "specifications.json" in artifacts:
            st.json(artifacts["specifications.json"])
        else:
            st.info("No specifications were recorded for this run.")
    with st.expander("All recorded parameters and metrics"):
        recorded_fields = [field for field in row.index
                           if field not in {"minimum_steps", "step_efficiency"}]
        step_position = recorded_fields.index("steps_taken") + 1
        recorded_fields[step_position:step_position] = ["minimum_steps", "step_efficiency"]
        st.dataframe(row.reindex(recorded_fields).astype(str).rename("Value"), width="stretch")
    details = {"run": json.loads(filtered[filtered.run_id == run_id].to_json(orient="records", date_format="iso"))[0],
               "artifacts": {k: v for k, v in artifacts.items() if k != "animations"}}
    st.download_button("Download run details (JSON)", json.dumps(details, indent=2),
                       file_name=f"{run_id}.json", mime="application/json")

with plots:
    st.caption("Plots use the sidebar filters for LLM, Short/Long specifications, and Iterative/One-shot mode. "
               "Difficulty is grouped into 0.1-wide intervals; points show interval averages. "
               "The final interval includes 1.0. Missing difficulty or metric values are excluded.")
    choices = {
        "Success rate": ("success", "Success rate (%)"),
        "System time": ("total_system_time_seconds", "Average system time (seconds)"),
        "Iterations": ("iterations", "Average iterations"),
        "CRM parse / trace failure percentage": ("crm_parse_fail_percentage", "Average CRM failure (%)"),
    }
    selected_metric = st.selectbox("Metric over difficulty", list(choices), key="difficulty_metric")
    field, label = choices[selected_metric]
    st.plotly_chart(difficulty_figure(filtered, field, label, split_modes=True), width="stretch",
                    key=f"difficulty_plot_bounded_{field}")
    st.caption("Success rate uses recorded outcomes only. CRM failure percentages include parse and trace checks. "
               "Hover over a point to see its sample count.")
    distributions, training, generation = st.tabs(["Distributions", "Training time", "Generation time"])
    with distributions:
        st.subheader("Difficulty distribution per LLM")
        st.plotly_chart(difficulty_distribution_figure(filtered), width="stretch")
        st.subheader("System time distribution per LLM")
        time_data = filtered.dropna(subset=["total_system_time_seconds"])
        if time_data.empty:
            st.info("No system times recorded for these filters.")
        else:
            facet_rows = (time_data.generator_model.nunique() + 1) // 2
            time_figure = px.histogram(
                time_data, x="total_system_time_seconds", color="generator_model",
                facet_col="generator_model", facet_col_wrap=2, nbins=20,
                facet_row_spacing=0.15 / facet_rows, facet_col_spacing=0.12,
                height=max(420, facet_rows * 350),
                labels={"total_system_time_seconds": "System time (seconds)", "generator_model": "LLM"},
            )
            time_figure.update_layout(showlegend=False, margin={"l": 80, "r": 30, "t": 60, "b": 65})
            time_figure.update_yaxes(showticklabels=True, automargin=True,
                                    title_standoff=15, tickfont={"size": 12})
            time_figure.update_yaxes(title_text="Experiment count", col=1)
            time_figure.update_yaxes(title_text=None, col=2)
            st.plotly_chart(time_figure, width="stretch")
    with training:
        st.subheader("Average training time over difficulty per LLM")
        st.plotly_chart(difficulty_figure(
            filtered, "average_training_time_seconds", "Average training time (seconds)",
        ), width="stretch")
        st.caption("Average of each run's recorded training time per iteration.")
    with generation:
        st.subheader("Average generation time over difficulty per LLM")
        st.plotly_chart(difficulty_figure(
            filtered, "average_generation_time_seconds", "Average generation time (seconds)",
        ), width="stretch")
        st.caption("Average of each run's recorded generation time per generator call.")

with counts:
    st.caption("Experiment counts for the runs matching the sidebar filters, including runs with missing metrics.")
    by_llm, by_length, by_mode = st.tabs(["Per LLM", "Short / Long", "Iterative / One-shot"])
    for tab, field, label in [
        (by_llm, "generator_model", "LLM"),
        (by_length, "specification_length", "Specifications"),
        (by_mode, "iterative_mode", "Mode"),
    ]:
        with tab:
            table = filtered.groupby(field, dropna=False).size().reset_index(name="Experiment count")
            table = table.rename(columns={field: label})
            st.dataframe(table, hide_index=True, width="stretch")
            st.plotly_chart(px.bar(table, x=label, y="Experiment count", color=label), width="stretch")

with comparison:
    st.caption("Compare identical task descriptions and difficulty scores where possible. Different task mixes can distort model averages; identical scores do not guarantee identical environments.")
    group_by_task = st.checkbox("Group by task and difficulty", value=True)
    groups = ["iterative_mode", "specification_length", "generator_model"]
    if group_by_task:
        groups += ["task_description", "overall_difficulty_score"]
    summary = summarize(filtered, groups)
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption("Success rate is a fraction (0–1). Runs counts every record; runs_with_recorded_outcome is the denominator after excluding legacy records with no true/false outcome.")
    metric = st.selectbox("Compare individual runs", ["step_efficiency", "reward_efficiency", "total_system_time_seconds",
                                                     "iterations",
                                                     "crm_parse_fail_percentage",
                                                     "overall_difficulty_score"])
    st.plotly_chart(comparison_figure(filtered, metric), width="stretch")
    if filtered[metric].isna().any():
        st.caption("Models without a recorded value for this metric have no plotted point.")
    st.download_button("Download filtered runs (CSV)", filtered.to_csv(index=False),
                       file_name="experiment_runs.csv", mime="text/csv")
    st.download_button("Download comparison (CSV)", summary.to_csv(index=False),
                       file_name="experiment_comparison.csv", mime="text/csv")
