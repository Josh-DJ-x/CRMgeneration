"""Offline checks for result semantics and dashboard interactions."""

import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_results as results
from export_report import export_report
from mlflow_tracking import configure_tracking, log_generation_failure, next_task_id


class Page(list):
    token = None


def run(identifier, params, metrics=None):
    return SimpleNamespace(data=SimpleNamespace(params=params, metrics=metrics or {}),
                           info=SimpleNamespace(run_id=identifier, status="FINISHED", start_time=1000))


class ResultsTests(unittest.TestCase):
    def test_tracking_restores_deleted_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            uri = Path(directory).as_uri()
            with patch.dict("os.environ", {
                "MLFLOW_TRACKING_URI": uri,
                "MLFLOW_EXPERIMENT_NAME": "Startup recovery",
                "MLFLOW_ALLOW_FILE_STORE": "true",
            }):
                api = results.client(uri)
                experiment_id = api.create_experiment("Startup recovery")
                run_id = api.create_run(experiment_id).info.run_id
                api.delete_experiment(experiment_id)
                mlflow = configure_tracking()
                self.assertEqual(api.get_experiment(experiment_id).lifecycle_stage, "active")
                self.assertEqual(api.get_run(run_id).info.lifecycle_stage, "active")
                api.delete_run(run_id)
                configure_tracking()
                self.assertEqual(api.get_run(run_id).info.lifecycle_stage, "deleted")
                with mlflow.start_run() as active:
                    self.assertEqual(active.info.experiment_id, experiment_id)

    def test_database_download_is_consistent_with_wal(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "tracking.db"
            writer = sqlite3.connect(source)
            writer.execute("PRAGMA journal_mode=WAL")
            writer.execute("create table example(value integer)")
            writer.execute("insert into example values (42)")
            writer.commit()
            try:
                snapshot = results.database_snapshot("sqlite:///" + source.as_posix())
                exported = Path(directory) / "download.db"
                exported.write_bytes(snapshot)
                reader = sqlite3.connect(exported)
                try:
                    self.assertEqual(reader.execute("select value from example").fetchone()[0], 42)
                    self.assertEqual(reader.execute("PRAGMA integrity_check").fetchone()[0], "ok")
                finally:
                    reader.close()
            finally:
                writer.close()
        with self.assertRaises(ValueError):
            results.database_snapshot("postgresql://localhost/results")

    def test_legacy_runtime_breakdown_uses_totals_not_averages(self):
        row = {"iterations": 2, "generator_call_count": 3, "judge_call_count": 1,
               "average_generation_time_seconds": 3.0, "average_judge_time_seconds": 4.0,
               "average_training_time_seconds": 10.0, "average_grid_world_generation_time_seconds": 0.5,
               "total_system_time_seconds": 40.0}
        table = results.runtime_breakdown(row)
        self.assertEqual(table.Seconds.tolist(), [9, 4, 20, 1, 6, 40])
        self.assertEqual(table.Activity.tolist()[-2:], ["Misc time", "Total system time"])

    def test_runtime_breakdown_prefers_totals_and_combines_other_work(self):
        row = {"total_generation_time_seconds": 15, "average_generation_time_seconds": 99,
               "generator_call_count": 3, "total_judge_time_seconds": 11,
               "total_training_time_seconds": 20, "total_grid_world_generation_time_seconds": 2,
               "setup_time_seconds": 2, "total_evaluation_time_seconds": 4,
               "postprocessing_time_seconds": 5, "orchestration_time_seconds": 0,
               "total_system_time_seconds": 59}
        table = results.runtime_breakdown(row)
        self.assertEqual(table.Seconds.tolist(), [15, 11, 20, 2, 11, 59])
        self.assertEqual(table.Seconds.iloc[:-1].sum(), table.Seconds.iloc[-1])
        row["misc_time_seconds"] = 11
        self.assertEqual(results.runtime_breakdown(row).Seconds.tolist(), table.Seconds.tolist())

    def test_runtime_breakdown_keeps_unknown_misc_time_missing(self):
        table = results.runtime_breakdown({"total_system_time_seconds": 40})
        self.assertTrue(results.pd.isna(table.Seconds.iloc[-2]))
        self.assertEqual(table.Seconds.iloc[-1], 40)

    def test_comparison_axes_keep_models_with_missing_metrics(self):
        frame = results.pd.DataFrame({
            "generator_model": ["a", "b", "c", "d", "e"],
            "judge_model": ["judge"] * 5,
            "task_description": ["task"] * 5,
            "run_id": list("abcde"), "outcome": ["Success"] * 5,
            "iterations": [1, 2, 3, 1, 2],
            "reward_efficiency": [1, None, None, 0.5, None],
            "total_system_time_seconds": [None] * 5,
        })
        for metric in ["iterations", "reward_efficiency", "total_system_time_seconds"]:
            figure = results.comparison_figure(frame, metric)
            self.assertEqual(list(figure.layout.xaxis.tickvals), list("abcde"))
            self.assertEqual(list(figure.layout.xaxis.categoryarray), list("abcde"))

    def test_difficulty_bins_keep_boundaries_unknown_outcomes_and_missing_values(self):
        frame = results.pd.DataFrame({
            "generator_model": ["model-a"] * 8,
            "overall_difficulty_score": [0, 0.09, 0.1, 0.3, 0.9, 1.0, None, 1.1],
            "success": results.pd.Series([True, False, None, True, False, True, True, True], dtype="boolean"),
            "average_training_time_seconds": [10, None, 20, 30, 40, 60, 999, 999],
        })
        counts = results.difficulty_summary(frame).set_index("interval")
        self.assertEqual(counts.value.tolist(), [2, 1, 0, 1, 0, 0, 0, 0, 0, 2])
        success = results.difficulty_summary(frame, "success").set_index("interval")
        self.assertEqual(success.loc["0.0–0.1", "value"], 50)
        self.assertEqual(success.loc["0.1–0.2", "samples"], 0)
        self.assertTrue(results.pd.isna(success.loc["0.1–0.2", "value"]))
        self.assertEqual(success.loc["0.9–1.0", "value"], 50)
        training = results.difficulty_summary(frame, "average_training_time_seconds").set_index("interval")
        self.assertEqual(training.loc["0.0–0.1", "value"], 10)
        self.assertEqual(training.loc["0.9–1.0", "value"], 50)
        figure = results.difficulty_figure(frame, "success", "Success rate (%)")
        self.assertEqual(figure.layout.xaxis.dtick, 0.1)
        self.assertEqual(tuple(figure.layout.xaxis.range), (0, 1))
        self.assertEqual(figure.layout.xaxis.minallowed, 0)
        self.assertEqual(figure.layout.xaxis.maxallowed, 1)
        self.assertFalse(figure.layout.xaxis.autorange)
        self.assertTrue(figure.layout.xaxis.fixedrange)
        frame["specification_length"] = "Long"
        frame["iterative_mode"] = ["Iterative", "One-shot"] * 4
        timing = results.difficulty_figure(
            frame, "average_training_time_seconds", "Time (seconds)", split_modes=True,
        )
        for axis in timing.select_xaxes():
            self.assertEqual(tuple(axis.range), (0, 1))
            self.assertEqual((axis.minallowed, axis.maxallowed), (0, 1))
            self.assertTrue(axis.fixedrange)
        for axis in timing.select_yaxes():
            self.assertTrue(axis.autorange)

    def test_comparison_graphs_label_generators_instead_of_openrouter_models(self):
        frame = self.frame()
        frame["generator_model"] = ["model-a", "model-b", "model-a"]
        frame["judge_model"] = ["OpenRouter (judge-a)"] * 3
        frame["iterations"] = [1, 2, 3]
        figure = results.comparison_figure(frame, "iterations")
        self.assertEqual({trace.name for trace in figure.data}, {"model-a", "model-b"})
        self.assertNotIn("OpenRouter", figure.to_json())

    def test_dashboard_new_plots_counts_and_metric_menu_follow_filters(self):
        from streamlit.testing.v1 import AppTest
        frame = self.frame()
        frame["generator_model"] = ["model-a", "model-b", "model-a"]
        frame["overall_difficulty_score"] = [0.1, 0.6, 1.0]
        frame["specification_length"] = ["Short", "Long", "Short"]
        frame["iterative_mode"] = ["Iterative", "One-shot", "Iterative"]
        for metric in ("total_system_time_seconds", "average_generation_time_seconds",
                       "average_training_time_seconds", "crm_parse_fail_percentage", "iterations"):
            frame[metric] = [1, 2, 3]
        with (patch.object(results, "experiments", return_value=[("1", "test")]),
              patch.object(results, "load_runs", new=lambda *args: frame),
              patch.object(results, "load_artifacts", new=lambda *args: {"warnings": [], "animations": []})):
            app = AppTest.from_file(str(results.ROOT / "dashboard.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertIn("Plots", [tab.label for tab in app.tabs])
            self.assertIn("Experiment counts", [tab.label for tab in app.tabs])
            self.assertTrue(any(metric.label == "Average reward efficiency" for metric in app.metric))
            for label in app.selectbox(key="difficulty_metric").options:
                app.selectbox(key="difficulty_metric").set_value(label).run()
                self.assertEqual(len(app.exception), 0)
            app.sidebar.multiselect[0].set_value(["model-a"]).run()
            app.sidebar.multiselect[4].set_value(["Short"]).run()
            app.sidebar.multiselect[3].set_value(["Iterative"]).run()
            self.assertEqual(len(app.exception), 0)
            llm_counts = next(item.value for item in app.dataframe if "LLM" in item.value.columns)
            self.assertEqual(llm_counts.to_dict("records"), [{"LLM": "model-a", "Experiment count": 2}])

    def test_judge_labels_distinguish_routing_from_actual_models(self):
        self.assertEqual(results.judge_label("openrouter/free", '["a", "b", "a"]', 3),
                         "OpenRouter (a, b)")
        self.assertEqual(results.judge_label("openrouter/free", None, 2),
                         "OpenRouter (actual model not recorded)")
        self.assertEqual(results.judge_label("openrouter/free", "[]", 0),
                         "OpenRouter (not called)")

    def test_early_generation_failure_appears_in_results(self):
        with tempfile.TemporaryDirectory() as directory:
            uri = Path(directory).as_uri()
            with patch.dict("os.environ", {
                "MLFLOW_TRACKING_URI": uri,
                "MLFLOW_EXPERIMENT_NAME": "Early failure test",
                "MLFLOW_ALLOW_FILE_STORE": "true",
            }):
                run_id = log_generation_failure({
                    "task_id": "1", "task_description": "Collect coins",
                    "generator_model": "llama3.2", "judge_model": "openrouter/free",
                    "iterative": True, "long": True,
                    "failure_reason": "Invalid compact trace",
                    "generator_call_count": 1,
                    "total_system_time": 4.0,
                    "average_generation_time": 1.0,
                    "failure_stage": "initial generation validation",
                    "generation_history": [{"call": 1, "kind": "initial", "response": "Invalid raw response"}],
                })
                experiment_id = results.experiments(uri)[0][0]
                frame = results.load_runs(experiment_id, uri)
                artifacts = results.load_artifacts(run_id, uri)
        self.assertEqual(frame.loc[0, "run_id"], run_id)
        self.assertEqual(frame.loc[0, "outcome"], "Failure")
        self.assertEqual(frame.loc[0, "failure_stage"], "initial generation validation")
        self.assertEqual(artifacts["generations.json"][0]["response"], "Invalid raw response")
        self.assertNotIn("trace_verification_failures", frame)
        self.assertTrue(frame.loc[0, "overall_difficulty_score"] != frame.loc[0, "overall_difficulty_score"])

    def test_next_task_id_ignores_legacy_labels(self):
        api = SimpleNamespace()
        api.get_experiment_by_name = lambda name: SimpleNamespace(experiment_id="2")
        api.search_runs = lambda **kwargs: [
            SimpleNamespace(data=SimpleNamespace(params={"task_id": "4"})),
            SimpleNamespace(data=SimpleNamespace(params={"task_id": "old-uuid"})),
        ]

        self.assertEqual(next_task_id(api), "5")

    def frame(self):
        with patch.object(results, "client") as factory:
            first = Page([run("a", {"success_status": "False", "task_description": "task"})])
            first.token = "next"
            factory.return_value.search_runs.side_effect = [first, Page([
                run("b", {"success_status": "True", "task_description": "task"}),
                run("c", {"task_description": "task"}),
            ])]
            frame = results.load_runs("1")
            self.assertEqual(factory.return_value.search_runs.call_args.kwargs["page_token"], "next")
            return frame

    def test_unknown_outcomes_and_missing_metrics(self):
        frame = self.frame()
        self.assertEqual(frame.outcome.tolist(), ["Failure", "Success", "Unknown"])
        summary = results.summarize(frame, ["generator_model"]).iloc[0]
        self.assertEqual(summary.runs, 3)
        self.assertEqual(summary.runs_with_recorded_outcome, 2)
        self.assertEqual(summary.success_rate, 0.5)
        self.assertTrue(frame.reward_efficiency.isna().all())

    def test_step_metrics_load_from_new_and_historical_runs(self):
        with patch.object(results, "client") as factory:
            factory.return_value.search_runs.return_value = Page([
                run("new", {"success_status": "True"}, {"minimum_steps": 95, "steps_taken": 99}),
                run("old", {"success_status": "True"}, {"environment_minimum_steps_to_complete_trace": 95, "steps_taken": 100}),
                run("unknown", {}, {"environment_minimum_steps_to_complete_trace": 0, "steps_taken": 100}),
                run("empty", {}, {"minimum_steps": 95, "steps_taken": 0}),
            ])
            frame = results.load_runs("1").set_index("run_id")
        self.assertEqual(frame.loc["new", "minimum_steps"], 95)
        self.assertAlmostEqual(frame.loc["new", "step_efficiency"], 95 / 99)
        self.assertEqual(frame.loc["old", "minimum_steps"], 95)
        self.assertEqual(frame.loc["old", "step_efficiency"], 0.95)
        self.assertTrue(results.pd.isna(frame.loc["unknown", "minimum_steps"]))
        self.assertTrue(results.pd.isna(frame.loc["unknown", "step_efficiency"]))
        self.assertTrue(results.pd.isna(frame.loc["empty", "step_efficiency"]))
        self.assertTrue(frame.loc["old", "success"])

    def test_dashboard_places_minimum_steps_under_steps_taken(self):
        from streamlit.testing.v1 import AppTest
        frame = self.frame()
        frame["steps_taken"] = 100
        frame["minimum_steps"] = 95
        frame["step_efficiency"] = 0.95
        with (patch.object(results, "experiments", return_value=[("1", "test")]),
              patch.object(results, "load_runs", new=lambda *args: frame),
              patch.object(results, "load_artifacts", new=lambda *args: {"warnings": [], "animations": []})):
            app = AppTest.from_file(str(results.ROOT / "dashboard.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        table = next(item.value for item in app.dataframe if "Value" in item.value.columns)
        index = table.index.tolist()
        position = index.index("steps_taken")
        self.assertEqual(index[position:position + 3], ["steps_taken", "minimum_steps", "step_efficiency"])
        self.assertEqual(table.loc["minimum_steps", "Value"], "95")

    def test_trace_length_mismatch(self):
        table = results.trace_table(["COIN", "GOAL"], ["COIN"])
        self.assertEqual(table.Match.tolist(), [True, False])
        self.assertEqual(table.Actual.tolist(), ["COIN", "—"])

    def test_unavailable_artifact_store(self):
        with patch.object(results, "client") as factory:
            factory.return_value.list_artifacts.side_effect = OSError("missing")
            self.assertTrue(results.load_artifacts("a")["warnings"])

    def test_snapshot_does_not_overwrite_and_escapes_html(self):
        frame = self.frame()
        frame.loc[0, "task_description"] = "<script>alert(1)</script>"
        with tempfile.TemporaryDirectory() as directory, patch("export_report.experiments", return_value=[("1", "test")]), patch("export_report.load_runs", return_value=frame):
            target = Path(directory) / "snapshot"
            export_report("test", target)
            self.assertTrue((target / "report.qmd").exists())
            self.assertNotIn("<script>", (target / "preview.html").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                export_report("test", target)

    def test_dashboard_filters_and_empty_state(self):
        from streamlit.testing.v1 import AppTest
        frame = self.frame()
        frame["task_id"] = ["1", "10", "2"]
        frame["task_number"] = [1, 10, 2]
        frame["task_description"] = ["task", "contains 1", "also contains 1"]
        frame["iterative_mode"] = ["Iterative", "Iterative", "One-shot"]
        frame["specification_length"] = ["Long", "Short", "Short"]
        def fake_runs(*args):
            return frame

        def fake_artifacts(*args):
            return {"warnings": [], "animations": []}

        with patch.object(results, "experiments", return_value=[("1", "test")]), patch.object(results, "load_runs", new=fake_runs), patch.object(results, "load_artifacts", new=fake_artifacts):
            app = AppTest.from_file(str(results.ROOT / "dashboard.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.sidebar.checkbox), 0)
            self.assertEqual(app.sidebar.multiselect[3].options, ["Iterative", "One-shot"])
            self.assertEqual(app.sidebar.multiselect[4].options, ["Long", "Short"])
            app.sidebar.text_input[0].set_value(" 1 ").run()
            self.assertEqual(app.dataframe[0].value.task_id.tolist(), [1])
            app.sidebar.text_input[0].set_value("").run()
            app.sidebar.multiselect[3].set_value(["Iterative"]).run()
            self.assertEqual(app.metric[0].value, "2")
            app.sidebar.multiselect[3].set_value(["One-shot"]).run()
            self.assertEqual(app.dataframe[0].value.task_id.tolist(), [2])
            app.sidebar.multiselect[3].set_value([]).run()
            app.sidebar.multiselect[4].set_value(["Short"]).run()
            self.assertEqual(app.metric[0].value, "2")
            app.sidebar.multiselect[4].set_value([]).run()
            app.sidebar.multiselect[2].set_value(["Failure"]).run()
            self.assertEqual(app.metric[0].value, "1")
            app.sidebar.text_input[0].set_value("no matching task").run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("No runs match" in item.value for item in app.info))

    def test_dashboard_local_time_multiple_generators_and_fixed_difficulty_range(self):
        from streamlit.testing.v1 import AppTest
        frame = self.frame()
        frame["generator_model"] = ["model-a", "model-b", "model-c"]
        frame["started_at"] = results.pd.to_datetime(["2026-10-03T23:30:00Z"] * 3)
        frame["overall_difficulty_score"] = [0.2, 0.6, float("nan")]
        with (patch.object(results, "experiments", return_value=[("1", "test")]),
              patch.object(results, "load_runs", new=lambda *args: frame),
              patch.object(results, "load_artifacts", new=lambda *args: {"warnings": [], "animations": []})):
            app = AppTest.from_file(str(results.ROOT / "dashboard.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.dataframe[0].value.started_at.tolist(), ["2026-10-04 01:30:00"] * 3)
            slider = app.sidebar.slider[0]
            self.assertEqual((slider.min, slider.max, slider.value), (0.0, 1.0, (0.0, 1.0)))
            app.sidebar.multiselect[0].set_value(["model-a", "model-b"]).run()
            self.assertEqual(app.dataframe[0].value.generator_model.tolist(), ["model-a", "model-b"])
            app.sidebar.slider[0].set_value((0.5, 1.0)).run()
            self.assertEqual(app.dataframe[0].value.generator_model.tolist(), ["model-b"])
            app.sidebar.multiselect[0].set_value([]).run()
            self.assertEqual(app.dataframe[0].value.generator_model.tolist(), ["model-b", "model-c"])

    def test_dashboard_hides_provider_errors_invalid_tasks_and_environment_failures(self):
        from streamlit.testing.v1 import AppTest
        frame = results.pd.concat([self.frame(), self.frame()], ignore_index=True)
        frame["run_id"] = list("abcdef")
        frame["failure_category"] = ["rate_limit", "invalid_task", "invalid_task",
                                     "other", "provider_capacity", None]
        frame["failure_reason"] = [
            "Generator LLM is rate limited. Please try again later.",
            "Please specify a valid task",
            "INVALID TASK SPECIFICATION",
            "Other failure mentioning rate limited and invalid task specification",
            "Arbitrary provider message; category determines filtering",
            None,
        ]
        frame["success"] = results.pd.Series([False] * 5 + [True], dtype="boolean")
        frame["outcome"] = ["Failure"] * 5 + ["Success"]
        environment_failures = frame.iloc[:3].copy()
        environment_failures["run_id"] = list("ghi")
        environment_failures["failure_category"] = ["invalid_environment", "environment_generation_limit", "route_search_limit"]
        environment_failures["failure_reason"] = [
            "Invalid environment specification.",
            "Environment generation could not find a valid layout after 100 attempts.",
            "Shortest-route search reached its search limit; a valid environment may still exist.",
        ]
        frame = results.pd.concat([frame, environment_failures], ignore_index=True)

        def fake_runs(*args):
            return frame

        def fake_artifacts(*args):
            return {"warnings": [], "animations": []}

        with (patch.object(results, "experiments", return_value=[("1", "test")]),
              patch.object(results, "load_runs", new=fake_runs),
              patch.object(results, "load_artifacts", new=fake_artifacts)):
            app = AppTest.from_file(str(results.ROOT / "dashboard.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertFalse(app.sidebar.toggle[0].value)
            self.assertEqual(app.metric[0].value, "9")
            app.sidebar.toggle[0].set_value(True).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.dataframe[0].value.run_id.tolist(), list("df"))
            self.assertEqual(app.metric[0].value, "2")
            self.assertEqual(app.metric[1].value, "50.00%")
            app.sidebar.multiselect[2].set_value(["Failure"]).run()
            self.assertEqual(app.dataframe[0].value.run_id.tolist(), list("d"))
            app.sidebar.toggle[0].set_value(False).run()
            self.assertEqual(app.dataframe[0].value.run_id.tolist(), list("abcdeghi"))

    def test_dashboard_hides_historical_environment_failures_for_tasks_52_and_57(self):
        from streamlit.testing.v1 import AppTest
        frame = self.frame()
        frame["task_id"] = ["52", "57", "58"]
        frame["task_number"] = [52, 57, 58]
        frame["failure_category"] = ["other", "other", None]
        frame["failure_reason"] = [
            "environment_generation: ValueError: Invalid environment specification.",
            "evaluation reward bounds: ValueError: No ordered route exists for the desired trace.",
            "Other failure mentioning invalid environment specification.",
        ]
        with (patch.object(results, "experiments", return_value=[("1", "test")]),
              patch.object(results, "load_runs", new=lambda *args: frame),
              patch.object(results, "load_artifacts", new=lambda *args: {"warnings": [], "animations": []})):
            app = AppTest.from_file(str(results.ROOT / "dashboard.py"), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.metric[0].value, "3")
            app.sidebar.toggle[0].set_value(True).run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.dataframe[0].value.task_id.tolist(), [58])
            self.assertEqual(app.metric[0].value, "1")
            app.sidebar.toggle[0].set_value(False).run()
            self.assertEqual(app.metric[0].value, "3")

    def test_failed_generation_remains_visible_when_main_filters_exclude_it(self):
        from streamlit.testing.v1 import AppTest
        frame = self.frame()
        frame.loc[0, "failure_reason"] = "Initial generation was invalid"
        artifacts = {"warnings": [], "animations": [], "generations.json": [
            {"call": 1, "kind": "initial", "response": "Broken initial CRM"},
        ]}
        with (patch.object(results, "experiments", return_value=[("1", "test")]),
              patch.object(results, "load_runs", new=lambda *args: frame),
              patch.object(results, "load_artifacts", new=lambda *args: artifacts)):
            app = AppTest.from_file(str(results.ROOT / "dashboard.py"), default_timeout=30).run()
            app.sidebar.multiselect[2].set_value(["Success"]).run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any(item.key == "selected_failed_generation" for item in app.selectbox))
            self.assertTrue(any(item.value == "Broken initial CRM" for item in app.code))
            self.assertTrue(any("Initial generation was invalid" in item.value for item in app.error))

    def test_dashboard_deletes_only_selected_task_run(self):
        from streamlit.testing.v1 import AppTest
        frame = self.frame()
        frame["task_id"] = ["1", "1", "2"]

        def fake_runs(*args):
            return frame.copy()

        def fake_artifacts(*args):
            return {"warnings": [], "animations": []}

        def fake_delete(run_id, uri):
            frame.drop(frame.index[frame.run_id == run_id], inplace=True)

        with (patch.object(results, "experiments", return_value=[("1", "test")]),
              patch.object(results, "load_runs", new=fake_runs),
              patch.object(results, "load_artifacts", new=fake_artifacts),
              patch.object(results, "delete_run", side_effect=fake_delete) as delete):
            app = AppTest.from_file(str(results.ROOT / "dashboard.py"), default_timeout=30).run()
            self.assertFalse(any(button.label == "Delete experiment" for button in app.button))
            app.selectbox(key="selected_task_run").set_value("b").run()
            app.button(key="delete_task_run").click().run()
            delete.assert_called_once_with("b", results.tracking_uri())
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.dataframe[0].value.run_id.tolist(), ["a", "c"])
            self.assertEqual(app.metric[0].value, "2")
            self.assertTrue(any("Deleted task run" in item.value for item in app.success))
            app.button(key="delete_task_run").click().run()
            app.button(key="delete_task_run").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("This experiment has no runs yet" in item.value for item in app.info))
            self.assertEqual(len(app.metric), 0)
            self.assertEqual(app.sidebar.selectbox[0].options, ["test"])

    def test_dashboard_reports_task_run_deletion_failure(self):
        from streamlit.testing.v1 import AppTest
        frame = self.frame()

        def fake_runs(*args):
            return frame

        def fake_artifacts(*args):
            return {"warnings": [], "animations": []}

        with (patch.object(results, "experiments", return_value=[("1", "test")]),
              patch.object(results, "load_runs", new=fake_runs),
              patch.object(results, "load_artifacts", new=fake_artifacts),
              patch.object(results, "delete_run", side_effect=RuntimeError("store unavailable"))):
            app = AppTest.from_file(str(results.ROOT / "dashboard.py"), default_timeout=30).run()
            app.button(key="delete_task_run").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Cannot delete task run: store unavailable" in item.value for item in app.error))
            self.assertEqual(app.metric[0].value, "3")

    def test_task_run_deletion_is_individual_and_recoverable(self):
        with tempfile.TemporaryDirectory() as directory:
            uri = Path(directory).as_uri()
            with patch.dict("os.environ", {"MLFLOW_ALLOW_FILE_STORE": "true"}):
                api = results.client(uri)
                experiment_id = api.create_experiment("Deletion test")
                run_id = api.create_run(experiment_id).info.run_id
                other_run_id = api.create_run(experiment_id).info.run_id
                results.delete_run(run_id, uri)
                self.assertEqual(results.experiments(uri), [(experiment_id, "Deletion test")])
                self.assertEqual(api.get_run(run_id).info.lifecycle_stage, "deleted")
                self.assertEqual(api.get_run(other_run_id).info.lifecycle_stage, "active")
                self.assertEqual(results.load_runs(experiment_id, uri).run_id.tolist(), [other_run_id])
                api.restore_run(run_id)
                self.assertEqual(api.get_run(run_id).info.lifecycle_stage, "active")

    def test_dashboard_displays_compact_iteration_history(self):
        from streamlit.testing.v1 import AppTest
        frame = self.frame()
        history = [{
            "iteration": 1,
            "environment_spec": {"coins": 1},
            "grid": [["A", "C"], [".", "G"]],
            "crm_specification": {},
            "desired_trace": ["COIN", "GOAL"],
            "agent_trace": ["COIN"],
            "run_information": {"steps": 10, "episode_reward": 1.0,
                                "reward_efficiency": 0.5},
            "trace_correct": False,
            "performance_passed": False,
            "judge_feedback": "Find the goal.",
        }]
        def fake_runs(*args):
            return frame

        def fake_artifacts(*args):
            return {"warnings": [], "animations": [], "iterations.json": history}

        with (patch.object(results, "experiments", return_value=[("1", "test")]),
              patch.object(results, "load_runs", new=fake_runs),
              patch.object(results, "load_artifacts", new=fake_artifacts)):
            app = AppTest.from_file(str(results.ROOT / "dashboard.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any("Find the goal." in item.value for item in app.text))


if __name__ == "__main__":
    unittest.main()
