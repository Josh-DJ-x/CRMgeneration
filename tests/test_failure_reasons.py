"""Failure explanations, abnormal exits, and historical run repairs."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import experiment_results as results
import system_loop
from failure_diagnostics import failure_reason
from mlflow_tracking import log_experiment_run, log_generation_failure
from semantic_specifications import GENERATION_FORMAT_SHORT
from system_functions import crm_accepts_trace, crm_parse, extract_generation_trace


class FailureReasonTests(unittest.TestCase):
    def test_crm_failure_percentage_counts_parse_and_alignment_failures_once(self):
        initial = ('<<<DESIRED_TRACE>>>\nGOAL\n<<<ENVIRONMENT>>>\n'
                   '{"coins":0,"bells":0,"keys":{"red":0,"blue":0,"green":0,"yellow":0}}'
                   '\n<<<CRM>>>\ninvalid CRM')
        for parses, accepts, expected in [
            ([None, object(), object(), object()], [False, False, True], 75.0),
            ([object()] * 4, [False] * 4, 100.0),
        ]:
            with self.subTest(expected=expected), \
                 patch("builtins.input", return_value="Reach goal"), \
                 patch.object(system_loop, "configure_tracking", return_value=None), \
                 patch.object(system_loop, "crm_parse", side_effect=parses), \
                 patch.object(system_loop, "crm_accepts_trace", side_effect=accepts) as accept, \
                 patch.object(system_loop, "crm_trace_alignment_error", return_value="trace rejected"), \
                 patch.object(system_loop, "generate_gridworld", side_effect=RuntimeError("stop after validation")), \
                 patch.object(system_loop, "log_generation_failure") as log_failure:
                responses = iter([initial] + ["<<<CRM>>>\ncorrected"] * 3)
                with self.assertRaises((RuntimeError, SystemExit)):
                    system_loop.main(gemini=SimpleNamespace(generate=lambda prompt: next(responses)),
                                     openrouter=SimpleNamespace())
                record = log_failure.call_args.args[0]
                self.assertEqual(record["generator_call_count"], 4)
                self.assertEqual(record["crm_parse_fail_percentage"], expected)
                self.assertEqual(accept.call_count, len(accepts))

    def test_crm_failure_percentage_handles_missing_section_and_provider_failure(self):
        initial = ('<<<DESIRED_TRACE>>>\nGOAL\n<<<ENVIRONMENT>>>\n'
                   '{"coins":0,"bells":0,"keys":{"red":0,"blue":0,"green":0,"yellow":0}}'
                   '\n<<<CRM>>>\ninvalid CRM')
        for correction in ["Missing CRM section", RuntimeError("network unavailable")]:
            with self.subTest(correction=correction), \
                 patch("builtins.input", return_value="Reach goal"), \
                 patch.object(system_loop, "configure_tracking", return_value=None), \
                 patch.object(system_loop, "crm_parse", return_value=None), \
                 patch.object(system_loop, "log_generation_failure") as log_failure:
                responses = ([initial] + [correction] * 3 if isinstance(correction, str)
                             else [initial, correction])
                generator = SimpleNamespace(generate=unittest.mock.Mock(side_effect=responses))
                with self.assertRaises((SystemExit, RuntimeError)):
                    system_loop.main(gemini=generator, openrouter=SimpleNamespace(), iterative=False)
                record = log_failure.call_args.args[0]
                self.assertEqual(record["crm_parse_fail_percentage"], 100.0)
                self.assertEqual(record["generator_call_count"], len(responses))

    def test_invalid_initial_generation_saves_raw_response(self):
        response = "Malformed generation without required sections"
        with (patch("builtins.input", return_value="Reach goal"),
              patch.object(system_loop, "configure_tracking", return_value=None),
              patch.object(system_loop, "log_generation_failure") as log_failure):
            with self.assertRaises(SystemExit):
                system_loop.main(gemini=SimpleNamespace(generate=lambda prompt: response),
                                 openrouter=SimpleNamespace())
        log_failure.assert_called_once()
        record = log_failure.call_args.args[0]
        self.assertEqual(record["failure_stage"], "initial generation validation")
        self.assertEqual(record["crm_parse_fail_percentage"], 100.0)
        self.assertEqual(record["generation_history"], [
            {"call": 1, "kind": "initial", "response": response},
        ])

    def test_task5_first_bell_must_also_decrement_counter(self):
        text = (Path(__file__).parent / "fixtures/task5_failed_crm.txt").read_text()
        trace = ("COIN",) * 5 + ("RED_KEY", "BLUE_KEY") + ("COIN",) * 3 + ("BELL",) * 11 + ("YELLOW_KEY", "GOAL")
        self.assertFalse(crm_accepts_trace(crm_parse(text), trace))
        corrected = text.replace("BELL / (-,Z,-) -> (0,0,0)", "BELL / (-,Z,-) -> (0,0,-1)")
        self.assertTrue(crm_accepts_trace(crm_parse(corrected), trace))

    def test_evaluation_reasons_cover_trace_performance_and_unknown_failure(self):
        record = {"success_status": False, "desired_trace": ["GOAL"], "final_trace": [],
                  "reward_efficiency": 0, "iterations": 3, "iterative": True,
                  "run_information": {"truncated": True, "steps": 200}}
        reason = failure_reason(record)
        self.assertIn("200-step limit", reason)
        self.assertIn("No events", reason)
        self.assertIn("0.940", reason)
        self.assertIn("3 evaluation", reason)
        record.update(final_trace=["GOAL"], reward_efficiency=0.93, run_information={})
        reason = failure_reason(record)
        self.assertIn("0.930", reason)
        self.assertNotIn("trace mismatch", reason)
        self.assertTrue(failure_reason({"success_status": False}))
        self.assertIsNone(failure_reason({"success_status": True, "failure_reason": "stale reason"}))

    def test_provider_errors_and_call_limits_are_logged_once(self):
        for generator, expected, limit in [
            (SimpleNamespace(generate=lambda prompt: (_ for _ in ()).throw(RuntimeError("network unavailable"))),
             "generator: RuntimeError: network unavailable", 6),
            (SimpleNamespace(generate=lambda prompt: "unused"), "Generator call limit reached", 0),
        ]:
            with self.subTest(expected=expected), patch("builtins.input", return_value="Reach goal"), \
                 patch.object(system_loop, "configure_tracking", return_value=None), \
                 patch.object(system_loop, "log_generation_failure") as log_failure, \
                 patch.object(system_loop, "MAX_GENERATOR_CALLS", limit):
                with self.assertRaises((RuntimeError, SystemExit)):
                    system_loop.main(gemini=generator, openrouter=SimpleNamespace())
                log_failure.assert_called_once()
                self.assertIn(expected, log_failure.call_args.args[0]["failure_reason"])

    def test_initialization_errors_have_reasons(self):
        with patch("builtins.input", return_value="Reach goal"), \
             patch.object(system_loop, "configure_tracking", return_value=None), \
             patch.object(system_loop, "Gemini", side_effect=ValueError("invalid model")), \
             patch.object(system_loop, "log_generation_failure") as log_failure:
            with self.assertRaisesRegex(ValueError, "invalid model"):
                system_loop.main(openrouter=SimpleNamespace())
            self.assertIn("initialization: ValueError: invalid model", log_failure.call_args.args[0]["failure_reason"])

    def test_training_errors_and_completed_failures_have_reasons(self):
        for training_error in [True, False]:
            with self.subTest(training_error=training_error):
                ground = SimpleNamespace(grid=SimpleNamespace(tolist=lambda: []))
                run_info = {"trace": (), "reward_efficiency": 0, "episode_reward": -2,
                            "minimum_reward": -2, "maximum_reward": 10, "steps": 200,
                            "terminated": False, "truncated": True, "grid_progression": [], "bell_rings": []}
                with patch("builtins.input", return_value="Reach goal"), \
                     patch.object(system_loop, "configure_tracking", return_value=None), \
                     patch.object(system_loop, "generate_gridworld", return_value=ground), \
                     patch.object(system_loop, "GridWorldLabellingFunction", return_value=object()), \
                     patch.object(system_loop, "GridWorldCrossProduct", return_value=SimpleNamespace(max_steps=200)), \
                     patch.object(system_loop, "train_agent", side_effect=ValueError("training crashed") if training_error else None,
                                  return_value=({}, [])), \
                     patch.object(system_loop, "calculate_trace_reward_bounds", return_value={"shortest_valid_steps": 1}), \
                     patch.object(system_loop, "run_agent", return_value=run_info), \
                     patch.object(system_loop, "combined_difficulty", return_value={"difficulty_score": 0}), \
                     patch.object(system_loop, "export_grid_animation", return_value=(None, False)), \
                     patch.object(system_loop, "log_generation_failure") as log_failure, \
                     patch.object(system_loop, "log_experiment_run", return_value="test") as log_run:
                    generator = SimpleNamespace(generate=lambda prompt: GENERATION_FORMAT_SHORT)
                    if training_error:
                        with self.assertRaisesRegex(ValueError, "training crashed"):
                            system_loop.main(gemini=generator, openrouter=SimpleNamespace(), iterative=False)
                        self.assertIn("training: ValueError: training crashed", log_failure.call_args.args[0]["failure_reason"])
                        log_run.assert_not_called()
                    else:
                        system_loop.main(gemini=generator, openrouter=SimpleNamespace(), iterative=False)
                        self.assertIn("200-step limit", log_run.call_args.args[0]["failure_reason"])
                        self.assertIn("One-shot", log_run.call_args.args[0]["failure_reason"])
                        log_failure.assert_not_called()

    def test_logger_and_historical_backfill_store_nonempty_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            uri = Path(directory).as_uri()
            with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": uri, "MLFLOW_EXPERIMENT_NAME": "Reasons",
                                          "MLFLOW_ALLOW_FILE_STORE": "true"}):
                api = results.client(uri)
                experiment_id = api.create_experiment("Reasons")
                run_id = api.create_run(experiment_id).info.run_id
                api.log_param(run_id, "success_status", False)
                api.log_param(run_id, "task_id", "old")
                for name, value in {
                    "traces.json": {"desired_trace": ["GOAL"], "final_trace": []},
                    "iterations.json": [{"run_information": {"trace": [], "steps": 200, "truncated": True,
                                                              "reward_efficiency": 0}}],
                }.items():
                    path = Path(directory) / name
                    path.write_text(json.dumps(value))
                    api.log_artifact(run_id, str(path))
                api.set_terminated(run_id)
                success_id = api.create_run(experiment_id).info.run_id
                api.log_param(success_id, "success_status", True)
                repaired = results.backfill_failure_reasons(experiment_id, uri)
                self.assertEqual([item["run_id"] for item in repaired], [run_id])
                self.assertIn("No events", api.get_run(run_id).data.params["failure_reason"])
                self.assertEqual(results.backfill_failure_reasons(experiment_id, uri), [])
                self.assertNotIn("failure_reason", api.get_run(success_id).data.params)
                record = {key: 0 for key in [
                    "training_episodes", "training_learning_rate", "training_discount_factor", "training_initial_epsilon",
                    "training_min_epsilon", "training_epsilon_decay", "training_seed", "iterations", "generator_call_count",
                    "judge_call_count", "average_training_time", "average_grid_world_generation_time", "average_generation_time",
                    "average_judge_time", "average_iteration_time", "total_system_time", "overall_difficulty_score",
                    "episode_reward", "maximum_reward", "minimum_reward", "steps_taken", "reward_efficiency", "crm_parse_fail_percentage",
                ]}
                record.update(task_id="new", task_description="Reach goal", success_status=False,
                              generator_model="test", judge_model="test", iterative=False, long=False,
                              specifications={"difficulty": {"environment": {"required_object_count": 0, "hazard_count": 0, "wall_count": 0}}},
                              desired_trace=["GOAL"], final_trace=[], judge_reflection="")
                new_id = log_experiment_run(record)
                self.assertIn("No events", api.get_run(new_id).data.params["failure_reason"])
                record.update(total_system_time=1, crm_parse_fail_percentage=75.0)
                failed_id = log_generation_failure(record)
                self.assertEqual(api.get_run(failed_id).data.metrics["crm_parse_fail_percentage"], 75.0)


if __name__ == "__main__":
    unittest.main()
