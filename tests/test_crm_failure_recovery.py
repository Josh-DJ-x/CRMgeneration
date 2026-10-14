"""CRM validation recovery and difficulty logging without provider calls."""

import io
import json
import sys
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crm_support import mlflow_tracking
import system_loop
from system_specifications import GENERATION_FORMAT_SHORT
from system_functions import (combined_difficulty, crm_parse, extract_generation_trace,
                              generate_gridworld, OpenRouterResponseError)


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


class RateLimitError(Exception):
    status_code = 429


class CRMFailureRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace, cls.crm_text, environment = extract_generation_trace(
            GENERATION_FORMAT_SHORT, initial=True
        )
        cls.spec = json.loads(environment)
        cls.ground = generate_gridworld(seed=42, environment_json=environment, task_trace=cls.trace)
        cls.crm = crm_parse(cls.crm_text)
        cls.difficulty = combined_difficulty(cls.ground.grid.tolist(), cls.spec, cls.crm, cls.trace)

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(
            system_loop, "OpenRouter", side_effect=AssertionError("Unmocked judge fallback")
        ))
        replacements = {
            "configure_tracking": None,
            "generate_gridworld": self.ground,
            "GridWorldLabellingFunction": object(),
            "GridWorldCrossProduct": SimpleNamespace(max_steps=200),
            "calculate_trace_reward_bounds": {"shortest_valid_steps": 1},
            "train_agent": ({}, []),
            "export_grid_animation": ("test.gif", True),
            "log_experiment_run": "test-run",
            "log_generation_failure": "failed-run",
        }
        self.mocks = {
            name: self.stack.enter_context(patch.object(system_loop, name, return_value=value))
            for name, value in replacements.items()
        }
        self.failed_run = {
            "trace": (), "reward_efficiency": 0.0, "episode_reward": 0.0,
            "minimum_reward": 0.0, "maximum_reward": 1.0, "steps": 200,
            "grid_progression": [], "bell_rings": [],
        }
        self.evaluate = self.stack.enter_context(patch.object(
            system_loop, "run_agent", return_value=self.failed_run
        ))

    def assert_difficulty_retained(self):
        self.mocks["log_generation_failure"].assert_called_once()
        record = self.mocks["log_generation_failure"].call_args.args[0]
        self.assertEqual(record["overall_difficulty_score"], self.difficulty["difficulty_score"])
        self.assertEqual(record["specifications"]["environment_spec"], self.spec)
        self.assertEqual(record["desired_trace"], list(self.trace))
        self.assertEqual(record["failure_category"], "rate_limit")
        return record

    def test_judge_rate_limit_retains_difficulty(self):
        generator = FakeLLM([GENERATION_FORMAT_SHORT])
        judge = FakeLLM([RateLimitError("too many requests")])
        fallback = FakeLLM([RateLimitError("fallback also rate limited")])
        fallback.model = "openrouter/free"
        with patch.object(system_loop, "OpenRouter", return_value=fallback) as factory, \
             self.assertRaisesRegex(SystemExit, "Judge LLM is rate limited"):
            system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        factory.assert_called_once()
        record = self.assert_difficulty_retained()
        self.assertEqual(record["specifications"]["difficulty"], self.difficulty)
        self.assertEqual(record["judge_call_count"], 2)
        self.assertTrue(record["judge_fallback_used"])
        self.assertEqual(record["iteration_history"][0]["run_information"]["steps"], 200)

    def test_rate_limited_judge_switches_for_all_remaining_judgements(self):
        judge = FakeLLM([RateLimitError("too many requests")])
        judge.model = "pinned-judge"
        fallback = FakeLLM(["<<<REFLECTION>>>Repair one", "<<<REFLECTION>>>Repair two"])
        fallback.model = "openrouter/free"
        fallback.last_response_model = "fallback-actual"
        generator = FakeLLM([GENERATION_FORMAT_SHORT] + ["<<<CRM>>>\n" + self.crm_text] * 2)
        with patch.object(system_loop, "OpenRouter", return_value=fallback) as factory:
            system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        factory.assert_called_once()
        self.assertEqual(len(judge.prompts), 1)
        self.assertEqual(len(fallback.prompts), 2)
        self.assertEqual(judge.prompts[0], fallback.prompts[0])
        self.assertIn("Iteration 2", fallback.prompts[1])
        self.mocks["log_generation_failure"].assert_not_called()
        record = self.mocks["log_experiment_run"].call_args.args[0]
        self.assertEqual(record["judge_model"], "pinned-judge")
        self.assertTrue(record["judge_fallback_used"])
        self.assertEqual(record["judge_call_count"], 3)
        self.assertEqual(record["iterations"], 3)

    def test_fallback_length_limit_stops_after_one_full_budget_attempt(self):
        judge = FakeLLM([RateLimitError("too many requests")])
        judge.model = "pinned-judge"
        fallback = FakeLLM([
            OpenRouterResponseError("output token budget", finish_reason="length"),
            "<<<REFLECTION>>>Repair one", "<<<REFLECTION>>>Repair two",
        ])
        fallback.model = "openrouter/free"
        fallback.max_tokens = 16384
        generator = FakeLLM([GENERATION_FORMAT_SHORT] + ["<<<CRM>>>\n" + self.crm_text] * 2)
        with patch.object(system_loop, "OpenRouter", return_value=fallback) as factory, \
             self.assertRaisesRegex(SystemExit, "output token budget"):
            system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        factory.assert_called_once()
        self.assertEqual(fallback.max_tokens, 16384)
        self.assertEqual(len(fallback.prompts), 1)
        record = self.mocks["log_generation_failure"].call_args.args[0]
        self.assertEqual(record["failure_category"], "judge_output_budget")
        self.assertEqual(record["judge_call_count"], 2)
        self.assertTrue(record["judge_fallback_used"])

    def test_length_limit_stops_without_retry(self):
        judge = FakeLLM([OpenRouterResponseError("output token budget", finish_reason="length")] * 2)
        judge.model = "pinned-judge"
        judge.max_tokens = 16384
        with self.assertRaisesRegex(SystemExit, "output token budget"):
            system_loop.main(gemini=FakeLLM([GENERATION_FORMAT_SHORT]), openrouter=judge,
                             task_description="Reach the goal")
        self.assertEqual(len(judge.prompts), 1)
        record = self.mocks["log_generation_failure"].call_args.args[0]
        self.assertEqual(record["failure_category"], "judge_output_budget")
        self.assertEqual(record["judge_call_count"], 1)

    def test_nvidia_503_switches_for_remaining_judgements(self):
        judge = FakeLLM([OpenRouterResponseError(
            "OpenRouter judge (nvidia/nemotron-3-ultra-550b-a55b:free) returned an error [503]: "
            "Upstream error from Nvidia: Service temporarily overloaded", status_code=503)])
        judge.model = "nvidia/nemotron-3-ultra-550b-a55b:free"
        fallback = FakeLLM(["<<<REFLECTION>>>Repair one", "<<<REFLECTION>>>Repair two"])
        fallback.model = "openrouter/free"
        generator = FakeLLM([GENERATION_FORMAT_SHORT] + ["<<<CRM>>>\n" + self.crm_text] * 2)
        with patch.object(system_loop, "OpenRouter", return_value=fallback) as factory:
            system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        factory.assert_called_once()
        self.assertEqual(len(judge.prompts), 1)
        self.assertEqual(len(fallback.prompts), 2)
        self.assertEqual(fallback.prompts[0], judge.prompts[0])
        self.mocks["log_generation_failure"].assert_not_called()
        record = self.mocks["log_experiment_run"].call_args.args[0]
        self.assertTrue(record["judge_fallback_used"])
        self.assertEqual(record["judge_call_count"], 3)

    def test_fallback_overload_stops_cleanly_without_repeated_switching(self):
        judge = FakeLLM([OpenRouterResponseError("overloaded", status_code=503)])
        fallback = FakeLLM([OpenRouterResponseError("fallback overloaded", status_code=503)])
        fallback.model = "openrouter/free"
        with patch.object(system_loop, "OpenRouter", return_value=fallback) as factory, \
             self.assertRaisesRegex(SystemExit, "high demand"):
            system_loop.main(gemini=FakeLLM([GENERATION_FORMAT_SHORT]), openrouter=judge,
                             task_description="Reach the goal")
        factory.assert_called_once()
        record = self.mocks["log_generation_failure"].call_args.args[0]
        self.assertEqual(record["failure_category"], "provider_capacity")
        self.assertTrue(record["judge_fallback_used"])
        self.assertEqual(record["judge_call_count"], 2)

    def test_mid_run_fallback_keeps_judge_history_and_resolved_models(self):
        judge = FakeLLM(["<<<REFLECTION>>>First feedback", RateLimitError("429")])
        judge.model = "pinned-judge"
        judge.resolved_models = ["original-actual"]
        fallback = FakeLLM(["<<<REFLECTION>>>Fallback feedback"])
        fallback.model = "openrouter/free"
        generator = FakeLLM([GENERATION_FORMAT_SHORT] + ["<<<CRM>>>\n" + self.crm_text] * 2)
        with patch.object(system_loop, "OpenRouter", return_value=fallback):
            system_loop.main(gemini=generator, openrouter=judge, judge_backend="openai",
                             task_description="Reach the goal")
        self.assertEqual(fallback.messages, [
            {"role": "user", "content": judge.prompts[0]},
            {"role": "assistant", "content": "<<<REFLECTION>>>First feedback"},
        ])
        self.assertEqual(fallback.resolved_models, ["original-actual"])
        record = self.mocks["log_experiment_run"].call_args.args[0]
        self.assertEqual(record["judge_backend"], "openai+openrouter")

    def test_default_router_rate_limit_stops_without_another_fallback(self):
        judge = FakeLLM([RateLimitError("too many requests")])
        judge.model = "openrouter/free"
        with patch.object(system_loop, "OpenRouter") as factory, \
             self.assertRaisesRegex(SystemExit, "Judge LLM is rate limited"):
            system_loop.main(gemini=FakeLLM([GENERATION_FORMAT_SHORT]), openrouter=judge,
                             task_description="Reach the goal")
        factory.assert_not_called()
        self.assertFalse(self.mocks["log_generation_failure"].call_args.args[0]["judge_fallback_used"])

    def test_judge_missing_answer_exits_cleanly_and_logs_evaluation(self):
        judge = FakeLLM([OpenRouterResponseError("no completion choices")])
        with self.assertRaisesRegex(SystemExit, "Judge LLM failed: no completion choices") as raised:
            system_loop.main(gemini=FakeLLM([GENERATION_FORMAT_SHORT]), openrouter=judge,
                             task_description="Reach the goal")
        self.assertTrue(raised.exception.__suppress_context__)
        self.mocks["log_generation_failure"].assert_called_once()
        record = self.mocks["log_generation_failure"].call_args.args[0]
        self.assertEqual(record["failure_category"], "judge_provider_error")
        self.assertEqual(record["failure_stage"], "judge")
        self.assertEqual(record["judge_call_count"], 1)
        self.assertEqual(record["iteration_history"][0]["run_information"]["steps"], 200)
        self.assertEqual(record["overall_difficulty_score"], self.difficulty["difficulty_score"])


    def test_refinement_rate_limit_retains_difficulty_and_feedback(self):
        generator = FakeLLM([GENERATION_FORMAT_SHORT, RateLimitError("too many requests")])
        judge = FakeLLM(["<<<REFLECTION>>>Repair the CRM"])
        with self.assertRaisesRegex(SystemExit, "Generator LLM is rate limited"):
            system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        record = self.assert_difficulty_retained()
        self.assertEqual(record["generator_call_count"], 2)
        self.assertEqual(record["judge_reflection"], "Repair the CRM")

    def test_correction_rate_limit_before_training_still_has_task_difficulty(self):
        initial = GENERATION_FORMAT_SHORT.split("<<<CRM>>>")[0] + "<<<CRM>>>\ninvalid CRM"
        generator = FakeLLM([initial, RateLimitError("too many requests")])
        judge = FakeLLM([])
        with self.assertRaisesRegex(SystemExit, "Generator LLM is rate limited"):
            system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        record = self.assert_difficulty_retained()
        self.assertIsNone(record["specifications"]["crm_specification"])
        self.assertEqual(record["failure_stage"], "generator")
        self.mocks["generate_gridworld"].assert_called_once()
        self.mocks["train_agent"].assert_not_called()
        self.assertEqual(judge.prompts, [])

    def test_failed_corrections_go_to_judge_and_repaired_crm_is_evaluated(self):
        for broken in ("invalid CRM", self.crm_text.replace("GOAL / (-) -> -1", "GOAL / (-) -> 0")):
            with self.subTest(broken=broken):
                for mock in self.mocks.values():
                    mock.reset_mock()
                initial = GENERATION_FORMAT_SHORT.split("<<<CRM>>>")[0] + "<<<CRM>>>\n" + broken
                generator = FakeLLM([initial] + ["<<<CRM>>>\n" + broken] * 3
                                    + ["<<<CRM>>>\n" + self.crm_text])
                judge = FakeLLM(["<<<REFLECTION>>>Use the terminal GOAL transition"])
                self.evaluate.return_value = {
                    **self.failed_run, "trace": self.trace, "steps": 1,
                    "reward_efficiency": 1.0, "episode_reward": 1.0,
                }
                system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
                self.assertEqual(len(generator.prompts), 5)
                self.assertEqual(len(judge.prompts), 1)
                self.assertIn(broken, judge.prompts[0])
                self.assertIn('"evaluation_skipped": true', judge.prompts[0])
                self.assertIn("validation_error", judge.prompts[0])
                self.assertIn("Use the terminal GOAL transition", generator.prompts[-1])
                self.mocks["train_agent"].assert_called_once()
                self.mocks["log_generation_failure"].assert_not_called()
                record = self.mocks["log_experiment_run"].call_args.args[0]
                self.assertTrue(record["success_status"])
                self.assertEqual(record["iterations"], 2)
                self.assertEqual(record["crm_parse_fail_percentage"], 80)
                history = record["iteration_history"]
                self.assertEqual(history[0]["training_episodes_completed"], 0)
                self.assertTrue(history[0]["run_information"]["evaluation_skipped"])
                self.assertEqual(history[0]["judge_feedback"], "Use the terminal GOAL transition")
                self.assertTrue(history[1]["trace_correct"])

    def test_initial_trace_parse_failure_reaches_judge_then_full_generation_repair(self):
        broken = ("<<<DESIRED_TRACE>>>\nHAZARD -> -1 (TERMINAL)\n<<<ENVIRONMENT>>>"
                  + GENERATION_FORMAT_SHORT.split("<<<ENVIRONMENT>>>")[1])
        generator = FakeLLM([broken] * 4 + [GENERATION_FORMAT_SHORT])
        judge = FakeLLM(["<<<REFLECTION>>>Use GOAL for avoiding hazards and repair the trace syntax"])
        self.evaluate.return_value = {**self.failed_run, "trace": self.trace, "steps": 1,
                                     "reward_efficiency": 1.0, "episode_reward": 1.0}
        system_loop.main(gemini=generator, openrouter=judge, task_description="Avoid hazards")
        self.assertEqual(len(generator.prompts), 5)
        self.assertEqual(len(judge.prompts), 1)
        self.assertIn("Avoid hazards", judge.prompts[0])
        self.assertIn("HAZARD -> -1 (TERMINAL)", judge.prompts[0])
        self.assertIn('"evaluation_skipped": true', judge.prompts[0])
        self.assertIn("Use GOAL for avoiding hazards", generator.prompts[-1])
        self.assertIn("Return all three sections", generator.prompts[-1])
        self.mocks["generate_gridworld"].assert_called_once()
        self.mocks["train_agent"].assert_called_once()
        record = self.mocks["log_experiment_run"].call_args.args[0]
        self.assertTrue(record["success_status"])
        self.assertEqual(record["iterations"], 2)
        self.assertEqual(record["iteration_history"][0]["grid"], [])
        self.assertEqual(record["desired_trace"], list(self.trace))

    def test_missing_initial_sections_are_corrected_before_evaluation(self):
        generator = FakeLLM(["missing all sections", GENERATION_FORMAT_SHORT])
        self.evaluate.return_value = {**self.failed_run, "trace": self.trace, "steps": 1,
                                     "reward_efficiency": 1.0, "episode_reward": 1.0}
        judge = FakeLLM([])
        system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        self.assertEqual(len(generator.prompts), 2)
        self.assertIn("Validation failure", generator.prompts[1])
        self.assertIn("Return all three sections", generator.prompts[1])
        self.assertEqual(judge.prompts, [])
        self.mocks["train_agent"].assert_called_once()

    def test_unrepaired_initial_trace_one_shot_stops_after_three_corrections(self):
        generator = FakeLLM(["malformed generation"] * 4)
        judge = FakeLLM([])
        with self.assertRaisesRegex(SystemExit, "after three corrections"):
            system_loop.main(gemini=generator, openrouter=judge, iterative=False,
                             task_description="Reach the goal")
        self.assertEqual(len(generator.prompts), 4)
        self.assertEqual(judge.prompts, [])
        self.mocks["generate_gridworld"].assert_not_called()
        self.mocks["train_agent"].assert_not_called()

    def test_invalid_initial_environment_json_reaches_judge_and_is_repaired(self):
        prefix, tail = GENERATION_FORMAT_SHORT.split("<<<ENVIRONMENT>>>")
        broken = prefix + "<<<ENVIRONMENT>>>\n{bad json}\n<<<CRM>>>" + tail.split("<<<CRM>>>")[1]
        generator = FakeLLM([broken] * 4 + [GENERATION_FORMAT_SHORT])
        judge = FakeLLM(["<<<REFLECTION>>>Fix the environment JSON"])
        self.evaluate.return_value = {**self.failed_run, "trace": self.trace, "steps": 1,
                                     "reward_efficiency": 1.0, "episode_reward": 1.0}
        system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        self.assertIn("ENVIRONMENT must contain valid JSON", judge.prompts[0])
        self.mocks["generate_gridworld"].assert_called_once()
        self.mocks["train_agent"].assert_called_once()

    def test_repaired_initial_sections_then_crm_only_correction_keep_fixed_task(self):
        bad_crm = GENERATION_FORMAT_SHORT.split("<<<CRM>>>")[0] + "<<<CRM>>>invalid CRM"
        generator = FakeLLM(["bad trace", bad_crm, "<<<CRM>>>\n" + self.crm_text])
        self.evaluate.return_value = {**self.failed_run, "trace": self.trace, "steps": 1,
                                     "reward_efficiency": 1.0, "episode_reward": 1.0}
        system_loop.main(gemini=generator, openrouter=FakeLLM([]), task_description="Reach the goal")
        self.assertEqual(len(generator.prompts), 3)
        self.assertNotIn("Return all three sections", generator.prompts[2])
        record = self.mocks["log_experiment_run"].call_args.args[0]
        self.assertEqual(record["desired_trace"], list(self.trace))
        self.assertEqual(record["specifications"]["environment_spec"], self.spec)

    def test_invalid_environment_specification_gets_preliminary_correction(self):
        for specification in [{**self.spec, "bells": 200}, {**self.spec, "keys": []},
                              {**self.spec, "coins": -1}, {**self.spec, "bells": True}]:
            with self.subTest(specification=specification):
                for mock in self.mocks.values():
                    mock.reset_mock()
                prefix = GENERATION_FORMAT_SHORT.split("<<<ENVIRONMENT>>>")[0]
                broken = prefix + "<<<ENVIRONMENT>>>\n" + json.dumps(specification) + "\n<<<CRM>>>\n" + self.crm_text
                generator = FakeLLM([broken, GENERATION_FORMAT_SHORT])
                self.evaluate.return_value = {**self.failed_run, "trace": self.trace, "steps": 1,
                                             "reward_efficiency": 1.0, "episode_reward": 1.0}
                system_loop.main(gemini=generator, openrouter=FakeLLM([]), task_description="Ring a bell repeatedly")
                self.assertEqual(len(generator.prompts), 2)
                self.assertIn("Invalid environment specification", generator.prompts[1])
                self.assertIn("one bell can be rung repeatedly", generator.prompts[1])
                self.mocks["generate_gridworld"].assert_called_once()
                self.mocks["train_agent"].assert_called_once()
                self.mocks["log_generation_failure"].assert_not_called()

    def test_invalid_environment_specification_reaches_judge_after_corrections(self):
        prefix = GENERATION_FORMAT_SHORT.split("<<<ENVIRONMENT>>>")[0]
        broken = prefix + "<<<ENVIRONMENT>>>\n" + json.dumps({**self.spec, "bells": 200}) + "\n<<<CRM>>>\n" + self.crm_text
        generator = FakeLLM([broken] * 4 + [GENERATION_FORMAT_SHORT])
        judge = FakeLLM(["<<<REFLECTION>>>Use one physical bell instead of 200 bell tiles"])
        self.evaluate.return_value = {**self.failed_run, "trace": self.trace, "steps": 1,
                                     "reward_efficiency": 1.0, "episode_reward": 1.0}
        system_loop.main(gemini=generator, openrouter=judge, task_description="Ring a bell repeatedly")
        self.assertEqual(len(generator.prompts), 5)
        self.assertEqual(len(judge.prompts), 1)
        self.assertIn("Invalid environment specification", judge.prompts[0])
        self.assertIn("Use one physical bell", generator.prompts[-1])
        self.mocks["generate_gridworld"].assert_called_once()
        self.mocks["train_agent"].assert_called_once()
        record = self.mocks["log_experiment_run"].call_args.args[0]
        self.assertTrue(record["success_status"])
        self.assertEqual(record["iterations"], 2)

    def test_layout_failure_is_repaired_before_training(self):
        self.mocks["generate_gridworld"].side_effect = [
            system_loop.EnvironmentGenerationError("No valid layout for generated objects"), self.ground]
        generator = FakeLLM([GENERATION_FORMAT_SHORT] * 2)
        self.evaluate.return_value = {**self.failed_run, "trace": self.trace, "steps": 1,
                                     "reward_efficiency": 1.0, "episode_reward": 1.0}
        system_loop.main(gemini=generator, openrouter=FakeLLM([]), task_description="Reach the goal")
        self.assertIn("No valid layout", generator.prompts[1])
        self.assertIn("Return all three sections", generator.prompts[1])
        self.mocks["train_agent"].assert_called_once()

    def test_reward_bound_crm_failure_reaches_judge_then_crm_repair(self):
        self.mocks["calculate_trace_reward_bounds"].side_effect = [
            ValueError("State u=8 is terminal or not defined in the transition function")] * 4 + [{"shortest_valid_steps": 1}]
        generator = FakeLLM([GENERATION_FORMAT_SHORT] + ["<<<CRM>>>\n" + self.crm_text] * 4)
        judge = FakeLLM(["<<<REFLECTION>>>Fix transitions during movement"])
        self.evaluate.return_value = {**self.failed_run, "trace": self.trace, "steps": 1,
                                     "reward_efficiency": 1.0, "episode_reward": 1.0}
        system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        self.assertIn("State u=8", judge.prompts[0])
        self.assertIn("Fix transitions during movement", generator.prompts[-1])
        self.assertNotIn("Return all three sections", generator.prompts[-1])
        self.mocks["train_agent"].assert_called_once()
        self.assertTrue(self.mocks["log_experiment_run"].call_args.args[0]["success_status"])

    def test_no_ordered_route_gets_full_generation_repair(self):
        self.mocks["calculate_trace_reward_bounds"].side_effect = [
            ValueError("No ordered route exists for the desired trace."), {"shortest_valid_steps": 1}]
        generator = FakeLLM([GENERATION_FORMAT_SHORT] * 2)
        self.evaluate.return_value = {**self.failed_run, "trace": self.trace, "steps": 1,
                                     "reward_efficiency": 1.0, "episode_reward": 1.0}
        system_loop.main(gemini=generator, openrouter=FakeLLM([]), task_description="Reach the goal")
        self.assertIn("No ordered route", generator.prompts[1])
        self.assertIn("Return all three sections", generator.prompts[1])
        self.mocks["train_agent"].assert_called_once()

    def test_route_timeout_trains_full_budget_and_qualifies_by_trace_only(self):
        self.mocks["calculate_trace_reward_bounds"].side_effect = system_loop.RouteSearchLimitError("60-second time limit")
        self.mocks["train_agent"].return_value = ({}, [0] * system_loop.TRAINING_EPISODES)
        self.evaluate.return_value = {**self.failed_run, "trace": self.trace, "steps": 100,
                                     "reward_efficiency": 0.01, "episode_reward": 1.0}
        system_loop.main(gemini=FakeLLM([GENERATION_FORMAT_SHORT]), openrouter=FakeLLM([]),
                         task_description="Reach the goal")
        args = self.mocks["train_agent"].call_args.kwargs
        self.assertEqual(args["episodes"], system_loop.TRAINING_EPISODES)
        self.assertIsNone(args["stop_training"])
        record = self.mocks["log_experiment_run"].call_args.args[0]
        self.assertTrue(record["success_status"])
        self.assertEqual(record["performance_criterion"], "trace_only")
        self.assertFalse(record["efficiency_baseline_available"])
        for key in ["step_efficiency", "reward_efficiency", "minimum_steps", "minimum_reward", "maximum_reward"]:
            self.assertIsNone(record[key])
        self.assertEqual(record["iteration_history"][0]["training_episodes_completed"], system_loop.TRAINING_EPISODES)
        self.assertEqual(self.mocks["calculate_trace_reward_bounds"].call_count, 1)
        with patch.object(mlflow_tracking, "configure_tracking") as configure:
            api = configure.return_value
            mlflow_tracking.log_experiment_run(record)
            metrics = {key: value for call in api.log_metrics.call_args_list for key, value in call.args[0].items()}
            for key in ["step_efficiency", "reward_efficiency", "minimum_steps", "minimum_reward", "maximum_reward"]:
                self.assertNotIn(key, metrics)
            self.assertNotIn(None, metrics.values())

    def test_collectible_threshold_skips_baseline_for_coins_and_keys(self):
        for coins, keys, expected_skip in [
            (16, {"red": 0, "blue": 0, "green": 0, "yellow": 0}, False),
            (17, {"red": 0, "blue": 0, "green": 0, "yellow": 0}, True),
            (15, {"red": 1, "blue": 1, "green": 0, "yellow": 0}, False),
            (0, {"red": 5, "blue": 4, "green": 4, "yellow": 4}, False),
            (0, {"red": 17, "blue": 0, "green": 0, "yellow": 0}, True),
        ]:
            with self.subTest(coins=coins, keys=keys):
                for mock in self.mocks.values():
                    mock.reset_mock()
                spec = {**self.spec, "coins": coins, "keys": keys}
                prefix = GENERATION_FORMAT_SHORT.split("<<<ENVIRONMENT>>>")[0]
                generation = prefix + "<<<ENVIRONMENT>>>\n" + json.dumps(spec) + "\n<<<CRM>>>\n" + self.crm_text
                self.mocks["train_agent"].return_value = ({}, [0] * system_loop.TRAINING_EPISODES)
                self.evaluate.return_value = {**self.failed_run, "trace": self.trace, "steps": 1,
                                             "reward_efficiency": 1.0, "episode_reward": 1.0}
                system_loop.main(gemini=FakeLLM([generation]), openrouter=FakeLLM([]), task_description="Collect items")
                record = self.mocks["log_experiment_run"].call_args.args[0]
                self.assertTrue(record["success_status"])
                self.assertEqual(record["efficiency_baseline_available"], not expected_skip)
                self.assertEqual(self.mocks["calculate_trace_reward_bounds"].call_count, 0 if expected_skip else 1)
                if expected_skip:
                    self.assertIsNone(self.mocks["train_agent"].call_args.kwargs["stop_training"])
                    self.assertIsNone(record["step_efficiency"])
                    self.assertIsNone(record["reward_efficiency"])
                else:
                    self.assertIsNotNone(self.mocks["train_agent"].call_args.kwargs["stop_training"])

    def test_route_timeout_bad_trace_still_uses_judge_and_full_training_each_iteration(self):
        self.mocks["calculate_trace_reward_bounds"].side_effect = system_loop.RouteSearchLimitError("state limit")
        self.mocks["train_agent"].return_value = ({}, [0] * system_loop.TRAINING_EPISODES)
        judge = FakeLLM(["<<<REFLECTION>>>Repair trace", "<<<REFLECTION>>>Repair again"])
        generator = FakeLLM([GENERATION_FORMAT_SHORT] + ["<<<CRM>>>\n" + self.crm_text] * 2)
        system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        self.assertEqual(len(judge.prompts), 2)
        self.assertIn("Efficiency values are null", judge.prompts[0])
        for call in self.mocks["train_agent"].call_args_list:
            self.assertIsNone(call.kwargs["stop_training"])
        record = self.mocks["log_experiment_run"].call_args.args[0]
        self.assertFalse(record["success_status"])
        self.assertEqual(record["iterations"], 3)
        self.assertIsNone(record["reward_efficiency"])
        self.assertEqual(self.mocks["calculate_trace_reward_bounds"].call_count, 1)

    def test_unparseable_crm_judge_rate_limit_still_has_task_difficulty(self):
        initial = GENERATION_FORMAT_SHORT.split("<<<CRM>>>")[0] + "<<<CRM>>>\ninvalid CRM"
        # Missing CRM sections in correction responses are validation failures too.
        generator = FakeLLM([initial] + ["invalid correction"] * 3)
        judge = FakeLLM([RateLimitError("too many requests")])
        fallback = FakeLLM([RateLimitError("fallback also rate limited")])
        fallback.model = "openrouter/free"
        with patch.object(system_loop, "OpenRouter", return_value=fallback), \
             self.assertRaisesRegex(SystemExit, "Judge LLM is rate limited"):
            system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        record = self.assert_difficulty_retained()
        self.assertIsNone(record["specifications"]["crm_specification"])
        self.assertIsNone(record["specifications"]["difficulty"]["crm"])
        self.assertEqual(record["crm_parse_fail_percentage"], 100)
        self.mocks["train_agent"].assert_not_called()

    def test_recovery_respects_generator_call_limit(self):
        initial = GENERATION_FORMAT_SHORT.split("<<<CRM>>>")[0] + "<<<CRM>>>\ninvalid CRM"
        generator = FakeLLM([initial] + ["<<<CRM>>>invalid CRM"] * 5)
        judge = FakeLLM(["<<<REFLECTION>>>Repair the CRM"])
        with self.assertRaisesRegex(SystemExit, "Generator call limit reached"):
            system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        self.assertEqual(len(generator.prompts), system_loop.MAX_GENERATOR_CALLS)
        self.assertEqual(len(judge.prompts), 1)
        self.mocks["train_agent"].assert_not_called()
        record = self.mocks["log_generation_failure"].call_args.args[0]
        self.assertEqual(record["overall_difficulty_score"], self.difficulty["difficulty_score"])

    def test_one_shot_validation_failure_still_skips_judge(self):
        initial = GENERATION_FORMAT_SHORT.split("<<<CRM>>>")[0] + "<<<CRM>>>\ninvalid CRM"
        generator = FakeLLM([initial] + ["<<<CRM>>>invalid CRM"] * 3)
        judge = FakeLLM([])
        with self.assertRaisesRegex(SystemExit, "after three corrections"):
            system_loop.main(gemini=generator, openrouter=judge, iterative=False,
                             task_description="Reach the goal")
        self.assertEqual(judge.prompts, [])
        self.mocks["train_agent"].assert_not_called()

    def test_failure_logger_uploads_difficulty_and_specifications(self):
        record = {
            "task_id": "1", "task_description": "Reach the goal", "generator_model": "test",
            "judge_model": "test", "iterative": True, "long": False,
            "generator_call_count": 2, "total_system_time": 3.0, "average_generation_time": 1.0,
            "failure_reason": "Generator LLM is rate limited", "failure_category": "rate_limit",
            "overall_difficulty_score": self.difficulty["difficulty_score"],
            "specifications": {"difficulty": self.difficulty, "environment_spec": self.spec,
                               "crm_specification": None},
            "desired_trace": list(self.trace), "final_trace": [],
            "judge_reflection": "Repair the CRM",
        }
        tracking = MagicMock()
        with patch.object(mlflow_tracking, "configure_tracking", return_value=tracking):
            mlflow_tracking.log_generation_failure(record)
        metrics = {key: value for call in tracking.log_metrics.call_args_list
                   for key, value in call.args[0].items()}
        self.assertEqual(metrics["overall_difficulty_score"], self.difficulty["difficulty_score"])
        self.assertEqual(metrics["crm_state_count"], self.difficulty["crm"]["states"])
        artifacts = {call.args[1]: call.args[0] for call in tracking.log_dict.call_args_list}
        self.assertEqual(artifacts["specifications.json"], record["specifications"])
        self.assertEqual(artifacts["traces.json"]["desired_trace"], list(self.trace))
        tracking.log_text.assert_called_once_with("Repair the CRM", "judge_reflection.txt")


if __name__ == "__main__":
    unittest.main()
