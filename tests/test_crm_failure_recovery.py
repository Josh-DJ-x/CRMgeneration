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

import mlflow_tracking
import system_loop
from semantic_specifications import GENERATION_FORMAT_SHORT
from system_functions import combined_difficulty, crm_parse, extract_generation_trace, generate_gridworld


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
        with self.assertRaisesRegex(SystemExit, "Judge LLM is rate limited"):
            system_loop.main(gemini=generator, openrouter=judge, task_description="Reach the goal")
        record = self.assert_difficulty_retained()
        self.assertEqual(record["specifications"]["difficulty"], self.difficulty)
        self.assertEqual(record["judge_call_count"], 1)
        self.assertEqual(record["iteration_history"][0]["run_information"]["steps"], 200)

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

    def test_unparseable_crm_judge_rate_limit_still_has_task_difficulty(self):
        initial = GENERATION_FORMAT_SHORT.split("<<<CRM>>>")[0] + "<<<CRM>>>\ninvalid CRM"
        # Missing CRM sections in correction responses are validation failures too.
        generator = FakeLLM([initial] + ["invalid correction"] * 3)
        judge = FakeLLM([RateLimitError("too many requests")])
        with self.assertRaisesRegex(SystemExit, "Judge LLM is rate limited"):
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
