"""Route efficiency, evaluation metrics, and tracking checks."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crm_support import mlflow_tracking
import system_functions
import system_loop
from crm_support.failure_diagnostics import failure_reason


class StepEfficiencyTests(unittest.TestCase):
    def test_ratio_and_zero_steps(self):
        self.assertEqual(system_functions.step_efficiency(95, 100), 0.95)
        self.assertGreater(system_functions.step_efficiency(95, 99), 0.95)
        self.assertEqual(system_functions.step_efficiency(95, 0), 0.0)

    def test_evaluation_includes_minimum_steps_and_step_efficiency(self):
        env = SimpleNamespace(
            reset=lambda: ((0,), {}),
            step=lambda action: ((0,), 1.0, True, False, {}),
            props={system_functions.Event.GOAL},
        )
        bounds = {"minimum_reward": -1.0, "maximum_reward": 1.0, "shortest_valid_steps": 1}
        with patch.object(system_functions, "valid_gridworld_actions", return_value=[0]):
            info = system_functions.run_agent(env, {(0,): [1.0]}, bounds)
        self.assertEqual(info["minimum_steps"], 1)
        self.assertEqual(info["steps"], 1)
        self.assertEqual(info["step_efficiency"], 1.0)

    def test_failure_reason_uses_strict_step_threshold(self):
        reason = failure_reason({
            "success_status": False, "desired_trace": ["GOAL"], "final_trace": ["GOAL"],
            "performance_criterion": "step_efficiency", "training_step_efficiency_threshold": 0.95,
            "minimum_steps": 95, "steps_taken": 100, "step_efficiency": 0.95,
            "reward_efficiency": 1.0, "iterative": False,
        })
        self.assertIn("95 minimum / 100 taken", reason)
        self.assertIn("must be greater than 0.950", reason)
        self.assertNotIn("Reward efficiency", reason)

    def test_both_efficiency_qualifiers_are_required(self):
        for steps, rewards, expected in (
            (0.96, 0.95, True),
            (0.95, 1.0, False),
            (1.0, 0.94, False),
            (0.9, 1.0, False),
            (1.0, 0.9, False),
        ):
            with (self.subTest(steps=steps, rewards=rewards),
                  patch.object(system_loop, "TRAINING_STEP_EFFICIENCY_THRESHOLD", 0.95),
                  patch.object(system_loop, "TRAINING_REWARD_EFFICIENCY_THRESHOLD", 0.94)):
                self.assertEqual(system_loop._performance_passed({
                    "step_efficiency": steps, "reward_efficiency": rewards,
                }), expected)

    def test_combined_failure_reason_reports_each_failed_qualifier(self):
        for steps, rewards, step_failure, reward_failure in (
            (0.95, 0.95, True, False),
            (0.96, 0.94, False, True),
            (0.95, 0.94, True, True),
        ):
            with self.subTest(steps=steps, rewards=rewards):
                reason = failure_reason({
                    "success_status": False, "performance_criterion": "step_and_reward_efficiency",
                    "step_efficiency": steps, "reward_efficiency": rewards,
                    "minimum_steps": 95, "steps_taken": 100,
                })
                self.assertEqual("Step efficiency" in reason, step_failure)
                self.assertEqual("Reward efficiency" in reason, reward_failure)

    def test_tracker_records_route_metrics_and_qualification_parameter(self):
        record = {key: 0 for key in (
            "training_episodes", "training_learning_rate", "training_discount_factor",
            "training_initial_epsilon", "training_min_epsilon", "training_epsilon_decay",
            "training_seed", "iterations", "generator_call_count", "judge_call_count",
            "average_training_time", "average_grid_world_generation_time", "average_generation_time",
            "average_judge_time", "average_iteration_time", "total_system_time",
            "overall_difficulty_score", "episode_reward", "maximum_reward", "minimum_reward",
            "reward_efficiency", "crm_parse_fail_percentage",
        )}
        record.update(
            task_id="test", task_description="Reach goal", success_status=True,
            generator_model="test", judge_model="test", iterative=False, long=False,
            specifications={"difficulty": {"environment": {
                "required_object_count": 0, "hazard_count": 0, "wall_count": 0,
                "minimum_steps_to_complete_trace": 95,
            }}}, desired_trace=["GOAL"], final_trace=["GOAL"], judge_reflection="",
            minimum_steps=95, steps_taken=99, step_efficiency=95 / 99,
            performance_criterion="step_and_reward_efficiency", training_step_efficiency_threshold=0.95,
            training_reward_efficiency_threshold=0.94,
        )
        tracking = MagicMock()
        with patch.object(mlflow_tracking, "configure_tracking", return_value=tracking):
            mlflow_tracking.log_experiment_run(record)
        params = tracking.log_params.call_args.args[0]
        self.assertEqual(params["training_step_efficiency_threshold"], 0.95)
        self.assertEqual(params["performance_criterion"], "step_and_reward_efficiency")
        self.assertEqual(params["training_reward_efficiency_threshold"], 0.94)
        metrics = tracking.log_metrics.call_args_list[0].args[0]
        self.assertEqual(metrics["steps_taken"], 99)
        self.assertEqual(metrics["minimum_steps"], 95)
        self.assertAlmostEqual(metrics["step_efficiency"], 95 / 99)


if __name__ == "__main__":
    unittest.main()
