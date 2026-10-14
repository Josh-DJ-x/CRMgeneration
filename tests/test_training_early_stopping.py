"""Offline coverage for continuous training and checkpoint success criteria."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import system_functions
import system_loop


class OneStepEnvironment:
    action_space = SimpleNamespace(n=1)
    max_steps = 1

    def reset(self):
        return (0,), {}

    def step(self, action):
        return (0,), 1.0, True, False, {}


class EarlyStoppingTests(unittest.TestCase):
    def test_checkpoints_preserve_q_table_and_stop_at_success(self):
        checkpoints = []

        def check(q_table, episodes):
            checkpoints.append((q_table, episodes, q_table[(0,)][0]))
            return episodes == 2000

        with patch.object(system_functions, "valid_gridworld_actions", return_value=[0]):
            q_table, returns = system_functions.train_agent(
                OneStepEnvironment(), episodes=5000, learning_rate=0.0001,
                stop_training=check,
            )
        self.assertEqual(len(returns), 2000)
        self.assertEqual([row[1] for row in checkpoints], [1000, 2000])
        self.assertTrue(all(row[0] is q_table for row in checkpoints))
        self.assertGreater(checkpoints[1][2], checkpoints[0][2])

    def test_failed_checkpoints_reach_maximum(self):
        checkpoints = []
        with patch.object(system_functions, "valid_gridworld_actions", return_value=[0]):
            _, returns = system_functions.train_agent(
                OneStepEnvironment(), episodes=5000,
                stop_training=lambda table, episode: checkpoints.append(episode) or False,
            )
        self.assertEqual(len(returns), 5000)
        self.assertEqual(checkpoints, [1000, 2000, 3000, 4000, 5000])

    def test_loop_requires_strict_threshold_and_matching_trace_and_reuses_result(self):
        outcomes = [
            {"trace": ("GOAL",), "reward_efficiency": 1.0, "steps": 100},
            {"trace": (), "reward_efficiency": 1.0, "steps": 95},
            {"trace": ("GOAL",), "reward_efficiency": 0.94, "steps": 99},
            {"trace": ("GOAL",), "reward_efficiency": 0.95, "steps": 99},
        ]
        outcomes = [{**result, "episode_reward": 1.0, "minimum_reward": 0.0,
                     "maximum_reward": 1.0,
                     "grid_progression": [], "bell_rings": []} for result in outcomes]
        generator = SimpleNamespace(generate=lambda prompt: (
            '<<<DESIRED_TRACE>>>\nGOAL\n<<<ENVIRONMENT>>>\n'
            '{"coins":0,"bells":0,"keys":{"red":0,"blue":0,"green":0,"yellow":0},"hazards":false}'
            '\n<<<CRM>>>\nfake CRM'))

        def training(env, **kwargs):
            self.assertEqual(kwargs["episodes"], 5000)
            self.assertEqual(kwargs["evaluation_interval"], 1000)
            for episode in (1000, 2000, 3000, 4000):
                stopped = kwargs["stop_training"]({}, episode)
                self.assertEqual(stopped, episode == 4000)
                if stopped:
                    return {}, [1.0] * episode
            self.fail("Training did not stop")

        with (
            patch("builtins.input", return_value="Reach the goal"),
            patch.object(system_loop, "TRAINING_EPISODES", 5000),
            patch.object(system_loop, "TRAINING_EVALUATION_INTERVAL", 1000),
            patch.object(system_loop, "TRAINING_STEP_EFFICIENCY_THRESHOLD", 0.95),
            patch.object(system_loop, "TRAINING_REWARD_EFFICIENCY_THRESHOLD", 0.94),
            patch.object(system_loop, "configure_tracking", return_value=None),
            patch.object(system_loop, "crm_parse", return_value=object()),
            patch.object(system_loop, "crm_accepts_trace", return_value=True),
            patch.object(system_loop, "generate_gridworld", return_value=SimpleNamespace(
                grid=SimpleNamespace(tolist=lambda: []))),
            patch.object(system_loop, "GridWorldLabellingFunction", return_value=object()),
            patch.object(system_loop, "GridWorldCrossProduct", return_value=SimpleNamespace(max_steps=200)),
            patch.object(system_loop, "calculate_trace_reward_bounds", return_value={"shortest_valid_steps": 95}),
            patch.object(system_loop, "train_agent", side_effect=training),
            patch.object(system_loop, "run_agent", side_effect=outcomes) as evaluate,
            patch.object(system_loop, "crm_specification", return_value={}),
            patch.object(system_loop, "combined_difficulty", return_value={"difficulty_score": 0.1}),
            patch.object(system_loop, "export_grid_animation", return_value=("test.gif", True)),
            patch.object(system_loop, "log_experiment_run", return_value="test") as log_run,
        ):
            system_loop.main(gemini=generator, iterative=False)
        self.assertEqual(evaluate.call_count, 4)
        record = log_run.call_args.args[0]
        self.assertTrue(record["success_status"])
        self.assertEqual(record["minimum_steps"], 95)
        self.assertAlmostEqual(record["step_efficiency"], 95 / 99)
        self.assertEqual(record["performance_criterion"], "step_and_reward_efficiency")
        self.assertEqual(record["training_step_efficiency_threshold"], 0.95)
        self.assertEqual(record["training_reward_efficiency_threshold"], 0.94)
        self.assertEqual(record["reward_efficiency"], 0.95)
        iteration = record["iteration_history"][0]
        self.assertEqual(iteration["training_episodes_completed"], 4000)
        self.assertEqual(len(iteration["training_evaluations"]), 4)
        self.assertTrue(iteration["training_evaluations"][-1]["stopped_early"])


if __name__ == "__main__":
    unittest.main()
