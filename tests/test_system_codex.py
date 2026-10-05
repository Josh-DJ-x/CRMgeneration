"""Offline test for the ``system_codex`` compatibility entry point.

``system_codex`` deliberately contains no loop logic. It should expose the
same injectable ``main`` function as ``system_loop`` so either entry point
uses the tested orchestration code.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import system
import system_loop


class SystemCodexTests(unittest.TestCase):
    def test_codex_entry_point_reexports_injectable_main(self):
        self.assertIs(system.main, system_loop.main)

    def test_comparison_modes_prompt_once_and_run_all_settings(self):
        for mode, settings in [
            ("1", [(True, False), (False, False)]),
            ("2", [(True, True), (False, True), (True, False)]),
        ]:
            with self.subTest(mode=mode), patch.object(system, "ITERATIVE", False), \
                 patch("builtins.input", return_value="Reach the goal") as task_input, \
                 patch.object(system, "main") as run, patch("builtins.print"):
                self.assertEqual(system.cli([mode]), 0)
                task_input.assert_called_once()
                self.assertEqual(run.call_count, len(settings))
                for call, (long, iterative) in zip(run.call_args_list, settings):
                    self.assertEqual(call.kwargs, {
                        "long": long, "iterative": iterative,
                        "task_description": "Reach the goal",
                        "generator_backend": system.GENERATOR_LLM,
                        "judge_model": system.JUDGE_LLM,
                    })

    def test_comparison_continues_after_recorded_failure(self):
        with patch("builtins.input", return_value="Reach the goal"), \
             patch.object(system, "main", side_effect=[SystemExit("rate limited"), None, None]) as run, \
             patch("builtins.print"):
            self.assertEqual(system.cli(["2"]), 1)
            self.assertEqual(run.call_count, 3)

    def test_comparison_stops_on_keyboard_interrupt(self):
        with patch("builtins.input", return_value="Reach the goal"), \
             patch.object(system, "main", side_effect=KeyboardInterrupt) as run, \
             patch("builtins.print"):
            with self.assertRaises(KeyboardInterrupt):
                system.cli(["2"])
            run.assert_called_once()

    def test_supplied_task_skips_input_and_preserves_run_context(self):
        with patch("builtins.input") as task_input, \
             patch.object(system_loop, "_run_task") as run:
            system_loop.main(task_description="Reach the goal", long=True, iterative=False)
            task_input.assert_not_called()
            self.assertEqual(run.call_args.args[0], "Reach the goal")
            self.assertEqual(run.call_args.args[3]["record"]["task_description"], "Reach the goal")

    def test_default_mode_preserves_configured_settings(self):
        with patch.object(system, "main") as run:
            system.cli([])
            run.assert_called_once_with(iterative=system.ITERATIVE, long=system.long,
                                        generator_backend=system.GENERATOR_LLM,
                                        judge_model=system.JUDGE_LLM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
